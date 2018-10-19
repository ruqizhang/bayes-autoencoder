import torch
import argparse
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets, transforms
import sys
import model_bn
from ais import AIS
import torch.distributions as td
from hmc import HMC
import data
parser = argparse.ArgumentParser(description='AIS with bayesian auto-encoder')
parser.add_argument('--data', type=str, default='./datasets/ptb',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--zdim', type=int, default=20,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=3,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--device_id',type = int, help = 'device id to use')
parser.add_argument('--num-steps', type=int, default = 100, help = 'number of steps to run AIS for')
parser.add_argument('--num-samples', type=int, default = 2, help='number of chains to run AIS over')

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
device_id = 0
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
args.device = None
if args.cuda:
    args.device = torch.device('cuda')
    print('using ', args.device)
else:
    args.device = torch.device('cpu')

corpus = data.Corpus(args.data)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.to(args.device)
    return data

eval_batch_size = args.batch_size
# train_data = batchify(corpus.train, args.batch_size)
# val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)
test_data = test_data[0:35,:]
ntokens = len(corpus.dictionary)
model = model_bn.VAE(args.model, ntokens, args.emsize, args.nhid, args.zdim,args.nlayers,device_id,args.batch_size, args.dropout, args.tied)
model.to(args.device)
#collect one single batch for speed
# for x in test_data:
#     test_tensor_list = x
# class myiterator:
#     def __iter__(self):
#         return iter([test_tensor_list])
# new_test_loader = myiterator()

model.load_state_dict(torch.load('./bn/model_a0.1_ppl_2%i.pt'%(100)))

#create the prior distribution for z
pmean = torch.zeros(args.zdim).to(args.device)
pstd = torch.ones(args.zdim).to(args.device)
priordist = torch.distributions.Normal(pmean, pstd)

def geom_average_loss(t1, data, backwards = False):
    """
    t1: scaling factor for the geometric average loss
    data: data tensor
    backwards: if we draw a simulated model and use that instead of the real data
    """
    #pass t1 to current device if necessary
    t1 = t1.to(args.device)

    #want to calculate log(q(z|x)^(1-t1) p(x,z)^t1)

    #backwards pass ignores the data argument and samples generatively
    if backwards:
        #sample z generatively
        z = priordist.rsample(((data.size(1),)))
    else:
        #perform forwards pass through model if "forwards"
        #add noise

        # noise_mean = torch.zeros(data.size(0), args.zdim).to(args.device)
        # noise_std = torch.ones(data.size(0), args.zdim).to(args.device)

        # noise_sample = td.Normal(noise_mean, noise_std).rsample().to(args.device)
        emb = model.embed(data)
        # augmented_data = torch.cat((emb, noise_sample),dim=1)
        z,_ = model.encoder(emb)

    #pass backwards
    emb = model.embed(data)
    x_probs = model.decoder(emb,z)

    #create distributions
    l_dist = td.Bernoulli(probs = x_probs)

    #now the "backwards pass"
    if backwards:
        #draw simulated data_location
        data = l_dist.sample()

    #now compute log(geometric average)
    d0 = data.size(0)
    d1 = data.size(1)
    if backwards:
        value,idx = torch.max(data.view(-1,ntokens),dim =1)
        datam = torch.zeros(d0*d1,10000).cuda(device_id)
        datam[range(d0*d1),idx]=1
    else:
        datam = torch.zeros(d0*d1,10000).cuda(device_id)
        datam[range(d0*d1),data.view(-1)]=1
    datam = datam.view(d0,d1,10000)
    recon_loss = l_dist.log_prob(datam).sum(dim=2).sum(dim=0)

    prior_loss = priordist.log_prob(z).sum(dim=1)

    theta_loss = 0.0
    for param in model.encoder.parameters():
        param_dist = td.Normal(torch.zeros_like(param), torch.ones_like(param))
        theta_loss += param_dist.log_prob(param).sum()

    total_loss = t1 * recon_loss + prior_loss + theta_loss/10000.
    return total_loss.sum()

#note, right now im using adam as the transition operator, which i won't be doing.
#ill be using HMC in the future
sampler = HMC(model.parameters(), lr = 1e-6, L = 3)
ais_for_vae = AIS(model, test_data, sampler, num_beta=args.num_steps, num_samples = args.num_samples,
                            nprint=1)

print('Now running forward AIS')
logprob_est, logprob_vec, _ = ais_for_vae.run_forward(geom_average_loss)
print(sampler.acc_rate())
print('Lower bound estimate: ' + str(logprob_est/args.batch_size*1.0/ntokens))

# print('Now running backward AIS')
# blogprob_est, blogprob_vec, _ = ais_for_vae.run_backward(geom_average_loss)

# print('Upper bound estimate: ' + str(blogprob_est/args.batch_size*1.0/ntokens))