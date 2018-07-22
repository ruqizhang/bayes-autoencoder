# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import data
import model_bae
import torch.optim as optim
import numpy as np
from torch.autograd.gradcheck import zero_gradients
parser = argparse.ArgumentParser(description='PyTorch PTB RNN/LSTM Language Model')
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
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
parser.add_argument('--device_id',type = int, help = 'device id to use')
args = parser.parse_args()
device_id = args.device_id
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
args.cuda = True
###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda(device_id)
    return data

eval_batch_size = args.batch_size
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model_bae.VAE(args.model, ntokens, args.emsize, args.nhid, args.zdim,args.nlayers,device_id,args.batch_size, args.dropout, args.tied)
if args.cuda:
    model.cuda(device_id)

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def loss_function(recon_batch, x):
    recon_batch = recon_batch.view(-1,ntokens)
    BCE = F.cross_entropy(recon_batch, x)
    return BCE

def en_loss(z_recon,z):
    z = Variable(z.data,requires_grad = False)
    loss = F.mse_loss(z_recon,z)
    return loss

def z_prior_loss(z):
    prior_loss = 0.5*torch.sum(z*z)
    return prior_loss

def z_noise_loss(z):

    learning_rate = lr

    noise_std = np.sqrt(2*learning_rate*alpha)
    noise_std = torch.from_numpy(np.array([noise_std])).float().cuda(device_id)
    noise_std = noise_std[0]
    means = torch.zeros(z.size()).cuda(device_id)
    noise_loss = torch.sum(z * Variable(torch.normal(means, std = noise_std).cuda(device_id),
                           requires_grad = False))
    #print('noise_loss',noise_loss)#1e-8
    return noise_loss

def compute_jacobian(inputs, output):
    assert inputs.requires_grad
    # inputs = inputs.squeeze()
    num_classes = output.size()[1]

    jacobian = torch.zeros(args.batch_size, inputs.size(1),inputs.size(1)).cuda(device_id)
    grad_output = torch.zeros(*output.size()).cuda(device_id)

    for i in range(num_classes):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1
        output.backward(grad_output, retain_variables=True)
        jacobian[:,i,:] = inputs.grad.view(args.batch_size,1,args.zdim).data
    return jacobian

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    total_kld = 0
    ntokens = len(corpus.dictionary)
    count=0
    # hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i)
        model.decoder.bsz = eval_batch_size
        recon_batch,z,xi = model(data)
        BCE = loss_function(recon_batch, targets)
        # jacobian = compute_jacobian(xi,z)
        # xi = xi.squeeze()
        # xi_dim = xi.size(1)
        # q_xi = 1/((2*np.pi)**(xi_dim/2.0))*(torch.sum(xi*xi,1).mul(-0.5).exp())
        # det = np.linalg.det(jacobian.cpu().numpy())
        # det = torch.from_numpy(det).cuda(device_id)
        # log_det = torch.log(det)
        # det_inverse = 1/det
        # tem = -q_xi.data*det_inverse*log_det
        # tem = torch.sum(tem)
        # xi_entropy = 0.5*np.log(2*np.pi*np.exp(1))
        # z_entropy = tem+det_inverse*xi_entropy
        # print(tem,det_inverse,xi_entropy)
        loss = BCE
        total_loss += loss.data[0]
        count+=1
    avg = total_loss / count
    print(' ppl_avg :%g avg_loss:%g ' % (math.exp(avg),avg ))
    return avg

def z_opt(z_sample):

    opt = optim.SGD([z_sample], lr=lr, momentum = 1-alpha)

    return opt
def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    model.decoder.bsz = args.batch_size
    # hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # hidden = repackage_hidden(hidden)
        for j in range(J):
            if j == 0:
                model.zero_grad()
                recon_batch,z,xi = model(data)
                # jacobian = compute_jacobian(xi,z)
                z_sample = Variable(z.data,requires_grad = True)
                z_optimizer = z_opt(z_sample)
                z_optimizer.zero_grad()
            else:
                emb = model.embed(data)
                recon_batch = model.decoder(emb,z_sample)

            BCE = loss_function(recon_batch, targets)

            prior_loss = model.prior_loss(prior_std)
            noise_loss = model.noise_loss(lr,alpha)
            prior_loss /=args.bptt*len(train_data)
            noise_loss /=args.bptt*len(train_data)
            prior_loss_z = z_prior_loss(z_sample)
            noise_loss_z = z_noise_loss(z_sample)
            prior_loss_z /=args.bptt*len(train_data)
            noise_loss_z /=args.bptt*len(train_data)
            loss = BCE+ prior_loss+noise_loss+prior_loss_z+noise_loss_z
            if j>burnin:
                loss_en = en_loss(z_sample,z)
                loss += loss_en
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            z_optimizer.step()

        total_loss += BCE.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:5.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} '.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
alpha = 0.1
bit = 0
prior_std = 1
epoch_cut = 0
burnin = 0
J = 2
best_val_loss = None
optimizer = optim.SGD(model.parameters(), lr=lr,momentum = 1-alpha)
# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)
        test_loss = evaluate(test_data)
        if epoch > 50:
            print('save!')
            torch.save(model.state_dict(),'./bn/model_a0.1_ppl_2%i.pt'%(epoch))
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           test_loss, math.exp(test_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print('New Best!')
            # torch.save(model.state_dict(),'model.pt')
            best_val_loss = val_loss
        # else:
        #     print('decay!')
        #     # Anneal the learning rate if no improvement has been seen in the validation dataset.
        #     lr /= 2.0
        #     lr = max(lr,1)
        #     # alpha *= 0.9
        #     # alpha = min(0.2,alpha)
        #     optimizer = optim.SGD(model.parameters(), lr=lr,momentum = 1-alpha)
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
# model.load_state_dict(torch.load('model.pt'))

# Run on test data.
# test_loss = evaluate(test_data)
# print('=' * 89)
# print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#     test_loss, math.exp(test_loss)))
# print('=' * 89)