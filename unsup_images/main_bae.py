# coding: utf-8
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import models
import torch.optim as optim
import numpy as np
from torch.autograd.gradcheck import zero_gradients
import sys, os
sys.path.append('..')
import data, utils
parser = argparse.ArgumentParser(description='PyTorch PTB RNN/LSTM Language Model')
parser.add_argument('--dataset', type = str, default = 'MNIST')
parser.add_argument('--data_path', type=str, default='./datasets/ptb',
                    help='location of the MNIST dataset ')
parser.add_argument('--model', type=str, default='BAE')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--zdim', type=int, default=20,
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float, default=1,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--no-cuda', action='store_true',
                    help='do not use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save-epochs', type=int, default=25, metavar='N',
                    help='how often to save the model')
parser.add_argument('--dir', type=str, required = True)
#parser.add_argument('--device_id',type = int, help = 'device id to use')
args = parser.parse_args()
#device_id = args.device_id

###############################################################################
# Load data and prepare model
###############################################################################
#set cuda if available, while also setting seed
torch.manual_seed(args.seed)
args.device = None
if torch.cuda.is_available() and not args.no_cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

#define the model
print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Preparing model')
print(*model_cfg.args)
print('using ', args.zdim, ' latent space')
model = model_cfg.base(*model_cfg.args, noise_dim=args.zdim, zdim=args.zdim, **model_cfg.kwargs)
model.cuda()

#prepare the dataset
print('Loading dataset %s from %s' % (args.dataset, args.data_path))
loaders = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    4,
    model_cfg.transform_train,
    model_cfg.transform_test,
    use_validation=not args.use_test,
    unsup=True
)
datasize = len(loaders['train'].dataset)

#make directory
print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
os.makedirs(args.dir+'/results/', exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write('python ' + ' '.join(sys.argv))
    f.write('\n')

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def loss_function(recon_batch, x):
    # sum(p(x_i | \theta, z))
    #x = x.view(-1,784)
    #BCE = F.cross_entropy(recon_batch, x)
    l_dist =  torch.distributions.Bernoulli(probs = recon_batch)
    likelihood = -l_dist.log_prob(x).sum(dim=1).mean()
    #print(likelihood.item())
    return likelihood

def en_loss(z_recon,z):
    z = Variable(z.data,requires_grad = False)
    loss = F.mse_loss(z_recon,z)
    return loss

def z_prior_loss(z):
    # sum(p(z_i))
    # assume a standard normal prior here
    prior_loss = 0.5*torch.sum(z*z)
    return prior_loss

def z_noise_loss(z):

    learning_rate = lr

    noise_std = np.sqrt(2*learning_rate*alpha)
    noise_std = torch.from_numpy(np.array([noise_std])).float().cuda()
    noise_std = noise_std[0]
    random_var = torch.ones_like(z).normal_() * noise_std
    noise_loss = torch.sum(z * random_var)
    #means = torch.zeros(z.size()).cuda()
    #noise_loss = torch.sum(z * Variable(torch.normal(means, std = noise_std).cuda(),
    #                       requires_grad = False))
    return noise_loss

def evaluate(data_source, epoch, dir):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    total_kld = 0
    #ntokens = len(corpus.dictionary)
    count=0
    # hidden = model.init_hidden(eval_batch_size)
    for i, (data,_) in enumerate(data_source):
        data = data.view(-1, 784).cuda()
        if i==0 and epoch%args.save_epochs is 0:
            model.reconstruct_samples(data, epoch = epoch, dir = dir)
        #data, targets = get_batch(data_source, i)
        #model.decoder.bsz = eval_batch_size
        recon_batch,z,xi = model(data)
        BCE = loss_function(recon_batch, data)

        loss = BCE #+ (z_prior_loss(z).sum() / z.size(0))

        total_loss += loss.item()
        count+=1
    avg = total_loss / count
    #print(' ppl_avg :%g avg_loss:%g ' % (math.exp(avg),avg ))
    return avg

def z_opt(z_sample):
    opt = optim.SGD([z_sample], lr=lr, momentum = 1-alpha)
    return opt

def train(data_source):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()

    for i, (data, _) in enumerate(data_source):
        data = data.view(-1, 784).cuda()

        for j in range(J):
            if j == 0:
                model.zero_grad()
                recon_batch,z,noise = model(data)
                z_sample = Variable(z.data,requires_grad = True)
                z_optimizer = z_opt(z_sample)
                z_optimizer.zero_grad()
            else:
                model.zero_grad()
                z_optimizer.zero_grad()
                recon_batch = model.decoder(z_sample)

            BCE = loss_function(recon_batch, data)

            prior_loss = model.prior_loss(prior_std) / datasize
            noise_loss = model.noise_loss(lr,alpha) / datasize
            
            prior_loss_z = z_prior_loss(z_sample) / z_sample.size(0)
            noise_loss_z = z_noise_loss(z_sample) / z_sample.size(0)
            
            #print(BCE, prior_loss, noise_loss, prior_loss_z, noise_loss_z)
            loss = BCE #+ prior_loss + noise_loss + prior_loss_z + noise_loss_z

            if j>burnin:
                loss_en = en_loss(z_sample,z)
                loss += loss_en
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            z_optimizer.step()

        total_loss += loss.item()

        if i % args.log_interval == 0 and i > 0:
            #print(loss.item())
            #cur_loss = total_loss / i
            cur_loss = loss.item()
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:5.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f}'.format(
                epoch, i, int(datasize/args.batch_size), lr,
                elapsed * 1000 / args.log_interval, cur_loss))
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
        train(loaders['train'])
        #val_loss = evaluate(val_data)
        #print('-' * 89)
        #print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
        #        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
        #                                   val_loss, math.exp(val_loss)))
        print('-' * 89)
        test_loss = evaluate(loaders['test'], epoch = epoch, dir = args.dir+ '/results')
        if epoch%args.save_epochs is 0:
            print('saving model')
            #torch.save(model.state_dict().clone().cpu(), 
            utils.save_model(epoch, model, optimizer, args.dir)
            model.generate_samples(dir = args.dir +'/results', epoch=epoch)
            
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                .format(epoch, (time.time() - epoch_start_time),
                                           test_loss))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        #if not best_val_loss or val_loss < best_val_loss:
        #    print('New Best!')
            # torch.save(model.state_dict(),'model.pt')
        #    best_val_loss = val_loss
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')