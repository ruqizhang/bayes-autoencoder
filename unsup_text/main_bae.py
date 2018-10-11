# coding: utf-8
import argparse
import time
import math,os,sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import data
import models
from models.bae import BAE
import torch.optim as optim
import numpy as np
from itertools import starmap, cycle

import utils

from torch.autograd.gradcheck import zero_gradients
parser = argparse.ArgumentParser(description='PyTorch PTB RNN/LSTM Language Model')
parser.add_argument('--dataset', type=str, default='ptb', 
                    help = 'dataset to use (only ptb is implemented currently)')
parser.add_argument('--data_path', type=str, default='./datasets/ptb',
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
parser.add_argument('--save_epochs', type = int, help = 'how often to save the model')
parser.add_argument('--dir', type = str, default = 'directory to save model to')

args = parser.parse_args()
device_id = args.device_id
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
args.cuda = True

###############################################################################
# Load data
###############################################################################
loaders, ntokens = data.loaders(args.dataset, args.data_path, args.batch_size, args.bptt, args.cuda)

#make directory
print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
os.makedirs(args.dir+'/results/', exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write('python '.join(sys.argv))
    f.write('\n')

###############################################################################
# Build the model
###############################################################################

model = BAE.base(ntokens, args.emsize, args.nhid, args.zdim,args.nlayers, args.batch_size, args.dropout, args.tied)
if args.cuda:
    model.cuda()

def train(loader):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch, (data, targets) in enumerate(loader):

        for j in range(J):
            if j == 0:
                optimizer.zero_grad()
                recon_batch,z,_ = model(data)

                z_sample = Variable(z.data,requires_grad = True)
                z_optimizer = utils.z_opt(z_sample, lr, alpha)
                z_optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                z_optimizer.zero_grad()

                if type(model)==models.bae.BAE_LSTM:
                    emb = model.embed(data)
                    recon_batch = model.decoder(emb,z_sample)
                else:
                    recon_batch = model.decoder(z_sample)

            BCE = utils.loss_function(recon_batch, targets, ntokens)

            prior_loss = model.prior_loss(prior_std) 
            noise_loss = model.noise_loss(lr,alpha)
            prior_loss /= len(loader)
            noise_loss /= len(loader)

            prior_loss_z = utils.z_prior_loss(z_sample)
            noise_loss_z = utils.z_noise_loss(z_sample, lr, alpha)
            prior_loss_z /= z_sample.size(0)
            noise_loss_z /= z_sample.size(0)

            loss = BCE + prior_loss + noise_loss + prior_loss_z + noise_loss_z
            if j>burnin:
                loss_en = utils.en_loss(z_sample,z)
                loss += loss_en
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            z_optimizer.step()

        total_loss += BCE.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:5.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} '.format(
                epoch, batch, len(loader.dataset) // args.bptt, lr,
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
best_val_loss = 1e10
optimizer = optim.SGD(model.parameters(), lr=lr,momentum = 1-alpha)
# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()

        #train
        train(loaders['train'])

        #validate
        with torch.no_grad():
            val_loss = utils.evaluate(loaders['valid'], model, ntokens)
            torch.cuda.synchronize()
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)

            #test
            test_loss = utils.evaluate(loaders['test'], model, ntokens)
            torch.cuda.synchronize()

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                    'test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            test_loss, math.exp(test_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                print('New Best!')
                best_val_loss = val_loss

            if epoch % args.save_epochs is 0:
                print('saving model')
                utils.save_model(epoch, model, optimizer, args.dir)
            
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')