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
###############################################################################
# Build the model
###############################################################################

model = BAE.base(ntokens, args.emsize, args.nhid, args.zdim,args.nlayers, args.batch_size, args.dropout, args.tied)
if args.cuda:
    model.cuda()



def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

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
        BCE = utils.loss_function(recon_batch, targets, ntokens)

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
                optimizer.zero_grad()
                recon_batch,z,xi = model(data)
                z_sample = Variable(z.data,requires_grad = True)
                z_optimizer = z_opt(z_sample)
                z_optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                z_optimizer.zero_grad()

                emb = model.embed(data)
                recon_batch = model.decoder(emb,z_sample)

            BCE = utils.loss_function(recon_batch, targets, ntokens)

            prior_loss = model.prior_loss(prior_std)
            noise_loss = model.noise_loss(lr,alpha)
            prior_loss /=args.bptt*len(train_data)
            noise_loss /=args.bptt*len(train_data)
            prior_loss_z = utils.z_prior_loss(z_sample)
            noise_loss_z = utils.z_noise_loss(z_sample, lr, alpha)
            prior_loss_z /= z_sample.size(0)
            noise_loss_z /= z_sample.size(0)
            loss = BCE+ prior_loss+noise_loss+prior_loss_z+noise_loss_z
            if j>burnin+1:
                loss_en = utils.en_loss(z_sample,z)
                loss += loss_en
            if j%2==0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
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
J = 4
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
            torch.save(model.state_dict(),'./bn/model_gibbs_a0.1_ppl_2%i.pt'%(epoch))
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           test_loss, math.exp(test_loss)))
        print('-' * 89)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or test_loss < best_val_loss:
            print('New Best!')
            # torch.save(model.state_dict(),'model.pt')
            best_val_loss = test_loss
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
