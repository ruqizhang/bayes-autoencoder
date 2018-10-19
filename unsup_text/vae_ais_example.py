#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 11:56 5/15/18
@author: wesleymaddox
working example of ais in practice on a pre-trained model
"""

import torch
import argparse
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets, transforms
import sys
import model_v
import model_b
from ais import AIS
import torch.distributions as td
from hmc import HMC
import data
parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
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
args = parser.parse_args()
device_id = args.device_id
args.cuda = True
torch.manual_seed(args.seed)
to_bernoulli = lambda x: transforms.ToTensor()(x).bernoulli()
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
# train_data = batchify(corpus.train, args.batch_size)
# val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)
test_data = test_data[0:35,:]
ntokens = len(corpus.dictionary)
model = model_b.VAE(args.model, ntokens, args.emsize, args.nhid, args.zdim,args.nlayers,device_id,args.batch_size, args.dropout, args.tied)
model.cuda(device_id)

model.load_state_dict(torch.load('./b/model_v_3.pt'))

#create the prior distribution for z
pmean = torch.zeros(args.zdim).cuda(device_id)
pstd = torch.ones(args.zdim).cuda(device_id)
priordist = torch.distributions.Normal(pmean, pstd)

def geom_average_loss(t1, data, backwards = False):
    """
    t1: scaling factor for the geometric average loss
    data: data tensor
    backwards: if we draw a simulated model and use that instead of the real data
    """
    #want to calculate log(q(z|x)^(1-t1) p(x,z)^t1)
    #backwards pass ignores the data argument and samples generatively
    if backwards:
        #sample z generatively
        z = priordist.rsample(((data.size(1),)))

    #pass forwards
    if not backwards:
        recon_batch, mu,logvar = model(data)

        #generate distribution
        q_dist = td.Normal(mu, torch.exp(logvar.mul(0.5)))

        #reparameterize and draw a random sample
        z = q_dist.rsample()

    #pass backwards
    emb = model.drop(model.word_embeddings(data))
    x_probs = model.decoder(emb,z)

    #create distributions
    l_dist = td.Bernoulli(probs = x_probs)

    #now the "backwards pass"
    if backwards:
        #draw simulated data_location
        data = l_dist.sample()
        d0 = data.size(0)
        d1 = data.size(1)
        value,idx = torch.max(data.view(-1,ntokens),dim =1)
        data = idx.view(d0,d1)
        recon_batch, mu,logvar = model(data)
        q_dist = td.Normal(mu, torch.exp(logvar.mul(0.5)))

    #now compute log(geometric average)
    d0 = data.size(0)
    d1 = data.size(1)
    datam = torch.zeros(d0*d1,10000).cuda(device_id)
    datam[range(d0*d1),data.view(-1)]=1
    datam = datam.view(d0,d1,10000)
    gen_loss = l_dist.log_prob(datam).sum(dim=2).sum(dim=0) + priordist.log_prob(z).sum(dim=1)
    variational_loss = q_dist.log_prob(z).sum(dim=1)
    t1 = t1.cuda(device_id)
    total_loss = (1 - t1) * variational_loss + t1 * gen_loss
    return total_loss.sum()
#note, right now im using adam as the transition operator, which i won't be doing.
#ill be using HMC in the future
sampler = HMC(model.parameters(), lr = 1e-6, L = 3)
ais_for_vae = AIS(model, test_data, sampler, num_beta=100, num_samples = 2, nprint=10)

print('Now running forward AIS')
logprob_est, logprob_vec, _ = ais_for_vae.run_forward(geom_average_loss)
print(sampler.acc_rate())
print(logprob_est)
print('Lower bound estimate: ' + str(logprob_est/args.batch_size*1.0/ntokens))

# print('Now running backward AIS')
# blogprob_est, blogprob_vec, _ = ais_for_vae.run_backward(geom_average_loss)

# print('Upper bound estimate: ' + str(blogprob_est/args.batch_size*1.0/ntokens))
