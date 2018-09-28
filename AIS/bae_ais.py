#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 4:40 5/16/18
@author: wesleymaddox
working example of ais in practice on a pre-trained bayesian auto-encoder
"""

import torch
import argparse
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets, transforms
import sys

from ais import AIS
import torch.distributions as td
from hmc import HMC
from torch import nn, optim

import sys
sys.path.append('..')
from bae_model import BAE

parser = argparse.ArgumentParser(description='AIS with bayesian auto-encoder')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--file', type=str, default ='', help = 'file to run test loading on')
parser.add_argument('--zdim', type=int, default = 20, metavar = 'S',
                    help='latent + noise dimension to use in model')
parser.add_argument('--num-steps', type=int, default = 500, help = 'number of steps to run AIS for')
parser.add_argument('--num-samples', type=int, default = 16, help='number of chains to run AIS over')
parser.add_argument('--device', type=int, default = 0, help = 'device')
parser.add_argument('--data_path', type=str, default='/scratch/datasets/', help='location of mnist dataset')

args = parser.parse_args()
kwargs = {'num_workers': 1, 'pin_memory': True}

to_bernoulli = lambda x: transforms.ToTensor()(x).bernoulli()

#use a variant of test loader so we don't have to re-generate the batch stuff
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(args.data_path, train=False, transform=to_bernoulli, download=True),
    batch_size=10000, shuffle=True, **kwargs)

for (data, label) in test_loader:
    test_tensor_list = (data.to(args.device), label.to(args.device))
class myiterator:
    def __iter__(self):
        return iter([test_tensor_list])
new_test_loader = myiterator()

x_dim = 784
z_dim = 20
hidden_dim = 400

model = BAE(x_dim,z_dim,hidden_dim)

model.to(args.device)
model_state_dict = torch.load(args.file)
model.load_state_dict(model_state_dict)
#model.cuda()

#create the prior distribution for z
pmean = torch.zeros(z_dim).to(args.device)
pstd = torch.ones(z_dim).to(args.device)
priordist = torch.distributions.Normal(pmean, pstd)

def geom_average_loss(t1, data, backwards = False):
    """
    t1: scaling factor for the geometric average loss
    data: data tensor
    backwards: if we draw a simulated model and use that instead of the real data
    """
    #pass t1 to current device if necessary
    #t1 = t1.to(args.device)

    #want to calculate log(q(z|x)^(1-t1) p(x,z)^t1)
    data, _ = data
    data = data.view(-1, 784)
    #data = data.view(-1,784).to(args.device)

    #backwards pass ignores the data argument and samples generatively
    if backwards:
        z = priordist.rsample(((data.size(1),)))
    else:
        z = model.encode(data)

    #pass backwards
    x_probs = model.decode(z)

    #create distributions
    l_dist = td.Bernoulli(probs = x_probs)

    #now the "backwards pass"
    if backwards:
        #draw simulated data_location
        data = l_dist.sample()

    #now compute log(geometric average)
    recon_loss = l_dist.log_prob(data).sum(dim=1)

    prior_loss = priordist.log_prob(z).sum(dim=1)

    theta_loss = 0.0
    for i, param in enumerate(model.parameters()):
        #print(i, 'here')
        param_dist = td.Normal(torch.zeros_like(param), torch.ones_like(param))
        theta_loss += param_dist.log_prob(param).sum()

    #print(recon_loss.mul(t1).size(), prior_loss.size(), theta_loss.size())
    total_loss = (recon_loss.mul(t1) + prior_loss).sum() + theta_loss/10000.
    return total_loss.sum()

#note, right now im using adam as the transition operator, which i won't be doing.
#ill be using HMC in the future
sampler = HMC(model.parameters(), lr = 1e-6, L = 3)
ais_for_vae = AIS(model, new_test_loader, sampler, num_beta=args.num_steps, num_samples = args.num_samples,
                            nprint=10)

print('Now running forward AIS')
logprob_est, logprob_vec, _ = ais_for_vae.run_forward(geom_average_loss)
print(sampler.acc_rate())
print(logprob_vec)
print('Lower bound estimate: ' + str(logprob_est/10000.))

# print('Now running backward AIS')
# blogprob_est, blogprob_vec, _ = ais_for_vae.run_backward(geom_average_loss)

# print('Upper bound estimate: ' + str(blogprob_est/10000.))






