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

sys.path.append('../')
from vae import VAE
from ais import AIS
import torch.distributions as td
from hmc import HMC

parser = argparse.ArgumentParser(description='generative test ll using mu')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--file', type=str, default ='', help = 'file to run test loading on')
parser.add_argument('--zdim', type=int, default = 20, metavar = 'S',
                    help='latent + noise dimension to use in model')
parser.add_argument('--num-steps', type=int, default = 500, help = 'number of steps to run AIS for')
parser.add_argument('--num-samples', type=int, default = 16, help='number of chains to run AIS over')
parser.add_argument('--bayesian', action='store_true', default=False,
                    help='if marginalization of the model parameters will also be performed (only use for BVAE)')
args = parser.parse_args()

#cuda stuff
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
args.device = None
if args.cuda:
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

data_location = '../../data'
to_bernoulli = lambda x: transforms.ToTensor()(x).bernoulli()

#use a variant of test loader so we don't have to re-generate the batch stuff
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_location, train=False, transform=to_bernoulli),
    batch_size=10000, shuffle=True, **kwargs)
#collect one single batch for speed
for x in test_loader:
    test_tensor_list = x
class myiterator:
    def __iter__(self):
        return iter([test_tensor_list])
new_test_loader = myiterator()

#initialize model
model = VAE(zdim = args.zdim).to(args.device)

#reload model
print(args.cuda)
if args.cuda:
    loaded = torch.load(args.file)
else:
    loaded = torch.load(args.file, map_location=lambda storage, loc: storage)

print(loaded['epoch'])
model.load_state_dict(loaded['state_dict'])

#create the prior distribution for z
pmean = torch.zeros(model.zdim).to(args.device)
pstd = torch.ones(model.zdim).to(args.device)
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
    data, _ = data
    data = data.view(-1,784).to(args.device)

    #backwards pass ignores the data argument and samples generatively
    if backwards:
        #sample z generatively
        z = priordist.rsample(((data.size(0),)))

    #pass forwards 
    if not backwards:
        mu, logvar = model(data)    
    
        #generate distribution
        q_dist = td.Normal(mu, torch.exp(logvar.mul(0.5)))
    
        #reparameterize and draw a random sample
        z = q_dist.rsample()

    #pass backwards
    x_probs = model.decode(z)

    #create distributions
    l_dist = td.Bernoulli(probs = x_probs)

    #now the "backwards pass"
    if backwards:
        #draw simulated data_location
        data = l_dist.sample()
        mu, logvar = model(data)
        q_dist = td.Normal(mu, torch.exp(logvar.mul(0.5)))
        
    #now compute log(geometric average)
    recon_loss = l_dist.log_prob(data).sum(dim=1)
    gen_loss = recon_loss + priordist.log_prob(z).sum(dim=1)
    variational_loss = q_dist.log_prob(z).sum(dim=1)

    #marginalize out the parameters if bayesian 
    if args.bayesian is True:
        theta_prior_loss = 0.0
        for p in model.parameters():
            theta_prior_loss +=  td.Normal(torch.zeros_like(p), torch.ones_like(p)).log_prob(p).sum()
       
        total_loss = ((1 - t1) * variational_loss + t1 * gen_loss).sum() + theta_prior_loss/10000.
    else:
        total_loss = (1 - t1) * variational_loss + t1 * gen_loss
    return total_loss.sum()

#note, right now im using adam as the transition operator, which i won't be doing. 
#ill be using HMC in the future
sampler = HMC(model.parameters(), lr = 1e-6, L = 3)
ais_for_vae = AIS(model, new_test_loader, sampler, num_beta=args.num_steps, num_samples = args.num_samples, 
                            nprint=10, data_len = 1.)

#print('Now running forward AIS')
#logprob_est, logprob_vec, _ = ais_for_vae.run_forward(geom_average_loss)

#print('Lower bound estimate: ' + str(logprob_est/10000.))

print('Now running backward AIS')
blogprob_est, blogprob_vec, _ = ais_for_vae.run_backward(geom_average_loss)

print('Upper bound estimate: ' + str(blogprob_est/10000.))
