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
#sys.path.append('/home/wm326/')
# sys.path.append('../../')
#from bvae.vae_images.inferences_general import VR_loss
from ais import AIS
import torch.distributions as td
from hmc import HMC
from torch import nn, optim
parser = argparse.ArgumentParser(description='AIS with bayesian auto-encoder')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--file', type=str, default ='', help = 'file to run test loading on')
parser.add_argument('--zdim', type=int, default = 20, metavar = 'S',
                    help='latent + noise dimension to use in model')
parser.add_argument('--num-steps', type=int, default = 500, help = 'number of steps to run AIS for')
parser.add_argument('--num-samples', type=int, default = 16, help='number of chains to run AIS over')
parser.add_argument('--device', type=int, default = 0, help = 'device')

args = parser.parse_args()
kwargs = {'num_workers': 1, 'pin_memory': True}

data_location = './data'
to_bernoulli = lambda x: transforms.ToTensor()(x).bernoulli()
#use a variant of test loader so we don't have to re-generate the batch stuff
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_location, train=False, transform=to_bernoulli),
    batch_size=10000, shuffle=True, **kwargs)
for x in test_loader:
    test_tensor_list = x
class myiterator:
    def __iter__(self):
        return iter([test_tensor_list])
new_test_loader = myiterator()
class VAE(nn.Module):
    def __init__(self,x_dim,z_dim,hidden_dim):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, x_dim)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        xi = Variable(torch.randn(x.size()).to(args.device),requires_grad=False)
        h1 = self.relu(self.fc1(x+xi))
        z = self.fc21(h1)
        return z

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def prior_loss(self,std):
        prior_loss = 0.0
        for var in self.parameters():
            nn = torch.div(var, std)
            prior_loss += torch.sum(nn*nn)

        prior_loss /= datasize
        #print(prior_loss)#1e-3
        return 0.5*prior_loss

    def noise_loss(self,lr,alpha):
        noise_loss = 0.0
        # learning_rate = base_lr * np.exp(-lr_decay *min(1.0, (train_iter*args.batch_size)/float(datasize)))
        # noise_std = 2*learning_rate*alpha
        noise_std = 2*lr*alpha
        for var in self.parameters():
            if args.cuda:
                noise_loss += torch.sum(var * Variable(torch.from_numpy(np.random.normal(0, noise_std, size=var.size())).float().to(args.device),
                               requires_grad = False))
            else:
                noise_loss += torch.sum(var * Variable(torch.from_numpy(np.random.normal(0, noise_std, size=var.size())).float(),
                               requires_grad = False))
        noise_loss /= datasize
        #print(noise_loss)#1e-8
        return noise_loss

    def forward(self, x):
        z = self.encode(x.view(-1, self.x_dim))
        recon_x = self.decode(z)
        return recon_x,z
x_dim = 784
z_dim = 20
hidden_dim = 400

model = VAE(x_dim,z_dim,hidden_dim)
model.to(args.device)
model.load_state_dict(torch.load('./results/bn_model.pt'))

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
    t1 = t1.to(args.device)

    #want to calculate log(q(z|x)^(1-t1) p(x,z)^t1)
    data, _ = data
    data = data.view(-1,784).to(args.device)

    #backwards pass ignores the data argument and samples generatively
    if backwards:
        #sample z generatively
        z = priordist.rsample(((data.size(1),)))
    else:
        #perform forwards pass through model if "forwards"
        #add noise
        # noise_mean = torch.zeros(data.size(0), args.zdim).to(args.device)
        # noise_std = torch.ones(data.size(0), args.zdim).to(args.device)

        # noise_sample = td.Normal(noise_mean, noise_std).rsample()

        # augmented_data = torch.cat((data, noise_sample),dim=1)
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
    for name,param in model.named_parameters():
        if 'fc1' in name or 'fc21' in name:
            param_dist = td.Normal(torch.zeros_like(param), torch.ones_like(param))
            theta_loss += param_dist.log_prob(param).sum()

    total_loss = t1 * recon_loss + prior_loss + theta_loss/10000.
    return total_loss.sum()

#note, right now im using adam as the transition operator, which i won't be doing.
#ill be using HMC in the future
sampler = HMC(model.parameters(), lr = 1e-6, L = 3)
ais_for_vae = AIS(model, new_test_loader, sampler, num_beta=args.num_steps, num_samples = args.num_samples,
                            nprint=10)

print('Now running forward AIS')
logprob_est, logprob_vec, _ = ais_for_vae.run_forward(geom_average_loss)
print(sampler.acc_rate())
print('Lower bound estimate: ' + str(logprob_est/10000.))

# print('Now running backward AIS')
# blogprob_est, blogprob_vec, _ = ais_for_vae.run_backward(geom_average_loss)

# print('Upper bound estimate: ' + str(blogprob_est/10000.))






