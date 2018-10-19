from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

class BAE(nn.Module):
    def __init__(self,x_dim,z_dim,hidden_dim, device_id=0):
        super(BAE, self).__init__()
        self.x_dim = x_dim
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, x_dim)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.device_id = device_id

    def encode(self, x):
        #xi = Variable(torch.randn(x.size()).cuda(self.device_id),requires_grad=False)
        xi = torch.ones_like(x).normal_()
        h1 = self.relu(self.fc1(x+xi))
        z = self.fc21(h1)
        return z

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          #eps = Variable(std.data.new(std.size()).normal_())
          eps = torch.ones_like(std.data).normal_()
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
            noise_loss += torch.sum(var * noise_std * torch.zeros_like(var).normal_())
            """                noise_loss += torch.sum(var * Variable(torch.from_numpy(np.random.normal(0, noise_std, size=var.size())).float().cuda(device_id),
                               requires_grad = False))
            else:
                noise_loss += torch.sum(var * Variable(torch.from_numpy(np.random.normal(0, noise_std, size=var.size())).float(),
                               requires_grad = False))"""
        noise_loss /= datasize
        #print(noise_loss)#1e-8
        return noise_loss

    def forward(self, x):
        z = self.encode(x.view(-1, self.x_dim))
        recon_x = self.decode(z)
        return recon_x,z
