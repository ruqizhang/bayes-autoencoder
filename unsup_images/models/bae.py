import torch
import torch.nn as nn

from torchvision import transforms

from .mlp import *

__all__ = ['BAE', 'BAE_SS']

class BayesAE(nn.Module):
    #currently designed for un-supervised learning
    #can have separate optimizers for self.deoder.parameters(), self.encoder.parameters()
    #so steps can be separate
    def __init__(self, dim = 784, noise_dim = 50, zdim = 50, hidden = 500, activation = nn.ReLU):
        super(BayesAE, self).__init__()
        self.dim, self.zdim, self.hidden, self.noise_dim = dim, zdim, hidden, noise_dim
        self.encoder = MLP(dim+noise_dim, hidden, zdim, activation = activation)
        self.decoder = MLP(zdim, hidden, dim, activation = activation)
        
    def encode(self, x):
        x_new = torch.cat((x, torch.ones(x.size(0), self.noise_dim, device=x.device, dtype=x.dtype).normal_()),dim=1)
        return self.encoder(x_new)
    
    def decode(self, z):
        return nn.Sigmoid()(self.decoder(z))
        #return self.decoder(z)
    
    def forward(self, x):
        x = x.view(-1,self.dim)

        mu, logvar = self.encode(x)
        return mu, logvar  

class BayesAE_SS(nn.Module):
    def __init__(self, dim = 784, noise_dim = 10, zdim = 50, hidden = 500, activation = nn.ReLU):
        super(BayesAE_SS, self).__init__()
        
        self.dim, self.zdim, self.hidden, self.noise_dim = dim, zdim, hidden, noise_dim

        self.encoder = MLP(dim+noise_dim, hidden, zdim, out_layer = Linear2, activation = activation)
        self.decoder = MLP(zdim, hidden, dim, activation = activation)
        
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        x = x.view(-1,self.dim)

        mu, logvar = self.encode(x)
        return mu, logvar  

class BAE:
    args = list()
    kwargs = {'hidden': 500, 'activation': nn.ReLU}

    base = BayesAE

    transform_train = lambda x: transforms.ToTensor()(x).bernoulli()
    transform_test = transform_train

class BAE_SS:
    args = list()
    kwargs = {'hidden': 500, 'activation': nn.ReLU}

    base = BayesAE_SS

    transform_train = lambda x: transforms.ToTensor()(x).bernoulli()
    transform_test = transform_train