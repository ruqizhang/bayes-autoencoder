import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from torch.autograd import Variable

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
        noise = torch.ones(x.size(0), self.noise_dim, device=x.device, dtype=x.dtype).normal_()
        x_new = torch.cat((x.view(-1, self.dim), noise),dim=1)
        return self.encoder(x_new), noise
    
    def decode(self, z):
        return nn.Sigmoid()(self.decoder(z))
        #return self.decoder(z) #returns a logit
    
    def forward(self, x):
        x = x.view(-1,self.dim)

        z, noise = self.encode(x)
        reconstructed_batch = self.decode(z)
        return reconstructed_batch, z, noise
    
    def prior_loss(self,prior_std):
        prior_loss = 0.0
        for var in self.parameters():
            nn = torch.div(var, prior_std)
            prior_loss += torch.sum(nn*nn)
        #print('prior_loss',prior_loss)#1e-3
        return 0.5*prior_loss

    def noise_loss(self,lr,alpha):
        noise_loss = 0.0
        # learning_rate = base_lr * np.exp(-lr_decay *min(1.0, (train_iter*args.batch_size)/float(datasize)))
        learning_rate = lr
        noise_std = np.sqrt(2*learning_rate*alpha)
        noise_std = torch.from_numpy(np.array([noise_std])).float().cuda()
        noise_std = noise_std[0]
        for var in self.parameters():
            #means = torch.zeros(var.size()).cuda()
            #means = torch.zeros_like(var)
            random_var = torch.ones_like(var).normal_() * noise_std
            #noise_loss += torch.sum(var * Variable(torch.normal(means, std = noise_std).cuda(),
            #                   requires_grad = False))
            noise_loss += torch.sum(var * random_var)
        #print('noise_loss',noise_loss)#1e-8
        return noise_loss 
    
    def reconstruct_samples(self, data, dir, epoch, **kwargs):
        z_gen, _ = self.encode(data.view(-1, self.dim))
    
        recon_batch = self.decode(z_gen)
        n = min(data.size(0), 8)
        comparison = torch.cat([data.view(-1, 1, 28, 28)[:n],
                    recon_batch.view(data.size(0), 1, 28, 28)[:n]])
        save_image(comparison.data.cpu(),
             dir + '/reconstruction_' + str(epoch) + '.png', nrow=n)
    
    def generate_samples(self, dir, epoch):
        #sample = z_prior.sample((64,))
        sample = torch.randn(64, self.zdim).cuda()
        
        sample = self.decode(sample).cpu()
        save_image(sample.data.view(64, 1, 28, 28),
                dir + '/sample_' + str(epoch) + '.png') 

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