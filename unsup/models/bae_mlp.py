import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.utils import save_image
from .mlp import *

import PIL
__all__ = ['BAE_MLP']

class baeMLP(nn.Module):
    #currently designed for un-supervised learning
    def __init__(self, dim = 784, noise_dim = 50, zdim = 50, nhidden = 500, activation = nn.ReLU, **kwargs):
        super(baeMLP, self).__init__()
        self.dim, self.zdim, self.hidden, self.noise_dim = dim, zdim, nhidden, noise_dim
        self.encode = MLP(dim+noise_dim, nhidden, zdim, activation = activation)
        self.decode = MLP(zdim, nhidden, dim, activation = activation)
        
    def encoder(self, x):
        noise = torch.ones(x.size(0), self.noise_dim, device=x.device, dtype=x.dtype).normal_()
        x_new = torch.cat((x.view(-1, self.dim), noise),dim=1)
        return self.encode(x_new), noise
    
    def decoder(self, z):
        return nn.Sigmoid()(self.decode(z))
    
    def forward(self, x, z = None):
        x = x.view(-1,self.dim)

        if z is None:
            z, noise = self.encoder(x)
        else:
            noise = None
            
        reconstructed_batch = self.decoder(z)
        return reconstructed_batch, z, noise
    
    def prior_loss(self,prior_std):
        prior_loss = 0.0
        for var in self.parameters():
            prior_dist = torch.distributions.Normal(torch.zeros_like(var), prior_std * torch.ones_like(var))
            prior_loss += -prior_dist.log_prob(var).sum()
        return 0.5*prior_loss

    def noise_loss(self,lr,alpha):
        noise_loss = 0.0
        noise_std = (2.0 * lr * alpha)**0.5
        for var in self.parameters():
            rand_like_var = torch.zeros_like(var).normal_() * noise_std
            noise_loss += torch.sum(var * rand_like_var)
        return noise_loss
    
    def reconstruct_samples(self, data, dir, epoch, **kwargs):
        z_gen, _ = self.encoder(data.view(-1, self.dim))
    
        recon_batch = self.decoder(z_gen)
        n = min(data.size(0), 8)
        comparison = torch.cat([data.view(-1, 1, 28, 28)[:n],
                    recon_batch.view(data.size(0), 1, 28, 28)[:n]])
        save_image(comparison.data.cpu(),
             dir + '/results/reconstruction_' + str(epoch) + '.png', nrow=n)
    
    def generate_samples(self, dir, epoch):
        #sample = z_prior.sample((64,))
        sample = torch.randn(64, self.zdim).cuda()
        
        sample = self.decoder(sample).cpu()
        save_image(sample.data.view(64, 1, 28, 28),
            dir + '/results/sample_' + str(epoch) + '.png')  

    def criterion(self, recon, data, target):
        BCE = torch.nn.functional.binary_cross_entropy(recon, data.view_as(recon), reduction='sum')
        BCE /= recon.size(0)
        return BCE


class BAE_MLP:
    args = list()
    kwargs = {'activation': nn.ReLU}

    base = baeMLP

    #transform_train = lambda x: transforms.ToTensor()(x).bernoulli() if type(x)==PIL.Image.Image else x.bernoulli()
    transform_train = lambda x: transforms.ToTensor()(x).bernoulli()
    
    """def transform_train(x):
        print('before: ', type(x))
        if type(x)==torch.Tensor:
            x = transforms.ToPILImage()(x.numpy())

        print('after: ', type(x))
        return transforms.ToTensor()(x).bernoulli()"""
    transform_test = transform_train
