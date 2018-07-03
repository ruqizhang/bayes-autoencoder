"""
    vae and M2 vae model definition
    from pytorch source
"""

import torch, torchvision
import numpy

from torch import nn
import torch.distributions
from torchvision.utils import save_image
from torchvision import transforms

from .mlp import MLP, Linear2

__all__=['VAE', 'M2VAE']

class baseVAE(nn.Module):
    #standard unsupervised vae architecture
    #re-parameterization takes place using Normal().rsample()
    def __init__(self, dim = 784, zdim = 50, hidden = 500, activation = nn.ReLU):
        super(baseVAE, self).__init__()
        self.dim, self.zdim, self.hidden = dim, zdim, hidden
        self.encoder = MLP(dim, hidden, zdim, out_layer = Linear2, activation = activation)
        self.decoder = MLP(zdim, hidden, dim, activation = activation)
        
    def encode(self, x):
        return self.encoder(x.view(-1, self.dim))
    
    def decode(self, z):
        return nn.Sigmoid()(self.decoder(z))
    
    def forward(self, x):
        x = x.view(-1,self.dim)

        mu, logvar = self.encode(x)
        return mu, logvar 

    def generate_samples(self, dir, epoch, priordist=None):
        if priordist is None:
            priordist = torch.distributions.Normal(torch.zeros(self.zdim), torch.ones(self.zdim)).to(self.device)

        sample = priordist.sample((64,))

        sample = self.decode(sample).cpu()
        save_image(sample.data.view(64, 1, 28, 28),
             dir + 'sample_' + str(epoch) + '.png') 
    
    def reconstruct_samples(self, data, y, dir, epoch):
        batch_size = data.size(0)
        mu, _ = self.forward(data)

        #eps = mu #use mean for reconstruction
        recon_batch = self.decode(mu)

        n = min(data.size(0), 8)
        comparison = torch.cat([data[:n],
                            recon_batch.view(batch_size, 1, 28, 28)[:n]])
        save_image(comparison.data.cpu(),
                dir +  '/reconstruction_' + str(epoch) + '.png', nrow=n)  

        #save samples only every 1000 epochs
        if self.zdim <= 3:
            print('saving model samples at ', epoch)
            z_ymat = torch.cat((mu.data.cpu(), y.float().view(-1,1)),dim=1).numpy()
            numpy.savetxt(dir + 'saved_samples_' + str(epoch) +'.csv', z_ymat, delimiter = ',')
    
class baseM2VAE(nn.Module):
    def __init__(self, dim = 784, zdim = 50, hidden = 500, nclasses = 10, activation = nn.ReLU):
        super(baseM2VAE, self).__init__()
        self.dim, self.zdim, self.hidden, self.nclasses = dim, zdim, hidden, nclasses
        
        self.encoder_y_real = MLP(dim, hidden, nclasses, activation = activation)
        self.encoder_z = MLP(dim, hidden, zdim, out_layer = Linear2, activation = activation)
        self.decoder = MLP(zdim+nclasses, hidden, dim, activation = activation)
        
        self.norm_classifier_outputs = nn.LogSoftmax(dim=1)

        self.args = list()
        self.kwargs = dict()

    def encode(self, x):
        return self.encoder_z(x.view(-1, self.dim))
    
    def decode(self, z, y):
        inp = torch.cat((z.view(-1,self.zdim), y.view(-1,self.nclasses)),dim=1)
        decoder_real = self.decoder(inp)
        return nn.Sigmoid()(decoder_real)
    
    def forward(self, x, y):
        x, y = x.view(-1,self.dim), y.view(-1,self.nclasses)
        
        y_probs_unnorm = self.encoder_y_real(x)
        logits = self.norm_classifier_outputs(y_probs_unnorm)
        
        mu, logvar = self.encode(x)
        return mu, logvar, logits 

class VAE:
    args = list()
    kwargs = {'dim': 784, 'hidden':500}
    base = baseVAE

    transform_train = lambda x: transforms.ToTensor()(x).bernoulli()
    transform_test = transform_train

class M2VAE:
    args = list()
    kwargs = {'dim': 784, 'hidden':500}
    base = baseM2VAE

    transform_train = lambda x: transforms.ToTensor()(x).bernoulli()
    transform_test = transform_train
