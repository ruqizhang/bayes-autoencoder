import torch
from torch import nn
from torchvision.utils import save_image
from mlp import MLP, Linear2
import torch.distributions

class mlpBAE(nn.Module):
    #currently designed for un-supervised learning
    #assume a 10-d noise dimension
    #can have separate optimizers for self.deoder.parameters(), self.encoder.parameters()
    #so steps can be separate
    def __init__(self, dim = 784, noise_dim = 10, zdim = 50, hidden = 500, activation = nn.ReLU):
        super(BAE, self).__init__()
        self.dim, self.zdim, self.hidden, self.noise_dim = dim, zdim, hidden, noise_dim
        self.encoder = MLP(dim+noise_dim, hidden, zdim, activation = activation)
        self.decoder = MLP(zdim, hidden, dim, activation = activation)
        
    def encode(self, x):
        return self.encoder(x.view(-1, self.dim + self.noise_dim))
    
    def decode(self, z):
        return nn.Sigmoid()(self.decoder(z))
    
    def forward(self, x):
        x = x.view(-1,self.dim)

        mu, logvar = self.encode(x)
        return mu, logvar  

class mlpSSBAE(nn.Module):
    def __init__(self, dim = 784, noise_dim = 10, zdim = 50, hidden = 500, activation = nn.ReLU):
        super(Ruqi_VAE, self).__init__()
        self.dim, self.zdim, self.hidden, self.noise_dim = dim, zdim, hidden, noise_dim
        self.encoder = MLP(dim+noise_dim, hidden, zdim, out_layer = Linear2, activation = activation)
        self.decoder = MLP(zdim, hidden, dim, activation = activation)
        
    def encode(self, x):
        return self.encoder(x.view(-1, self.dim + self.noise_dim))
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        x = x.view(-1,self.dim)

        mu, logvar = self.encode(x)
        return mu, logvar   