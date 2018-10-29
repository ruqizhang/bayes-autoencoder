import torch
import torch.nn as nn

from torchvision import transforms
from torchvision.utils import save_image
from .mlp import *

__all__ = ['VAE_MLP']
    
class vaeMLP(nn.Module):
    #standard unsupervised vae architecture
    #re-parameterization takes place using Normal().rsample()
    def __init__(self, dim = 784, zdim = 50, hidden = 500, activation = nn.ReLU, **kwargs):
        super(vaeMLP, self).__init__()
        self.dim, self.zdim, self.hidden = dim, zdim, hidden
        self.encoder = MLP(dim, hidden, zdim, out_layer = Linear2, activation = activation)
        self.decoder = MLP(zdim, hidden, dim, activation = activation)
        
    #def encoder(self, x):
    #    return self.encode(x.view(-1, self.dim))
    
    def decode(self, z):
        return nn.Sigmoid()(self.decoder(z))
    
    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          #eps = Variable(std.data.new(std.size()).normal_())
          eps = torch.zeros_like(std).normal_()
          #eps.cuda(self.device_id)
          return eps.mul(std).add_(mu)
        else:
          return mu

    def forward(self, input, z = None):
        #emb = self.drop(self.word_embeddings(input))
        #mu,logvar = self.encoder(emb)
        mu, logvar = self.encoder(input)

        if z is None:
            z = self.reparameterize(mu, logvar)
            #print('rsampled ll: ', torch.distributions.Normal(mu, logvar.exp()).log_prob(z).sum(dim=1).mean())
            recon_batch = self.decode(z)
            return recon_batch, mu,logvar
        else:
            recon_batch = self.decode(z)
            return recon_batch, z, None

    def criterion(self, recon, data, target, **kwargs):
        BCE = torch.nn.functional.binary_cross_entropy(recon, data.view_as(recon), **kwargs)
        BCE /= recon.size(0)
        return BCE

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

class VAE_MLP:
    args = list()
    kwargs = {'activation':nn.ReLU}
    base = vaeMLP

    transform_train = lambda x: transforms.ToTensor()(x).bernoulli()
    transform_test = transform_train