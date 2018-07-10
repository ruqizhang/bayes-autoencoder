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
import inferences

__all__=['VAE', 'M2VAE', 'BVAE', 'M2BVAE']

def compute_prior_loss(model, scale = 1.0):
    loss = 0.0
    for param in model.parameters():
        param_dist = torch.distributions.Normal(torch.zeros_like(param), scale * torch.ones_like(param))
        loss -= param_dist.log_prob(param).sum()

    return loss

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

    def loss(self, data, K=1, alpha = 1, iw_function=inferences.VR, ss = False):
        priordist = torch.distributions.Normal(torch.zeros(data[0].size(0), self.zdim).to(self.device), torch.ones(data[0].size(0), self.zdim).to(self.device))
        prior_x = [priordist] * K
        z = [None] * K
        l_dist = [None] * K
        q_dist = [None] * K
        
        #pass forwards
        if not ss:
            mu, logvar = self.forward(data)  
        else:
            mu, logvar, logits = self.forward(data[0], data[1])  
        
        for k in range(K):
            #generate distribution
            q_dist[k] = torch.distributions.Normal(mu, torch.exp(logvar.mul(0.5)))
            
            #reparameterize
            z[k] = q_dist[k].rsample()
        
            #pass backwards
            if not ss:
                x_probs = self.decode(z[k])
            else:
                x_probs = self.decode(z[k], data[1])
        
            #create distributions
            l_dist[k] = torch.distributions.Bernoulli(probs = x_probs)
            
        #compute loss function
        loss = iw_function(z, data[0].view(-1,784), l_dist, prior_x, q_dist, K = K, alpha = alpha)

        if not ss:
            return loss
        else:
            return loss, logits

class baseBVAE(baseVAE):
    def __init__(self, dim = 784, zdim = 50, hidden = 500, activation = nn.ReLU):
        super(baseBVAE, self).__init__(dim, zdim, hidden, activation)

    def loss(self, data, **kwargs):
        main_loss = super(baseBVAE, self).loss(data, **kwargs)

        prior_loss = compute_prior_loss(self) 

        return main_loss + prior_loss/len(data) 

class baseM2VAE(baseVAE):
    def __init__(self, dim = 784, zdim = 50, hidden = 500, nclasses = 10, activation = nn.ReLU):
        super(baseM2VAE, self).__init__(dim, zdim, hidden, activation)
        self.dim, self.zdim, self.hidden, self.nclasses = dim, zdim, hidden, nclasses
        self.encoder_y_real = MLP(dim, hidden, nclasses, activation = activation)
        #we're now using self.encoder from the super-class
        #self.encoder_z = MLP(dim, hidden, zdim, out_layer = Linear2, activation = activation)
        self.decoder = MLP(zdim+nclasses, hidden, dim, activation = activation)
        
        self.norm_classifier_outputs = nn.LogSoftmax(dim=1)

        self.args = list()
        self.kwargs = dict()

    def encode(self, x):
        return self.encoder(x.view(-1, self.dim))
    
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

    def loss(self, data, y = None, K = 1, alpha = 1, iw_function=inferences.VR, weight=300., num_classes=10):
        if y is not None:
            ul_loss, logits = super(baseM2VAE, self).loss((data, y), K, alpha, iw_function, ss=True)

            c_dist = torch.distributions.OneHotCategorical(logits = logits)
            c_loss = - weight * c_dist.log_prob(y)

            total_loss = ul_loss.view(-1) + c_loss
            secondary_loss = c_loss.sum()
        else:
            secondary_loss = None
            total_loss = 0.0
            for yy in range(num_classes):
                #create on-hot vector
                ycurrent = torch.zeros(data.size(0), num_classes)
                ycurrent[:,yy] = 1
                ycurrent.to(self.device)

                ul_loss, logits = super(baseM2VAE, self).loss((data, ycurrent), K, alpha, iw_function, ss=True)
                y_dist = torch.distributions.OneHotCategorical(logits = logits)
                y_lprob = y_dist.log_prob(ycurrent)

                total_loss += torch.exp(y_lprob) * (ul_loss.view(-1) - y_lprob)
        return total_loss.sum(), secondary_loss

    def calc_accuracy(self, data, y_one_hot, yvec):
        r"""
        simple function for computing accuracy
        """
        mu, logvar, logits = self.forward(data, y_one_hot)
        _, pred = torch.max(logits, dim = 1)
        misclass = (pred.data.long() - yvec.long()).ne(int(0)).cpu().long().sum()
        return misclass.item()/data.size(0)

class baseM2BVAE(baseM2VAE):
    def __init__(self, dim = 784, zdim = 50, hidden = 500, nclasses = 10, activation = nn.ReLU):
        super(baseM2BVAE, self).__init__(dim, zdim, hidden, nclasses, activation)
        
    def loss(self, data, y, **kwargs):
        main_loss, secondary_loss = super(baseM2BVAE, self).loss(data, y, **kwargs)

        prior_loss = compute_prior_loss(self)  
        return main_loss + prior_loss/len(data), secondary_loss

class VAE:
    args = list()
    kwargs = {'dim': 784, 'hidden':500}
    base = baseVAE

    transform_train = lambda x: transforms.ToTensor()(x).bernoulli()
    transform_test = transform_train

class BVAE(VAE):
    def __init__(self):
        super(BVAE, self).__init__()
    base = baseBVAE

class M2VAE:
    args = list()
    kwargs = {'dim': 784, 'hidden':500}
    base = baseM2VAE

    transform_train = lambda x: transforms.ToTensor()(x).bernoulli()
    transform_test = transform_train

class M2BVAE(VAE):
    def __init__(self):
        super(M2VAE, self).__init__()
    base = baseM2BVAE    
