import torch
from torch import nn
from torchvision.utils import save_image
from .mlp import MLP, Linear2
import torch.distributions
from torchvision import transforms
__all__ = ['BAE', 'SSBAE']

class mlpBAE(nn.Module):
    #currently designed for un-supervised learning
    #assume a 10-d noise dimension
    #can have separate optimizers for self.deoder.parameters(), self.encoder.parameters()
    #so steps can be separate
    def __init__(self, dim = 784, noise_dim = 10, zdim = 50, hidden = 500, activation = nn.ReLU, 
                J = 2, burnin = 2, batch_size = 100, gibbs = False):
        super(mlpBAE, self).__init__()

        self.dim, self.zdim, self.hidden, self.noise_dim = dim, zdim, hidden, noise_dim
        self.J, self.burnin, self.gibbs = J, burnin, gibbs

        #construct networks
        self.encoder = MLP(dim+noise_dim, hidden, zdim, activation = activation)
        self.decoder = MLP(zdim, hidden, dim, activation = activation)
        
        #phi_loss_fn
        self.phi_loss_fn = torch.nn.MSELoss()

        #hidden units
        #register parameter
        self.z = torch.nn.Parameter(torch.zeros(batch_size, self.zdim))
        self.z_prior = None
        #prior distribution on z
        #self.z_prior = torch.distributions.Normal(torch.zeros(self.zdim).to(self.device), torch.ones(self.zdim).to(self.device))
        #self.param_dist = torch.distributions.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))

    def encode(self, x):
        return self.encoder(x.view(-1, self.dim + self.noise_dim))
    
    def decode(self, z):
        return nn.Sigmoid()(self.decoder(z))
    
    def forward(self, x):
        x = x.view(-1,self.dim)

        mu, logvar = self.encode(x)
        return mu, logvar  

    def loss(self, data, y=None, ss=False, **kwargs):
        if not self.training:
            return self.test_loss(data, y=y, ss=ss, **kwargs)

        self.z_prior = torch.distributions.Normal(torch.zeros(self.zdim).to(self.device), torch.ones(self.zdim).to(self.device))
        
        def joint_posterior(z):
            #likelihood 
            x_probs = self.decode(z)
            l_dist = torch.distributions.Bernoulli(probs = x_probs)
            del x_probs
            likelihood = -l_dist.log_prob(data.view(-1,self.dim)).sum(dim=1)
            
            #prior on z
            z_prior_loss = -self.z_prior.log_prob(z).sum(dim=1)
            
            #prior on theta
            theta_loss = 0.0
            for param in self.encoder.parameters():
                param_dist = torch.distributions.Normal(torch.zeros_like(param), torch.ones_like(param))
                theta_loss += -param_dist.log_prob(param).sum()
            
            #rescale for minibatches
            self.scaling = 55000.0 #figure out where this goes
            loss = (self.scaling * (likelihood + z_prior_loss).sum(dim=0,keepdim=True) + theta_loss)/data.size(0)
            
            del likelihood, z_prior_loss, theta_loss
            return loss
        
        #create noise distribution
        noise_mean = torch.zeros(data.size(0), self.noise_dim).to(self.device)
        noise_std = torch.ones(data.size(0), self.noise_dim).to(self.device)         
        noise_sample = torch.distributions.Normal(noise_mean, noise_std).rsample()
        
        #augment data by appending noise
        augmented_data = torch.cat((data.view(-1,784), noise_sample),dim=1)
        
        #update z with current decoder
        z_gen = self.encode(augmented_data)
        self.z.data = z_gen.data
        """tmp = z_gen.clone()
        sampler.param_groups[0]['params'][0].data = tmp.data
        self.z = sampler.param_groups[0]['params'][0]       
        del tmp, augmented_data, noise_sample, noise_mean, noise_std"""

        #consecutive updates here
        encoder_post_loss = 0.0
        model_post_loss = 0.0
        for j in range(self.J + self.burnin):       
            self.sampler.zero_grad()
            curr_post = joint_posterior(self.z)
            curr_post.backward(retain_graph = True)
            self.sampler.step()
            
            if j>=self.burnin and ss:
                if y is not None:
                    self.sampler.zero_grad()
                    #y_optim.zero_grad()
                    aug_input = torch.cat((self.z, data.view(-1, self.dim)),dim=1)
                    y_logits = self.y_model(aug_input)
                    labeled_loss = -torch.distributions.OneHotCategorical(logits = y_logits).log_prob(y).sum()
                    labeled_loss.backward(retain_graph = True)
                    #y_optim.step()
                    self.sampler.step()

                    #store predictions so we know how accurate we are
                    pred_matrix = torch.zeros(len(y), self.J).long()
                    _, pred = torch.max(y_logits, dim = 1)
                    pred_matrix[:,(j-self.burnin)] = pred.data.long()

            if not self.gibbs:
                #store z if j > burn-in
                if j >= self.burnin:
                    #phi_optim.zero_grad()
                    self.sampler.zero_grad()
                    #z_curr = Variable(z.data, requires_grad = False)
                    z_curr = self.z.data.detach()
                    encoder_loss = self.phi_loss_fn(z_gen, z_curr)
                    encoder_loss.backward(retain_graph = True)
            
                    #phi_optim.step()
                    self.sampler.step()
                    
                    model_post_loss += curr_post
                    encoder_post_loss += encoder_loss
                    
                    del encoder_loss, z_curr
            else:
                pass
                #TODO: write gibbs version
            del curr_post
        
        del z_gen
        del data  
        
            
        if ss:
            self.misclass = torch.zeros(1)
            if y is not None and ss:
                #get mode of prediction matrix
                pred_mode, _ = torch.mode(pred_matrix.cpu(),dim = 1)
                yvec = torch.nonzero(y)[:,1].data.cpu()

                self.misclass = (pred_mode - yvec).ne(int(0)).sum()
        #return model_post_loss/self.J, self.scaling * encoder_post_loss/self.J

        if not ss:
            return (model_post_loss + self.scaling * encoder_post_loss)/self.J
        else:
            return model_post_loss/self.J, self.scaling * encoder_post_loss/self.J#, misclass.item()


    def test_loss(self, data, y=None, ss=False, **kwargs):
        data = data.view(-1,self.dim)
        torch.cuda.empty_cache()
        
        noise_mean, noise_std = torch.zeros(data.size(0), self.noise_dim).to(self.device), torch.ones(data.size(0), self.noise_dim).to(self.device)
        
        #add random noise to sample, but force noise to be differentiable for jacobian
        noise_dist = torch.distributions.Normal(noise_mean, noise_std)
        noise_sample = noise_dist.sample()  
        if self.z_prior is None:
            self.z_prior = torch.distributions.Normal(torch.zeros(self.zdim).to(self.device), torch.ones(self.zdim).to(self.device))

        def test_from_augmented(augmented_data):
            #produce z'
            z_gen = self.encode(augmented_data)
            
            #compute generative loss
            probs = self.decode(z_gen)
            lik_loss = torch.distributions.Bernoulli(probs = probs).log_prob(data).sum(dim=1)
            prior_loss = self.z_prior.log_prob(z_gen).sum(dim=1)
            
            generative_loss = -(lik_loss + prior_loss)
        
            total_loss = generative_loss.sum()
            del probs, lik_loss, prior_loss
            return total_loss, z_gen
        
        #compute both zero mean loss & noisy loss
        aug_noise_data = torch.cat((data, noise_sample),dim=1)
        aug_zero_data = torch.cat((data, noise_mean), dim= 1)
        
        noisy_loss, z_gen = test_from_augmented(aug_noise_data)
        noiseless_loss, _ = test_from_augmented(aug_zero_data)

        if ss:
            aug_input = torch.cat((z_gen, data),dim=1)
            y_logits = self.y_model(aug_input)
            _, pred = torch.max(y_logits, dim = 1)
            
            correct = y[torch.arange(data.size(0)).long(), pred.data].sum()
            self.misclass = 1.0 - correct/data.size(0)
    
        #print('Zero Mean loss: ', noisy_loss.data.item()/data.size(0))
        #print('Noisy loss: ', noiseless_loss.data.item()/data.size(0))
        #print(-lik_loss.cpu().data.sum(), -prior_loss.cpu().data.sum())
        
        #garbage collection    
        torch.cuda.empty_cache()
        del noise_mean, noise_std, noise_dist, noise_sample
        del aug_noise_data, aug_zero_data
        
        if not ss:
            return noisy_loss
        else:
            return noisy_loss, 0.0

    def reconstruct_samples(self, data, y, dir, epoch, **kwargs):
        noise_mean = torch.zeros(data.size(0), self.noise_dim).to(self.device)
        noise_std = torch.ones(data.size(0), self.noise_dim).to(self.device)         
        noise_sample = torch.distributions.Normal(noise_mean, noise_std).rsample()

        augmented_data = torch.cat((data.view(-1, 784), noise_sample),dim=1)
        
        z_gen = self.encode(augmented_data)
        
        recon_batch = self.decode(z_gen)

        n = min(data.size(0), 8)
        comparison = torch.cat([data[:n],
                            recon_batch.view(data.size(0), 1, 28, 28)[:n]])
        save_image(comparison.data.cpu(),
                dir +  '/reconstruction_' + str(epoch) + '.png', nrow=n)
        
        #save samples only every 1000 epochs
        if self.zdim <= 3 and epoch%1000 is 0:
            import numpy
            print('saving model at ', epoch)
            z_ymat = torch.cat((z_gen.cpu().data, y.float().view(-1,1)),dim=1).numpy()
            numpy.savetxt(dir + '/saved_samples_'+ str(epoch)+'saved_samples.csv', z_ymat, delimiter = ',')
    
    def generate_samples(dir, epoch):
        sample = self.z_prior.sample((64,))
        
        sample = self.decode(sample).cpu()
        save_image(sample.data.view(64, 1, 28, 28),
                    dir + '/sample_' + str(epoch) + '.png')  

class mlpSSBAE(mlpBAE):
    def __init__(self, dim = 784, noise_dim = 10, zdim = 50, hidden = 500, nclasses = 10, activation = nn.ReLU):
        super(mlpSSBAE, self).__init__(dim, noise_dim, zdim, hidden)
        #self.dim, self.zdim, self.hidden, self.noise_dim = dim, zdim, hidden, noise_dim
        #self.encoder = MLP(dim+noise_dim, hidden, zdim, out_layer = Linear2, activation = activation)
        #self.decoder = MLP(zdim, hidden, dim, activation = activation)

        self.y_model = torch.nn.Sequential(torch.nn.Linear(self.zdim+self.dim, 100), 
                              torch.nn.ReLU(),
                              torch.nn.Linear(100, nclasses),
                              torch.nn.LogSoftmax(dim=1))
        
    def calc_accuracy(self, *argv, **kwargs):
        return self.misclass.item()
    """def encode(self, x):
        return self.encoder(x.view(-1, self.dim + self.noise_dim))
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        x = x.view(-1,self.dim)

        mu, logvar = self.encode(x)
        return mu, logvar  """ 

class BAE:
    args = list()
    kwargs = {'dim': 784, 'hidden':500}
    base = mlpBAE

    transform_train = lambda x: transforms.ToTensor()(x).bernoulli()
    transform_test = transform_train

class SSBAE:
    args = list()
    kwargs = {'dim': 784, 'hidden':500}
    base = mlpSSBAE

    transform_train = lambda x: transforms.ToTensor()(x).bernoulli()
    transform_test = transform_train