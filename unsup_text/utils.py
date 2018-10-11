import torch
import copy
import math

import torch.nn.functional as F

import itertools

from torch.autograd import Variable

def save_model(epoch, model, optimizer, dir):
    print('saving model at epoch ', epoch)
    torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()},
        f = dir + '/ckpt_' + str(epoch+1) +'.pth.tar')

def loss_function(recon_batch, x, dim):
    recon_batch = recon_batch.view(-1,dim)
    BCE = F.cross_entropy(recon_batch, x)
    return BCE

def en_loss(z_recon,z):
    z = Variable(z.data,requires_grad = False)
    loss = F.mse_loss(z_recon,z)
    return loss

def z_prior_loss(z):
    prior_distribution = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
    prior_loss = -prior_distribution.log_prob(z).sum()
    return prior_loss

def z_noise_loss(z, lr, alpha):
    noise_std = (2 * lr * alpha) ** 0.5
    rand_like_z = torch.zeros_like(z).normal_() * noise_std

    noise_loss = torch.sum(z * rand_like_z)
    return noise_loss

def evaluate(data_source, model, dim):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    
    for i, (data, targets) in enumerate(data_source):

        recon_batch,z,_ = model(data)
        BCE = loss_function(recon_batch, targets, dim)

        loss = BCE
        total_loss += loss.item()

    avg = total_loss / i
    print(' ppl_avg :%g avg_loss:%g ' % (math.exp(avg),avg ))
    return avg

def z_opt(z_sample, lr, alpha):

    opt = torch.optim.SGD([z_sample], lr=lr, momentum = 1-alpha)

    return opt

