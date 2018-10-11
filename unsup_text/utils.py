import torch
import copy
import math, time
import itertools

import torch.nn.functional as F
from torch.autograd import Variable

import models

def z_opt(z_sample, lr, alpha):

    opt = torch.optim.SGD([z_sample], lr=lr, momentum = 1-alpha)

    return opt

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

def train(epoch, loader, model, optimizer, dim, lr, alpha, J, burnin, prior_std, clip, log_interval, bptt, gibbs = False):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    epoch_start_time = start_time

    for batch, (data, targets) in enumerate(loader):

        for j in range(J):
            if j == 0:
                optimizer.zero_grad()
                recon_batch,z,_ = model(data)

                z_sample = Variable(z.data,requires_grad = True)
                z_optimizer = z_opt(z_sample, lr, alpha)
                z_optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                z_optimizer.zero_grad()

                if type(model)==models.bae.BAE_LSTM:
                    emb = model.embed(data)
                    recon_batch = model.decoder(emb,z_sample)
                else:
                    recon_batch = model.decoder(z_sample)

            BCE = loss_function(recon_batch, targets, dim)

            prior_loss = model.prior_loss(prior_std) 
            noise_loss = model.noise_loss(lr,alpha)
            prior_loss /= len(loader)
            noise_loss /= len(loader)

            prior_loss_z = z_prior_loss(z_sample)
            noise_loss_z = z_noise_loss(z_sample, lr, alpha)
            prior_loss_z /= z_sample.size(0)
            noise_loss_z /= z_sample.size(0)

            loss = BCE + prior_loss + noise_loss + prior_loss_z + noise_loss_z
            
            if gibbs:
                if j>burnin+1:
                    loss_en = en_loss(z_sample,z)
                    loss += loss_en
                if j%2==0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    z_optimizer.step()
            else:
                if j>burnin:
                    loss_en = en_loss(z_sample,z)
                    loss += loss_en
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            z_optimizer.step()

        total_loss += BCE.data

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:5.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} '.format(
                epoch, batch, len(loader.dataset) // bptt, lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

