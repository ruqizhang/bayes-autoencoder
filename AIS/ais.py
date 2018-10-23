import numpy as np
import time

import torch
from torch.autograd import grad as torchgrad
from utils import log_normal, log_bernoulli, log_mean_exp, discretized_logistic, safe_repeat
from hmc import hmc_trajectory, accept_reject
from tqdm import tqdm


def ais_trajectory(model, loader, mode='forward', schedule=np.linspace(0., 1., 500), n_sample=100):
    """Compute annealed importance sampling trajectories for a batch of data. 
    Could be used for *both* forward and reverse chain in bidirectional Monte Carlo
    (default: forward chain with linear schedule).

    Args:
        model (vae.VAE): VAE model
        loader (iterator): iterator that returns pairs, with first component being `x`,
            second would be `z` or label (will not be used)
        mode (string): indicate forward/backward chain; must be either `forward` or 'backward'
        schedule (list or 1D np.ndarray): temperature schedule, i.e. `p(z)p(x|z)^t`;
            foward chain has increasing values, whereas backward has decreasing values
        n_sample (int): number of importance samples (i.e. number of parallel chains 
            for each datapoint)

    Returns:
        A list where each element is a torch.autograd.Variable that contains the 
        log importance weights for a single batch of data
    """

    assert mode == 'forward' or mode == 'backward', 'Should have either forward/backward mode'

    def log_f_i(z, emb, data, target, t, log_likelihood_fn=log_bernoulli):
        """Unnormalized density for intermediate distribution `f_i`:
            f_i = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
        =>  log f_i = log p(z) + t * log p(x|z)
        """

        log_prior = log_normal(z, torch.zeros_like(z), torch.zeros_like(z))
        #log_likelihood = log_likelihood_fn(model.decoder(emb, z).view_as(data), data)
        ## LL = -valid_loss * sentence_length
        log_likelihood = -model.criterion(model.decoder(emb, z), data, target, reduction='none').view_as(data).sum(dim=0) * data.size(0)
        #print(log_likelihood.size(), log_prior.size())
        
        return log_prior + log_likelihood.mul_(t)

    # shorter aliases
    try:
        z_size = model.hps.z_size
    except:
        try:
            z_size = model.z_dim
        except:
            z_size = model.zdim
    #mdtype = model.dtype

    _time = time.time()
    logws = []  # for output

    print ('In %s mode' % mode)

    for i, (batch, target) in enumerate(loader):
    #for (batch, _) in loader:
        batch, target = batch.cuda(), target.cuda()

        B = batch.size(-1) * n_sample
        batch = safe_repeat(batch, n_sample)

        """d0 = batch.size(0)
        d1 = B
        #print('data size: ', batch.size())
        data = torch.zeros(d0*d1,10000, device = batch.device)
        data[range(d0*d1),batch.view(-1)]=1
        data = data.view(d0,d1,10000)"""

        # batch of step sizes, one for each chain
        epsilon = torch.ones(B, device = batch.device).mul_(1.0)
        # accept/reject history for tuning step size
        accept_hist = torch.zeros(B, device = batch.device)

        # record log importance weight; volatile=True reduces memory greatly
        logw = torch.zeros(B, device = batch.device, requires_grad = False)
        #logw.requires_grad = False

        # initial sample of z
        if mode == 'forward':
            current_z = torch.randn(B, z_size, device = batch.device, requires_grad = True)
        else:
            current_z = safe_repeat(post_z, n_sample).type(batch.dtype).device(batch.device)
            current_z.requires_grad = True

        for j, (t0, t1) in tqdm(enumerate(zip(schedule[:-1], schedule[1:]), 1)):
            emb = model.embed(batch)

            # update log importance weight
            log_int_1 = log_f_i(current_z, emb, batch, target, t0).data
            log_int_2 = log_f_i(current_z, emb, batch, target, t1).data
            #print(log_int_1.mean(), log_int_2.mean())
            logw.data.add_(log_int_2 - log_int_1)

            del log_int_1, log_int_2

            # resample speed
            #current_v = torch.randn(current_z.size(), dtype=batch.dtype, device=batch.device, requires_grad = False)
            current_v = torch.ones_like(current_z).normal_()

            def U(z):
                return -log_f_i(z, emb, batch, target, t1)

            def grad_U(z):
                # grad w.r.t. outputs; mandatory in this case
                grad_outputs = torch.ones(B, device = batch.device)
                grad = torchgrad(U(z), z, grad_outputs=grad_outputs)[0]
                # clip by norm
                grad = torch.clamp(grad, -B*z_size*100, B*z_size*100)
                grad.requires_grad = True
                return grad

            def normalized_kinetic(v):
                zeros = torch.zeros(B, z_size, device = batch.device)
                # this is superior to the unnormalized version
                return -log_normal(v, zeros, zeros)

            z, v = hmc_trajectory(current_z, current_v, U, grad_U, epsilon)

            # accept-reject step
            current_z, epsilon, accept_hist = accept_reject(current_z, current_v,
                                                            z, v,
                                                            epsilon,
                                                            accept_hist, j,
                                                            U, K=normalized_kinetic)

        # IWAE lower bound
        logw = log_mean_exp(logw.view(n_sample, -1).transpose(0, 1))
        if mode == 'backward':
            logw = -logw

        logws.append(logw.data)

        print ('Time elapse %.4f, last batch stats %.4f' % (time.time()-_time, logw.mean().cpu().data.numpy()))
        _time = time.time()

    return logws
