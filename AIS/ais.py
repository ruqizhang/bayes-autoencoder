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

    def log_f_i(z, data, t, log_likelihood_fn=log_bernoulli):
        """Unnormalized density for intermediate distribution `f_i`:
            f_i = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
        =>  log f_i = log p(z) + t * log p(x|z)
        """
        zeros = torch.zeros(B, z_size, dtype = z.dtype, device = z.device)
        log_prior = log_normal(z, zeros, zeros)
        log_likelihood = log_likelihood_fn(model.decoder(z), data)

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

    for i, (batch, post_z) in enumerate(loader):

        B = batch.size(0) * n_sample
        batch = safe_repeat(batch, n_sample)
        # batch of step sizes, one for each chain
        epsilon = torch.ones(B, dtype = batch.dtype, device = batch.device).mul_(0.01)
        # accept/reject history for tuning step size
        accept_hist = torch.zeros(B, dtype = batch.dtype, device = batch.device)

        # record log importance weight; volatile=True reduces memory greatly
        logw = torch.zeros(B, dtype = batch.dtype, device = batch.device)
        logw.requires_grad = False

        # initial sample of z
        if mode == 'forward':
            current_z = torch.randn(B, z_size, dtype=batch.dtype, device = batch.device)
            #current_z = model.encode(batch)
            #current_z.detach_()
            current_z.requires_grad = True
        else:
            current_z = safe_repeat(post_z, n_sample).type(batch.dtype).device(batch.device)
            current_z.requires_grad = True

        for j, (t0, t1) in tqdm(enumerate(zip(schedule[:-1], schedule[1:]), 1)):
            # update log importance weight
            log_int_1 = log_f_i(current_z, batch, t0).data
            log_int_2 = log_f_i(current_z, batch, t1).data
            logw.data.add_(log_int_2 - log_int_1)

            del log_int_1, log_int_2

            # resample speed
            current_v = torch.randn(current_z.size(), dtype=batch.dtype, device=batch.device, requires_grad = False)

            def U(z):
                return -log_f_i(z, batch, t1)

            def grad_U(z):
                # grad w.r.t. outputs; mandatory in this case
                grad_outputs = torch.ones(B, dtype = batch.dtype, device = batch.device)
                grad = torchgrad(U(z), z, grad_outputs=grad_outputs)[0]
                # clip by norm
                grad = torch.clamp(grad, -B*z_size*100, B*z_size*100)
                grad.requires_grad = True
                return grad

            def normalized_kinetic(v):
                zeros = torch.zeros(B, z_size, dtype = batch.dtype, device = batch.device)
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
        print(logw.size())
        if mode == 'backward':
            logw = -logw

        logws.append(logw.data)

        print ('Time elapse %.4f, last batch stats %.4f' % (time.time()-_time, logw.mean().cpu().data.numpy()))
        _time = time.time()

    return logws
