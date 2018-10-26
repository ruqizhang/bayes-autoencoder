import numpy as np
import time

import torch
from torch.autograd import grad as torchgrad
from utils import log_normal, log_bernoulli, log_mean_exp, discretized_logistic, safe_repeat, z_prior_loss
from hmc import hmc_trajectory, accept_reject
from tqdm import tqdm

#import sys
#sys.path.append('..')

def ais_trajectory(model, loader, mode='forward', schedule=np.linspace(0., 1., 500), n_sample=100, prior_fn=lambda z: -z_prior_loss(z)):
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

    def log_f_i(z, data, target, t, prior_fn=prior_fn):
        """Unnormalized density for intermediate distribution `f_i`:
            f_i = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
        =>  log f_i = log p(z) + t * log p(x|z)
        """
        #zeros = torch.zeros(B, z.size(1), dtype = z.dtype, device = z.device)
        #log_prior = log_normal(z, zeros, zeros).sum()
        #log_prior = -z_prior_loss(z)
        log_prior = -prior_fn(z, data) / (z.size(0) * z.size(1) * data.size(0))
        #log_prior/= (z.size(0) * z.size(1) * data.size(0))

        model_output = model.forward(data, z)
        recon_batch = model_output[0]
        log_likelihood = -model.criterion(recon_batch, data, target, reduction='none').view_as(data).mean(0)
        #print(log_likelihood)
        
        #likelihood_prior =  -z_prior_loss(z)/(z.size(0) * z.size(1) * data.size(0))
        likelihood_prior = z.pow(2.0).sum(dim=1)/(z.size(0) * z.size(1) * data.size(0))
        log_joint_likelihood = log_likelihood + likelihood_prior
        #log_joint_likelihood = likelihood_prior
        #log_prior = torch.zeros_like(log_joint_likelihood)
        #print(log_prior.sum().cpu().item(), likelihood_prior.sum().cpu().item(), log_likelihood.sum().cpu().item())
        return log_prior.mul(1-t) + log_joint_likelihood.mul(t)

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
        #print(batch.size(), target.size())
        B = batch.size(1) * n_sample
        batch = safe_repeat(batch, n_sample)
        # batch of step sizes, one for each chain
        epsilon = torch.ones(B, device = batch.device).mul_(1e-4)
        # accept/reject history for tuning step size
        accept_hist = torch.zeros(B, device = batch.device)

        # record log importance weight; volatile=True reduces memory greatly
        logw = torch.zeros(B, device = batch.device, requires_grad = False)
        #logw.requires_grad = False

        # initial sample of z
        if mode == 'forward':
            #initialize with the mean of the encoder
            #current_z = torch.randn(B, z_size, device = batch.device, requires_grad = True)
            current_z = model.forward(batch)[1]
        else:
            current_z = safe_repeat(post_z, n_sample).type(batch.dtype).device(batch.device)
            current_z.requires_grad = True

        for j, (t0, t1) in tqdm(enumerate(zip(schedule[:-1], schedule[1:]), 1)):
            model.zero_grad()

            with torch.no_grad():
                # update log importance weight
                log_int_1 = log_f_i(current_z, batch, target, t0).data
                log_int_2 = log_f_i(current_z, batch, target, t1).data
                logw.data.add_(log_int_2 - log_int_1)

                del log_int_1, log_int_2

            # resample speed
            current_v = torch.randn(current_z.size(), device=batch.device, requires_grad = False) * 0.0

            def U(z):
                return -log_f_i(z, batch, target, t1)

            def grad_U(z):
                # grad w.r.t. outputs; mandatory in this case
                grad_outputs = torch.ones(B, device = batch.device)
                #grad_outputs = torch.ones_like(z)
                grad = torchgrad(U(z), z, grad_outputs=grad_outputs)[0]
                #grad = torch.autograd.grad(U(z), z)[0]
                # clip by norm
                grad = torch.clamp(grad, -B*z_size*100, B*z_size*100)
                grad.requires_grad = True
                #print(grad.norm())
                return grad

            def normalized_kinetic(v):
                zeros = torch.zeros(B, z_size, device = batch.device)
                # this is superior to the unnormalized version
                return -log_normal(v, zeros, zeros)

            z, v = hmc_trajectory(current_z, current_v, U, grad_U, epsilon)
            #print('norm of difference: ',  (current_z - z).norm() )
            # accept-reject step
            current_z, epsilon, accept_hist = accept_reject(current_z, current_v,
                                                            z, v,
                                                            epsilon,
                                                            accept_hist, j,
                                                            U, K=normalized_kinetic)
            #print('acceptance rate: ', accept_hist.mean()/(j+1) )
            #print('mean epsilon:', epsilon.mean())
            """with torch.no_grad():
                model.eval()
                out,_,_ = model(batch, current_z)
                print('eval p(x|z): ', model.criterion(out, batch, target).cpu().item(), current_z.var().cpu().item())
                model.train()"""
            
        # IWAE lower bound
        print(logw)
        #raise(EnvironmentError('done'))
        logw = log_mean_exp(logw.view(n_sample, -1).transpose(0, 1))
        print(logw.size())
        if mode == 'backward':
            logw = -logw

        logws.append(logw.data)

        print ('Time elapse %.4f, last batch stats %.4f' % (time.time()-_time, logw.mean().cpu().data.numpy()))
        _time = time.time()

    return logws
