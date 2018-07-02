""" 
date: 1/3/18
general inference methods file, currently includes 
    Importance weighted loss (variant of standard KLqp loss)
        Note: K = 1 is the standard KLqp loss
    Variational Renyi loss (generalization of KLqp to minimize Renyi alpha divergences)
    Reweighted wake sleep
    perturbative variational inference loss (third order Taylor approximation of KLqp)
"""
import torch
import math

from LogSumExp import LogSumExp

#TODO: implement chivi

def IWAE(z, data, likelihood, prior, var_approx, K, reduce = True):
    r"""
    z: parameters that are being inferred
    data: minibatch
    K: number of samples
    
    NOTE: z, likelihood, var_approx must be lists due to multiple samples
    
    This is the IWAE bound of Burda et al. 2016
    """
    for k in range(K):
        #calculate log likelihoods
        l_lprob = likelihood[k].log_prob(data).sum(dim=1)
        p_lprob = prior[k].log_prob(z[k]).sum(dim=1)
        v_lprob = var_approx[k].log_prob(z[k]).sum(dim=1)
        logprob = l_lprob + p_lprob - v_lprob
        
        """print('L:',l_lprob.sum())
        print('P:',p_lprob.sum())
        print('V:',v_lprob.sum())"""
        if k is 0:
            logprob_total = logprob.view(-1,1)
        else:
            logprob_total = torch.cat((logprob_total, logprob.view(-1,1)),dim=1)
            
    loss = -LogSumExp(logprob_total,dim=1) + math.log(K)
    if reduce is True:
        return loss.sum()
    else:
        return loss

def VR(z, data, likelihood, prior, var_approx, K, alpha = 0, reduce = True):
    r"""
    z: parameters that are being inferred
    data: minibatch
    K: number of samples
    alpha: hyperparameter of Renyi alpha-divergence
    
    NOTE: z, likelihood, var_approx must be lists due to multiple samples
    
    This is the VR bound of Li and Turner 2016
    """
    if alpha == 1.0:
        return IWAE(z, data, likelihood, prior, var_approx, K, reduce)
    
    for k in range(K):
        #calculate log likelihoods
        l_lprob = likelihood[k].log_prob(data).sum(dim=1)
        p_lprob = prior[k].log_prob(z[k]).sum(dim=1)
        v_lprob = var_approx[k].log_prob(z[k]).sum(dim=1)
        logprob = l_lprob + p_lprob - v_lprob
            
        if k is 0:
            logprob_total = logprob.view(-1,1)
        else:
            logprob_total = torch.cat((logprob_total, logprob.view(-1,1)),dim=1)
        
    lse_sum = -LogSumExp((1-alpha) * logprob_total,dim=1) + math.log(K)

    loss = lse_sum/(1-alpha)
    if reduce is True:
        return loss.sum()
    else:
        return loss
    
def IWAE_RWS(wake, z, data, likelihood, prior, var_approx, K):
    r"""
    z: parameters that are being inferred
    data: minibatch
    K: number of samples
    alpha: hyperparameter of Renyi alpha-divergence
    
    This is the reweighted wake sleep method of Bornstein and Bengio 2015
    
    NOTE: z, likelihood, var_approx must be lists due to multiple samples
     
    fix the sleep phase
    """
    if wake:
        #wake phase is identical to the IWAE loss
        return IWAE_loss(z, data, likelihood, prior, var_approx, K)
    
    else:
        return -var_approx.log_prob(z).sum()
        """#in sleep stage, generate forwards from prior
        h_prime = prior.sample()
            
        #use prior sample to get simulated data
        x_sleep_probs = model.decode(h_prime)
        likelihood = Bernoulli(x_sleep_probs)
        x_simulated = Variable(likelihood.sample().data)
            
        #get new mean and std
        approx.mean, logvar = model.encode(x_simulated)
        approx.std = torch.exp(logvar.mul(0.5))
            
        v_lprob = approx.log_prob(h_prime).sum()
        
        return -v_lprob"""
            
def PVI(z, data, likelihood, prior, var_approx, K, alpha = 0):
    r"""
    z: parameters that are being inferred
    data: minibatch
    K: number of samples
    alpha: the V0 term for rescaling
    
    This is the surrogate objective (Eq 7) of Bamler et al 2017
    """
    for k in range(K):
        l_lprob = likelihood[k].log_prob(data).sum(dim=1)
        p_lprob = prior[k].log_prob(z[k]).sum(dim=1)
        v_lprob = var_approx[k].log_prob(z[k]).sum(dim=1)
        logprob = l_lprob + p_lprob - v_lprob
        V = -logprob
        
        if k is 0:
            currloss = 1 + (alpha - V) + 0.5 * (alpha - V).pow(2) + 1/6 * (alpha - V).pow(3)
            loss = currloss/K #mean contribution to loss
        else:
            currloss = 1 + (alpha - V) + 0.5 * (alpha - V).pow(2) + 1/6 * (alpha - V).pow(3)
            loss += currloss/K
    return -loss