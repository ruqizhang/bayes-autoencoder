import torch
import torch.distributions as td

def log_prob_conjugate_form_poisson(theta, covar, response, a0, tau=1.0):
    r"""
    theta: random vector (possibly batched)
    covar: covariance matrix of observed data
    response: response vector
    a0: prior scaling factor
    tau: known scaling factor
    
    form of the D(..) conjugate prior as in Chen & Ibrahim Stat. Sin. 2003
    the log-density is of the form
    a0 tau (response^T covar.theta - sum(exp(-covar.theta))
    """

    #covar.theta
    linear_pred = covar.matmul(theta.t())

    #sum(exp(-covar.theta))
    bfun = torch.sum(torch.exp(-linear_pred), dim=0)

    #response^T covar.theta
    reg_pred = torch.matmul(response.squeeze(), linear_pred)

    logprob = a0 * tau * (reg_pred - bfun)

    return logprob

def construct_simulated_data(N = 100, beta = [1, -1.], seed = 1):
    """
    N: number of data points
    beta: true global parameters
    seed: rng seed to be set for repoducibility

    y \sim poisson(exp((x,z).beta))

    returns y, x, z
    """

    torch.manual_seed(seed)

    # y \sim U(-1, 1)
    y = 2. * torch.rand(N) - 1.
    
    # z \sim N(0, 1)
    z = torch.ones(N).normal_()

    # canonical link: log(lambda) = M \beta
    loglambda = beta[0] * y + beta[1] * z

    # note tau is 1 in this example
    poissondist = td.poisson.Poisson(torch.exp(loglambda))
    x = poissondist.sample()

    return x, y, z

def prof_beta_cond_trueZ(x, y, a0 = 1., ngrid = 100, seed = 2):

    #prior marginal mean is 0
    x0 = torch.zeros_like(x).normal_()
    
    # z \sim N(0, 1): the true latent distribution
    z = torch.randn(x.size(0))

    # M = (y, z)
    M = torch.stack((y, z),dim=1)

    # an = a0 + 1
    aprime = a0 + 1.

    # xn = 1/an * (x + a0 * x0)
    response_post = (x + a0 * x0)/aprime

    # create beta sequence and grid of betas
    beta_start = -4.
    step = (-2*beta_start)/ngrid
    beta_sequence = torch.arange(beta_start, -beta_start, step)
    beta_tuple = torch.meshgrid([beta_sequence, beta_sequence])
    beta_grid = torch.stack((beta_tuple[0].contiguous().view(-1), beta_tuple[1].contiguous().view(-1)),1)

    # then compute log probability over beta
    logprob_vector = log_prob_conjugate_form_poisson(beta_grid, covar=M, response=response_post, a0=aprime)

    return logprob_vector, beta_sequence, beta_grid

    
#test to ensure this is working
def test_logprob():
    theta = torch.randn(4,2) 
    x = torch.randn(100, 2)
    y = torch.randn(100, 1)

    a0 = torch.ones(1)
    tau = torch.ones(1)

    #output should be a four dimensional vector
    print('log prob test: ')
    print(log_prob_conjugate_form_poisson(theta, x, y, a0, tau))

def test_data():
    print('Simulation test: ')
    print( construct_simulated_data(N=10))

def test_prof_beta(N = 100, ngrid = 10):
    x, y, z = construct_simulated_data(N=N)
    #grid, vals = prof_beta_cond_trueZ(x, y, ngrid=10)
    vals, seq, grid = prof_beta_cond_trueZ(x, y, ngrid = ngrid)
    #print('beta test')
    #print(vals)
    return vals, seq

#test_logprob()
#test_data()
import matplotlib.pyplot as plt
vals, seq = test_prof_beta(N = 500, ngrid=500)

CS = plt.contourf(seq.numpy(), seq.numpy(), vals.contiguous().view(seq.size(0), seq.size(0)).numpy(), 20)
nm, lbl = CS.legend_elements()
plt.legend(nm, lbl, title= 'MyTitle', fontsize= 8) 
plt.xlabel('beta1')
plt.ylabel('beta2')
plt.show()
