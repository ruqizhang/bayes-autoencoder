"""
new version of ais using the sghmc_optimizer

re-written based off of the old ais code and the sghmc_optimizer code

fix this documentation

"""

#from sghmc_optimizer import SGHMC
import torch
import numpy as np
import math
import copy
from torch.autograd import Variable

def LogSumExp(x,dim=0):
    m,_ = torch.max(x,dim=dim,keepdim=True)
    return m + torch.log((x - m).exp().sum(dim=dim, keepdim=True))

def get_schedule(num, rad=4):
    #copied from ais generative model paper's code
    #geometric averages
    if num == 1:
        return np.array([0.0, 1.0])
    t = np.linspace(-rad, rad, num)
    s = 1.0 / (1.0 + np.exp(-t))
    return (s - np.min(s)) / (np.max(s) - np.min(s))

class AIS(object):
    def __init__(self, model, dataset, optimizer = None,
                 data_len = None, num_samples = 5, num_beta = 500, nprint = 25):

        #store required things
        self.model = model
        self.model_init_params = copy.deepcopy(model.state_dict())
        self.dataset = dataset

        if data_len is None:
            self.data_len = 1
        else:
            self.data_len = data_len
        self.num_samples = num_samples
        self.num_beta = num_beta
        self.nprint = nprint

        #store schedule and reverse schedule
        self.schedule = get_schedule(num_beta)
        self.rev_schedule = torch.Tensor(self.schedule[::-1].copy())
        #self.schedule = torch.from_numpy(self.schedule).float()

        #store optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-6, amsgrad = True)
        else:
            self.optimizer = optimizer

    def ais_step(self, schedule, log_prob, input_data, backwards = False):
        r"""
        schedule: annealing schedule
        backwards: number of data points to draw if/when we are sampling simulated data in the backwards setting
        """
        logw_k = 0.0

        #compute maximum number of epochs

        #print(self.num_beta/data_len)
        max_epochs = math.ceil(self.num_beta/self.data_len)
        print('Maximum Number of Epochs: ', max_epochs)

        #iterate through number of samples
        b = 0
        num_epochs = 0

        while b < (self.num_beta-2):
            if num_epochs%self.nprint is 0 and b > 0:
                print('Epoch: ', num_epochs, '/', max_epochs, ' Current ll:', logw_k.cpu().numpy())

                #only print acceptance rate if we're using HMC
                if self.optimizer.__class__.__name__ is 'HMC':
                    print('Current Acceptance Rate: ', self.optimizer.acc_rate())

            for _, data in enumerate(input_data):
                #put breaking condition in so we don't crap out
                if b+1 == self.num_beta:
                    break

                t0 = schedule[b]
                t1 = schedule[b+1]

                def closure():
                    self.optimizer.zero_grad()

                    #perform log weight update
                    #use negative here so we can use optimizer to minimize it
                    loss = -log_prob(t1, data, backwards)
                    #perform transition update based on loss
                    loss.backward()

                    return loss

                with torch.no_grad():
                    update = log_prob(t1, data, backwards) - log_prob(t0, data, backwards)
                    logw_k += update

                self.optimizer.step(closure)

                b += 1
            num_epochs += 1

        return logw_k

    def run_forward(self, log_prob_fn):
        r"""
        x: our data
        forwards direction: anneal from prior to unnormalized posterior
        log prob is set up for bayesian inference case
        """
        model_state_list = [None] * self.num_samples

        for k in range(self.num_samples):
            print('Sample: ', k)

            #reload from initial state dict
            self.model.load_state_dict(self.model_init_params)


            #perform ais inner loop through schedule
            logw_k = self.ais_step(self.schedule, log_prob_fn, self.dataset)

            #save current model state in forwards step
            tmp = self.model.state_dict()
            model_state_list[k] = copy.deepcopy(tmp)

            if k is 0:
                logw = logw_k
            else:
                # print(logw.size(), logw_k.size())
                logw = torch.cat((logw.view(-1), logw_k.view(-1)),0)

        logw_total = -math.log(self.num_samples) + LogSumExp(logw)

        return logw_total, logw, model_state_list

    def run_backward(self, log_prob_fn):
        r"""
        x: our data
        reverse direction: anneal from posterior back to prior
        again, log prob is set up for bayesian inference case
        """

        #preinitialize
        model_state_list = [None] * self.num_samples

        for k in range(self.num_samples):
            print('Sample: ', k)

            #generate new sample for model parameters, initial pt is N(0, 0.5^2)
            self.model.load_state_dict(self.model_init_params)

            #perform ais inner loop through schedule
            logw_k = self.ais_step(self.rev_schedule, log_prob_fn, self.dataset, True)

            #save current model state in forwards step
            model_state_list[k] = self.model.state_dict()

            if k is 0:
                logw = logw_k
            else:
                logw = torch.cat((logw.view(-1), logw_k.view(-1)),0)

        logw_total = math.log(self.num_samples) - LogSumExp(logw)

        return logw_total, logw, model_state_list
