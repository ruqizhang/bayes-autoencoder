"""
Author: Wesley Maddox
Date: 2/1/18
Copied off of sgd based code
source of sgd: https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py

hmc as an optimizer, code built off of sgd source code
only update should be the noise
"""

import torch
from torch.optim.optimizer import Optimizer, required
from torch.distributions import Normal
import copy

class HMC(Optimizer):

    def __init__(self, params, lr=required, L = 1, propsd = 1, momentum=0, dampening=0,
                 weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, L = L, propsd = propsd)
        #if nesterov and (momentum <= 0 or dampening != 0):
        #    raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(HMC, self).__init__(params, defaults)
        self.total = 0.0
        self.reject = 0.0

    def __setstate__(self, state):
        super(HMC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def acc_rate(self):
        #print(self.total, self.reject)
        #print('Current Acceptance Rate: ', (self.total-self.reject)/self.total)
        return (self.total-self.reject)/self.total

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable, required): A closure that reevaluates the model
                and returns the loss. Here, loss is the Negative log-likelihood.
        """
        self.total += 1.
        loss = closure()
        loss_orig = loss.clone()

        for group in self.param_groups:
            #print('at start: ', group)
            #old_state = copy.deepcopy(group)

            momentum_change = 0.0
            old_state = []
            for q in group['params']:
                old_state.append(q.clone())
                epsilon = group['lr']

                #resample momentum
                p_dist = Normal(torch.zeros_like(q), group['propsd'] * torch.ones_like(q))
                p_old = p_dist.sample()
                p = copy.deepcopy(p_old)

                #half step for momentum
                if q.grad is None:
                    # print(q)
                    continue
                p.data.add_(-epsilon/2, q.grad.data)

                for l in range(group['L']-1):
                    q.data.add_(epsilon, p.data)

                    loss = closure()

                    p.data.add_(epsilon, q.grad.data)

                #final half step for momentum
                q.data.add_(epsilon, p.data)
                loss = closure()
                p.data.add_(epsilon/2, q.grad.data)

                #flip
                p.data.mul_(-1)

                momentum_change += -p_dist.log_prob(p_old).sum() + p_dist.log_prob(p).sum()

            loss_change = loss_orig - loss

            u = torch.zeros_like(loss_change).uniform_()
            total_change = torch.exp(loss_change + momentum_change)
            #print(u.data.numpy(), total_change.data.numpy())
            if u > total_change:
                self.reject += 1.
                #print('before rejection: ', group['params'])
                #print('rejected')
                for idx, q in enumerate(group['params']):
                    q.data = old_state[idx].data
                #group = old_state
                #print('after rejection: ', group['params'])


        return loss





#        loss = closure()
#        loss_orig = loss.clone()
#
#        old_state = copy.deepcopy(self.param_groups)
#
#        """i = 0
#        for group in self.param_groups:
#            for q in group['params']:
#                if i<1:
#                    print(q.data)
#                i+=1"""
#
#        for group in self.param_groups:
#            if closure is None and group['L']>1:
#                raise ValueError("L > 1 requires a closure")
#
#            weight_decay = group['weight_decay']
#            propsd = group['propsd']
#
#            momentum = 0.0
#            for q in group['params']:
#                #resample momentum for each parameter group in turn
#                p_dist = Normal(torch.zeros_like(q), propsd * torch.ones_like(q))
#                p_old = p_dist.sample()
#                p = p_old.clone()
#
#                #half step for momentum
#                p.data.add_(-group['lr']/2, q.grad.data)
#
#                #now for loop
#                for l in range(group['L']):
#                    q.data.add_(group['lr'], p.data)
#
#                    #now re-evaluate the loss function to re-set the gradients
#                    loss = closure()
#
#                    if q.grad is None:
#                            continue
#                    #update momentum
#                    if l < group['L']-1:
#                        #print(q.grad.data)
#                        d_q = q.grad.data
#                        p.data.add_(-group['lr'], d_q)
#
#                    """if weight_decay != 0:
#                        d_q.add_(weight_decay, q.data)
#                    if momentum != 0:
#                        param_state = self.state[q]
#                        if 'momentum_buffer' not in param_state:
#                            buf = param_state['momentum_buffer'] = torch.zeros_like(q.data)
#                            buf.mul_(momentum).add_(d_q)
#                        else:
#                            buf = param_state['momentum_buffer']
#                            buf.mul_(momentum).add_(1 - dampening, d_q)
#                        if nesterov:
#                            d_q = d_q.add(momentum, buf)
#                        else:
#                            d_q = buf"""
#
#                #half step for end of momentum
#                p.data.add_(-group['lr']/2, q.grad.data)
#                p = -p
#
#                #add proposal energies
#                #print(p.data.sum(), p_old.data.sum())
#                k_ll_diff = -p_dist.log_prob(p_old) + p_dist.log_prob(p)
#                momentum += k_ll_diff.sum()
#
#            loss_difference = closure() - loss_orig
#
#            print(loss_difference.data.sum(), momentum.data.sum())
#        #now metropolis accept reject
#        u = torch.rand(1)
#        print(u)
#        if torch.rand(1) > torch.exp(loss_difference + momentum).data:
#            print('rejected')
#            #reset to old state
#            self.param_groups = old_state
#
#        """i = 0
#        for group in self.param_groups:
#            for q in group['params']:
#                if i<1:
#                    print(q.data)
#                i+=1"""
#        return loss

