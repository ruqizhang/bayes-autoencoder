import sys
import os
import math
import torch
import numpy as np

import torch
from torch.autograd import Variable


def hmc_trajectory(current_z, current_v, U, grad_U, epsilon, L=10):
    """This version of HMC follows https://arxiv.org/pdf/1206.1901.pdf.

    Args:
        U: function to compute potential energy/minus log-density
        grad_U: function to compute gradients w.r.t. U
        epsilon: (adaptive) step size
        L: number of leap-frog steps
        current_z: current position
    """

    # as of `torch-0.3.0.post4`, there still is no proper scalar support
    assert isinstance(epsilon, Variable)

    eps = epsilon.view(-1, 1)
    z = current_z
    v = current_v - grad_U(z).mul(eps).mul_(.5)

    for i in range(1, L+1):
        z = z + v.mul(eps)
        if i < L:
            v = v - grad_U(z).mul(eps)

    v = v - grad_U(z).mul(eps).mul_(.5)
    v = -v  # this is not needed; only here to conform to the math

    return z.detach(), v.detach()


def accept_reject(current_z, current_v, 
                  z, v, 
                  epsilon, 
                  accept_hist, hist_len, 
                  U, K=lambda v: torch.sum(v * v, 1)):
    """Accept/reject based on Hamiltonians for current and propose.

    Args:
        current_z: position BEFORE leap-frog steps
        current_v: speed BEFORE leap-frog steps
        z: position AFTER leap-frog steps
        v: speed AFTER leap-frog steps
        epsilon: step size of leap-frog.
                (This is only needed for adaptive update)
        U: function to compute potential energy (MINUS log-density)
        K: function to compute kinetic energy (default: kinetic energy in physics w/ mass=1)
    """

    mdtype = current_z.dtype
    mdevice = current_z.device

    current_Hamil = K(current_v) + U(current_z)
    propose_Hamil = K(v) + U(z)

    prob = torch.exp(current_Hamil - propose_Hamil)
    uniform_sample = torch.rand(prob.size(), dtype=mdtype, device=mdevice)
    #uniform_sample = Variable(uniform_sample.type(mdtype))
    accept = (prob > uniform_sample).float()
    z = z.mul(accept.view(-1, 1)) + current_z.mul(1. - accept.view(-1, 1))

    accept_hist = accept_hist.add(accept)
    criteria = (accept_hist / hist_len > 0.65).float()
    adapt = 1.02 * criteria + 0.98 * (1. - criteria)
    epsilon = epsilon.mul(adapt).clamp(1e-4, .5)

    # clear previous history & save memory, similar to detach
    #z.detach_()
    #z.requries_grad = True
    new_z = z.data.clone()
    new_z.requires_grad = True
    #z = Variable(z.data, requires_grad=True)
    #epsilon = Variable(epsilon.data)
    #accept_hist = Variable(accept_hist.data)
    epsilon.requires_grad = False
    accept_hist.requires_grad = False

    return new_z, epsilon, accept_hist
