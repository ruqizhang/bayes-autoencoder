#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:26:27 2017

@author: wesleymaddox

numerically stable logSumExp function
numerically stable calculation of log variance, only in 1d though
"""
import torch
import math

def LogSumExp(x,dim=0):
    m,_ = torch.max(x,dim=dim,keepdim=True)
    return m + torch.log((x - m).exp().sum(dim=dim,keepdim=True))

def LogVar(x):
    N = len(x)
    v1 = LogSumExp(2 * x) - math.log(N)
    v2 = 2 * (LogSumExp(x) - math.log(N))
    return LogSumExp(torch.cat((v1, v2)))