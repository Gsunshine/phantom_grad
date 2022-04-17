# Modified based on the DEQ repo.

import torch
from torch import nn
import torch.nn.functional as functional
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

from modules.deq2d import *



def power_method(f, h, iters=10):
    v = torch.randn_like(h).to(h)
    for _ in range(iters):
        vTJ = torch.autograd.grad(f, h, grad_outputs=v, retain_graph=True)[0]
        v = F.normalize(vTJ.flatten(start_dim=1), dim=1).view_as(h)
    
    # v is unit vector
    r = torch.einsum('bnd,bnd->b', vTJ, v).abs()

    return r


class MDEQWrapper(DEQModule2d):
    def __init__(self, func, tau, pg_steps):
        super(MDEQWrapper, self).__init__(func)

        self.tau = tau
        self.pg_steps = pg_steps

    def forward(self, z1_list, u, **kwargs):
        train_step = kwargs.get('train_step', -1)
        threshold = kwargs.get('threshold', 30)
        writer = kwargs.get('writer', None)

        if u is None:
            raise ValueError("Input injection is required.")

        z1_list = list(DEQFunc2d.apply(self.func, z1_list, u, threshold, train_step, writer))

        if self.training:
            cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z1_list]
            for _ in range(self.pg_steps):
                z1_list = pg_f_z1_list(self.func, z1_list, u, self.tau, cutoffs)
           
        return z1_list

