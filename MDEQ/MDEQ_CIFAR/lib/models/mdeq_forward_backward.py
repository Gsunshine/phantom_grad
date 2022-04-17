# Modified based on the DEQ repo.

import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np

import sys
sys.path.append("../../")
from modules.deq2d import *


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
        
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z1_list]

        z1_list = list(DEQFunc2d.apply(self.func, z1_list, u, threshold, train_step, writer))
        
        if self.training:
            for _ in range(self.pg_steps):
                z1_list = pg_f_z1_list(self.func, z1_list, u, self.tau, cutoffs)

        return z1_list


