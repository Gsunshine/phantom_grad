# Modified based on the DEQ repo.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import torch.autograd as autograd
import numpy as np
import pickle
import sys
import os
from scipy.optimize import root
import time
from termcolor import colored
import copy
from modules.broyden import broyden, analyze_broyden
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)


def norm_diff(new, old, show_list=False):
    if show_list:
        return [(new[i] - old[i]).norm().item() for i in range(len(new))]
    return np.sqrt(sum((new[i] - old[i]).norm().item()**2 for i in range(len(new))))


class DEQFunc2d(Function):
    """ Generic DEQ module that uses Broyden's method to find the equilibrium state """

    @staticmethod
    def f(func, z1, u, *args):
        return func(z1, u, *args)

    @staticmethod
    def g(func, z1, u, cutoffs, *args):
        z1_list = DEQFunc2d.vec2list(z1, cutoffs)
        return DEQFunc2d.list2vec(DEQFunc2d.f(func, z1_list, u, *args)) - z1

    @staticmethod
    def list2vec(z1_list):
        bsz = z1_list[0].size(0)
        return torch.cat([elem.reshape(bsz, -1, 1) for elem in z1_list], dim=1)

    @staticmethod
    def vec2list(z1, cutoffs):
        bsz = z1.shape[0]
        z1_list = []
        start_idx, end_idx = 0, cutoffs[0][0] * cutoffs[0][1] * cutoffs[0][2]
        for i in range(len(cutoffs)):
            z1_list.append(z1[:, start_idx:end_idx].view(bsz, *cutoffs[i]))
            if i < len(cutoffs)-1:
                start_idx = end_idx
                end_idx += cutoffs[i + 1][0] * cutoffs[i + 1][1] * cutoffs[i + 1][2]
        return z1_list

    @staticmethod
    def broyden_find_root(func, z1, u, eps, *args):
        bsz = z1[0].size(0)
        z1_est = DEQFunc2d.list2vec(z1)
        cutoffs = [(elem.size(1), elem.size(2), elem.size(3)) for elem in z1]
        threshold, train_step, writer = args[-3:]

        g = lambda x: DEQFunc2d.g(func, x, u, cutoffs, *args)
        result_info = broyden(g, z1_est, threshold=threshold, eps=eps, name="forward")
        z1_est = result_info['result']
        nstep = result_info['nstep']
        lowest_step = result_info['lowest_step']
        diff = result_info['diff']
        r_diff = min(result_info['new_trace'][1:])

        if z1_est.get_device() == 0:
            if writer is not None:
                writer.add_scalar('forward/diff', result_info['diff'], train_step)
                writer.add_scalar('forward/nstep', result_info['nstep'], train_step)
                writer.add_scalar('forward/lowest_step', result_info['lowest_step'], train_step)
                writer.add_scalar('forward/final_trace', result_info['new_trace'][lowest_step], train_step)

        status = analyze_broyden(result_info, judge=True)
        if status:
            err = {"z1": z1}
            analyze_broyden(result_info, err=err, judge=False, name="forward", save_err=False)

        if threshold > 30:
            torch.cuda.empty_cache()
        
        # print(result_info['lowest_step'])
        # print(result_info['trace'])

        return DEQFunc2d.vec2list(z1_est.clone().detach(), cutoffs)

    @staticmethod
    def forward(ctx, func, z1, u, *args):
        nelem = sum([elem.nelement() for elem in z1])
        eps = 1e-5 * np.sqrt(nelem)
        ctx.args_len = len(args)
        with torch.no_grad():
            z1_est = DEQFunc2d.broyden_find_root(func, z1, u, eps, *args)  # args include pos_emb, threshold, train_step

            # If one would like to analyze the convergence process (e.g., failures, stability), should
            # insert here or in broyden_find_root.
            return tuple(z1_est)

    @staticmethod
    def backward(ctx, grad_z1):
        grad_args = [None for _ in range(ctx.args_len)]
        return (None, grad_z1, None, *grad_args)


def pg_f(func, z1, u, tau, cutoffs):
    z1_list = DEQFunc2d.vec2list(z1, cutoffs)
    f = DEQFunc2d.list2vec(func(z1_list, u))
    z1 = (1 - tau) * z1 + tau * f

    return z1


def pg_f_z1_list(func, z1_list, u, tau, cutoffs):
    f = DEQFunc2d.list2vec(func(z1_list, u))
    z1 = DEQFunc2d.list2vec(z1_list)
    z1 = (1 - tau) * z1 + tau * f
    z1_list = DEQFunc2d.vec2list(z1, cutoffs)

    return z1_list


class DEQModule2d(nn.Module):
    def __init__(self, func):
        super(DEQModule2d, self).__init__()
        self.func = func

    def forward(self, z1s, us, z0, **kwargs):
        raise NotImplemented