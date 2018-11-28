import math
import numpy
import random
import warnings

from scipy import linalg
from web.util import timing

import torch


SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class NormOfDifferenceTest:
    def __init__(self, tol=1e-8, order=1):
        self._tol = tol
        self._ord = order
        self._prev_params = None

    def __call__(self, params, update=True):
        params = numpy.asarray(params) - numpy.mean(params)
        if self._prev_params is None:
            if update:
                self._prev_params = params
            return False
        dist = numpy.linalg.norm(self._prev_params - params, ord=self._ord)
        self.dist = dist
        if update:
            self._prev_params = params
        return dist <= self._tol * len(params)

def log_transform(weights):
    params = numpy.log(weights)
    return params - params.mean()


def exp_transform(params):
    weights = numpy.exp(numpy.asarray(params) - numpy.mean(params))
    return (len(weights) / weights.sum()) * weights

def statdist(generator):
    with timing("statdist"):
        #with timing("statdist::asarray"):
        generator = numpy.asarray(generator)
        n = generator.shape[0]
        torch_1 = False
        torch_2 = True
        if not torch_1 or not torch.cuda.is_available():
            with timing("statdist::lu_factor_numpy"):
                lu, piv = linalg.lu_factor(generator.T, check_finite=False)
        else:
            with timing("statdist::to_pytorch"):
                t_generator = torch.t(torch.from_numpy(generator).to(device))
            with timing("statdist::lu_factor_torch"):
                lu, piv = torch.btrifact(t_generator.reshape((-1,) + t_generator.shape),pivot=False)
                lu = lu.reshape(t_generator.shape).cpu().numpy()
        # The last row contains 0's only.
        left = lu[:-1,:-1]
        right = -lu[:-1,-1]
        # Solves system `left * x = right`. Assumes that `left` is
        # upper-triangular (ignores lower triangle).
        #print("left shape:", left.shape, "right shape:", right.shape)
        #with timing("statdist::pytorch readback 1"):
        if not torch_2 or not torch.cuda.is_available():
            lc = left#.cpu().numpy()
            rc = right#.cpu().numpy()
            #with timing("statdist::solve_triangular"):
            res = linalg.solve_triangular(lc, rc, check_finite=False)
        else:
            with timing("pytorch version"):
                t_res, _ = torch.trtrs(torch.from_numpy(right.reshape(right.shape+(-1,))).to(device), torch.from_numpy(left).to(device))
            with timing("pytorch readback"):
                t_res = t_res.reshape([-1]).cpu().numpy()
                res = t_res
        #print("res", numpy.min(res), numpy.max(res), numpy.mean(res), numpy.std(res))
        #print("t_res", numpy.min(t_res), numpy.max(t_res), numpy.mean(t_res), numpy.std(t_res))
        #diff = t_res - res
        #print("diff", numpy.min(diff), numpy.max(diff), numpy.mean(diff), numpy.std(diff))
        res = numpy.append(res, 1.0)
        return (n / res.sum()) * res



def _init_lsr(n_items, alpha, initial_params):
    if initial_params is None:
        weights = numpy.ones(n_items)
    else:
        weights = exp_transform(initial_params)
    chain = alpha * numpy.ones((n_items, n_items), dtype=float)
    return weights, chain


def _ilsr(fun, params, max_iter, tol):
    converged = NormOfDifferenceTest(tol, order=1)
    for iteration in range(max_iter):
        params = fun(initial_params=params)
        if converged(params):
            break
    return params, iteration, converged.dist, converged.dist / len(params)


def lsr_pairwise(n_items, data, alpha=0.0, initial_params=None):
    weights, chain = _init_lsr(n_items, alpha, initial_params)
    for winner, loser in data:
        chain[loser, winner] += 1 / (weights[winner] + weights[loser])
    chain -= numpy.diag(chain.sum(axis=1))
    return log_transform(statdist(chain))


def ilsr_pairwise(n_items, data, alpha=0.0, initial_params=None, max_iter=100, tol=1e-4):
    import functools
    fun = functools.partial(lsr_pairwise, n_items=n_items, data=data, alpha=alpha)
    return _ilsr(fun, initial_params, max_iter, tol)
