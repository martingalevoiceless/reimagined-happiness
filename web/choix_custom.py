import math
import sys
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
    with timing("start cuda"):
        torch.zeros(1, device=device)
else:
    device = torch.device('cpu')

def nanguardt(val, warning=None):
    #if not torch.isfinite(val).all():
    #    raise ValueError(f"Not Finite: {val} ({warning})")
    return val
def posguardt(val, warning=None):
    #if not (val > 0).all():
    #    raise ValueError(f"Not positive: {val} ({warning})")
    #nanguardt(val, warning)
    return val

class NormOfDifferenceTest:
    def __init__(self, tol=1e-8, order=1):
        self._tol = tol
        self._ord = order
        self._prev_params = None

    def __call__(self, params, update=True):
        if self._prev_params is None:
            if update:
                self._prev_params = nanguardt(params)
            return False
        delta = self._prev_params - params
        dist = nanguardt(torch.sum(torch.abs(delta)).cpu()).numpy()
        self.dist = dist
        if update:
            self._prev_params = params
        sys.stdout.write(f"converged dist: {dist} /param: {dist/len(params)}                                 \r")
        sys.stdout.flush()
        return dist <= self._tol * len(params)

def log_transform(weights):
    weights = torch.max(weights, torch.full((1,), 0.0001, device=device))
    posguardt(weights, weights[-20:])
    params = nanguardt(torch.log(weights), weights[-20:])
    return params - torch.mean(params)


def exp_transform(params):
    weights = torch.exp(params - torch.mean(params))
    return (len(weights) / torch.sum(weights)) * weights

def statdist(t_generator):
    with timing("statdist"):
        n = t_generator.shape[0]
        nanguardt(t_generator, "t_generator")
        with timing("statdist::lu_factor_torch"):
            _, lu = torch.gesv(torch.ones([n, 1], dtype=torch.float32).to(device), t_generator)
        nanguardt(lu, "lu")
        # The last row contains 0's only.
        with timing("statdist::slices"):
            left = lu[:-1,:-1]
            right = -lu[:-1,-1]
        # Solves system `left * x = right`. Assumes that `left` is
        # upper-triangular (ignores lower triangle).
        #print("left shape:", left.shape, "right shape:", right.shape)
        #with timing("statdist::pytorch readback 1"):
        with timing("pytorch version"):
            res, _ = torch.trtrs(right.reshape(right.shape+(-1,)), left)
            nanguardt(res, "res")
            res = res.view(-1)
            res = torch.cat((res, torch.ones(1, device=device)))
            return nanguardt((n / torch.sum(res)), "n/sum") * res



def eq(a, b):
    assert a.shape == b.shape
    print(a)
    print(b)
    q = numpy.abs(a.reshape([-1]) - b.reshape([-1]))
    print(q)
    max_err = numpy.max(q)
    mean_error = numpy.mean(q)
    median_error = numpy.median(q)
    percentiles = numpy.percentile(q, [60, 70, 80, 90, 95, 99, 99.5, 99.9, 99.99])
    print(max_err, mean_error, median_error, percentiles)
    assert percentiles[-1] < 4e-5

def nonzeros(x, thresh):
    x = x.cpu().numpy()
    assert len(x.shape) == 1
    mask = numpy.abs(x) > thresh
    return x[mask], numpy.arange(x.shape[0])[mask]

def ilsr_pairwise(n_items, data, alpha=0.0, params=None, max_iter=100, tol=1e-5):
    converged = NormOfDifferenceTest(tol, order=1)
    with timing("check"):
        assert len(set([(a, b) for a, b, c in data])) == len(data)
        assert all(c for a, b, c in data)
    with timing("setup"):
        winners        = torch.tensor([x[0] for x in data], device=device)
        losers         = torch.tensor([x[1] for x in data], device=device)
        pairs          = winners * n_items + losers
    with timing("setup2"):
        counts = nanguardt(torch.tensor([x[2] for x in data], dtype=torch.float32, device=device) + alpha, "counts")
        #counts /= torch.min(counts)
        diag_indices   = torch.arange(n_items, device=device) + torch.arange(n_items, device=device) * n_items
        if params is not None:
            params = posguardt(torch.from_numpy(params.astype("float32")).to(device), "params")
            converged(params)
    with timing("loop"):
        for iteration in range(max_iter):
            with timing("loop body"):
                with timing("loop setup"):
                    if params is None:
                        weights = torch.ones(n_items, dtype=torch.float32, device=device)
                    else:
                        weights = posguardt(exp_transform(params))
                with timing("chain build"):
                    t_chain = torch.full((n_items, n_items), alpha, dtype=torch.float32, device=device)
                    t_chain.view(-1).scatter_(0, pairs, nanguardt(counts * (1 / (weights[winners] + weights[losers])), "scatter"))
                    t_chain.view(-1).scatter_add_(0, diag_indices, nanguardt(-torch.sum(t_chain, dim=0), "sum scatter"))
                with timing("core"):
                    #print(alpha)
                    #print("t_chain[-1, :] = ", nonzeros(t_chain[-1], alpha))
                    #print("t_chain[:, -1] = ", nonzeros(t_chain[:, -1], alpha))
                    #if params is not None:
                    #    print("params[-1] = ", params[-1])
                    #print("weights[-1] = ", weights[-1])
                    params = nanguardt(log_transform(statdist(t_chain)))
                with timing("converged"):
                    if converged(params):
                        break
    with timing("return"):
        return params.cpu().numpy().astype(float), iteration, converged.dist, converged.dist / len(params)
