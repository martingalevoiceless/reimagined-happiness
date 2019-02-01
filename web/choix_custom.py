import math
import itertools
import sys
import numpy
import random
import warnings
from torch import autograd

from scipy import linalg
from web.util import nanguard
import contextlib
@contextlib.contextmanager
def timing(name):
    yield

import torch


SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)

def setup(cuda=True):
    global device
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        with timing("start cuda"):
            torch.zeros(1, device=device)
    else:
        device = torch.device('cpu')

ng_enabled=False
def nanguardt(val, warning=None, force=False):
    if ng_enabled or force:
        if not torch.isfinite(val).all():
            raise ValueError(f"Not Finite: {val} ({warning})")
    return val
def posguardt(val, warning=None, force=False):
    if ng_enabled or force:
        if not (val > 0).all():
            raise ValueError(f"Not positive: {val} ({warning})")
    nanguardt(val, warning, force)
    return val

class NormOfDifferenceTest:
    def __init__(self, tol=1e-8, order=1):
        self._tol = tol
        self._ord = order
        self._prev_params = None
        self._last_delta = None
        self._last_delta2 = None
        self.dist = None
        self.delta = None

    def __call__(self, params, update=True):
        self._prev_params2 = self._prev_params
        if self._prev_params is None:
            if update:
                self._prev_params = nanguardt(params)
            return False
        delta = (self._prev_params - params).cpu().numpy()
        mags = numpy.abs(delta)
        dist = nanguard(numpy.sum(mags))
        self._last_delta2 = self._last_delta
        self._last_delta = self.delta
        self.delta = delta
        self.dist = dist
        delta_2step = nanguard(numpy.sum(numpy.abs(self._last_delta2 - delta))) if self._last_delta2 is not None else 0
        delta_1step = nanguard(numpy.sum(numpy.abs(self._last_delta - delta))) if self._last_delta is not None else 0
        largest = numpy.argsort(delta)
        if update:
            self._prev_params = params
        spaces = " " * 6
        ups = ', '.join(f'{idx}={delta[idx]:0.3f}' for idx in largest[:-5:-1])
        mag_std = numpy.std(mags)
        sys.stdout.write(f"converged dist: {dist:0.4f}. /param: {dist/len(params):0.3e}. two step dist: {delta_2step:0.3f}, 1step: {delta_1step:0.3f}, mag std: {mag_std}. top updates: {ups}{spaces}\n")
        sys.stdout.flush()
        return dist <= self._tol * len(params) or (delta_2step < delta_1step / 2 and delta_2step > 0)

def log_transform(weights):
    weights = torch.max(weights, torch.full((1,), 0.0001, device=device))
    posguardt(weights, weights[-20:])
    params = nanguardt(torch.log(weights), weights[-20:])
    return params - torch.mean(params)


def exp_transform(params):
    weights = torch.exp(params - torch.mean(params))
    return (len(weights) / torch.sum(weights)) * weights

def statdist(v):
    v = v.pop()
    with timing("statdist"):
        n = v.shape[0]
        nanguardt(v, "t_generator")
        with timing("statdist::lu_factor_torch"):
            _, v = torch.gesv(torch.ones([n, 1], dtype=torch.float32).to(device), v)
            del _
        nanguardt(v, "lu")
        # The last row contains 0's only.
        with timing("statdist::slices"):
            left = v[:-1,:-1]
            right = -v[:-1,-1]
            del v
        # Solves system `left * x = right`. Assumes that `left` is
        # upper-triangular (ignores lower triangle).
        #print("left shape:", left.shape, "right shape:", right.shape)
        #with timing("statdist::pytorch readback 1"):
        with timing("pytorch version"):
            res, _ = torch.trtrs(right.reshape(right.shape+(-1,)), left)
            del _
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
        counts = nanguardt(torch.tensor([x[2] for x in data], dtype=torch.float32, device=device), "counts")
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
                    t_chain.view(-1).scatter_add_(0, pairs, nanguardt(counts * (1 / (weights[winners] + weights[losers])), "scatter"))
                    t_chain.view(-1).scatter_add_(0, diag_indices, nanguardt(-torch.sum(t_chain, dim=0), "sum scatter"))
                with timing("core"):
                    #print(alpha)
                    #print("t_chain[-1, :] = ", nonzeros(t_chain[-1], alpha))
                    #print("t_chain[:, -1] = ", nonzeros(t_chain[:, -1], alpha))
                    #if params is not None:
                    #    print("params[-1] = ", params[-1])
                    #print("weights[-1] = ", weights[-1])
                    t_chain = [t_chain]
                    params = nanguardt(log_transform(statdist(t_chain)))
                with timing("converged"):
                    if converged(params):
                        break
    with timing("return"):
        return params.cpu().numpy().astype(float), iteration, converged.dist, converged.dist / len(params)




def normpdf_log(mean, std, x):
    var = std ** 2
    scaling = numpy.log(numpy.sqrt(2*numpy.pi*var))
    main = -(x-mean)**2/(2*var) - scaling
    return main
    
def err(vecs, targ, edge_ratios, edges, edge_idxs, edge_mags):
    def cg(label, val):
        #assert val.requires_grad, label
        #grads_to_check.append((val, label))
        #def extract(grad):
        #    try:
        #        nanguardt(grad, label)
        #    except ValueError:
        #        torch.set_printoptions(profile="full")
        #        for i, v in enumerate(val):
        #            if not torch.isfinite(grad[i]).all():
        #                print(i, v)
        #        for i, v in enumerate(grad):
        #            if not torch.isfinite(v).all():
        #                print(i, v)
        #        torch.set_printoptions(profile="default")
        #        raise
        #val.register_hook(extract)
        return val
    dists = cg("dists", torch.sqrt(cg("edges_sum", 1e-15+torch.sum(cg("edges_exp", cg("edges", (vecs[[x[0] for x in edges]] - vecs[[x[1] for x in edges]]))**2), 1))))

    q = cg("gather", torch.gather(dists, 0, edge_idxs).reshape(-1, 2))
    ld = cg("ld", cg("ld_left", (1e-10+q[:, 0])) / cg("ld_right", 1e-10 + torch.sum(q, 1)))
    nanguardt(ld)
    error = cg("error", (ld - targ))
    loss = cg("loss", torch.mean(error**2 * edge_mags) * 100)
    l2_loss = cg("l2_loss", torch.sum(cg("normpdf_out", -normpdf_log(1, 0.5, cg("l2", torch.sqrt(cg("l2_sqrt", torch.sum(cg("vecs", vecs) ** 2, 1)))))))/10)
    l1_loss = cg("l1_loss", torch.sum(cg("abs(vecs)", torch.abs(vecs)))/800)
    nanguardt(l1_loss)
    nanguardt(l2_loss)
    nanguardt(loss)
    nanguardt(error)
    nanguardt(q)
    
    return loss + l2_loss + l1_loss, loss, l2_loss, l1_loss, torch.mean(error)

def pca(x, k=2):
    # preprocess the data
    x_mean = torch.mean(x,0)
    x = x - x_mean.expand_as(x)

    # svd
    u,_,_ = torch.svd(torch.t(x))
    return torch.mm(x,u[:,:k]), u

def init(count, vec_means=None, vec_stds=None):
    vecdim = 64
    if vec_means is not None:
        shape = (count-vec_means.shape[0], vecdim)
    else:
        shape = (count, vecdim)
    initstd = 0.5/numpy.sqrt(vecdim)
    vec_means_ = torch.normal(0, torch.full(shape, 1/numpy.sqrt(vecdim)))
    vec_stds_ = torch.full(shape, initstd)
    if vec_means is not None:
        vec_means = torch.cat((vec_means.type(torch.float32).to(device), vec_means_.to(device)))
        vec_stds = torch.cat((vec_stds.type(torch.float32).to(device), vec_stds_.to(device)))
        try:
            nanguardt(vec_means)
            nanguardt(vec_stds)
        except ValueError as e:
            print(e)
            vec_means = torch.normal(0, torch.full((count,vecdim), 1/numpy.sqrt(vecdim)))
            vec_stds = torch.full((count,vecdim), initstd)
    else:
        vec_means = vec_means_
        vec_stds = vec_stds_
    vec_means = torch.tensor(vec_means.detach(), requires_grad=True, device=device)
    vec_stds = torch.tensor(vec_stds.detach(), requires_grad=True, device=device)
    opt = torch.optim.Adam([vec_means, vec_stds], 0.01)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=400,factor=0.1,verbose=True)
    return vec_means, vec_stds, (count,vecdim), opt, sch, initstd

def embedding(hashes, edges, edge_ratios, targ, vec_means=None, vec_stds=None, max_iters=4000):
    vec_means, vec_stds, shape, opt, sch, initstd = init(len(hashes), vec_means, vec_stds)
    targ = torch.tensor(targ).to(device)
    edge_mags = torch.tensor([x[2] for x in edge_ratios], device=device)
    edge_mags /= torch.mean(edge_mags)
    
    edge_pos = {tuple(sorted(e)): i for i, e in enumerate(edges)}
    edge_idxs = torch.tensor(list(itertools.chain.from_iterable((edge_pos[tuple(sorted(e))] for e in ep[0]) for ep in edge_ratios)), device=device)
    with timing("train"):
        for x in range(max_iters):
            try:
                first = True
                #with autograd.detect_anomaly():
                def closure():
                    nonlocal first
                    opt.zero_grad()
                    nanguardt(vec_means)
                    nanguardt(vec_stds)
                    vecs = torch.normal(0, torch.full(shape, 1.0)).to(device)*torch.abs(vec_stds) + vec_means
                    q1 = torch.abs(vec_stds) / (initstd)
                    q2 = -torch.log(q1)
                    q3 = torch.max(q2, torch.full(shape, -0.5).to(device))
                    std_loss = torch.sum(q3) / 30000
                    #print("q1", q1.detach().cpu().numpy())
                    #print("q2", q2.detach().cpu().numpy())
                    #print("q3", q3.detach().cpu().numpy())
                    #print("std_loss", std_loss.detach().cpu().numpy())
                    loss, main_loss, l2_loss, l1_loss, dists = err(nanguardt(vecs), targ, edge_ratios, edges, edge_idxs, edge_mags)
                    loss = loss + std_loss
                    nanguardt(loss)
                    loss.backward()
                    if x % 100 == 0 and first:
                        first=False
                        print(torch.max(torch.sqrt(torch.sum(vec_stds**2, 1))).detach().cpu().numpy())
                        print("loss", loss.detach().cpu().numpy(), main_loss.detach().cpu().numpy(), l2_loss.detach().cpu().numpy(), l1_loss.detach().cpu().numpy(), std_loss.detach().cpu().numpy(), dists.detach().cpu().numpy())
                    del std_loss
                    del dists
                    del l1_loss
                    del l2_loss
                    del vecs
                    del q1
                    del q2
                    del q3

                    nanguardt(vec_means.grad)
                    nanguardt(vec_stds.grad)
                    return loss, main_loss
                
                _,l = closure()
                opt.step()
                lr=opt.state_dict()["param_groups"][0]["lr"]
                if lr < 0.0003: break
                sch.step(l)
            except ValueError as e:
                print("failure on iter:", x)
                raise
    return nanguardt(vec_means,force=True).type(torch.float16), nanguardt(vec_stds,force=True).type(torch.float16)
