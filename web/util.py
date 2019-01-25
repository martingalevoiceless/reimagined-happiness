from contextlib import contextmanager
import re
import threading
import numpy
import time

import time
import contextlib

timing_enabled = True
timing_stack = threading.local()

class Timer(object):
    def __init__(self, label, thresh=0.1):
        self.start_time = time.time()
        self.thresh = thresh
        self.label = label
        self.end_time = None
        self.stack = vars(timing_stack).setdefault("stack", [])
        self.stack.append(self)
        self.stack_copy = list(self.stack)

    def end(self):
        self.end_time = time.time()
        if self.stack and self.stack[-1] is self:
            self.stack.pop()
        else:
            print("warning: timer stack corrupted", self)

    def format(self):
        return " ".join(str(t) for t in self.stack_copy)

    def end_print(self):
        self.end()
        if self.thresh is None or self.end_time - self.start_time > self.thresh:
            print("timing", self.format())

    def __str__(self):
        if self.end_time is None:
            return f"{self.label} >"
        else:
            return f"{self.label}: {self.end_time-self.start_time:0.5f}"

@contextmanager
def timing(*a, **kw):
    if not timing_enabled:
        yield
        return

    timer = Timer(*a, **kw)
    try:
        yield
    finally:
        timer.end_print()

def nanguard(val, warning=None, default=0):
    if not numpy.isfinite(val).all():
        if not numpy.isscalar(val) or warning is None:
            raise ValueError(f"Not Finite: {val}")
        print("\033[31mNOT FINITE:", warning, val,"\033[m")
        return default
    return val

def softmax(x):
    e_x = numpy.exp(x - numpy.max(x))
    return nanguard(e_x / e_x.sum(axis=0))

def squash(x, amount):
    amount /= 2
    return nanguard(amount * (4/(1+numpy.exp(-min(max(x/amount,-300), 300))) - 2))

def sigmoid(x):
    return 1/(1+numpy.exp(-x))

def clamp(x, m, M):
    return nanguard(min(M, max(m, x)))

def as_pair(v1, v2, v3=None, extras=(None, None, None), strip=False):
    if type(v1) == dict: v1 = v1.get("hash")
    if type(v2) == dict: v2 = v2.get("hash")
    if type(v3) == dict: v3 = v3.get("hash")
    if strip:
        v1 = v1.partition(":")[0]
        v2 = v2.partition(":")[0]
        if v3 is not None:
            v3 = v3.partition(":")[0]

    pw = sorted([
        (v1, extras[0]),
        (v2, extras[1]),
    ] + ([(v3, extras[2])] if v3 is not None else []))
    pair, values = zip(*pw)
    if extras is not None:
        return pair, values
    else:
        return pair
