from contextlib import contextmanager
import re
import threading
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

