
import time
import contextlib
@contextlib.contextmanager
def timing(label):
    before = time.time()
    try:
        yield
    finally:
        after = time.time()
        print(f"Ran {label} in {after - before:0.3f}")
