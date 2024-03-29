import contextlib

import matplotlib as mpl


@contextlib.contextmanager
def mpl_batch_mode():
    backend = mpl.get_backend()
    mpl.use("Agg")
    try:
        yield
    finally:
        mpl.use(backend)
