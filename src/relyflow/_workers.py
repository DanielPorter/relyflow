"""Entry points for worker processes. Kept free of heavy imports.

JAX reads its threading flags when first imported, and spawned workers import
the parent's main module (and possibly JAX) before running anything else, so
the flags must already be in the environment when workers start; see
``single_threaded_environment``.
"""

from __future__ import annotations

import contextlib
import os

SINGLE_THREADED = {
    "JAX_PLATFORMS": "cpu",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    "OMP_NUM_THREADS": "1",
}


@contextlib.contextmanager
def single_threaded_environment():
    """Set single-thread flags for processes started inside the block, then restore."""
    saved = {key: os.environ.get(key) for key in SINGLE_THREADED}
    os.environ.update(SINGLE_THREADED)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def nuts_reference(model_name: str, data: tuple, seed: int, settings: dict) -> dict:
    """Run NUTS from prior draws for one operator; returns draws and diagnostics."""
    from relyflow.mcmc import nuts
    from relyflow.models import get_model

    result = nuts(get_model(model_name), *data, seed=seed, **settings)
    return {
        "draws": result.flat(),
        "max_rhat": max(result.rhat.values()),
        "min_ess": min(result.ess.values()),
        "divergences": result.divergences,
        "seconds": result.seconds,
    }
