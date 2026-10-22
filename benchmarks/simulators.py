"""Simulator throughput (roadmap milestone 2).

Run with ``python benchmarks/simulators.py``. Prints simulated trials per second
for each model at a few batch sizes, using a design of 200 unaided trials
followed by 400 trials with an 85%-reliable aid.
"""

import platform
import time

import numpy as np

from relyflow import make_design
from relyflow.models import MODELS, get_model

N_TRIALS = 600
BATCHES = (100, 1_000, 10_000)
REPEATS = 3


def trials_per_second(name: str, n_sims: int, rng: np.random.Generator) -> float:
    model = get_model(name)
    stimulus, advice = make_design([(200, None), (400, 0.85)], rng, n_sims=n_sims)
    params = model.sample_prior(n_sims, rng)
    best = np.inf
    for _ in range(REPEATS):
        start = time.perf_counter()
        model.simulate(params, stimulus, advice, rng)
        best = min(best, time.perf_counter() - start)
    return n_sims * N_TRIALS / best


def main() -> None:
    rng = np.random.default_rng(0)
    print(f"{platform.processor() or platform.machine()}, numpy {np.__version__}")
    print(f"{'model':<10}" + "".join(f"{n:>12,} sims" for n in BATCHES))
    for name in sorted(MODELS):
        rates = [trials_per_second(name, n, rng) for n in BATCHES]
        print(f"{name:<10}" + "".join(f"{r / 1e6:>12.1f}M/s " for r in rates))


if __name__ == "__main__":
    main()
