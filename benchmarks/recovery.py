"""Quick parameter recovery for trained networks (roadmap milestone 3).

Usage: ``python benchmarks/recovery.py checkpoints/reliance [n_operators]``

Simulates operators from the prior, draws posteriors with the network, and
prints per parameter: R² of posterior means against the truth, normalized RMSE,
posterior contraction (1 - posterior variance / prior variance) and the share
of 90% intervals containing the truth. Milestone 4 replaces this with the full
validation suite.

Static models: 200 unaided trials, then 400 with a 60-98%-reliable aid.
Learning model: 200 unaided, then 300 with an 85-98% aid, then 300 with a
55-75% aid, so that learning leaves a trace.
"""

import sys
import time
import warnings

import numpy as np

from relyflow import TrialData, make_design
from relyflow.inference import load


def designs(model_name: str, n: int, rng: np.random.Generator):
    if model_name == "learning":
        blocks = [
            (200, None),
            (300, rng.uniform(0.85, 0.98, n)),
            (300, rng.uniform(0.55, 0.75, n)),
        ]
    else:
        blocks = [(200, None), (400, rng.uniform(0.6, 0.98, n))]
    return make_design(blocks, rng, n_sims=n)


def main(path: str, n_operators: int = 500) -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    amortizer = load(path)
    model = amortizer.model
    rng = np.random.default_rng(99)
    truth = model.sample_prior(n_operators, rng)
    stimulus, advice = designs(model.name, n_operators, rng)
    choice, rt = model.simulate(truth, stimulus, advice, rng)

    start = time.perf_counter()
    draws = amortizer.sample_batch(stimulus, advice, choice, rt, num_samples=1000, seed=1)
    batch_seconds = time.perf_counter() - start
    prior = model.sample_prior(20_000, rng)

    print(f"{model.name}: {n_operators} operators x {stimulus.shape[1]} trials")
    print(f"{'parameter':<15}{'R2':>6}{'NRMSE':>7}{'contraction':>12}{'cover90':>8}")
    for name in model.param_names:
        samples, true = draws[name], truth[name]
        mean = samples.mean(axis=1)
        lower, upper = np.quantile(samples, [0.05, 0.95], axis=1)
        r2 = 1 - np.sum((mean - true) ** 2) / np.sum((true - true.mean()) ** 2)
        nrmse = np.sqrt(np.mean((mean - true) ** 2)) / np.ptp(true)
        contraction = 1 - samples.var(axis=1).mean() / prior[name].var()
        coverage = np.mean((true >= lower) & (true <= upper))
        print(f"{name:<15}{r2:6.2f}{nrmse:7.3f}{contraction:12.2f}{coverage:8.2f}")

    one = TrialData(stimulus[0], advice[0], choice[0], rt[0])
    amortizer.sample(one, num_samples=2000, seed=2)  # warm-up
    start = time.perf_counter()
    amortizer.sample(one, num_samples=2000, seed=3)
    single_seconds = time.perf_counter() - start
    print(f"posterior time: {single_seconds:.2f}s for one operator (2,000 draws), "
          f"{batch_seconds:.1f}s for all {n_operators} (1,000 draws each)")  # fmt: skip


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:3]))
