"""How much could a learning-model posterior know about alpha and r0? (milestone 3)

Usage: ``python benchmarks/learning_reference.py checkpoints/learning [n_operators]``

The learning model has an exact likelihood, so a near-exact posterior for
(alpha, r0) can be computed on a grid. For each operator, the network's draws of
the other 12 parameters are combined with the exact conditional posterior of
(alpha, r0) given each draw, then averaged. The network recovers those 12
parameters well, so this approximates the true marginal posterior. Comparing it
with the network's own posterior shows how much information the network misses.
Slow: about 5 s per operator. Uses the learning design from ``recovery.py``.
"""

import sys
import warnings

import numpy as np
from recovery import designs

from relyflow.inference import load

N_DRAWS = 40
LOG_ALPHA_GRID = np.linspace(np.log(0.002), np.log(0.8), 30)
R0_GRID = np.linspace(0.02, 0.98, 30)


def r2(estimate: np.ndarray, truth: np.ndarray) -> float:
    return 1 - np.sum((estimate - truth) ** 2) / np.sum((truth - truth.mean()) ** 2)


def main(path: str, n_operators: int = 120) -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    amortizer = load(path)
    model = amortizer.model
    priors = {p.name: p.prior for p in model.parameters}
    rng = np.random.default_rng(99)
    truth = model.sample_prior(n_operators, rng)
    stimulus, advice = designs(model.name, n_operators, rng)
    choice, rt = model.simulate(truth, stimulus, advice, rng)
    draws = amortizer.sample_batch(stimulus, advice, choice, rt, num_samples=N_DRAWS, seed=1)

    log_alpha, r0 = (g.ravel() for g in np.meshgrid(LOG_ALPHA_GRID, R0_GRID, indexing="ij"))
    alpha = np.exp(log_alpha)
    # Prior density on the (log alpha, r0) grid, including the log-transform Jacobian.
    log_prior = priors["alpha"].logpdf(alpha) + log_alpha + priors["r0"].logpdf(r0)

    moments = []
    for i in range(n_operators):
        posterior = np.zeros(alpha.size)
        for d in range(N_DRAWS):
            theta = {name: np.full(alpha.size, draws[name][i, d]) for name in model.param_names}
            theta["alpha"], theta["r0"] = alpha, r0
            log_post = (
                model.log_likelihood(theta, stimulus[i], advice[i], choice[i], rt[i]) + log_prior
            )
            weights = np.exp(log_post - log_post.max())
            posterior += weights / weights.sum()
        posterior /= N_DRAWS
        mean_la, mean_r0 = posterior @ log_alpha, posterior @ r0
        moments.append(
            (
                mean_la,
                posterior @ (log_alpha - mean_la) ** 2,
                mean_r0,
                posterior @ (r0 - mean_r0) ** 2,
            )
        )
    moments = np.array(moments)

    prior_draws = model.sample_prior(200_000, rng)
    contraction_la = 1 - moments[:, 1].mean() / np.log(prior_draws["alpha"]).var()
    contraction_r0 = 1 - moments[:, 3].mean() / prior_draws["r0"].var()
    print(f"{model.name}: {n_operators} operators x {stimulus.shape[1]} trials")
    print(
        f"near-exact posterior: log alpha R2 {r2(moments[:, 0], np.log(truth['alpha'])):.2f} "
        f"(contraction {contraction_la:.2f}); r0 R2 {r2(moments[:, 2], truth['r0']):.2f} "
        f"(contraction {contraction_r0:.2f})"
    )
    print(
        f"network posterior:    log alpha R2 "
        f"{r2(np.log(draws['alpha']).mean(axis=1), np.log(truth['alpha'])):.2f}; "
        f"r0 R2 {r2(draws['r0'].mean(axis=1), truth['r0']):.2f}"
    )


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:3]))
