"""Parameter recovery, interval coverage and simulation-based calibration."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from relyflow.inference.designs import DesignPrior
from relyflow.models.base import AutomationModel


def simulate_operators(
    model: AutomationModel,
    n: int,
    design: DesignPrior,
    rng: np.random.Generator,
    batch_size: int = 50,
) -> Iterator[tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield batches ``(truth, stimulus, advice, choice, rt)`` of simulated operators.

    Designs come from ``design``, the distribution the network was trained on, so
    that calibration checks are valid. Operators in a batch share a block layout.
    """
    done = 0
    while done < n:
        size = min(batch_size, n - done)
        truth = model.sample_prior(size, rng)
        stimulus, advice = design.sample(size, rng)
        choice, rt = model.simulate(truth, stimulus, advice, rng)
        yield truth, stimulus, advice, choice, rt
        done += size


def recovery_table(
    model: AutomationModel,
    truth: dict[str, np.ndarray],
    draws: dict[str, np.ndarray],
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Per-parameter recovery: R², NRMSE, posterior contraction and interval coverage.

    ``truth[name]`` has shape ``(n,)`` and ``draws[name]`` ``(n, num_draws)``.
    """
    prior = model.sample_prior(50_000, rng or np.random.default_rng(0))
    rows = []
    for name in model.param_names:
        samples, true = draws[name], truth[name]
        mean = samples.mean(axis=1)
        row = {
            "parameter": name,
            "r2": 1 - np.sum((mean - true) ** 2) / np.sum((true - true.mean()) ** 2),
            "nrmse": np.sqrt(np.mean((mean - true) ** 2)) / np.ptp(true),
            "contraction": 1 - samples.var(axis=1).mean() / prior[name].var(),
        }
        for level in (0.5, 0.9, 0.95):
            lower, upper = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2], axis=1)
            row[f"coverage_{round(level * 100)}"] = np.mean((true >= lower) & (true <= upper))
        rows.append(row)
    return pd.DataFrame(rows)


def sbc_table(
    model: AutomationModel,
    truth: dict[str, np.ndarray],
    draws: dict[str, np.ndarray],
    family_alpha: float = 0.05,
) -> pd.DataFrame:
    """Simulation-based calibration (Talts et al., 2018) per parameter.

    Uses the log-gamma test of rank uniformity (Modrák et al., 2025; BayesFlow's
    ``calibration_log_gamma``). The per-parameter level is Bonferroni-corrected so
    that the whole model fails with probability ``family_alpha`` when calibrated.
    A parameter passes when log-gamma ≥ 0.
    """
    from bayesflow.diagnostics.metrics import calibration_log_gamma

    names = list(model.param_names)
    estimates = np.stack([draws[name] for name in names], axis=-1)
    targets = np.stack([truth[name] for name in names], axis=-1)
    result = calibration_log_gamma(
        estimates, targets, variable_names=names, quantile=family_alpha / len(names)
    )
    log_gamma = np.asarray(result["values"]).reshape(-1)
    ranks = (estimates < targets[:, None, :]).mean(axis=1)
    return pd.DataFrame(
        {
            "parameter": names,
            "log_gamma": log_gamma,
            "passed": log_gamma >= 0,
            # Mean fractional rank: < 0.5 means the posterior tends to sit above the truth.
            "mean_rank": ranks.mean(axis=0),
        }
    )
