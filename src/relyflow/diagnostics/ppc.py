"""Posterior predictive checks on accuracy and RT quantiles."""

from __future__ import annotations

import warnings

import numpy as np

from relyflow.data import NO_ADVICE
from relyflow.models.base import AutomationModel

CELLS = ("unaided", "aid_right", "aid_wrong")
RT_QUANTILES = (0.1, 0.5, 0.9)
PPC_STATISTICS = tuple(
    [f"accuracy_{cell}" for cell in CELLS]
    + [f"correct_rt_q{round(q * 100)}_{cell}" for cell in CELLS for q in RT_QUANTILES]
)


def ppc_statistics(
    stimulus: np.ndarray, advice: np.ndarray, choice: np.ndarray, rt: np.ndarray
) -> np.ndarray:
    """Accuracy and correct-RT quantiles per trial type, shape ``(..., len(PPC_STATISTICS))``.

    Statistics for trial types an operator never saw are NaN.
    """
    aided = advice != NO_ADVICE
    masks = {
        "unaided": ~aided,
        "aid_right": aided & (advice == stimulus),
        "aid_wrong": aided & (advice != stimulus),
    }
    correct = choice == stimulus
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # empty cells give NaN
        accuracy = [np.nanmean(np.where(masks[c], correct, np.nan), axis=-1) for c in CELLS]
        quantiles = []
        for cell in CELLS:
            correct_rt = np.where(masks[cell] & correct, rt, np.nan)
            quantiles.extend(np.nanquantile(correct_rt, q, axis=-1) for q in RT_QUANTILES)
    return np.stack(accuracy + quantiles, axis=-1)


def posterior_predictive(
    model: AutomationModel,
    draws: dict[str, np.ndarray],
    stimulus: np.ndarray,
    advice: np.ndarray,
    rng: np.random.Generator,
    num_replicates: int = 200,
) -> np.ndarray:
    """PPC statistics of datasets replicated from posterior draws, ``(num_replicates, n_stats)``.

    Each replicate uses a different posterior draw and the operator's own design.
    """
    n_draws = len(next(iter(draws.values())))
    index = rng.choice(n_draws, size=num_replicates, replace=num_replicates > n_draws)
    params = {name: values[index] for name, values in draws.items()}
    stimulus = np.broadcast_to(stimulus, (num_replicates, len(stimulus)))
    advice = np.broadcast_to(advice, (num_replicates, len(advice)))
    choice, rt = model.simulate(params, stimulus, advice, rng)
    return ppc_statistics(stimulus, advice, choice, rt)


def ppc_check(observed: np.ndarray, replicated: np.ndarray) -> dict[str, np.ndarray]:
    """Two-sided posterior predictive p-values and whether each statistic falls outside
    the central 95% predictive interval."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        upper_tail = np.nanmean(replicated >= observed, axis=0)
        lower_tail = np.nanmean(replicated <= observed, axis=0)
        p_value = np.minimum(1.0, 2 * np.minimum(upper_tail, lower_tail))
        lo, hi = np.nanquantile(replicated, [0.025, 0.975], axis=0)
    outside = (observed < lo) | (observed > hi)
    return {"p_value": p_value, "outside_95": np.where(np.isnan(observed), np.nan, outside)}
