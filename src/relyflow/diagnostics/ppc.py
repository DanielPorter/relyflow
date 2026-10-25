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
        quantiles.extend(nan_quantiles(correct_rt, RT_QUANTILES))
    return np.stack(accuracy + quantiles, axis=-1)


def nan_quantiles(x: np.ndarray, q) -> np.ndarray:
    """``np.nanquantile(x, q, axis=-1)`` (linear interpolation) with one sort.

    Several times faster than NumPy's version on many rows. Rows with no
    finite values give NaN. Returns shape ``(len(q), *x.shape[:-1])``.
    """
    q = np.asarray(q, dtype=np.float64)
    count = np.count_nonzero(~np.isnan(x), axis=-1)[..., None]
    ordered = np.sort(x, axis=-1)  # NaN sorts last
    position = np.maximum(count - 1, 0) * q
    lower = np.floor(position).astype(np.intp)
    upper = np.minimum(lower + 1, np.maximum(count - 1, 0))
    low = np.take_along_axis(ordered, lower, axis=-1)
    high = np.take_along_axis(ordered, upper, axis=-1)
    with np.errstate(invalid="ignore"):
        values = low + (position - lower) * (high - low)
    return np.moveaxis(np.where(count > 0, values, np.nan), -1, 0)


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
