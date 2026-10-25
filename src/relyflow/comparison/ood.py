"""Out-of-distribution check: could any relyflow model plausibly have produced these data?

The observed operator's learned summary (from the model-comparison network) is
compared with the summaries of reference operators simulated from all candidate
models, with parameters drawn from the priors, *on the observed operator's own
design* (the same stimulus and advice sequence). Under the null hypothesis that
one of the models generated the data, the observed dataset is exchangeable with
the references, so the rank of its Mahalanobis distance among all ``n + 1``
distances gives an exact p-value: the false-alarm rate is at most ``alpha`` by
construction (a prior predictive check in learned summary space; cf. Schmitt et
al., 2024).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from relyflow.comparison.classifier import ModelComparison
from relyflow.data import NO_ADVICE, TrialData
from relyflow.models import get_model

FEATURES = ("summary", "statistics", "both")


@dataclass
class OODResult:
    """Outcome of the out-of-distribution check for one operator."""

    p_value: float
    distance: float
    flagged: bool
    alpha: float
    n_reference: int
    uncalibrated_p_value: float | None = None


def ood_statistics(
    stimulus: np.ndarray, advice: np.ndarray, choice: np.ndarray, rt: np.ndarray
) -> np.ndarray:
    """Interpretable statistics that complement the learned summary, ``(n, 17)``.

    Accuracy and correct-RT quantiles by trial type (as in the posterior predictive
    checks), the median log RT of errors, the spread of log RTs, the rate of following
    advice, and session trends: the slope of log RT over trials and the change in
    accuracy from the first to the last third. The learned summary is trained only to
    tell the models apart, so it can ignore features none of them produce, such as
    trends over the session.
    """
    return _fill_missing(_raw_statistics(stimulus, advice, choice, rt))


def _raw_statistics(stimulus, advice, choice, rt) -> np.ndarray:
    """:func:`ood_statistics` with NaN where a statistic is undefined."""
    from relyflow.diagnostics.ppc import ppc_statistics

    log_rt = np.log(rt)
    correct = choice == stimulus
    aided = advice != NO_ADVICE
    n_trials = rt.shape[-1]
    position = np.linspace(-0.5, 0.5, n_trials)
    slope = ((log_rt - log_rt.mean(axis=-1, keepdims=True)) * position).sum(axis=-1) / (
        position**2
    ).sum()
    third = max(n_trials // 3, 1)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # undefined statistics give NaN
        error_rt = np.nanmedian(np.where(correct, np.nan, log_rt), axis=-1)
        followed = np.where(aided, choice == advice, np.nan)
        stats = np.column_stack(
            [
                ppc_statistics(stimulus, advice, choice, log_rt),
                error_rt,
                np.subtract(*np.percentile(log_rt, [75, 25], axis=-1)),
                np.nanmean(followed, axis=-1),
                slope,
                correct[..., -third:].mean(axis=-1) - correct[..., :third].mean(axis=-1),
            ]
        )
    return stats


def _fill_missing(stats: np.ndarray) -> np.ndarray:
    """Statistics undefined for a dataset (e.g. no errors) take the median over the
    datasets compared with it (axis -2), or 0 if undefined for all of them."""
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(stats, axis=-2, keepdims=True)
    return np.nan_to_num(np.where(np.isnan(stats), medians, stats), nan=0.0)


def mahalanobis_ranks(summaries: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Mahalanobis distance of every row from the pooled mean, using the pooled covariance.

    Treating all rows symmetrically keeps the observed row exchangeable with the
    references, which makes the rank-based p-value exact.
    """
    centered = summaries - summaries.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    cov += ridge * np.trace(cov) / len(cov) * np.eye(len(cov))
    precision = np.linalg.inv(cov)
    return np.sqrt(np.einsum("ij,jk,ik->i", centered, precision, centered))


def rank_p_value(distances: np.ndarray, observed_index: int = 0) -> float:
    """Share of all distances (observed included) at least as large as the observed one."""
    return float(np.mean(distances >= distances[observed_index]))


def ood_check(
    comparison: ModelComparison,
    data: TrialData,
    n_reference: int = 500,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    features: str = "both",
) -> OODResult:
    """Flag data that the candidate models cannot plausibly produce.

    ``n_reference`` reference operators are simulated per check; with 500, the
    smallest attainable p-value is about 0.002 and the check takes 1-2 seconds.
    ``features`` selects the space the distance is measured in: the comparison
    network's learned ``"summary"``, the interpretable ``"statistics"`` of
    :func:`ood_statistics`, or ``"both"`` (default). All give exact false-alarm rates.
    """
    if features not in FEATURES:
        raise ValueError(f"features must be one of {FEATURES}")
    rng = rng or np.random.default_rng()
    models = [get_model(name) for name in comparison.models]
    n = data.n_trials
    stimulus = np.broadcast_to(data.stimulus, (n_reference, n))
    advice = np.broadcast_to(data.advice, (n_reference, n))
    index = rng.integers(0, len(models), size=n_reference)
    choice = np.empty((n_reference, n), dtype=np.int8)
    rt = np.empty((n_reference, n))
    for k, model in enumerate(models):
        chosen = index == k
        if chosen.any():
            params = model.sample_prior(int(chosen.sum()), rng)
            choice[chosen], rt[chosen] = model.simulate(
                params, stimulus[chosen], advice[chosen], rng
            )

    all_stimulus = np.vstack([data.stimulus, stimulus])
    all_advice = np.vstack([data.advice, advice])
    all_choice = np.vstack([data.choice, choice])
    all_rt = np.vstack([data.rt, rt])
    parts = []
    if features in ("summary", "both"):
        summaries = comparison.estimate_batch(all_stimulus, all_advice, all_choice, all_rt)
        parts.append(np.asarray(summaries["summaries"], dtype=np.float64))
    if features in ("statistics", "both"):
        parts.append(ood_statistics(all_stimulus, all_advice, all_choice, all_rt))
    distances = mahalanobis_ranks(np.hstack(parts))
    p_value = rank_p_value(distances, observed_index=0)
    return OODResult(
        p_value=p_value,
        distance=float(distances[0]),
        flagged=p_value <= alpha,
        alpha=alpha,
        n_reference=n_reference,
    )


def posterior_predictive_check(
    amortizer,
    data: TrialData,
    n_replicates: int = 200,
    n_calibration: int = 100,
    alpha: float = 0.05,
    seed: int | None = None,
) -> OODResult:
    """Does the fitted model reproduce this operator's data?

    Draws ``n_replicates`` parameter sets from the amortized posterior
    (``relyflow.inference.Amortizer``), simulates one dataset from each on the
    operator's own design, and measures how far the observed :func:`ood_statistics`
    lie from the replicates (Mahalanobis distance, standardized by the replicates'
    distances). This is much more sensitive than :func:`ood_check` to subtle misfit
    such as lapses or trends over the session, while :func:`ood_check` is better at
    gross departures that a fitted model partly absorbs.

    A posterior predictive p-value on its own is conservative: the posterior was
    fitted to the same data. It is calibrated (Hjort et al., 2006) by repeating the
    whole procedure, posterior included, for ``n_calibration`` datasets simulated
    from the model's prior on the same design, and ranking the observed discrepancy
    among theirs. If the model with prior-plausible parameters produced the data,
    the false-alarm rate is then exactly ``alpha`` (up to the rank resolution
    ``1 / (n_calibration + 1)``). Amortization makes this affordable: the
    ``n_calibration`` extra posteriors take one network pass. With the defaults the
    check takes a few seconds; ``n_calibration=0`` skips calibration (about 0.2 s)
    and returns the conservative p-value.
    """
    rng = np.random.default_rng(seed)
    model = amortizer.model
    choice, rt = data.choice[None], data.rt[None]
    if n_calibration:
        n = data.n_trials
        null_choice, null_rt = model.simulate(
            model.sample_prior(n_calibration, rng),
            np.broadcast_to(data.stimulus, (n_calibration, n)),
            np.broadcast_to(data.advice, (n_calibration, n)),
            rng,
        )
        choice, rt = np.vstack([choice, null_choice]), np.vstack([rt, null_rt])
    distance, discrepancy, raw = _posterior_discrepancies(
        amortizer, data.stimulus, data.advice, choice, rt, n_replicates, rng, seed
    )
    p_value = rank_p_value(discrepancy) if n_calibration else raw[0]
    return OODResult(
        p_value=float(p_value),
        distance=float(distance[0]),
        flagged=bool(p_value <= alpha),
        alpha=alpha,
        n_reference=n_calibration or n_replicates,
        uncalibrated_p_value=float(raw[0]),
    )


def _posterior_discrepancies(
    amortizer,
    stimulus: np.ndarray,
    advice: np.ndarray,
    choice: np.ndarray,
    rt: np.ndarray,
    n_replicates: int,
    rng: np.random.Generator,
    seed: int | None,
    chunk: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior predictive comparison for ``m`` datasets sharing one design.

    ``stimulus`` and ``advice`` have shape ``(n_trials,)``; ``choice`` and ``rt``
    ``(m, n_trials)``. Every dataset is treated identically: its own posterior, its
    own replicates. Returns, each of shape ``(m,)``, the dataset's Mahalanobis
    distance among its replicates, that distance standardized by the replicates'
    distances, and the uncalibrated posterior predictive p-value.
    """
    model = amortizer.model
    m, n = choice.shape
    draws = amortizer.sample_batch(
        np.broadcast_to(stimulus, (m, n)),
        np.broadcast_to(advice, (m, n)),
        choice,
        rt,
        num_samples=n_replicates,
        seed=seed,
    )
    distance, discrepancy, p_value = np.empty(m), np.empty(m), np.empty(m)
    group = n_replicates + 1
    for start in range(0, m, chunk):
        stop = min(start + chunk, m)
        k = stop - start
        params = {name: values[start:stop].reshape(-1) for name, values in draws.items()}
        rep_choice, rep_rt = model.simulate(
            params,
            np.broadcast_to(stimulus, (k * n_replicates, n)),
            np.broadcast_to(advice, (k * n_replicates, n)),
            rng,
        )
        # Each group of rows is one dataset followed by its replicates.
        all_choice = np.concatenate(
            [choice[start:stop, None], rep_choice.reshape(k, n_replicates, n)], axis=1
        ).reshape(k * group, n)
        all_rt = np.concatenate(
            [rt[start:stop, None], rep_rt.reshape(k, n_replicates, n)], axis=1
        ).reshape(k * group, n)
        stats = _raw_statistics(
            np.broadcast_to(stimulus, (k * group, n)),
            np.broadcast_to(advice, (k * group, n)),
            all_choice,
            all_rt,
        )
        stats = _fill_missing(stats.reshape(k, group, -1))
        for j in range(k):
            d = mahalanobis_ranks(stats[j])
            distance[start + j] = d[0]
            discrepancy[start + j] = (d[0] - d[1:].mean()) / d[1:].std()
            p_value[start + j] = rank_p_value(d)
    return distance, discrepancy, p_value
