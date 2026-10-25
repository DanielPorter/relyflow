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
    with np.errstate(invalid="ignore"):
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
    # Statistics undefined for a dataset (e.g. no errors) take the column median.
    medians = np.nanmedian(stats, axis=0)
    return np.where(np.isnan(stats), medians, stats)


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
    n_replicates: int = 300,
    alpha: float = 0.05,
    seed: int | None = None,
) -> OODResult:
    """Does the fitted model reproduce this operator's data?

    Draws ``n_replicates`` parameter sets from the amortized posterior
    (``relyflow.inference.Amortizer``), simulates one dataset from each on the
    operator's own design, and ranks the observed :func:`ood_statistics` among the
    replicates by Mahalanobis distance. Because the posterior has already been fitted
    to the observed data, this is conservative (false alarms below ``alpha``), but it
    is much more sensitive than :func:`ood_check` to subtle misfit such as lapses or
    trends over the session. Complements rather than replaces :func:`ood_check`,
    which catches gross departures that a fitted model partly absorbs.
    """
    rng = np.random.default_rng(seed)
    model = amortizer.model
    draws = amortizer.sample(data, num_samples=n_replicates, seed=seed)
    n = data.n_trials
    stimulus = np.broadcast_to(data.stimulus, (n_replicates, n))
    advice = np.broadcast_to(data.advice, (n_replicates, n))
    choice, rt = model.simulate(draws, stimulus, advice, rng)
    stats = ood_statistics(
        np.vstack([data.stimulus, stimulus]),
        np.vstack([data.advice, advice]),
        np.vstack([data.choice, choice]),
        np.vstack([data.rt, rt]),
    )
    distances = mahalanobis_ranks(stats)
    p_value = rank_p_value(distances, observed_index=0)
    return OODResult(
        p_value=p_value,
        distance=float(distances[0]),
        flagged=p_value <= alpha,
        alpha=alpha,
        n_reference=n_replicates,
    )
