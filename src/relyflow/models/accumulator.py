"""Static advisory and reliance models.

A two-accumulator linear ballistic accumulator (Brown & Heathcote, 2008) in which
the aid's advice changes accumulation rates, following Strickland et al. (2021,
2023). The published models estimate a free rate for every design cell; here the
advice effects are two parameters, excitation and inhibition, matching the
papers' definitions of those mechanisms. The full specification is in
``docs/models/static.md``.

Accumulator ``k`` produces response ``k``. All arrays are shaped
``(n_sims, n_trials)`` or ``(n_sims, n_trials, 2)`` with the last axis indexing
the accumulator.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import stats

from relyflow.data import NO_ADVICE
from relyflow.models.base import AutomationModel, Parameter, truncnorm
from relyflow.models.lba import race_density, sample_race

_ACCUMULATORS = np.array([0, 1])

LBA_PARAMETERS = (
    Parameter(
        "v_match_0",
        "Unaided mean rate of the correct accumulator when the stimulus is 0",
        truncnorm(2.5, 1.0, 0.0, 6.0),
    ),
    Parameter(
        "v_match_1",
        "Unaided mean rate of the correct accumulator when the stimulus is 1",
        truncnorm(2.5, 1.0, 0.0, 6.0),
    ),
    Parameter(
        "v_mismatch_0",
        "Unaided mean rate of the error accumulator when the stimulus is 0",
        truncnorm(0.5, 1.0, -3.0, 4.0),
    ),
    Parameter(
        "v_mismatch_1",
        "Unaided mean rate of the error accumulator when the stimulus is 1",
        truncnorm(0.5, 1.0, -3.0, 4.0),
    ),
    Parameter(
        "sv_match",
        "SD of the correct accumulator's rate (the error accumulator's SD is fixed at 1)",
        truncnorm(1.0, 0.5, 0.1, 3.0),
    ),
    Parameter(
        "A", "Upper bound of the uniform start-point distribution", truncnorm(2.0, 1.0, 0.1, 5.0)
    ),
    Parameter(
        "B_0",
        "Threshold gap b - A of the accumulator for response 0",
        truncnorm(2.0, 1.0, 0.1, 6.0),
    ),
    Parameter(
        "B_1",
        "Threshold gap b - A of the accumulator for response 1",
        truncnorm(2.0, 1.0, 0.1, 6.0),
    ),
    Parameter(
        "B_aided_scale",
        "Multiplier on both threshold gaps on aided trials",
        stats.lognorm(s=0.2),
    ),
    Parameter("t0", "Non-decision time in seconds", truncnorm(0.3, 0.15, 0.05, 1.0)),
)

INHIBITION = Parameter(
    "inhibition",
    "Rate decrease of the accumulator that disagrees with the advice",
    stats.halfnorm(scale=1.0),
)
EXCITATION = Parameter(
    "excitation",
    "Rate increase of the accumulator that agrees with the advice",
    stats.halfnorm(scale=1.0),
)


def as_design(stimulus: np.ndarray, advice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast a design to 2-D ``(n_sims or 1, n_trials)`` integer arrays."""
    return np.broadcast_arrays(np.atleast_2d(stimulus), np.atleast_2d(advice))


def unaided_rate_means(params: Mapping[str, np.ndarray], stimulus: np.ndarray) -> np.ndarray:
    """Mean rates of both accumulators without advice, shape ``(n_sims, n_trials, 2)``."""
    stim0 = stimulus == 0
    match = np.where(stim0, params["v_match_0"][:, None], params["v_match_1"][:, None])
    mismatch = np.where(stim0, params["v_mismatch_0"][:, None], params["v_mismatch_1"][:, None])
    is_match = stimulus[..., None] == _ACCUMULATORS
    return np.where(is_match, match[..., None], mismatch[..., None])


def rate_sds(params: Mapping[str, np.ndarray], stimulus: np.ndarray) -> np.ndarray:
    """Rate SDs: ``sv_match`` for the correct accumulator, 1 for the error accumulator."""
    is_match = stimulus[..., None] == _ACCUMULATORS
    return np.where(is_match, params["sv_match"][:, None, None], 1.0)


def advice_effects(
    advice: np.ndarray, excitation: np.ndarray, inhibition: np.ndarray
) -> np.ndarray:
    """Change in each accumulator's mean rate caused by the advice.

    The accumulator agreeing with the advice gains ``excitation`` and the other
    loses ``inhibition``. Both arguments broadcast against ``(n_sims, n_trials)``.
    Unaided trials are unaffected.
    """
    agrees = advice[..., None] == _ACCUMULATORS
    effect = np.where(agrees, np.asarray(excitation)[..., None], -np.asarray(inhibition)[..., None])
    return np.where((advice != NO_ADVICE)[..., None], effect, 0.0)


def thresholds(params: Mapping[str, np.ndarray], advice: np.ndarray) -> np.ndarray:
    """Thresholds ``b = A + B_k`` per accumulator, with ``B_k`` scaled on aided trials."""
    gaps = np.stack([params["B_0"], params["B_1"]], axis=-1)[:, None, :]
    scale = np.where(advice != NO_ADVICE, params["B_aided_scale"][:, None], 1.0)
    return params["A"][:, None, None] + gaps * scale[..., None]


class StaticAccumulator(AutomationModel):
    """Shared machinery of the advisory and reliance models."""

    def rate_means(
        self, params: Mapping[str, np.ndarray], stimulus: np.ndarray, advice: np.ndarray
    ) -> np.ndarray:
        """Mean rates of both accumulators on every trial, shape ``(n_sims, n_trials, 2)``."""
        stimulus, advice = as_design(stimulus, advice)
        excitation = params.get("excitation", np.zeros_like(params["inhibition"]))
        effects = advice_effects(advice, excitation[:, None], params["inhibition"][:, None])
        return unaided_rate_means(params, stimulus) + effects

    def race_inputs(
        self, params: Mapping[str, np.ndarray], stimulus: np.ndarray, advice: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mean rates, rate SDs and thresholds, each ``(n_sims, n_trials, 2)``."""
        stimulus, advice = as_design(stimulus, advice)
        return (
            self.rate_means(params, stimulus, advice),
            rate_sds(params, stimulus),
            thresholds(params, advice),
        )

    def simulate(self, params, stimulus, advice, rng):
        means, sds, bounds = self.race_inputs(params, stimulus, advice)
        return sample_race(rng, means, sds, bounds, params["A"][:, None], params["t0"][:, None])

    def log_likelihood(
        self,
        params: Mapping[str, np.ndarray],
        stimulus: np.ndarray,
        advice: np.ndarray,
        choice: np.ndarray,
        rt: np.ndarray,
    ) -> np.ndarray:
        """Exact log-likelihood of each parameter set given the trials, shape ``(n_sims,)``.

        Trials broadcast like the design, so one operator's data can be scored
        against many parameter sets at once.
        """
        means, sds, bounds = self.race_inputs(params, stimulus, advice)
        density = race_density(
            choice, rt, means, sds, bounds, params["A"][:, None], params["t0"][:, None]
        )
        tiny = np.finfo(np.float64).tiny
        density = np.where(np.isfinite(density) & (density > tiny), density, tiny)
        return np.log(density).sum(axis=-1)


class Advisory(StaticAccumulator):
    """The aid acts as an advisor: it inhibits the disagreeing accumulator only.

    Advice cannot trigger a decision on its own, so the operator must still
    process the task (Strickland et al., 2021).
    """

    name = "advisory"
    parameters = (*LBA_PARAMETERS, INHIBITION)


class Reliance(StaticAccumulator):
    """The operator also accumulates the advice directly (excitation).

    This grants the aid some autonomy in triggering decisions, as found for a
    95%-reliable aid by Strickland et al. (2023).
    """

    name = "reliance"
    parameters = (*LBA_PARAMETERS, INHIBITION, EXCITATION)
