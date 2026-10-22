"""Reliability-learning extension of the reliance model.

The operator's perceived reliability of the aid is updated after every aided
trial with the delta rule of Strickland et al. (2024, Eq. 2), and scales the
advice effects of the accumulator model. Coupling the learning rule to the
accumulator was proposed by Strickland et al. (2024) but not fitted there; the
coupling used here is relyflow's own and is specified in ``docs/models/learning.md``.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import stats

from relyflow.data import NO_ADVICE
from relyflow.models.accumulator import (
    LBA_PARAMETERS,
    StaticAccumulator,
    advice_effects,
    as_design,
    unaided_rate_means,
)
from relyflow.models.base import Parameter


def delta_rule_beliefs(
    r0: np.ndarray, alpha: np.ndarray, stimulus: np.ndarray, advice: np.ndarray
) -> np.ndarray:
    """Perceived reliability held *before* each trial, shape ``(n_sims, n_trials)``.

    After an aided trial, ``r <- r + alpha * (aid_correct - r)``. Unaided trials
    give no evidence about the aid and leave ``r`` unchanged. The operator is
    assumed to learn whether the aid was right after each trial (e.g. through
    error feedback).
    """
    stimulus, advice = as_design(stimulus, advice)
    n_trials = advice.shape[1]
    aided = advice != NO_ADVICE
    aid_correct = (advice == stimulus).astype(np.float64)
    r = np.array(r0, dtype=np.float64)
    beliefs = np.empty(np.broadcast_shapes((*r.shape, 1), advice.shape))
    for t in range(n_trials):
        beliefs[:, t] = r
        r = np.where(aided[:, t], r + alpha * (aid_correct[:, t] - r), r)
    return beliefs


def perceived_validity(reliability: np.ndarray) -> np.ndarray:
    """How much the advice is worth acting on: 0 at chance (r <= 0.5), 1 for a perfect aid."""
    return np.clip(2.0 * np.asarray(reliability) - 1.0, 0.0, 1.0)


class ReliabilityLearning(StaticAccumulator):
    """Reliance model whose advice effects track learned reliability trial by trial."""

    name = "learning"
    parameters = (
        *LBA_PARAMETERS,
        Parameter(
            "inhibition",
            "Inhibition when the aid is believed perfectly reliable",
            stats.halfnorm(scale=1.0),
        ),
        Parameter(
            "excitation",
            "Excitation when the aid is believed perfectly reliable",
            stats.halfnorm(scale=1.0),
        ),
        Parameter(
            "r0", "Perceived reliability before any experience with the aid", stats.beta(4, 2)
        ),
        Parameter("alpha", "Delta-rule learning rate", stats.beta(1, 8)),
    )

    def rate_means(
        self, params: Mapping[str, np.ndarray], stimulus: np.ndarray, advice: np.ndarray
    ) -> np.ndarray:
        stimulus, advice = as_design(stimulus, advice)
        beliefs = delta_rule_beliefs(params["r0"], params["alpha"], stimulus, advice)
        weight = perceived_validity(beliefs)
        effects = advice_effects(
            advice,
            params["excitation"][:, None] * weight,
            params["inhibition"][:, None] * weight,
        )
        return unaided_rate_means(params, stimulus) + effects
