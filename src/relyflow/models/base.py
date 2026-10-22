"""Interface every automation-use model implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Parameter:
    """A model parameter: its name, what it means, and its prior.

    ``prior`` is a frozen ``scipy.stats`` distribution.
    """

    name: str
    description: str
    prior: Any

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.asarray(self.prior.rvs(size=n, random_state=rng), dtype=np.float64)

    @property
    def support(self) -> tuple[float, float]:
        lower, upper = self.prior.support()
        return float(lower), float(upper)


def truncnorm(mean: float, sd: float, lower: float = -np.inf, upper: float = np.inf):
    """Normal(mean, sd) truncated to [lower, upper], as a frozen scipy distribution."""
    return stats.truncnorm((lower - mean) / sd, (upper - mean) / sd, loc=mean, scale=sd)


class AutomationModel(ABC):
    """A generative model of an operator's choices and RTs given the aid's advice.

    Simulators must be vectorized over parameter sets: simulating 10,000
    operators is one call, not a Python loop. Training sets for the inference
    networks are built from millions of such draws.
    """

    name: ClassVar[str]
    parameters: ClassVar[tuple[Parameter, ...]]

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    def sample_prior(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw ``n`` parameter sets from the independent priors; each value has shape ``(n,)``."""
        return {p.name: p.sample(n, rng) for p in self.parameters}

    @abstractmethod
    def simulate(
        self,
        params: Mapping[str, np.ndarray],
        stimulus: np.ndarray,
        advice: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate choices and response times.

        Parameters
        ----------
        params
            Parameter arrays of shape ``(n_sims,)``, keyed by ``param_names``.
        stimulus, advice
            Arrays of shape ``(n_sims, n_trials)`` in presentation order, with
            ``advice == NO_ADVICE`` on unaided trials. Order matters for models
            that learn the aid's reliability.
        rng
            Source of randomness; simulators must not use global RNG state.

        Returns
        -------
        choice, rt
            Arrays of shape ``(n_sims, n_trials)``: 0/1 choices and RTs in seconds.
        """
