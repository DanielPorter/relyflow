"""Interface every automation-use model implements.

Provisional: the parameter sets and priors are fixed in roadmap milestone 1,
and this contract may change with them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

import numpy as np


class AutomationModel(ABC):
    """A generative model of an operator's choices and RTs given the aid's advice.

    Simulators must be vectorized over parameter sets: simulating 10,000
    operators is one call, not a Python loop. Training sets for the inference
    networks are built from millions of such draws.
    """

    name: ClassVar[str]
    param_names: ClassVar[tuple[str, ...]]

    @abstractmethod
    def sample_prior(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Draw ``n`` parameter sets. Each value has shape ``(n,)``."""

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
            Arrays of shape ``(n_sims, n_trials)`` in presentation order. Order
            matters for models that learn the aid's reliability.
        rng
            Source of randomness; simulators must not use global RNG state.

        Returns
        -------
        choice, rt
            Arrays of shape ``(n_sims, n_trials)``: 0/1 choices and RTs in seconds.
        """
