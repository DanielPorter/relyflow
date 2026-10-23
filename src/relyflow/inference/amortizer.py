"""Load a trained network and draw posterior samples for operators' data."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import keras
import numpy as np

from relyflow.data import NO_ADVICE, TrialData
from relyflow.inference import pretrained
from relyflow.inference.features import design_features, learning_statistics, network_inputs
from relyflow.inference.training import TrainingConfig, model_signature
from relyflow.models import MODELS, AutomationModel, get_model


class Amortizer:
    """A trained posterior network for one model.

    Use :meth:`load` to read one saved by ``relyflow.inference.training``.
    """

    def __init__(self, approximator, config: TrainingConfig, metadata: dict):
        self.approximator = approximator
        self.config = config
        self.metadata = metadata
        self.model: AutomationModel = get_model(config.model)

    @classmethod
    def load(cls, directory: str | Path) -> Amortizer:
        directory = Path(directory)
        saved = json.loads((directory / "config.json").read_text())
        config = TrainingConfig.from_dict(saved["config"])
        metadata = saved["metadata"]
        if metadata["model_signature"] != _jsonable(model_signature(get_model(config.model))):
            warnings.warn(
                f"The {config.model!r} model's parameters or priors have changed since this "
                "network was trained. Its posteriors are for the old model; retrain it.",
                stacklevel=2,
            )
        approximator = keras.saving.load_model(directory / "approximator.keras")
        return cls(approximator, config, metadata)

    def sample(
        self, data: TrialData, num_samples: int = 2000, seed: int | None = None
    ) -> dict[str, np.ndarray]:
        """Posterior draws for one operator, each of shape ``(num_samples,)``."""
        draws = self.sample_batch(
            data.stimulus, data.advice, data.choice, data.rt, num_samples=num_samples, seed=seed
        )
        return {name: values[0] for name, values in draws.items()}

    def sample_batch(
        self,
        stimulus: np.ndarray,
        advice: np.ndarray,
        choice: np.ndarray,
        rt: np.ndarray,
        num_samples: int = 2000,
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Posterior draws for datasets sharing a trial count.

        Arrays have shape ``(n_datasets, n_trials)`` (or ``(n_trials,)`` for one);
        each returned parameter has shape ``(n_datasets, num_samples)``.
        """
        stimulus, advice, choice, rt = (np.atleast_2d(x) for x in (stimulus, advice, choice, rt))
        for message in self.design_warnings(advice):
            warnings.warn(message, stacklevel=2)
        conditions = {
            "trials": network_inputs(stimulus, advice, choice, rt, self.config.history_features),
            "design": design_features(advice),
        }
        if self.config.learning_statistics:
            conditions["statistics"] = learning_statistics(stimulus, advice, choice, rt)
        draws = self.approximator.sample(num_samples=num_samples, conditions=conditions, seed=seed)
        return {name: np.asarray(draws[name]).reshape(len(stimulus), num_samples)
                for name in self.model.param_names}  # fmt: skip

    def design_warnings(self, advice: np.ndarray) -> list[str]:
        """Ways in which a design falls outside what the network was trained on."""
        design = self.config.design
        aided = np.atleast_2d(advice) != NO_ADVICE
        n_trials = aided.shape[-1]
        messages = []
        if not design.n_trials[0] <= n_trials <= design.n_trials[1]:
            messages.append(
                f"{n_trials} trials is outside the trained range {design.n_trials}; "
                "posteriors may be unreliable."
            )
        if np.any((~aided).sum(axis=-1) == 0):
            messages.append(
                "No unaided trials: excitation and inhibition are not identifiable without them."
            )
        return messages


def _jsonable(value):
    return json.loads(json.dumps(value))


def load(name_or_path: str | Path) -> Amortizer:
    """Load a trained network.

    Pass a model name (``"advisory"``, ``"reliance"`` or ``"learning"``) for the
    pretrained network, downloaded on first use (see ``relyflow.inference.pretrained``),
    or the path of a directory saved by ``relyflow.inference.training``.
    """
    if isinstance(name_or_path, str) and name_or_path in MODELS:
        return Amortizer.load(pretrained.fetch(name_or_path))
    return Amortizer.load(name_or_path)


__all__ = ["Amortizer", "load"]
