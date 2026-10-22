"""Automation-use models: parameters, priors and simulators (milestones 1-2).

Specifications, with sources, are in ``docs/models/``.
"""

from relyflow.models.accumulator import Advisory, Reliance, StaticAccumulator
from relyflow.models.base import AutomationModel, Parameter
from relyflow.models.learning import ReliabilityLearning

MODELS: dict[str, type[AutomationModel]] = {
    model.name: model for model in (Advisory, Reliance, ReliabilityLearning)
}


def get_model(name: str) -> AutomationModel:
    """Instantiate a model by its registered name ("advisory", "reliance" or "learning")."""
    try:
        return MODELS[name]()
    except KeyError:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}") from None


__all__ = [
    "MODELS",
    "Advisory",
    "AutomationModel",
    "Parameter",
    "ReliabilityLearning",
    "Reliance",
    "StaticAccumulator",
    "get_model",
]
