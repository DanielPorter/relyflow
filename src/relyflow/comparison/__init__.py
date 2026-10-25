"""Amortized model comparison and out-of-distribution checks (milestone 5).

- :class:`ModelComparison`: posterior probabilities of the advisory, reliance and
  learning models for an operator.
- :func:`ood_check`: flags data that none of the models can plausibly produce.
- :func:`posterior_predictive_check`: flags data the fitted model does not reproduce.
"""

# BayesFlow must be imported before Keras (see relyflow.inference).
import bayesflow  # noqa: F401

from relyflow.comparison.classifier import MODELS, ComparisonConfig, ModelComparison, train
from relyflow.comparison.ood import OODResult, ood_check, posterior_predictive_check

__all__ = [
    "MODELS",
    "ComparisonConfig",
    "ModelComparison",
    "OODResult",
    "ood_check",
    "posterior_predictive_check",
    "train",
]
