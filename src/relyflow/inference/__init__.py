"""Amortized posterior estimation with BayesFlow (milestone 3).

BayesFlow and Keras are imported here, never from the top-level package, so
`import relyflow` stays fast and works without a Keras backend installed.
"""

# BayesFlow must be imported before Keras: when KERAS_BACKEND is unset it selects an
# installed backend, whereas Keras alone falls back to its config (often TensorFlow).
import bayesflow  # noqa: F401

from relyflow.inference.amortizer import Amortizer, load
from relyflow.inference.designs import DesignPrior
from relyflow.inference.features import DESIGN_FEATURES, TRIAL_FEATURES
from relyflow.inference.training import TrainingConfig, train

__all__ = [
    "DESIGN_FEATURES",
    "TRIAL_FEATURES",
    "Amortizer",
    "DesignPrior",
    "TrainingConfig",
    "load",
    "train",
]
