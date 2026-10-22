"""relyflow: amortized Bayesian inference for models of human reliance on automation."""

from importlib.metadata import PackageNotFoundError, version

from relyflow.data import TrialData, split_operators

try:
    __version__ = version("relyflow")
except PackageNotFoundError:  # source tree without an install
    __version__ = "0+unknown"

__all__ = ["TrialData", "__version__", "split_operators"]
