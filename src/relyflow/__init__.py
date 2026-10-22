"""relyflow: amortized Bayesian inference for models of human reliance on automation."""

from importlib.metadata import PackageNotFoundError, version

from relyflow.data import NO_ADVICE, TrialData, split_operators, summary_stats
from relyflow.design import make_design

try:
    __version__ = version("relyflow")
except PackageNotFoundError:  # source tree without an install
    __version__ = "0+unknown"

__all__ = [
    "NO_ADVICE",
    "TrialData",
    "__version__",
    "make_design",
    "split_operators",
    "summary_stats",
]
