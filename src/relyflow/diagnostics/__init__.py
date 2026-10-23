"""Validation suite (milestone 4): recovery, simulation-based calibration,
posterior predictive checks and the MCMC benchmark.

Run everything for one network with ``python -m relyflow.diagnostics validate``.
"""

from relyflow.diagnostics.calibration import recovery_table, sbc_table, simulate_operators
from relyflow.diagnostics.ppc import (
    PPC_STATISTICS,
    posterior_predictive,
    ppc_check,
    ppc_statistics,
)

__all__ = [
    "PPC_STATISTICS",
    "posterior_predictive",
    "ppc_check",
    "ppc_statistics",
    "recovery_table",
    "sbc_table",
    "simulate_operators",
]
