"""Linear ballistic accumulator race: sampler and analytic density.

Each accumulator starts at ``x ~ U(0, A)``, rises at a rate drawn from a normal
distribution truncated below at zero, and finishes at ``(b - x) / rate``. The
first to finish gives the response; RT adds the non-decision time ``t0``
(Brown & Heathcote, 2008). Because rates are truncated at zero, every
accumulator eventually finishes and the densities are normalized by
``Phi(v / s)``, as in the source papers' software.

Array arguments broadcast against each other; the last axis indexes the
accumulator.
"""

from __future__ import annotations

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri_exp

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def sample_race(
    rng: np.random.Generator,
    mean_rates: np.ndarray,
    rate_sds: np.ndarray,
    thresholds: np.ndarray,
    A: np.ndarray,
    t0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample responses and RTs.

    ``mean_rates``, ``rate_sds`` and ``thresholds`` have shape ``(..., n_accumulators)``;
    ``A`` and ``t0`` broadcast against ``(...)``. Returns the index of the winning
    accumulator (int8) and the RT, each of shape ``(...)``.
    """
    mean_rates, rate_sds, thresholds = np.broadcast_arrays(mean_rates, rate_sds, thresholds)
    A = np.asarray(A)[..., None]
    starts = rng.random(mean_rates.shape) * A
    rates = _truncated_normal(rng, mean_rates, rate_sds)
    times = (thresholds - starts) / rates
    choice = times.argmin(axis=-1).astype(np.int8)
    return choice, np.asarray(t0) + times.min(axis=-1)


def _truncated_normal(rng: np.random.Generator, mean: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Normal(mean, sd) truncated to (0, inf), by inversion in log space.

    Inverting the lower tail stays accurate when ``mean / sd`` is very negative,
    where the naive ``Phi^-1(U(Phi(-m/s), 1))`` would round to infinity.
    """
    log_mass = log_ndtr(mean / sd)  # log P(rate > 0) before truncation
    log_u = np.log1p(-rng.random(mean.shape))  # log of U(0, 1], avoids log(0)
    return mean - sd * ndtri_exp(log_u + log_mass)


def accumulator_pdf_cdf(
    t: np.ndarray, v: np.ndarray, s: np.ndarray, b: np.ndarray, A: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Density and CDF of one accumulator's finishing time ``t`` (decision time, t > 0).

    Brown & Heathcote (2008), Eqs. 1-2, divided by ``Phi(v / s)`` for rates
    truncated at zero. Returns zeros for ``t <= 0``.
    """
    t, v, s, b, A = (np.array(x, dtype=np.float64) for x in np.broadcast_arrays(t, v, s, b, A))
    positive = t > 0
    ts = np.where(positive, t, 1.0) * s
    tv = np.where(positive, t, 1.0) * v
    z_lo = (b - A - tv) / ts
    z_hi = (b - tv) / ts
    phi_lo = np.exp(-0.5 * z_lo**2) / _SQRT_2PI
    phi_hi = np.exp(-0.5 * z_hi**2) / _SQRT_2PI
    pdf = (-v * ndtr(z_lo) + s * phi_lo + v * ndtr(z_hi) - s * phi_hi) / A
    cdf = (
        1.0 + (b - A - tv) / A * ndtr(z_lo) - (b - tv) / A * ndtr(z_hi) + ts / A * (phi_lo - phi_hi)
    )
    mass = ndtr(v / s)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdf = np.where(positive, np.clip(pdf, 0.0, None) / mass, 0.0)
        cdf = np.where(positive, np.clip(cdf / mass, 0.0, 1.0), 0.0)

    # Far in the truncation tail the closed form is a cancellation of O(1) terms
    # divided by a vanishing Phi(v/s); use stable quadrature there instead.
    tail = positive & (v / s < _TAIL_Z)
    if tail.any():
        pdf[tail], cdf[tail] = _tail_pdf_cdf(t[tail], v[tail], s[tail], b[tail], A[tail])
    return pdf, cdf


_TAIL_Z = -5.0
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(32)


def _tail_pdf_cdf(t, v, s, b, A):
    """Finishing-time density and CDF by quadrature over the start point, in log space.

    Given start point x, the accumulator finishes by t when its rate exceeds
    c = (b - x) / t, and the density of t is g(c) * c / t, where g is the
    truncated-normal rate density. Both are averaged over x ~ U(0, A) with
    Gauss-Legendre quadrature, using log-space normal tails throughout.
    """
    x = 0.5 * A[:, None] * (1.0 + _GL_NODES)  # (n, nodes)
    c = (b[:, None] - x) / t[:, None]
    z = (c - v[:, None]) / s[:, None]
    log_mass = log_ndtr(v / s)[:, None]
    log_g = -0.5 * z**2 - np.log(_SQRT_2PI * s[:, None]) - log_mass
    pdf = 0.5 * np.sum(_GL_WEIGHTS * np.exp(log_g) * c / t[:, None], axis=1)
    cdf = 0.5 * np.sum(_GL_WEIGHTS * np.exp(log_ndtr(-z) - log_mass), axis=1)
    return pdf, np.clip(cdf, 0.0, 1.0)


def race_density(
    choice: np.ndarray,
    rt: np.ndarray,
    mean_rates: np.ndarray,
    rate_sds: np.ndarray,
    thresholds: np.ndarray,
    A: np.ndarray,
    t0: np.ndarray,
) -> np.ndarray:
    """Joint density of ``(choice, rt)``: the winner's density times the others' survival."""
    decision_time = np.asarray(rt) - np.asarray(t0)
    pdf, cdf = accumulator_pdf_cdf(
        decision_time[..., None], mean_rates, rate_sds, thresholds, np.asarray(A)[..., None]
    )
    n_accumulators = pdf.shape[-1]
    won = np.asarray(choice)[..., None] == np.arange(n_accumulators)
    return np.prod(np.where(won, pdf, 1.0 - cdf), axis=-1)
