import itertools

import numpy as np
import pytest
from scipy import integrate, stats

from relyflow.models.lba import (
    _truncated_normal,
    accumulator_pdf_cdf,
    race_density,
    sample_race,
)

# Two-accumulator races covering typical, slow, biased and low-variability regimes,
# including negative mean rates (rates are truncated at zero).
CASES = {
    "typical": dict(v=[2.5, 0.5], s=[1.0, 1.0], b=[4.0, 4.0], A=2.0, t0=0.3),
    "negative_mismatch_rate": dict(v=[2.0, -1.5], s=[0.5, 1.0], b=[3.0, 3.5], A=1.0, t0=0.2),
    "both_rates_low": dict(v=[0.3, -0.5], s=[1.0, 1.0], b=[2.5, 2.0], A=2.0, t0=0.25),
    "narrow_rates": dict(v=[3.0, 1.0], s=[0.2, 1.0], b=[1.5, 1.5], A=0.5, t0=0.1),
}


def density(case, choice, t):
    c = CASES[case]
    return float(race_density(choice, t, c["v"], c["s"], c["b"], c["A"], c["t0"]))


def bin_probability(case, choice, lower, upper):
    return integrate.quad(lambda t: density(case, choice, t), lower, upper, limit=200)[0]


@pytest.mark.parametrize("case", CASES)
def test_density_integrates_to_one_over_both_responses(case):
    t0 = CASES[case]["t0"]
    total = sum(bin_probability(case, k, t0, np.inf) for k in (0, 1))
    assert total == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("case", CASES)
def test_sampler_matches_the_analytic_density(case):
    # Chi-square goodness of fit on (response, RT bin) cells. Bins are each
    # response's own RT quantiles; rare responses get a single bin.
    c = CASES[case]
    n = 100_000
    shape = (n, 2)
    choice, rt = sample_race(
        np.random.default_rng(0),
        np.broadcast_to(c["v"], shape),
        np.broadcast_to(c["s"], shape),
        np.broadcast_to(c["b"], shape),
        c["A"],
        c["t0"],
    )
    observed, expected = [], []
    for k in (0, 1):
        rts = rt[choice == k]
        inner = np.quantile(rts, [0.1, 0.3, 0.5, 0.7, 0.9]) if len(rts) >= 1000 else []
        edges = np.concatenate([[c["t0"]], inner, [np.inf]])
        observed.extend(np.histogram(rts, edges)[0])
        expected.extend(
            n * bin_probability(case, k, lo, hi) for lo, hi in itertools.pairwise(edges)
        )
    observed, expected = np.array(observed), np.array(expected)
    assert expected.min() > 5
    chi2 = ((observed - expected) ** 2 / expected).sum()
    p_value = stats.chi2.sf(chi2, df=len(observed) - 1)
    assert p_value > 1e-3, f"chi2={chi2:.1f}, p={p_value:.2g}"


def test_accumulator_pdf_is_the_derivative_of_its_cdf():
    t = np.linspace(0.05, 6, 400)
    for v, s, b, A in [(2.0, 1.0, 4.0, 2.0), (-1.0, 0.5, 3.0, 1.0), (4.0, 0.2, 1.0, 0.5)]:
        pdf, cdf = accumulator_pdf_cdf(t, v, s, b, A)
        h = 1e-6
        _, cdf_hi = accumulator_pdf_cdf(t + h, v, s, b, A)
        _, cdf_lo = accumulator_pdf_cdf(t - h, v, s, b, A)
        np.testing.assert_allclose(pdf, (cdf_hi - cdf_lo) / (2 * h), atol=1e-5)
        assert np.all(np.diff(cdf) >= -1e-12)


def test_density_is_zero_before_non_decision_time():
    c = CASES["typical"]
    assert density("typical", 0, c["t0"] - 0.01) == 0.0
    assert density("typical", 0, c["t0"]) == 0.0


@pytest.mark.parametrize(("mean", "sd"), [(1.0, 1.0), (-2.0, 1.0), (-6.0, 0.1), (5.0, 0.5)])
def test_truncated_normal_rates_are_positive_and_correctly_distributed(mean, sd):
    # (-6, 0.1) is 60 SDs into the tail, where naive inversion returns infinity.
    draws = _truncated_normal(np.random.default_rng(1), np.full(20_000, mean), np.full(20_000, sd))
    assert np.all(np.isfinite(draws)) and np.all(draws > 0)
    reference = stats.truncnorm(-mean / sd, np.inf, loc=mean, scale=sd)
    assert stats.kstest(draws, reference.cdf).pvalue > 1e-3


def test_tail_quadrature_matches_the_closed_form_where_both_are_accurate():
    from relyflow.models.lba import _tail_pdf_cdf

    t = np.linspace(0.2, 8.0, 50)
    v, s, b, A = np.full(50, -3.5), np.full(50, 1.0), np.full(50, 3.0), np.full(50, 1.5)
    pdf_closed, cdf_closed = accumulator_pdf_cdf(t, v, s, b, A)  # v/s = -3.5: closed form
    pdf_quad, cdf_quad = _tail_pdf_cdf(t, v, s, b, A)
    # Near t = 0 the closed form rounds the CDF to 0; quadrature keeps ~1e-12.
    np.testing.assert_allclose(pdf_quad, pdf_closed, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(cdf_quad, cdf_closed, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize(("v", "s"), [(-4.0, 0.1), (-1.0, 0.01), (-3.0, 0.5)])
def test_density_stays_finite_and_normalized_deep_in_the_truncation_tail(v, s):
    # v/s from -6 to -100: the closed form would divide rounding noise by ~0.
    # Rates sit near 0, so finishing times can be ~1e4 s; integrate over log t.
    def pdf_log_t(y):
        t = np.exp(y)
        return float(accumulator_pdf_cdf(t, v, s, 2.0, 1.0)[0]) * t

    t = np.geomspace(0.01, 1e12, 300)
    values, cdf = accumulator_pdf_cdf(t, v, s, 2.0, 1.0)
    assert np.all(np.isfinite(values)) and np.all(values >= 0)
    assert np.all(np.diff(cdf) >= -1e-12) and cdf[-1] == pytest.approx(1.0, abs=1e-3)
    total = integrate.quad(pdf_log_t, np.log(0.01), np.log(1e12), limit=500)[0]
    assert total == pytest.approx(1.0, abs=1e-3)
