import warnings

import numpy as np
import pytest

from relyflow import NO_ADVICE
from relyflow.diagnostics import (
    PPC_STATISTICS,
    ppc_check,
    ppc_statistics,
    recovery_table,
    sbc_table,
)
from relyflow.diagnostics.benchmark import compare
from relyflow.models import get_model


@pytest.fixture(scope="module")
def prior_as_posterior():
    """Posteriors equal to the prior (as with uninformative data) are calibrated."""
    model = get_model("reliance")
    rng = np.random.default_rng(0)
    n, draws_per = 400, 200
    truth = model.sample_prior(n, rng)
    draws = {k: v.reshape(n, draws_per) for k, v in model.sample_prior(n * draws_per, rng).items()}
    return model, truth, draws


def test_calibrated_posteriors_pass_sbc_and_have_nominal_coverage(prior_as_posterior):
    model, truth, draws = prior_as_posterior
    sbc = sbc_table(model, truth, draws)
    assert sbc["passed"].all()
    recovery = recovery_table(model, truth, draws)
    assert np.allclose(recovery["coverage_90"], 0.9, atol=0.05)
    assert np.allclose(recovery["contraction"], 0.0, atol=0.1)


# Grossly miscalibrated ranks give gamma = 0, so log-gamma = -inf: a (correct) failure.
@pytest.mark.filterwarnings("ignore:divide by zero encountered in log:RuntimeWarning")
def test_overconfident_posteriors_fail_sbc(prior_as_posterior):
    model, truth, draws = prior_as_posterior
    rng = np.random.default_rng(1)
    # Narrow posteriors centered on a noisy estimate: far too confident.
    narrow = {}
    for k, v in draws.items():
        center = truth[k][:, None] + 0.3 * v.std() * rng.normal(size=(len(v), 1))
        narrow[k] = center + 0.015 * v.std() * rng.normal(size=v.shape)
    assert not sbc_table(model, truth, narrow)["passed"].any()
    assert (recovery_table(model, truth, narrow)["coverage_90"] < 0.5).all()


def test_ppc_statistics_split_by_trial_type():
    stimulus = np.array([1, 0, 1, 0, 1, 0])
    advice = np.array([NO_ADVICE, NO_ADVICE, 1, 0, 0, 1])
    choice = np.array([1, 1, 1, 0, 0, 0])
    rt = np.array([1.0, 2.0, 0.5, 0.7, 3.0, 1.5])
    stats = dict(zip(PPC_STATISTICS, ppc_statistics(stimulus, advice, choice, rt), strict=True))
    assert stats["accuracy_unaided"] == 0.5
    assert stats["accuracy_aid_right"] == 1.0
    assert stats["accuracy_aid_wrong"] == 0.5
    assert stats["correct_rt_q50_aid_right"] == pytest.approx(0.6)
    assert stats["correct_rt_q50_aid_wrong"] == pytest.approx(1.5)


def test_ppc_check_flags_observations_outside_the_predictive_interval():
    replicated = np.random.default_rng(2).normal(size=(500, 2))
    result = ppc_check(np.array([0.0, 5.0]), replicated)
    assert result["outside_95"].tolist() == [0.0, 1.0]
    assert result["p_value"][0] > 0.5 and result["p_value"][1] < 0.01


def test_compare_measures_agreement_in_reference_sds():
    rng = np.random.default_rng(3)
    reference = {"x": rng.normal(0, 2, 20_000)}
    same = compare({"x": rng.normal(0, 2, 20_000)}, reference).iloc[0]
    assert abs(same["mean_error"]) < 0.05 and abs(same["sd_ratio"] - 1) < 0.05
    shifted = compare({"x": rng.normal(1, 1, 20_000)}, reference).iloc[0]
    assert shifted["mean_error"] == pytest.approx(0.5, abs=0.05)
    assert shifted["sd_ratio"] == pytest.approx(0.5, abs=0.05)


def test_nan_quantiles_match_numpy():
    from relyflow.diagnostics.ppc import nan_quantiles

    rng = np.random.default_rng(11)
    x = rng.lognormal(size=(40, 3, 25))
    x[rng.random(x.shape) < 0.4] = np.nan
    x[0, 0] = np.nan  # a row with no values
    x[1, 1, 1:] = np.nan  # a row with one value
    q = (0.1, 0.5, 0.9)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.nanquantile(x, q, axis=-1)
    np.testing.assert_allclose(nan_quantiles(x, q), expected, equal_nan=True, rtol=1e-12)
