import numpy as np
import pytest

from relyflow import NO_ADVICE
from relyflow.models import MODELS, Advisory, ReliabilityLearning, Reliance, get_model
from relyflow.models.accumulator import thresholds
from relyflow.models.learning import delta_rule_beliefs, perceived_validity

# Two unaided trials, two where the aid is right, two where it is wrong.
STIM = np.array([0, 1, 0, 1, 0, 1])
ADV = np.array([NO_ADVICE, NO_ADVICE, 0, 1, 1, 0])


@pytest.fixture(params=sorted(MODELS))
def model(request):
    return get_model(request.param)


def one_set(model, **values):
    """A single parameter set at the prior medians, with some values overridden."""
    params = {p.name: np.array([p.prior.median()]) for p in model.parameters}
    params.update({name: np.array([value]) for name, value in values.items()})
    return params


def test_prior_draws_have_the_right_shape_and_support(model):
    draws = model.sample_prior(500, np.random.default_rng(0))
    assert tuple(draws) == model.param_names
    assert len(set(model.param_names)) == len(model.param_names)
    for p in model.parameters:
        lower, upper = p.support
        assert draws[p.name].shape == (500,)
        assert np.all((draws[p.name] >= lower) & (draws[p.name] <= upper))


def test_prior_draws_are_reproducible(model):
    a = model.sample_prior(5, np.random.default_rng(1))
    b = model.sample_prior(5, np.random.default_rng(1))
    for name in a:
        np.testing.assert_array_equal(a[name], b[name])


def test_only_reliance_models_have_excitation():
    assert "excitation" not in Advisory().param_names
    assert {"inhibition", "excitation"} <= set(Reliance().param_names)
    assert {"r0", "alpha"} <= set(ReliabilityLearning().param_names)


def test_unknown_model_name_lists_the_choices():
    with pytest.raises(ValueError, match="advisory"):
        get_model("nope")


def test_rates_follow_the_published_definitions_of_excitation_and_inhibition():
    # Strickland et al. (2023): excitation raises the accumulator agreeing with the
    # advice, inhibition lowers the one disagreeing, relative to matched unaided trials.
    params = one_set(
        Reliance(),
        v_match_0=2.0, v_match_1=3.0, v_mismatch_0=0.5, v_mismatch_1=0.2,
        excitation=0.4, inhibition=0.7,
    )  # fmt: skip
    rates = Reliance().rate_means(params, STIM, ADV)[0]
    expected = [
        [2.0, 0.5],  # unaided, stimulus 0: accumulator 0 is correct
        [0.2, 3.0],  # unaided, stimulus 1
        [2.0 + 0.4, 0.5 - 0.7],  # aid right: correct excited, error inhibited
        [0.2 - 0.7, 3.0 + 0.4],
        [2.0 - 0.7, 0.5 + 0.4],  # aid wrong: correct inhibited, error excited
        [0.2 + 0.4, 3.0 - 0.7],
    ]
    np.testing.assert_allclose(rates, expected)


def test_advisory_is_reliance_with_excitation_fixed_at_zero():
    reliance_params = one_set(Reliance(), excitation=0.0)
    advisory_params = {k: v for k, v in reliance_params.items() if k != "excitation"}
    np.testing.assert_allclose(
        Advisory().rate_means(advisory_params, STIM, ADV),
        Reliance().rate_means(reliance_params, STIM, ADV),
    )


def test_aided_thresholds_scale_the_gap_but_not_the_start_point_range():
    params = one_set(Reliance(), A=1.0, B_0=2.0, B_1=3.0, B_aided_scale=0.5)
    b = thresholds(params, np.atleast_2d(ADV))[0]
    np.testing.assert_allclose(b[0], [3.0, 4.0])
    np.testing.assert_allclose(b[2], [2.0, 2.5])


def test_rates_are_vectorized_over_parameter_sets(model):
    rng = np.random.default_rng(2)
    draws = model.sample_prior(7, rng)
    stim = rng.integers(0, 2, (7, 12))
    adv = rng.integers(-1, 2, (7, 12))
    rates = model.rate_means(draws, stim, adv)
    assert rates.shape == (7, 12, 2)
    single = {name: values[3:4] for name, values in draws.items()}
    np.testing.assert_allclose(rates[3], model.rate_means(single, stim[3], adv[3])[0])


def test_delta_rule_matches_its_recency_weighted_average_form():
    # Strickland et al. (2024), Eq. 3: r_t = (1-a)^t r0 + sum_{i<t} a (1-a)^i AC_{t-i}
    rng = np.random.default_rng(3)
    stim = rng.integers(0, 2, 30)
    adv = np.where(rng.random(30) < 0.8, stim, 1 - stim)
    r0, a = 0.6, 0.15
    beliefs = delta_rule_beliefs(np.array([r0]), np.array([a]), stim, adv)[0]
    aid_correct = (adv == stim).astype(float)
    for t in range(30):
        expected = (1 - a) ** t * r0 + sum(
            a * (1 - a) ** i * aid_correct[t - 1 - i] for i in range(t)
        )
        assert beliefs[t] == pytest.approx(expected)


def test_unaided_trials_leave_perceived_reliability_unchanged():
    stim = np.array([1, 1, 1, 1])
    adv = np.array([0, NO_ADVICE, NO_ADVICE, 1])
    beliefs = delta_rule_beliefs(np.array([0.7]), np.array([0.5]), stim, adv)[0]
    np.testing.assert_allclose(beliefs, [0.7, 0.35, 0.35, 0.35])


def test_perceived_validity_is_zero_at_chance_and_one_for_a_perfect_aid():
    np.testing.assert_allclose(perceived_validity([0.2, 0.5, 0.75, 1.0]), [0.0, 0.0, 0.5, 1.0])


def test_learning_model_without_learning_is_a_rescaled_reliance_model():
    learner = one_set(ReliabilityLearning(), r0=0.9, alpha=0.0, excitation=0.5, inhibition=1.0)
    static = dict(learner, excitation=np.array([0.5 * 0.8]), inhibition=np.array([1.0 * 0.8]))
    np.testing.assert_allclose(
        ReliabilityLearning().rate_means(learner, STIM, ADV),
        Reliance().rate_means(static, STIM, ADV),
    )


def test_a_run_of_correct_advice_strengthens_its_effects():
    n = 50
    stim = np.ones(n, dtype=int)
    params = one_set(ReliabilityLearning(), r0=0.6, alpha=0.1)
    rates = ReliabilityLearning().rate_means(params, stim, stim)[0]
    assert np.all(np.diff(rates[:, 1]) > 0)  # agreeing accumulator: more excitation
    assert np.all(np.diff(rates[:, 0]) < 0)  # disagreeing accumulator: more inhibition
