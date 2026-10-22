"""Behavioral validation of the model simulators (roadmap milestone 2).

Pattern tests simulate each operator twice on the same stimuli with the same
random numbers ("common random numbers"): once with a 75%-reliable aid and once
unaided. Differences between an aided trial and its unaided twin then come only
from the advice. Expected patterns are documented in docs/models/static.md.
"""

import warnings

import numpy as np
import pytest

from relyflow import NO_ADVICE, TrialData, make_design, summary_stats
from relyflow.models import MODELS, get_model

N_DRAWS = 300
N_TRIALS = 4000


def prior_draws(model, n=N_DRAWS, seed=0, **fixed):
    params = model.sample_prior(n, np.random.default_rng(seed))
    params.update({name: np.full(n, value) for name, value in fixed.items()})
    return params


def at_medians(model, n, **values):
    params = {p.name: np.full(n, p.prior.median()) for p in model.parameters}
    params.update({name: np.full(n, value) for name, value in values.items()})
    return params


def simulate_twins(model, params, n_trials=N_TRIALS, seed=1):
    """Statistics for aid-right and aid-wrong trials, aided and for their unaided twins."""
    n = len(next(iter(params.values())))
    stimulus, advice = make_design([(n_trials, 0.75)], np.random.default_rng(seed), n_sims=n)
    runs = {
        "aided": model.simulate(params, stimulus, advice, np.random.default_rng(seed + 1)),
        "unaided": model.simulate(
            params, stimulus, np.full_like(advice, NO_ADVICE), np.random.default_rng(seed + 1)
        ),
    }
    trial_types = {"right": advice == stimulus, "wrong": advice != stimulus}
    cells = {}
    for run, (choice, rt) in runs.items():
        correct = choice == stimulus
        for trial_type, mask in trial_types.items():
            cells[run, trial_type] = {
                "accuracy": _masked_mean(correct, mask),
                "followed": _masked_mean(choice == advice, mask),
                # Medians, because LBA RTs have power-law tails and infinite variance.
                # Per stimulus type, as in the source papers: pooling stimuli of
                # different difficulty can reverse RT differences (Simpson's paradox).
                "correct_rt": np.stack(
                    [_masked_median(rt, mask & correct & (stimulus == s)) for s in (0, 1)], axis=1
                ),
                "n_correct": np.stack(
                    [(mask & correct & (stimulus == s)).sum(axis=1) for s in (0, 1)], axis=1
                ),
            }
        cells[run, "rt"] = rt
    return cells


def _masked_mean(values, mask):
    with np.errstate(invalid="ignore"):
        return np.where(mask, values, 0).sum(axis=1) / mask.sum(axis=1)


def _masked_median(values, mask):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows give NaN
        return np.nanmedian(np.where(mask, values, np.nan), axis=1)


def wrong_advice_slowing(cells):
    """Median correct-RT slowing on aid-wrong trials vs. their unaided twins, per stimulus.

    Cells with fewer than 30 aided correct responses are NaN.
    """
    aided, unaided = cells["aided", "wrong"], cells["unaided", "wrong"]
    slowing = aided["correct_rt"] - unaided["correct_rt"]
    return np.where(aided["n_correct"] >= 30, slowing, np.nan)


# --- Effects of advice -------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(MODELS))
def test_accuracy_rises_when_the_aid_is_right_and_falls_when_it_is_wrong(name):
    model = get_model(name)
    cells = simulate_twins(model, prior_draws(model, B_aided_scale=1.0))
    benefit = cells["aided", "right"]["accuracy"] - cells["unaided", "right"]["accuracy"]
    cost = cells["unaided", "wrong"]["accuracy"] - cells["aided", "wrong"]["accuracy"]
    assert np.all(benefit >= 0) and np.all(cost >= 0)
    assert np.mean(benefit) > 0.02 and np.mean(cost) > 0.05


def test_inhibition_slows_correct_responses_when_the_aid_is_wrong():
    # The expected slowing is positive for every prior draw, but tiny when inhibition
    # is near zero, so the simulated check uses draws with a detectable effect.
    advisory = get_model("advisory")
    params = prior_draws(advisory, B_aided_scale=1.0)
    slowing = wrong_advice_slowing(simulate_twins(advisory, params))
    detectable = slowing[params["inhibition"] >= 0.3]
    assert np.all(detectable[~np.isnan(detectable)] > 0)
    assert np.nanmedian(slowing) > 0.1

    # In the reliance model the slowing holds when inhibition dominates excitation.
    # (I >= 2E is an empirical rule over the prior, not a theorem.)
    reliance = get_model("reliance")
    params = prior_draws(reliance, n=1000, B_aided_scale=1.0)
    dominant = (params["inhibition"] >= 2 * params["excitation"]) & (params["inhibition"] >= 0.3)
    slowing = wrong_advice_slowing(simulate_twins(reliance, params))[dominant]
    assert dominant.sum() > 100
    assert np.all(slowing[~np.isnan(slowing)] > 0)


def test_strong_excitation_without_inhibition_speeds_correct_responses_on_wrong_advice():
    # A selection effect: only fast correct responses beat an excited error accumulator.
    # So the RT cost of wrong advice is a signature of inhibition, not of aid use in general.
    reliance = get_model("reliance")
    params = at_medians(reliance, 50, excitation=2.0, inhibition=0.0, B_aided_scale=1.0)
    assert np.all(wrong_advice_slowing(simulate_twins(reliance, params)) < 0)


def test_misuse_rises_with_excitation():
    reliance = get_model("reliance")
    base = prior_draws(reliance, n=100)
    misuse = [
        simulate_twins(reliance, dict(base, excitation=np.full(100, e)))["aided", "wrong"][
            "followed"
        ]
        for e in (0.0, 0.5, 1.0, 2.0)
    ]
    assert np.all(np.diff(misuse, axis=0) >= 0)
    assert np.mean(misuse[-1] - misuse[0]) > 0.1


def test_advice_alone_cannot_trigger_a_decision_without_excitation():
    # With inhibition only, every aided trial is at least as slow as its unaided twin.
    advisory = get_model("advisory")
    cells = simulate_twins(advisory, prior_draws(advisory, n=100, B_aided_scale=1.0))
    assert np.all(cells["aided", "rt"] >= cells["unaided", "rt"])
    # With excitation, some aided decisions come faster than they would unaided.
    reliance = get_model("reliance")
    cells = simulate_twins(reliance, prior_draws(reliance, n=100, B_aided_scale=1.0))
    assert np.any(cells["aided", "rt"] < cells["unaided", "rt"])


def test_lower_aided_thresholds_speed_up_aided_trials():
    reliance = get_model("reliance")
    params = at_medians(reliance, 50, excitation=0.0, inhibition=0.0, B_aided_scale=0.7)
    cells = simulate_twins(reliance, params)
    assert np.all(cells["aided", "rt"] <= cells["unaided", "rt"])
    assert np.mean(cells["aided", "rt"]) < np.mean(cells["unaided", "rt"])


# --- Learning model: tracking the aid's reliability ---------------------------------


def learning_vs_static(blocks, seed):
    n = 2000
    rng = np.random.default_rng(seed)
    stimulus, advice = make_design(blocks, rng, n_sims=n)
    results = {}
    for name, extra in (("learning", dict(r0=0.8, alpha=0.3)), ("reliance", {})):
        model = get_model(name)
        params = at_medians(model, n, excitation=1.5, inhibition=1.5, **extra)
        choice, rt = model.simulate(params, stimulus, advice, np.random.default_rng(seed + 1))
        results[name] = (stimulus, advice, choice, rt)
    return results


def test_operators_follow_the_aid_less_right_after_it_fails():
    for name, (stimulus, advice, choice, _) in learning_vs_static([(400, 0.8)], seed=3).items():
        followed = (choice == advice)[:, 1:]
        after_failure = (advice != stimulus)[:, :-1]
        drop = followed[~after_failure].mean() - followed[after_failure].mean()
        if name == "learning":
            assert drop > 0.05
        else:  # the static model has no memory of the last trial
            assert abs(drop) < 0.01


def test_misuse_falls_after_the_aid_becomes_less_reliable():
    results = learning_vs_static([(300, 0.95), (300, 0.6)], seed=4)
    for name, (stimulus, advice, choice, rt) in results.items():
        early = summary_stats(stimulus[:, :300], advice[:, :300], choice[:, :300], rt[:, :300])
        late = summary_stats(stimulus[:, 450:], advice[:, 450:], choice[:, 450:], rt[:, 450:])
        change = np.nanmean(late["misuse_rate"]) - np.nanmean(early["misuse_rate"])
        if name == "learning":
            assert change < -0.2
        else:
            assert abs(change) < 0.03


# --- Priors and likelihood -----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(MODELS))
def test_prior_predictive_data_are_plausible_and_cover_the_source_study(name):
    # Standard design: 200 unaided trials, then 400 with an 85%-reliable aid.
    model = get_model(name)
    rng = np.random.default_rng(10)
    n = 2000
    stimulus, advice = make_design([(200, None), (400, 0.85)], rng, n_sims=n)
    choice, rt = model.simulate(model.sample_prior(n, rng), stimulus, advice, rng)
    stats = summary_stats(stimulus, advice, choice, rt)
    median_rt = np.median(rt, axis=1)

    assert np.all(np.isfinite(rt))
    assert np.mean((median_rt > 0.3) & (median_rt < 5.0)) > 0.99
    assert np.mean(stats["accuracy_unaided"] < 0.5) < 0.05

    # Strickland et al. (2023) data, participants 1-24: unaided accuracy 0.85,
    # median RT about 1.4 s, misuse 0.22 (75% aid) and 0.50 (95% aid).
    def covers(values, observed):
        lower, upper = np.nanquantile(values, [0.05, 0.95])
        return lower < observed < upper

    assert covers(stats["accuracy_unaided"], 0.85)
    assert covers(median_rt, 1.4)
    assert covers(stats["misuse_rate"], 0.22) and covers(stats["misuse_rate"], 0.50)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_log_likelihood_is_vectorized_and_favors_the_generating_parameters(name):
    model = get_model(name)
    rng = np.random.default_rng(11)
    truth = model.sample_prior(1, rng)
    stimulus, advice = make_design([(200, None), (400, 0.85)], rng)
    choice, rt = model.simulate(truth, stimulus, advice, rng)

    candidates = model.sample_prior(50, rng)
    loglik = model.log_likelihood(candidates, stimulus, advice, choice, rt)
    assert loglik.shape == (50,) and np.all(np.isfinite(loglik))
    one = {name: values[7:8] for name, values in candidates.items()}
    assert model.log_likelihood(one, stimulus, advice, choice, rt)[0] == pytest.approx(loglik[7])
    assert model.log_likelihood(truth, stimulus, advice, choice, rt)[0] > np.median(loglik)


def test_vectorized_summaries_match_single_operator_summaries():
    rng = np.random.default_rng(12)
    model = get_model("reliance")
    stimulus, advice = make_design([(50, None), (100, 0.7)], rng, n_sims=4)
    choice, rt = model.simulate(model.sample_prior(4, rng), stimulus, advice, rng)
    batch = summary_stats(stimulus, advice, choice, rt)
    for i in range(4):
        single = TrialData(stimulus[i], advice[i], choice[i], rt[i]).summary()
        for stat, value in single.items():
            np.testing.assert_allclose(batch[stat][i], value)
