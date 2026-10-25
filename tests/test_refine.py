import numpy as np
import pytest

from relyflow import TrialData, make_design
from relyflow.inference import DesignPrior, TrainingConfig, load, train
from relyflow.inference.features import design_features, learning_statistics, network_inputs
from relyflow.inference.refine import importance_resample, network_log_density


@pytest.fixture(scope="module", params=["reliance", "learning"])
def tiny(request, tmp_path_factory):
    out = tmp_path_factory.mktemp(request.param)
    config = TrainingConfig(
        model=request.param, epochs=1, steps_per_epoch=2, batch_size=8, summary_dim=4,
        flow_depth=1, validation_sims=8, design=DesignPrior(n_trials=(60, 90), n_trials_step=30),
    )  # fmt: skip
    train(config, out, verbose=0)
    amortizer = load(out)
    model = amortizer.model
    rng = np.random.default_rng(0)
    stimulus, advice = make_design([(30, None), (50, 0.8)], rng)
    choice, rt = model.simulate(model.sample_prior(1, rng), stimulus, advice, rng)
    return amortizer, TrialData(stimulus[0], advice[0], choice[0], rt[0])


def test_fast_network_density_matches_bayesflow_log_prob(tiny):
    amortizer, data = tiny
    draws = amortizer.sample(data, num_samples=16, seed=1)
    config = amortizer.config
    s, a, c, r = (np.atleast_2d(x) for x in (data.stimulus, data.advice, data.choice, data.rt))
    conditions = {
        "trials": network_inputs(s, a, c, r, config.history_features),
        "design": design_features(a),
    }
    if config.learning_statistics:
        conditions["statistics"] = learning_statistics(s, a, c, r)
    batch = {k: np.repeat(v, 16, axis=0) for k, v in conditions.items()}
    batch |= {k: v[:, None] for k, v in draws.items()}
    expected = np.asarray(amortizer.approximator.log_prob(batch)).reshape(-1)
    np.testing.assert_allclose(network_log_density(amortizer, data, draws), expected, rtol=1e-4)


def test_importance_resampling_returns_draws_and_diagnostics(tiny):
    amortizer, data = tiny
    refined = importance_resample(amortizer, data, num_samples=50, num_proposals=200, seed=2)
    assert refined.method == "importance"
    assert set(refined.draws) == set(amortizer.model.param_names)
    assert all(v.shape == (50,) for v in refined.draws.values())
    assert 1 <= refined.diagnostics["ess"] <= 200


def test_log_evidence_is_finite_with_a_positive_ess(tiny):
    from relyflow.inference.refine import log_evidence

    amortizer, data = tiny
    value, ess = log_evidence(amortizer, data, num_proposals=200, seed=3)
    assert np.isfinite(value) and value < 0
    assert 1 <= ess <= 200
