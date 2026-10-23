import json
import shutil
import warnings

import numpy as np
import pytest

from relyflow import NO_ADVICE, TrialData, make_design
from relyflow.inference import DesignPrior, TrainingConfig, load, train
from relyflow.inference.features import (
    DESIGN_FEATURES,
    LOG_RT_BOUNDS,
    TRIAL_FEATURES,
    design_features,
    trial_features,
)
from relyflow.models import get_model


def test_trial_features_encode_each_trial():
    stimulus = np.array([1, 0, 1])
    advice = np.array([NO_ADVICE, 0, 0])
    choice = np.array([1, 0, 0])
    rt = np.array([0.5, 1.0, 1000.0])
    features = trial_features(stimulus, advice, choice, rt)
    assert features.shape == (3, len(TRIAL_FEATURES)) and features.dtype == np.float32
    columns = dict(zip(TRIAL_FEATURES, features.T, strict=True))
    np.testing.assert_array_equal(columns["correct"], [1, 1, 0])
    np.testing.assert_array_equal(columns["aided"], [0, 1, 1])
    np.testing.assert_array_equal(columns["aid_correct"], [0, 1, 0])
    np.testing.assert_array_equal(columns["followed"], [0, 1, 1])
    np.testing.assert_allclose(columns["log_rt"], [np.log(0.5), 0.0, LOG_RT_BOUNDS[1]], rtol=1e-6)


def test_design_features_are_log_trial_counts():
    advice = np.array([[NO_ADVICE] * 3 + [1] * 7])
    features = design_features(advice)
    assert features.shape == (1, len(DESIGN_FEATURES))
    np.testing.assert_allclose(features, [[np.log(3), np.log(7)]], rtol=1e-6)


def test_design_prior_respects_its_ranges():
    prior = DesignPrior(
        n_trials=(100, 300), n_trials_step=50, reliability=(0.7, 0.9), max_aided_blocks=3
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        stimulus, advice = prior.sample(8, rng)
        n_trials = stimulus.shape[1]
        aided = advice != NO_ADVICE
        assert stimulus.shape == (8, n_trials) and n_trials in (100, 150, 200, 250, 300)
        assert aided.any(axis=1).all() and (~aided).any(axis=1).all()
        share = (~aided).mean(axis=1)
        assert np.all((share > 0.15) & (share < 0.55))
        reliability = (advice == stimulus)[aided].mean()
        assert 0.6 < reliability < 1.0


@pytest.fixture(scope="module")
def tiny_amortizer(tmp_path_factory):
    """A barely trained network: enough to exercise the pipeline, not to be accurate."""
    out = tmp_path_factory.mktemp("amortizer")
    config = TrainingConfig(
        model="reliance",
        epochs=1,
        steps_per_epoch=2,
        batch_size=8,
        summary_dim=4,
        flow_depth=1,
        validation_sims=8,
        design=DesignPrior(n_trials=(60, 90), n_trials_step=30),
    )
    train(config, out, verbose=0)
    return out


def simulated_operator(n_unaided, n_aided, seed=0):
    model = get_model("reliance")
    rng = np.random.default_rng(seed)
    stimulus, advice = make_design([(n_unaided, None), (n_aided, 0.8)], rng)
    choice, rt = model.simulate(model.sample_prior(1, rng), stimulus, advice, rng)
    return TrialData(stimulus[0], advice[0], choice[0], rt[0])


def test_a_trained_network_round_trips_through_disk(tiny_amortizer):
    saved = json.loads((tiny_amortizer / "config.json").read_text())
    assert saved["config"]["model"] == "reliance"
    assert saved["metadata"]["keras_backend"]

    amortizer = load(tiny_amortizer)
    draws = amortizer.sample(simulated_operator(30, 50), num_samples=100, seed=1)
    assert list(draws) == list(amortizer.model.param_names)
    for parameter in amortizer.model.parameters:
        lower, upper = parameter.support
        values = draws[parameter.name]
        assert values.shape == (100,)
        assert np.all((values >= lower) & (values <= upper))


def test_out_of_range_designs_trigger_warnings(tiny_amortizer):
    amortizer = load(tiny_amortizer)
    data = simulated_operator(0, 80)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        amortizer.sample(data, num_samples=10)
    assert any("No unaided trials" in str(w.message) for w in caught)
    assert amortizer.design_warnings(np.full((1, 500), 1))[0].startswith("500 trials")


def test_changed_priors_are_detected_on_load(tiny_amortizer, tmp_path):
    stale = tmp_path / "stale"
    shutil.copytree(tiny_amortizer, stale)
    saved = json.loads((stale / "config.json").read_text())
    saved["metadata"]["model_signature"][0][2] = [0.0, 1.0]
    (stale / "config.json").write_text(json.dumps(saved))
    with pytest.warns(UserWarning, match="priors have changed"):
        load(stale)


def test_config_chooses_summary_networks_per_model():
    assert TrainingConfig(model="learning").summary_network == "deep_set"
    assert TrainingConfig(model="advisory").summary_network == "set_transformer"
    restored = TrainingConfig.from_dict(
        json.loads(json.dumps(TrainingConfig("reliance").to_dict()))
    )
    assert restored.design == DesignPrior()


def test_history_features_match_the_learning_models_beliefs():
    # Each track-record column is the delta-rule belief with r0 = 0.5 at that rate.
    from relyflow.inference.features import HISTORY_FEATURES, HISTORY_RATES, history_features
    from relyflow.models.learning import delta_rule_beliefs

    stimulus, advice = make_design([(20, None), (60, 0.7), (20, None)], np.random.default_rng(3))
    history = history_features(stimulus, advice)[0]
    assert history.shape == (100, len(HISTORY_FEATURES))
    for column, rate in enumerate(HISTORY_RATES, start=2):
        beliefs = delta_rule_beliefs(np.array([0.5]), np.array([rate]), stimulus, advice)[0]
        np.testing.assert_allclose(history[:, column], beliefs, rtol=1e-6)
    np.testing.assert_array_equal(history[:21, 0], 0)  # no aid experience yet
    np.testing.assert_allclose(history[-1, 1], np.log1p(60), rtol=1e-6)


def test_learning_inputs_are_on_by_default_only_for_the_learning_model():
    learning, reliance = TrainingConfig(model="learning"), TrainingConfig(model="reliance")
    assert learning.history_features and learning.learning_statistics
    assert not (reliance.history_features or reliance.learning_statistics)
    # Configs saved before the options existed load as "off".
    old = TrainingConfig.from_dict({"model": "advisory"})
    assert not (old.history_features or old.learning_statistics)


def test_learning_statistics_detect_following_that_tracks_the_aid():
    from relyflow.inference.features import LEARNING_STATISTICS, learning_statistics

    rng = np.random.default_rng(4)
    stimulus, advice = make_design([(50, None), (400, 0.75)], rng, n_sims=2)
    # Operator 0 follows the advice only right after it was correct; operator 1 at random.
    previous_right = np.zeros_like(advice, dtype=bool)
    previous_right[:, 51:] = (advice == stimulus)[:, 50:-1]
    follow = np.stack([previous_right[0], rng.random(advice.shape[1]) < 0.7])
    choice = np.where(follow, advice, 1 - advice)
    choice[:, :50] = stimulus[:, :50]
    stats = learning_statistics(stimulus, advice, choice, np.full(advice.shape, 1.0))
    assert stats.shape == (2, len(LEARNING_STATISTICS)) and np.all(np.isfinite(stats))
    drop = LEARNING_STATISTICS.index("follow_after_failure_minus_success")
    assert stats[0, drop] < -0.9 and abs(stats[1, drop]) < 0.15


def test_pretrained_networks_download_once_and_are_verified(tiny_amortizer, tmp_path, monkeypatch):
    from relyflow.inference import pretrained

    archive = tmp_path / "relyflow-reliance.zip"
    digest = pretrained.package(tiny_amortizer, archive)
    downloads = []

    def fake_urlopen(url):
        downloads.append(url)
        return archive.open("rb")

    monkeypatch.delenv("RELYFLOW_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("RELYFLOW_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(pretrained.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setitem(pretrained.REGISTRY, "reliance", ("weights-test", archive.name, digest))

    amortizer = load("reliance")
    load("reliance")
    assert amortizer.config.model == "reliance"
    assert len(downloads) == 1 and downloads[0].endswith("/weights-test/relyflow-reliance.zip")

    monkeypatch.setitem(pretrained.REGISTRY, "advisory", ("weights-test", archive.name, "0" * 64))
    with pytest.raises(RuntimeError, match="integrity check"):
        pretrained.fetch("advisory")


def test_local_weights_directory_overrides_downloads(tiny_amortizer, monkeypatch):
    from relyflow.inference import pretrained

    monkeypatch.setenv("RELYFLOW_WEIGHTS_DIR", str(tiny_amortizer.parent))
    assert pretrained.fetch(tiny_amortizer.name) == tiny_amortizer
    with pytest.raises(FileNotFoundError):
        pretrained.fetch("missing-model")


def test_a_learning_network_with_history_and_statistics_round_trips(tmp_path):
    config = TrainingConfig(
        model="learning",
        epochs=1,
        steps_per_epoch=2,
        batch_size=8,
        summary_dim=4,
        flow_depth=1,
        validation_sims=8,
        design=DesignPrior(n_trials=(60, 90), n_trials_step=30),
    )
    train(config, tmp_path, verbose=0)
    amortizer = load(tmp_path)
    assert amortizer.config.history_features and amortizer.config.learning_statistics
    model = amortizer.model
    rng = np.random.default_rng(5)
    stimulus, advice = make_design([(30, None), (50, 0.8)], rng)
    choice, rt = model.simulate(model.sample_prior(1, rng), stimulus, advice, rng)
    draws = amortizer.sample(TrialData(stimulus[0], advice[0], choice[0], rt[0]), num_samples=50)
    assert {"r0", "alpha"} <= set(draws) and draws["alpha"].shape == (50,)
