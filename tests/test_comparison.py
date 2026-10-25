import numpy as np
import pytest

from relyflow import NO_ADVICE, TrialData, make_design
from relyflow.comparison import (
    ComparisonConfig,
    ModelComparison,
    ood_check,
    posterior_predictive_check,
    train,
)
from relyflow.comparison.ood import mahalanobis_ranks, rank_p_value
from relyflow.diagnostics.misspecification import SCENARIOS, always_follow
from relyflow.inference import DesignPrior, TrainingConfig, load
from relyflow.inference import train as train_amortizer


@pytest.fixture(scope="module")
def comparison_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("comparison")
    config = ComparisonConfig(
        epochs=1, steps_per_epoch=2, batch_size=12, summary_dim=4, validation_sims=12,
        design=DesignPrior(n_trials=(60, 90), n_trials_step=30),
    )  # fmt: skip
    train(config, out, verbose=0)
    return out


@pytest.fixture(scope="module")
def tiny_comparison(comparison_dir):
    return ModelComparison.load(comparison_dir)


@pytest.fixture(scope="module")
def tiny_amortizer(tmp_path_factory):
    out = tmp_path_factory.mktemp("reliance")
    config = TrainingConfig(
        model="reliance", epochs=1, steps_per_epoch=2, batch_size=8, summary_dim=4,
        flow_depth=1, validation_sims=8, design=DesignPrior(n_trials=(60, 90), n_trials_step=30),
    )  # fmt: skip
    train_amortizer(config, out, verbose=0)
    return load(out)


def operator(seed=0, n_unaided=30, n_aided=60):
    from relyflow.models import get_model

    model = get_model("reliance")
    rng = np.random.default_rng(seed)
    stimulus, advice = make_design([(n_unaided, None), (n_aided, 0.8)], rng)
    choice, rt = model.simulate(model.sample_prior(1, rng), stimulus, advice, rng)
    return TrialData(stimulus[0], advice[0], choice[0], rt[0])


def test_model_probabilities_cover_all_models_and_sum_to_one(tiny_comparison):
    probs = tiny_comparison.probabilities(operator())
    assert list(probs) == ["advisory", "reliance", "learning"]
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-5)
    assert all(0 <= p <= 1 for p in probs.values())


def test_pretrained_comparison_network_downloads_and_loads(comparison_dir, tmp_path, monkeypatch):
    from relyflow.inference import pretrained

    archive = tmp_path / "relyflow-comparison.zip"
    digest = pretrained.package(comparison_dir, archive)
    monkeypatch.delenv("RELYFLOW_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("RELYFLOW_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(pretrained.urllib.request, "urlopen", lambda url: archive.open("rb"))
    monkeypatch.setitem(pretrained.REGISTRY, "comparison", ("weights-test", archive.name, digest))
    comparison = ModelComparison.load()
    assert list(comparison.models) == ["advisory", "reliance", "learning"]
    assert (tmp_path / "cache" / "weights-test" / "comparison" / "classifier.keras").exists()


def test_loading_warns_when_model_priors_changed(comparison_dir, tmp_path):
    import json
    import shutil

    copy = tmp_path / "stale"
    shutil.copytree(comparison_dir, copy)
    saved = json.loads((copy / "config.json").read_text())
    saved["metadata"]["model_signatures"]["reliance"] = {"changed": True}
    (copy / "config.json").write_text(json.dumps(saved))
    with pytest.warns(UserWarning, match=r"\['reliance'\].*retrain"):
        ModelComparison.load(copy)


def test_ood_check_returns_a_valid_p_value(tiny_comparison):
    result = ood_check(tiny_comparison, operator(1), n_reference=50, rng=np.random.default_rng(2))
    assert 1 / 51 <= result.p_value <= 1
    assert result.flagged == (result.p_value <= 0.05)


def test_calibrated_posterior_predictive_check_returns_valid_p_values(tiny_amortizer):
    result = posterior_predictive_check(
        tiny_amortizer, operator(7), n_replicates=40, n_calibration=19, seed=8
    )
    assert 1 / 20 <= result.p_value <= 1
    assert 1 / 41 <= result.uncalibrated_p_value <= 1
    assert result.flagged == (result.p_value <= 0.05)
    assert result.n_reference == 19


def test_uncalibrated_posterior_predictive_check(tiny_amortizer):
    result = posterior_predictive_check(
        tiny_amortizer, operator(7), n_replicates=40, n_calibration=0, seed=8
    )
    assert result.p_value == result.uncalibrated_p_value
    assert 1 / 41 <= result.p_value <= 1
    assert result.n_reference == 40


def test_rank_p_values_have_the_nominal_false_alarm_rate():
    # Under the null, the observed row is exchangeable with the references, so
    # P(p <= alpha) is at most alpha even with a covariance estimated from the data.
    rng = np.random.default_rng(3)
    p_values = [
        rank_p_value(mahalanobis_ranks(rng.standard_t(5, size=(200, 8)))) for _ in range(2000)
    ]
    assert np.mean(np.array(p_values) <= 0.05) == pytest.approx(0.05, abs=0.015)


def test_rank_p_values_flag_an_outlier():
    rng = np.random.default_rng(4)
    summaries = rng.normal(size=(300, 6))
    summaries[0] += 6.0
    assert rank_p_value(mahalanobis_ranks(summaries)) < 0.01


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_misspecified_generators_produce_valid_data(name):
    rng = np.random.default_rng(5)
    stimulus, advice = make_design([(40, None), (80, 0.8)], rng, n_sims=6)
    choice, rt = SCENARIOS[name](stimulus, advice, rng)
    assert choice.shape == rt.shape == stimulus.shape
    assert np.isin(choice, (0, 1)).all()
    assert np.all(np.isfinite(rt)) and np.all(rt > 0)
    TrialData(stimulus[0], advice[0], choice[0], rt[0])  # passes validation


def test_always_follow_copies_the_advice_on_aided_trials():
    rng = np.random.default_rng(6)
    stimulus, advice = make_design([(20, None), (50, 0.7)], rng, n_sims=3)
    choice, _ = always_follow(stimulus, advice, rng)
    aided = advice != NO_ADVICE
    np.testing.assert_array_equal(choice[aided], advice[aided])
