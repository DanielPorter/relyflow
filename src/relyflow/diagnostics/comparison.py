"""Validation of model comparison and the out-of-distribution check (milestone 5).

Run with ``python -m relyflow.diagnostics compare checkpoints/comparison --out DIR``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from relyflow import TrialData, make_design
from relyflow.comparison import ModelComparison, ood_check, posterior_predictive_check
from relyflow.comparison.classifier import ComparisonSimulator
from relyflow.diagnostics.misspecification import SCENARIOS
from relyflow.inference import load
from relyflow.inference.refine import log_evidence
from relyflow.models import get_model


def _standard_design(n: int, rng: np.random.Generator, shift: bool):
    """200 unaided trials, then 400 aided; with ``shift`` the aid drops from 85-98% to
    55-75% reliability halfway, which makes learning identifiable."""
    if shift:
        blocks = [(200, None), (200, rng.uniform(0.85, 0.98, n)), (200, rng.uniform(0.55, 0.75, n))]
    else:
        blocks = [(200, None), (400, rng.uniform(0.6, 0.98, n))]
    return make_design(blocks, rng, n_sims=n)


def calibration(comparison: ModelComparison, n: int, seed: int) -> dict:
    """Accuracy, confusion and calibration on held-out simulations (training distribution)."""
    simulator = ComparisonSimulator(comparison.models, comparison.config.design, seed)
    probs, truth = [], []
    for start in range(0, n, 50):
        index, stimulus, advice, choice, rt = simulator.simulate(min(50, n - start))
        probs.append(comparison.estimate_batch(stimulus, advice, choice, rt)["model_probs"])
        truth.append(index)
    probs, truth = np.concatenate(probs), np.concatenate(truth)
    k = len(comparison.models)
    onehot = np.eye(k)[truth]
    confidence = probs.max(axis=1)
    bins = np.minimum((confidence * 10).astype(int), 9)
    correct = probs.argmax(axis=1) == truth
    ece = sum(
        abs(correct[bins == b].mean() - confidence[bins == b].mean()) * np.mean(bins == b)
        for b in range(10)
        if np.any(bins == b)
    )
    return {
        "probs": probs,
        "onehot": onehot,
        "accuracy": float(correct.mean()),
        "expected_calibration_error": float(ece),
        "mean_probability_by_true_model": _mean_probs(probs, truth, comparison.models),
        "mean_log_score": float(
            np.mean(np.log(np.maximum(probs[np.arange(len(truth)), truth], 1e-12)))
        ),
    }


def _mean_probs(probs: np.ndarray, truth: np.ndarray, models) -> dict:
    """Mean posterior probability of each model, by true model."""
    table = {}
    for i, true_name in enumerate(models):
        row = probs[truth == i].mean(axis=0).round(3).tolist()
        table[true_name] = dict(zip(models, row, strict=True))
    return table


def crosscheck(comparison: ModelComparison, amortizers: dict, n_per_model: int, seed: int) -> dict:
    """Compare classifier probabilities with probabilities from importance-sampled
    marginal likelihoods (the exact likelihood), on a design with a reliability shift."""
    rng = np.random.default_rng(seed)
    rows = []
    for true_name in comparison.models:
        model = get_model(true_name)
        stimulus, advice = _standard_design(n_per_model, rng, shift=True)
        choice, rt = model.simulate(model.sample_prior(n_per_model, rng), stimulus, advice, rng)
        classifier = comparison.estimate_batch(stimulus, advice, choice, rt)["model_probs"]
        for i in range(n_per_model):
            data = TrialData(stimulus[i], advice[i], choice[i], rt[i])
            evidence, ess = [], []
            for name in comparison.models:
                log_z, n_eff = log_evidence(amortizers[name], data, seed=seed + i)
                evidence.append(log_z)
                ess.append(n_eff)
            exact = np.exp(np.array(evidence) - max(evidence))
            rows.append(
                {
                    "true": true_name,
                    "classifier": classifier[i],
                    "exact": exact / exact.sum(),
                    "min_ess": float(min(ess)),
                }
            )
    classifier = np.array([r["classifier"] for r in rows])
    exact = np.array([r["exact"] for r in rows])
    reliable = np.array([r["min_ess"] >= 50 for r in rows])

    def log_bf(p, a, b):
        return np.log(np.maximum(p[:, a], 1e-12)) - np.log(np.maximum(p[:, b], 1e-12))

    names = list(comparison.models)
    truth = np.array([names.index(r["true"]) for r in rows])
    correlations = {}
    for a, b in [("reliance", "advisory"), ("learning", "reliance")]:
        ia, ib = names.index(a), names.index(b)
        x = log_bf(classifier, ia, ib)[reliable]
        y = log_bf(exact, ia, ib)[reliable]
        correlations[f"{a} vs {b}"] = float(np.corrcoef(x, y)[0, 1])
    return {
        "n_operators": len(rows),
        "n_reliable_evidence": int(reliable.sum()),
        "mean_abs_probability_difference": float(np.abs(classifier - exact)[reliable].mean()),
        "same_most_probable_model": float(np.mean(classifier.argmax(1) == exact.argmax(1))),
        "log_bayes_factor_correlation": correlations,
        "accuracy": {
            "classifier": float(np.mean(classifier.argmax(1) == truth)),
            "exact": float(np.mean(exact.argmax(1) == truth)),
        },
    }


def ood_rates(
    comparison: ModelComparison, nets: dict, n: int, n_reference: int, alpha: float, seed: int
) -> dict:
    """Flag rates of the misspecification checks on model data and misspecified data.

    Reports the prior check (``ood_check``), the posterior check
    (``posterior_predictive_check``) before and after calibration, and the two
    calibrated checks combined, each at ``alpha / 2`` (Bonferroni), which keeps
    the combined false-alarm rate at most ``alpha``. Model data are split evenly
    across the models, each checked against its own posterior network.
    Misspecified data are checked against the reliance model, the most flexible
    static model.
    """
    rng = np.random.default_rng(seed)

    def p_values(data: TrialData, net, i: int) -> tuple[float, float, float]:
        prior = ood_check(comparison, data, n_reference=n_reference, rng=rng)
        posterior = posterior_predictive_check(net, data, seed=seed + i)
        return prior.p_value, posterior.uncalibrated_p_value, posterior.p_value

    def summarize(p) -> dict:
        p = np.array(p)
        return {
            "prior": float(np.mean(p[:, 0] <= alpha)),
            "posterior_uncalibrated": float(np.mean(p[:, 1] <= alpha)),
            "posterior": float(np.mean(p[:, 2] <= alpha)),
            "combined": float(np.mean(np.minimum(p[:, 0], p[:, 2]) <= alpha / 2)),
        }

    p = []
    for k, name in enumerate(comparison.models):
        model = get_model(name)
        m = n // len(comparison.models) + (k < n % len(comparison.models))
        stimulus, advice = _standard_design(m, rng, shift=False)
        choice, rt = model.simulate(model.sample_prior(m, rng), stimulus, advice, rng)
        for i in range(m):
            data = TrialData(stimulus[i], advice[i], choice[i], rt[i])
            p.append(p_values(data, nets[name], i))
    results = {"false alarms (data from the models)": summarize(p)}

    for name, generator in SCENARIOS.items():
        stimulus, advice = _standard_design(n, rng, shift=False)
        choice, rt = generator(stimulus, advice, rng)
        p = [
            p_values(TrialData(stimulus[i], advice[i], choice[i], rt[i]), nets["reliance"], i)
            for i in range(n)
        ]
        results[f"detection: {name}"] = summarize(p)
    return results


def evaluate(
    comparison_path: str,
    out_dir: str | Path,
    amortizers: str | None = None,
    n_calibration: int = 3000,
    n_crosscheck_per_model: int = 20,
    n_ood: int = 150,
    n_reference: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Run all model-comparison checks and write ``metrics.json``, ``summary.md`` and figures.

    ``amortizers`` is a directory with ``<model>/`` training outputs for the exact
    cross-check; by default the pretrained networks are used.
    """
    import matplotlib

    matplotlib.use("Agg")
    from bayesflow.diagnostics.plots import mc_calibration, mc_confusion_matrix

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    comparison = ModelComparison.load(comparison_path)
    names = list(comparison.models)
    started = time.perf_counter()

    cal = calibration(comparison, n_calibration, seed)
    mc_calibration(cal["probs"], cal["onehot"], model_names=names).savefig(
        out / "calibration.png", dpi=80, bbox_inches="tight"
    )
    mc_confusion_matrix(cal["probs"], cal["onehot"], model_names=names).savefig(
        out / "confusion.png", dpi=80, bbox_inches="tight"
    )
    nets = {m: load(str(Path(amortizers) / m)) if amortizers else load(m) for m in names}
    metrics = {
        "comparison": comparison_path,
        "calibration": {k: v for k, v in cal.items() if k not in ("probs", "onehot")},
        "n_calibration": n_calibration,
        "crosscheck": crosscheck(comparison, nets, n_crosscheck_per_model, seed + 1),
        "ood": {
            "alpha": alpha,
            "n_per_condition": n_ood,
            "n_reference": n_reference,
            "rates": ood_rates(comparison, nets, n_ood, n_reference, alpha, seed + 2),
        },
    }
    metrics["seconds"] = round(time.perf_counter() - started, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "summary.md").write_text(render(metrics))
    return metrics


def render(m: dict) -> str:
    cal, cc, ood = m["calibration"], m["crosscheck"], m["ood"]
    lines = [
        "# Validation: model comparison and out-of-distribution check",
        "",
        f"Generated by `python -m relyflow.diagnostics compare` from `{m['comparison']}`. "
        f"Total time {m['seconds']:.0f} s.",
        "",
        "## Model comparison",
        "",
        f"On {m['n_calibration']} held-out simulations from the training distribution "
        f"(equal prior odds): accuracy of the most probable model {cal['accuracy']:.3f}, "
        f"expected calibration error {cal['expected_calibration_error']:.3f}. "
        "Figures: `calibration.png`, `confusion.png`.",
        "",
        "Mean posterior probability assigned to each model (rows: true model):",
        "",
        "| True model | " + " | ".join(cal["mean_probability_by_true_model"]) + " |",
        "| --- |" + " --- |" * len(cal["mean_probability_by_true_model"]),
    ]
    for true, row in cal["mean_probability_by_true_model"].items():
        lines.append(f"| {true} | " + " | ".join(f"{v:.2f}" for v in row.values()) + " |")
    lines += [
        "",
        "## Cross-check against the exact likelihood",
        "",
        f"{cc['n_operators']} operators on a design whose aid drops from 85-98% to 55-75% "
        "reliability. Classifier probabilities are compared with probabilities from "
        f"importance-sampled marginal likelihoods ({cc['n_reliable_evidence']} operators had "
        "ESS ≥ 50 for every model and are used for the difference and correlations).",
        "",
        "- Mean absolute difference in model probabilities: "
        f"{cc['mean_abs_probability_difference']:.3f}",
        f"- Same most probable model: {cc['same_most_probable_model']:.2f}",
        "- Correlation of log Bayes factors: "
        + ", ".join(f"{k} {v:.2f}" for k, v in cc["log_bayes_factor_correlation"].items()),
        f"- Accuracy: classifier {cc['accuracy']['classifier']:.2f}, "
        f"exact {cc['accuracy']['exact']:.2f}",
        "",
        "## Out-of-distribution check",
        "",
        f"alpha = {ood['alpha']}, {ood['n_reference']} reference operators per check, "
        f"{ood['n_per_condition']} operators per condition (200 unaided + 400 aided trials).",
        "",
        "Prior check: could any model, with prior-plausible parameters, produce the data?",
        "Posterior check: does the fitted model reproduce them? Uncalibrated, it is",
        "conservative; calibrated against datasets from the prior (Hjort et al., 2006), its",
        "false-alarm rate is alpha. Combined: either calibrated check at alpha / 2.",
        "",
        "| Condition | Prior check | Posterior, uncalibrated | Posterior, calibrated | Combined |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {k} | {v['prior']:.2f} | {v['posterior_uncalibrated']:.2f} "
        f"| {v['posterior']:.2f} | {v['combined']:.2f} |"
        for k, v in ood["rates"].items()
    ]
    return "\n".join(lines) + "\n"
