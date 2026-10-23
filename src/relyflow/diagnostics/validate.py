"""Run the full validation suite for one trained network and write a report."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from relyflow import TrialData, make_design
from relyflow.diagnostics.calibration import recovery_table, sbc_table, simulate_operators
from relyflow.diagnostics.ppc import PPC_STATISTICS, posterior_predictive, ppc_check, ppc_statistics
from relyflow.inference import load
from relyflow.inference.refine import importance_resample

# Pass criteria.
COVERAGE_TOLERANCE_SE = 3.0  # 90% coverage may differ from 0.9 by 3 binomial SEs
PPC_MAX_OUTSIDE = 0.075  # share of statistics outside 95% predictive intervals
MCMC_MAX_MEAN_ERROR = 0.3  # median |mean error| in reference SDs
MCMC_SD_RATIO = (0.75, 1.33)  # acceptable median SD ratio


def benchmark_operators(model, n: int, rng: np.random.Generator) -> list[TrialData]:
    """Operators on a standard design: 200 unaided trials, then aided blocks.

    The learning model's aid drops from 85-98% to 55-75% reliability midway, so
    that learning leaves a trace; the static models get one 60-98% block.
    """
    truth = model.sample_prior(n, rng)
    if model.name == "learning":
        blocks = [(200, None), (300, rng.uniform(0.85, 0.98, n)), (300, rng.uniform(0.55, 0.75, n))]
    else:
        blocks = [(200, None), (400, rng.uniform(0.6, 0.98, n))]
    stimulus, advice = make_design(blocks, rng, n_sims=n)
    choice, rt = model.simulate(truth, stimulus, advice, rng)
    return [TrialData(stimulus[i], advice[i], choice[i], rt[i]) for i in range(n)]


def validate(
    name_or_path: str,
    out_dir: str | Path,
    n_sbc: int = 1000,
    n_draws: int = 1000,
    n_ppc: int = 200,
    n_mcmc: int = 20,
    refine: str | None = None,
    seed: int = 0,
    workers: int = 10,
) -> dict:
    """Recovery, SBC, posterior predictive checks and (optionally) the MCMC benchmark.

    ``refine="importance"`` validates importance-resampled posteriors instead of
    raw network draws. Writes ``metrics.json``, ``summary.md`` and figures to
    ``out_dir`` and returns the metrics.
    """
    import matplotlib

    matplotlib.use("Agg")
    from bayesflow.diagnostics.plots import calibration_ecdf, recovery

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    amortizer = load(name_or_path)
    model = amortizer.model
    names = list(model.param_names)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    # 1. Posteriors for operators simulated from the training distribution.
    truths, draws, operators = [], [], []
    for truth, stimulus, advice, choice, rt in simulate_operators(
        model, n_sbc, amortizer.config.design, rng
    ):
        # Fresh seeds per batch and operator keep the SBC ranks independent.
        batch_seed = seed + len(operators)
        batch = amortizer.sample_batch(
            stimulus, advice, choice, rt, num_samples=n_draws, seed=batch_seed
        )
        for i in range(len(stimulus)):
            data = TrialData(stimulus[i], advice[i], choice[i], rt[i])
            if refine == "importance":
                posterior = importance_resample(
                    amortizer, data, num_samples=n_draws, seed=seed + len(operators)
                )
                draws.append(posterior.draws)
            else:
                draws.append({name: batch[name][i] for name in names})
            operators.append(data)
        truths.append(truth)
    truth = {name: np.concatenate([t[name] for t in truths]) for name in names}
    stacked = {name: np.stack([d[name] for d in draws]) for name in names}

    # 2. Recovery, coverage and SBC.
    rec = recovery_table(model, truth, stacked, rng)
    sbc = sbc_table(model, truth, stacked)
    tolerance = COVERAGE_TOLERANCE_SE * np.sqrt(0.9 * 0.1 / n_sbc)
    rec["coverage_90_ok"] = (rec["coverage_90"] - 0.9).abs() <= tolerance
    estimates = np.stack([stacked[n] for n in names], axis=-1)
    targets = np.stack([truth[n] for n in names], axis=-1)
    calibration_ecdf(estimates, targets, variable_names=names, difference=True).savefig(
        out / "sbc_ecdf.png", dpi=80, bbox_inches="tight"
    )
    recovery(estimates, targets, variable_names=names).savefig(
        out / "recovery.png", dpi=80, bbox_inches="tight"
    )

    # 3. Posterior predictive checks.
    outside = []
    for i, data in enumerate(operators[:n_ppc]):
        observed = ppc_statistics(data.stimulus, data.advice, data.choice, data.rt)
        replicated = posterior_predictive(
            model, {n: stacked[n][i] for n in names}, data.stimulus, data.advice, rng
        )
        outside.append(ppc_check(observed, replicated)["outside_95"])
    outside = np.array(outside, dtype=float)
    ppc = pd.DataFrame(
        {"statistic": PPC_STATISTICS, "share_outside_95": np.nanmean(outside, axis=0)}
    )
    ppc_share = float(np.nanmean(outside))

    metrics = {
        "model": model.name,
        "network": str(name_or_path),
        "refine": refine,
        "n_sbc": n_sbc,
        "n_draws": n_draws,
        "recovery": rec.to_dict(orient="records"),
        "sbc": sbc.to_dict(orient="records"),
        "ppc": ppc.to_dict(orient="records"),
        "ppc_share_outside_95": ppc_share,
        "passed": {
            "sbc": bool(sbc["passed"].all()),
            "coverage_90": bool(rec["coverage_90_ok"].all()),
            "ppc": ppc_share <= PPC_MAX_OUTSIDE,
        },
    }

    # 4. MCMC benchmark.
    if n_mcmc:
        from relyflow.diagnostics.benchmark import run_benchmark

        bench_ops = benchmark_operators(model, n_mcmc, np.random.default_rng(seed + 1))
        agreement, timing = run_benchmark(amortizer, bench_ops, seed=seed, workers=workers)
        agreement.to_csv(out / "mcmc_agreement.csv", index=False)
        timing.to_csv(out / "mcmc_timing.csv", index=False)
        summary = (
            agreement.assign(abs_mean_error=agreement["mean_error"].abs())
            .groupby(["method", "parameter"], sort=False)[
                ["abs_mean_error", "sd_ratio", "wasserstein"]
            ]
            .median()
            .reset_index()
        )
        nuts = timing[timing["method"] == "nuts"]
        metrics["mcmc"] = {
            "agreement": summary.to_dict(orient="records"),
            "median_seconds": timing.groupby("method")["seconds"].median().to_dict(),
            "nuts_converged": int(((nuts["max_rhat"] < 1.01) & (nuts["min_ess"] > 400)).sum()),
            "nuts_divergent_runs": int((nuts["divergences"] > 0).sum()),
            "importance_efficiency_median": float(nuts["efficiency"].median()),
            "n_operators": n_mcmc,
        }
        for method in ("network", "importance"):
            rows = summary[summary["method"] == method]
            metrics["passed"][f"mcmc_{method}"] = bool(
                (rows["abs_mean_error"] <= MCMC_MAX_MEAN_ERROR).all()
                and rows["sd_ratio"].between(*MCMC_SD_RATIO).all()
            )

    metrics["seconds"] = round(time.perf_counter() - started, 1)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float) + "\n")
    (out / "summary.md").write_text(render_summary(metrics))
    return metrics


def render_summary(m: dict) -> str:
    """A Markdown report of a validation run."""
    title = f"{m['model']} model" + (f", {m['refine']}-refined" if m["refine"] else "")
    lines = [
        f"# Validation: {title}",
        "",
        f"Generated by `python -m relyflow.diagnostics validate` from `{m['network']}`; "
        f"{m['n_sbc']} simulated operators from the training design distribution, "
        f"{m['n_draws']} posterior draws each. Total time {m['seconds']:.0f} s.",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    lines += [f"| {k} | {'pass' if v else '**FAIL**'} |" for k, v in m["passed"].items()]
    lines += ["", "## Recovery, coverage and SBC", ""]
    lines += [
        "| Parameter | R² | NRMSE | Contraction | 50% cov. | 90% cov. | 95% cov. | SBC log-gamma |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r, s in zip(m["recovery"], m["sbc"], strict=True):
        flag = "" if s["passed"] else " ✗"
        lines.append(
            f"| `{r['parameter']}` | {r['r2']:.2f} | {r['nrmse']:.3f} | {r['contraction']:.2f} "
            f"| {r['coverage_50']:.2f} | {r['coverage_90']:.2f} | {r['coverage_95']:.2f} "
            f"| {s['log_gamma']:.2f}{flag} |"
        )
    lines += [
        "",
        "SBC uses the log-gamma rank-uniformity test with a Bonferroni-corrected family-wise",
        "level of 5%; log-gamma < 0 (✗) rejects calibration. Figures: `sbc_ecdf.png`,",
        "`recovery.png`.",
        "",
        "## Posterior predictive checks",
        "",
        f"Share of statistics outside their 95% predictive interval: "
        f"{m['ppc_share_outside_95']:.3f} (pass if ≤ {PPC_MAX_OUTSIDE}).",
        "",
        "| Statistic | Share outside 95% |",
        "| --- | --- |",
    ]
    lines += [f"| `{p['statistic']}` | {p['share_outside_95']:.3f} |" for p in m["ppc"]]
    if "mcmc" in m:
        mc = m["mcmc"]
        lines += [
            "",
            f"## MCMC benchmark ({mc['n_operators']} operators, NUTS reference)",
            "",
            f"NUTS: {mc['nuts_converged']} of {mc['n_operators']} runs converged "
            f"(R̂ < 1.01, ESS > 400); {mc['nuts_divergent_runs']} had divergences. "
            f"Importance-sampling efficiency (median): {mc['importance_efficiency_median']:.3f}.",
            "",
            "Median single-core seconds per operator: "
            + ", ".join(f"{k} {v:.2f}" for k, v in mc["median_seconds"].items())
            + ".",
            "",
            "Median agreement with NUTS, in NUTS posterior SDs (pass: |mean error| ≤ "
            f"{MCMC_MAX_MEAN_ERROR}, SD ratio in {list(MCMC_SD_RATIO)}):",
            "",
            "| Method | Parameter | abs mean error | SD ratio | Wasserstein |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {a['method']} | `{a['parameter']}` | {a['abs_mean_error']:.2f} "
            f"| {a['sd_ratio']:.2f} | {a['wasserstein']:.2f} |"
            for a in mc["agreement"]
        ]
    return "\n".join(lines) + "\n"
