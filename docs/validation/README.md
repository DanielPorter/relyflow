# Validation results

Generated with `python -m relyflow.diagnostics validate` (milestone 4). Each
folder holds a model's `summary.md`, `metrics.json`, SBC and recovery figures,
and, where run, the NUTS benchmark tables. The `*-importance` folders validate
importance-resampled posteriors (`relyflow.inference.refine.importance_resample`)
instead of raw network draws.

## What is checked

- **Recovery and coverage:** R², NRMSE, posterior contraction, and 50/90/95%
  interval coverage. 90% coverage must be within 3 binomial SEs of 0.9.
- **Simulation-based calibration (SBC):** operators simulated from the prior
  and the training design distribution, 1,000 per network (300 for refined
  posteriors). Rank uniformity uses the log-gamma test at a Bonferroni-corrected
  family-wise level of 5%.
- **Posterior predictive checks:** for 200 operators, accuracy and correct-RT
  quantiles (10/50/90%) on unaided, aid-right and aid-wrong trials, against 200
  replicated datasets each. The check passes if at most 7.5% of statistics fall
  outside their 95% predictive interval.
- **NUTS benchmark:** 20 operators per model on a standard design, with NUTS
  (4 chains × 1,000 warmup + 1,000 draws, started from the prior) as the
  reference. A method passes if, for every parameter, its median mean error is at
  most 0.3 reference SDs and its median SD ratio is in [0.75, 1.33].

## Results

| Model | Posterior | SBC operators | SBC | 90% coverage | PPC outside 95% | Matches NUTS | Time per operator |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [advisory](advisory/summary.md) | network | 1000 | pass | pass (0.91–0.93) | 0.007 | **fail** | 0.03 s |
| [advisory](advisory-importance/summary.md) | importance-refined | 300 | pass | pass (0.86–0.93) | 0.010 | pass | 1.2 s |
| | NUTS (reference) | | | | | | 200 s |
| [reliance](reliance/summary.md) | network | 1000 | **fail** (`v_mismatch_0`, `v_mismatch_1`) | **fail** (0.89–0.93) | 0.005 | **fail** | 0.03 s |
| [reliance](reliance-importance/summary.md) | importance-refined | 300 | pass | pass (0.87–0.92) | 0.007 | pass | 1.2 s |
| | NUTS (reference) | | | | | | 240 s |
| [learning](learning/summary.md) | network | 1000 | pass | pass (0.89–0.92) | 0.009 | **fail** | 0.03 s |
| [learning](learning-importance/summary.md) | importance-refined | 300 | pass | pass (0.88–0.92) | 0.015 | pass | 1.7 s |
| | NUTS (reference) | | | | | | 315 s |

NUTS times are single-core medians over the 20 benchmark operators; network and
importance times are on one CPU core plus the GPU when present.

### What the results mean

- **Raw networks are calibrated to conservative.** Their posteriors are
  1.1–1.8× wider than NUTS, with mean errors of 0.2–0.8 NUTS SDs, so they fail the
  benchmark's tolerance. They lose information rather than claiming false
  precision: coverage is at or slightly above nominal. The reliance network's SBC
  failure on the two mismatch rates is of this kind, too-wide posteriors plus a
  slight downward bias for `v_mismatch_1` (mean rank 0.525). It is detectable only
  because 1,000 operators make SBC sensitive to ECDF deviations of about 0.05.
- **The learning rate is where the networks lose the most.** The learning
  network's `alpha` posterior is 1.76× wider than NUTS (see
  [../inference.md](../inference.md#the-learning-model-getting-at-the-learning-rate)).
- **Importance resampling closes the gap for every model.** Against NUTS, mean
  errors are at most 0.07 NUTS SDs and SD ratios 0.96–1.03, at 1.2–1.7 s per
  operator against 200–315 s for NUTS on one core. Refined posteriors pass SBC and
  coverage for all three models, including the reliance parameters the raw network
  failed on.
- **Posterior predictive checks pass easily for every model.** Only 0.5–1.5% of
  statistics fall outside their 95% intervals, which is expected because
  posterior predictive checks are conservative when the model is correctly
  specified. They become informative for real data (milestone 8).
- **The NUTS reference itself is sound:** 58 of 60 runs converged (R̂ < 1.01,
  ESS > 400), and one run per model had divergences.

### Recommendation

Use importance resampling by default when fitting operators (milestone 6). It is
fast enough for interactive use, matches NUTS, and its effective sample size
flags the operators for which the network was a poor proposal. Those operators
can then be refined with NUTS (`nuts_refine`). The importance-sampling efficiency
is 17.5% for advisory, 12.7% for reliance and 3.6% for learning (medians).
