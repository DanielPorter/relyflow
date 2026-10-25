# Roadmap

relyflow is built in nine milestones. Each one lists what it delivers and what counts
as done. Later milestones depend on earlier ones, especially on milestone 2: every
network, diagnostic and report inherits any simulator bug.

## 1. Model specification ✅

Specs with sources, equations, parameters and priors are in [docs/models/](docs/models/).

- [x] Static **advisory** accumulator: the aid's advice inhibits the accumulator that
      disagrees with it (Strickland et al., 2021).
- [x] Static **reliance** variant: the advice also excites the agreeing accumulator,
      giving the aid autonomy to trigger decisions (Strickland et al., 2023). Advisory
      is reliance with excitation fixed at 0, which sets up milestone 5's comparison.
- [x] **Reliability-learning** extension: delta-rule perceived reliability
      (Strickland et al., 2024) scales the advice effects trial by trial. The learning
      rule is published; its coupling to the accumulator is relyflow's own and is
      documented as such.
- [x] Parameter tables (meaning, support, prior) and trial-level data requirements.
- [x] Unaided (manual) trials: **required**. Without them, excitation and inhibition
      are not identifiable. `TrialData` marks them with `NO_ADVICE`.

**Done:** `Advisory`, `Reliance` and `ReliabilityLearning` define priors and
mean-rate and threshold mappings, tested against the papers' definitions. The
race simulator is a stub until milestone 2.

## 2. Validated simulators ✅

Results are in [docs/models/](docs/models/README.md#simulators-and-likelihood).

- [x] Vectorized NumPy LBA race simulator (`relyflow.models.lba`), plus an exact,
      vectorized `log_likelihood` for all three models.
- [x] Design generator (`relyflow.make_design`) for unaided and aided blocks with
      exact aid-error counts.
- [x] Sampler checked against the analytic density by chi-square goodness of fit in
      four regimes. The tests catch planted bugs, including a 10% rate-SD error.
- [x] Qualitative-pattern tests across prior draws, using paired aided/unaided twins
      with common random numbers. Findings: the RT cost of wrong advice is a
      signature of inhibition and can reverse under strong excitation; pooled RT
      effects can reverse across stimulus types (Simpson's paradox).
- [x] Learning signatures: less following right after an aid failure, and less
      misuse after reliability drops. The static model shows neither.
- [x] Prior predictive checks: the priors cover the 2023 data. Time scale decided:
      RTs will be expressed in task-specific units at fit and simulation time
      (milestones 6–7) instead of widening the priors.
- [x] Throughput: about 6M trials/s per core for every model (`benchmarks/simulators.py`).

## 3. Amortized inference networks ✅

Details and results are in [docs/inference.md](docs/inference.md).

- [x] Online training-set generation with fixed seeds; every run saves its full
      `TrainingConfig`, versions and the model's prior signature
      (`python -m relyflow.inference train`).
- [x] Long RT tail: networks see log RTs clipped to [0.05 s, 30 s]. Simulations
      are not truncated, so they stay consistent with the exact likelihood used in
      milestone 4. Fitting real data with deadlines is revisited in milestone 6.
- [x] Static models: set transformer + affine coupling flow. Recovery R² 0.60–0.90,
      calibrated 90% intervals, 0.03 s per operator.
- [x] Learning model: deep set over trial features plus each trial's aid track
      record (sequence-aware history features), with 14 learning statistics passed
      to the flow. A recurrent network did worse on every parameter. Non-learning
      parameters R² 0.72–0.90; log learning rate R² 0.28, against 0.58 for a
      near-exact posterior (0.02 without the statistics). The starting belief `r0`
      is barely identifiable (near-exact R² 0.19).
- [x] Variable trial counts (120–1,200 in training, on a grid of 60 to avoid
      recompiling), with warnings outside the trained range or without unaided
      trials.
- [ ] Publish pretrained networks as versioned GitHub release assets
      (`relyflow.inference.pretrained`: download once, verify SHA-256, cache;
      `RELYFLOW_WEIGHTS_DIR` for local networks). Code done; the release follows
      the milestone 3 commit.

## 4. Validation suite ✅

Results are in [docs/validation/](docs/validation/README.md). Run with
`python -m relyflow.diagnostics validate <network> --out <dir> [--refine importance]`.

- [x] Parameter recovery and coverage: R², NRMSE, contraction, 50/90/95% intervals.
- [x] Simulation-based calibration: log-gamma rank-uniformity test (family-wise 5%)
      and ECDF plots, on operators from the training design distribution.
- [x] Posterior predictive checks on accuracy and correct-RT quantiles by trial type.
- [x] MCMC benchmark against NUTS (`relyflow.mcmc`: NumPyro on a JAX twin of the
      likelihood, tested equal to the NumPy one), run in parallel workers. Covers
      all three models, since all have exact likelihoods.
- [x] Exact-likelihood refinement (`relyflow.inference.refine`): importance
      resampling (1–2 s per operator) and NUTS from network draws.

**Outcome:** raw networks are calibrated to conservative, with posteriors 1.1–1.8×
wider than NUTS (reliance fails SBC on two mismatch rates for this reason).
Importance-refined posteriors pass every check for every model and match NUTS
(mean error ≤ 0.07 SD, SD ratio 0.96–1.03), about 170× faster than NUTS. Along
the way, the LBA density was made stable deep in the truncation tail.

## 5. Model comparison and misspecification

- [x] Amortized model comparison returning posterior model probabilities (advisory vs.
      reliance vs. learning), with calibration and a confusion matrix on simulated data
      (`relyflow.comparison.ModelComparison`, an evidence network).
- [x] Out-of-distribution check that flags data the simulators cannot plausibly produce:
      a prior predictive rank test in learned-summary plus statistic space with an exact
      false-alarm rate (`ood_check`), and a calibrated posterior predictive check
      against the fitted model (`posterior_predictive_check`).
- [x] Cross-check of classifier probabilities against importance-sampled marginal
      likelihoods (`relyflow.inference.refine.log_evidence`).

**Done when** model probabilities are calibrated on held-out simulations and the OOD
check catches deliberately misspecified data at a documented false-alarm rate.

**Outcome** ([report](docs/validation/comparison/summary.md)): on 3,000 held-out
simulations, model probabilities are calibrated (ECE 0.009, accuracy 0.73; errors are
mostly nested models with negligible excitation or learning, for which the data cannot
decide). They agree with the exact likelihood (same most probable model for 90% of
operators, log Bayes factor correlation 0.94 for reliance vs. advisory and 0.80 for
learning vs. reliance). At alpha = 0.05, the posterior predictive check, calibrated
against prior-predictive datasets (Hjort et al., 2006; affordable because every extra
posterior is one network pass), flags 4% of operators simulated from the models and
catches 69–95% of six misspecified generators (lapses, random responding, always
following the aid, a diffusion model with start-point bias, fatigue, a slow task).
Combined with the prior check at alpha / 2 each, it flags 3% and catches 65–95%.

## 6. Per-operator fitting interface and reports

- [ ] `relyflow.fit(data, model=...)` returning a posterior object (samples, summaries,
      plots) in seconds.
- [ ] Calibration summary: is this operator over-relying, under-relying, or tracking
      the aid's actual reliability? Decision rule stated in terms of posterior
      probabilities.
- [ ] Short report as text, a DataFrame and an optional HTML/PDF one-pager.

## 7. Synthetic operator mode

- [ ] Sample a population of simulated operators from a population-level distribution
      (fitted or user-specified).
- [ ] Run them against an aid at a grid of reliability levels.
- [ ] Output predicted accuracy, RT distributions, misuse and disuse rates, with
      uncertainty intervals.

## 8. Real-data demonstration

- [ ] Pick a dataset. Candidates found in milestone 1 (licenses still to check): the
      Strickland et al. (2021) data (<https://osf.io/28g3j/>; 24 × 2,400 trials, 90% aid
      vs. manual) and the Strickland et al. (2023) data (<https://osf.io/q3b2r/>;
      24 × about 4,680 trials, 95% and 75% aids vs. manual, with trust ratings). Both
      have trial order, choices, RTs, aid correctness and unaided baselines.
- [ ] Test the learning model's coupling, which is not yet validated on data.
- [ ] Run the full pipeline: OOD check, model comparison, fits, reports, PPCs.
- [ ] Show the results are sensible and interpretable, and compare with published
      findings where possible.

## 9. Packaging and publication

- [ ] Dockerfile with a pinned environment and pretrained weights.
- [ ] Documentation site (API reference, model specs, validation results).
- [ ] Tutorial notebooks: fitting one operator, comparing models, evaluating an aid.
- [ ] Test coverage for every public API; slow tests marked and run on a schedule.
- [ ] Make the repository public, release on PyPI with trusted publishing, and archive
      on Zenodo for a DOI.
- [ ] JOSS paper (`paper/paper.md`, `paper/paper.bib`) and submission.
