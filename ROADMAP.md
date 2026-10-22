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

## 3. Amortized inference networks

- [ ] Training-set generation with fixed seeds and saved configs.
- [ ] Handle the LBA's long RT tail (infinite RT variance; 0.5–0.8% of simulated
      trials over 10 s): censor at a task deadline and/or summarize log RTs, and
      match whatever preprocessing real data get.
- [ ] Static models: permutation-invariant set encoder (e.g. SetTransformer or DeepSet)
      feeding a normalizing-flow or flow-matching posterior.
- [ ] Learning model: sequence-aware summary network, because trial order carries the
      information about reliability updating.
- [ ] Support variable trial counts, so the same network serves short and long sessions.
- [ ] Ship or download pretrained weights, versioned alongside the simulator code.

**Done when** each model has a trained network that returns posteriors for one operator
in under a few seconds on a laptop CPU.

## 4. Validation suite

- [ ] Parameter recovery: posterior mean vs. truth, R², NRMSE, posterior contraction.
- [ ] Simulation-based calibration: rank histograms and ECDF difference plots.
- [ ] Posterior predictive checks on accuracy, RT quantiles, misuse and disuse.
- [ ] MCMC benchmark: compare amortized and NUTS posteriors (means, SDs,
      distributional distance) and wall-clock time. All three models have exact
      likelihoods (`model.log_likelihood`), so the learning model can be
      benchmarked too, not just the static ones.

**Done when** all diagnostics are produced by one command and pass for every model, and
the benchmark results are recorded in the docs.

## 5. Model comparison and misspecification

- [ ] Amortized model comparison returning posterior model probabilities (advisory vs.
      reliance vs. learning), with calibration and a confusion matrix on simulated data.
- [ ] Out-of-distribution check that flags data the simulators cannot plausibly produce,
      e.g. by distance in the learned summary space from the training distribution.

**Done when** model probabilities are calibrated on held-out simulations and the OOD
check catches deliberately misspecified data at a documented false-alarm rate.

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
