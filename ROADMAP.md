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

## 2. Validated simulators

- [ ] Vectorized NumPy LBA race simulator, one call for many operators × many trials,
      built on the mean-rate and threshold mappings from milestone 1.
- [ ] Prior predictive checks with the validated simulator (milestone 1 ran only a
      preliminary check), including whether a time-scale option is needed for tasks
      much faster or slower than 0.5–3 s.
- [ ] Qualitative-pattern tests: higher accuracy when the aid is right; slower correct
      responses when the aid is wrong; misuse rising with excitation; learned trust
      tracking the aid's running reliability and dropping after aid failures. Target
      values are in [docs/models/static.md](docs/models/static.md).
- [ ] Where a likelihood exists, check simulated choice and RT distributions against it.
- [ ] Throughput benchmark, so that training-set size is not limited by simulation.

**Done when** the pattern tests pass across the prior range and simulated data from
each model match its analytic likelihood (where one exists) by a goodness-of-fit test.

## 3. Amortized inference networks

- [ ] Training-set generation with fixed seeds and saved configs.
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
- [ ] MCMC benchmark for the static model (tractable likelihood): compare amortized and
      NUTS posteriors (means, SDs, distributional distance) and wall-clock time.

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
