# Roadmap

relyflow is built in nine milestones. Each one lists what it delivers and what counts
as done. Later milestones depend on earlier ones, especially on milestone 2: every
network, diagnostic and report inherits any simulator bug.

## 1. Model specification

Choose two or three published automation-use models and pin each one down precisely.

- [ ] Static **advisory** accumulator: the aid's advice changes the rates of the evidence
      accumulators (excitation of the advised response, inhibition of the other).
- [ ] Static **reliance** variant: on some trials the operator adopts the aid's advice
      instead of deciding independently. Needed for the advising-vs-reliance comparison
      in milestone 5.
- [ ] **Reliability-learning** extension: trust in the aid is updated trial by trial with
      a delta rule (prediction error on aid correctness) and scales the advice effect.
- [ ] For each model, a document in `docs/models/` giving source citations, equations,
      the parameter table (name, meaning, support, prior), and the trial-level data it
      needs.
- [ ] Decide whether unaided (manual) trials are supported, since they help identify
      baseline accumulation rates.

**Done when** each model has a written spec reviewed against its source paper and a
matching `AutomationModel` subclass with priors (simulator still a stub).

## 2. Validated simulators

- [ ] Vectorized NumPy simulators, one call for many operators × many trials.
- [ ] Qualitative-pattern tests: higher accuracy when the aid is right; slower correct
      responses when the aid is wrong; misuse rising with reliance; learned trust
      tracking the aid's running reliability and dropping after aid failures.
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

- [ ] Pick a dataset: a publicly posted automation-use dataset (check its license and
      that it has trial-level choices, RTs, advice and aid correctness), or run a small
      online study.
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
