# Model comparison and out-of-distribution checks

Code: `relyflow/comparison/`. Validation: [validation/comparison/](validation/comparison/summary.md).

## Which mechanism describes this operator?

`ModelComparison` returns posterior probabilities of the three models for an
operator's data, assuming equal prior odds:

```python
from relyflow.comparison import ModelComparison

comparison = ModelComparison.load()  # pretrained; downloads once, checks SHA-256
comparison.probabilities(trial_data)
# {"advisory": 0.08, "reliance": 0.81, "learning": 0.11}
```

- **advisory vs. reliance** asks whether the operator only lets the aid inhibit
  the disagreeing response, or also lets it drive decisions (excitation).
- **learning vs. reliance** asks whether the aid's influence tracks its recent
  track record.

The models are nested: advisory is reliance with excitation fixed at 0, and
reliance is the learning model without learning. When an operator's excitation is
near 0, or learning is negligible, the data cannot favour one model strongly, and
the probabilities stay close to the prior. That is the correct answer, not a
failure.

### How it works

A classifier (an evidence network; Jeffrey & Wandelt, 2024, as implemented in
BayesFlow) is trained on simulations from all three models, drawn with equal
probability, on the same design distribution as the posterior networks. It sees
the learning model's inputs: per-trial features, each trial's aid track record,
and the 14 learning statistics. Trained with cross-entropy, its softmax outputs
are posterior model probabilities. `estimate_batch` also returns log Bayes factors
relative to the advisory model.

The pretrained network (release `weights-v2`) was trained for 20,000 steps. To
train your own, for example after changing a model's priors (loading warns when
the priors no longer match the network), run:

```bash
KERAS_BACKEND=torch python -m relyflow.comparison train --out checkpoints/comparison
```

and load it with `ModelComparison.load("checkpoints/comparison")`.

### Exact cross-check

Every model has an exact likelihood, so each operator's marginal likelihood can
also be estimated by importance sampling with the posterior network as the
proposal (`relyflow.inference.refine.log_evidence`). Normalizing these gives
model probabilities that do not depend on the classifier. The validation compares
the two (see the report).

## Could any model have produced these data?

`ood_check` flags operators whose data none of the models can plausibly produce:

```python
from relyflow.comparison import ood_check

result = ood_check(comparison, trial_data)  # 1-2 s
result.flagged, result.p_value
```

It simulates 500 reference operators from all three models, with parameters drawn
from the priors, **on the operator's own design** (the same stimulus and advice
sequence). It describes every dataset by a feature vector (below) and ranks the
observed operator's Mahalanobis distance among all 501. Under the
null hypothesis that one of the models produced the data, the observed operator
is exchangeable with the references. The rank is then a valid p-value, and the
false-alarm rate is at most α by construction: a prior predictive check in
learned summary space (cf. Schmitt et al., 2024).

The distance is measured on the comparison network's learned summary together with
17 interpretable statistics (`ood_statistics`): accuracy and correct-RT quantiles
by trial type, error RTs, RT spread, the rate of following advice, and trends over
the session. The learned summary is trained only to tell the models apart, so on
its own it ignores features none of the models produce, such as a slowdown over
the session.

### Does the fitted model reproduce the data?

The prior check asks whether *any* model with *any* prior-plausible parameters
could produce the data. That makes it blind to misfit that some other parameter
setting happens to mimic: 10% lapses look like a slower, noisier operator. The
posterior predictive check asks the sharper question, using the operator's fitted
posterior:

```python
from relyflow.comparison import posterior_predictive_check
from relyflow.inference import load

result = posterior_predictive_check(load("reliance"), trial_data)  # ~3 s
result.flagged, result.p_value
```

It simulates one replicate dataset from each of 200 posterior draws on the
operator's design, and measures how far the observed statistics lie from the
replicates: the Mahalanobis distance, standardized by the replicates' own
distances.

A posterior predictive p-value on its own is conservative. The posterior was
fitted to the observed data, so the observed operator sits closer to the
replicates than an exchangeable one would, and real misfit is flagged less often
than it should be. relyflow calibrates it (Hjort et al., 2006). It repeats the
whole procedure, a fresh posterior included, for 100 datasets simulated from the
model's prior on the same design, and ranks the observed discrepancy among
theirs. If the model, with prior-plausible parameters, produced the data, the
observed operator is exchangeable with those 100, and the false-alarm rate is α
(to within the rank resolution, 1/101). Like the prior check's, this rate is an
average over prior-plausible operators; it can differ for parameters deep in the
prior's tails. With MCMC this would mean 100 extra posterior fits per operator; with
amortized inference it is one network pass. Pass `n_calibration=0` for the fast
(~0.2 s), conservative version.

The two checks are complementary. The posterior check is far more sensitive to
subtle misfit, such as lapses, fatigue or a different decision process. Gross
departures, such as an operator who copies the aid without looking at the
stimulus, are partly absorbed by the fit and are caught better by the prior
check. To use both with a known overall false-alarm rate, flag an operator when
either p-value is at most α / 2 (Bonferroni): the combined rate is then at most α.

A flag means the posterior and model probabilities for that operator should not
be trusted. Common causes are lapses or contaminant responses, response strategies
outside the models, trends over the session such as fatigue, or a task much
faster or slower than the priors cover. Some misspecification
cannot be detected from one operator's data: an operator who responds at random
with model-like RTs looks like a model operator whose accumulation rates do not
depend on the stimulus.

### Validation

`python -m relyflow.diagnostics compare` checks the classifier's calibration, its
agreement with the exact likelihood, the false-alarm rate on data from the models,
and the detection rate on six deliberately misspecified generators
(`relyflow.diagnostics.misspecification`). Full results are in the
[validation report](validation/comparison/summary.md). In brief:

- **Calibration.** On 3,000 held-out simulations, the most probable model is the
  true one 73% of the time, and stated probabilities match observed frequencies
  (expected calibration error 0.009). Most errors assign a reliance or learning
  operator to the advisory model; those operators have little excitation or
  learning, so the simpler model is a reasonable answer.
- **Agreement with the exact likelihood.** On 60 operators with a mid-session drop
  in aid reliability, the classifier and the importance-sampled marginal
  likelihoods pick the same model 90% of the time. Log Bayes factors correlate
  0.94 (reliance vs. advisory) and 0.80 (learning vs. reliance). Both pick the true
  model about as often (0.87 and 0.88).
- **Misspecification checks.** At alpha = 0.05, with 150 operators per condition
  (600 trials each). Misspecified data are checked against the reliance model.

  | Condition | Prior check | Posterior, uncalibrated | Posterior, calibrated | Combined |
  | --- | --- | --- | --- | --- |
  | Data from the models (false alarms) | 0.04 | 0.00 | 0.04 | 0.03 |
  | 10% lapses | 0.16 | 0.52 | 0.79 | 0.67 |
  | Random responder | 0.57 | 0.55 | 0.69 | 0.65 |
  | Always follows the aid | 0.94 | 0.68 | 0.85 | 0.95 |
  | Diffusion model, advice biases the start point | 0.07 | 0.78 | 0.95 | 0.86 |
  | Fatigue (rates halve over the session) | 0.06 | 0.49 | 0.75 | 0.65 |
  | Slow task (RT + 2 s) | 0.72 | 0.71 | 0.94 | 0.91 |

  Calibration brings the posterior check's false-alarm rate up to its nominal
  level and raises its detection rate by 0.14 to 0.27. On its own it is the most
  sensitive check for every generator except the operator who always follows the
  aid. Combined with the prior check (each at α / 2), it catches 65–95% of every
  generator at a 3% false-alarm rate.

## References

- Hjort, N. L., Dahl, F. A., & Steinbakk, G. H. (2006). Post-processing posterior
  predictive p values. *Journal of the American Statistical Association, 101*(475),
  1157–1174. https://doi.org/10.1198/016214505000001393
- Jeffrey, N., & Wandelt, B. D. (2024). Evidence networks: Simple losses for fast,
  amortized, neural Bayesian model comparison. *Machine Learning: Science and
  Technology, 5*(1), 015008. https://doi.org/10.1088/2632-2153/ad1a4d
- Schmitt, M., Bürkner, P.-C., Köthe, U., & Radev, S. T. (2024). Detecting model
  misspecification in amortized Bayesian inference with neural networks. In
  *DAGM German Conference on Pattern Recognition 2023* (LNCS, pp. 541–557).
  https://doi.org/10.1007/978-3-031-54605-1_35
