# Amortized inference

Code: `relyflow/inference/`. relyflow trains one neural posterior estimator per
model on simulated data ([BayesFlow 2](https://bayesflow.org)). After that, an
operator's posterior takes about a second instead of an MCMC run.

## Pipeline

1. **Simulate.** `TrainingSimulator` draws parameters from the model's prior and a
   design from the training design prior (below), then simulates the operator.
2. **Featurize.** Each trial becomes a vector of the 7 values in `TRIAL_FEATURES`:
   stimulus, choice, correct, aided, aid correct, followed advice, and log RT. Log
   RTs are clipped to [0.05 s, 30 s], because the LBA's power-law RT tail carries
   no information beyond that. The trial counts (log unaided, log aided) are passed
   separately, because pooling layers discard how many trials there were, and that
   sets how narrow the posterior should be. The same functions
   (`relyflow.inference.features`) prepare real data, so training and fitting
   preprocess identically.
   For the learning model, each trial also carries **history features**: the
   aid's track record as the operator saw it before that trial
   (`features.history_features`). These are the outcome of the previous aided trial,
   log(1 + aided trials so far), and exponentially weighted aid accuracy at rates
   0.01, 0.03, 0.1 and 0.3. In the learning model, perceived reliability *is* such a
   weighted average, for an unknown rate and starting point, and the track record
   depends only on the design, so it is known exactly. These features are
   controlled by `TrainingConfig.history_features`, which is on by default only for
   the learning model.
3. **Summarize.** A summary network compresses all trials into a fixed-length
   vector. Both are permutation-invariant set networks:
   - Static models use a **set transformer** with 32 inducing points. Their
     likelihood does not depend on trial order.
   - The learning model uses a **deep set** over trial and history features. Its
     summary is still sequence-aware: reordering the trials changes each trial's
     history features. Pilots (below) favored this over a recurrent network.
4. **Estimate.** An affine coupling flow learns the posterior. It is conditioned on
   the summary, the trial counts and, for the learning model, 14 learning
   statistics (see below). Bounded parameters are mapped to the real line
   with the bounds of their priors, so posterior draws always respect the prior's
   support.

## Supported designs

A network is only trustworthy for designs like those it was trained on. The
default `DesignPrior` covers:

| Aspect | Range |
| --- | --- |
| Total trials | 120–1,200, in steps of 60 during training (any count works when fitting) |
| Unaided share | 20–50% of trials |
| Block layout | 1–2 unaided and 1–3 aided blocks, in random order and of random sizes |
| Aid reliability | 55–98% per session; in half of sessions each aided block has its own reliability, so the aid can change mid-session |
| Stimulus base rate | 50% |

`Amortizer.sample` warns when data have no unaided trials or a trial count outside
this range. Milestone 5 adds a full out-of-distribution check.

Training uses a grid of trial counts. JAX and PyTorch/MPS recompile for every new
input shape, and drawing the trial count anew for each batch made JAX training
3.3× slower.

## The learning model: getting at the learning rate

The learning model was the hard case. Each change below was tested on 300–500
simulated operators, each with 200 unaided trials, then 300 trials with an 85–98%
aid, then 300 with a 55–75% aid (`benchmarks/recovery.py`).

**1. Summary network.** Three 3,000-step pilots compared encoders. The table gives
R² of posterior means:

| Encoder | Basic LBA parameters | inhibition | excitation | `r0`, `alpha` |
| --- | --- | --- | --- | --- |
| Recurrent (conv + GRU), trial features only | 0.09–0.68 | 0.01 | 0.08 | ≈ 0 |
| Recurrent, + history features | 0.20–0.57 | 0.05 | 0.34 | ≈ 0 |
| Deep set, + history features | 0.54–0.84 | 0.06 | 0.36 | ≈ 0 |

The recurrent network has to tally hundreds of trials step by step, and it
struggled even with the basic parameters. The deep set recovered those as well as
the static models' set transformer did. All pilots were calibrated, with 90%
intervals covering the truth 87–95% of the time.

**2. Longer training.** At 30,000 steps the deep set recovered inhibition (0.76)
and excitation (0.78) well, but still not `r0` or `alpha` (R² ≈ 0, posteriors ≈
prior).

**3. Is the information there?** The likelihood is exact, so a near-exact
posterior for (`alpha`, `r0`) can be computed on a grid, given the network's draws
of the other 12 parameters (`benchmarks/learning_reference.py`). On 120 operators:

| | log `alpha` R² (contraction) | `r0` R² (contraction) |
| --- | --- | --- |
| Near-exact posterior | 0.56 (0.50) | 0.18 (0.21) |
| Network, 30,000 steps | 0.02 | 0.00 |

So the data carry real information about the learning rate, and modest information
about the starting belief. The network was missing it: the signal is spread thinly
over many trials, as small relationships between behavior and the aid's recent
track record.

**4. Learning statistics.** `features.learning_statistics` computes 14 such
relationships directly:

- For each history rate, correlations between the aid's track record and following
  the advice (separately when the aid was right and wrong) and log RT.
- The change in following and in log RT right after an aid failure, relative to
  right after a success.

They are passed to the inference network alongside the learned summary
(`TrainingConfig.learning_statistics`, on by default only for the learning model).
On their own, in a quadratic regression, they explain 28% of the variance in log
`alpha` on this design. In a 3,000-step pilot they lifted `alpha` from R² 0.02 to
0.14, inhibition from 0.06 to 0.23 and excitation from 0.36 to 0.48.

**5. Result.** After 30,000 steps with the statistics, on the same 120 operators as
step 3:

| | log `alpha` R² | `r0` R² |
| --- | --- | --- |
| Near-exact posterior | 0.58 | 0.19 |
| Network with learning statistics | 0.28 | −0.10 |
| Network without them | 0.02 | 0.00 |

The network now captures about half of the information about the learning rate
that the data carry. The starting belief `r0` is barely identifiable with these
designs even in principle. Closing the remaining gap is planned for milestone 4:
the learning model's exact likelihood allows refining the network's draws of
(`alpha`, `r0`), as `benchmarks/learning_reference.py` already does at about 5 s
per operator.

## Training

```bash
KERAS_BACKEND=torch python -m relyflow.inference train --model reliance --out checkpoints/reliance
KERAS_BACKEND=jax python -m relyflow.inference train --model learning --out checkpoints/learning
```

Every `TrainingConfig` field can be set from the command line (e.g. `--epochs`,
`--steps-per-epoch`, `--batch-size`, `--summary-dim`). Training is online: every
batch is fresh simulations. The optimizer is Adam with cosine learning-rate decay
and gradient clipping. A fixed set of 512 validation simulations tracks
overfitting.

The output directory holds:

- `approximator.keras`: the trained networks, loadable on any Keras backend.
- `config.json`: the full `TrainingConfig` plus metadata (versions, backend,
  training time, and the model's parameter and prior signature).
- `history.json`: losses per epoch.

**Backend choice.** On an Apple M2 Pro:

- Set transformers train about 4× faster on PyTorch with the Apple GPU (MPS) than
  on JAX/CPU.
- The recurrent time-series network is about 6× faster on JAX/CPU than on MPS.

Saved networks load on either backend (tested both ways).

## Fitting

```python
from relyflow.inference import load

amortizer = load("reliance")  # pretrained network, downloaded and cached on first use
draws = amortizer.sample(trial_data, num_samples=2000)  # dict: parameter -> (2000,)
```

`load` also accepts the path of a training output directory.
`Amortizer.sample_batch` handles many operators at once, provided they have the
same number of trials. Loading warns if the model's parameters or priors have
changed since the network was trained, because such a network's posteriors belong
to the old model.

## Refining with the exact likelihood

Every relyflow model has an exact likelihood, so an amortized posterior can be
corrected after the fact (`relyflow.inference.refine`):

```python
from relyflow.inference.refine import importance_resample, nuts_refine

refined = importance_resample(amortizer, trial_data)  # a few seconds
refined.draws, refined.diagnostics  # {"ess": ..., "efficiency": ...}
refined = nuts_refine(amortizer, trial_data)  # NUTS from network draws; ~1-2 min
```

- **Importance resampling** weights 8,000 network draws by exact posterior ÷
  network density and resamples them. The network density is evaluated with one
  pass of the summary network, which takes 3 s instead of 43 s for the learning
  model. Low efficiency (ESS / proposals) means the network's posterior was far
  from the exact one for these data. Median efficiency is about 20% for the
  reliance network and about 5% for the learning network.
- **NUTS refinement** (`mcmc` extra) starts NUTS chains at network draws and
  reports R̂, ESS and divergences.

How well each option agrees with full NUTS is measured in the validation suite
([validation/](validation/README.md)).

## Pretrained networks

Pretrained networks are published as assets of versioned GitHub releases
(`relyflow.inference.pretrained`). The download happens once, is checked against a
SHA-256 hash recorded in the package, and is cached in `~/.cache/relyflow`
(override with `RELYFLOW_CACHE`). Set `RELYFLOW_WEIGHTS_DIR` to a directory of
training outputs (`<dir>/<model>/approximator.keras`) to use local networks
instead, for example offline or while the repository is private.

## Results

Recovery was measured on 500 new simulated operators per model
(`benchmarks/recovery.py`):

- Static models: 200 unaided trials, then 400 with a 60–98%-reliable aid.
- Learning model: 200 unaided, then 300 with an 85–98% aid, then 300 with a 55–75%
  aid.

Values are R² of posterior means, posterior contraction (1 − posterior variance /
prior variance), and the share of 90% intervals containing the true value.

| Model | Training | R² | Contraction | 90% coverage |
| --- | --- | --- | --- | --- |
| Advisory | 20,000 steps, 122 min (PyTorch/MPS) | 0.60–0.90 | 0.57–0.89 | 0.90–0.93 |
| Reliance | 20,000 steps, 109 min (PyTorch/MPS) | 0.66–0.90 | 0.63–0.88 | 0.90–0.95 |
| Learning, 12 non-learning parameters | 30,000 steps, 131 min (PyTorch/MPS) | 0.72–0.90 | 0.66–0.89 | 0.89–0.93 |
| Learning, `alpha` / `r0` | | 0.33 / −0.05 | 0.25 / −0.08 | 0.86 / 0.91 |

- The weakest non-learning parameters are `t0` and `A`, which trade off in LBA
  models.
- Inhibition (advisory 0.90, reliance 0.85, learning 0.73) and excitation
  (reliance 0.90, learning 0.81) are recovered well, which the advising vs.
  reliance comparison depends on.
- The learning model's `alpha` intervals are slightly too narrow (86% coverage);
  milestone 4's simulation-based calibration will check this properly.
- One operator's posterior (2,000 draws) takes 0.03 s for every model, and 500
  operators take under 3 s.

Full simulation-based calibration, posterior predictive checks and the NUTS
benchmark are in [validation/](validation/README.md). In short, the raw networks
are calibrated to conservative, and importance-refined posteriors match NUTS.
