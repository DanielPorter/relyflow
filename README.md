# relyflow

**Amortized Bayesian inference for cognitive models of how people use automated decision aids.**

> **Status: pre-alpha.** The trial-data layer and the model specifications
> ([docs/models/](docs/models/)) are done. Everything else below is the target design.
> Progress is tracked in [ROADMAP.md](ROADMAP.md).

relyflow fits evidence-accumulation models of automation use to one operator's
trial-by-trial data in seconds. It does this with neural networks trained once on
simulated data ([BayesFlow](https://bayesflow.org)) instead of running MCMC for every
person. It is built to answer three questions:

- **Mechanism.** Does this operator treat the aid as an *advisor*, where advice shifts
  their evidence accumulation, or *rely* on it, handing the decision over?
- **Calibration.** Does their reliance track the aid's actual reliability, or are they
  over-relying (misuse) or under-relying (disuse)?
- **Design.** How would a population of operators perform with an aid that is 70%,
  85% or 95% reliable, before running a human study?

## Planned capabilities

| Area | What it provides |
| --- | --- |
| Models | Static advisory and reliance accumulator models (Strickland et al., 2021, 2023) and a reliability-learning (delta-rule) extension, with documented parameters and priors |
| Simulators | Vectorized simulators, validated against known qualitative patterns |
| Inference | Pretrained amortized posterior networks: a set encoder for static models, a sequence encoder for learning models |
| Validation | Parameter recovery, simulation-based calibration, posterior predictive checks and an MCMC benchmark |
| Model checks | Amortized model comparison (advising vs. reliance) and out-of-distribution flags |
| Reports | Per-operator posteriors plus a calibration verdict: over-relying, under-relying or calibrated |
| Synthetic operators | Simulated operator populations run against aids of different reliability |

## Data format

Each operator contributes one row per trial, **in presentation order**:

| Column | Meaning |
| --- | --- |
| `stimulus` | True state of the world (0/1) |
| `advice` | The aid's recommendation (0/1), or empty on unaided trials |
| `choice` | The operator's response (0/1) |
| `rt` | Response time in seconds |

Whether the aid was correct is derived as `advice == stimulus`. Include some
unaided (manual) trials: the models measure the aid's influence relative to them.

```python
import pandas as pd
import relyflow

df = pd.read_csv("study.csv")
operators = relyflow.split_operators(df, "participant", order_col="trial", rt="rt_seconds")
operators["P07"].summary()
# {'aid_reliability': 0.85, 'accuracy': 0.91, 'misuse_rate': 0.42, 'disuse_rate': 0.04, ...}
```

## Installation

relyflow is not on PyPI yet. To set up a development install:

```bash
git clone https://github.com/DanielPorter/relyflow.git
cd relyflow
uv venv && uv pip install -e ".[dev,jax]"
```

BayesFlow runs on Keras 3, which needs a backend. Install the `jax` or `torch` extra
and set `KERAS_BACKEND` to match. Python 3.12 or 3.13 is required.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
