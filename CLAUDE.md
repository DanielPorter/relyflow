# relyflow

Python package for amortized Bayesian inference (BayesFlow 2 / Keras 3) on cognitive
models of human reliance on automated decision aids. The nine-milestone plan is in
ROADMAP.md; check it to see what stage the project is at.

## Commands

- Environment: `uv venv && uv pip install -e ".[dev,jax,mcmc]"` (Python 3.12–3.13)
- Tests: `.venv/bin/pytest` (fast; slow tests deselected by default); `.venv/bin/pytest -m slow`
- Train: `python -m relyflow.inference train --model <name> --out checkpoints/<name>`
- Validate: `python -m relyflow.diagnostics validate checkpoints/<name> --out docs/validation/<name>`
- Model comparison: `KERAS_BACKEND=torch python -m relyflow.comparison train --out checkpoints/comparison`,
  then `python -m relyflow.diagnostics compare checkpoints/comparison --out docs/validation/comparison --amortizers checkpoints`
  (about 10–40 min per model; run in the background under `caffeinate -i`)
- Lint/format: `.venv/bin/ruff check . && .venv/bin/ruff format .`

## Layout

- `src/relyflow/data.py`: `TrialData`, the per-operator trial container (stimulus, advice, choice, rt)
- `src/relyflow/models/`: `AutomationModel`/`Parameter` interface (`base.py`), LBA sampler
  and density (`lba.py`), static advisory/reliance models (`accumulator.py`), delta-rule
  learning extension (`learning.py`)
- `src/relyflow/design.py`: `make_design` for unaided/aided block sequences
- `src/relyflow/inference/`: BayesFlow training (`training.py`), network inputs (`features.py`,
  NumPy only), training design prior (`designs.py`), loading/sampling (`amortizer.py`),
  release downloads (`pretrained.py`), exact-likelihood refinement (`refine.py`)
- `src/relyflow/mcmc.py`: NUTS (NumPyro) on a JAX twin of the likelihood; must match
  `models/` exactly (`tests/test_mcmc.py`)
- `src/relyflow/diagnostics/`: recovery/SBC (`calibration.py`), posterior predictive checks
  (`ppc.py`), NUTS benchmark (`benchmark.py`, workers in `relyflow/_workers.py`), the
  `validate` orchestrator; reports in `docs/validation/`; model-comparison validation
  (`comparison.py`) with misspecified generators (`misspecification.py`)
- `src/relyflow/comparison/`: model-comparison classifier (`classifier.py`), prior and
  posterior predictive misspecification checks (`ood.py`)
- `checkpoints/`: local training outputs (git-ignored); set `RELYFLOW_WEIGHTS_DIR` to use them
- `benchmarks/`: throughput scripts, run directly with Python
- `docs/models/`: model specs with sources; keep them in sync with the priors in code
- `src/relyflow/{inference,diagnostics,comparison,synthetic,reports}/`: one subpackage per roadmap milestone

## Rules

- Never import bayesflow/keras/jax/torch from `relyflow/__init__.py`, `data.py` or `models/`.
  `tests/test_package.py` enforces this.
- Simulators take an `np.random.Generator`, are vectorized over parameter sets, and come
  with qualitative-pattern tests.
- LBA RTs have infinite variance: compare medians/quantiles, per stimulus type, in tests.
- Train set encoders with `KERAS_BACKEND=torch` (Apple GPU), recurrent nets with `jax`
  (CPU). Never run two GPU (MPS) jobs at once: it caused a Metal internal error.
- Changing a model's priors or a network's inputs requires retraining; loading warns
  when a network's saved prior signature no longer matches the code.
- Any change to the likelihood in `models/` must be mirrored in `mcmc.py` (JAX).
- Do not invent model equations or citations. Every model spec cites its source paper,
  and anything relyflow adds beyond the papers (e.g. the learning coupling) is labelled.
