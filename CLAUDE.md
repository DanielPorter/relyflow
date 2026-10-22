# relyflow

Python package for amortized Bayesian inference (BayesFlow 2 / Keras 3) on cognitive
models of human reliance on automated decision aids. The nine-milestone plan is in
ROADMAP.md; check it to see what stage the project is at.

## Commands

- Environment: `uv venv && uv pip install -e ".[dev,jax]"` (Python 3.12–3.13)
- Tests: `.venv/bin/pytest` (fast); `.venv/bin/pytest -m slow` (training)
- Lint/format: `.venv/bin/ruff check . && .venv/bin/ruff format .`

## Layout

- `src/relyflow/data.py`: `TrialData`, the per-operator trial container (stimulus, advice, choice, rt)
- `src/relyflow/models/`: `AutomationModel`/`Parameter` interface (`base.py`), LBA sampler
  and density (`lba.py`), static advisory/reliance models (`accumulator.py`), delta-rule
  learning extension (`learning.py`)
- `src/relyflow/design.py`: `make_design` for unaided/aided block sequences
- `benchmarks/`: throughput scripts, run directly with Python
- `docs/models/`: model specs with sources; keep them in sync with the priors in code
- `src/relyflow/{inference,diagnostics,comparison,synthetic,reports}/`: one subpackage per roadmap milestone

## Rules

- Never import bayesflow/keras/jax/torch from `relyflow/__init__.py`, `data.py` or `models/`.
  `tests/test_package.py` enforces this.
- Simulators take an `np.random.Generator`, are vectorized over parameter sets, and come
  with qualitative-pattern tests.
- LBA RTs have infinite variance: compare medians/quantiles, per stimulus type, in tests.
- Do not invent model equations or citations. Every model spec cites its source paper,
  and anything relyflow adds beyond the papers (e.g. the learning coupling) is labelled.
