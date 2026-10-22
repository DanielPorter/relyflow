# Contributing to relyflow

## Development setup

You need Python 3.12 or 3.13. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install -e ".[dev,jax]"
pre-commit install
```

With plain pip, create a virtual environment and run `pip install -e ".[dev,jax]"`.
Use the `torch` extra instead of `jax` if you prefer PyTorch, and set `KERAS_BACKEND`
to match.

## Checks

```bash
pytest                  # fast test suite
pytest -m slow          # training and large simulation tests
ruff check . && ruff format --check .
```

CI runs lint and the fast tests on every push and pull request.

## Conventions

- **Simulators are vectorized** over parameter sets and trials. No Python loops over
  operators; a loop over trials is acceptable only where the model is sequential (e.g.
  reliability learning).
- **Randomness is explicit.** Functions take an `np.random.Generator`; nothing uses
  global RNG state.
- **The deep-learning stack loads lazily.** Only `relyflow.inference` and modules that
  need it import BayesFlow or Keras, so the data and simulator layers work without a
  backend.
- **Models are tested on behavior.** Every simulator gets tests for the qualitative
  patterns it should produce, not just for array shapes.
- Every model's parameters, priors and source citations are documented before it is
  implemented.

## Workflow

Work on a branch, open a pull request against `main`, and make sure CI passes.
