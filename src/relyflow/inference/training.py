"""Train amortized posterior networks with BayesFlow.

Run from the command line, for example::

    python -m relyflow.inference train --model reliance --out checkpoints/reliance

The output directory receives ``approximator.keras`` (the trained networks),
``config.json`` (everything needed to reproduce the run) and ``history.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import bayesflow as bf
import keras
import numpy as np

import relyflow
from relyflow.inference.designs import DesignPrior
from relyflow.inference.features import design_features, learning_statistics, network_inputs
from relyflow.models import AutomationModel, get_model

SUMMARY_NETWORKS = ("set_transformer", "deep_set", "time_series")


@dataclass
class TrainingConfig:
    """Everything that determines a training run."""

    model: str
    summary_network: str = "auto"
    """``"set_transformer"``, ``"deep_set"`` or ``"time_series"``. ``"auto"`` uses a
    set transformer for the static models and a deep set for the learning model.
    The learning model's dependence on trial order enters through its history
    features; in pilots this beat a recurrent time-series network on every
    parameter (docs/inference.md)."""
    history_features: bool | None = None
    """Append each trial's aid track record (``features.history_features``) to its
    inputs. ``None`` turns it on for the learning model only."""
    learning_statistics: bool | None = None
    """Pass ``features.learning_statistics`` to the inference network alongside the
    learned summary. ``None`` turns it on for the learning model only."""
    summary_dim: int = 32
    flow_depth: int = 6
    epochs: int = 100
    steps_per_epoch: int = 200
    batch_size: int = 64
    learning_rate: float = 5e-4
    validation_sims: int = 512
    seed: int = 2026
    design: DesignPrior = field(default_factory=DesignPrior)

    def __post_init__(self):
        if isinstance(self.design, dict):
            self.design = DesignPrior(**{k: tuple(v) if isinstance(v, list) else v
                                         for k, v in self.design.items()})  # fmt: skip
        if self.history_features is None:
            self.history_features = self.model == "learning"
        if self.learning_statistics is None:
            self.learning_statistics = self.model == "learning"
        if self.summary_network == "auto":
            self.summary_network = "deep_set" if self.model == "learning" else "set_transformer"
        if self.summary_network not in SUMMARY_NETWORKS:
            raise ValueError(f"summary_network must be one of {SUMMARY_NETWORKS}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> TrainingConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})


class TrainingSimulator(bf.simulators.Simulator):
    """Draws parameters from the prior and designs from the design prior, then simulates.

    Returns each parameter as ``(batch, 1)``, per-trial features as ``"trials"``
    ``(batch, n_trials, n_features)`` and trial counts as ``"design"`` ``(batch, 2)``.
    The trial count varies between batches, not within one.
    """

    def __init__(
        self,
        model: AutomationModel,
        design: DesignPrior,
        seed: int,
        history: bool,
        statistics: bool = False,
    ):
        self.model = model
        self.design = design
        self.history = history
        self.statistics = statistics
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_shape, **kwargs) -> dict[str, np.ndarray]:
        n = int(np.prod(batch_shape))
        params = self.model.sample_prior(n, self.rng)
        stimulus, advice = self.design.sample(n, self.rng)
        choice, rt = self.model.simulate(params, stimulus, advice, self.rng)
        batch = {name: values[:, None] for name, values in params.items()}
        batch["trials"] = network_inputs(stimulus, advice, choice, rt, self.history)
        batch["design"] = design_features(advice)
        if self.statistics:
            batch["statistics"] = learning_statistics(stimulus, advice, choice, rt)
        return batch


def build_adapter(model: AutomationModel, statistics: bool = False) -> bf.Adapter:
    """Map simulator output to network inputs, transforming bounded parameters to the real line."""
    adapter = bf.Adapter().convert_dtype("float64", "float32")
    for parameter in model.parameters:
        lower, upper = parameter.support
        adapter = adapter.constrain(
            parameter.name,
            lower=None if np.isinf(lower) else lower,
            upper=None if np.isinf(upper) else upper,
        )
    adapter = adapter.concatenate(list(model.param_names), into="inference_variables").rename(
        "trials", "summary_variables"
    )
    if statistics:
        return adapter.concatenate(["design", "statistics"], into="inference_conditions")
    return adapter.rename("design", "inference_conditions")


def build_summary_network(config: TrainingConfig) -> bf.networks.SummaryNetwork:
    if config.summary_network == "set_transformer":
        return bf.networks.SetTransformer(summary_dim=config.summary_dim, num_inducing_points=32)
    if config.summary_network == "deep_set":
        return bf.networks.DeepSet(summary_dim=config.summary_dim)
    return bf.networks.TimeSeriesNetwork(summary_dim=config.summary_dim, recurrent_dim=64)


def build_approximator(config: TrainingConfig) -> bf.ContinuousApproximator:
    model = get_model(config.model)
    approximator = bf.ContinuousApproximator(
        inference_network=bf.networks.CouplingFlow(depth=config.flow_depth),
        summary_network=build_summary_network(config),
        adapter=build_adapter(model, config.learning_statistics),
    )
    schedule = keras.optimizers.schedules.CosineDecay(
        config.learning_rate, decay_steps=config.epochs * config.steps_per_epoch
    )
    approximator.compile(optimizer=keras.optimizers.Adam(schedule, clipnorm=1.0))
    return approximator


def model_signature(model: AutomationModel) -> list:
    """Parameter names and priors, used to detect networks trained on a different model."""
    return [
        [p.name, p.prior.dist.name, [float(a) for a in p.prior.args], sorted(p.prior.kwds.items())]
        for p in model.parameters
    ]


def train(config: TrainingConfig, out_dir: str | Path, verbose: int | str = 2):
    """Train an amortized posterior network and save it to ``out_dir``. Returns an ``Amortizer``."""
    from relyflow.inference.amortizer import Amortizer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(config.seed)
    model = get_model(config.model)
    approximator = build_approximator(config)
    adapter = approximator.adapter

    train_data = bf.OnlineDataset(
        TrainingSimulator(
            model, config.design, config.seed, config.history_features, config.learning_statistics
        ),
        batch_size=config.batch_size,
        num_batches=config.steps_per_epoch,
        adapter=adapter,
    )
    validation = TrainingSimulator(
        model, config.design, config.seed + 1, config.history_features, config.learning_statistics
    ).sample(config.validation_sims)
    validation_data = bf.OfflineDataset(
        validation, batch_size=config.batch_size, adapter=adapter, shuffle=False
    )

    start = time.perf_counter()
    history = approximator.fit(
        dataset=train_data,
        epochs=config.epochs,
        validation_data=validation_data,
        verbose=verbose,
    )
    metadata = {
        "relyflow_version": relyflow.__version__,
        "bayesflow_version": bf.__version__,
        "keras_backend": keras.backend.backend(),
        "training_seconds": round(time.perf_counter() - start, 1),
        "model_signature": model_signature(model),
    }
    approximator.save(out_dir / "approximator.keras")
    (out_dir / "config.json").write_text(
        json.dumps({"config": config.to_dict(), "metadata": metadata}, indent=2)
    )
    (out_dir / "history.json").write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in history.history.items()})
    )
    return Amortizer(approximator, config, metadata)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m relyflow.inference train",
        description="Train a relyflow amortized posterior network.",
    )
    parser.add_argument("--model", required=True, choices=["advisory", "reliance", "learning"])
    parser.add_argument("--out", required=True, help="output directory")
    defaults = TrainingConfig(model="reliance")
    for name in ("summary_network", "summary_dim", "flow_depth", "epochs", "steps_per_epoch",
                 "batch_size", "learning_rate", "validation_sims", "seed"):  # fmt: skip
        default = getattr(defaults, name)
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(default), default=None)
    for flag in ("--history-features", "--learning-statistics"):
        parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
    args = vars(parser.parse_args(argv))
    out = args.pop("out")
    config = TrainingConfig(**{k: v for k, v in args.items() if v is not None})
    print(json.dumps(config.to_dict(), indent=2), flush=True)
    train(config, out)
