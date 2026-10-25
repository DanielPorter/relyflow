"""Amortized model comparison: which mechanism best describes an operator?

A classifier is trained on simulations from all candidate models (advisory,
reliance, learning) with equal prior probability, and returns posterior model
probabilities for an operator's data (an evidence network; Jeffrey & Wandelt,
2024, as implemented in BayesFlow). Inputs match the learning model's: trial
features, each trial's aid track record, and the learning statistics, so that
learning can be told apart from static reliance.

Train with ``python -m relyflow.comparison train --out checkpoints/comparison``.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import bayesflow as bf
import keras
import numpy as np

import relyflow
from relyflow.data import TrialData
from relyflow.inference import pretrained
from relyflow.inference.designs import DesignPrior
from relyflow.inference.features import design_features, learning_statistics, network_inputs
from relyflow.inference.training import model_signature
from relyflow.models import get_model

MODELS = ("advisory", "reliance", "learning")


@dataclass
class ComparisonConfig:
    """Everything that determines a model-comparison training run."""

    models: tuple[str, ...] = MODELS
    summary_dim: int = 32
    epochs: int = 100
    steps_per_epoch: int = 200
    batch_size: int = 64
    learning_rate: float = 5e-4
    validation_sims: int = 512
    seed: int = 2026
    design: DesignPrior = field(default_factory=DesignPrior)

    def __post_init__(self):
        self.models = tuple(self.models)
        if isinstance(self.design, dict):
            self.design = DesignPrior(**{k: tuple(v) if isinstance(v, list) else v
                                         for k, v in self.design.items()})  # fmt: skip

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> ComparisonConfig:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in known})


def comparison_inputs(stimulus, advice, choice, rt) -> dict[str, np.ndarray]:
    """Network inputs for a batch of datasets sharing a trial count."""
    stimulus, advice, choice, rt = (np.atleast_2d(x) for x in (stimulus, advice, choice, rt))
    return {
        "trials": network_inputs(stimulus, advice, choice, rt, history=True),
        "design": design_features(advice),
        "statistics": learning_statistics(stimulus, advice, choice, rt),
    }


class ComparisonSimulator(bf.simulators.Simulator):
    """Simulates operators from a uniform mixture of the candidate models.

    Every dataset in a batch shares one block layout (so arrays stack); each
    dataset's model is drawn independently. ``"model_indices"`` is one-hot.
    """

    def __init__(self, models: tuple[str, ...], design: DesignPrior, seed: int):
        self.models = [get_model(name) for name in models]
        self.design = design
        self.rng = np.random.default_rng(seed)

    def simulate(self, n: int) -> tuple[np.ndarray, ...]:
        """Model index, stimulus, advice, choice and RT for ``n`` operators."""
        stimulus, advice = self.design.sample(n, self.rng)
        index = self.rng.integers(0, len(self.models), size=n)
        choice = np.empty(stimulus.shape, dtype=np.int8)
        rt = np.empty(stimulus.shape)
        for k, model in enumerate(self.models):
            chosen = index == k
            if chosen.any():
                params = model.sample_prior(int(chosen.sum()), self.rng)
                choice[chosen], rt[chosen] = model.simulate(
                    params, stimulus[chosen], advice[chosen], self.rng
                )
        return index, stimulus, advice, choice, rt

    def sample(self, batch_shape, **kwargs) -> dict[str, np.ndarray]:
        index, stimulus, advice, choice, rt = self.simulate(int(np.prod(batch_shape)))
        batch = comparison_inputs(stimulus, advice, choice, rt)
        batch["model_indices"] = np.eye(len(self.models), dtype=np.float32)[index]
        return batch


def build_adapter() -> bf.Adapter:
    return (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .rename("model_indices", "inference_variables")
        .rename("trials", "summary_variables")
        .concatenate(["design", "statistics"], into="inference_conditions")
    )


def build_approximator(config: ComparisonConfig) -> bf.ModelComparisonApproximator:
    approximator = bf.ModelComparisonApproximator(
        inference_network=bf.networks.ScoringRuleNetwork(
            scoring_rules={"cross_entropy": bf.scoring_rules.CrossEntropyScore()}
        ),
        summary_network=bf.networks.DeepSet(summary_dim=config.summary_dim),
        adapter=build_adapter(),
    )
    schedule = keras.optimizers.schedules.CosineDecay(
        config.learning_rate, decay_steps=config.epochs * config.steps_per_epoch
    )
    approximator.compile(optimizer=keras.optimizers.Adam(schedule, clipnorm=1.0))
    return approximator


class ModelComparison:
    """A trained model-comparison network. Load with :meth:`load`."""

    def __init__(self, approximator, config: ComparisonConfig, metadata: dict):
        self.approximator = approximator
        self.config = config
        self.metadata = metadata
        self.models = config.models

    @classmethod
    def load(cls, name_or_path: str | Path = pretrained.COMPARISON) -> ModelComparison:
        """Load the pretrained network (``"comparison"``, the default; downloaded on first
        use, see ``relyflow.inference.pretrained``) or a directory saved by :func:`train`."""
        if name_or_path == pretrained.COMPARISON:
            name_or_path = pretrained.fetch(pretrained.COMPARISON)
        directory = Path(name_or_path)
        saved = json.loads((directory / "config.json").read_text())
        config = ComparisonConfig.from_dict(saved["config"])
        signatures = json.loads(
            json.dumps({name: model_signature(get_model(name)) for name in config.models})
        )
        changed = [
            name
            for name in config.models
            if saved["metadata"]["model_signatures"].get(name) != signatures[name]
        ]
        if changed:
            warnings.warn(
                f"The parameters or priors of {changed} have changed since this "
                "model-comparison network was trained. Its probabilities are for the old "
                "models; retrain it.",
                stacklevel=2,
            )
        approximator = keras.saving.load_model(directory / "classifier.keras")
        return cls(approximator, config, saved["metadata"])

    def estimate_batch(self, stimulus, advice, choice, rt) -> dict[str, np.ndarray]:
        """Posterior model probabilities ``(n, n_models)``, log Bayes factors relative to
        the first model, and learned summaries ``(n, summary_dim)``."""
        prior = np.full(len(self.models), 1.0 / len(self.models))
        result = self.approximator.estimate(
            conditions=comparison_inputs(stimulus, advice, choice, rt),
            return_summaries=True,
            model_prior=prior,
        )
        return {
            "model_probs": result["model_probs"],
            "log_bayes_factors": result["log_bayes_factors"],
            "summaries": result["_summaries"],
        }

    def probabilities(self, data: TrialData) -> dict[str, float]:
        """Posterior probability of each model for one operator (equal prior odds)."""
        probs = self.estimate_batch(data.stimulus, data.advice, data.choice, data.rt)
        return dict(zip(self.models, probs["model_probs"][0].tolist(), strict=True))


def train(config: ComparisonConfig, out_dir: str | Path, verbose: int | str = 2) -> ModelComparison:
    """Train the model-comparison network and save it to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keras.utils.set_random_seed(config.seed)
    approximator = build_approximator(config)
    adapter = approximator.adapter
    train_data = bf.OnlineDataset(
        ComparisonSimulator(config.models, config.design, config.seed),
        batch_size=config.batch_size,
        num_batches=config.steps_per_epoch,
        adapter=adapter,
    )
    validation = ComparisonSimulator(config.models, config.design, config.seed + 1).sample(
        config.validation_sims
    )
    validation_data = bf.OfflineDataset(
        validation, batch_size=config.batch_size, adapter=adapter, shuffle=False
    )
    start = time.perf_counter()
    history = approximator.fit(
        dataset=train_data, epochs=config.epochs, validation_data=validation_data, verbose=verbose
    )
    metadata = {
        "relyflow_version": relyflow.__version__,
        "bayesflow_version": bf.__version__,
        "keras_backend": keras.backend.backend(),
        "training_seconds": round(time.perf_counter() - start, 1),
        "model_signatures": {name: model_signature(get_model(name)) for name in config.models},
    }
    approximator.save(out_dir / "classifier.keras")
    (out_dir / "config.json").write_text(
        json.dumps({"config": config.to_dict(), "metadata": metadata}, indent=2) + "\n"
    )
    (out_dir / "history.json").write_text(
        json.dumps({k: [float(v) for v in vals] for k, vals in history.history.items()}) + "\n"
    )
    return ModelComparison(approximator, config, metadata)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m relyflow.comparison train",
        description="Train the relyflow model-comparison network.",
    )
    parser.add_argument("--out", required=True)
    defaults = ComparisonConfig()
    for name in ("summary_dim", "epochs", "steps_per_epoch", "batch_size", "learning_rate",
                 "validation_sims", "seed"):  # fmt: skip
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(getattr(defaults, name)))
    args = vars(parser.parse_args(argv))
    out = args.pop("out")
    config = ComparisonConfig(**{k: v for k, v in args.items() if v is not None})
    print(json.dumps(config.to_dict(), indent=2), flush=True)
    train(config, out)
