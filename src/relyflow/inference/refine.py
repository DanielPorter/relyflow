"""Refine amortized posteriors with the models' exact likelihood.

Every relyflow model has an exact likelihood, so an amortized posterior can be
corrected after the fact:

- :func:`importance_resample` reweights network draws by exact posterior ÷
  network density and resamples them (neural importance sampling). It takes a
  few seconds and reports its effective sample size. A low ESS means the
  network's posterior was far from the exact one for these data.
- :func:`nuts_refine` runs NUTS chains started at network draws (needs the
  ``mcmc`` extra). It is slower but exact as chains converge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from relyflow.data import TrialData
from relyflow.inference.amortizer import Amortizer
from relyflow.inference.features import design_features, learning_statistics, network_inputs


@dataclass
class RefinedPosterior:
    """Refined draws, shape ``(num_samples,)`` per parameter, and how refinement went."""

    draws: dict[str, np.ndarray]
    method: str
    diagnostics: dict[str, float] = field(default_factory=dict)


def network_log_density(amortizer: Amortizer, data: TrialData, draws: dict[str, np.ndarray]):
    """Log-density of parameter draws under the network's posterior for ``data``.

    Equivalent to ``approximator.log_prob`` with the data repeated once per draw,
    but runs the summary network once instead of once per draw (thousands of
    times faster for large draw counts). This uses one private BayesFlow method;
    ``tests/test_refine.py`` checks the result against ``log_prob``.
    """
    import keras

    approximator = amortizer.approximator
    config = amortizer.config
    s, a, c, r = (np.atleast_2d(x) for x in (data.stimulus, data.advice, data.choice, data.rt))
    conditions = {
        "trials": network_inputs(s, a, c, r, config.history_features),
        "design": design_features(a),
    }
    if config.learning_statistics:
        conditions["statistics"] = learning_statistics(s, a, c, r)
    params = {name: np.asarray(values)[:, None] for name, values in draws.items()}
    n = len(next(iter(params.values())))

    adapted_conditions = approximator.adapter(conditions, strict=False, stage="inference")
    adapted_conditions = keras.tree.map_structure(keras.ops.convert_to_tensor, adapted_conditions)
    resolved, _ = approximator._standardize_and_resolve(
        adapted_conditions.get("inference_conditions"),
        adapted_conditions.get("summary_variables"),
        stage="inference",
    )
    resolved = keras.ops.repeat(resolved, n, axis=0)

    adapted, log_det_jac = approximator.adapter(
        params, strict=False, log_det_jac=True, stage="inference"
    )
    variables, log_det_jac_std = approximator.standardizer.maybe_standardize(
        keras.ops.convert_to_tensor(adapted["inference_variables"]),
        key="inference_variables",
        stage="inference",
        log_det_jac=True,
    )
    log_prob = approximator.inference_network.log_prob(variables, conditions=resolved)
    return (
        keras.ops.convert_to_numpy(log_prob)
        + np.asarray(log_det_jac.get("inference_variables", 0.0))
        + keras.ops.convert_to_numpy(log_det_jac_std)
    ).reshape(-1)


def log_posterior(model, data: TrialData, draws: dict[str, np.ndarray]) -> np.ndarray:
    """Unnormalized exact log posterior of each draw."""
    log_prior = sum(p.prior.logpdf(draws[p.name]) for p in model.parameters)
    loglik = model.log_likelihood(draws, data.stimulus, data.advice, data.choice, data.rt)
    return loglik + log_prior


def importance_resample(
    amortizer: Amortizer,
    data: TrialData,
    num_samples: int = 2000,
    num_proposals: int = 8000,
    seed: int | None = None,
) -> RefinedPosterior:
    """Importance-resample network draws toward the exact posterior.

    Weights are truncated at ``mean * sqrt(num_proposals)`` (Ionides, 2008) to
    limit the variance from rare large weights. Diagnostics: ``ess`` (effective
    sample size of the proposals) and ``efficiency`` (ESS / number of proposals).
    """
    rng = np.random.default_rng(seed)
    proposals = amortizer.sample(data, num_samples=num_proposals, seed=seed)
    log_weights = log_posterior(amortizer.model, data, proposals) - network_log_density(
        amortizer, data, proposals
    )
    log_weights = np.where(np.isfinite(log_weights), log_weights, -np.inf)
    weights = np.exp(log_weights - log_weights.max())
    weights = np.minimum(weights, weights.mean() * np.sqrt(num_proposals))
    weights /= weights.sum()
    ess = 1.0 / np.sum(weights**2)
    index = rng.choice(num_proposals, size=num_samples, p=weights)
    return RefinedPosterior(
        draws={name: values[index] for name, values in proposals.items()},
        method="importance",
        diagnostics={"ess": float(ess), "efficiency": float(ess / num_proposals)},
    )


def nuts_refine(
    amortizer: Amortizer,
    data: TrialData,
    num_samples: int = 2000,
    num_chains: int = 4,
    num_warmup: int = 300,
    seed: int = 0,
) -> RefinedPosterior:
    """NUTS started at network draws (``relyflow.mcmc.nuts``). Needs the ``mcmc`` extra."""
    from relyflow.mcmc import nuts

    init = amortizer.sample(data, num_samples=num_chains, seed=seed)
    result = nuts(
        amortizer.model,
        data.stimulus,
        data.advice,
        data.choice,
        data.rt,
        seed=seed,
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=-(-num_samples // num_chains),
        init=init,
    )
    draws = {name: values[:num_samples] for name, values in result.flat().items()}
    return RefinedPosterior(
        draws=draws,
        method="nuts",
        diagnostics={
            "max_rhat": max(result.rhat.values()),
            "min_ess": min(result.ess.values()),
            "divergences": float(result.divergences),
            "seconds": result.seconds,
        },
    )
