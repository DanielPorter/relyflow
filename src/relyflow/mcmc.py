"""Reference posteriors by NUTS, with a JAX version of the models' exact likelihood.

Used as the gold standard in the MCMC benchmark (milestone 4) and to refine
amortized posteriors (``relyflow.inference.refine``). Requires the ``mcmc``
extra (JAX and NumPyro). Computations run in float64 inside
``jax.enable_x64``, so other JAX users in the same process (such as Keras) are
unaffected.

The likelihood mirrors ``relyflow.models``: the same rate, SD and threshold
mappings and the same LBA density, including the quadrature used far in the
truncation tail. ``tests/test_mcmc.py`` checks that both agree.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import log_ndtr, ndtr
from jax.scipy.stats import beta as jbeta
from jax.scipy.stats import norm as jnorm
from scipy.special import ndtr as np_ndtr

from relyflow.data import NO_ADVICE
from relyflow.models.base import AutomationModel

_SQRT_2PI = np.sqrt(2.0 * np.pi)
_TAIL_Z = -5.0
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(32)
_TINY = np.finfo(np.float64).tiny


@dataclass
class MCMCResult:
    """Draws of shape ``(n_chains, n_samples)`` per parameter, with diagnostics."""

    draws: dict[str, np.ndarray]
    rhat: dict[str, float]
    ess: dict[str, float]
    divergences: int
    seconds: float

    def flat(self) -> dict[str, np.ndarray]:
        """Draws pooled across chains."""
        return {name: values.reshape(-1) for name, values in self.draws.items()}

    @property
    def converged(self) -> bool:
        return max(self.rhat.values()) < 1.01 and min(self.ess.values()) > 400


# --- Likelihood --------------------------------------------------------------------


def _accumulator_pdf_cdf(t, v, s, b, A):
    """JAX twin of ``relyflow.models.lba.accumulator_pdf_cdf``.

    Both branches are computed and selected with ``where``; inputs to the
    unselected branch are masked to safe values so gradients stay finite.
    """
    positive = t > 0
    t_safe = jnp.where(positive, t, 1.0)
    tail = v / s < _TAIL_Z

    v_c = jnp.where(tail, 0.0, v)
    ts, tv = t_safe * s, t_safe * v_c
    z_lo, z_hi = (b - A - tv) / ts, (b - tv) / ts
    phi_lo = jnp.exp(-0.5 * z_lo**2) / _SQRT_2PI
    phi_hi = jnp.exp(-0.5 * z_hi**2) / _SQRT_2PI
    pdf_c = (-v_c * ndtr(z_lo) + s * phi_lo + v_c * ndtr(z_hi) - s * phi_hi) / A
    cdf_c = (
        1.0 + (b - A - tv) / A * ndtr(z_lo) - (b - tv) / A * ndtr(z_hi) + ts / A * (phi_lo - phi_hi)
    )
    mass = ndtr(v_c / s)
    pdf_c = jnp.clip(pdf_c, 0.0, None) / mass
    cdf_c = jnp.clip(cdf_c / mass, 0.0, 1.0)

    pdf_c = jnp.where(positive, pdf_c, 0.0)
    cdf_c = jnp.where(positive, cdf_c, 0.0)

    def with_tail(_):
        x = 0.5 * A[..., None] * (1.0 + _GL_NODES)
        c = (b[..., None] - x) / t_safe[..., None]
        z = (c - v[..., None]) / s[..., None]
        log_mass = log_ndtr(v / s)[..., None]
        log_g = -0.5 * z**2 - jnp.log(_SQRT_2PI * s[..., None]) - log_mass
        pdf_q = 0.5 * jnp.sum(_GL_WEIGHTS * jnp.exp(log_g) * c / t_safe[..., None], axis=-1)
        cdf_q = 0.5 * jnp.sum(_GL_WEIGHTS * jnp.exp(log_ndtr(-z) - log_mass), axis=-1)
        use = positive & tail
        return jnp.where(use, pdf_q, pdf_c), jnp.where(use, jnp.clip(cdf_q, 0.0, 1.0), cdf_c)

    # The quadrature is only needed (and only paid for) when some entry is in the tail.
    return jax.lax.cond(jnp.any(positive & tail), with_tail, lambda _: (pdf_c, cdf_c), None)


def _beliefs(r0, alpha, aided, aid_correct):
    """Delta-rule perceived reliability before each trial (``models.learning``)."""

    def step(r, inputs):
        was_aided, correct = inputs
        return jnp.where(was_aided, r + alpha * (correct - r), r), r

    _, beliefs = jax.lax.scan(step, r0, (aided, aid_correct))
    return beliefs


def make_log_likelihood(model: AutomationModel, stimulus, advice, choice, rt):
    """A JAX function of a parameter dict returning one operator's log-likelihood.

    Call inside ``jax.enable_x64(True)``.
    """
    stimulus, advice, choice = (
        jnp.asarray(np.asarray(x), dtype=jnp.int32) for x in (stimulus, advice, choice)
    )
    rt = jnp.asarray(np.asarray(rt), dtype=jnp.float64)
    aided = advice != NO_ADVICE
    aid_correct = (advice == stimulus).astype(jnp.float64)
    accumulators = jnp.arange(2)
    is_match = stimulus[:, None] == accumulators
    agrees = advice[:, None] == accumulators
    won = choice[:, None] == accumulators
    stim0 = stimulus == 0
    learning = model.name == "learning"

    def log_likelihood(p):
        match = jnp.where(stim0, p["v_match_0"], p["v_match_1"])
        mismatch = jnp.where(stim0, p["v_mismatch_0"], p["v_mismatch_1"])
        means = jnp.where(is_match, match[:, None], mismatch[:, None])
        excitation = jnp.asarray(p.get("excitation", 0.0))
        inhibition = jnp.asarray(p["inhibition"])
        if learning:
            beliefs = _beliefs(p["r0"], p["alpha"], aided, aid_correct)
            weight = jnp.clip(2.0 * beliefs - 1.0, 0.0, 1.0)
            excitation, inhibition = excitation * weight, inhibition * weight
        effect = jnp.where(
            agrees,
            jnp.broadcast_to(excitation, stimulus.shape)[:, None],
            -jnp.broadcast_to(inhibition, stimulus.shape)[:, None],
        )
        means = means + jnp.where(aided[:, None], effect, 0.0)
        sds = jnp.where(is_match, p["sv_match"], 1.0)
        gaps = jnp.stack([p["B_0"], p["B_1"]])
        scale = jnp.where(aided, p["B_aided_scale"], 1.0)
        bounds = p["A"] + gaps[None, :] * scale[:, None]
        A = jnp.broadcast_to(p["A"], means.shape)
        decision = jnp.broadcast_to((rt - p["t0"])[:, None], means.shape)
        pdf, cdf = _accumulator_pdf_cdf(decision, means, sds, bounds, A)
        density = jnp.prod(jnp.where(won, pdf, 1.0 - cdf), axis=-1)
        density = jnp.where(jnp.isfinite(density) & (density > _TINY), density, _TINY)
        return jnp.sum(jnp.log(density))

    return log_likelihood


# --- Priors and transforms -------------------------------------------------------------


def _prior_logpdf(prior):
    """JAX log-density of one of the scipy prior families used by relyflow models."""
    name, args, kwds = prior.dist.name, prior.args, prior.kwds
    loc, scale = kwds.get("loc", 0.0), kwds.get("scale", 1.0)
    if name == "truncnorm":
        a, b = args
        log_norm = float(np.log(np_ndtr(b) - np_ndtr(a)))
        return lambda x: jnorm.logpdf((x - loc) / scale) - np.log(scale) - log_norm
    if name == "halfnorm":
        return lambda x: np.log(2.0) + jnorm.logpdf((x - loc) / scale) - np.log(scale)
    if name == "lognorm":
        s = args[0] if args else kwds["s"]
        return lambda x: (
            -jnp.log((x - loc) * s * _SQRT_2PI) - jnp.log((x - loc) / scale) ** 2 / (2 * s**2)
        )
    if name == "beta":
        a, b = args if args else (kwds["a"], kwds["b"])
        return lambda x: jbeta.logpdf(x, a, b, loc, scale)
    raise NotImplementedError(f"no JAX log-density for prior family {name!r}")


class _Space:
    """Maps parameters to and from the real line, with log-Jacobians."""

    def __init__(self, model: AutomationModel):
        self.names = list(model.param_names)
        self.bounds = [p.support for p in model.parameters]
        self.logpdfs = [_prior_logpdf(p.prior) for p in model.parameters]

    def constrain(self, u):
        params, log_jac = {}, 0.0
        for j, (name, (lo, hi)) in enumerate(zip(self.names, self.bounds, strict=True)):
            if np.isfinite(lo) and np.isfinite(hi):
                params[name] = lo + (hi - lo) * jax.nn.sigmoid(u[j])
                log_jac += np.log(hi - lo) + jax.nn.log_sigmoid(u[j]) + jax.nn.log_sigmoid(-u[j])
            elif np.isfinite(lo):
                params[name] = lo + jnp.exp(u[j])
                log_jac += u[j]
            elif np.isfinite(hi):
                params[name] = hi - jnp.exp(u[j])
                log_jac += u[j]
            else:
                params[name] = u[j]
        return params, log_jac

    def unconstrain(self, x: np.ndarray) -> np.ndarray:
        """``x`` of shape ``(..., n_params)`` in the original space."""
        u = np.array(x, dtype=np.float64)
        for j, (lo, hi) in enumerate(self.bounds):
            if np.isfinite(lo) and np.isfinite(hi):
                q = np.clip((u[..., j] - lo) / (hi - lo), 1e-9, 1 - 1e-9)
                u[..., j] = np.log(q) - np.log1p(-q)
            elif np.isfinite(lo):
                u[..., j] = np.log(np.maximum(u[..., j] - lo, 1e-12))
            elif np.isfinite(hi):
                u[..., j] = np.log(np.maximum(hi - u[..., j], 1e-12))
        return u

    def log_prior(self, params):
        return sum(f(params[name]) for name, f in zip(self.names, self.logpdfs, strict=True))


# --- NUTS --------------------------------------------------------------------------------


def nuts(
    model: AutomationModel,
    stimulus,
    advice,
    choice,
    rt,
    *,
    seed: int = 0,
    num_chains: int = 4,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    init: dict[str, np.ndarray] | None = None,
    target_accept: float = 0.9,
) -> MCMCResult:
    """Sample one operator's posterior with NUTS (NumPyro).

    ``init`` gives each chain's starting values, shape ``(num_chains,)`` per
    parameter (for example amortized posterior draws); by default chains start
    at prior draws.
    """
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    from numpyro.infer import MCMC, NUTS

    start = time.perf_counter()
    space = _Space(model)
    if init is None:
        init = model.sample_prior(num_chains, np.random.default_rng(seed))
    init_x = np.column_stack([np.asarray(init[name])[:num_chains] for name in space.names])

    with jax.enable_x64(True):
        log_likelihood = make_log_likelihood(model, stimulus, advice, choice, rt)

        def potential(u):
            params, log_jac = space.constrain(u)
            log_post = log_likelihood(params) + space.log_prior(params) + log_jac
            return -jnp.where(jnp.isfinite(log_post), log_post, -1e300)

        mcmc = MCMC(
            NUTS(potential_fn=potential, target_accept_prob=target_accept),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method="sequential",
            progress_bar=False,
        )
        mcmc.run(
            jax.random.PRNGKey(seed),
            init_params=jnp.asarray(space.unconstrain(init_x)),
            extra_fields=("diverging",),
        )
        u = mcmc.get_samples(group_by_chain=True)  # (chains, samples, d)
        constrained, _ = jax.vmap(jax.vmap(space.constrain))(u)
        draws = {name: np.asarray(constrained[name]) for name in space.names}
        divergences = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())

    return MCMCResult(
        draws=draws,
        rhat={n: float(split_gelman_rubin(v)) for n, v in draws.items()},
        ess={n: float(effective_sample_size(v)) for n, v in draws.items()},
        divergences=divergences,
        seconds=time.perf_counter() - start,
    )
