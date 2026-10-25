"""Deliberately misspecified operators, for testing the out-of-distribution check.

Each generator returns choices and RTs for given stimulus/advice arrays of shape
``(n, n_trials)``, from a process that none of relyflow's models implements.
"""

from __future__ import annotations

import numpy as np

from relyflow.data import NO_ADVICE
from relyflow.models import get_model
from relyflow.models.lba import sample_race


def _model_data(stimulus, advice, rng, model="reliance", params=None):
    m = get_model(model)
    params = params or m.sample_prior(len(stimulus), rng)
    return m.simulate(params, stimulus, advice, rng)


def contaminated(stimulus, advice, rng, share=0.1):
    """Reliance-model data with ``share`` of trials replaced by random guesses with
    uniform RTs on [0.2, 7] s (lapses, distraction)."""
    choice, rt = _model_data(stimulus, advice, rng)
    lapse = rng.random(stimulus.shape) < share
    choice = np.where(lapse, rng.integers(0, 2, stimulus.shape), choice).astype(np.int8)
    return choice, np.where(lapse, rng.uniform(0.2, 7.0, stimulus.shape), rt)


def random_responder(stimulus, advice, rng):
    """Choices independent of stimulus and advice; RTs from the reliance model."""
    _, rt = _model_data(stimulus, advice, rng)
    return rng.integers(0, 2, stimulus.shape).astype(np.int8), rt


def always_follow(stimulus, advice, rng):
    """Pure automation heuristic: on aided trials, copy the advice after a fast,
    stimulus-independent delay; unaided trials follow the reliance model."""
    choice, rt = _model_data(stimulus, advice, rng)
    aided = advice != NO_ADVICE
    fast = 0.3 + rng.gamma(4.0, 0.12, stimulus.shape)
    return np.where(aided, advice, choice).astype(np.int8), np.where(aided, fast, rt)


def diffusion_bias(stimulus, advice, rng, dt=0.005, deadline=10.0):
    """A diffusion model in which advice shifts the starting point toward the advised
    response (a prior-like bias) instead of changing accumulation rates."""
    n, n_trials = stimulus.shape
    drift = rng.uniform(0.8, 2.5, (n, 1)) * np.where(stimulus == 1, 1.0, -1.0)
    bound = rng.uniform(1.2, 2.2, (n, 1))
    t0 = rng.uniform(0.2, 0.4, (n, 1))
    bias = rng.uniform(0.1, 0.35, (n, 1))  # share of the half-width moved toward the advice
    toward = np.where(advice == NO_ADVICE, 0.0, np.where(advice == 1, 1.0, -1.0))
    x = toward * bias * bound / 2
    rt = np.full((n, n_trials), np.nan)
    choice = np.zeros((n, n_trials), dtype=np.int8)
    active = np.ones((n, n_trials), dtype=bool)
    for step in range(int(deadline / dt)):
        x = x + np.where(active, drift * dt + np.sqrt(dt) * rng.standard_normal((n, n_trials)), 0)
        hit = active & (np.abs(x) >= bound / 2)
        choice[hit] = (x[hit] > 0).astype(np.int8)
        rt[hit] = (np.broadcast_to(t0, (n, n_trials)) + (step + 1) * dt)[hit]
        active &= ~hit
        if not active.any():
            break
    leftover = np.isnan(rt)
    rt[leftover] = (np.broadcast_to(t0, rt.shape) + deadline)[leftover]
    choice[leftover] = (x[leftover] > 0).astype(np.int8)
    return choice, rt


def fatigue(stimulus, advice, rng, final_factor=0.5):
    """Reliance-model data whose accumulation rates decline linearly over the
    session to ``final_factor`` of their starting value."""
    model = get_model("reliance")
    params = model.sample_prior(len(stimulus), rng)
    means, sds, bounds = model.race_inputs(params, stimulus, advice)
    factor = np.linspace(1.0, final_factor, stimulus.shape[1])[None, :, None]
    return sample_race(
        rng, means * factor, sds, bounds, params["A"][:, None], params["t0"][:, None]
    )


def slow_task(stimulus, advice, rng, shift=2.0):
    """Reliance-model data with every RT ``shift`` seconds longer: a task far slower
    than the priors' time scale."""
    choice, rt = _model_data(stimulus, advice, rng)
    return choice, rt + shift


SCENARIOS = {
    "contaminated (10% lapses)": contaminated,
    "random responder": random_responder,
    "always follows the aid": always_follow,
    "diffusion, advice biases start point": diffusion_bias,
    "fatigue (rates halve over session)": fatigue,
    "slow task (RT + 2 s)": slow_task,
}
