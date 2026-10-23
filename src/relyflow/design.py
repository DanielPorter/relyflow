"""Stimulus and advice sequences for simulated sessions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from relyflow.data import NO_ADVICE


def make_design(
    blocks: Sequence[tuple[int, float | np.ndarray | None]],
    rng: np.random.Generator,
    n_sims: int = 1,
    p_stimulus: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build stimulus and advice arrays of shape ``(n_sims, n_trials)``.

    ``blocks`` lists ``(n_trials, reliability)`` pairs run in order, with
    ``reliability=None`` for an unaided block. As in Strickland et al.'s designs,
    each block has exactly ``round(p_stimulus * n)`` stimulus-1 trials and exactly
    ``round((1 - reliability) * n)`` aid errors, both at random positions.
    ``reliability`` may also be an array of shape ``(n_sims,)``, giving each
    simulated session its own aid.

    Example: ``make_design([(200, None), (400, 0.9)], rng)`` is 200 unaided trials
    followed by 400 trials with a 90%-reliable aid.
    """
    if not blocks:
        raise ValueError("a design needs at least one block")
    stimulus, advice = [], []
    for n_trials, reliability in blocks:
        stim = _random_flags(round(p_stimulus * n_trials), n_trials, n_sims, rng).astype(np.int8)
        if reliability is None:
            adv = np.full((n_sims, n_trials), NO_ADVICE, dtype=np.int8)
        else:
            reliability = np.asarray(reliability, dtype=np.float64)
            if np.any((reliability < 0.0) | (reliability > 1.0)):
                raise ValueError(f"reliability must be in [0, 1] or None, got {reliability}")
            n_wrong = np.rint((1.0 - reliability) * n_trials).astype(np.int64)
            wrong = _random_flags(n_wrong, n_trials, n_sims, rng)
            adv = np.where(wrong, 1 - stim, stim).astype(np.int8)
        stimulus.append(stim)
        advice.append(adv)
    return np.concatenate(stimulus, axis=1), np.concatenate(advice, axis=1)


def _random_flags(k, n: int, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Boolean ``(n_sims, n)`` array with exactly ``k`` True values per row, at random positions.

    ``k`` is an int or an array of shape ``(n_sims,)``.
    """
    permutation = rng.random((n_sims, n)).argsort(axis=1)
    return permutation < np.reshape(k, (-1, 1))
