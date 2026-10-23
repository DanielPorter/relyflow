"""Network inputs computed from trial data. Shared by training and fitting.

NumPy only, so the same preprocessing is available without the deep-learning stack.
"""

from __future__ import annotations

import numpy as np

from relyflow.data import NO_ADVICE

TRIAL_FEATURES = ("stimulus", "choice", "correct", "aided", "aid_correct", "followed", "log_rt")
DESIGN_FEATURES = ("log_n_unaided", "log_n_aided")

# LBA RTs have power-law tails; log RTs beyond these bounds carry no extra information.
LOG_RT_BOUNDS = (np.log(0.05), np.log(30.0))


def trial_features(
    stimulus: np.ndarray, advice: np.ndarray, choice: np.ndarray, rt: np.ndarray
) -> np.ndarray:
    """Per-trial features, shape ``(..., n_trials, len(TRIAL_FEATURES))``, float32."""
    aided = advice != NO_ADVICE
    features = (
        stimulus,
        choice,
        choice == stimulus,
        aided,
        aided & (advice == stimulus),
        aided & (choice == advice),
        np.clip(np.log(rt), *LOG_RT_BOUNDS),
    )
    return np.stack(np.broadcast_arrays(*features), axis=-1).astype(np.float32)


HISTORY_RATES = (0.01, 0.03, 0.1, 0.3)
HISTORY_FEATURES = (
    "previous_aid_outcome",
    "log_aid_experience",
    *(f"aid_track_record_{rate}" for rate in HISTORY_RATES),
)


def history_features(stimulus: np.ndarray, advice: np.ndarray) -> np.ndarray:
    """The aid's track record as the operator saw it before each trial.

    Shape ``(..., n_trials, len(HISTORY_FEATURES))``, float32:

    - the outcome of the most recent aided trial (+1 right, -1 wrong, 0 none yet);
    - log(1 + number of aided trials so far);
    - exponentially weighted aid accuracy at each rate in ``HISTORY_RATES``,
      starting from 0.5.

    In the learning model, perceived reliability is such a weighted average for
    some unknown rate and starting point. Giving every trial its history lets the
    summary network relate behavior to it without counting through the sequence.
    Depends only on the design, not on the operator's responses.
    """
    stimulus, advice = np.broadcast_arrays(np.atleast_2d(stimulus), np.atleast_2d(advice))
    batch, n_trials = advice.shape[:-1], advice.shape[-1]
    aided = advice != NO_ADVICE
    outcome = (advice == stimulus).astype(np.float64)
    rates = np.asarray(HISTORY_RATES)

    previous = np.zeros(batch)
    experience = np.zeros(batch)
    track = np.full((*batch, len(rates)), 0.5)
    out = np.empty((*batch, n_trials, len(HISTORY_FEATURES)), dtype=np.float32)
    for t in range(n_trials):
        out[..., t, 0] = previous
        out[..., t, 1] = np.log1p(experience)
        out[..., t, 2:] = track
        was_aided = aided[..., t]
        previous = np.where(was_aided, 2.0 * outcome[..., t] - 1.0, previous)
        experience = experience + was_aided
        update = rates * (outcome[..., t, None] - track)
        track = np.where(was_aided[..., None], track + update, track)
    return out


LEARNING_STATISTICS = (
    *(
        f"{stat}_{rate}"
        for rate in HISTORY_RATES
        for stat in (
            "corr_follow_track_aid_right",
            "corr_follow_track_aid_wrong",
            "corr_logrt_track",
        )
    ),
    "follow_after_failure_minus_success",
    "logrt_after_failure_minus_success",
)


def learning_statistics(
    stimulus: np.ndarray, advice: np.ndarray, choice: np.ndarray, rt: np.ndarray
) -> np.ndarray:
    """Summary statistics of learning from the aid, shape ``(..., len(LEARNING_STATISTICS))``.

    For each history rate: correlations, over aided trials, between the aid's track
    record and following the advice (separately when the aid was right and wrong)
    and log RT. Plus the change in following and log RT right after an aid failure,
    relative to right after a success. They are passed to the inference network
    directly, because the learning rate's signal is spread thinly over many trials
    and a learned summary alone missed it (docs/inference.md). Undefined values
    (e.g. no variation) are 0.
    """
    stimulus, advice, choice, rt = np.broadcast_arrays(
        *(np.atleast_2d(x) for x in (stimulus, advice, choice, rt))
    )
    history = history_features(stimulus, advice)
    aided = advice != NO_ADVICE
    followed = (choice == advice).astype(np.float64)
    log_rt = np.log(rt)
    right, wrong = aided & (advice == stimulus), aided & (advice != stimulus)
    stats = []
    for k in range(len(HISTORY_RATES)):
        track = history[..., 2 + k].astype(np.float64)
        stats += [
            _masked_corr(followed, track, right),
            _masked_corr(followed, track, wrong),
            _masked_corr(log_rt, track, aided),
        ]
    after_failure = aided & (history[..., 0] < 0)
    after_success = aided & (history[..., 0] > 0)
    for values in (followed, log_rt):
        stats.append(_masked_mean0(values, after_failure) - _masked_mean0(values, after_success))
    return np.stack(stats, axis=-1).astype(np.float32)


def _masked_mean0(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    count = mask.sum(axis=-1)
    return np.where(mask, values, 0.0).sum(axis=-1) / np.maximum(count, 1)


def _masked_corr(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    dx = np.where(mask, x - _masked_mean0(x, mask)[..., None], 0.0)
    dy = np.where(mask, y - _masked_mean0(y, mask)[..., None], 0.0)
    denominator = np.sqrt((dx**2).sum(axis=-1) * (dy**2).sum(axis=-1))
    return np.where(
        denominator > 1e-12, (dx * dy).sum(axis=-1) / np.maximum(denominator, 1e-12), 0.0
    )


def network_inputs(
    stimulus: np.ndarray,
    advice: np.ndarray,
    choice: np.ndarray,
    rt: np.ndarray,
    history: bool,
) -> np.ndarray:
    """Per-trial network inputs: trial features, plus history features if ``history``."""
    features = trial_features(stimulus, advice, choice, rt)
    if not history:
        return features
    return np.concatenate([features, history_features(stimulus, advice)], axis=-1)


def design_features(advice: np.ndarray) -> np.ndarray:
    """Log counts of unaided and aided trials, shape ``(..., 2)``, float32.

    Pooling in the summary networks discards how many trials there were, which
    sets how concentrated the posterior should be, so the counts are passed
    to the inference network directly.
    """
    aided = advice != NO_ADVICE
    counts = np.stack([(~aided).sum(axis=-1), aided.sum(axis=-1)], axis=-1)
    return np.log(np.maximum(counts, 1)).astype(np.float32)
