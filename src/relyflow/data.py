"""Trial-level data container shared by every model, simulator and fitter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NO_ADVICE = -1
"""Value of ``TrialData.advice`` on unaided (manual) trials."""


@dataclass(frozen=True, eq=False)
class TrialData:
    """One operator's trials, in the order they were presented.

    Stimuli, advice and responses are binary categories coded 0/1 (for example
    "no conflict" / "conflict"). Trial order matters: reliability-learning models
    read the sequence trial by trial, so rows must not be shuffled.

    Unaided (manual) trials have ``advice == NO_ADVICE``. The accumulator models
    need some unaided trials as a baseline: without them, excitation and
    inhibition cannot be told apart (see ``docs/models/static.md``).

    Whether the aid was correct is not stored separately; it is derived as
    ``advice == stimulus``.

    Parameters
    ----------
    stimulus
        True state of the world on each trial (0 or 1).
    advice
        The aid's recommendation on each trial (0 or 1), or ``NO_ADVICE`` (-1)
        on unaided trials. NaN and None are also read as unaided.
    choice
        The operator's response on each trial (0 or 1).
    rt
        Response time in seconds.
    operator_id
        Optional identifier carried through to fits and reports.
    """

    stimulus: np.ndarray
    advice: np.ndarray
    choice: np.ndarray
    rt: np.ndarray
    operator_id: str | None = None

    def __post_init__(self) -> None:
        arrays: dict[str, np.ndarray] = {}
        for name in ("stimulus", "choice"):
            arrays[name] = _coded(getattr(self, name), name, allowed=(0, 1))
        advice = np.asarray(self.advice, dtype=np.float64)
        arrays["advice"] = _coded(
            np.where(np.isnan(advice), NO_ADVICE, advice), "advice", allowed=(NO_ADVICE, 0, 1)
        )

        rt = np.array(self.rt, dtype=np.float64)
        if rt.ndim != 1:
            raise ValueError(f"rt must be 1-D, got shape {rt.shape}")
        if not (np.isfinite(rt).all() and (rt > 0).all()):
            raise ValueError("rt must be positive and finite (seconds)")
        arrays["rt"] = rt

        lengths = {name: len(arr) for name, arr in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"all fields must have the same length, got {lengths}")
        if lengths["rt"] == 0:
            raise ValueError("TrialData needs at least one trial")

        for name, arr in arrays.items():
            arr.flags.writeable = False
            object.__setattr__(self, name, arr)

    def __len__(self) -> int:
        return self.n_trials

    @property
    def n_trials(self) -> int:
        return len(self.rt)

    @property
    def aided(self) -> np.ndarray:
        """True on trials where the aid gave advice."""
        return self.advice != NO_ADVICE

    @property
    def aid_correct(self) -> np.ndarray:
        """True on aided trials where the advice matched the stimulus."""
        return self.advice == self.stimulus

    @property
    def correct(self) -> np.ndarray:
        """True on trials where the operator's choice matched the stimulus."""
        return self.choice == self.stimulus

    @property
    def followed_advice(self) -> np.ndarray:
        """True on aided trials where the operator's choice matched the advice."""
        return self.choice == self.advice

    def summary(self) -> dict[str, float]:
        """Descriptive statistics used by validation checks and reports.

        See :func:`summary_stats` for definitions.
        """
        stats = summary_stats(self.stimulus, self.advice, self.choice, self.rt)
        return {name: float(value) for name, value in stats.items()}

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        stimulus: str = "stimulus",
        advice: str = "advice",
        choice: str = "choice",
        rt: str = "rt",
        operator_id: str | None = None,
    ) -> TrialData:
        """Build from a DataFrame with one row per trial, already in presentation order.

        Keyword arguments name the columns to read when they differ from the defaults.
        Missing values in the advice column mark unaided trials.
        """
        missing = [col for col in (stimulus, advice, choice, rt) if col not in df.columns]
        if missing:
            raise KeyError(f"DataFrame is missing columns: {missing}")
        return cls(
            stimulus=df[stimulus].to_numpy(),
            advice=df[advice].to_numpy(dtype=np.float64, na_value=np.nan),
            choice=df[choice].to_numpy(),
            rt=df[rt].to_numpy(),
            operator_id=operator_id,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """One row per trial, with derived columns included. Unaided advice is <NA>."""
        df = pd.DataFrame(
            {
                "trial": np.arange(self.n_trials),
                "stimulus": self.stimulus,
                "advice": pd.Series(self.advice, dtype="Int8").mask(~self.aided),
                "choice": self.choice,
                "rt": self.rt,
                "aided": self.aided,
                "aid_correct": self.aid_correct,
                "correct": self.correct,
            }
        )
        if self.operator_id is not None:
            df.insert(0, "operator", self.operator_id)
        return df


def split_operators(
    df: pd.DataFrame,
    operator_col: str = "operator",
    *,
    order_col: str | None = None,
    **columns: str,
) -> dict[str, TrialData]:
    """Split a long-format DataFrame of many operators into one ``TrialData`` each.

    Rows keep their existing order within each operator unless ``order_col`` is
    given, in which case they are sorted by it. Extra keyword arguments rename
    columns as in :meth:`TrialData.from_dataframe`.
    """
    if operator_col not in df.columns:
        raise KeyError(f"DataFrame has no operator column {operator_col!r}")
    operators: dict[str, TrialData] = {}
    for operator, rows in df.groupby(operator_col, sort=False):
        if order_col is not None:
            rows = rows.sort_values(order_col, kind="stable")
        operators[str(operator)] = TrialData.from_dataframe(
            rows, operator_id=str(operator), **columns
        )
    return operators


def _coded(values, name: str, allowed: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if not np.isin(arr, allowed).all():
        raise ValueError(f"{name} must contain only {', '.join(map(str, allowed))}")
    return arr.astype(np.int8)


def summary_stats(
    stimulus: np.ndarray, advice: np.ndarray, choice: np.ndarray, rt: np.ndarray
) -> dict[str, np.ndarray]:
    """Descriptive statistics over the last axis, for one operator or a batch of simulations.

    Arrays have shape ``(..., n_trials)`` and each statistic has shape ``(...)``.
    Misuse is following the aid when it is wrong; disuse is rejecting it when it
    is right (Parasuraman & Riley, 1997). With binary responses these are
    ``1 - accuracy_aid_incorrect`` and ``1 - accuracy_aid_correct``. The
    ``unaided`` entries are the manual baseline. Statistics conditioned on trials
    that never occurred (e.g. misuse with a perfect aid) are NaN.
    """
    aided = advice != NO_ADVICE
    aid_right = advice == stimulus
    aid_wrong = aided & ~aid_right
    unaided = ~aided
    correct = choice == stimulus
    followed = choice == advice
    return {
        "n_trials": np.full(aided.shape[:-1], aided.shape[-1], dtype=np.float64),
        "n_aided": aided.sum(axis=-1).astype(np.float64),
        "aid_reliability": _masked_mean(aid_right, aided),
        "accuracy": correct.mean(axis=-1),
        "accuracy_unaided": _masked_mean(correct, unaided),
        "accuracy_aid_correct": _masked_mean(correct, aid_right),
        "accuracy_aid_incorrect": _masked_mean(correct, aid_wrong),
        "mean_rt_correct_unaided": _masked_mean(rt, correct & unaided),
        "mean_rt_correct_aid_correct": _masked_mean(rt, correct & aid_right),
        "mean_rt_correct_aid_incorrect": _masked_mean(rt, correct & aid_wrong),
        "misuse_rate": _masked_mean(followed, aid_wrong),
        "disuse_rate": _masked_mean(~followed, aid_right),
    }


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean of ``values`` where ``mask`` is True, over the last axis; NaN where nothing is."""
    count = mask.sum(axis=-1)
    total = np.where(mask, values, 0).sum(axis=-1, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)
