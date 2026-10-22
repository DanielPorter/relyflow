"""Trial-level data container shared by every model, simulator and fitter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_BINARY_FIELDS = ("stimulus", "advice", "choice")


@dataclass(frozen=True, eq=False)
class TrialData:
    """One operator's trials, in the order they were presented.

    Stimuli, advice and responses are binary categories coded 0/1 (for example
    "no conflict" / "conflict"). Trial order matters: reliability-learning models
    read the sequence trial by trial, so rows must not be shuffled.

    Whether the aid was correct is not stored separately; it is derived as
    ``advice == stimulus``.

    Parameters
    ----------
    stimulus
        True state of the world on each trial (0 or 1).
    advice
        The aid's recommendation on each trial (0 or 1).
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
        for name in _BINARY_FIELDS:
            arr = np.asarray(getattr(self, name))
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
            if not np.isin(arr, (0, 1)).all():
                raise ValueError(f"{name} must contain only 0 and 1")
            arrays[name] = arr.astype(np.int8)

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
    def aid_correct(self) -> np.ndarray:
        """True on trials where the aid's advice matched the stimulus."""
        return self.advice == self.stimulus

    @property
    def correct(self) -> np.ndarray:
        """True on trials where the operator's choice matched the stimulus."""
        return self.choice == self.stimulus

    @property
    def followed_advice(self) -> np.ndarray:
        """True on trials where the operator's choice matched the aid's advice."""
        return self.choice == self.advice

    def summary(self) -> dict[str, float]:
        """Descriptive statistics used by validation checks and reports.

        Misuse is following the aid when it is wrong; disuse is rejecting it when
        it is right (Parasuraman & Riley, 1997). With binary responses these are
        ``1 - accuracy_aid_incorrect`` and ``1 - accuracy_aid_correct``. Rates
        conditioned on trials that never occurred (e.g. misuse with a perfect
        aid) are NaN.
        """
        aid_right = self.aid_correct
        aid_wrong = ~aid_right
        correct = self.correct
        return {
            "n_trials": float(self.n_trials),
            "aid_reliability": float(aid_right.mean()),
            "accuracy": float(correct.mean()),
            "accuracy_aid_correct": _masked_mean(correct, aid_right),
            "accuracy_aid_incorrect": _masked_mean(correct, aid_wrong),
            "mean_rt_correct_aid_correct": _masked_mean(self.rt, correct & aid_right),
            "mean_rt_correct_aid_incorrect": _masked_mean(self.rt, correct & aid_wrong),
            "misuse_rate": _masked_mean(self.followed_advice, aid_wrong),
            "disuse_rate": _masked_mean(~self.followed_advice, aid_right),
        }

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
        """
        missing = [col for col in (stimulus, advice, choice, rt) if col not in df.columns]
        if missing:
            raise KeyError(f"DataFrame is missing columns: {missing}")
        return cls(
            stimulus=df[stimulus].to_numpy(),
            advice=df[advice].to_numpy(),
            choice=df[choice].to_numpy(),
            rt=df[rt].to_numpy(),
            operator_id=operator_id,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """One row per trial, with the derived correctness columns included."""
        df = pd.DataFrame(
            {
                "trial": np.arange(self.n_trials),
                "stimulus": self.stimulus,
                "advice": self.advice,
                "choice": self.choice,
                "rt": self.rt,
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


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if mask.any() else float("nan")
