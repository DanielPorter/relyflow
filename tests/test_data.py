import math

import numpy as np
import pandas as pd
import pytest

from relyflow import NO_ADVICE, TrialData, split_operators

# Eight hand-checked trials. The aid is wrong on trials 3 and 7.
# Trial 2 and 6 reject correct advice (disuse); trial 3 follows wrong advice (misuse).
STIMULUS = [1, 1, 1, 1, 0, 0, 0, 0]
ADVICE = [1, 1, 1, 0, 0, 0, 0, 1]
CHOICE = [1, 1, 0, 0, 0, 0, 1, 0]
RT = [0.5, 0.6, 0.7, 0.8, 0.5, 0.6, 0.7, 0.9]


@pytest.fixture
def data() -> TrialData:
    return TrialData(stimulus=STIMULUS, advice=ADVICE, choice=CHOICE, rt=RT, operator_id="op1")


def test_derived_columns(data):
    assert data.n_trials == len(data) == 8
    np.testing.assert_array_equal(data.aid_correct, [1, 1, 1, 0, 1, 1, 1, 0])
    np.testing.assert_array_equal(data.correct, [1, 1, 0, 0, 1, 1, 0, 1])
    np.testing.assert_array_equal(data.followed_advice, [1, 1, 0, 1, 1, 1, 0, 0])


def test_summary_matches_hand_calculation(data):
    s = data.summary()
    assert s["n_trials"] == 8
    assert s["aid_reliability"] == pytest.approx(6 / 8)
    assert s["accuracy"] == pytest.approx(5 / 8)
    assert s["accuracy_aid_correct"] == pytest.approx(4 / 6)
    assert s["accuracy_aid_incorrect"] == pytest.approx(1 / 2)
    assert s["mean_rt_correct_aid_correct"] == pytest.approx(0.55)
    assert s["mean_rt_correct_aid_incorrect"] == pytest.approx(0.9)
    assert s["misuse_rate"] == pytest.approx(1 - s["accuracy_aid_incorrect"])
    assert s["disuse_rate"] == pytest.approx(1 - s["accuracy_aid_correct"])


def test_summary_is_nan_for_conditions_that_never_occur():
    perfect_aid = TrialData(stimulus=[0, 1], advice=[0, 1], choice=[0, 1], rt=[0.4, 0.5])
    s = perfect_aid.summary()
    assert math.isnan(s["misuse_rate"])
    assert math.isnan(s["accuracy_aid_incorrect"])
    assert s["disuse_rate"] == 0.0


def test_unaided_trials_form_the_manual_baseline():
    nan = float("nan")
    data = TrialData(
        stimulus=[1, 0, 1, 0, 1, 1, 0, 0],
        advice=[nan, None, NO_ADVICE, nan, 1, 0, 0, 1],
        choice=[1, 0, 0, 0, 1, 0, 0, 1],
        rt=[1.0, 1.2, 1.4, 1.6, 0.5, 0.6, 0.7, 0.8],
    )
    np.testing.assert_array_equal(data.aided, [0, 0, 0, 0, 1, 1, 1, 1])
    assert not (data.aid_correct[:4].any() or data.followed_advice[:4].any())
    s = data.summary()
    assert s["n_aided"] == 4
    assert s["aid_reliability"] == pytest.approx(0.5)
    assert s["accuracy_unaided"] == pytest.approx(0.75)
    assert s["mean_rt_correct_unaided"] == pytest.approx((1.0 + 1.2 + 1.6) / 3)
    assert s["misuse_rate"] == 1.0
    assert s["disuse_rate"] == 0.0


def test_unaided_trials_survive_a_dataframe_round_trip():
    data = TrialData(stimulus=[0, 1, 1], advice=[NO_ADVICE, 1, 0], choice=[0, 1, 1], rt=[1, 1, 1])
    df = data.to_dataframe()
    assert df["advice"].isna().tolist() == [True, False, False]
    np.testing.assert_array_equal(TrialData.from_dataframe(df).advice, data.advice)


def test_arrays_are_read_only_copies():
    rt = np.array(RT)
    data = TrialData(stimulus=STIMULUS, advice=ADVICE, choice=CHOICE, rt=rt)
    with pytest.raises(ValueError):
        data.rt[0] = 1.0
    assert rt.flags.writeable


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"choice": [0, 1, 2, 0, 0, 0, 0, 0]}, "only 0, 1"),
        ({"advice": [1, 1, 1, 0, 0, 0, 0, 2]}, "only -1, 0, 1"),
        ({"advice": [[1, 1, 1, 0, 0, 0, 0, 1]]}, "1-D"),
        ({"rt": RT[:-1]}, "same length"),
        ({"rt": [0.5, -0.1, 0.7, 0.8, 0.5, 0.6, 0.7, 0.9]}, "positive and finite"),
        ({"rt": [0.5, np.nan, 0.7, 0.8, 0.5, 0.6, 0.7, 0.9]}, "positive and finite"),
    ],
)
def test_invalid_input_is_rejected(override, message):
    fields = {"stimulus": STIMULUS, "advice": ADVICE, "choice": CHOICE, "rt": RT} | override
    with pytest.raises(ValueError, match=message):
        TrialData(**fields)


def test_empty_data_is_rejected():
    with pytest.raises(ValueError, match="at least one trial"):
        TrialData(stimulus=[], advice=[], choice=[], rt=[])


def test_dataframe_round_trip_with_renamed_columns(data):
    df = data.to_dataframe().rename(columns={"rt": "rt_seconds", "choice": "response"})
    rebuilt = TrialData.from_dataframe(df, rt="rt_seconds", choice="response")
    np.testing.assert_array_equal(rebuilt.choice, data.choice)
    np.testing.assert_allclose(rebuilt.rt, data.rt)


def test_from_dataframe_reports_missing_columns():
    with pytest.raises(KeyError, match="advice"):
        TrialData.from_dataframe(pd.DataFrame({"stimulus": [1], "choice": [1], "rt": [0.5]}))


def test_split_operators_sorts_each_operator_by_trial():
    df = pd.DataFrame(
        {
            "subject": ["a", "b", "a", "b"],
            "trial": [1, 0, 0, 1],
            "stimulus": [1, 0, 0, 1],
            "advice": [1, 0, 0, 1],
            "choice": [1, 0, 0, 1],
            "rt": [0.2, 0.3, 0.4, 0.5],
        }
    )
    ops = split_operators(df, "subject", order_col="trial")
    assert list(ops) == ["a", "b"]
    np.testing.assert_allclose(ops["a"].rt, [0.4, 0.2])
    np.testing.assert_allclose(ops["b"].rt, [0.3, 0.5])
    assert ops["b"].operator_id == "b"
