import numpy as np
import pytest

from relyflow import NO_ADVICE, make_design


def test_blocks_have_exact_stimulus_and_aid_error_counts():
    stimulus, advice = make_design(
        [(100, None), (200, 0.9), (40, 0.75)], np.random.default_rng(0), n_sims=5
    )
    assert stimulus.shape == advice.shape == (5, 340)
    assert np.all(advice[:, :100] == NO_ADVICE)
    for start, stop, errors in [(100, 300, 20), (300, 340, 10)]:
        block_stim, block_adv = stimulus[:, start:stop], advice[:, start:stop]
        assert np.all((block_adv != block_stim).sum(axis=1) == errors)
        assert np.all(block_stim.sum(axis=1) == (stop - start) // 2)


def test_aid_errors_are_placed_differently_in_each_simulation():
    stimulus, advice = make_design([(100, 0.8)], np.random.default_rng(1), n_sims=3)
    errors = advice != stimulus
    assert not np.array_equal(errors[0], errors[1])


def test_each_simulation_can_have_its_own_aid_reliability():
    reliability = np.array([1.0, 0.9, 0.5])
    stimulus, advice = make_design([(100, reliability)], np.random.default_rng(2), n_sims=3)
    np.testing.assert_array_equal((advice != stimulus).sum(axis=1), [0, 10, 50])


def test_invalid_reliability_is_rejected():
    with pytest.raises(ValueError, match="reliability"):
        make_design([(10, 1.2)], np.random.default_rng(0))
    with pytest.raises(ValueError, match="at least one block"):
        make_design([], np.random.default_rng(0))
