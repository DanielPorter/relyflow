"""The distribution of experimental designs that training data are simulated from.

An amortized network is only trustworthy for designs like those it was trained
on, so this distribution defines the supported range of studies: how many trials,
how they split into unaided and aided blocks, and how reliable the aid is.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from relyflow.design import make_design


@dataclass(frozen=True)
class DesignPrior:
    """Random block designs for training.

    Each draw picks a total trial count (a multiple of ``n_trials_step``, so the
    networks see few distinct input shapes and are not recompiled for every batch),
    an unaided share, 1 to
    ``max_unaided_blocks`` unaided blocks and 1 to ``max_aided_blocks`` aided
    blocks in random order with random sizes. Each simulated session gets its own
    aid reliability. With probability ``p_shared_reliability`` all aided blocks
    share it; otherwise each aided block draws its own, so reliability can change
    mid-session.
    """

    n_trials: tuple[int, int] = (120, 1200)
    n_trials_step: int = 60
    unaided_share: tuple[float, float] = (0.2, 0.5)
    max_unaided_blocks: int = 2
    max_aided_blocks: int = 3
    reliability: tuple[float, float] = (0.55, 0.98)
    p_shared_reliability: float = 0.5
    p_stimulus: float = 0.5

    def sample(self, n_sims: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Stimulus and advice arrays, ``(n_sims, n_trials)``, sharing one block layout."""
        n_total = int(
            rng.choice(np.arange(self.n_trials[0], self.n_trials[1] + 1, self.n_trials_step))
        )
        n_unaided = round(n_total * rng.uniform(*self.unaided_share))
        unaided_sizes = _split(n_unaided, int(rng.integers(1, self.max_unaided_blocks + 1)), rng)
        aided_sizes = _split(
            n_total - n_unaided, int(rng.integers(1, self.max_aided_blocks + 1)), rng
        )

        shared = rng.random() < self.p_shared_reliability
        base = rng.uniform(*self.reliability, size=n_sims)
        blocks = [(size, None) for size in unaided_sizes] + [
            (size, base if shared else rng.uniform(*self.reliability, size=n_sims))
            for size in aided_sizes
        ]
        order = rng.permutation(len(blocks))
        return make_design([blocks[i] for i in order], rng, n_sims, self.p_stimulus)

    def to_dict(self) -> dict:
        return asdict(self)


def _split(n: int, k: int, rng: np.random.Generator) -> list[int]:
    """Split ``n`` into ``k`` random positive parts."""
    k = min(k, n)
    cuts = np.sort(rng.choice(np.arange(1, n), size=k - 1, replace=False))
    return np.diff(np.concatenate([[0], cuts, [n]])).tolist()
