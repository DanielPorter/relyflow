# Reliability-learning extension

Code: `relyflow/models/learning.py` (`ReliabilityLearning`).

## Status of the sources

The **learning rule** is published. Strickland et al. (2024) compared eight
models of how people learn an aid's reliability. They fitted each to trial-by-trial
judgments of "the probability that the automation's next classification will be
correct" from 240 participants across three experiments. The delta rule fitted
well. A two-kernel delta rule fitted best in seven of eight conditions.

**Coupling the rule to the accumulator model is not published.** Strickland et al.
(2024) propose it as a direction: learning could act as a "front end" to the
evidence accumulation model, with higher learned reliability increasing
accumulation in favour of agreeing with the aid. As of October 2026 we found no
published fit of such a combined model. The coupling below is relyflow's own
formalization. It is tested on data in milestone 8, and its results should be
described as an extension of the published models.

## Specification

**Perceived reliability.** Let $r_t$ be the reliability the operator believes in
before trial *t*, with $r_1 = r_0$. After an aided trial, with
$c_t = [a_t = s_t]$ marking whether the aid was right (Strickland et al., 2024,
Eq. 2):

$$
r_{t+1} = r_t + \alpha\,(c_t - r_t)
$$

Unaided trials give no evidence about the aid, so $r_{t+1} = r_t$. Equivalently,
$r_t$ is a recency-weighted average of the aid's past accuracy (their Eq. 3).

**Coupling.** In a two-choice task, advice from an aid believed to be at chance is
worthless. The *perceived validity*

$$
w_t = \operatorname{clip}(2 r_t - 1,\ 0,\ 1)
$$

is 0 at or below chance and 1 for an aid believed to be perfect. It scales both
advice effects of the reliance model ([static.md](static.md)):

$$
E_t = E \cdot w_t, \qquad I_t = I \cdot w_t
$$

Everything else (unaided rates, thresholds, the race) is as in the reliance
model. $E$ and $I$ are now the effects the operator would show if they believed
the aid were perfectly reliable.

This choice fits the evidence summarized above. Inhibition was stronger with the
95% aid than with the 75% aid, and excitation appeared mainly with the 95% aid.
A single multiplier is the simplest coupling that produces both, and the
alternatives it rules out are listed under "Variants not implemented".

## Parameters

All parameters of the reliance model, with $E$ and $I$ reinterpreted as above,
plus two more:

| Name | Meaning | Prior |
| --- | --- | --- |
| `inhibition` | $I$ at perceived validity 1 | HalfNormal(1) |
| `excitation` | $E$ at perceived validity 1 | HalfNormal(1) |
| `r0` | Perceived reliability before using the aid | Beta(4, 2) |
| `alpha` | Learning rate | Beta(1, 8) |

That is 14 parameters in total. The prior on `r0` (mean 0.67) reflects the
two-kernel fits in Strickland et al. (2024, Table 3), whose initial estimates
averaged 0.58–0.84 by condition. The prior on `alpha` (median 0.08, 95% below
0.31) spans the slow (0.01–0.04) and fast (0.11–0.44) learning rates estimated
there.

## Assumptions

- The operator learns after every aided trial whether the aid was right.
- Beliefs carry over unchanged across unaided trials, blocks and sessions.
- One aid, whose reliability may change over the session. Changing reliability is
  where the learning model differs most from the static models.

## Identifiability

- When `alpha` ≈ 0, beliefs stay at `r0` and only the products $E\,w(r_0)$ and
  $I\,w(r_0)$ are identified. This leaves a ridge between `r0` and the effect
  sizes. `r0` is identified separately only when beliefs move: when the aid's
  actual reliability differs from `r0`, or changes over time.
- If `r0` ≤ 0.5 and learning is slow, $w = 0$ and the advice is ignored. The
  likelihood is then flat in $E$ and $I$.
- Trial order carries the information about `alpha`. That is why this model
  needs a sequence-aware summary network rather than the set encoder used for the
  static models (milestone 3).

## Simulated behavior

`tests/test_simulators.py` checks two signatures of learning, with the reliance
model as a control. The table is from one run of 2,000 simulated operators with
`r0` = 0.8, `alpha` = 0.3, and excitation = inhibition = 1.5:

| | Learning | Reliance (control) |
| --- | --- | --- |
| Follows advice after the aid was wrong vs. right (80% aid) | 0.78 vs. 0.90 | 0.94 vs. 0.94 |
| Misuse with a 95% aid, then late in a 60% block | 0.65 → 0.28 | 0.71 → 0.72 |

## Variants not implemented

- **Two-kernel delta rule**, the best fit to reliability judgments in Strickland
  et al. (2024). Two delta learners share `r0`, with rates `alpha_slow` and
  `alpha_fast`. The fast estimate is used when the two differ by more than a
  threshold `T`. It adds 2 parameters and is worth adding if the single delta rule
  misfits real data.
- **Trust ratings as a second observed channel.** For example, reliability
  judgments ~ TN($r_t$, σ) on [0, 1], as in Strickland et al. (2024), for designs
  that collect them.
- **Separate couplings** for excitation and inhibition (e.g. excitation only
  above some perceived reliability).

References are listed in [README.md](README.md#references).
