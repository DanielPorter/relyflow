# Static models: advisory and reliance

Code: `relyflow/models/accumulator.py` (`Advisory`, `Reliance`).

## Mechanism

Strickland et al. (2021) proposed that two sources of input drive evidence
accumulation in an aided decision: the task itself and the decision aid. Inputs
can change accumulation in two ways:

- **Excitation** raises the rate of the accumulator that agrees with the input.
  Excitation from the aid lets the aid trigger decisions without the operator
  processing the task. This hands it autonomy, which is what this package calls
  *reliance*.
- **Inhibition** lowers the rate of the accumulator that disagrees with the input.
  Inhibition alone cannot trigger a decision, so the operator must still process
  the task. The aid then acts as an *advisor*.

With a 90%-reliable aid, Strickland et al. (2021) found inhibition and little evidence of
excitation. With a 95%-reliable aid, Strickland et al. (2023) found both. With a
75%-reliable aid they found inhibition with weak and inconsistent excitation.
Thresholds were lower with the aid, but contributed little to its effects.

## Specification

Notation for trial *t*:

- Stimulus $s_t \in \{0, 1\}$.
- Advice $a_t \in \{0, 1\}$, or none on unaided trials.
- Accumulator $k \in \{0, 1\}$ produces response *k*.

**Mean rates.** On unaided trials:

$$
\mu_{tk} = \begin{cases} v^{\text{match}}_{s_t} & k = s_t \\ v^{\text{mismatch}}_{s_t} & k \neq s_t \end{cases}
$$

On aided trials, the advice adds excitation $E$ to the agreeing accumulator and
subtracts inhibition $I$ from the other:

$$
\mu_{tk} = \mu^{\text{unaided}}_{tk} + E \cdot [k = a_t] - I \cdot [k \neq a_t]
$$

This gives four aided cells, each defined exactly as in Strickland et al. (2023),
relative to matched unaided trials:

| Trial type | Correct accumulator ($k = s_t$) | Error accumulator ($k \neq s_t$) |
| --- | --- | --- |
| Aid correct ($a_t = s_t$) | $v^{\text{match}} + E$ | $v^{\text{mismatch}} - I$ |
| Aid incorrect ($a_t \neq s_t$) | $v^{\text{match}} - I$ | $v^{\text{mismatch}} + E$ |

**Rate SDs.** $\sigma_{tk}$ = `sv_match` for the correct accumulator and 1 for the
error accumulator.

**Thresholds.** $b_{tk} = A + B_k \cdot \kappa^{[\text{aided}_t]}$, with
$\kappa$ = `B_aided_scale`. The multiplier keeps $b > A$, which follows the
threshold-multiplier variant in the 2021 OSF code.

**Race.** $x_k \sim U(0, A)$, $d_k \sim N^{+}(\mu_{tk}, \sigma_{tk})$,
$T_k = (b_{tk} - x_k)/d_k$, choice $= \arg\min_k T_k$, and
RT $= t_0 + \min_k T_k$.

**Variants.** `Reliance` estimates $E \ge 0$ and $I \ge 0$. `Advisory` fixes
$E = 0$.

## Parameters

| Name | Meaning | Prior |
| --- | --- | --- |
| `v_match_0`, `v_match_1` | Unaided mean rate of the correct accumulator, by stimulus | TN(2.5, 1) on [0, 6] |
| `v_mismatch_0`, `v_mismatch_1` | Unaided mean rate of the error accumulator, by stimulus | TN(0.5, 1) on [-3, 4] |
| `sv_match` | Rate SD of the correct accumulator (error accumulator's SD = 1) | TN(1, 0.5) on [0.1, 3] |
| `A` | Start-point range | TN(2, 1) on [0.1, 5] |
| `B_0`, `B_1` | Threshold gap $b - A$ per response | TN(2, 1) on [0.1, 6] |
| `B_aided_scale` | Threshold-gap multiplier on aided trials | LogNormal(0, 0.2) |
| `t0` | Non-decision time (s) | TN(0.3, 0.15) on [0.05, 1] |
| `inhibition` | $I$: rate decrease of the accumulator disagreeing with the advice | HalfNormal(1) |
| `excitation` | $E$: rate increase of the accumulator agreeing with the advice (reliance only) | HalfNormal(1) |

That is 11 parameters for advisory and 12 for reliance. Negative mean rates are
allowed, because rates are drawn from a normal truncated at zero.

## Relation to the published model

The published model is a cell-means LBA, fitted to each participant with the
Dynamic Models of Choice R suite (Heathcote et al., 2019). It has:

- Free mean rates for every combination of stimulus × condition (manual / aid)
  × aid correctness × accumulator match.
- Thresholds by condition × session × accumulator.
- One each of `A`, `t0` and `sv_match`, with the error accumulator's SD fixed at 1
  and no non-decision-time variability.

Excitation and inhibition are then read off as differences between aided and
matched unaided rates. relyflow makes the following changes:

1. **Effects are parameters.** One $E$ and one $I$ replace eight aided cell
   rates. This makes the mechanism directly estimable, and comparable across
   operators, from far fewer trials than the 2,400–4,680 per participant of the
   source studies. The cost is that $E$ and $I$ cannot differ by stimulus type
   or aid correctness. Strickland et al. (2023) reported one such difference: with
   the 75% aid, the non-conflict rate was lower, not higher, on aid-correct trials.
   If posterior predictive checks show misfit, a variant with stimulus-specific
   effects can be added.
2. **Unaided rates do not depend on whether the aid would have failed.** The
   published model kept separate manual rates for stimuli matched to aid failures.
   In the source designs, failure trials were randomly chosen stimuli. The 2023
   data confirm the cells do not differ: unaided accuracy was 0.849 vs. 0.851,
   and mean RT 1.92 s for both.
3. **One threshold multiplier for aided trials**, instead of per-condition,
   per-session thresholds. Session effects are not modelled. Fit sessions
   separately if practice effects matter.
4. **One aid per fit** (the 2023 study's two aids are fitted separately).

## Identifiability

**Unaided trials are necessary.** Without unaided trials, the transformation

$$
(v^{\text{match}}_s,\ v^{\text{mismatch}}_s,\ E,\ I) \mapsto (v^{\text{match}}_s + c,\ v^{\text{mismatch}}_s + c,\ E - c,\ I + c)
$$

leaves every aided mean rate in the table above unchanged, for any $c$. Only
$E + I$ and the unaided rates shifted by $E$ are identified, so advisory and
reliance cannot be distinguished. Unaided trials pin down the unaided rates and
break this symmetry.

**Excitation vs. lower thresholds.** Both speed aided responses. They separate
because excitation also speeds and increases errors when the aid is wrong, while
a lower threshold lowers accuracy on every trial type. Milestone 4's recovery
checks will measure how well this works at realistic trial counts.

**Model comparison.** Advisory is nested in reliance at $E = 0$, so comparison
depends on the prior on $E$ (HalfNormal(1)). That prior is part of the
specification and should be reported with any model-comparison result.

## Qualitative patterns the simulator must reproduce

These come from Strickland et al. (2021, 2023). The numbers below were computed by
us from the 2023 OSF data for participants 1–24, as accuracy and mean RT including
errors:

| Condition | Accuracy, aid right | Accuracy, aid wrong | RT, aid right | RT, aid wrong |
| --- | --- | --- | --- | --- |
| Unaided (matched trials) | 0.851 | 0.849 | 1.92 s | 1.92 s |
| 75% aid | 0.885 | 0.784 | 1.90 s | 2.01 s |
| 95% aid | 0.948 | 0.498 | 1.73 s | 2.04 s |

1. Accuracy rises when the aid is right and falls when it is wrong, compared
   with unaided trials.
2. Correct responses slow down when the aid is wrong (inhibition of the correct
   accumulator).
3. Misuse (following wrong advice) increases with excitation.
4. With $E = 0$, wrong advice slows responses but advice can never produce a
   response on its own.

References are listed in [README.md](README.md#references).
