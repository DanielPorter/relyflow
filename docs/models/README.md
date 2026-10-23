# Model specifications

relyflow implements one family of evidence-accumulation models of automation use,
built on the linear ballistic accumulator (LBA; Brown & Heathcote, 2008) model of
human-automation teaming developed by Strickland, Loft, Heathcote and colleagues.

| Model | `get_model(...)` | How the aid's advice acts | Source |
| --- | --- | --- | --- |
| Advisory | `"advisory"` | Inhibits the accumulator that disagrees with the advice | Strickland et al. (2021) |
| Reliance | `"reliance"` | Inhibition, plus excitation of the accumulator that agrees with the advice | Strickland et al. (2023) |
| Reliability learning | `"learning"` | As reliance, scaled trial by trial by delta-rule learned reliability | Learning rule: Strickland et al. (2024). Coupling to the accumulator: relyflow; a related model was presented by Lin et al. (2026) (see [learning.md](learning.md)) |

- [static.md](static.md) specifies the advisory and reliance models.
- [learning.md](learning.md) specifies the reliability-learning extension.

The models are nested. Advisory is reliance with excitation fixed at zero, and
reliance is the learning model with no learning (`alpha = 0`, after rescaling the
effects). Comparing advisory with reliance answers "does this operator use the aid
as an advisor, or let it trigger decisions?" (roadmap milestone 5).

## Trial-level data

Each model reads one operator's trials in presentation order (`relyflow.TrialData`):

| Field | Meaning |
| --- | --- |
| `stimulus` | True state of the world, 0 or 1 |
| `advice` | The aid's recommendation, 0 or 1, or `NO_ADVICE` on unaided trials |
| `choice` | The operator's response, 0 or 1 |
| `rt` | Response time in seconds |

Whether the aid was correct is derived as `advice == stimulus`.

Requirements and assumptions:

1. **Unaided trials are required.** Excitation and inhibition are defined relative
   to unaided performance, and without unaided trials they cannot be separated
   (proof in [static.md](static.md#identifiability)). Both source studies
   interleaved manual blocks with aided blocks for this reason.
2. **One aid per fit.** Data from aids of different reliability are fitted
   separately, each with the unaided trials.
3. **Aid errors are strongly recommended.** The static models can be estimated
   from aid-correct trials alone. However, the misuse estimates, the RT cost of
   wrong advice, and learning from aid failures all depend on observing errors.
4. **Preprocessing.** Remove non-responses (time-outs) and anticipations
   (RT < 0.2 s) before fitting, as Strickland et al. did. Response deadlines are
   not modelled. Non-responses were rare in the source data: 96 of 57,600 trials
   in the 2021 study.
5. **Feedback (learning model only).** The operator is assumed to learn after each
   aided trial whether the aid was right. In the source paradigm, every incorrect
   response produced feedback naming the correct answer.

## Shared accumulator core

On each trial two accumulators race, one per response. Accumulator *k* starts at
$x_k \sim U(0, A)$ and rises linearly at rate $d_k$, drawn from a normal
distribution with mean $\mu_k$ and SD $\sigma_k$ truncated below at 0. It finishes at
$T_k = (b_k - x_k) / d_k$. The first accumulator to finish gives the response,
and RT $= t_0 + \min_k T_k$. Thresholds are reported as gaps $B_k = b_k - A > 0$. The
error accumulator's rate SD is fixed at 1 to set the scale, as in the source
papers. Advice changes the mean rates $\mu_k$ and, through one multiplier, the
thresholds. The details are in [static.md](static.md).

## Priors

The priors are proper and independent. They are written in `relyflow.models` as
`scipy.stats` distributions, and the tables in these documents list the same
values. They follow the scale of the published priors and are
narrowed so that training simulations stay plausible.

| Parameter | Published prior (DMC, per participant) | relyflow prior |
| --- | --- | --- |
| `t0` | TN(0.3, 1) on [0.1, 5] | TN(0.3, 0.15) on [0.05, 1] |
| `A` | TN(3, 1) on [0, 10] (2021); TN(2.5, 0.25) on [0.01, 10] (2023) | TN(2, 1) on [0.1, 5] |
| `B_k` | TN(2, 1) on [0, ∞) | TN(2, 1) on [0.1, 6] |
| `sv_match` | TN(1, 1) on [0, ∞) | TN(1, 0.5) on [0.1, 3] |
| matching rates | N(1, 2) | TN(2.5, 1) on [0, 6] |
| mismatching rates | N(0, 2) | TN(0.5, 1) on [-3, 4] |

TN(m, s) on [l, u] is a normal truncated to [l, u]. Published priors are from
`R/3-model_specification*.R` in the OSF projects listed below.

### Prior predictive check

Each model simulated 5,000 operators drawn from its prior, on a design of 200
unaided trials followed by 400 trials with an 85%-reliable aid. The table gives
5th–95th percentiles across operators:

| Statistic | Advisory | Reliance | Learning |
| --- | --- | --- | --- |
| Unaided accuracy | 0.55–0.96 | 0.55–0.96 | 0.55–0.96 |
| Accuracy, aid right | 0.63–0.99 | 0.70–1.00 | 0.66–0.99 |
| Accuracy, aid wrong | 0.35–0.93 | 0.18–0.88 | 0.32–0.92 |
| Misuse rate | 0.07–0.65 | 0.12–0.82 | 0.08–0.68 |
| Median RT (s) | 0.83–2.41 | 0.77–2.11 | 0.80–2.18 |

These intervals cover the 2023 data: unaided accuracy 0.85, median RT about 1.4 s,
and misuse of 0.22 with the 75% aid and 0.50 with the 95% aid.
`tests/test_simulators.py` checks this coverage.

**Time scale.** The priors are calibrated to decisions with median RTs of roughly
0.8–2.4 s. Rather than widening them, much faster or slower tasks will be handled
by expressing RTs in units of a task-specific time scale. This is applied when
fitting (milestone 6) and simulating operators (milestone 7). The models
themselves are unit-free: 1 unit = 1 s by default.

**Long RT tails.** Rates are truncated at zero, so near-zero rates occur and each
accumulator's finishing time has a power-law tail, P(T > t) ∝ 1/t. Race RTs
therefore have finite means but infinite variance, and 0.5–0.8% of prior-predictive
trials exceed 10 s, with a few over 100 s. Real tasks impose deadlines (7–8 s in
the source studies) and drop non-responses. Two consequences:

- Validation compares medians and quantiles, not means.
- Milestone 3 must handle the tail when building training data (e.g. censoring at
  a deadline, or a log-RT transform).

## Simulators and likelihood

Every model simulates with `model.simulate(params, stimulus, advice, rng)` and
scores data with `model.log_likelihood(params, stimulus, advice, choice, rt)`.
Both are vectorized over parameter sets. The likelihood is exact for all three
models: the learning model's beliefs depend only on the aid's observed accuracy,
not on the operator's responses, so they are deterministic given the design.

The sampler draws truncated-normal rates by inversion in log space, which stays
accurate 60 or more SDs into the tail. It is checked against the analytic density
(Brown & Heathcote, 2008, normalized for truncation) by chi-square goodness of fit
on 100,000 trials in four regimes, including negative mean rates. The tests catch
planted bugs as small as a 10% error in the rate SD.

Throughput is about 6 million trials per second on one CPU core for every model
(`benchmarks/simulators.py`; Apple Silicon, NumPy 2.5).

## Source datasets

Both source studies posted trial-level data and model code on OSF. Both include
unaided baselines, and both are candidates for the real-data demonstration
(milestone 8, licenses still to be checked):

- Strickland et al. (2021): <https://osf.io/28g3j/>. 24 participants × 2,400
  trials, 90%-reliable aid vs. manual.
- Strickland et al. (2023): <https://osf.io/q3b2r/>. 28 participants (24 analysed)
  × about 4,680 trials, 95%- and 75%-reliable aids vs. manual, with trust ratings.

## References

- Brown, S. D., & Heathcote, A. (2008). The simplest complete model of choice
  response time: Linear ballistic accumulation. *Cognitive Psychology, 57*(3),
  153–178. https://doi.org/10.1016/j.cogpsych.2007.12.002
- Heathcote, A., Lin, Y.-S., Reynolds, A., Strickland, L., Gretton, M., & Matzke, D.
  (2019). Dynamic models of choice. *Behavior Research Methods, 51*(2), 961–985.
  https://doi.org/10.3758/s13428-018-1067-y
- Lin, W., Loft, S., Heathcote, A., Miletić, S., & Strickland, L. (2026, July 18). An
  evidence accumulation model of adaptive automation use [Conference presentation].
  MathPsych/ICCM 2026. https://mathpsych.org/session/514
- Parasuraman, R., & Riley, V. (1997). Humans and automation: Use, misuse, disuse,
  abuse. *Human Factors, 39*(2), 230–253. https://doi.org/10.1518/001872097778543886
- Strickland, L., Heathcote, A., Bowden, V. K., Boag, R. J., Wilson, M. K., Khan, S.,
  & Loft, S. (2021). Inhibitory cognitive control allows automated advice to improve
  accuracy while minimizing misuse. *Psychological Science, 32*(11), 1768–1781.
  https://doi.org/10.1177/09567976211012676
- Strickland, L., Boag, R. J., Heathcote, A., Bowden, V., & Loft, S. (2023).
  Automated decision aids: When are they advisors and when do they take control of
  human decision making? *Journal of Experimental Psychology: Applied, 29*(4),
  849–868. https://doi.org/10.1037/xap0000463
- Strickland, L., Farrell, S., Wilson, M. K., Hutchinson, J., & Loft, S. (2024). How
  do humans learn about the reliability of automation? *Cognitive Research:
  Principles and Implications, 9*, 8. https://doi.org/10.1186/s41235-024-00533-1
