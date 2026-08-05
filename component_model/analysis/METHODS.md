# Methods: reliability vs. sample size (script 13)

Script 12 asks "how much of Pitching+'s unpredictability is noise vs. missing
skill" at a fixed, full-season sample size. Script 13 asks a different
question: **at a given pitch count, which metric is most predictive of
next-season adjT, and how does that change as pitch count grows?** The two
share the same noise/drift/stable-skill machinery but script 13 applies it
per-metric (adjT, Stuff+, Location+) rather than only to Pitching+, and adds
the sample-size dimension script 12 doesn't need.

Motivating fact: the three metrics have different noise-per-pitch and
different asymptotic ceilings. Stuff+ is the least noisy per pitch but has the
lowest ceiling; adjT (predicting itself) is the noisiest per pitch but has the
highest ceiling; Location+ sits between both. The most predictive metric at
n pitches therefore depends on n, and the reliability curves cross.

## Part A: per-pitch noise, by splitting a season against itself

Take one pitcher's season, split games (not pitches) randomly into two halves,
and compare the two halves' means. If nothing about the pitcher changed all
year, any disagreement between the halves has to be noise -- there was no time
within one season for real change.

**Why whole games, not individual pitches.** Pitches within one game share
batter, park, umpire, and day effects. A pitch-level split leaves that shared
variance in both halves and understates the noise (this is script 06's
mistake, corrected in script 12; script 13 inherits the correction).

**Why /4.** Square the gap between the two halves' means, multiply by the
pitcher's total pitch count, divide by 4:

```
noise_per_pitch ≈ n × (mean_A − mean_B)² / 4
```

The 4 is two separate factors of 2, not one:

1. Cutting a sample in half doubles the variance of its mean (a smaller
   sample is a less reliable readout of the true value -- not because
   per-pitch noise changed, it didn't, but because there's less data for
   randomness to average out over).
2. Subtracting two independent noisy numbers adds their variances (each
   half's own wobble contributes; they don't cancel).

Multiplying by the pitcher's own pitch count (not a fixed number) matters
too: without it, a pitcher who happened to throw more pitches would look
artificially steadier per pitch, purely from averaging over more data, not
from actually being less noisy. Full derivation with a worked numeric
example: `variance_components.py`'s `effective_noise_scale` docstring.

Script 13 repeats this split 200 times with a fresh random assignment each
time (`--splits`) and reports the mean and a 95% CI, rather than the single
deterministic parity split script 12's Part A uses -- the point here is a
distribution over the estimate, not one fixed answer.

## Part B: each metric's ceiling against next-season adjT

For each metric X (adjT, Stuff+, Location+), the cross-year covariance of X
this season with adjT next season identifies X's ceiling: only X's permanent,
stable component can covary with a DIFFERENT season's outcome, since noise and
drift are fresh, independent draws each season and contribute ~0 to that
covariance in expectation.

The raw cross-year correlation still needs an attenuation correction, because
the observed VARIANCES of X and next-season adjT include noise and drift on
top of the permanent component, which bias a naive correlation toward zero.
Correcting requires dividing by each metric's STABLE-ONLY share of its own
variance (not stable+drift together) -- get this wrong and the correction
undershoots by roughly the metric's own persistence ratio. Caught here by a
synthetic test where a metric predicting ITSELF next season has a true
ceiling of exactly 1.0 by construction; the wrong correction recovered ~0.71.

## Part C: reliability curves and crossovers

The naive expectation is that R²(n) rises from 0 to a metric's ceiling² as
pitch count grows. It doesn't, for two reasons that both cap the curve BELOW
the idealized ceiling:

- **persistence_x** = stable / (stable + drift) for metric X. X's own drift
  never averages away no matter how many pitches you have in ONE season --
  drift is a real per-season effect, not sampling error. So R(n) tops out at
  `ceiling² × persistence_x`, not `ceiling²`, as n → infinity within a
  single season.
- **share_y** = next-season adjT's own stable share of its own variance, at
  whatever pitch count IT actually has. adjT's own noise and drift cap how
  correlated anything can be with it, independent of how good the predictor
  is. Easy to drop by mistake since it has nothing to do with X -- doing so
  overstated a numeric check here by more than 2x.

Full closed form:

```
R²(n) = ceiling² × persistence_x × share_y × n / (n + noise_per_pitch / signal_variance)
```

where `signal_variance` = stable + drift (X's total repeatable variance,
which noise erodes but drift does not). Verified against a brute-force
simulation that actually varies X's pitch count and checks the closed form
against the empirical squared correlation, not just against itself
(`tests/test_reliability_curves.py`).

Pairwise crossovers (the pitch count where two metrics' curves are equal) are
solved numerically via bisection, since the crossover of two saturating
curves like this has no clean algebraic form.

## Part D: precision-weighted blend

A GLS combination of all three metrics, solved per pitch count n:

```
Σ(n) w = c
```

where `Σ_ii(n)` = metric i's own variance at n pitches (signal_variance + 
noise/n), `Σ_ij` (i≠j) = the stable-skill covariance BETWEEN two different
metrics (identified the same way as Part B: via cross-year, cross-metric
covariance, since only the permanent components covary across years), and
`c_i` = metric i's covariance with next-season adjT. The blend's R² is
`c'Σ(n)⁻¹c / Var(adjT)`.

**Simplifying assumption, flagged explicitly:** the off-diagonal terms use
ONLY the cross-metric stable-skill covariance, assuming same-season
cross-metric noise and drift are negligible. This is plausible but unchecked
-- Stuff+ and Location+ are computed from the same pitches, so their
within-season noise could plausibly correlate a little. If the blend weights
end up load-bearing for a real decision, validate this assumption empirically
rather than trusting the closed form.

By construction, the blend's R² can never be below the best single metric's
(it can always fall back to weighting only that one) -- verified in tests,
modulo a negligible numerical-precision artifact at extreme n (~0.002, from
`np.linalg.solve` near a poorly-conditioned Σ, not a modeling error).

## Part E: transfer portal table

Parts C and D evaluated at realistic portal sample sizes (50, 150, 300, 600,
1200, 2500 pitches), reporting each single metric's R², the blend's R², and
which option wins at each pitch count. This is the coach-facing deliverable.

## Real-data run: what it found, and three more bugs it caught

Everything in Parts A-E is unit-tested against SYNTHETIC panels with a known
ground truth (same reasoning as `variance_components.py`'s own tests:
nothing in real data reveals the true noise/drift/stable split, so there's
no way to check correctness against real data directly). But synthetic
recovery tests only prove a formula is INTERNALLY correct against the
generative model it was checked against -- they can't catch two
independently-written functions silently drifting apart on a convention
neither test alone exercises. Running against the real D1 2024-2026 caches
(`workdir_2425_d1` + `workdir_2526_d1`, 1,377 pitchers, 2,945 pitcher-seasons
-- exactly matching script 12's published panel size) surfaced three such
bugs, none visible in any synthetic test:

1. `pooled_cross_covariance` pooled only ADJACENT season pairs (year1/year2),
   while `persistence_of`/`stable_share_of` (via `vc.variance_components`)
   pool ALL pairs (year1/year2 AND year1/year3, per the task spec). Since
   stable skill decays somewhat by lag-2, the lag-1-only estimate read
   systematically higher, and adjT's own same-metric ceiling came out 1.080
   instead of the required 1.000.
2. `signal_variance` used `noise_per_pitch / mean(n)`; `vc.variance_components`
   uses `noise_per_pitch * mean(1/n)`. These differ under Jensen's
   inequality whenever pitch counts vary across pitcher-seasons -- which
   they do, a lot, on real data (~100 to 1000+ FF) but not in early
   synthetic tests that happened to use a narrow, near-uniform n range.
3. Three separate call sites computed a metric's variance without
   season-centering it first, while `vc.variance_components` always centers.
   Consolidated into one `centered_variance` helper all three now share.

The tell was a regression test that shouldn't have needed real data to
fail: `optimal_blend` with a SINGLE predictor (a metric predicting itself)
must reproduce `reliability_curve`'s own number for that metric, to machine
precision -- both describe the identical quantity through two independent
code paths. Before these fixes, the real run's blend scored BELOW adjT alone
by up to 0.027 R² at large n, which is impossible for a correctly-computed
GLS blend (it can always fall back to weighting only the best predictor).
After the fixes: exact agreement (~1e-17), ceiling exactly 1.000, and the
blend gain over the best single metric is positive everywhere on the real
data (min +0.010, was min -0.027 before the fix). A permanent regression
test for this now lives in `tests/test_reliability_curves.py`.

**Real numbers from that run** (D1, 2024-2026, `--splits 50`; treat as a
first pass, not final -- rerun with more splits and both `2425` and `2526`
D1 caches for the production numbers):

- Noise/pitch: adjT 0.0476, Location+ 0.0057, Stuff+ 0.0002 -- confirms the
  anticipated ordering exactly (Stuff+ least noisy per pitch, adjT noisiest).
- adjT's own stable share (0.263) matches RESULTS.md's published adjT
  robustness row (26.3%) almost exactly -- independent confirmation the
  panel and estimators here are consistent with script 12's already-vetted
  pipeline.
- Ceilings vs next-season adjT: adjT 1.000 (by construction), Location+
  0.783, Stuff+ 0.488 -- Location+ well ahead of Stuff+, consistent in
  direction with the already-published "Location+ carries ~4x what Stuff+
  does" finding (script 12), though not the same statistic.
- Crossovers cluster tightly: Stuff+ vs Location+ at n=128, adjT vs Stuff+
  at n=140, adjT vs Location+ at n=161 -- Stuff+ wins at very low pitch
  counts (near-zero noise dominates), then both adjT and Location+ overtake
  it in a narrow band around 130-160 pitches. Directly confirms the
  motivating premise: the most predictive metric depends on sample size, and
  the curves genuinely cross in a realistic, coach-relevant range.
- The blend wins at every tested portal sample size (50 to 2500 pitches).

This is one run at `--splits 50` on one caches pair; before treating any
specific number above as a citable project finding, rerun with more splits
and confirm stability, per the usual replication discipline in this suite.

## The 24% ceiling caveat (per script 12, restated for this context)

Script 12's ~24% stable-skill figure is the ceiling for predicting FROM PAST
OUTCOMES. It is not a hard ceiling on what a BETTER, lower-noise physical
metric could recover. Some of what script 12's decomposition currently books
as "drift" may actually be stable skill that adjT (a noisy, results-based
measurement) simply can't see clearly enough to attribute correctly --
a lower-noise metric could recover that portion. What no metric, however
precise, can recover is genuinely NEW change after the fact: an injury, an
off-season mechanical overhaul. That part is drift in the strict sense and
is not recoverable by any amount of measurement precision, physical or
otherwise.
