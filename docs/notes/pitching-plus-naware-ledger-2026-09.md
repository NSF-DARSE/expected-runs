# Sample-size-aware four-seam Pitching+, 2026-09: ledger

Scope: can the four-seam Pitching+ be trusted on fewer than 200 four-seams if each part counts
only as much as it can be trusted at the pitcher's count? Everything under "Pre-registration" was
written and committed BEFORE any forecast was correlated with the criterion. Results are appended
in a later commit; the pre-registration is not edited after its commit.

Sign convention: Stuff+ (ridge_pred), Location+ (map value), results (adjT) and the criterion are
expected run value from the pitcher's perspective, LOWER = BETTER. The harness keeps that
orientation, so a working forecast correlates POSITIVELY with the criterion. None of these are
"runs"; they are run value relative to an average pitcher.

## Why this, why now

The page mutes Pitching+ below 200 four-seams (`SAMPLE_FLOORS.pitch` in the app's `derive.ts`,
whose own comment says no analysis in either repo stands behind that number). The median arm in
the gate's pool throws 117. For most of a staff the headline grade is greyed out.

The cause is the blend itself. Pitching+ is script 08's equal-z blend, (z Stuff+ + z Location+ +
z results) / 3, with one third on results at 40 pitches and at 400. Results are the slowest part
to stabilise. The Pitching+ mix card, driven by the GLS weights contract, already tells a coach
that on a small sample "this is mostly the pitch's shape"; the number on the same card does not.

This is team-first work: it would put a trustworthy Pitching+ on arms the staff asks about in
February and March, without moving any Pitching+ the staff can already see.

## Pre-registration (committed before any read)

### Harness

`component_model/analysis/coach_pitching_plus_naware.py`, tests
`tests/test_pitching_plus_naware.py` (6). Caches read, never rebuilt: score build `workdir_ext3`
(2024 and 2025 D1), criterion build `workdir_ext3_crit` (2025 and 2026 D1, role-relabeled).

**SHIPPED** is script 08 reproduced: the shipped four-seam ridge (`fc.ridge_for_group`, asserted
equal to `fc.FEATS`), the pooled location map trained on 2024 four-seams, and mean adjT, over
2025 four-seams with plate location. Each component is z'd against 2025 pitchers with 100+
four-seams (2,470), then averaged in equal thirds.

**NAWARE** scales each component's z by

    k_c(n) = min(1, rel_c(n) / rel_c(200)),   rel_c(n) = n / (n + n0_c)

where n is the pitcher's 2025 four-seam count and n0_c is the component's game-split noise
ratio from script 16's estimator (alternating whole games, 20+ pitches per half) on the 2025
graded season alone. Nothing is fitted to an outcome, and no 2026 data enters the weights. At
200+ four-seams k = 1 for every component, so NAWARE equals SHIPPED for every Pitching+ the app
shows today. 200 is the app's existing floor, not a tuned value.

### Measured in the dry run (no criterion touched)

`--dry` computes the noise ratios, the readability curves, the pools and the power; it never
correlates anything with the 2026 criterion.

- n0: Stuff+ 4.0 (half r 0.945), Location+ 107.5 (0.374), results 274.0 (0.187); panel 3,474.
- k at n = 30 / 50 / 100 / 150: Stuff+ 0.90 / 0.94 / 0.98 / 0.99, Location+ 0.34 / 0.49 / 0.74 /
  0.90, results 0.23 / 0.37 / 0.63 / 0.84.
- Split-half reliability, SHIPPED / NAWARE: 15 pitches 0.19 / 0.58, 30 0.31 / 0.60, 50 0.43 /
  0.62, 100 0.60 / 0.68, 150 0.69 / 0.72, 200 0.75 / 0.75.
- Pool (gate pair C, 15+ four-seams and 10%+ share in 2025 and 2026): 2,007 pitchers; 1,454
  under 200, 553 at 200+. Bins: 15-49 404, 50-99 472, 100-199 578.

**The readability floor rule, fixed in the harness before the dry run, returned 200.** The rule
was "the smallest n at which NAWARE's reliability reaches SHIPPED's at 200 (0.752)". It cannot
move: NAWARE's composite gains weight on the noisy parts as n rises, so its reliability climbs
slowly from 0.58 to 0.75 rather than sprinting to it. The rule was the wrong standard for a
composite whose makeup changes with n. It is reported and not replaced here. For context only,
the reliability each existing app floor implies, from the same n0s: Stuff+ at 5 pitches 0.56,
Location+ at 50 0.32, adjusted results at 200 0.42, Pitching+ at 200 0.75. NAWARE at 15 (0.58)
clears every one of those except Pitching+'s own. Any floor change is Jack's call, after the
validity read below.

### The test

**Primary stratum:** pool pitchers under 200 four-seams (n = 1,454), the only ones whose
forecast changes. Delta = r(NAWARE, criterion) - r(SHIPPED, criterion), criterion = 2026 mean
four-seam adjT. 2,000 paired bootstrap resamples of pitchers, seed 20260817.

**Decision rule:** NAWARE **improves** iff P(Delta > 0) >= 0.95 and |Delta| >= 1 SE.

**Power, stated before the read.** r(SHIPPED, NAWARE) = 0.880 on the stratum. With the one-season
forecast r of about 0.24 on this pool (two-season ledger), the Steiger SE of the paired
difference is 0.0124, so the minimum detectable Delta at P >= 0.95 is about **+0.021**. A
smaller true gain will read as a tie.

**Reported, no bar:** the full pool (MDE 0.016); bins 15-49 (MDE 0.047), 50-99 (0.019), 100-199
(0.008); the 200+ bin's r (identical by construction). Also a descriptive decomposition: NAWARE
with only results down-weighted, and with only Location+ down-weighted, so a gain can be
attributed to a component.

**Reads.** 1. Cap 2: a second read only for a bug found after the first, recorded here first.

**Nothing ships from this pass.** A pass is a recommendation to Jack: script 08 would compute
NAWARE, and whether to lower the Pitching+ floor is a product decision this read informs and does
not make.

### Contamination, stated

- SHIPPED's one-season four-seam blend has been read on pair C before, in the shape ledger's
  three-way blend and the two-season ledger (with gate-pool z's, not script 08's). NAWARE and any
  n-dependent weighting of the displayed blend have never been read on any pair.
- Script 12 and the GLS contract used 2026 data. Neither enters here: n0s are re-measured on 2025
  alone, and N_REF is the app floor.
- The dry run above computed within-2025 reliability and predictor-only correlations before
  this registration. It did not compute any correlation with the criterion.

## Results, read 1 (2026-09-28)

One invocation, 147 s, `coach_pitching_plus_naware.json` in the score workdir. No bug found, no
second read.

### Verdict: a tie. NAWARE does not clear the bar; script 08's Pitching+ stands.

| stratum | n | SHIPPED r | NAWARE r | Delta | SE | P(Delta>0) |
|---|---|---|---|---|---|---|
| **under 200 (primary)** | 1,454 | +0.186 | +0.203 | **+0.016** | 0.016 | **0.851** |
| full pool | 2,007 | +0.234 | +0.249 | +0.016 | 0.013 | 0.885 |
| 15-49 | 404 | +0.122 | +0.134 | +0.013 | 0.031 | 0.664 |
| 50-99 | 472 | +0.131 | +0.151 | +0.020 | 0.013 | 0.925 |
| 100-199 | 578 | +0.280 | +0.282 | +0.002 | 0.005 | 0.681 |
| 200+ | 553 | identical by construction | | | | |

P = 0.851 is under 0.95 and Delta is 1.0 SE, so NAWARE does not improve by the registered rule.

### What the read shows beyond the verdict

**Direction is consistent, size is small.** The point estimate is positive in every bin and the
95% interval on the primary stratum is -0.016 to +0.047, so NAWARE is unlikely to be a worse
forecast than the number it would replace. The gain it does show is about the size of the
minimum detectable effect, and the read could not separate it from zero.

**Results carry it.** Down-weighting results alone reaches r 0.197 on the primary stratum,
Location+ alone 0.190, both together 0.203. The one-third weight on results at small counts is
where the current blend loses most.

**The power estimate was optimistic.** The bootstrap SE on the primary stratum was 0.016, about
30% above the Steiger approximation registered (0.0124), so the real minimum detectable Delta
was about +0.027, not +0.021. The approximation assumed r = 0.24 on a stratum whose actual r is
0.19 and ignores the heavy tails of run value. Future registrations should get power from a
simulation on the dry-run predictors, or inflate the analytic SE by a third.

**Not run: the discovery pair.** Reading pair D now and pooling it with this read would be
optional stopping after a near-miss, so it is not done.

### What this means for the page

Nothing changes on its own. Two facts for Jack's product call, neither of them a pass:

- On arms under 200 four-seams, NAWARE forecasts at least as well as script 08's blend (point
  estimate higher in every bin, interval lower bound -0.016) and its split-half reliability is
  far higher at small counts (0.58 vs 0.19 at 15 pitches, 0.62 vs 0.43 at 50).
- The pre-registered floor rule kept the floor at 200; the reliability NAWARE reaches at 15
  pitches already clears the standard the app applies to Stuff+, Location+ and adjusted results.

If Jack wants a Pitching+ shown below 200, NAWARE is the version to show, labelled as leaning on
the pitch's shape at small counts, which is what the mix card already says. That decision is
his; this read did not earn it on its own.

### Carried forward: blind 2027 read

Locked now, before any 2027 data exists: NAWARE exactly as registered (n0 re-measured on the
graded season alone, N_REF 200), primary stratum under 200 four-seams, same decision rule, on
the 2026 -> 2027 pair. It has no parameter to tune, so the 2027 read is a clean test.
