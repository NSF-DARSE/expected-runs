# Reliability decomposition: noise vs. missing skill

**Date:** 2026-07-27
**Status:** design, pending user review
**Scope:** A + B only (defer the missing-channel horse race, C, until the size of the residual is known)

## Question

How much of our inability to predict pitcher success year over year is irreducible
noise, and how much is persistent skill that Pitching+ fails to capture?

"Noise" is actually two distinct buckets, and only the third bucket is a Pitching+
problem. The deliverable's job is to size all three against the fair criterion.

| bucket | what it is | fixable by better features? |
|---|---|---|
| 1. Measurement noise | a college season is too few pitches to pin down true talent | no (needs more data / a smoother metric) |
| 2. True year-to-year change | pitchers genuinely change: velo, health, mechanics, role | no (static model cannot recover it) |
| 3. Missing skill | persistent, year-1-visible talent the physical model does not encode (command, deception, sequencing, unfeatured pitch shape) | **yes — this is the Pitching+ gap** |

Buckets 1 and 2 are both "noise" in the colloquial sense but fail differently and set
the ceiling; bucket 3 is the headline number the analysis exists to produce.

## Target and conventions

- **Success metric: the fair criterion `xT`** (defense/luck-stripped expected run
  value). Chosen because it is the least noisy target available, so any residual
  attributed to bucket 3 is a conservative, defensible floor on missing skill.
- Run-value units, **lower = better**; orientation per `fair_criterion.py` docstring.
  A physical trait helps when it correlates *negatively* with the future criterion.
- **D1-only** for headline calls (the feed's D2 share doubled from 2024 to 2026;
  all-levels numbers dilute; report D1 as the call, all-levels as a footnote).
- **Four-seam fastballs first.** Per-type repetition is out of scope here; note it as
  a follow-on.
- Reuse `fair_criterion.py` for all shared math. Do not modify it (it is a fixed
  reference); compose its functions. Read source from `STUFFPLUS_DATA`, cache/write
  only under `STUFFPLUS_WORKDIR`. Never commit derived values or per-pitcher output.

## Part A — within-season reliability (the honest ceiling)

Separates bucket 1 (measurement noise) from bucket 2 (true drift), which the existing
across-season number (~0.30, attenuating to ~0.27 in 2025→26) conflates.

**Method.** Within a single season, split each qualified pitcher's FF by **odd/even
`GameID`** (game is the right unit of variation, not pitch), take mean `xT` per half,
correlate across pitchers, and Spearman-Brown correct to full-season length:
`rho_full = 2r / (1 + r)`. Report for 2024, 2025, and 2026 separately (D1).

`GameID` is dropped by `ff_panel`'s slim frame, so A composes
`load_pitches(args)` → `add_xt(df)` → filter to FF + panel, keeping `GameID`, rather
than reading the slim panel.

**Estimands.**
- `rho_within` = full-season measurement reliability (bucket 1 = `1 - rho_within`).
- `persistence ≈ r_across / rho_within` (disattenuated year-to-year stability of true
  skill). Interpretation gate:
  - `rho_within ≈ r_across` → the metric is just noisy; bucket 1 dominates and there
    is little stable skill for *any* model to chase.
  - `rho_within >> r_across` → pitchers are stable within a year but drift across
    years; bucket 2 dominates and no static model beats that ceiling regardless of
    features.

## Part B — variance components (the headline number)

Decompose the between/within variance of `xT` and measure how much of the *stable*
part the physical model already explains.

**Panel.** Three seasons (2024/25/26), D1, pitchers with 100+ qualifying FF in the
seasons used (mirror `PANEL_MIN_FF`; flag <50 FF unreliable per the sample floor). The
loader takes a two-year pair, so assembling a three-season panel is an explicit plan
step: either compose two overlapping pairs and union on `PitcherId`, or point
`STUFFPLUS_DATA` at a combined three-season CSV. Either way, `add_xt` must be fit on a
fixed reference vintage so `xT` is comparable across seasons (do not refit the EV/LA
map per season).

**Model.** Three-level decomposition of pitch-level (or pitcher-season-mean, precision-
weighted) `xT`:

```
xT ~ season + (1 | pitcher) + (1 | pitcher:season)
```

- `sigma2_pitcher` = **stable skill** (bucket 3 lives in what's left of this)
- `sigma2_pitcher:season` = **year-to-year drift** (bucket 2)
- `sigma2_resid` = **within-season sampling noise** (bucket 1)
- `season` fixed effect absorbs environment drift (feed growth, level-mix shift).

If a pitch-level three-level fit on ~2M rows is too heavy, aggregate to precision-
weighted pitcher-season means and plug `rho_within` from Part A in as the known
measurement-error component; the three-bucket split is identical, just cheaper. With
only two seasons conveniently assembled, the 2-season fallback still identifies the
split (stable = cross-year covariance; drift = between-season residual net of
measurement error from A) — three seasons simply identify drift far better.

**Missing-skill step.** Refit adding **physical Pitching+ as a fixed effect** —
`Stuff+` (from `ridge_pred`) and `Location+` (from the pooled/count-relative location
map), computed from year-appropriate data. Deliberately **exclude the results
component**: the point is how much persistent skill the *physical* model captures, so
using a results-based predictor to explain a results-based target would be circular.

- drop in `sigma2_pitcher` = **stable skill Pitching+ captures**
- remaining `sigma2_pitcher` = **missing skill** (bucket 3), reported as a share of
  stable skill and as a share of total next-year variance.

Sanity cross-check: the standalone-validity framing must agree. Physical Pitching+
reaches ~0.39 correlation vs a ~0.55 criterion-reliability ceiling; the variance-
components "captured fraction" should land in the same neighborhood once converted.
Report both so they corroborate.

## Deliverable

- `component_model/analysis/12_reliability_decomposition.py`, following the numbered-
  script pattern: imports `fair_criterion.py`, reads `STUFFPLUS_DATA`/`STUFFPLUS_WORKDIR`,
  prints the A and B tables, writes no per-pitcher output to the repo.
- A verdict section appended to `component_model/RESULTS.md`: the three-bucket split
  for `xT` (percentages), the missing-skill number with its two denominators, the
  standalone-validity cross-check, and the interpretation gate outcome from A.
- Update `analysis/README.md` run-order table with script 12.
- Replication caveat: numbers are on the 2024–2026 panel; treat as current best
  estimate, re-run when more seasons land.

## Explicitly out of scope

- Part C (incremental-validity horse race to name the missing channel) — the natural
  follow-on once bucket 3's size justifies it.
- Per-pitch-type repetition beyond four-seam.
- Any change to `fair_criterion.py`, the fixed Stuff+ ridge, or the criterion.
