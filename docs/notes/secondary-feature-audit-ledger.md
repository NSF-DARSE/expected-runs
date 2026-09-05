# Secondary-pitch feature audit — ledger (2026-09-05)

Question from the pitching coach (2026-08-17 meeting): one staff splitter/changeup grades far
below where he ranks it, and the "vs his fastball" differentials were the suspected cause. The
sinker loop (`sinker-cutter-loop-ledger.md`) also left two model-side threads open for the
secondaries: movement-vector geometry (the one sinker gain) and whether the release-deviation
terms carry off the four-seam. Rather than patch one pitch, all three were posed as RULES and
asked of every secondary type at once, pre-registered, on the frozen gate machinery.

Two scripts, both new, neither touching the gate, the criterion, or any shipped score:

- `component_model/analysis/coach_secondary_feature_audit.py` — CH, SL, CB, SW; seven feature
  candidates per type (eight for CH); paired on 200 shared cluster-bootstrap draws against the
  type's shipping list. "base" replicates `coach_incremental_gate.json` row for row.
- `component_model/analysis/coach_gb_criterion_shadow.py` — shadow evaluation of two
  ground-ball-conditioned criteria on all six types, blind, eligible types read first.

Sign convention: adjT and every grade are expected run value, pitcher's perspective, LOWER =
BETTER; a correctly oriented grade correlates POSITIVELY with the criterion. Standardized
coefficients below are in that frame: NEGATIVE = more of the feature predicts fewer runs.

Decision rules were fixed in each script's docstring before it ran: ADD recommended only at
P(Δstuff_r > 0) ≥ 0.95 AND P(Δsemipartial > 0) ≥ 0.95 AND the Target-criterion delta agreeing
in sign; DROP recommended only at P(Δstuff_r > −0.01) ≥ 0.95 (non-inferiority, the release-gate
margin). Everything else is "no recommendation".

## Part 1 — feature audit (the coach's question, as a rule)

Δ = candidate minus the type's shipping list, paired on the same draws. Gate P is the type's
own P(blend gain > 0) under that candidate (all eligible types stay PASS under every
candidate except where noted).

### Changeup (pooled ChangeUp + Splitter; n=622 pitchers, 154k train rows)

| candidate | stuff_r | Δstuff_r [95% CI] | P(Δ>0) | P(Δ>−.01) | Δsemipartial | gate P | verdict |
|---|---|---|---|---|---|---|---|
| base (shipping list) | +0.185 | — | — | — | — | 1.000 | — |
| −diff (drop all three differentials) | +0.102 | −0.081 [−0.123, −0.035] | 0.000 | 0.000 | −0.073 | **0.625 (fails gate)** | costs validity |
| −breakdiff (keep velo diff only) | +0.181 | −0.001 [−0.018, +0.016] | 0.41 | 0.86 | −0.001 | 1.000 | not shown non-inferior |
| −dev (drop release-deviation V terms) | +0.183 | −0.001 [−0.013, +0.013] | 0.43 | 0.92 | +0.000 | 1.000 | not shown non-inferior |
| +movgeo | +0.189 | +0.004 [−0.003, +0.012] | 0.84 | 1.00 | +0.004 | 1.000 | no |
| +movgeo+angsq | +0.190 | +0.005 [−0.003, +0.014] | 0.84 | 1.00 | +0.005 | 1.000 | no |
| −diff+movgeo | +0.110 | −0.074 [−0.118, −0.027] | 0.000 | 0.000 | −0.066 | 0.670 | no |
| +spin (restore SpinRate) | +0.195 | +0.009 [−0.010, +0.025] | 0.87 | 0.96 | +0.009 | 0.995 | no |

The coach's hypothesis is **rejected for the changeup model**: the differentials are the single
most important feature family on a changeup. Dropping them cuts validity nearly in half and
takes the type from PASS to a gate failure. Almost all of that is `velocity_differential`
(standardized coef +0.0063, the largest term in the model: a changeup further below the
fastball predicts fewer runs). The two break differentials are near-inert on a changeup
(−breakdiff Δ ≈ 0) but the non-inferiority bar was not reached at this sample.

Restoring SpinRate would buy ≈ +0.009 validity (P=0.87, not significant). The 2026-08-17
construct decision to drop it (the coefficient rewards high spin; the coach wants low spin on a
cambio) costs at most that, and the decision stands.

### Slider (n=1263 pitchers, 289k train rows)

| candidate | stuff_r | Δstuff_r [95% CI] | P(Δ>0) | P(Δ>−.01) | Δsemipartial | verdict |
|---|---|---|---|---|---|---|
| base | +0.179 | — | — | — | — | — |
| −diff | +0.169 | −0.010 [−0.022, +0.000] | 0.035 | 0.56 | −0.011 | costs validity |
| −breakdiff | +0.177 | −0.001 [−0.008, +0.005] | 0.28 | **0.995** | −0.002 | **DROP-OK** (non-inferior) |
| −dev | +0.166 | −0.012 [−0.024, −0.001] | 0.005 | 0.39 | −0.010 | costs validity |
| +movgeo | +0.180 | +0.001 [−0.003, +0.005] | 0.77 | 1.00 | +0.001 | no |
| +movgeo+angsq | +0.182 | +0.003 [−0.002, +0.007] | 0.89 | 1.00 | +0.002 | no |
| −diff+movgeo | +0.171 | −0.008 [−0.019, +0.004] | 0.07 | 0.64 | −0.009 | no |

On the slider the release-deviation V terms DO carry (dropping them costs −0.012, P(worse)=0.995),
and the velocity differential carries; the two break differentials are shown inert (the one
DROP-OK in the audit). Not acted on: a slider-only feature list diverging from the other
breaking balls for a ≈0 gain is not worth the explanation.

### Curveball (n=301 pitchers, 84k train rows)

| candidate | stuff_r | Δstuff_r [95% CI] | P(Δ>0) | P(Δ>−.01) | Δsemipartial | verdict |
|---|---|---|---|---|---|---|
| base | +0.310 | — | — | — | — | — |
| −diff | +0.287 | −0.017 [−0.061, +0.027] | 0.20 | 0.39 | −0.018 | costs validity |
| −breakdiff | +0.307 | +0.002 [−0.028, +0.034] | 0.52 | 0.80 | +0.002 | not shown non-inferior |
| −dev | +0.304 | −0.002 [−0.023, +0.020] | 0.38 | 0.83 | −0.002 | not shown non-inferior |
| +movgeo | +0.303 | −0.007 [−0.030, +0.009] | 0.24 | 0.71 | −0.007 | no |
| +movgeo+angsq | +0.308 | −0.004 [−0.029, +0.012] | 0.38 | 0.77 | −0.004 | no |
| −diff+movgeo | +0.280 | −0.024 [−0.069, +0.020] | 0.15 | 0.29 | −0.025 | no |

Curveball is the smallest eligible pool and the CIs are wide; nothing moves it in either
direction. Movement geometry is slightly negative here (the break vector on a curveball is
already well described by IVB + HB magnitudes).

### Sweeper

Skipped: 0 train rows. The `Sweeper` tag enters the D1 feed in 2025, so there is no 2024
training season. It cannot be modelled until the 2025→2026 pair is the score frame.

### Part 1 verdict

- **Keep the differentials on every secondary.** The rule the coach asked about goes the other
  way: "how far below the fastball" is the strongest single term on a changeup, and it carries on
  slider and curveball too.
- **Movement geometry does not generalise from the sinker.** +0.001 to +0.005 on CH/SL, negative
  on CB; nowhere near the bar. The sinker gain was a sinker fact.
- **Release-deviation terms stay** (they carry on SL; neutral elsewhere).
- Nothing ships from Part 1. The shipping FEATS_BY_PITCH lists are the right lists.

## Part 2 — the finding that actually explains the coach's complaint

While setting up the staff readout for the flagged splitter, the deployed pitcher-page grade
turned out NOT to be the model Part 1 audited.

`arsenal.fit_type` (called by `14_pitcher_pages.py` for every `ar.PITCH_TYPES` entry) calls
`fc.stuff_ridge(pit, pitch_mask=mask)` **without `feats=`**, so every pitch type on the pitcher
page is graded on the four-seam feature list (`fc.FEATS`: SpinRate in, differentials out), and
`Splitter` is fitted as its own tag-only ridge (`PITCH_TYPES` lists it separately from
`ChangeUp`), scaled on the ~20 pitchers with 100+ splitters in the season. `FEATS_BY_PITCH` /
`feats_for()` and the `PITCH_GROUPS` "CH" pooling exist in `fair_criterion.py` and are what the
gate, the eligibility verdicts (contract v3) and Part 1 above were all computed on, but nothing
in the page-building path calls them. `build_model_artifact` ships one global `featureOrder =
fc.FEATS`, which is why the gap was invisible: the artifact is internally consistent, just with
the wrong model.

Consequences, measured locally on the deploy frame (2025→2026 pair, all levels, DEL_BLU staff,
no names here; the per-pitcher printout stays in the scratch workdir):

| grade for the flagged splitter | model | staff rank among CH-group rows ≥15 pitches |
|---|---|---|
| **81** (what the page shows) | splitter-only ridge, four-seam feature list, 20-pitcher scale | 12th of 14 |
| 88 | pooled CH group, four-seam feature list | 9th of 16 |
| **103** | pooled CH group, `feats_for("CH")` — the validated model | **2nd of 16** |
| 107 | pooled CH group, `feats_for("CH")` minus differentials | 4th of 16 |

The deployed 81 is a SpinRate penalty: the four-seam list carries SpinRate with a "more spin is
better" coefficient, and a splitter is a low-spin pitch by design. Under the model the gate
validated, the flagged splitter is the second-best changeup-group pitch on the staff — which is
where the coach put it. His reading was right; the differentials were the wrong suspect.

The same gap applies to every non-four-seam type on the page: the eligibility verdicts in the
contract describe models the page does not use. Curveball and slider grades on the page carry
no differentials and DO carry SpinRate; the audited/validated models are the reverse.

**This is a wiring defect, not a modelling question, and it is NOT fixed on this branch.**
Fixing it changes the JSON contract: `model.featureOrder` is one global list today and the
app's `TraitPanel`/`PitchTooltip` index `typical`, `percentiles` and `pitch.f` positionally
against it. Two ways to close it, Jack's call:

- (A) per-type `featureOrder` inside `model.byPitchType[t]`, arsenal rows and pitch `f` arrays
  positional against their own type's order; app reads the type's order. Honest contract,
  touches publisher schema + app + fixtures.
- (B) keep one global `featureOrder` = union of all per-type lists; each type's `coef`,
  `scalerMean`, `scalerScale`, `populationMeanZ` are padded (coef 0, population moments) for
  features its model does not use. App contract unchanged; the trait panel would show 0-point
  rows for unused features unless it hides zero-coefficient features.

Either way Splitter should be graded by the pooled CH model (train pooled, display by tag) per
the 2026-08-17 pooling decision, with the scale from the pooled group's qualifying pitchers.

## Part 3 — GB-conditioned criterion shadow (Jack's option 2, run blind)

Two shadow criteria, both leaving every non-ground-ball pitch exactly as the frozen criterion
values it: `gb_realized` (ground balls keep realized Target — the ceiling of any GB map, luck
included) and `gb_fine` (GB-only EV/LA grid at 2.5 mph × 5°, cell n≥30, 97–98% coverage,
fallback to frozen). Both passed through the same opponent adjustment on both frames. The
grade never changes; only what it is measured against.

Readout step 1 (eligible types must be non-inferior on both criterion reliability and grade
validity, Δ > −0.01 at P ≥ 0.95, before SI is even read):

| type | gb_realized Δstuff_r (P>−.01) | Δresults_r (P>−.01) | gb_fine Δstuff_r (P>−.01) | Δresults_r (P>−.01) |
|---|---|---|---|---|
| FF | −0.033 (0.000) | −0.020 (0.21) | −0.004 (1.00) | +0.004 (1.00) |
| SL | −0.009 (0.52) | −0.064 (0.005) | −0.001 (1.00) | −0.001 (1.00) |
| CB | −0.037 (0.12) | +0.052 (0.96) | −0.011 (0.44) | +0.002 (1.00) |
| CH | −0.069 (0.000) | −0.044 (0.10) | −0.003 (1.00) | −0.009 (0.58) |

**Neither shadow criterion carries forward.** `gb_realized` degrades four-seam and changeup
validity outright and slider criterion reliability; `gb_fine` is close to a wash everywhere
but fails non-inferiority on curveball validity and changeup reliability. Step 2, for the
record only: under `gb_realized` the SI criterion's own reliability falls from +0.241 to
+0.139 (the GB residual is noise, as `coach_si_change_tests.py` TEST B said) and SI's gate P
moves 0.29 → 0.44; under `gb_fine` it moves 0.29 → 0.27. FC: 0.78 → 0.56 / 0.77.

Conclusion: the criterion stays frozen, on evidence rather than by rule. The sinker path (a)
from the sinker ledger is closed at this sample; path (b), the 2026→2027 pair, is what remains.

## Outputs (score workdir, never committed)

`coach_secondary_feature_audit.json`, `log_secondary_feature_audit.txt`,
`coach_gb_criterion_shadow.json`, `log_gb_criterion_shadow.txt`. Run either script with
`STUFFPLUS_DATA`, `STUFFPLUS_WORKDIR` (2024/2025 build) and `STUFFPLUS_WORKDIR_CRIT`
(2025/2026 build) set; `AUDIT_GROUPS=SW` reruns one type and merges.
