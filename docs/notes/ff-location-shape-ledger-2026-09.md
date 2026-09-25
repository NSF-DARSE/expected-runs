# Four-seam Location+ by pitch shape, 2026-09: ledger

Scope: can the shipped four-seam Location+ be improved by valuing each location for the pitch's
own shape instead of for an average four-seam? Everything in "Pre-registration" was written and
committed BEFORE any candidate's gate statistic was computed. Results are appended below it in
later commits; the pre-registration text is not edited after that commit.

Sign convention for every number here: xT, location values, adjT, the prior and ridge_pred are
expected run value from the pitcher's perspective, LOWER = BETTER. A grade is correctly oriented
when it correlates POSITIVELY with the future criterion. None of these quantities are "runs";
they are run value relative to an average pitcher.

## Why shape, and what is already settled

The four-seam Location+ is the strongest score we ship (script 12: 50.1% of stable skill vs
Stuff+ 12.2%; the frozen gate passes it at P=1.000 on both pairs). Two ways of improving it have
already been read and do not move it, so they are not re-read here:

- Count-relative value (script 04): location-given-count ties the pooled map; the raw count
  map's gain is count occupancy, not location.
- Batter-mirrored and platoon-split frames (script 09): "the four-seam map is unaffected".

What has never been tried is the pitch's own shape. The shipped map values a location by what an
average four-seam does there, so a steep fastball at the letters is credited as if it were a
flat one. A criterion-free look at the 2024 train rows (xT only, no prior, no criterion; the
scratch script is not committed) shows the interaction is real and sits mostly OUTSIDE the zone:

| xT x100, middle third of the plate | 0.5-1.5 ft | 1.5-2.0 | 2.0-2.5 | 2.5-3.0 | 3.0-3.5 | 3.5-4.0 | 4.0-5.0 |
|---|---|---|---|---|---|---|---|
| steepest third of approach angle | +4.92 | -4.16 | -2.78 | -3.32 | -5.06 | +1.94 | +8.97 |
| flattest third | +6.29 | -6.25 | -5.40 | -6.08 | -6.99 | -1.82 | +6.79 |
| flat minus steep | +1.37 | -2.09 | -2.62 | -2.76 | -1.94 | **-3.76** | -2.18 |

Flat fastballs are better everywhere (a level effect the candidates must NOT reward, since that
is Stuff+), and better by the most just above the zone, where a flat fastball gets chased and a
steep one is a ball. Below the zone the order reverses. Arm-side run shows a smaller interaction
by batter side: to a same-side batter, more run makes the inside edge cheaper (+2.85 least run,
+0.09 most run); to an opposite-side batter, more run makes the inside edge dearer (+3.45 to
+5.10).

Expectation, stated before the read: the headroom is small. At the grade level the candidates
correlate 0.98 (SHAPE_V) and 0.88-0.90 (SHAPE_VH) with the shipped grade on both pairs, so a
real gain would be a few thousandths of blend r. The likely outcome is a tie.

## Pre-registration (committed before any read)

### Harness

`component_model/analysis/coach_ff_location_shape_gate.py` (tests:
`tests/test_ff_location_shape.py`). It imports the frozen gate's constants and statistic from
`coach_location_gate.py` and the map machinery from `location_maps.py`, both from the secondary
research pass (this branch is stacked on it). Gate: 50/50 z-blend of grade and prior, 200
refits resampling train pitchers (frequency weights on every map fit) and criterion pitchers
separately, pool 15+ pitches and 10%+ share in both seasons, MIN_PITCHERS 60, seed 20260817,
criterion = next-season mean four-seam adjT. Pairs as in the secondary ledger: C (2024 map,
2025 grade, 2026 criterion) is confirmation; D (2024 grade from a 5-fold cross-fitted 2024 map,
fold seed 20260924, 2025 criterion) is discovery.

The comparison is PAIRED: every candidate and the control are refit on the same 200 draws, and
the statistic is the difference in blend gain, Delta = gain_candidate - gain_CONTROL, per draw.
Graded rows are identical for every candidate (the secondary harness's row filter), so every
candidate is scored on one pool.

Caches read (no rebuild; the harness exits if a cache is missing or would be rebuilt): score
build `workdir_ext3`, criterion build `workdir_ext3_crit`. Same environment variables as the
secondary pass.

### Candidates

All maps are fitted on xT with `location_maps.CellMap`, count=False: 0.25 ft cells with a
50-pitch minimum, 0.5 ft fallback, then the surface's own mean. Shape bins are folded into the
platoon code, so each (bin, platoon) gets its own surface and the per-pitch value is relative
to that surface's mean, E[xT | location, bin] - E[xT | bin]. A pitcher earns nothing for
throwing a flatter or harder fastball; only for putting his fastball where fastballs of its
shape succeed.

- **CONTROL.** The shipped four-seam Location+: catcher frame, one surface. Harness check: it
  uses the secondary harness's rows, pool and draws, so it must reproduce that pass's read 1
  (D: n 1839, gain +0.062, CI [+0.039, +0.082], P 1.000) and read 5 (C: n 2007, gain +0.042,
  CI [+0.023, +0.063], P 1.000) exactly. If it does not, nothing on that pair is interpreted.
- **SHAPE_V.** Catcher frame, one surface per tercile of height-adjusted approach angle
  (`vaa_flat`, `coach_si_feature_gate.add_vaa_flat`: release speed, extension, induced vertical
  break and release height only; the pitch's own plate location never enters). Cut points from
  the 2024 train rows, fixed in every refit: -6.25 and -5.77 degrees.
- **SHAPE_VH.** Batter_platoon frame (x mirrored so positive is inside; separate same-side and
  opposite-side surfaces, which is what makes arm side vs glove side known), one surface per
  vaa_flat tercile x arm-side horizontal break tercile x platoon: 18 surfaces. HorzBreak_arm
  cut points 8.79 and 13.52 in. The frame change is not an attribution confound worth a
  separate read: script 09 found the four-seam map unaffected by it.

A row missing any shape input (0.04% of rows) goes to its own bin (one extra surface per platoon).

### Reads and the total

| # | read | pair | purpose |
|---|---|---|---|
| 1 | CONTROL, SHAPE_V, SHAPE_VH jointly | D | discovery; harness check |
| 2 | CONTROL, SHAPE_V, SHAPE_VH jointly | C | confirmation; harness check |

**Total: 2 reads, cap 4.** Read 2 runs whatever read 1 shows. The confirmatory tests on C are
SHAPE_V vs CONTROL and SHAPE_VH vs CONTROL: under a global null, about a 10% chance that at
least one clears 0.95 by luck.

### Decision rules

- **A candidate "improves on the shipped Location+"** iff on C: P(Delta > 0) >= 0.95, and on
  D: mean Delta > 0. A C pass without the D direction is "C pass, not replicated on D".
- **Tie.** |mean Delta| under 1 SE on C is a tie, whatever P says. A tie keeps the shipped map:
  switching re-grades every pitcher a coach has already seen, which needs evidence, not parity.
- **Audit, reported for every candidate, no bar:** (a) r(grade, four-seam Stuff+ grade) against
  CONTROL's; a rise of more than 0.10 is a construct warning (the shape conditioning would be
  leaking stuff into Location+). (b) The three-way equal-z blend, results + Stuff+ + Location+,
  paired against the same blend with CONTROL, on the same draws, Stuff+ held at its full-sample
  fit (the shipped 2024 ridge; the location maps are what is being compared). An adoption
  recommendation needs this difference > 0 on C. (c) On D, year-over-year reliability of the
  grade (2024 vs 2025 pitches, both valued by the pitcher's fold-excluded 2024 map), paired
  pitcher bootstrap against CONTROL; a candidate more than 1 SE less reliable is flagged.
- **Nothing ships from this pass.** An improvement is a recommendation to Jack.
- **No re-runs with tweaks.** Any change after a read is a new candidate, registered here
  before it is read, and counts toward the cap of 4.

### Contamination, stated

- CONTROL has been read on both pairs (script 03; secondary pass reads 1 and 5). It is a harness
  check here, not a test.
- No shape-conditioned location map has been read on either pair, by any script.
- Before registering I ran the harness with `--dry` on both pairs (maps, cut points, grade
  correlations between candidates, top-vs-bottom-of-zone values; no prior, no criterion) and the
  scratch xT table above (2024 train rows' own xT only). Those fixed the cut points and the
  expectation of a small effect. Nothing else was looked at.

## Results: discovery pair D (2024 -> 2025), read 1

Appended after the read; the pre-registration above is unchanged. Read count: 1 of 2 (cap 4).
Read 2 (pair C) was started before these results were seen and is not yet looked at.

| candidate | n | location r | semipartial | r(grade, Stuff+) | gain mean | 95% CI | P(gain>0) | Delta vs CONTROL (SE) | P(Delta>0) | 3-way Delta (SE) | reliability r |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CONTROL | 1839 | +0.218 | +0.164 | +0.193 | +0.062 | [+0.039, +0.082] | 1.000 | | | | 0.352 |
| SHAPE_V | 1839 | +0.208 | +0.153 | +0.110 | +0.056 | [+0.033, +0.077] | 1.000 | -0.006 (0.003) | 0.030 | +0.000 (0.002) | 0.332 (-0.020, SE 0.008) |
| SHAPE_VH | 1839 | +0.186 | +0.131 | +0.099 | +0.044 | [+0.023, +0.066] | 1.000 | -0.018 (0.006) | 0.000 | -0.009 (0.005) | 0.262 (-0.092, SE 0.014) |

Two-way blend (results + Stuff+, no Location+) on this pool: r +0.234. Reliability is the
2024-to-2025 grade on 1850 pitchers with 15+ four-seams in both seasons.

**Harness check passes.** CONTROL reproduces the secondary pass's read 1 exactly (n 1839, gain
+0.0615, CI [+0.0393, +0.0821], P 1.000).

**Neither candidate can improve on D.** Both are below the shipped map on the blend gain
(SHAPE_V by about 1.9 SE, SHAPE_VH by about 2.8 SE) and on reliability. Under the registered
rule the D direction fails for both, so neither can be "improves" whatever C shows.

**What the decomposition says.** Valuing each location for the pitch's own shape cuts the
grade's correlation with Stuff+ from 0.19 to 0.11: about 40% of the shipped Location+'s overlap
with Stuff+ comes from the map crediting pitchers whose fastballs are shaped to play where
fastballs play well. That overlap is real predictive signal (standalone validity drops with it)
and it is stable (reliability drops with it), but it is signal Stuff+ already carries. In the
three-way blend SHAPE_V ties CONTROL (+0.0004, SE 0.0023). So the shipped Location+ leans a
little on stuff, and inside Pitching+ that costs nothing. SHAPE_VH loses on every count; its 18
surfaces thin the cells enough that map noise shows up as lost reliability.

## Results: confirmation pair C (2025 -> 2026), read 2

Run once, as registered. Read count: 2 of 2 (cap 4). The shape line is closed.

| candidate | n | location r | semipartial | r(grade, Stuff+) | gain mean | 95% CI | P(gain>0) | Delta vs CONTROL (SE) | P(Delta>0) | 3-way Delta (SE) | P(3-way Delta>0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CONTROL | 2007 | +0.175 | +0.122 | +0.105 | +0.042 | [+0.023, +0.063] | 1.000 | | | | |
| SHAPE_V | 2007 | +0.157 | +0.104 | +0.025 | +0.032 | [+0.014, +0.052] | 1.000 | -0.010 (0.003) | 0.000 | -0.004 (0.003) | 0.085 |
| SHAPE_VH | 2007 | +0.144 | +0.093 | +0.009 | +0.024 | [+0.004, +0.044] | 0.990 | -0.018 (0.007) | 0.005 | -0.009 (0.005) | 0.030 |

Two-way blend (results + Stuff+) on this pool: r +0.218; with CONTROL +0.242.

**Harness check passes.** CONTROL reproduces the secondary pass's read 5 exactly (n 2007, gain
+0.0417, CI [+0.0225, +0.0631], P 1.000).

**Verdict: the shipped four-seam Location+ stays.** Both shape candidates are worse than it on
both pairs, by 2-3 SE on the blend gain, and neither is "improves" under any reading of the
rules. Conditioning a four-seam's location value on its shape removes most of the grade's
overlap with Stuff+ (0.105 to 0.025 on C, 0.19 to 0.11 on D), and what it removes is useful:
on C it costs a little even inside the three-way blend (-0.004, about 1.4 SE), so part of what
the shipped map credits is not fully carried by the four-seam Stuff+ ridge either. The
interaction the criterion-free table showed (flat fastballs gain the most just above the zone)
is real in xT, but grading pitchers on it measures less of what predicts next season than
grading them on the average fastball's map does.

What this closes, with the earlier reads: for the four-seam, count (script 04), batter frame
(script 09) and pitch shape (here) have all been tried against the pooled catcher-frame map
and none beats it. Remaining headroom for four-seam Location+ is precision (sample size), not
the map's conditioning.
