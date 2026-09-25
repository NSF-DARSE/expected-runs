# Secondary-pitch research pass, 2026-09: ledger

Scope: another look at the pitch types that have no confirmed grade or no Location+ at all,
under the frozen incremental gate. Everything in the "Pre-registration" section was written
and committed BEFORE any candidate's gate statistic was computed. The commit that adds this
section is the audit trail; results are appended below it in later commits and the
pre-registration text is not edited after that commit.

Sign convention for every number here: xT, location values, adjT and the prior are expected
run value from the pitcher's perspective, LOWER = BETTER. A grade is correctly oriented when it
correlates POSITIVELY with the future criterion (both are in the lower-is-better frame). None
of these quantities are "runs"; they are run value relative to an average pitcher.

## Where the headroom is, and where it is not

The honest question before choosing anything: which untested construct could plausibly pass a
gate whose binding constraint is criterion noise (script 12: ~70% of single-season variance in
the fair criterion is sampling noise)?

- **Stuff+ precision levers are not worth a read.** Shrinking or pooling a Stuff+ grade attacks
  noise in the grade, and the grade is not noisy: sinker and cutter Stuff+ repeat year over
  year at 0.77 and 0.84. A physics mean over 15+ pitches is already close to its asymptote;
  shrinkage by n would barely reorder pitchers. The binding noise is in the prior and the
  criterion, and both are frozen.
- **Sinker Stuff+ is locked** (blind `pooled_all` read on 2026->2027). No sinker Stuff+
  candidate is registered here.
- **Cutter Stuff+ is withheld** with physics that repeats and results that do not (tag audit,
  2026-09-10). No cutter Stuff+ candidate is registered here.
- **Location+ is the untested construct slot.** It has never been put through the frozen gate
  for any pitch but the four-seam. On four-seams it carries 4x Stuff+'s share of stable skill
  (script 12, 50.1% vs 12.2%). On secondaries the 2024->2025 read was reliable but invalid
  (slider 0.020, changeup 0.013, 100+ panel), and the 2025->2026 replication found the
  validities no longer ~0 (slider 0.058, changeup 0.150, curveball 0.245, all levels, 100+
  panel). The cutter is the most command-dependent pitch type in the public literature and our
  tag audit pointed the same way. Whether a per-type Location+ adds to a pitcher's own results
  is the one question with real, unmeasured headroom.
- **Location+ is also where a precision lever can bite.** Unlike Stuff+, a secondary
  Location+ is noisy at gate-pool sample sizes: the design dry run below puts the
  empirical-Bayes shrinkage constant k (the pitch count at which a pitcher's own mean gets half
  weight) at 45-137 pitches by type, against a 15-pitch pool floor. Two shrinkage priors are
  registered: the population, and the pitcher's own location on his OTHER pitch types (the
  "command is a pitcher trait" hypothesis).
- **Not registered, with reasons.** Multi-season pooling of the grade: the prior is one season,
  on pair C a 2024 grade season would be scored by an in-sample 2024 map, and script 12's lag-2
  retention (0.48) says an older season carries about half the signal; deferred. A
  stuff-conditioned location map (value of a location given the pitch's movement): a larger
  model with more cells than the secondary samples support; deferred. Criterion changes: frozen.
  Sweeper: no 2024 train rows.

## Pre-registration (committed before any read)

### Harness

`component_model/analysis/coach_location_gate.py`, maps and shrinkage in
`component_model/analysis/location_maps.py` (tests: `tests/test_location_maps.py`, 13 tests,
including exact reproduction of `fair_criterion.PooledLocationMap` in the catcher frame and
integer-weight = duplicated-row equality for the bootstrap).

The gate is copied from `coach_incremental_gate.py` unchanged: 50/50 z-blend of grade and prior,
statistic P(blend gain > 0) over 200 refits resampling train pitchers (frequency weights on the
map fit) and criterion pitchers separately, pool = 15+ pitches and 10%+ share in both seasons,
MIN_PITCHERS 60, bar 0.95, seed 20260817, criterion = next-season mean adjT on the type. Only
the grade differs: a pitcher's mean per-pitch location value in place of his mean ridge
prediction. Graded rows: the type's pitches with plate location, count, a Left/Right batter
side and pitcher hand, and non-null Target (one row filter for every candidate, so all
candidates on a pair share one pool).

Caches read (no rebuild; the harness exits if a cache is missing or would be rebuilt):
score build `workdir_ext3` (2024, 2025 D1), criterion build `workdir_ext3_crit` (2025, 2026 D1,
role-relabeled). Environment: STUFFPLUS_DATA = the 2024-2025 relspeed CSV, STUFFPLUS_DATA_CRIT
= the 2025-2026 realvelo v3 CSV, STUFFPLUS_WORKDIR / STUFFPLUS_WORKDIR_CRIT = those two
workdirs.

### Pairs: which is discovery, which is confirmation

- **C, confirmation: 2025 -> 2026.** Exactly the frozen gate's configuration: map trained on
  2024 (score build), grade on 2025 pitches, prior = 2025 adjT, criterion = 2026 adjT (criterion
  build). This is the pair the gate verdict is read on.
- **D, discovery: 2024 -> 2025**, inside the score build: grade on 2024 pitches, prior = 2024
  adjT, criterion = 2025 adjT. There is no 2023 to train on, so the 2024 map is CROSS-FITTED:
  pitchers are split into 5 fixed folds (seed 20260924, assigned from sorted PitcherId, blind to
  outcomes) and every pitcher's pitches are valued by a map fitted without him. Bootstrap
  weights apply inside every fold map.

### Candidates

All maps are fitted on xT (the shipped four-seam Location+ input), 0.25 ft cells with a 50-pitch
minimum, 0.5 ft fallback, then the platoon's overall mean (`location_maps.CellMap`).

- **CONTROL** (harness check, FF only). The shipped four-seam Location+: catcher-frame pooled
  map, no count, no shrinkage. Expected to pass on both pairs (script 03 increment P=0.994 /
  0.983 on the 100+ panel). If it fails on either pair the harness is suspect and no candidate
  on that pair is interpreted.
- **H1, per-type Location+** (SI, FC, SL, CB, CH). Per-type map in the `batter_platoon` frame:
  x mirrored so positive is inside for every batter, separate surfaces for same-side and
  opposite-side batters. Count-conditioned on the 12 counts, each count-cell shrunk toward
  pooled(location) + count effect with prior weight m; the per-pitch value is count-RELATIVE
  (value minus E[xT | count, platoon]) so count and platoon occupancy earn nothing (FRAMEWORK.md
  rule 1). m is tuned per type by the script-09 holdout on 2024 xT only (grid 1, 2, 5, 10, 25,
  100) and fixed in every refit. Tuned values, fixed now: SI 25, FC 100, SL 25, CB 25, CH 100
  (FF 5, used only inside H3). Grade = pitcher's mean relative value.
- **H2, H1 + population shrinkage.** grade = mu + n/(n + k) * (mean - mu), mu = mean relative
  value over the type's graded pitches, k = s2 / tau2, s2 = pooled within-pitcher variance of
  the relative value, tau2 = var(pitcher means) - mean(s2 / n) over pitchers with 15+ pitches,
  floored at 1% of var(pitcher means). All estimated from the grade season's location values,
  never an outcome, and re-estimated inside every refit. Dry-run k on pair C: SI 137, FC 91,
  SL 45, CB 52, CH 54.
- **H3, H1 + other-pitch shrinkage.** Same EB form, prior = a + b * (O - mean O), where O is
  the pitcher's mean standardised H1 value (each pitch divided by its own type's pitch-level
  SD) over his OTHER types among FF, SI, FC, SL, CB, CH in the grade season, fitted by OLS over
  pitchers with 15+ of the type and 15+ other pitches; tau2 is the residual variance around
  that prior (same floor). Pitchers without 15 other pitches shrink toward mu as in H2. The
  bootstrap resamples train pitchers across all six types, because all six maps are refit.
  Construct warning, stated before the read: H3 borrows location evidence from the pitcher's
  other pitches (mostly his four-seam). A pass would be partly a pitcher-level command score,
  not purely the pitch's own Location+; the audit below measures how much.

### Reads, in order, and the total

| # | read | pair | purpose |
|---|---|---|---|
| 1 | CONTROL | D | harness check (cross-fit harness) |
| 2 | H1 | D | discovery |
| 3 | H2 | D | discovery |
| 4 | H3 | D | discovery |
| 5 | CONTROL | C | harness check |
| 6 | H1 | C | confirmation, run whatever D shows (it is the construct question) |
| 7 | winner of H2 vs H3 | C | confirmation of one precision lever |
| 8 | AUDIT_O | C | only if H3 is read 7 and passes for any type: O alone as the grade |
| 9 | three-way audit | C | only if any type passes: results + Stuff+ + Location+ vs results + Stuff+ |

**Total: 7 fixed reads, at most 9.** Each read covers every target type of that candidate, so
the confirmatory per-type tests on C number 10 (H1 x 5, read 7 x 5). Under a global null that
is about 0.5 expected false passes at P >= 0.95, roughly a 40% chance of at least one, which is
why a pass here is "discovered", never "confirmed".

**Selection rule for read 7** (fixed now): of H2 and H3, the one with the higher MEDIAN
gain_mean on D across the five target types. Tie: H2 (fewer moving parts). The loser is never
read on C.

### Decision rules

- **Per type, a candidate is "discovered"** iff on C: P(blend gain > 0) >= 0.95 AND its
  location r > 0 (correct orientation), and on D: gain_mean > 0 for the same candidate and type.
  A C pass without the D direction is reported as "C pass, not replicated on D" and is not
  discovered.
- **Nothing ships from this pass.** A discovered type is a recommendation to Jack, subject to
  the decomposition audit: (a) which construct carries the gain (for H3, read 8 splits own-pitch
  from borrowed; for any pass, the grade's correlation with the type's shipped Stuff+ and with
  the prior), and (b) whether it survives next to Stuff+ in the combined score (read 9: equal-z
  three-way blend vs two-way, same bootstrap structure, P(gain > 0) reported, no bar implied).
- **No re-runs with tweaks.** Any change after a read is a new candidate, pre-registered in this
  file before it is read, and it counts toward the total. The cap is 10 reads for the pass.

### Contamination, stated

- Catcher-frame pooled per-type Location+ validities for SL, CH, CB were seen on BOTH pairs
  (script 09 and the 2026 replication, 100+ panel, not the gate pool). Batter-mirrored and
  platoon-conditioned variants were seen on D for SL and CH (~0.02-0.03). H1's exact
  construction (per-type, platoon-split batter frame, count-relative with the additive prior,
  gate pool, gate statistic) has not been read on either pair. The earlier looks were near zero
  on D, so any contamination biases the discovery pair against H1, not toward it.
- Four-seam Location+ increments were read on both pairs (script 03). The control is a harness
  check, not a test.
- Cutter and sinker Location+ have never been read on either pair.
- Before registration I ran the harness with `--dry` only: it builds maps and pools and prints
  pool sizes, tuned m, shrinkage k and map orientation from the xT training alone. It computes
  no correlation with the prior or the criterion. Those runs fixed the m values above and
  confirmed the C pools reproduce the gate's n exactly (FF 2007, SI 272, FC 190, SL 1263,
  CB 301, CH 622); D pools are FF 1839, SI 160, FC 79, SL 1145, CB 175, CH 493.
- Map orientation from the dry run, per-pitch relative value x100 (lower = better), agrees with
  how the pitches are used: slider to a same-side batter, away edge +2.2 vs inside edge +10.4;
  changeup to an opposite-side batter, away edge +1.9 vs inside edge +7.7. The heart is the
  cheapest region for every type because the xT surface charges balls; that is the shipped
  four-seam map's behaviour too.

## Results: discovery pair D (2024 -> 2025), reads 1-4

Appended after the reads; the pre-registration above is unchanged. Read count so far: 4.

| read | type | n | location r | prior r | semipartial | gain mean | 95% CI | P(gain>0) | refits used |
|---|---|---|---|---|---|---|---|---|---|
| 1 CONTROL | FF | 1839 | +0.218 | +0.164 | +0.164 | +0.062 | [+0.039, +0.082] | 1.000 | 200 |
| 2 H1 | SI | 160 | -0.050 | +0.037 | -0.065 | -0.047 | [-0.128, +0.044] | 0.110 | 200 |
| 2 H1 | FC | 79 | +0.061 | +0.031 | +0.054 | +0.002 | n/a | (0.500) | 2 |
| 2 H1 | SL | 1145 | +0.124 | +0.143 | +0.081 | +0.019 | [-0.007, +0.048] | 0.890 | 200 |
| 2 H1 | CB | 175 | +0.044 | +0.177 | -0.009 | -0.036 | [-0.105, +0.026] | 0.155 | 200 |
| 2 H1 | CH | 493 | +0.088 | +0.174 | +0.045 | -0.008 | [-0.050, +0.032] | 0.350 | 200 |
| 3 H2 | SI | 160 | +0.023 | +0.037 | +0.013 | -0.004 | [-0.073, +0.067] | 0.460 | 200 |
| 3 H2 | FC | 79 | +0.038 | +0.031 | +0.030 | +0.009 | n/a | (1.000) | 2 |
| 3 H2 | SL | 1145 | +0.113 | +0.143 | +0.073 | +0.014 | [-0.012, +0.042] | 0.825 | 200 |
| 3 H2 | CB | 175 | +0.024 | +0.177 | -0.027 | -0.047 | [-0.113, +0.016] | 0.070 | 200 |
| 3 H2 | CH | 493 | +0.068 | +0.174 | +0.024 | -0.021 | [-0.058, +0.017] | 0.140 | 200 |
| 4 H3 | SI | 160 | +0.053 | +0.037 | +0.043 | +0.019 | [-0.043, +0.087] | 0.710 | 200 |
| 4 H3 | FC | 79 | +0.099 | +0.031 | +0.094 | +0.039 | n/a | (1.000) | 5 |
| 4 H3 | SL | 1145 | +0.133 | +0.143 | +0.094 | +0.026 | [-0.003, +0.053] | 0.960 | 200 |
| 4 H3 | CB | 175 | +0.062 | +0.177 | +0.014 | -0.023 | [-0.088, +0.048] | 0.275 | 200 |
| 4 H3 | CH | 493 | +0.098 | +0.174 | +0.053 | -0.004 | [-0.045, +0.044] | 0.390 | 200 |

**The control passes** (P=1.000), so the cross-fitted D harness is sound.

**The cutter is unreadable on D, and I did not anticipate it.** Its D pool is 79 pitchers. The
gate's criterion-side resample keeps the unique pitchers drawn (about 63%, ~50), which is
below MIN_PITCHERS 60, so all but 2-5 of the 200 refits are discarded. The frozen gate has the
same property; it never bit before because every earlier read was on C (FC n=190). The FC rows
on D above are point estimates plus a handful of refits, and the parenthesised P values mean
nothing. Consequence under the pre-registered decision rule: the FC "gain_mean > 0 on D"
condition is mechanically met for all three candidates but carries no information, so I will
treat any FC pass on C as NOT replicated on D. That is the conservative reading and is decided
here, before the C reads.

**Shrinkage diagnostics on D.** H2 k (pitches for half weight on the pitcher's own mean): SI
107, FC 113, SL 47, CB 62, CH 49. H3 cross-pitch slope b is positive for every type (+0.014 to
+0.026 run value per SD of other-pitch location score): pitchers who locate their other pitches
well locate this one well too, in the correct direction.

**Selection for read 7 (pre-registered rule: higher median D gain_mean across the five types).**
H2 median -0.004, H3 median +0.019. **H3 goes to C; H2 is never read on C.** The choice does not
depend on the unreadable FC rows: on SI, SL, CB, CH alone the medians are H2 -0.013, H3 +0.007.

Discovery picture before confirmation: the only type with a positive, near-bar D signal is the
slider (H1 0.890, H3 0.960). Changeup and curveball are negative on D for every candidate even
though their location r is positive: their prior results are the stronger predictor on this
pair (+0.17 to +0.18) and the grade adds little beyond them. The sinker's own-pitch map is
wrongly oriented on D (H1 r -0.050) and only becomes positive when shrunk.

## Results: confirmation pair C (2025 -> 2026), reads 5-7

Run once each, in the registered order, after the D results and the H2-vs-H3 selection were
committed. Read count so far: 7 (all fixed reads done). Pools match the frozen Stuff+ gate's n
for every type.

| read | type | n | location r | prior r | semipartial | r(grade, prior) | gain mean | 95% CI | P(gain>0) | D gain mean | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 CONTROL | FF | 2007 | +0.175 | +0.150 | +0.122 | +0.428 | +0.042 | [+0.023, +0.063] | 1.000 | +0.062 | harness check passes |
| 6 H1 | SI | 272 | +0.244 | +0.243 | +0.167 | +0.365 | +0.048 | [-0.003, +0.107] | **0.965** | -0.047 | C pass, NOT replicated on D |
| 6 H1 | FC | 190 | -0.046 | +0.121 | -0.082 | +0.270 | -0.074 | [-0.143, +0.009] | 0.045 | (unreadable) | no |
| 6 H1 | SL | 1263 | +0.136 | +0.181 | +0.067 | +0.416 | +0.006 | [-0.023, +0.034] | 0.700 | +0.019 | no |
| 6 H1 | CB | 301 | +0.086 | +0.108 | +0.050 | +0.367 | +0.007 | [-0.052, +0.069] | 0.570 | -0.036 | no |
| 6 H1 | CH | 622 | +0.113 | +0.173 | +0.054 | +0.361 | -0.003 | [-0.043, +0.036] | 0.465 | -0.008 | no |
| 7 H3 | SI | 272 | +0.243 | +0.243 | +0.175 | +0.315 | +0.053 | [+0.002, +0.109] | **0.985** | +0.019 | **discovered** |
| 7 H3 | FC | 190 | +0.035 | +0.121 | +0.003 | +0.260 | -0.023 | [-0.093, +0.056] | 0.270 | (unreadable) | no |
| 7 H3 | SL | 1263 | +0.167 | +0.181 | +0.106 | +0.383 | +0.028 | [+0.002, +0.056] | **0.980** | +0.026 | **discovered** |
| 7 H3 | CB | 301 | +0.074 | +0.108 | +0.037 | +0.368 | -0.002 | [-0.055, +0.061] | 0.485 | -0.023 | no |
| 7 H3 | CH | 622 | +0.147 | +0.173 | +0.097 | +0.321 | +0.023 | [-0.018, +0.059] | 0.880 | -0.004 | no |

Every "location r" is positive except H1 FC, so orientation is correct wherever it matters: a
LOWER (better) location value goes with a LOWER (better) next-season run value.

**Verdicts under the pre-registered rule.** H3 (per-type Location+ shrunk toward the pitcher's
location on his other pitches) is DISCOVERED for the sinker (P=0.985) and the slider (P=0.980).
H1 passes the bar for the sinker on C (0.965) but its D gain is negative, so it is "C pass, not
replicated on D", not discovered. Nothing passes for the cutter, curveball or changeup. Of the
10 confirmatory per-type tests, 3 cleared 0.95 against about 0.5 expected by chance under a
global null; two of them are the same type (SI) under nested candidates. Both discovered rows
sit near the bar with CIs whose lower ends touch zero (+0.002). Discovered, not confirmed.

**The cutter.** The hypothesis that motivated this pass fails cleanly: cutter Location+ has no
validity on C (H1 r -0.046, H3 +0.035, n=190) and borrowing the pitcher's other-pitch command
does not rescue it. Physics repeats, location does not predict, results barely repeat. At this
sample there is no cutter grade of either kind to validate.

These two discoveries trigger the pre-registered audits (reads 8 and 9).
