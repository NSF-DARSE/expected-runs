"""Is the pooled ridge's RelSide sign an artifact of mixing two handednesses?

Adapted from coach_horzbreak_hand.py, same four parts, same three verdicts. Read that
script first; only the differences are justified here.

BACKGROUND: the feature-importance run found two features where the coach's card and our
pooled ridge assign OPPOSITE signs -- HorzBreak, and RelSide. HorzBreak was tested and came
back "not established" for his term (see RETRACTED in the README). RelSide was never tested
and has the same structure, with one difference that makes it worse: a left-hander releases
from the opposite side of the rubber, so RelSide is close to a handedness INDICATOR on its
own. Our ridge pools both hands with only an additive is_lhp flag, so a single pooled
RelSide coefficient is largely fitting handedness rather than release mechanics. His card
uses per-hand weight tables and does not have this problem.

WHY THE COLLINEARITY IS THE HEADLINE HERE AND WAS NOT FOR HORZBREAK. Both features are
mirrored by hand, but they differ in how much within-hand variation survives. HorzBreak:
hand means +10.6 / -11.4 against a within-hand SD of ~5.4, so roughly a quarter of the
variance is within-hand. RelSide: the hand gap is several feet against a within-hand SD
under a foot, so almost none is. Part A measures this rather than assuming it, because it
decides whether the POOLED coefficient in part B is estimable at all. If RelSide and is_lhp
are collinear past the point of identification, the shipping coefficient is a ridge-penalty
artifact -- an arbitrary split of shared variance between two columns -- and `notes/`
already records one withdrawn reading of exactly that kind (joint models over scores
correlated 0.70-0.94, coefficients "are not identified and the reading was withdrawn").
Reporting it as "release mechanics" would repeat that error.

WHAT MIRRORING IS, AND WHY IT IS NOT THE SAME MOVE AS THE EXTENSION/RelSpeed SWAP.
`RelSide_mirrored = RelSide * (1 - 2*is_lhp)` is a PRODUCT of two model columns, so it is
not in the span of {RelSide, is_lhp}. Mirroring is therefore a genuinely different model,
not a reparameterisation, and part D's out-of-sample check is the right test of it. It also
de-confounds: mirroring collapses the between-hand gap, so the mirrored column is close to
orthogonal to is_lhp and its coefficient becomes estimable where the pooled one is not.
That is the most informative cell in part B's table for this feature.

MIRRORING RelSide ALONE IS A CHOICE, AND IT IS DELIBERATE. An arm-side frame would flip
HorzBreak too. This script flips only RelSide so the result is attributable to RelSide;
MIRROR_ALSO below flips the break columns as well for anyone who wants the joint version.
Do not read the two as interchangeable.

THE MIRROR'S OWN ASSUMPTION, WHICH IS WEAKER HERE THAN FOR BREAK: mirroring asserts that a
lefty at X feet arm-side behaves like a righty at X feet arm-side. Geometrically fine; not
outcome-preserving, because the batter-handedness mix is NOT symmetric across pitcher hands
(the league is majority RHB, so mirrored-equivalent release points face different platoon
mixes). README open item #2 already flags is_lhb riding in the model for a related reason.
For HorzBreak the residual asymmetry is small next to the break signal. For RelSide, where
most of the raw spread IS the hand, more of what survives mirroring may be platoon mix than
release mechanics. Part D is the check; a null there is not evidence the mirror is right.

THE INTERACTION VARIANT, AND WHY IT SITS BETWEEN POOLED AND MIRRORED (Jack, 2026-08-16).
Add `RelSide * is_lhp` to the pooled model and KEEP is_lhp. This gives RelSide a separate
slope per hand (RHP = the RelSide coef; LHP = RelSide coef + interaction coef) while the
level gap between hands stays in the is_lhp coefficient. So it answers the exact worry that
a hand-aware RelSide term might smuggle in "the value of being left-handed": that value is
is_lhp's job here, and the interaction carries only the per-hand RESPONSE to release side.
It NESTS the mirror -- mirroring is the special case where the two per-hand slopes are equal
and opposite in the raw frame -- so the interaction is strictly more flexible and is the
right model when you are unwilling to assume the mirror symmetry. The catch (README item #2):
the clean level/slope split holds only while is_lhp remains in the model; drop it and the
interaction starts absorbing the level gap too. Non-linear terms (splines on raw RelSide)
were considered and rejected for now -- the raw feature is bimodal by hand, so a spline
mostly re-learns the handedness split; non-linearity is worth revisiting only AFTER RelSide
is in a hand-neutral frame, not as a fix for the confounding.

FOUR PARTS (as in the HorzBreak script; parts B and D gain the interaction variant):
  A. Descriptive: is RelSide mirrored by hand, and how collinear is it with is_lhp?
  B. Refit the reference ridge: pooled, RHP-only, LHP-only, mirrored-pooled, and the
     interaction (per-hand slopes in one pooled model, is_lhp kept for the level).
  C. His isolated RelSide term, re-evaluated within each hand.
  D. Does re-parameterising RelSide help predictive validity? Mirrored AND interaction,
     each against pooled.

THREE CHANGES TO THE INHERITED DESIGN, all of which also apply to the HorzBreak script:

  1. THE VERDICT NOW TURNS ON THE RHP-vs-LHP DIFFERENCE, NOT ON TWO SEPARATE SIGN TESTS.
     coach_horzbreak_hand.py calls RHP-vs-LHP "the decisive comparison" (its line 198),
     computes it, prints it, stores it as `rhp_minus_lhp` -- and then `verdict_from` never
     reads it, gating instead on each hand's coefficient being individually distinguishable
     from zero. Those are different questions: two coefficients can both sit within a SE of
     zero and still be confidently different FROM EACH OTHER, which is precisely what
     "pooling averages two opposite relationships" means. On the HorzBreak run that
     mattered -- RHP -0.0006 (P(>0)=0.25, ambiguous) and LHP +0.0036 (P(>0)=1.000) returned
     "not established", while the difference it ignored was -0.0042 +/- 0.0015, P~1.000.
     The confounding WAS established; the verdict function could not say so.

  2. PART B BOOTSTRAPS EVERY COEFFICIENT, NOT JUST THIS FEATURE'S. The cluster bootstrap
     already refits the whole model on each draw, so the other eleven standardised
     coefficients are sitting in `coef_` and get thrown away. Keeping them is free and
     answers a question the coach page needs answered: the page prints a plain-words
     direction ("more is better" / "less is better") for all twelve features with no error
     bar anywhere, and the smallest of them are small enough that the direction may be
     noise. HorzBreak's own pooled CI, [-0.0027, +0.0033], straddles zero -- yet the page
     states "Horizontal break, less is better" as fact. `direction_established` per feature
     is emitted so the page can gate its labels instead of guessing.

  3. THE PARTS GET SEPARATE VERDICTS. B is powered and C and D are not (see POWER), so one
     combined verdict lets the underpowered parts silently veto the powered one. That is
     what produced the "not established, limited by the LHP pool" line in the README, which
     misattributes the cause: part B's LHP estimate was the CONFIDENT one (P=1.000) and the
     RHP estimate was the ambiguous one.

POWER, STATED BEFORE THE RUN because it decides whether the run is worth making. Taking the
HorzBreak run's measured SEs as the prior, at the same floors and the same pools:
  Part B  WELL POWERED. Fits on ~1.35M pitches; the cluster bootstrap resamples PITCHERS,
          and the training-year LHP pool is far larger than the 219 that survive part C's
          both-seasons join. Coefficient SEs came in at 0.0009-0.0015 and the hand
          difference resolved at P~1.000. For RelSide, whose hand split is larger than
          HorzBreak's, part B should resolve at least as sharply.
  Part C  UNDERPOWERED, and not marginally. LHP n=219 gave a bootstrap SE of 0.0758 on the
          correlation, so nothing under |r| ~ 0.125 can clear P>=0.95. For scale, the
          ENTIRE twelve-feature ridge correlates 0.112 with this criterion: a single
          isolated scorecard term would have to out-predict the whole model to register.
          Detecting a realistic |r| ~ 0.05 needs roughly 1,080 LHP; the pool has 219.
          Expect "not established" and do not read it as evidence of no effect.
  Part D  UNDERPOWERED. HorzBreak's mirrored-minus-pooled validity gain was +0.0037 +/-
          0.0075; clearing P>=0.95 needs ~+0.0123, an 11% relative improvement in validity
          from re-parameterising one feature. RelSide's shipping coefficient is smaller
          than HorzBreak's (0.00022 vs 0.00060), so expect a smaller gain and a null.
So: this run can establish whether OUR pooled RelSide coefficient is confounded, and cannot
establish anything about HIS term or about validity. That is still worth the electricity --
the first question is the one the coach page makes a claim about -- but the other two
should be reported as unresolved by design, not spun as ties.

SIGN CONVENTIONS (fair_criterion.py; unchanged from the HorzBreak script, do not guess):
  - Target, xT, adjT and ridge_pred are expected run value from the PITCHER's perspective,
    LOWER = BETTER, relative to an average pitcher. Never "runs" or "runs allowed".
  - RAW TRAITS (a physical measurement, or an isolated scorecard term in the scorecard's
    own native higher-is-better units, as in part C) predict better outcomes when they
    correlate NEGATIVELY with the criterion.
  - FITTED PREDICTIONS of run value (part D) are already in the criterion's frame, so a
    correctly oriented prediction correlates POSITIVELY. Do not apply the raw-trait rule.
  - A standardised ridge coefficient is on run value, so NEGATIVE coefficient = more of the
    feature is better for the pitcher. `direction()` is the single place that is converted
    to words.
  - Never negate twice.

LIMITATIONS INHERITED FROM THE HORZBREAK SCRIPT:
  - Part B's RHP-only and LHP-only bootstraps resample disjoint pitcher pools, so
    "RHP - LHP" is an independent Monte Carlo combination of two bootstrap distributions,
    not a same-resample paired bootstrap. Labelled as such where it is printed.
  - This answers whether pooling is confounded, not whether his card's magnitude is
    calibrated -- only its sign.

Data rules: reads cached workdirs only via coach_model_ff_criterion._frame (licensed
Level II TrackMan data). Writes one JSON under SCORE_WORKDIR, no pitcher names, never
committed. Imports coach_model_comparison, coach_model_ff_criterion and fair_criterion
without modifying any of them.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import coach_model_comparison as cm
import coach_model_ff_criterion as ffc
import fair_criterion as fc

FEATURE = "RelSide"
MIRROR = [FEATURE]              # columns flipped for LHP to reach an arm-side frame
MIRROR_ALSO = ["HorzBreak", "horzbreakdiff"]   # NOT applied; see the docstring
INT_COL = "relside_x_lhp"      # RelSide * is_lhp, added for the interaction variant
FLOOR = 100
N_BOOT_COEF = 200
N_BOOT_CORR = 3000
# Identification threshold for the pooled fit. Above this correlation with is_lhp the
# pooled coefficient is a split of shared variance rather than an estimate of anything,
# and part B says so instead of printing a number that invites interpretation.
COLLINEAR_MAX = 0.90
OUT_PATH = os.path.join(ffc.SCORE_WORKDIR, "coach_relside_hand.json")


def direction(coef: float) -> str:
    """Coefficient is on run value (lower = better), so negative => more is better."""
    return "more is better" if coef < 0 else "less is better"


def mirror_frame(ff: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Flip `cols` for LHP so both hands share an arm-side-positive convention."""
    out = ff.copy()
    flip = np.where(out["is_lhp"] == 1, -1.0, 1.0)
    for c in cols:
        out[c] = out[c] * flip
    return out


def add_interaction(ff: pd.DataFrame) -> pd.DataFrame:
    """Add RelSide * is_lhp in place and return the frame.

    Keeping is_lhp as a main effect alongside this product is the whole point: the level
    difference between hands (the value of BEING a left-hander) lives in the is_lhp
    coefficient, and the per-hand RESPONSE to release side lives in the two slopes the
    interaction implies (RHP = RelSide coef; LHP = RelSide coef + interaction coef). That
    is the separation Jack asked for -- release-point value without the handedness premium.

    CAVEAT tied to README open item #2: if is_lhp is ever dropped or neutralised, this
    interaction alone would start absorbing the level gap too, because it would be the only
    term free to differ between hands. The clean split holds only while is_lhp stays in."""
    ff[INT_COL] = ff[FEATURE].values * ff["is_lhp"].astype(float).values
    return ff


# ---------------------------------------------------------------------------
# part A
# ---------------------------------------------------------------------------

def part_a(ff: pd.DataFrame) -> dict:
    print(f"\n=== A. Is {FEATURE} mirrored by handedness, and is it separable from is_lhp? ===")
    print("  (pooled across both role-years' four-seams, D1 level; raw values, no")
    print("   orientation applied -- a units/identification check, not a validity one)")
    desc = {}
    for label, sub in [("RHP", ff[ff["is_lhp"] == 0]), ("LHP", ff[ff["is_lhp"] == 1])]:
        v = sub[FEATURE]
        desc[label] = dict(n=int(len(v)), mean=float(v.mean()), sd=float(v.std()),
                           p10=float(v.quantile(0.10)), p50=float(v.quantile(0.50)),
                           p90=float(v.quantile(0.90)))
        d = desc[label]
        print(f"  {label:<4} n={d['n']:>7}  mean={d['mean']:+7.2f}  sd={d['sd']:6.2f}  "
              f"p10={d['p10']:+7.2f}  p50={d['p50']:+7.2f}  p90={d['p90']:+7.2f}")

    m_r, m_l = desc["RHP"]["mean"], desc["LHP"]["mean"]
    avg_mag = (abs(m_r) + abs(m_l)) / 2
    mirrored = bool(np.sign(m_r) != np.sign(m_l) and avg_mag > 0
                    and abs(m_r + m_l) / avg_mag < 0.3)

    # The RelSide-specific part: how much of the raw spread is just the hand? A feature
    # can be perfectly mirrored (part A's original test) and still be estimable pooled if
    # enough variation survives within a hand -- HorzBreak is that case. This separates the
    # two questions instead of letting "mirrored" stand in for both.
    r_hand = float(fc.R(ff[FEATURE], ff["is_lhp"].astype(float)))
    within_sd = float(np.sqrt(((desc["RHP"]["sd"] ** 2 * desc["RHP"]["n"]
                                + desc["LHP"]["sd"] ** 2 * desc["LHP"]["n"])
                               / (desc["RHP"]["n"] + desc["LHP"]["n"]))))
    gap = abs(m_r - m_l)
    vif = float(1.0 / max(1e-12, 1.0 - r_hand ** 2))
    mir_ff = mirror_frame(ff, MIRROR)
    r_hand_mir = float(fc.R(mir_ff[FEATURE], mir_ff["is_lhp"].astype(float)))

    print(f"\n  between-hand gap        {gap:7.2f}")
    print(f"  within-hand SD (pooled) {within_sd:7.2f}   gap/SD = {gap / within_sd:.2f}")
    print(f"  corr({FEATURE}, is_lhp) {r_hand:+7.3f}   VIF = {vif:.1f}   "
          f"({100 * (1 - r_hand ** 2):.1f}% of its variance survives conditioning on hand)")
    print(f"  after mirroring         {r_hand_mir:+7.3f}   "
          f"(mirroring is what makes the pooled coefficient estimable)")

    identified = bool(abs(r_hand) < COLLINEAR_MAX)
    if not identified:
        print(f"\n  -> NOT IDENTIFIED POOLED: |corr| >= {COLLINEAR_MAX}. The shipping "
              f"pooled {FEATURE}\n     coefficient is a ridge-penalty split of variance "
              f"shared with is_lhp, not an\n     estimate of release mechanics. Read the "
              f"hand-split and mirrored rows in part B;\n     do not interpret the pooled "
              f"row, and do not let the page print a direction for it.")
    verdict = ("MIRRORED: hand means sit on opposite sides of zero and roughly cancel."
               if mirrored else
               "NOT clearly mirrored at the 30% tolerance used here.")
    print(f"  -> {verdict}")
    return dict(by_hand=desc, mirrored=mirrored, verdict=verdict,
                corr_with_is_lhp=round(r_hand, 4),
                corr_with_is_lhp_mirrored=round(r_hand_mir, 4),
                vif_vs_is_lhp=round(vif, 2), between_hand_gap=round(gap, 4),
                within_hand_sd=round(within_sd, 4),
                gap_over_within_sd=round(gap / within_sd, 3),
                pooled_identified=identified, collinear_max=COLLINEAR_MAX)


# ---------------------------------------------------------------------------
# part B
# ---------------------------------------------------------------------------

def fit_coefs(train: pd.DataFrame, feats: list[str]) -> np.ndarray:
    model = make_pipeline(StandardScaler(), Ridge(alpha=fc.RIDGE_ALPHA))
    model.fit(train[feats].values, train["Target"].values)
    return model.named_steps["ridge"].coef_


def cluster_bootstrap_coefs(train: pd.DataFrame, feats: list[str], n_boot: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Resample PITCHERS (whole clusters of rows) with replacement and refit.

    Returns (n_boot, n_feats): the full standardised coefficient vector per draw, not one
    column of it. Same cost as returning one -- the refit is the expensive step -- and it
    is what lets every feature's direction carry an error bar."""
    groups = train.groupby("PitcherId").indices
    ids = np.array(list(groups.keys()))
    X = train[feats].values
    y = train["Target"].values
    out = np.empty((n_boot, len(feats)))
    for b in range(n_boot):
        samp = rng.choice(ids, len(ids), replace=True)
        idx = np.concatenate([groups[i] for i in samp])
        model = make_pipeline(StandardScaler(), Ridge(alpha=fc.RIDGE_ALPHA))
        model.fit(X[idx], y[idx])
        out[b] = model.named_steps["ridge"].coef_
        if (b + 1) % 50 == 0:
            print(f"      {b + 1}/{n_boot}", flush=True)
    return out


def _summ(coef: float, boot: np.ndarray) -> dict:
    p_gt0 = float((boot > 0).mean())
    established = bool(max(p_gt0, 1 - p_gt0) >= 0.95)
    return dict(coef=round(coef, 4), boot_mean=round(float(boot.mean()), 4),
                boot_se=round(float(boot.std()), 4),
                ci95=[round(float(np.percentile(boot, 2.5)), 4),
                      round(float(np.percentile(boot, 97.5)), 4)],
                p_gt0=round(p_gt0, 4), reads_as=direction(coef),
                direction_established=established)


def part_b(ff: pd.DataFrame, a: dict) -> dict:
    print(f"\n=== B. Ridge coefficient on {FEATURE}, four ways (train = 2024 role only) ===")
    train_all = ff[ff["year"] == 2024]
    rng = np.random.default_rng(2024)
    feats_pooled = list(fc.FEATS)
    feats_hand = [f for f in fc.FEATS if f != "is_lhp"]   # constant within one hand
    j_pooled = feats_pooled.index(FEATURE)
    j_hand = feats_hand.index(FEATURE)
    out, boots = {}, {}

    print(f"    pooled ({len(train_all):,} pitches)...", flush=True)
    cf = fit_coefs(train_all, feats_pooled)
    bt = cluster_bootstrap_coefs(train_all, feats_pooled, N_BOOT_COEF, rng)
    out["pooled"] = _summ(float(cf[j_pooled]), bt[:, j_pooled])
    boots["pooled"] = bt[:, j_pooled]

    # Every other coefficient from the same refits, so the page can stop printing
    # directions for terms whose CI straddles zero. Free: these fits already happened.
    out["all_features_pooled"] = {
        f: _summ(float(cf[i]), bt[:, i]) for i, f in enumerate(feats_pooled)}

    for label, mask in [("RHP", train_all["is_lhp"] == 0),
                        ("LHP", train_all["is_lhp"] == 1)]:
        sub = train_all[mask]
        print(f"    {label}-only ({len(sub):,} pitches, "
              f"{sub['PitcherId'].nunique():,} pitchers)...", flush=True)
        cf_h = fit_coefs(sub, feats_hand)
        bt_h = cluster_bootstrap_coefs(sub, feats_hand, N_BOOT_COEF, rng)
        out[label] = _summ(float(cf_h[j_hand]), bt_h[:, j_hand])
        out[label]["n_pitchers"] = int(sub["PitcherId"].nunique())
        boots[label] = bt_h[:, j_hand]

    print(f"    mirrored-pooled...", flush=True)
    mir_train = mirror_frame(ff, MIRROR)
    mir_train = mir_train[mir_train["year"] == 2024]
    cf_m = fit_coefs(mir_train, feats_pooled)
    bt_m = cluster_bootstrap_coefs(mir_train, feats_pooled, N_BOOT_COEF, rng)
    out["mirrored_pooled"] = _summ(float(cf_m[j_pooled]), bt_m[:, j_pooled])
    boots["mirrored_pooled"] = bt_m[:, j_pooled]

    # interaction: one pooled model, is_lhp kept for the level, RelSide allowed a separate
    # slope per hand. The implied per-hand slopes are the interpretable quantities, not the
    # raw split of RelSide vs interaction coefficient (those two are collinear and the
    # ridge penalty splits them arbitrarily; their hand-specific SUMS are stable).
    print(f"    interaction (is_lhp kept, RelSide slope per hand)...", flush=True)
    feats_int = feats_pooled + [INT_COL]
    j_rs, j_int = feats_int.index(FEATURE), feats_int.index(INT_COL)
    int_train = ff[ff["year"] == 2024]
    cf_i = fit_coefs(int_train, feats_int)
    bt_i = cluster_bootstrap_coefs(int_train, feats_int, N_BOOT_COEF, rng)
    rhp_slope = bt_i[:, j_rs]                      # is_lhp = 0
    lhp_slope = bt_i[:, j_rs] + bt_i[:, j_int]     # is_lhp = 1
    idiff = bt_i[:, j_int]                          # LHP slope minus RHP slope
    p_idiff = float((idiff > 0).mean())
    out["interaction"] = dict(
        rhp_slope=_summ(float(cf_i[j_rs]), rhp_slope),
        lhp_slope=_summ(float(cf_i[j_rs] + cf_i[j_int]), lhp_slope),
        slope_diff=dict(mean=round(float(idiff.mean()), 4), se=round(float(idiff.std()), 4),
                        p_gt0=round(p_idiff, 4),
                        differ_established=bool(max(p_idiff, 1 - p_idiff) >= 0.95)),
        is_lhp_level=_summ(float(cf_i[feats_int.index("is_lhp")]),
                           bt_i[:, feats_int.index("is_lhp")]))
    boots["interaction_rhp"], boots["interaction_lhp"] = rhp_slope, lhp_slope

    print(f"\n  {'variant':<17}{'coef':>9}{'boot SE':>10}{'P(coef>0)':>11}"
          f"{'reads as':>18}{'established':>13}")
    for k in ("pooled", "RHP", "LHP", "mirrored_pooled"):
        v = out[k]
        note = ""
        if k == "pooled" and not a["pooled_identified"]:
            note = "   <- NOT IDENTIFIED, do not interpret"
        print(f"  {k:<17}{v['coef']:>+9.4f}{v['boot_se']:>10.4f}{v['p_gt0']:>11.3f}"
              f"{v['reads_as']:>18}{str(v['direction_established']):>13}{note}")
    for k, v in (("interaction:RHP", out["interaction"]["rhp_slope"]),
                 ("interaction:LHP", out["interaction"]["lhp_slope"])):
        print(f"  {k:<17}{v['coef']:>+9.4f}{v['boot_se']:>10.4f}{v['p_gt0']:>11.3f}"
              f"{v['reads_as']:>18}{str(v['direction_established']):>13}")
    sd = out["interaction"]["slope_diff"]
    print(f"  interaction gives the two per-hand slopes IN ONE POOLED MODEL, with is_lhp "
          f"holding the\n  level. Slope difference LHP-RHP = {sd['mean']:+.4f} +/- "
          f"{sd['se']:.4f} (differ established={sd['differ_established']}). These implied "
          f"slopes\n  should track the disjoint RHP-only/LHP-only fits above; if they do, "
          f"the per-hand\n  reading is robust to whether the other features are pooled.")

    print("\n  DECISIVE COMPARISON, RHP-only vs LHP-only (disjoint pitcher pools, so this")
    print("  is an independent Monte Carlo combination of two bootstraps, not a paired")
    print("  same-resample bootstrap). This is the quantity the verdict turns on:")
    diff = boots["RHP"] - boots["LHP"]
    fc.boot_report("RHP coef - LHP coef", diff)
    p_gt0 = float((diff > 0).mean())
    out["rhp_minus_lhp"] = dict(
        mean=round(float(diff.mean()), 4), se=round(float(diff.std()), 4),
        p_gt0=round(p_gt0, 4),
        differ_established=bool(max(p_gt0, 1 - p_gt0) >= 0.95))

    print(f"\n  every pooled coefficient, ranked by |coef| -- the page prints a direction")
    print(f"  for all of these and an error bar for none of them:")
    print(f"    {'feature':<24}{'coef':>10}{'boot SE':>10}{'reads as':>18}{'established':>13}")
    for f in sorted(feats_pooled, key=lambda f: -abs(out["all_features_pooled"][f]["coef"])):
        v = out["all_features_pooled"][f]
        print(f"    {f:<24}{v['coef']:>+10.4f}{v['boot_se']:>10.4f}"
              f"{v['reads_as']:>18}{str(v['direction_established']):>13}")
    unest = [f for f in feats_pooled
             if not out["all_features_pooled"][f]["direction_established"]]
    print(f"    -> {len(unest)} of {len(feats_pooled)} feature directions NOT established: "
          f"{', '.join(unest) if unest else 'none'}")
    return out


# ---------------------------------------------------------------------------
# part C
# ---------------------------------------------------------------------------

def part_c(score_ff: pd.DataFrame, crit_ff: pd.DataFrame) -> dict:
    print(f"\n=== C. Coach's isolated {FEATURE} term, re-evaluated per hand ===")
    print("  (raw-trait convention: his term stays in his native higher-is-better units,")
    print("   so a well-signed term correlates NEGATIVELY with the criterion)")
    print("  UNDERPOWERED BY CONSTRUCTION -- see POWER in the docstring. At LHP n~219 the")
    print("  correlation SE is ~0.076, so nothing under |r|~0.125 can clear P>=0.95, and")
    print("  the whole twelve-feature ridge only reaches 0.112. Expect 'not established'.")
    terms = cm.load_coach_terms("FourSeamFastBall")
    rs_terms = [t for t in terms if t["col"] == FEATURE]
    if not rs_terms:
        print(f"  his card has no {FEATURE} term; part C does not apply")
        return {"applicable": False}
    print(f"\n  his {FEATURE} rows in the workbook:")
    for t in rs_terms:
        print(f"    hand={t['hand']:<6} coef={t['coef']:+.3f} avg={t['avg']:+.3f} "
              f"relative={t['relative']} direction={t['direction']}")
    hands_differ = len({(t["coef"], t["avg"], t["direction"]) for t in rs_terms}) > 1
    print(f"  -> per-hand {FEATURE} weights differ (avg and/or direction): {hands_differ}")

    graded = score_ff[score_ff["year"] == 2025].copy()
    graded["iso_raw"] = cm.coach_score(graded, rs_terms, 1.0)
    pit = graded.groupby("PitcherId").agg(n=("iso_raw", "size"), iso=("iso_raw", "mean"),
                                          hand=("PitcherThrows", "first"))
    pit = pit[pit["n"] >= FLOOR]

    nxt = crit_ff[crit_ff["year"] == 2025]
    k = nxt.groupby("PitcherId").agg(n26=("Target", "size"), crit=("Target", "mean"))
    k = k[k["n26"] >= FLOOR]
    k["crit100"] = k["crit"] * 100          # run value per 100 FF, lower = better

    j = pit.join(k, how="inner")
    print(f"\n  joined panel: {len(j)} pitchers "
          f"({int((j['hand'] == 'Right').sum())} RHP, {int((j['hand'] == 'Left').sum())} LHP)")

    out = {"applicable": True, "terms": [dict(t) for t in rs_terms],
           "hands_differ": bool(hands_differ), "by_hand": {}}
    rng = np.random.default_rng(56)
    print(f"\n  {'group':<8}{'n':>5}{'r':>9}{'SE':>8}   min |r| detectable at P>=0.95")
    for label, sub in [("pooled", j), ("RHP", j[j["hand"] == "Right"]),
                       ("LHP", j[j["hand"] == "Left"])]:
        n = len(sub)
        if n < 8:
            print(f"  {label:<8}{n:>5}   too small to bootstrap")
            out["by_hand"][label] = dict(n=n)
            continue
        r = float(fc.R(sub["iso"].values, sub["crit100"].values))
        idx = sub.index.values
        boots = np.empty(N_BOOT_CORR)
        for b in range(N_BOOT_CORR):
            draw = sub.loc[rng.choice(idx, n, replace=True)]
            boots[b] = fc.R(draw["iso"].values, draw["crit100"].values)
        se = float(boots.std())
        p_gt0 = float((boots > 0).mean())
        mde = 1.645 * se
        print(f"  {label:<8}{n:>5}{r:>+9.3f}{se:>8.3f}   |r| >= {mde:.3f}")
        out["by_hand"][label] = dict(
            n=n, r=round(r, 4), boot_mean=round(float(boots.mean()), 4),
            boot_se=round(se, 4), p_gt0=round(p_gt0, 4), p_lt0=round(1 - p_gt0, 4),
            established=bool(max(p_gt0, 1 - p_gt0) >= 0.95),
            mde_95=round(mde, 4))
    return out


# ---------------------------------------------------------------------------
# part D
# ---------------------------------------------------------------------------

def _predict_variant(score_ff: pd.DataFrame, feats: list[str]) -> pd.Series:
    """Refit on the 2024 role year over `feats`, predict the 2025 role year."""
    train = score_ff[(score_ff["year"] == 2024) & score_ff["Target"].notna()]
    graded = score_ff[score_ff["year"] == 2025]
    model = make_pipeline(StandardScaler(), Ridge(alpha=fc.RIDGE_ALPHA))
    model.fit(train[feats].values, train["Target"].values)
    return pd.Series(model.predict(graded[feats].values), index=graded.index)


def part_d(score_ff: pd.DataFrame, crit_ff: pd.DataFrame) -> dict:
    print("\n=== D. Does re-parameterising RelSide help predictive validity? ===")
    print("  (predictions and criterion are both lower=better run value, so a correctly")
    print("   oriented prediction correlates POSITIVELY -- fitted-prediction rule)")
    print("  Two candidates against pooled: MIRRORED (one arm-side slope, shared across")
    print("  hands) and INTERACTION (per-hand slopes, is_lhp kept for the level). The")
    print("  interaction nests the mirror, so it can only match or beat it in-sample; the")
    print("  question is whether either buys OUT-of-sample validity.")
    print("  UNDERPOWERED -- HorzBreak's mirror counterpart was +0.0037 +/- 0.0075, needing")
    print("  ~+0.0123 to clear P>=0.95. Expect ties; a tie is not endorsement.")
    graded = score_ff[score_ff["year"] == 2025].copy()

    mir_pred = _predict_variant(mirror_frame(score_ff, MIRROR), list(fc.FEATS))
    int_pred = _predict_variant(add_interaction(score_ff.copy()), list(fc.FEATS) + [INT_COL])

    g = pd.DataFrame({"PitcherId": graded["PitcherId"],
                      "pooled": graded["ridge_pred"].values,
                      "mirrored": mir_pred.reindex(graded.index).values,
                      "interaction": int_pred.reindex(graded.index).values},
                     index=graded.index)
    p = g.groupby("PitcherId").agg(n=("pooled", "size"), pooled=("pooled", "mean"),
                                   mirrored=("mirrored", "mean"),
                                   interaction=("interaction", "mean"))
    p = p[p["n"] >= FLOOR]

    nxt = crit_ff[crit_ff["year"] == 2025]
    k = nxt.groupby("PitcherId").agg(n26=("Target", "size"), crit=("Target", "mean"))
    k = k[k["n26"] >= FLOOR]
    k["crit100"] = k["crit"] * 100

    j = p.join(k, how="inner")
    print(f"  panel: {len(j)} pitchers")
    r = {v: float(fc.R(j[v].values, j["crit100"].values))
         for v in ("pooled", "mirrored", "interaction")}
    for v in ("pooled", "mirrored", "interaction"):
        print(f"  {v:<12} predictive validity: r={r[v]:+.4f}")
    if abs(r["pooled"] - 0.1121) > 0.005:
        print(f"  WARNING: pooled validity {r['pooled']:.4f} does not reproduce the 0.1121 "
              f"on record\n           (coach_handedness.json, variant 'full', n=825). "
              f"The pool has moved; stop\n           and reconcile before reading any "
              f"comparison below.")

    rng = np.random.default_rng(78)
    idx = j.index.values
    out = dict(n=int(len(j)), r_pooled=round(r["pooled"], 4),
               r_mirrored=round(r["mirrored"], 4),
               r_interaction=round(r["interaction"], 4), vs_pooled={})
    for v in ("mirrored", "interaction"):
        diffs = np.empty(N_BOOT_CORR)
        for b in range(N_BOOT_CORR):
            s = j.loc[rng.choice(idx, len(idx), replace=True)]
            diffs[b] = (fc.R(s[v].values, s["crit100"].values)
                        - fc.R(s["pooled"].values, s["crit100"].values))
        fc.boot_report(f"{v} - pooled predictive validity", diffs)
        p_gt0 = float((diffs > 0).mean())
        out["vs_pooled"][v] = dict(
            diff_boot_mean=round(float(diffs.mean()), 4),
            diff_boot_se=round(float(diffs.std()), 4),
            p_gt_pooled=round(p_gt0, 4), established=bool(p_gt0 >= 0.95),
            mde_95=round(1.645 * float(diffs.std()), 4))
    # back-compat keys so a reader of the old shape still finds the mirror result
    out["diff_boot_mean"] = out["vs_pooled"]["mirrored"]["diff_boot_mean"]
    out["diff_boot_se"] = out["vs_pooled"]["mirrored"]["diff_boot_se"]
    out["p_mirrored_gt_pooled"] = out["vs_pooled"]["mirrored"]["p_gt_pooled"]
    out["established"] = out["vs_pooled"]["mirrored"]["established"]
    out["mde_95"] = out["vs_pooled"]["mirrored"]["mde_95"]
    return out


# ---------------------------------------------------------------------------
# verdicts -- one per part, because they are not equally powered
# ---------------------------------------------------------------------------

def verdict_from(out: dict) -> tuple[str, str]:
    """Same three labels as the HorzBreak script, decided on the hand DIFFERENCE.

    The old gate (both hands individually distinguishable from zero) answers "does this
    feature matter within each hand", which is not the question. "Is pooling averaging two
    different relationships" is a question about the difference, and two coefficients can
    both straddle zero while differing confidently from each other."""
    b, a = out["B"], out["A"]
    d = b["rhp_minus_lhp"]
    rhp, lhp = b["RHP"]["coef"], b["LHP"]["coef"]
    opposite = bool(np.sign(rhp) != np.sign(lhp))
    mir = b["mirrored_pooled"]

    if d["differ_established"]:
        # built outside the f-string: a multi-line expression inside one needs PEP 701
        # (Python 3.12+), and the suite has to run on 3.11.
        how = ("with opposite signs" if opposite else
               "in the same direction but by a confidently non-zero amount")
        p_differ = max(d["p_gt0"], 1 - d["p_gt0"])
        label = "pooling is the artifact -- our pooled coefficient is not a finding"
        reasoning = (
            f"RHP-only ({rhp:+.4f}) and LHP-only ({lhp:+.4f}) {FEATURE} coefficients differ "
            f"by {d['mean']:+.4f} +/- {d['se']:.4f} (P={p_differ:.3f}), {how}. "
            f"The pooled coefficient ({b['pooled']['coef']:+.4f}) is therefore an average "
            f"over two different relationships"
            + (f", and with corr({FEATURE}, is_lhp) = {a['corr_with_is_lhp']:+.3f} it is "
               f"not separately identified from the handedness flag in any case"
               if not a["pooled_identified"] else "")
            + f". Mirrored-pooled reads {mir['coef']:+.4f} ({mir['reads_as']}, "
              f"established={mir['direction_established']}), which is the estimable version. "
            f"The coach's card carries per-hand weights and does not have this problem. "
            f"The page should not print a pooled direction for {FEATURE}.")
    elif not d["differ_established"] and b["RHP"]["direction_established"] \
            and b["LHP"]["direction_established"]:
        label = "pooling is fine -- the hands agree"
        reasoning = (
            f"RHP-only ({rhp:+.4f}) and LHP-only ({lhp:+.4f}) coefficients are each "
            f"distinguishable from zero and are NOT confidently different from each other "
            f"({d['mean']:+.4f} +/- {d['se']:.4f}). Pooling is not averaging two different "
            f"relationships, so the pooled coefficient stands on its own terms and part C "
            f"can be read at face value.")
    else:
        label = "not established"
        reasoning = (
            f"The hand difference is {d['mean']:+.4f} +/- {d['se']:.4f} "
            f"(P={max(d['p_gt0'], 1 - d['p_gt0']):.3f}), short of the P>=0.95 bar, and the "
            f"individual coefficients (RHP {rhp:+.4f}, established="
            f"{b['RHP']['direction_established']}; LHP {lhp:+.4f}, established="
            f"{b['LHP']['direction_established']}) do not settle it either. Note this is a "
            f"statement about part B, which is the WELL-POWERED part; parts C and D are "
            f"underpowered by construction and cannot rescue it.")
    return label, reasoning


def main() -> int:
    args = fc.paths()   # CLI parity with the rest of the suite; the real sources are the
    del args            # fixed SCORE_WORKDIR/CRIT_WORKDIR pair, as in the HorzBreak script

    print(f"loading graded-season build ({ffc.SCORE_WORKDIR}, 2024/2025 role years)...")
    score_ff = ffc._frame(ffc.SCORE_WORKDIR, "2024,2025")
    print(f"loading criterion build ({ffc.CRIT_WORKDIR}, 2025/2026 role years)...")
    crit_ff = ffc._frame(ffc.CRIT_WORKDIR, "2025,2026")

    for col in MIRROR + [FEATURE, "is_lhp"]:
        if col not in score_ff.columns:
            raise SystemExit(f"{col} missing from the loaded frame; cannot run")
    if FEATURE not in fc.FEATS:
        raise SystemExit(f"{FEATURE} is not in fc.FEATS; this script's premise is gone")
    add_interaction(score_ff)   # part B reads INT_COL off this frame; part D rebuilds its own

    out = {"feature": FEATURE, "mirrored_columns": MIRROR, "floor": FLOOR,
           "n_boot_coef": N_BOOT_COEF, "n_boot_corr": N_BOOT_CORR}
    out["A"] = part_a(score_ff)
    out["B"] = part_b(score_ff, out["A"])
    out["C"] = part_c(score_ff, crit_ff)
    out["D"] = part_d(score_ff, crit_ff)

    label, reasoning = verdict_from(out)
    print(f"\n=== VERDICT (part B, the powered part): {label} ===")
    print(reasoning)
    out["verdict"] = label
    out["reasoning"] = reasoning
    out["verdict_scope"] = ("Part B only. Part C (his term) and part D (validity) are "
                            "underpowered by construction at these pool sizes and are "
                            "reported as unresolved, not as ties.")
    c_est = [k for k, v in out["C"].get("by_hand", {}).items() if v.get("established")]
    out["C_verdict"] = ("not established -- underpowered" if not c_est
                        else f"established for: {', '.join(c_est)}")
    d_win = [v for v in ("mirrored", "interaction")
             if out["D"]["vs_pooled"][v]["established"]]
    out["D_verdict"] = (f"improves validity: {', '.join(d_win)}" if d_win
                        else "not established -- underpowered (neither mirror nor "
                             "interaction beats pooled at P>=0.95)")
    print(f"  part C: {out['C_verdict']}")
    print(f"  part D: {out['D_verdict']}")

    os.makedirs(ffc.SCORE_WORKDIR, exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
