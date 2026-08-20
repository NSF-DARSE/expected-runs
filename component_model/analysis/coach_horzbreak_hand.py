"""Is the pooled ridge's HorzBreak sign an artifact of mixing two handednesses?

BACKGROUND: a feature-importance run found the coach's hand-weighted scorecard and our
pooled ridge assign OPPOSITE signs to HorzBreak (his card: more break is better; our
ridge: less break is better, |r| 0.274, our 4th-strongest feature). A follow-up
concluded his term was "signed backwards" because his isolated HorzBreak contribution
correlated +0.056 with next season's run value -- the wrong direction for a raw trait
that predicts better outcomes (see SIGN CONVENTION below). That finding was computed
POOLED across both handednesses.

THE WORRY THIS SCRIPT TESTS: a left-hander's horizontal break points the opposite way
from a right-hander's for physically equivalent stuff. If the feed stores raw
(unmirrored) HorzBreak, a single pooled linear coefficient averages two opposite-signed
relationships and both the pooled ridge coefficient AND the pooled "+0.056, backwards"
finding are confounded. His card already carries separate D1 averages and a separate
+1/-1 direction per hand for HorzBreak (verified against the workbook below) -- if the
feed is unmirrored, his per-hand design is simply the right shape for this feature and
the "backwards" claim needs to be retracted before a coaches' meeting repeats it.

FOUR PARTS:
  A. Descriptive: is HorzBreak's distribution mirrored across RHP/LHP in the feed?
  B. Refit the reference ridge (fair_criterion.stuff_ridge, alpha=10, StandardScaler,
     trained on the 2024-role year, FEATS) four ways: pooled (shipping model), RHP-only,
     LHP-only, and mirrored-pooled (HorzBreak/horzbreakdiff x -1 for LHP so both hands
     share an arm-side-positive convention). RHP-only vs LHP-only is decisive: opposite
     signs means pooling is averaging mirrored relationships.
  C. Redo the coach-term correlation from part A of the earlier finding, but split by
     hand instead of pooled, using his own per-hand HorzBreak weights.
  D. Predictive-validity check: does the mirrored-pooled ridge beat the pooled ridge at
     predicting next season's four-seam run value?

SIGN CONVENTIONS (fair_criterion.py; do not guess, an inverted interpretation shipped
once already):
  - Target, xT, adjT and ridge_pred are expected run value from the PITCHER's
    perspective, LOWER = BETTER.
  - RAW TRAITS (a physical measurement, or an isolated linear-scorecard term expressed
    in the scorecard's own native higher-is-better convention, as used in part C here)
    predict better outcomes when they correlate NEGATIVELY with the run-value criterion.
    This is the convention the original "+0.056, backwards" finding used, and part C
    reproduces it exactly, split by hand.
  - FITTED PREDICTIONS of run value (ridge_pred and its mirrored counterpart, part D)
    are already in the same lower-is-better frame as the criterion. A correctly
    oriented prediction correlates POSITIVELY with the criterion; do not apply the
    raw-trait rule to these.
  - Never negate a quantity twice. Each reported number states its own orientation at
    the point it is computed.

LIMITATIONS:
  - Part B's RHP-only and LHP-only bootstraps resample disjoint pitcher pools (no
    pitcher is both), so the "RHP - LHP" comparison is an independent Monte Carlo
    combination of the two bootstrap distributions, not a same-resample paired
    bootstrap. Framed and labelled as such below.
  - Part C's floor (100+ FF in the graded season, 100+ in the next season) shrinks
    further once split by hand; LHP samples are the D1 minority and any LHP-only
    result here is intentionally reported with wider uncertainty rather than a false
    tie broken by a handful of pitchers.
  - This script answers whether pooling is confounded, not whether his card's
    magnitude (coef 3.0 either hand) is well-calibrated -- only its sign.

Data rules: reads cached workdirs only via coach_model_ff_criterion._frame (licensed
Level II TrackMan data). Writes one JSON summary under SCORE_WORKDIR, no pitcher names,
never committed. Imports coach_model_comparison, coach_model_ff_criterion, and
fair_criterion without modifying any of them.
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

FLOOR = 100                 # four-seams required per pitcher per season, matches the suite
N_BOOT_COEF = 200            # ridge refits per bootstrap draw (part B) -- refitting is the
                             # expensive step, kept modest across 4 variants x 200 fits
N_BOOT_CORR = 3000           # correlation-only bootstraps (parts C, D) are cheap
OUT_PATH = os.path.join(ffc.SCORE_WORKDIR, "coach_horzbreak_hand.json")


# ---------------------------------------------------------------------------
# part A
# ---------------------------------------------------------------------------

def describe_by_hand(ff: pd.DataFrame) -> dict:
    out = {}
    for label, sub in [("RHP", ff[ff["is_lhp"] == 0]), ("LHP", ff[ff["is_lhp"] == 1])]:
        v = sub["HorzBreak"]
        out[label] = dict(n=int(len(v)), mean=float(v.mean()), sd=float(v.std()),
                           p10=float(v.quantile(0.10)), p50=float(v.quantile(0.50)),
                           p90=float(v.quantile(0.90)))
    return out


def part_a(ff: pd.DataFrame) -> dict:
    print("\n=== A. Is HorzBreak mirrored by handedness in the feed? ===")
    print("  (pooled across both role-years' four-seams, D1 level; raw HorzBreak, no "
          "orientation applied -- this is a units/convention check, not a validity one)")
    desc = describe_by_hand(ff)
    for hand in ("RHP", "LHP"):
        d = desc[hand]
        print(f"  {hand:<4} n={d['n']:>7}  mean={d['mean']:+7.2f}  sd={d['sd']:6.2f}  "
              f"p10={d['p10']:+7.2f}  p50={d['p50']:+7.2f}  p90={d['p90']:+7.2f}")
    m_r, m_l = desc["RHP"]["mean"], desc["LHP"]["mean"]
    avg_mag = (abs(m_r) + abs(m_l)) / 2
    mirrored = bool(np.sign(m_r) != np.sign(m_l) and avg_mag > 0
                     and abs(m_r + m_l) / avg_mag < 0.3)
    verdict = ("MIRRORED: RHP and LHP means sit on opposite sides of zero and roughly "
               "cancel (raw arm-side/glove-side convention, not pre-flipped to a common "
               "frame). Pooling HorzBreak without a per-hand sign flip averages two "
               "opposite-signed physical relationships."
               if mirrored else
               "NOT clearly mirrored: RHP and LHP means do not sit as mirror images "
               "around zero at the 30% tolerance used here.")
    print(f"  -> {verdict}")
    return dict(by_hand=desc, mirrored=mirrored, verdict=verdict)


# ---------------------------------------------------------------------------
# part B
# ---------------------------------------------------------------------------

def fit_coef(train: pd.DataFrame, feats: list[str], feat_name: str) -> float:
    model = make_pipeline(StandardScaler(), Ridge(alpha=fc.RIDGE_ALPHA))
    model.fit(train[feats].values, train["Target"].values)
    return float(model.named_steps["ridge"].coef_[feats.index(feat_name)])


def cluster_bootstrap_coef(train: pd.DataFrame, feats: list[str], feat_name: str,
                            n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Resample pitchers (whole clusters of rows) with replacement, refit, return the
    standardised coefficient on feat_name each draw."""
    groups = train.groupby("PitcherId").indices
    ids = np.array(list(groups.keys()))
    X = train[feats].values
    y = train["Target"].values
    j = feats.index(feat_name)
    out = np.empty(n_boot)
    for b in range(n_boot):
        samp = rng.choice(ids, len(ids), replace=True)
        idx = np.concatenate([groups[i] for i in samp])
        model = make_pipeline(StandardScaler(), Ridge(alpha=fc.RIDGE_ALPHA))
        model.fit(X[idx], y[idx])
        out[b] = model.named_steps["ridge"].coef_[j]
    return out


def _summ(coef: float, boot: np.ndarray) -> dict:
    p_gt0 = float((boot > 0).mean())
    return dict(coef=round(coef, 4), boot_mean=round(float(boot.mean()), 4),
                boot_se=round(float(boot.std()), 4),
                ci95=[round(float(np.percentile(boot, 2.5)), 4),
                      round(float(np.percentile(boot, 97.5)), 4)],
                p_gt0=round(p_gt0, 4),
                # direction distinguishable from zero at the project's P>=0.95 bar
                direction_established=bool(max(p_gt0, 1 - p_gt0) >= 0.95))


def part_b(ff: pd.DataFrame) -> dict:
    print("\n=== B. Ridge coefficient on HorzBreak, four ways (train = 2024 role only) ===")
    train_all = ff[ff["year"] == 2024]
    rng = np.random.default_rng(2024)
    feats_pooled = list(fc.FEATS)
    feats_hand = [f for f in fc.FEATS if f != "is_lhp"]  # constant within one hand
    out = {}

    coef = fit_coef(train_all, feats_pooled, "HorzBreak")
    boot = cluster_bootstrap_coef(train_all, feats_pooled, "HorzBreak", N_BOOT_COEF, rng)
    out["pooled"] = _summ(coef, boot)
    boots = {"pooled": boot}

    for label, mask in [("RHP", train_all["is_lhp"] == 0), ("LHP", train_all["is_lhp"] == 1)]:
        sub = train_all[mask]
        coef = fit_coef(sub, feats_hand, "HorzBreak")
        boot = cluster_bootstrap_coef(sub, feats_hand, "HorzBreak", N_BOOT_COEF, rng)
        out[label] = _summ(coef, boot)
        boots[label] = boot

    mir = ff.copy()
    flip = np.where(mir["is_lhp"] == 1, -1.0, 1.0)
    mir["HorzBreak"] = mir["HorzBreak"] * flip
    mir["horzbreakdiff"] = mir["horzbreakdiff"] * flip
    mir_train = mir[mir["year"] == 2024]
    coef = fit_coef(mir_train, feats_pooled, "HorzBreak")
    boot = cluster_bootstrap_coef(mir_train, feats_pooled, "HorzBreak", N_BOOT_COEF, rng)
    out["mirrored_pooled"] = _summ(coef, boot)
    boots["mirrored_pooled"] = boot

    print(f"  {'variant':<17}{'coef':>9}{'boot mean':>12}{'boot SE':>10}{'P(coef>0)':>11}")
    for k in ("pooled", "RHP", "LHP", "mirrored_pooled"):
        v = out[k]
        print(f"  {k:<17}{v['coef']:>+9.3f}{v['boot_mean']:>+12.3f}{v['boot_se']:>10.3f}"
              f"{v['p_gt0']:>11.3f}")

    print("\n  decisive comparison, RHP-only vs LHP-only coefficient (disjoint pitcher")
    print("  pools -- independent Monte Carlo combination of the two bootstraps, not a")
    print("  same-resample paired bootstrap):")
    diff = boots["RHP"] - boots["LHP"]
    fc.boot_report("RHP coef - LHP coef", diff)
    p_gt0 = float((diff > 0).mean())
    out["rhp_minus_lhp"] = dict(mean=round(float(diff.mean()), 4),
                                 se=round(float(diff.std()), 4),
                                 p_gt0=round(p_gt0, 4),
                                 # the hands differ CONFIDENTLY, which is what "pooling
                                 # averages two relationships" actually requires -- not that
                                 # each hand separately clears zero
                                 differ_established=bool(max(p_gt0, 1 - p_gt0) >= 0.95))
    return out


# ---------------------------------------------------------------------------
# part C
# ---------------------------------------------------------------------------

def part_c(score_ff: pd.DataFrame, crit_ff: pd.DataFrame) -> dict:
    print("\n=== C. Coach's isolated HorzBreak term, re-evaluated per hand ===")
    print("  (raw-trait convention: his term stays in his native higher-is-better")
    print("  units, not negated -- the earlier +0.056 finding used this same frame,")
    print("  so a well-signed term should correlate NEGATIVELY with the criterion)")
    terms = cm.load_coach_terms("FourSeamFastBall")
    hb_terms = [t for t in terms if t["col"] == "HorzBreak"]
    print("\n  his HorzBreak rows in the workbook:")
    for t in hb_terms:
        print(f"    hand={t['hand']:<6} coef={t['coef']:+.3f} avg={t['avg']:+.3f} "
              f"relative={t['relative']} direction={t['direction']}")
    hands_differ = len({(t["coef"], t["avg"], t["direction"]) for t in hb_terms}) > 1
    print(f"  -> per-hand HorzBreak weights differ (avg and/or direction): {hands_differ}")

    graded = score_ff[score_ff["year"] == 2025].copy()
    graded["iso_raw"] = cm.coach_score(graded, hb_terms, 1.0)  # his native frame, higher=better
    pit = graded.groupby("PitcherId").agg(n=("iso_raw", "size"), iso=("iso_raw", "mean"),
                                           hand=("PitcherThrows", "first"))
    pit = pit[pit["n"] >= FLOOR]

    nxt = crit_ff[crit_ff["year"] == 2025]
    k = nxt.groupby("PitcherId").agg(n26=("Target", "size"), crit=("Target", "mean"))
    k = k[k["n26"] >= FLOOR]
    k["crit100"] = k["crit"] * 100  # runs/100 FF, pitcher's perspective, lower = better

    j = pit.join(k, how="inner")
    n_rhp = int((j["hand"] == "Right").sum())
    n_lhp = int((j["hand"] == "Left").sum())
    print(f"\n  joined panel: {len(j)} pitchers ({n_rhp} RHP, {n_lhp} LHP)")

    out = {"hb_terms": [dict(t) for t in hb_terms], "hands_differ": bool(hands_differ),
           "by_hand": {}}
    rng = np.random.default_rng(56)
    print(f"\n  {'group':<8}{'n':>5}   r (raw trait: expect r<0 if well-signed)")
    for label, sub in [("pooled", j), ("RHP", j[j["hand"] == "Right"]),
                       ("LHP", j[j["hand"] == "Left"])]:
        n = len(sub)
        if n < 8:
            print(f"  {label:<8}{n:>5}   too small to bootstrap")
            out["by_hand"][label] = dict(n=n)
            continue
        r = fc.R(sub["iso"].values, sub["crit100"].values)
        idx = sub.index.values
        boots = np.empty(N_BOOT_CORR)
        for b in range(N_BOOT_CORR):
            draw = sub.loc[rng.choice(idx, n, replace=True)]
            boots[b] = fc.R(draw["iso"].values, draw["crit100"].values)
        print(f"  {label:<8}{n:>5}   r={r:+.3f}")
        fc.boot_report(f"    {label} iso-HorzBreak vs next-season criterion", boots)
        out["by_hand"][label] = dict(
            n=n, r=round(float(r), 4), boot_mean=round(float(boots.mean()), 4),
            boot_se=round(float(boots.std()), 4),
            p_gt0=round(float((boots > 0).mean()), 4),
            p_lt0=round(float((boots < 0).mean()), 4))
    return out


# ---------------------------------------------------------------------------
# part D
# ---------------------------------------------------------------------------

def part_d(score_ff: pd.DataFrame, crit_ff: pd.DataFrame) -> dict:
    print("\n=== D. Does mirroring help predictive validity? ===")
    print("  (both predictions and the criterion are lower=better run value; a")
    print("  correctly oriented prediction correlates POSITIVELY -- fitted-prediction")
    print("  rule, not the raw-trait rule)")
    graded = score_ff[score_ff["year"] == 2025].copy()

    mir_all = score_ff.copy()
    flip = np.where(mir_all["is_lhp"] == 1, -1.0, 1.0)
    mir_all["HorzBreak"] = mir_all["HorzBreak"] * flip
    mir_all["horzbreakdiff"] = mir_all["horzbreakdiff"] * flip
    mir_train = mir_all[mir_all["year"] == 2024]
    mir_graded = mir_all[mir_all["year"] == 2025]

    model = make_pipeline(StandardScaler(), Ridge(alpha=fc.RIDGE_ALPHA))
    model.fit(mir_train[fc.FEATS].values, mir_train["Target"].values)
    mir_pred = pd.Series(model.predict(mir_graded[fc.FEATS].values), index=mir_graded.index)

    g = pd.DataFrame({"PitcherId": graded["PitcherId"], "pooled": graded["ridge_pred"].values,
                       "mirrored": mir_pred.reindex(graded.index).values},
                      index=graded.index)
    p = g.groupby("PitcherId").agg(n=("pooled", "size"), pooled=("pooled", "mean"),
                                    mirrored=("mirrored", "mean"))
    p = p[p["n"] >= FLOOR]

    nxt = crit_ff[crit_ff["year"] == 2025]
    k = nxt.groupby("PitcherId").agg(n26=("Target", "size"), crit=("Target", "mean"))
    k = k[k["n26"] >= FLOOR]
    k["crit100"] = k["crit"] * 100

    j = p.join(k, how="inner")
    print(f"  panel: {len(j)} pitchers")

    r_pooled = fc.R(j["pooled"].values, j["crit100"].values)
    r_mirrored = fc.R(j["mirrored"].values, j["crit100"].values)
    print(f"  pooled predictive validity:   r={r_pooled:+.3f}")
    print(f"  mirrored predictive validity: r={r_mirrored:+.3f}")

    rng = np.random.default_rng(78)
    idx = j.index.values
    diffs = np.empty(N_BOOT_CORR)
    for b in range(N_BOOT_CORR):
        s = j.loc[rng.choice(idx, len(idx), replace=True)]
        diffs[b] = (fc.R(s["mirrored"].values, s["crit100"].values)
                    - fc.R(s["pooled"].values, s["crit100"].values))
    fc.boot_report("mirrored - pooled predictive validity", diffs)
    return dict(n=len(j), r_pooled=round(float(r_pooled), 4),
                r_mirrored=round(float(r_mirrored), 4),
                diff_boot_mean=round(float(diffs.mean()), 4),
                diff_boot_se=round(float(diffs.std()), 4),
                p_mirrored_gt_pooled=round(float((diffs > 0).mean()), 4))


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

def verdict_from(out: dict) -> tuple[str, str]:
    """Gate on whether the HANDS DIFFER, not on each hand clearing zero separately.

    BACKPORTED FIX (2026-08-16): the first version of this function computed
    `rhp_minus_lhp`, printed it as "the decisive comparison", and then never read it --
    gating instead on rhp_conf AND lhp_conf, i.e. each hand's coefficient being individually
    distinguishable from zero. That is a different question. "Pooling averages two different
    relationships" is a claim about the DIFFERENCE, and two coefficients can both sit within
    a SE of zero while being confidently different from each other. On the recorded run it
    mattered: RHP -0.0006 (P(>0)=0.25, ambiguous) and LHP +0.0036 (P(>0)=1.000) returned
    "not established", while the difference the old gate ignored was -0.0042 +/- 0.0015,
    P~1.000 -- the confounding WAS established. Same fix now lives in coach_relside_hand.py;
    the two verdicts are deliberately parallel."""
    b, a = out["B"], out["A"]
    d = b["rhp_minus_lhp"]
    rhp, lhp = b["RHP"]["coef"], b["LHP"]["coef"]
    opposite = bool(np.sign(rhp) != np.sign(lhp))
    mir = b["mirrored_pooled"]
    p_differ = max(d["p_gt0"], 1 - d["p_gt0"])
    how = ("with opposite signs" if opposite else
           "in the same direction but by a confidently non-zero amount")

    if d["differ_established"]:
        label = "pooling is the artifact -- retract the claim about his card"
        reasoning = (
            f"HorzBreak is {'' if a['mirrored'] else 'not clearly '}mirrored in the raw feed "
            f"(part A), and RHP-only ({rhp:+.4f}) and LHP-only ({lhp:+.4f}) ridge "
            f"coefficients differ by {d['mean']:+.4f} +/- {d['se']:.4f} (P={p_differ:.3f}), "
            f"{how}. The pooled coefficient ({b['pooled']['coef']:+.4f}, "
            f"P(coef>0)={b['pooled']['p_gt0']:.3f}) is therefore an average of two different "
            f"physical relationships, not a real pooled effect. Mirrored-pooled reads "
            f"{mir['coef']:+.4f} (established={mir['direction_established']}), the estimable "
            f"version. The coach's card already carries separate per-hand averages and a "
            f"+1/-1 direction for HorzBreak (part C), which is the correct shape. The earlier "
            f"'+0.056, backwards' pooled finding should be retracted or restated per-hand -- "
            f"note that is a statement about OUR pooling, not proof his term is well-signed; "
            f"part C is underpowered to settle his term (LHP n~219)."
        )
    elif (not d["differ_established"] and b["RHP"]["direction_established"]
          and b["LHP"]["direction_established"]):
        label = "pooling is fine -- his term is backwards as previously stated"
        reasoning = (
            f"RHP-only ({rhp:+.4f}) and LHP-only ({lhp:+.4f}) coefficients are each "
            f"distinguishable from zero and NOT confidently different from each other "
            f"({d['mean']:+.4f} +/- {d['se']:.4f}), so pooling is not averaging opposite "
            f"relationships. The pooled coefficient stands, and part C can be read at face "
            f"value: if his term is confidently positive there (raw-trait convention, wrong "
            f"direction), it really is backwards as previously stated."
        )
    else:
        label = "not established"
        reasoning = (
            f"The hand difference is {d['mean']:+.4f} +/- {d['se']:.4f} (P={p_differ:.3f}), "
            f"short of the P>=0.95 bar, and the individual coefficients (RHP {rhp:+.4f}, "
            f"established={b['RHP']['direction_established']}; LHP {lhp:+.4f}, established="
            f"{b['LHP']['direction_established']}) do not settle it either. Most likely an "
            f"LHP sample-size problem given the D1 handedness split; would be settled by a "
            f"larger LHP four-seam pool or by part D showing mirroring confidently improving "
            f"out-of-sample fit even while the coefficient split stays ambiguous."
        )
    return label, reasoning


def main() -> int:
    args = fc.paths()  # accepts --data/--workdir/--years/--level for CLI parity with the
    del args           # rest of the suite; this script's actual sources are the fixed
                        # SCORE_WORKDIR/CRIT_WORKDIR pair below, matching coach_model_ff_
                        # criterion.py, so the comparison is pinned regardless of what a
                        # caller passes on the command line.

    print(f"loading graded-season build ({ffc.SCORE_WORKDIR}, 2024/2025 role years)...")
    score_ff = ffc._frame(ffc.SCORE_WORKDIR, "2024,2025")
    print(f"loading next-season criterion build ({ffc.CRIT_WORKDIR}, 2025/2026 role years)...")
    crit_ff = ffc._frame(ffc.CRIT_WORKDIR, "2025,2026")

    out = {}
    out["A"] = part_a(score_ff)
    out["B"] = part_b(score_ff)
    out["C"] = part_c(score_ff, crit_ff)
    out["D"] = part_d(score_ff, crit_ff)

    label, reasoning = verdict_from(out)
    print(f"\n=== VERDICT: {label} ===")
    print(reasoning)
    out["verdict"] = label
    out["reasoning"] = reasoning

    os.makedirs(ffc.SCORE_WORKDIR, exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
