"""Does swapping EffectiveVelo for RelSpeed make the Extension coefficient interpretable?

THE PROBLEM: our ridge reports "less extension is better" (4th-ish strongest feature). That
is almost certainly suppression, not a finding. EffectiveVelo is perceived velocity, which
is ALREADY a function of release speed and extension. Holding perceived velo fixed, more
extension necessarily implies less raw arm speed -- so the leftover Extension term prices
"got here without arm speed", and reads backwards. The current design bakes in TrackMan's
fixed effective-velo formula and then bolts a residual extension term onto it.

THE TEST: give the ridge RelSpeed (real release velocity) instead, and let it learn the
velo/extension tradeoff itself rather than inheriting a formula.

  baseline   FEATS as shipped (EffectiveVelo, no RelSpeed)
  swap       EffectiveVelo -> RelSpeed
  both       RelSpeed added alongside EffectiveVelo (expected to be badly collinear;
             reported so the collinearity is visible rather than assumed)

ADOPT ONLY IF BOTH HOLD: Extension turns sensibly positive (more is better) AND predictive
validity does not drop. A cleaner-reading coefficient bought with worse prediction is not a
fix, it is a preference. Everything on the coach page derives from ridge_pred, so adopting
means re-rendering all four sections -- do not swap the shipping FEATS from inside here.

PLUMBING: this worktree's fair_criterion lacks the OPTIONAL_COLS reader that lives on
branch real-velo-context, so RelSpeed is appended to USECOLS at import time instead of
editing the shared module. RelSpeed is NOT in source_2025_2026.csv -- pass
source_2025_2026_relspeed.csv (or _realvelo_v2) and a FRESH --workdir, or the existing
cache will be served without the column and/or clobbered.

SIGN CONVENTION: Target and ridge_pred are expected run value from the pitcher's
perspective, LOWER = BETTER, relative to an average pitcher. A standardised coefficient is
reported in that frame, so a NEGATIVE coefficient means more of the feature predicts FEWER
runs, i.e. better for the pitcher -- that is the plain-words direction printed. Predictive
validity correlates a fitted prediction with the criterion, both lower=better, so it is
POSITIVE when correctly oriented and larger is better (fitted-prediction rule, not the
raw-trait rule).

Data rules: reads the source CSV and its own workdir; writes one JSON there. No names.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import fair_criterion as fc

if "RelSpeed" not in fc.USECOLS:          # must happen before any load_pitches call
    fc.USECOLS = fc.USECOLS + ["RelSpeed"]

import coach_model_comparison as cm  # noqa: E402  (after the USECOLS patch)

FLOOR = 100
N_BOOT = 2000          # validity bootstrap (cheap: resamples pitcher-level rows only)
N_BOOT_COEF = 200      # coefficient bootstrap (expensive: refits the ridge each draw)
# Non-inferiority margin on validity, in criterion units (runs per 100). The swap is a
# reparameterisation, not a bid to predict better, so the gate is "does not lose more than
# this", NOT "predicts better" -- see the review. 0.01 is ~9% of the 0.112 baseline.
MARGIN = 0.01
BASE = list(fc.FEATS)
VARIANTS = {
    "baseline": BASE,
    "swap": [("RelSpeed" if f == "EffectiveVelo" else f) for f in BASE],
    "both": BASE + ["RelSpeed"],
}


def frame(data: str, workdir: str, level: str, years: str) -> pd.DataFrame:
    """Build one year-pair. workdir is EXPLICIT so the scoring and criterion pairs land in
    SEPARATE caches -- reusing one workdir for both --years lets load_pitches serve the
    first build for the second, which silently turns validity into an inflated same-season
    number. That was the bug in the first draft of this script."""
    import sys
    saved, sys.argv = sys.argv, ["x", "--data", data, "--workdir", workdir,
                                 "--years", years, "--level", level or "D1"]
    a = fc.paths()
    sys.argv = saved
    df = fc.load_pitches(a)
    fc.add_xt(df)
    fc.add_adjusted(df)
    return fc.stuff_ridge(df)


def cluster_bootstrap_ext(train: pd.DataFrame, feats: list[str], n_boot: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Resample PITCHERS (whole clusters), refit, return the standardised Extension
    coefficient each draw. This is the only way to put an error bar on the SIGN: the
    validity bootstrap below freezes the coefficients, so it says nothing about whether
    the extension flip is real or resampling noise on a coefficient that is the smallest
    in the model."""
    j = feats.index("Extension")
    groups = train.groupby("PitcherId").indices
    ids = np.array(list(groups.keys()))
    X, y = train[feats].values, train["Target"].values
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.choice(ids, len(ids), replace=True)])
        m = make_pipeline(StandardScaler(), Ridge(alpha=10))
        m.fit(X[idx], y[idx])
        out[b] = m[-1].coef_[j]
        if (b + 1) % 50 == 0:
            print(f"      coef boot {b + 1}/{n_boot}", flush=True)
    return out


def fit(ff: pd.DataFrame, feats: list[str]):
    train = ff[(ff["year"] == 2024) & ff["Target"].notna()]
    m = make_pipeline(StandardScaler(), Ridge(alpha=10))
    m.fit(train[feats].values, train["Target"].values)
    return m, dict(zip(feats, m[-1].coef_))


def main() -> int:
    import argparse
    args = fc.paths()
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--crit-workdir", default=None,
                    help="FRESH criterion workdir; defaults to <workdir>_crit. Must differ "
                         "from --workdir or validity is silently inflated.")
    crit_workdir = (ap.parse_known_args()[0].crit_workdir
                    or args.workdir.rstrip("\\/") + "_crit")
    if crit_workdir == args.workdir:
        raise SystemExit("--crit-workdir must differ from --workdir")
    print(f"  scores    {args.workdir}")
    print(f"  criterion {crit_workdir}")

    # Fail fast on the CSV side of the plumbing (instant) before the minutes-long frame
    # build. This catches "wrong extract"; it does NOT prove the loader keeps the column --
    # that is the real-velo-context question, and the post-load `missing` check below is
    # what catches a loader that drops RelSpeed despite the USECOLS patch.
    header = pd.read_csv(args.data, nrows=0).columns
    if "RelSpeed" not in header:
        raise SystemExit(
            f"{args.data} has no RelSpeed column. Pass source_2025_2026_relspeed.csv or "
            f"_realvelo_v2 -- source_2025_2026.csv does not carry it. (Checked the header "
            f"only, so this failed in seconds rather than after the frame build.)")

    ff = frame(args.data, args.workdir, args.level, "2024,2025")
    need = sorted({c for v in VARIANTS.values() for c in v})
    missing = [c for c in need if c not in ff.columns]
    if missing:
        raise SystemExit(
            f"source lacks {missing}. Two independent causes -- check both: (1) --data must "
            f"point at a CSV carrying RelSpeed (source_2025_2026.csv does not; use "
            f"source_2025_2026_relspeed.csv or _realvelo_v2). (2) fair_criterion must read "
            f"it through -- the USECOLS patch at the top of this file covers the load, but a "
            f"STALE cache in --workdir will be served without the column; use a fresh workdir.")
    ff = ff.dropna(subset=need + ["Target"]).copy()

    terms = cm.load_coach_terms("FourSeamFastBall")
    ff["coach_raw"] = cm.coach_score(ff, terms, 1.0)

    c = frame(args.data, crit_workdir, args.level, "2025,2026")
    k = c[c["year"] == 2025].groupby("PitcherId").agg(
        n26=("Target", "size"), crit=("Target", "mean"))
    k = k[k["n26"] >= FLOOR]

    season = ff[ff["year"] == 2025]
    preds, coefs = {}, {}
    for name, feats in VARIANTS.items():
        model, co = fit(ff, feats)
        coefs[name] = co
        preds[name] = model.predict(season[feats].values)

    g = pd.DataFrame({"PitcherId": season["PitcherId"].values,
                      "coach": season["coach_raw"].values,
                      **{n: p for n, p in preds.items()}}).groupby("PitcherId").agg(
        n=("coach", "size"), coach=("coach", "mean"),
        **{n: (n, "mean") for n in VARIANTS})
    g = g[g["n"] >= FLOOR].join(k, how="inner")
    g["crit100"] = g["crit"] * 100
    print(f"\n  pool: {len(g)} pitchers, {FLOOR}+ four-seams both seasons")

    # TRIPWIRE: the baseline must reproduce the validity on record (coach_handedness.json,
    # variant 'full', n=825 -> 0.1121). If it does not, the pool moved -- most likely the
    # RelSpeed extract has different missingness so dropna shrank it differently -- and
    # baseline-vs-swap is then confounded by pool drift, not the feature change.
    base_val = float(fc.R(g["baseline"], g["crit100"]))
    if abs(len(g) - 825) > 40 or abs(base_val - 0.1121) > 0.01:
        print(f"  WARNING: baseline n={len(g)} val={base_val:.4f} does not match the "
              f"n~825, val~0.1121 on record.\n           The pool has drifted (likely "
              f"RelSpeed missingness); baseline-vs-swap is confounded.\n           "
              f"Reconcile before trusting the verdict below.")

    idx = g.index.values
    rng = np.random.default_rng(31)
    boots = [rng.choice(idx, len(idx)) for _ in range(N_BOOT)]

    print(f"\n  {'variant':<10}{'Extension coef':>16}{'reads as':>22}"
          f"{'validity':>10}{'r vs v1':>9}")
    out = {"n": int(len(g)), "floor": FLOOR, "margin": MARGIN,
           "baseline_validity": round(base_val, 4), "variants": {}}
    for name in VARIANTS:
        ext = float(coefs[name]["Extension"])
        # coefficient is on run value (lower = better), so negative => more is better
        reads = "more is better" if ext < 0 else "less is better"
        val = float(fc.R(g[name], g["crit100"]))
        r_v1 = float(fc.R(pd.Series(g["coach"].values), pd.Series(-g[name].values)))
        print(f"  {name:<10}{ext:>+16.4f}{reads:>22}{val:>10.3f}{r_v1:>9.3f}")
        out["variants"][name] = dict(features=VARIANTS[name],
                                     extension_coef=round(ext, 5), reads_as=reads,
                                     validity=round(val, 4), r_vs_v1=round(r_v1, 4),
                                     all_coefs={k2: round(float(v), 5)
                                                for k2, v in coefs[name].items()})

    # Q1 -- the SIGN, with an error bar. Refit per resample; the point estimate alone is a
    # direction read off the smallest coefficient in the model and can flip on noise.
    print(f"\n  Q1 sign: coefficient bootstrap ({N_BOOT_COEF} refits) for baseline & swap...")
    train = ff[(ff["year"] == 2024) & ff["Target"].notna()]
    for name in ("baseline", "swap"):
        cb = cluster_bootstrap_ext(train, VARIANTS[name], N_BOOT_COEF,
                                   np.random.default_rng(31))
        p_more = float((cb < 0).mean())          # coef<0 == more is better
        out["variants"][name].update(
            ext_boot_se=round(float(cb.std()), 5),
            ext_ci95=[round(float(np.percentile(cb, 2.5)), 5),
                      round(float(np.percentile(cb, 97.5)), 5)],
            p_more_is_better=round(p_more, 4),
            sign_established=bool(p_more >= 0.95 or p_more <= 0.05))
        v = out["variants"][name]
        print(f"    {name:<10} coef {coefs[name]['Extension']:+.5f} "
              f"CI[{v['ext_ci95'][0]:+.5f},{v['ext_ci95'][1]:+.5f}] "
              f"P(more is better)={p_more:.3f} established={v['sign_established']}")

    # Q2 -- validity, paired on shared resamples, as a NON-INFERIORITY test.
    print("\n  Q2 validity: paired bootstrap, variant minus baseline:")
    base = np.array([float(np.corrcoef(g.loc[b, "baseline"], g.loc[b, "crit100"])[0, 1])
                     for b in boots])
    for name in ("swap", "both"):
        alt = np.array([float(np.corrcoef(g.loc[b, name], g.loc[b, "crit100"])[0, 1])
                        for b in boots])
        d = alt - base
        fc.boot_report(f"{name} - baseline", d)
        p_ni = float((d > -MARGIN).mean())
        out["variants"][name].update(
            delta_validity=round(float(d.mean()), 4),
            delta_validity_se=round(float(d.std()), 4),
            p_noninferior=round(p_ni, 4),
            validity_holds=bool(p_ni >= 0.95))
        print(f"    {name:<10} P(delta > -{MARGIN}) = {p_ni:.3f} "
              f"(non-inferior={out['variants'][name]['validity_holds']})")

    # VERDICT on the swap only. 'both' is a span check (badly collinear by construction),
    # never a candidate. Adopt requires the sign ESTABLISHED positive AND validity
    # non-inferior -- a point-estimate flip is not enough.
    sw = out["variants"]["swap"]
    adopt = (sw["reads_as"] == "more is better" and sw["sign_established"]
             and sw["validity_holds"])
    verdict = ("ADOPT: extension is established more-is-better AND validity is non-inferior"
               if adopt else
               "DO NOT ADOPT on this evidence; reframe/hide the extension row instead")
    print(f"\n  => {verdict}")
    print(f"     sign established more-is-better: "
          f"{sw['reads_as'] == 'more is better' and sw['sign_established']}; "
          f"validity non-inferior: {sw['validity_holds']}")
    print("     ('both' is a collinearity span-check, excluded from the verdict.)")
    print("\n  IF ADOPT: this changes ridge_pred, so re-render all four page sections; it "
          "is not\n  an edit to the extension row. See RUNBOOK.md.")
    out["verdict"] = verdict
    out["adopt"] = bool(adopt)

    os.makedirs(args.workdir, exist_ok=True)
    dest = os.path.join(args.workdir, "coach_extension_fix.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
