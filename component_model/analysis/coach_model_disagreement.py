"""The 25% where our Stuff+ and his card disagree, plus his card's location blind spot.

The head-to-head (coach_model_ff_criterion.py) is CLOSED as an expected tie: across
seven criterion variants no comparison ever cleared P>=0.95 in either direction, and
the power requirement (~9,100 pitcher-seasons) is nowhere near the ~940 available. This
script does NOT reopen that comparison. It asks the two follow-up questions that still
have power because they do not depend on separating two r=0.73-0.75 correlated scores
outright:

  A1  Residual test. Regress each score on the other (z-scored, pitcher level) and ask
      whether the LEFTOVER part of either score -- the part the other model does not
      contain -- predicts next season's four-seam run value. This is a fair two-sided
      test: if neither residual predicts anything, that is real evidence the two cards
      differ only by noise, not a null result to explain away.
  A2  Names the residual in physical/card terms: which of our 12 ridge features, and
      which of his card's own term columns, correlate with our residual, so a finding
      can be stated as "our model rewards X, which his card ignores or discounts."
      Also reports his card's own per-term isolated contribution against the criterion,
      factually -- if a term is signed backwards, this is where it would show, and it
      is reported as a number, not editorialized.
  A3  His card has no location information at all. This is a clean structural split
      (not a residual): cross-tab his tercile against Location+'s tercile and look at
      pitchers his card is lukewarm/cold on but Location+ likes (and the mirror image),
      to see whether Location+ is catching something real that his card cannot see by
      construction.

SIGN CONVENTION (fair_criterion.py, do not re-derive): Target/xT/adjT/ridge_pred and
location-map values are expected runs from the PITCHER's perspective, LOWER = BETTER.
`coach_hi`, `stuff_hi`, `loc_hi` (from coach_model_ff_criterion.build) are already
flipped ONCE into higher-is-better display frames. The criterion columns
(crit100_Target, crit100_xT, crit100_adjT) stay in the lower-is-better runs frame.
A higher-is-better score/residual that is doing its job therefore correlates
NEGATIVELY with a crit100_* column, and a tercile `spread` (worst minus best, from
coach_model_ff_criterion.spread) is POSITIVE when it sorts correctly. Nothing here is
negated a second time.

KNOWN LIMITATIONS:
  - adjT-based criteria (crit100_adjT) are our own run model; crit100_Target is the
    external, unmodeled comparator (what actually happened). Both are reported.
  - A1's residuals are linear-regression residuals off z-scored pitcher-season means,
    not a held-out prediction; interpret magnitudes as descriptive, not as an
    out-of-sample forecast.
  - Sample floors of 51 and 100 four-seams per pitcher-season are both well under the
    ~9,100 pitcher-season power requirement for the head-to-head; this script tests
    narrower, higher-power questions instead, but small-cell reads in A3 are still
    guarded (n<15 is reported as an anecdote, not a trend).

Data rules: reads via coach_model_ff_criterion's fixed workdir/data constants (never
invents new paths). Writes ONE summary JSON to SCORE_WORKDIR, aggregate numbers only --
no PitcherId-level rows, no names. Never committed (this file itself is untracked;
do not `git add` it).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import coach_model_comparison as cm
import coach_model_ff_criterion as cfc
import fair_criterion as fc

FLOORS = [51, 100]  # 51 = measured project median FF/season; 100 = the "full read" floor
CRITS = ["crit100_adjT", "crit100_Target"]
RESID_COLS = ["ours_resid", "his_resid"]
RESID_LABELS = {"ours_resid": "Our Stuff+ residual (net of his card)",
                "his_resid": "His card residual (net of our Stuff+)"}
N_BOOT = 4000
N_BOOT_A3 = 4000
MIN_CELL = 15
PRIMARY_FLOOR = 100  # A2/A3 run once, on the "full read" pool


def _ols_resid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residual of y regressed on x with an intercept (matches the lstsq pattern used
    elsewhere in this suite, e.g. coach_model_coach_units.effect_per_sd)."""
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def add_resid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    zc = fc.z(df["coach_hi"]).values
    zs = fc.z(df["stuff_hi"]).values
    df["ours_resid"] = _ols_resid(zs, zc)  # what Stuff+ has that his card does not
    df["his_resid"] = _ols_resid(zc, zs)   # what his card has that Stuff+ does not
    return df


def add_tercile(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col + "_t"] = pd.qcut(fc.z(df[col]), 3, labels=cfc.TERCILES)
    return df


# ---------------------------------------------------------------------------
# A1 -- residual test
# ---------------------------------------------------------------------------

def run_a1(floor: int, seed: int) -> tuple[pd.DataFrame, dict]:
    df = cfc.build(floor, floor)
    df = add_resid(df)
    for c in RESID_COLS:
        add_tercile(df, c)

    print(f"\n=== A1: residual test, floor {floor} (n={len(df)}) ===")
    out = {"n": len(df), "by_criterion": {}}
    idx = df.index.values

    for crit in CRITS:
        print(f"\n  -- criterion {crit} (runs/100, lower=better; pool mean "
              f"{df[crit].mean():+.2f}) --")
        point = {}
        for c in RESID_COLS:
            r = fc.R(df[c], df[crit])
            sp = cfc.spread(df, c, crit)
            point[c] = dict(corr=r, spread=sp)
            print(f"    {RESID_LABELS[c]:<45} corr={r:+.3f}   "
                  f"tercile spread={sp:+.2f} runs/100")

        rng = np.random.default_rng(seed)
        B = {c: [] for c in RESID_COLS}
        for _ in range(N_BOOT):
            s = df.loc[rng.choice(idx, len(idx))]
            for c in RESID_COLS:
                s[c + "_t"] = pd.qcut(fc.z(s[c].values), 3, labels=cfc.TERCILES)
                B[c].append(cfc.spread(s, c, crit))
        B = {k: np.array(v) for k, v in B.items()}

        print("    bootstrap on the spread itself (does each residual sort at all?):")
        fc.boot_report(f"      {RESID_LABELS['ours_resid']}", B["ours_resid"])
        fc.boot_report(f"      {RESID_LABELS['his_resid']}", B["his_resid"])
        print("    paired bootstrap on the DIFFERENCE (same resamples):")
        diff = B["ours_resid"] - B["his_resid"]
        fc.boot_report("      ours_resid - his_resid", diff)

        p_ours = float((B["ours_resid"] > 0).mean())
        p_his = float((B["his_resid"] > 0).mean())
        p_diff_pos = float((diff > 0).mean())
        neither = p_ours < 0.95 and p_his < 0.95
        if neither:
            print("    READ: neither residual clears P>=0.95 on its own spread -- the "
                  "honest conclusion here is that the difference between the two cards "
                  "is noise on this criterion, not evidence for either side.")
        elif p_ours >= 0.95 and p_his < 0.95:
            print("    READ: only OUR residual sorts at P>=0.95 -- directional evidence "
                  "our model carries something his card does not.")
        elif p_his >= 0.95 and p_ours < 0.95:
            print("    READ: only HIS residual sorts at P>=0.95 -- directional evidence "
                  "his card carries something our model does not.")
        else:
            print("    READ: both residuals sort individually; the DIFFERENCE bootstrap "
                  "above is the one that says whether either edge is distinguishable.")

        out["by_criterion"][crit] = dict(
            pool_mean=round(float(df[crit].mean()), 3),
            point=point,
            boot=dict(
                ours_resid=dict(mean=round(float(B["ours_resid"].mean()), 3),
                                 se=round(float(B["ours_resid"].std()), 3),
                                 p_gt0=round(p_ours, 3)),
                his_resid=dict(mean=round(float(B["his_resid"].mean()), 3),
                                se=round(float(B["his_resid"].std()), 3),
                                p_gt0=round(p_his, 3)),
                diff=dict(mean=round(float(diff.mean()), 3), se=round(float(diff.std()), 3),
                          p_gt0=round(p_diff_pos, 3))))
    return df, out


# ---------------------------------------------------------------------------
# A2 -- name the residual
# ---------------------------------------------------------------------------

def run_a2(df100: pd.DataFrame) -> dict:
    print(f"\n=== A2: what does the Stuff+ residual reward? (floor {PRIMARY_FLOOR} pool) ===")
    terms = cm.load_coach_terms("FourSeamFastBall")
    used = sorted({t["col"] for t in terms})

    ff = cfc._frame(cfc.SCORE_WORKDIR, "2024,2025")
    ff = ff[ff["year"] == 2025].dropna(subset=fc.FEATS)

    pf = ff.groupby("PitcherId")[fc.FEATS].mean()
    pf = pf.loc[pf.index.intersection(df100.index)]
    merged = pf.join(df100[["ours_resid"]], how="inner")
    print(f"  pitcher-level feature means, n={len(merged)}")

    rows = []
    for feat in fc.FEATS:
        r = fc.R(merged["ours_resid"], merged[feat])
        rows.append((feat, r, feat in used))
    rows.sort(key=lambda x: -abs(x[1]))

    print(f"\n  {'feature':<24}{'r vs ours_resid':>17}   in his card?")
    for feat, r, in_card in rows:
        print(f"  {feat:<24}{r:>+17.3f}   {'yes' if in_card else 'no'}")

    # his card's per-term isolated contribution vs the next-year criterion, factually
    sub = ff[ff["PitcherId"].isin(df100.index)].dropna(subset=used)
    print("\n  his card, per-term isolated contribution vs next-year criterion "
          "(higher-is-better term score; a well-oriented term correlates NEGATIVE "
          "with runs/100):")
    term_rows = []
    for col in used:
        only = [t for t in terms if t["col"] == col]
        part = cm.coach_score(sub, only, 1.0)  # his frame, higher = better, per-term
        pv = pd.DataFrame({"p": part, "PitcherId": sub["PitcherId"]}).groupby("PitcherId").mean()
        pv = pv.join(df100[["crit100_adjT", "crit100_Target"]], how="inner")
        r_adj = fc.R(pv["p"], pv["crit100_adjT"])
        r_tgt = fc.R(pv["p"], pv["crit100_Target"])
        term_rows.append(dict(term=col, r_vs_crit100_adjT=r_adj, r_vs_crit100_Target=r_tgt))
        print(f"    {col:<22} r vs crit100_adjT={r_adj:+.3f}   r vs crit100_Target={r_tgt:+.3f}")

    return dict(n=len(merged),
                feature_corrs=[dict(feature=f, r=round(r, 3), in_coach_card=b)
                               for f, r, b in rows],
                term_isolated_contribution=[
                    dict(term=t["term"], r_vs_crit100_adjT=round(t["r_vs_crit100_adjT"], 3),
                         r_vs_crit100_Target=round(t["r_vs_crit100_Target"], 3))
                    for t in term_rows])


# ---------------------------------------------------------------------------
# A3 -- the Location+ blind spot
# ---------------------------------------------------------------------------

def _boot_group_diff(gA: pd.DataFrame, gB: pd.DataFrame, crit: str, n_boot: int, rng) -> np.ndarray:
    """Two-sample bootstrap of mean(gA[crit]) - mean(gB[crit]); groups resampled
    independently within their own (fixed) membership."""
    idxA, idxB = gA.index.values, gB.index.values
    out = np.empty(n_boot)
    for i in range(n_boot):
        a = gA.loc[rng.choice(idxA, len(idxA))][crit].mean()
        b = gB.loc[rng.choice(idxB, len(idxB))][crit].mean()
        out[i] = a - b
    return out


def run_a3(df: pd.DataFrame, floor: int) -> dict:
    df = df.copy()
    add_tercile(df, "coach_hi")
    add_tercile(df, "loc_hi")

    print(f"\n=== A3: Location+ blind spot, floor {floor} (n={len(df)}) ===")
    ct = pd.crosstab(df["coach_hi_t"], df["loc_hi_t"])
    ct = ct.reindex(index=cfc.TERCILES, columns=cfc.TERCILES)
    print("  cross-tab, rows=his card tercile, cols=Location+ tercile (counts):")
    print(ct.to_string())

    cell_counts = {}
    for r in cfc.TERCILES:
        for c in cfc.TERCILES:
            n = int(((df["coach_hi_t"] == r) & (df["loc_hi_t"] == c)).sum())
            cell_counts[f"{r}|{c}"] = n
            if n < MIN_CELL:
                print(f"    [anecdote] coach={r}, loc={c}: n={n} (<{MIN_CELL}, not a trend)")

    def cell(coach_terciles: list[str], loc_tercile: str) -> pd.DataFrame:
        return df[df["coach_hi_t"].isin(coach_terciles) & (df["loc_hi_t"] == loc_tercile)]

    groups = {
        "headline_loc_best": cell(["middle", "worst third"], "best third"),
        "headline_loc_worst": cell(["middle", "worst third"], "worst third"),
        "honesty_loc_worst": cell(["best third"], "worst third"),
        "honesty_loc_best": cell(["best third"], "best third"),
    }
    pool_mean = {crit: float(df[crit].mean()) for crit in CRITS}

    print(f"\n  pool mean: " + "  ".join(f"{c}={pool_mean[c]:+.2f}" for c in CRITS))
    group_out = {}
    for name, g in groups.items():
        n = len(g)
        flag = f"  [anecdote, n<{MIN_CELL}]" if n < MIN_CELL else ""
        means = {c: float(g[c].mean()) if n else float("nan") for c in CRITS}
        group_out[name] = dict(n=n, means=means)
        print(f"  {name:<20} n={n:>4}{flag}  " +
              "  ".join(f"{c}={means[c]:+.2f}" for c in CRITS))

    print("\n  headline exhibit: his card says middle-or-worst; does Location+ still "
          "separate them? (positive = his-worse-location pitchers really do allow more "
          "runs, i.e. Location+ is catching something his card structurally cannot)")
    print("  honesty exhibit: his card says best; does Location+ still separate them "
          "the same way, even when his card is already happy?")

    rng = np.random.default_rng(53)
    boot_out = {}
    for label, (nameA, nameB) in [
        ("headline: coach mid/worst, loc worst minus loc best", ("headline_loc_worst", "headline_loc_best")),
        ("honesty: coach best, loc worst minus loc best", ("honesty_loc_worst", "honesty_loc_best")),
    ]:
        gA, gB = groups[nameA], groups[nameB]
        if len(gA) < MIN_CELL or len(gB) < MIN_CELL:
            print(f"\n  {label}: skipped, a group has n<{MIN_CELL} "
                  f"({nameA}={len(gA)}, {nameB}={len(gB)}) -- anecdote, not tested")
            boot_out[label] = None
            continue
        print(f"\n  {label}:")
        crit_out = {}
        for crit in CRITS:
            d = _boot_group_diff(gA, gB, crit, N_BOOT_A3, rng)
            fc.boot_report(f"    {crit}", d)
            crit_out[crit] = dict(mean=round(float(d.mean()), 3), se=round(float(d.std()), 3),
                                   p_gt0=round(float((d > 0).mean()), 3))
        boot_out[label] = crit_out

    return dict(n=len(df), cell_counts=cell_counts, pool_mean=pool_mean,
                groups=group_out, bootstrap=boot_out)


# ---------------------------------------------------------------------------

def main() -> int:
    # No fc.paths() call here: coach_model_ff_criterion.build() stubs its own sys.argv
    # with fixed SCORE_WORKDIR/CRIT_WORKDIR/DATA constants internally, so this script
    # takes no --data/--workdir of its own (matches how coach_model_ff_criterion.main
    # is invoked).
    summary = {"a1": {}, "a2": None, "a3": None}
    df_by_floor = {}
    for floor in FLOORS:
        df, a1_out = run_a1(floor, seed=41 + floor)
        df_by_floor[floor] = df
        summary["a1"][floor] = a1_out

    df100 = df_by_floor[PRIMARY_FLOOR]
    summary["a2"] = run_a2(df100)
    summary["a3"] = run_a3(df100, PRIMARY_FLOOR)

    dest = os.path.join(cfc.SCORE_WORKDIR, "coach_model_disagreement.json")
    with open(dest, "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
