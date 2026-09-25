"""Four-seam Location+ valued for the pitch's own shape, against the shipped map. Pre-registration
and verdicts: docs/notes/ff-location-shape-ledger-2026-09.md. Read that before running anything.

WHY. The shipped four-seam Location+ values a location by what an AVERAGE four-seam does there.
A flat, high-ride fastball and a steep, sinking one are not the same pitch at the top of the
zone: the flat one misses bats up there and the steep one gets hit. A pitcher with a steep
fastball who lives at the letters is credited by the shipped map for a location that is good
for other people's fastballs. The candidates here value each pitch against pitches of its own
shape: E[xT | location, shape bin] - E[xT | shape bin]. Subtracting the bin's mean means a
better fastball earns nothing by itself; only putting it where pitches of its shape succeed
does (one construct per score, FRAMEWORK.md).

What earlier reads already settled for the four-seam, so nothing here re-reads them: a
count-relative map ties the pooled map (script 04), and batter-mirrored or platoon-split maps
do not move it (script 09). Shape is the untested axis.

THE GATE is coach_location_gate.py's, imported not re-derived: 50/50 z-blend of grade and prior
results, P(blend gain > 0) over 200 cluster-bootstrap refits that resample TRAIN pitchers
(frequency weights on every map fit) and CRITERION pitchers separately, pool 15+ pitches and
10%+ share in both seasons, bar 0.95, seed 20260817, criterion = next-season mean four-seam
adjT. The pair structure (C confirmation, D cross-fitted discovery) is that harness's too.
What is new is the PAIRED comparison: every candidate is refit on the same 200 draws as the
control, and the statistic is P(gain_candidate - gain_control > 0).

CANDIDATES (exact definitions in the ledger)
  CONTROL   the shipped four-seam Location+: catcher-frame pooled map on xT.
  SHAPE_V   CONTROL's frame, one surface per tercile of height-adjusted approach angle
            (vaa_flat, coach_si_feature_gate.add_vaa_flat: release physics only, never the
            pitch's own plate location), value relative to the tercile's mean.
  SHAPE_VH  batter_platoon frame (so arm side vs glove side is known), one surface per
            vaa_flat tercile x arm-side-run tercile x platoon, relative to that cell's mean.
Tercile cut points come from the 2024 train rows and are fixed in every refit. A row missing
a shape input goes to its own bin.

AUDIT (reported for every candidate, no bar): r(grade, four-seam Stuff+ grade); the three-way
z-blend (results + Stuff+ + Location+) paired against the same blend with CONTROL, Stuff+ held
at its full-sample fit; and on pair D the year-over-year reliability of the grade.

SIGN CONVENTION: xT, every location value, adjT, the prior and ridge_pred are expected run
value from the pitcher's perspective, LOWER = BETTER. A correctly oriented grade correlates
POSITIVELY with the criterion. Nothing is negated.

Data rules: reads the existing D1 pitch caches only and refuses to run if a cache is missing or
would be rebuilt; writes one JSON of aggregates to the score workdir. No pitcher names, no
per-pitcher output, no absolute paths. Usage, from component_model/analysis with
STUFFPLUS_DATA, STUFFPLUS_DATA_CRIT, STUFFPLUS_WORKDIR and STUFFPLUS_WORKDIR_CRIT set:
    python coach_ff_location_shape_gate.py --pair D
    python coach_ff_location_shape_gate.py --pair C --dry   # no criterion touched
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import fair_criterion as fc
import location_maps as lm
from coach_location_gate import (ABS_MIN, CRIT_WORKDIR, DATA, DATA_CRIT, FOLD_SEED,
                                 MIN_PITCHERS, N_BOOT, N_FOLDS, PASS_BAR, SCORE_WORKDIR,
                                 SEED, SHARE, _refuse_rebuild, _stats, _z)
from coach_si_feature_gate import add_vaa_flat

GRP = "FF"
N_Q = 3
CANDIDATES = {
    "CONTROL": {"frame": "catcher", "shape": ()},
    "SHAPE_V": {"frame": "catcher", "shape": ("vaa_flat",)},
    "SHAPE_VH": {"frame": "batter_platoon", "shape": ("vaa_flat", "HorzBreak_arm")},
}
TESTS = ("SHAPE_V", "SHAPE_VH")


def shape_columns(rows: pd.DataFrame) -> pd.DataFrame:
    """vaa_flat and arm-side horizontal break per row, from release physics only."""
    d = rows[["RelSpeed", "Extension", "InducedVertBreak", "RelHeight"]].astype(float).copy()
    add_vaa_flat(d)
    hb = rows["HorzBreak"].astype(float) * (1 - 2 * rows["is_lhp"].astype(float))
    return pd.DataFrame({"vaa_flat": d["vaa_flat"].values, "HorzBreak_arm": hb.values},
                        index=rows.index)


def shape_bins(shape: pd.DataFrame, cols, train_mask, n_q=N_Q):
    """(bin per row, cut points). Tercile cut points per column from the train rows only; the
    bin index combines columns in order (first column slowest). A row missing any shape input
    gets bin n_q ** len(cols), its own bin. No columns: every row is bin 0."""
    n = len(shape)
    if not cols:
        return np.zeros(n, dtype=np.int64), {}
    b = np.zeros(n, dtype=np.int64)
    bad = np.zeros(n, dtype=bool)
    cuts = {}
    for c in cols:
        v = shape[c].values.astype(float)
        fin = np.isfinite(v)
        q = np.quantile(v[train_mask & fin], np.arange(1, n_q) / n_q)
        cuts[c] = [float(x) for x in q]
        b = b * n_q + np.searchsorted(q, np.where(fin, v, 0.0), side="right")
        bad |= ~fin
    b[bad] = n_q ** len(cols)
    return b, cuts


class FFData:
    """Four-seam rows for one pair: per-candidate cell codes, xT, pitchers, masks, folds."""

    def __init__(self, frame_df, train_year, years, folds_of, cross_fit):
        m = fc.pitch_mask(frame_df, GRP) & frame_df["year"].isin(years)
        rows = frame_df[m]
        # One row filter for every candidate (the batter_platoon validity rule, as in
        # coach_location_gate.GroupData), so all candidates share one pool.
        ok = lm.frame_columns(rows, "batter_platoon")["ok"]
        rows = rows[ok & rows["Target"].notna()]
        self.year = rows["year"].values
        self.xt = rows["xT"].values.astype(float)
        self.pid = rows["PitcherId"].values
        self.adjT = rows["adjT"].values.astype(float)
        self.train = (self.year == train_year) & np.isfinite(self.xt)
        self.cross_fit = cross_fit
        self.fold = np.array([folds_of.get(p, 0) for p in self.pid])
        self.xz = lm.frame_columns(rows, "catcher")[["x", "z"]].reset_index(drop=True)
        shape = shape_columns(rows)
        self.shape_missing = float((~np.isfinite(shape.values)).any(axis=1).mean())
        self.codes, self.cuts, self.bins = {}, {}, {}
        for name, spec in CANDIDATES.items():
            fr = lm.frame_columns(rows, spec["frame"])
            n_p = 1 if spec["frame"] == "catcher" else 2
            b, cuts = shape_bins(shape, spec["shape"], self.train)
            # Shape bin folded into the platoon code: one surface per (bin, platoon), each
            # falling back to its own mean, and relative() subtracts E[xT | bin, platoon].
            fr["p"] = fr["p"].values + n_p * b
            self.codes[name] = lm.cell_codes(fr)
            self.cuts[name], self.bins[name] = cuts, b

    def values(self, name, w_row):
        """Relative location value for EVERY row, maps fitted on train rows with weights
        w_row. Cross-fit: a row is valued by the map fitted without its pitcher's fold."""
        codes = self.codes[name]
        if not self.cross_fit:
            t = np.where(self.train)[0]
            mp = lm.CellMap(lm.subset_codes(codes, t), self.xt[t], w_row[t])
            return mp.relative(codes)
        out = np.empty(len(self.pid))
        for f in range(N_FOLDS):
            t = np.where(self.train & (self.fold != f))[0]
            e = np.where(self.fold == f)[0]
            mp = lm.CellMap(lm.subset_codes(codes, t), self.xt[t], w_row[t])
            out[e] = mp.relative(lm.subset_codes(codes, e))
        return out


def grade_of(vals, pid, sel):
    """Pitcher mean relative value over the selected rows, pitchers with ABS_MIN+ rows."""
    g = pd.DataFrame({"v": vals[sel], "p": pid[sel]}).groupby("p")["v"].agg(["size", "mean"])
    return g.loc[g["size"] >= ABS_MIN, "mean"].rename("grade")


def design_diagnostics(D, eval_sel):
    """Criterion-free checks from the maps' own xT training: does the value of the top and
    bottom of the zone move with shape, and how different are the candidates' grades?"""
    ones = np.ones(len(D.pid))
    vals = {n: D.values(n, ones) for n in CANDIDATES}
    x, z = D.xz["x"].values, D.xz["z"].values
    top = (np.abs(x) < 0.83) & (z > 2.9) & (z < 3.6)
    bot = (np.abs(x) < 0.83) & (z > 1.5) & (z < 2.2)
    lines = []
    b = D.bins["SHAPE_V"]
    for k, lab in enumerate(["steepest third", "middle third", "flattest third"]):
        # vaa_flat is negative (downward); the HIGHEST tercile is the flattest approach.
        s = eval_sel & (b == k)
        lines.append("%s: top of zone %+.2f, bottom %+.2f (x100, SHAPE_V relative value; "
                     "CONTROL %+.2f / %+.2f)" % (
                         lab, 100 * vals["SHAPE_V"][s & top].mean(),
                         100 * vals["SHAPE_V"][s & bot].mean(),
                         100 * vals["CONTROL"][s & top].mean(),
                         100 * vals["CONTROL"][s & bot].mean()))
    g = pd.DataFrame({n: grade_of(v, D.pid, eval_sel) for n, v in vals.items()}).dropna()
    corr = g.corr()
    return lines, {"grade_r_CONTROL_SHAPE_V": round(float(corr.loc["CONTROL", "SHAPE_V"]), 4),
                   "grade_r_CONTROL_SHAPE_VH": round(float(corr.loc["CONTROL", "SHAPE_VH"]), 4),
                   "grade_sd_x100": {n: round(100 * float(g[n].std()), 3) for n in g},
                   "n_pitchers_15": int(len(g))}


def reliability(D, vals_by_name, pool_ids):
    """Year-over-year r of each candidate's grade (2024 vs 2025 score build, both valued by
    the pitcher's fold-excluded 2024 map), plus a paired pitcher bootstrap of r - r_CONTROL."""
    tabs = {}
    for n, v in vals_by_name.items():
        a = grade_of(v, D.pid, D.year == 2024)
        b = grade_of(v, D.pid, D.year == 2025)
        tabs[n] = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    ids = sorted(set.intersection(*[set(t.index) for t in tabs.values()]) & set(pool_ids))
    out = {n: {"r": round(float(fc.R(t.loc[ids, "a"], t.loc[ids, "b"])), 4)}
           for n, t in tabs.items()}
    rng = np.random.default_rng(SEED)
    arr = {n: t.loc[ids, ["a", "b"]].values for n, t in tabs.items()}
    diffs = {n: [] for n in TESTS}
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(ids), len(ids))
        rc = np.corrcoef(arr["CONTROL"][ix].T)[0, 1]
        for n in TESTS:
            diffs[n].append(np.corrcoef(arr[n][ix].T)[0, 1] - rc)
    for n in TESTS:
        d = np.array(diffs[n])
        out[n].update({"diff_mean": round(float(d.mean()), 4),
                       "diff_se": round(float(d.std(ddof=1)), 4),
                       "p_diff_positive": float((d > 0).mean())})
    out["n"] = len(ids)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=["C", "D"], required=True)
    ap.add_argument("--dry", action="store_true",
                    help="build maps and pools, print design diagnostics, touch no criterion "
                         "correlation and write nothing")
    args = ap.parse_args()
    t0 = time.time()

    _refuse_rebuild(DATA, SCORE_WORKDIR, "2024,2025")
    score = fc.load_frame(DATA, SCORE_WORKDIR, "2024,2025")
    if args.pair == "C":
        _refuse_rebuild(DATA_CRIT, CRIT_WORKDIR, "2025,2026")
        crit = fc.load_frame(DATA_CRIT, CRIT_WORKDIR, "2025,2026")
        train_year, eval_year = 2024, 2025
        crit_frame, crit_year = crit, 2025
        years, cross_fit = [2024, 2025], False
    else:
        train_year, eval_year = 2024, 2024
        crit_frame, crit_year = score, 2025
        # 2025 rows ride along only for the reliability diagnostic.
        years, cross_fit = [2024, 2025], True
    print("  frames loaded in %.0fs" % (time.time() - t0), flush=True)
    tot = score[score["year"] == eval_year].groupby("PitcherId").size().rename("tot")
    ctot = crit_frame[crit_frame["year"] == crit_year].groupby("PitcherId").size().rename("ctot")

    all_p = np.sort(score["PitcherId"].dropna().unique())
    perm = np.random.default_rng(FOLD_SEED).permutation(len(all_p))
    folds_of = dict(zip(all_p, perm % N_FOLDS))

    D = FFData(score, train_year, years, folds_of, cross_fit)
    ev = D.year == eval_year
    print("  train rows %d, eval rows %d, shape input missing on %.2f%% of rows" % (
        D.train.sum(), ev.sum(), 100 * D.shape_missing), flush=True)
    for n in CANDIDATES:
        print("  %s: %d surfaces, cut points %s" % (
            n, D.codes[n]["n_p"], {k: [round(c, 2) for c in v] for k, v in D.cuts[n].items()}))

    c = crit_frame[fc.pitch_mask(crit_frame, GRP) & (crit_frame["year"] == crit_year)]
    k = c.groupby("PitcherId").agg(cn=("adjT", "size"), crit=("adjT", "mean")).join(ctot)
    k = k[(k["cn"] >= ABS_MIN) & (k["cn"] / k["ctot"] >= SHARE)]
    ed = pd.DataFrame({"PitcherId": D.pid[ev], "adjT": D.adjT[ev]})
    gpool_all = ed.groupby("PitcherId").agg(sn=("adjT", "size"), prior=("adjT", "mean")).join(tot)
    gpool_all = gpool_all[(gpool_all["sn"] >= ABS_MIN) & (gpool_all["sn"] / gpool_all["tot"]
                                                           >= SHARE)]

    if args.dry:
        lines, diag = design_diagnostics(D, ev)
        print("\n  [dry] top/bottom-of-zone value by approach-angle tercile (lower = better):")
        for line in lines:
            print("    " + line)
        print("  [dry] %s" % diag)
        print("\n  dry run: nothing written   total %.0fs" % (time.time() - t0))
        return 0

    # Four-seam Stuff+ for the audit: the shipped ridge, held at its full-sample fit.
    ffr, _, _ = fc.ridge_for_group(score, GRP)
    sr = ffr[ffr["year"] == eval_year].groupby("PitcherId")["ridge_pred"].agg(["size", "mean"])
    stuff = sr.loc[sr["size"] >= ABS_MIN, "mean"].rename("stuff")

    tr_codes, tr_ids = pd.factorize(D.pid[D.train])
    idx_of = {p: i for i, p in enumerate(tr_ids)}
    row_idx = np.array([idx_of.get(p, -1) for p in D.pid])
    ev_ids = pd.unique(D.pid[ev])

    def build(cnt=None, keep=None):
        w = np.ones(len(D.pid)) if cnt is None else np.where(
            row_idx >= 0, cnt[np.maximum(row_idx, 0)], 0.0)
        sel = ev if keep is None else ev & np.isin(D.pid, keep)
        gp = gpool_all if keep is None else gpool_all[gpool_all.index.isin(keep)]
        vals, js = {}, {}
        for n in CANDIDATES:
            vals[n] = D.values(n, w)
            j = grade_of(vals[n], D.pid, sel).to_frame().join(gp, how="inner").join(
                k, how="inner").dropna(subset=["grade", "prior", "crit"])
            js[n] = j
        return js, vals

    def three_way(j):
        a = j.join(stuff, how="inner").dropna(subset=["stuff"])
        r3 = float(fc.R(_z(a["grade"]) + _z(a["prior"]) + _z(a["stuff"]), a["crit"]))
        r2 = float(fc.R(_z(a["prior"]) + _z(a["stuff"]), a["crit"]))
        return r3, r2, float(fc.R(a["grade"], a["stuff"])), len(a)

    t1 = time.time()
    js, vals0 = build()
    out = {"pair": args.pair, "share": SHARE, "abs_min": ABS_MIN, "n_boot": N_BOOT,
           "pass_bar": PASS_BAR, "seed": SEED, "n_folds": N_FOLDS if cross_fit else None,
           "shape_missing_frac": round(D.shape_missing, 5),
           "cuts": {n: D.cuts[n] for n in CANDIDATES}, "by_candidate": {}}
    print("\n=== FF shape-conditioned Location+ / pair %s  n=%d pitchers  (build %.1fs) ===" % (
        args.pair, len(js["CONTROL"]), time.time() - t1), flush=True)
    point = {}
    for n in CANDIDATES:
        j = js[n]
        rs, rp, rb, rsp = _stats(j)
        r3, r2, r_stuff, n3 = three_way(j)
        point[n] = (rb - rp, r3)
        print("  %-8s n=%d  location r %+.4f  prior r %+.4f  blend gain %+.4f  semipartial %+.4f"
              "  r(grade,Stuff+) %+.3f  3-way %+.4f (2-way %+.4f, n=%d)" % (
                  n, len(j), rs, rp, rb - rp, rsp, r_stuff, r3, r2, n3), flush=True)
        out["by_candidate"][n] = {
            "n": int(len(j)), "loc_r": round(rs, 4), "results_r": round(rp, 4),
            "blend_r": round(rb, 4), "blend_gain": round(rb - rp, 4),
            "semipartial": round(rsp, 4),
            "grade_prior_r": round(float(fc.R(j["grade"], j["prior"])), 4),
            "grade_stuff_r": round(r_stuff, 4), "three_way_r": round(r3, 4),
            "two_way_r": round(r2, 4), "n_three_way": int(n3)}
    if len(js["CONTROL"]) < MIN_PITCHERS:
        raise SystemExit("pool below MIN_PITCHERS; nothing to read")

    rng = np.random.default_rng(SEED)
    gains = {n: [] for n in CANDIDATES}
    d3 = {n: [] for n in TESTS}
    for bi in range(N_BOOT):
        cnt = np.bincount(rng.integers(0, len(tr_ids), len(tr_ids)),
                          minlength=len(tr_ids)).astype(float)
        keep = np.array(sorted(set(rng.choice(ev_ids, len(ev_ids)))))
        jb, _ = build(cnt, keep)
        if len(jb["CONTROL"]) < MIN_PITCHERS:
            continue
        for n in CANDIDATES:
            s = _stats(jb[n])
            gains[n].append(s[2] - s[1])
        r3c = three_way(jb["CONTROL"])[0]
        for n in TESTS:
            d3[n].append(three_way(jb[n])[0] - r3c)
        if (bi + 1) % 50 == 0:
            print("    refit %d/%d  (%.0fs)" % (bi + 1, N_BOOT, time.time() - t1), flush=True)

    g = {n: np.array(v) for n, v in gains.items()}
    for n in CANDIDATES:
        lo, hi = np.percentile(g[n], [2.5, 97.5])
        rec = out["by_candidate"][n]
        rec.update({"n_boot_used": int(len(g[n])), "gain_mean": round(float(g[n].mean()), 4),
                    "gain_ci": [round(float(lo), 4), round(float(hi), 4)],
                    "p_gain_positive": float((g[n] > 0).mean())})
        line = "  %-8s own gate: gain mean %+.4f CI [%+.4f,%+.4f] P(gain>0)=%.3f" % (
            n, g[n].mean(), lo, hi, rec["p_gain_positive"])
        if n in TESTS:
            d = g[n] - g["CONTROL"]
            dlo, dhi = np.percentile(d, [2.5, 97.5])
            e3 = np.array(d3[n])
            rec.update({"diff_vs_control_mean": round(float(d.mean()), 4),
                        "diff_vs_control_se": round(float(d.std(ddof=1)), 4),
                        "diff_vs_control_ci": [round(float(dlo), 4), round(float(dhi), 4)],
                        "p_diff_positive": float((d > 0).mean()),
                        "three_way_diff_mean": round(float(e3.mean()), 4),
                        "three_way_diff_se": round(float(e3.std(ddof=1)), 4),
                        "p_three_way_diff_positive": float((e3 > 0).mean())})
            line += ("\n  %-8s vs CONTROL: diff %+.4f (SE %.4f) CI [%+.4f,%+.4f] P(diff>0)=%.3f"
                     "   3-way diff %+.4f (SE %.4f) P=%.3f" % (
                         "", d.mean(), rec["diff_vs_control_se"], dlo, dhi,
                         rec["p_diff_positive"], e3.mean(), rec["three_way_diff_se"],
                         rec["p_three_way_diff_positive"]))
        print(line, flush=True)

    if cross_fit:
        rel = reliability(D, vals0, gpool_all.index)
        out["reliability"] = rel
        print("  reliability 2024->2025 grade (n=%d): %s" % (
            rel["n"], {n: rel[n] for n in CANDIDATES}), flush=True)

    dest = os.path.join(SCORE_WORKDIR, "coach_ff_location_shape_gate_%s.json" % args.pair)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\n  wrote %s   total %.0fs" % (os.path.basename(dest), time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
