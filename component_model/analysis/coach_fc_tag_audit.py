"""Is the cutter criterion unreliable because of the pitch, or because of the tag?

Task zero of the sinker/cutter loop parked the cutter: year-over-year reliability of a
pitcher's cutter results is indistinguishable from zero (2025->2026 gate panel +0.127
[-0.04,+0.27]), so no feature work can be validated against it. That verdict assumed the
noise is outcome noise at a small sample. The literature pass of 2026-09-10 raised a second
mechanism: the cutter is the one tag that straddles two families (a hard cutter is tagged a
fastball, a soft one a slider, and the same pitcher's tag can flip between seasons). If the
tag churns, the pitcher-year "cutter" is not the same pitch twice, and its results cannot
repeat however real the skill.

This script separates the two with physics only, on aggregates only:

  placement    per pitcher-year, where the cutter centroid sits relative to the pitcher's
               OWN four-seam and slider centroids in (RelSpeed, IVB, HorzBreak_arm), in
               pooled within-pitcher-tag SD units. A cutter closer than 1 SD to either
               neighbour is "overlapping": the tag is doing work the physics does not.
  pitch-level  share of tagged cutters whose nearest of the pitcher's own centroids is not
               the cutter's (a mis-tag proxy that needs no external classifier).
  persistence  of pitchers with 15+ cutters in year 1 and a real year 2, how many still
               carry the tag; of those who do not, how many have a year-2 four-seam or
               slider centroid within 1 SD of the year-1 cutter ("renamed").
  reliability  year-over-year r of the cutter's PHYSICS (mean velo, IVB, run) beside the
               r of its RESULTS (mean adjT), on the gate panel. Reliable physics with
               unreliable results is a real pitch with noisy outcomes; unreliable physics
               says the tag is not naming one pitch. Results reliability is also split by
               placement (distinct vs overlapping) and recomputed with the cutter POOLED
               into its nearest neighbour tag within pitcher (the BP StuffPro bucket
               idea) against the neighbour alone as the control.

Both real pairs are run (2024->2025 score build, 2025->2026 criterion build). Pitcher
bootstrap CIs, 1000 reps. Writes one JSON to the score workdir; no names, no per-pitcher
rows, no absolute paths (fair_criterion.workdirs()).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

import fair_criterion as fc

MIN_TAG = 15
SHARE = 0.10
MIN_TOTAL_Y2 = 100
OVERLAP_SD = 1.0
N_BOOT = 1000
SEED = 20260910
PHYS = ["RelSpeed", "InducedVertBreak", "HorzBreak_arm"]
TAGS = {"FF": "FF", "SL": "SL", "FC": "FC"}

DATA, SCORE_WORKDIR, CRIT_WORKDIR = fc.workdirs()
DATA_CRIT = os.environ.get("STUFFPLUS_DATA_CRIT", DATA)


def _boot_r(a, b, rng):
    r = float(fc.R(a, b))
    idx = np.arange(len(a))
    rs = []
    for _ in range(N_BOOT):
        s = rng.integers(0, len(idx), len(idx))
        if np.std(a[s]) == 0 or np.std(b[s]) == 0:
            continue
        rs.append(fc.R(a[s], b[s]))
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return {"r": round(r, 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
            "p_positive": round(float((np.array(rs) > 0).mean()), 3), "n": int(len(a))}


def prep(df):
    df = df[fc.pitch_mask(df, "FF") | fc.pitch_mask(df, "SL") | fc.pitch_mask(df, "FC")].copy()
    df["HorzBreak_arm"] = df["HorzBreak"] * (1 - 2 * df["is_lhp"])
    df["tag"] = np.where(fc.pitch_mask(df, "FF"), "FF", np.where(fc.pitch_mask(df, "SL"), "SL", "FC"))
    return df.dropna(subset=PHYS + ["adjT"])


def centroids(df):
    """Per pitcher-year-tag: n, physics means, adjT mean. Pooled within-tag SD per feature."""
    g = df.groupby(["PitcherId", "year", "tag"])
    c = g.agg(n=("adjT", "size"), adjT=("adjT", "mean"),
              **{p: (p, "mean") for p in PHYS}).reset_index()
    # pooled within pitcher-year-tag SD, over tags with enough pitches to have a spread
    within = g[PHYS].std()
    nn = g.size()
    sd = within[nn >= MIN_TAG].mean()
    return c, sd


def placement(c, sd, year):
    """Cutter centroid distance to own FF and SL centroids, SD units; nearest neighbour."""
    y = c[(c["year"] == year) & (c["n"] >= MIN_TAG)]
    w = y.pivot(index="PitcherId", columns="tag", values=PHYS)
    fcs = w.xs("FC", axis=1, level=1).dropna()
    out = pd.DataFrame(index=fcs.index)
    for nb in ("FF", "SL"):
        nbc = w.xs(nb, axis=1, level=1).reindex(fcs.index)
        d = ((fcs - nbc) / sd) ** 2
        out[f"d_{nb}"] = np.sqrt(d.sum(axis=1)).where(nbc.notna().all(axis=1))
    out["d_min"] = out[["d_FF", "d_SL"]].min(axis=1)
    out["nearest"] = np.where(out["d_FF"].fillna(9e9) <= out["d_SL"].fillna(9e9), "FF", "SL")
    out.loc[out["d_min"].isna(), "nearest"] = "none"
    out["overlapping"] = out["d_min"] < OVERLAP_SD
    return out


def pitch_level_overlap(df, c, sd, year):
    """Share of tagged cutters nearer another of the pitcher's own centroids than the cutter's."""
    y = c[(c["year"] == year) & (c["n"] >= MIN_TAG)]
    p = df[(df["year"] == year) & (df["tag"] == "FC")]
    p = p[p["PitcherId"].isin(y.loc[y["tag"] == "FC", "PitcherId"])]
    best = None
    for tag in ("FF", "SL", "FC"):
        cen = y[y["tag"] == tag].set_index("PitcherId")[PHYS]
        m = p[["PitcherId"]].join(cen, on="PitcherId")[PHYS]
        d = np.sqrt((((p[PHYS].values - m.values) / sd.values) ** 2).sum(axis=1))
        d = np.where(np.isnan(d), np.inf, d)
        if best is None:
            best, best_tag = d, np.full(len(d), tag, dtype=object)
        else:
            better = d < best
            best = np.where(better, d, best)
            best_tag = np.where(better, tag, best_tag)
    return {"n_cutter_pitches": int(len(p)),
            "share_nearest_not_cutter": round(float((best_tag != "FC").mean()), 4),
            "share_nearest_ff": round(float((best_tag == "FF").mean()), 4),
            "share_nearest_sl": round(float((best_tag == "SL").mean()), 4)}


def run_pair(df, label, rng):
    df = prep(df)
    c, sd = centroids(df)
    tot = df.groupby(["PitcherId", "year"]).size()
    res = {"pair": label, "pooled_within_tag_sd": {p: round(float(sd[p]), 3) for p in PHYS}}

    # ---- placement and pitch-level overlap, per year ----
    pl = {}
    for year in (2024, 2025):
        p = placement(c, sd, year)
        pl[year] = p
        res[f"placement_y{year - 2023}"] = {
            "n_cutter_pitchers": int(len(p)),
            "with_ff_and_sl": int(p[["d_FF", "d_SL"]].notna().all(axis=1).sum()),
            "median_d_ff": round(float(p["d_FF"].median()), 3),
            "median_d_sl": round(float(p["d_SL"].median()), 3),
            "share_nearest_ff": round(float((p["nearest"] == "FF").mean()), 3),
            "share_nearest_sl": round(float((p["nearest"] == "SL").mean()), 3),
            "share_overlapping_lt_1sd": round(float(p["overlapping"].mean()), 3),
            "share_lt_1.5sd": round(float((p["d_min"] < 1.5).mean()), 3),
            **pitch_level_overlap(df, c, sd, year)}

    # ---- tag persistence year 1 -> year 2 ----
    y1 = c[(c["year"] == 2024) & (c["tag"] == "FC") & (c["n"] >= MIN_TAG)].set_index("PitcherId")
    tot2 = tot.xs(2025, level="year")
    active = y1.index.intersection(tot2[tot2 >= MIN_TOTAL_Y2].index)
    y2fc = c[(c["year"] == 2025) & (c["tag"] == "FC") & (c["n"] >= MIN_TAG)].set_index("PitcherId")
    kept = active.intersection(y2fc.index)
    lost = active.difference(y2fc.index)
    renamed = 0
    y2 = c[(c["year"] == 2025) & (c["n"] >= MIN_TAG)]
    for pid in lost:
        cen1 = y1.loc[pid, PHYS].values.astype(float)
        nb = y2[(y2["PitcherId"] == pid) & (y2["tag"].isin(["FF", "SL"]))]
        if len(nb) and (np.sqrt((((nb[PHYS].values - cen1) / sd.values) ** 2).sum(axis=1)) < OVERLAP_SD).any():
            renamed += 1
    res["persistence"] = {
        "y1_cutter_pitchers_active_y2": int(len(active)),
        "kept_tag": int(len(kept)), "share_kept": round(len(kept) / max(len(active), 1), 3),
        "lost_tag": int(len(lost)),
        "lost_but_renamed_within_1sd": renamed,
        "share_lost_renamed": round(renamed / max(len(lost), 1), 3)}

    # ---- reliability: physics vs results, gate panel ----
    def panel(tag):
        rows = c[(c["tag"] == tag) & (c["n"] >= MIN_TAG)].copy()
        rows["tot"] = rows.set_index(["PitcherId", "year"]).index.map(tot)
        rows = rows[rows["n"] / rows["tot"] >= SHARE]
        w = rows.pivot(index="PitcherId", columns="year", values=PHYS + ["adjT"]).dropna()
        return w

    w = panel("FC")
    rel = {"n_pairs": int(len(w))}
    for v in PHYS + ["adjT"]:
        rel[v] = _boot_r(w[(v, 2024)].values, w[(v, 2025)].values, rng)
    res["fc_reliability_gate_panel"] = rel

    # split by placement in BOTH years: distinct (>=1 SD from both neighbours) vs overlapping
    both = pl[2024][["overlapping"]].join(pl[2025][["overlapping"]], lsuffix="_1", rsuffix="_2", how="inner")
    distinct = both.index[~both["overlapping_1"] & ~both["overlapping_2"]]
    overl = both.index[both["overlapping_1"] | both["overlapping_2"]]
    for name, ids in (("distinct_both_years", distinct), ("overlapping_either_year", overl)):
        ww = w[w.index.isin(ids)]
        res[f"fc_adjT_reliability_{name}"] = (
            _boot_r(ww[("adjT", 2024)].values, ww[("adjT", 2025)].values, rng)
            if len(ww) >= 20 else {"n": int(len(ww)), "skipped": "fewer than 20 pairs"})

    # ---- BP-style pooling: cutter folded into its nearest own tag ----
    # For each pitcher with a cutter in both years and the SAME nearest neighbour both
    # years, pool cutter + neighbour pitches per year and take mean adjT; compare to the
    # neighbour alone (control) and the cutter alone on the same pitchers.
    near = pl[2024][["nearest"]].join(pl[2025][["nearest"]], lsuffix="_1", rsuffix="_2", how="inner")
    near = near[(near["nearest_1"] == near["nearest_2"]) & (near["nearest_1"] != "none")]
    pooled = {}
    for nb in ("FF", "SL"):
        ids = near.index[near["nearest_1"] == nb]
        sub = df[df["PitcherId"].isin(ids) & df["tag"].isin([nb, "FC"])]
        pool = sub.groupby(["PitcherId", "year"])["adjT"].mean().unstack().dropna()
        alone = sub[sub["tag"] == nb].groupby(["PitcherId", "year"])["adjT"].mean().unstack().dropna()
        fco = sub[sub["tag"] == "FC"].groupby(["PitcherId", "year"])["adjT"].mean().unstack().dropna()
        common = pool.index.intersection(alone.index).intersection(fco.index)
        if len(common) < 20:
            pooled[nb] = {"n": int(len(common)), "skipped": "fewer than 20 pairs"}
            continue
        pooled[nb] = {
            "n": int(len(common)),
            "cutter_alone": _boot_r(fco.loc[common, 2024].values, fco.loc[common, 2025].values, rng),
            "neighbour_alone": _boot_r(alone.loc[common, 2024].values, alone.loc[common, 2025].values, rng),
            "cutter_pooled_with_neighbour": _boot_r(pool.loc[common, 2024].values, pool.loc[common, 2025].values, rng)}
    res["pooled_into_nearest_tag"] = pooled
    return res


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    out = {"min_tag": MIN_TAG, "share": SHARE, "overlap_sd": OVERLAP_SD, "n_boot": N_BOOT,
           "phys": PHYS, "pairs": []}
    for data, wd, years, label in ((DATA, SCORE_WORKDIR, "2024,2025", "2024->2025"),
                                   (DATA_CRIT, CRIT_WORKDIR, "2025,2026", "2025->2026")):
        df = fc.load_frame(data, wd, years)
        print("  %s loaded in %.0fs" % (label, time.time() - t0), flush=True)
        r = run_pair(df, label, rng)
        out["pairs"].append(r)
        print(json.dumps(r, indent=1), flush=True)
    dest = os.path.join(SCORE_WORKDIR, "coach_fc_tag_audit.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print("  wrote %s   total %.0fs" % (dest, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
