"""Per-type Location+ through the UNCHANGED incremental gate. Pre-registration and verdicts:
docs/notes/secondary-research-ledger-2026-09.md. Read that before running anything here.

WHY. Location+ has been a four-seam-only score since the 2024->2025 run found secondary
Location+ reliable but not valid (slider 0.020, changeup 0.013). The 2025->2026 replication
found those validities "no longer ~0" (slider 0.058, changeup 0.150, curveball 0.245, all
levels, 100+ panel), and no secondary or cutter Location+ has ever been put through the
frozen gate. The cutter's Stuff+ is withheld with the working conclusion that cutter skill is
a command question. This harness asks, with the gate statistic unchanged, whether a per-type
Location+ adds to a pitcher's own results.

THE GATE IS coach_incremental_gate.py's, copied not re-derived: 50/50 z-blend of grade and
prior results, P(blend gain > 0) over 200 cluster-bootstrap refits that resample TRAIN
pitchers (frequency weights on the map fit) and CRITERION pitchers separately, pool = 15+
pitches and 10%+ of the pitcher's pitches in both the grade and criterion seasons, bar 0.95,
seed 20260817, criterion = next-season mean adjT on the type. The only thing that changes is
the grade: a mean of per-pitch location values instead of a mean of ridge predictions.

PAIRS
  C  (confirmation) the gate's own configuration: map trained on 2024 (score build), grade on
     2025 pitches, prior = 2025 adjT (score build), criterion = 2026 adjT (criterion build,
     stored under the relabeled year 2025).
  D  (discovery) 2024->2025 inside the score build: grade on 2024 pitches, prior = 2024 adjT,
     criterion = 2025 adjT. There is no 2023 to train on, so the map is trained on 2024
     CROSS-FITTED: pitchers are split into 5 fixed folds and each pitcher's pitches are
     valued by a map fitted without him. The bootstrap weights apply inside every fold map.

CANDIDATES (exact definitions in the ledger)
  CONTROL  the shipped four-seam Location+: catcher-frame pooled map on xT. FF only.
  H1       per-type map, batter_platoon frame, count-relative (location_maps.CellMap,
           count=True, m tuned per type on a train-year holdout of xT, then fixed).
  H2       H1 grade shrunk toward the population mean by empirical Bayes.
  H3       H1 grade shrunk toward a prior predicted from the pitcher's location on his OTHER
           pitch types (standardised H1 values), by empirical Bayes.
  AUDIT_O  decomposition audit for H3: the other-pitch location score alone as the grade.

SIGN CONVENTION: xT, the location values, adjT and the prior are all expected run value from
the pitcher's perspective, LOWER = BETTER. A correctly oriented grade correlates POSITIVELY
with the criterion. Nothing is negated.

Data rules: reads the existing D1 pitch caches only and refuses to run if a cache is missing
or would be rebuilt; writes one JSON of aggregates to the score workdir. No pitcher names,
no per-pitcher output, no absolute paths (fair_criterion.workdirs()). Usage, from
component_model/analysis with STUFFPLUS_DATA, STUFFPLUS_DATA_CRIT, STUFFPLUS_WORKDIR and
STUFFPLUS_WORKDIR_CRIT set:
    python coach_location_gate.py --pair D --candidate H1
    python coach_location_gate.py --pair C --candidate CONTROL --dry   # no criterion touched
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import fair_criterion as fc
import location_maps as lm

# ---- frozen gate constants (coach_incremental_gate.py) ----
SHARE = 0.10
ABS_MIN = 15
MIN_PITCHERS = 60
N_BOOT = 200
PASS_BAR = 0.95
SEED = 20260817

# ---- pre-registered design constants (ledger) ----
N_FOLDS = 5
FOLD_SEED = 20260924
M_GRID = (1, 2, 5, 10, 25, 100)
ALL_GROUPS = ["FF", "SI", "FC", "SL", "CB", "CH"]
TARGETS = ["SI", "FC", "SL", "CB", "CH"]
CANDIDATES = {
    "CONTROL": {"frame": "catcher", "count": False, "shrink": None, "groups": ["FF"]},
    "H1": {"frame": "batter_platoon", "count": True, "shrink": None, "groups": TARGETS},
    "H2": {"frame": "batter_platoon", "count": True, "shrink": "population", "groups": TARGETS},
    "H3": {"frame": "batter_platoon", "count": True, "shrink": "other_pitch", "groups": TARGETS},
    "AUDIT_O": {"frame": "batter_platoon", "count": True, "shrink": "other_only",
                "groups": TARGETS},
}
NEEDS_ALL = {"other_pitch", "other_only"}

DATA, SCORE_WORKDIR, CRIT_WORKDIR = fc.workdirs()
DATA_CRIT = os.environ.get("STUFFPLUS_DATA_CRIT", DATA)


def _refuse_rebuild(data, workdir, years):
    """Exit unless the cache exists and load_pitches would serve it as-is."""
    tag = "" if years == "2024,2025" else "_" + years.replace(",", "_")
    cache = os.path.join(workdir, f"pitches_cache{tag}_D1.parquet")
    if not os.path.exists(cache):
        sys.exit(f"refusing to run: no cache at the score/criterion workdir for {years}; "
                 "this harness never builds one")
    import pyarrow.parquet as pq
    have = set(pq.read_schema(cache).names)
    header = pd.read_csv(data, nrows=0).columns
    want = fc.USECOLS + [c for c in fc.OPTIONAL_COLS if c in header] + fc.ANCHOR_COLS
    stale = [c for c in want if c not in have]
    if stale:
        sys.exit(f"refusing to run: the {years} cache would be REBUILT from the CSV "
                 f"(missing {stale}); check STUFFPLUS_DATA / STUFFPLUS_DATA_CRIT")


def _z(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd else s * 0.0


def _stats(j):
    """(grade r, prior r, blend r, semipartial), identical to coach_incremental_gate._stats."""
    rs = float(fc.R(j["grade"], j["crit"]))
    rp = float(fc.R(j["prior"], j["crit"]))
    rb = float(fc.R(_z(j["grade"]) + _z(j["prior"]), j["crit"]))
    b = np.polyfit(j["prior"].values, j["grade"].values, 1)
    resid = j["grade"].values - np.polyval(b, j["prior"].values)
    rsp = float(fc.R(pd.Series(resid, index=j.index), j["crit"]))
    return rs, rp, rb, rsp


class GroupData:
    """One pitch group's rows for one pair: codes, xT, pitchers, train/eval masks, folds."""

    def __init__(self, frame_df, grp, spec, train_year, eval_year, folds_of):
        m = fc.pitch_mask(frame_df, grp) & frame_df["year"].isin([train_year, eval_year])
        rows = frame_df[m]
        # Validity is ALWAYS judged in the batter_platoon frame so every candidate on a pair
        # is scored on the same rows and the same pool, whatever frame its map uses.
        ok = lm.frame_columns(rows, "batter_platoon")["ok"]
        rows = rows[ok & rows["Target"].notna()]
        fr = lm.frame_columns(rows, spec["frame"])
        self.grp = grp
        self.codes = lm.cell_codes(fr)
        self.xt = rows["xT"].values.astype(float)
        self.pid = rows["PitcherId"].values
        self.train = (rows["year"].values == train_year) & np.isfinite(self.xt)
        self.eval = rows["year"].values == eval_year
        self.cross_fit = train_year == eval_year
        self.fold = np.array([folds_of.get(p, 0) for p in self.pid]) if self.cross_fit else None
        self.eval_df = rows.loc[self.eval, ["PitcherId", "adjT"]].reset_index(drop=True)
        self.eval_xzp = fr.loc[self.eval, ["x", "z", "p"]].reset_index(drop=True)
        self.count = spec["count"]
        self.m = None
        if self.count:
            t = np.where(self.train)[0]
            self.m, self.m_errs = lm.tune_m(lm.subset_codes(self.codes, t), self.xt[t],
                                            self.pid[t], M_GRID)

    def eval_values(self, w_row):
        """Relative location value for every eval row, maps fitted with row weights w_row
        (weights on train rows; ignored elsewhere)."""
        ev = np.where(self.eval)[0]
        out = np.empty(len(ev))
        if not self.cross_fit:
            t = np.where(self.train)[0]
            mp = lm.CellMap(lm.subset_codes(self.codes, t), self.xt[t], w_row[t],
                            count=self.count, m=self.m or 5)
            return mp.relative(lm.subset_codes(self.codes, ev))
        for f in range(N_FOLDS):
            t = np.where(self.train & (self.fold != f))[0]
            e = self.fold[ev] == f
            mp = lm.CellMap(lm.subset_codes(self.codes, t), self.xt[t], w_row[t],
                            count=self.count, m=self.m or 5)
            out[e] = mp.relative(lm.subset_codes(self.codes, ev[e]))
        return out


def grade_table(target, values, eval_pids, shrink, others=None):
    """Pitcher-level grade for the target group from per-row eval values.

    values/eval_pids are the target group's eval rows (already restricted to kept pitchers).
    others: {group: (values, pids)} for the other groups' eval rows, used by H3 / AUDIT_O.
    Returns a DataFrame indexed by PitcherId with 'grade' (lower = better) and diagnostics.
    """
    s2, _, tab = lm.eb_moments(values, eval_pids)
    tab = tab[tab["n"] >= ABS_MIN].copy()
    if shrink is None:
        tab["grade"] = tab["mean"]
        return tab, {}
    mu = float(np.mean(values))
    if shrink == "population":
        tau2 = lm.tau2_between(tab["mean"].values, tab["n"].values, s2)
        tab["grade"] = lm.eb_shrink(tab["mean"], tab["n"], np.full(len(tab), mu), s2, tau2)
        return tab, {"k": s2 / tau2 if tau2 > 0 else None}
    # Other-pitch command score: each other pitch's value standardised by its own group's
    # pitch-level SD, averaged over the pitcher's other pitches.
    zs, ps = [], []
    for g, (v, p) in others.items():
        if g == target:
            continue
        sd = float(np.std(v))
        zs.append(v / sd if sd > 0 else v * 0.0)
        ps.append(p)
    o = pd.DataFrame({"z": np.concatenate(zs), "p": np.concatenate(ps)}).groupby("p")["z"]
    O = o.mean().reindex(tab.index)
    nO = o.size().reindex(tab.index).fillna(0)
    has_o = (nO >= ABS_MIN).values
    if shrink == "other_only":
        tab = tab[has_o].copy()
        tab["grade"] = O[has_o].values
        return tab, {"n_with_other": int(has_o.sum())}
    prior, b = lm.hier_prior(tab["mean"].values, O.fillna(0).values, has_o, mu)
    tau2_o = lm.tau2_between(tab["mean"].values[has_o], tab["n"].values[has_o], s2,
                             prior[has_o])
    tau2_p = lm.tau2_between(tab["mean"].values, tab["n"].values, s2)
    grade = np.empty(len(tab))
    grade[has_o] = lm.eb_shrink(tab["mean"].values[has_o], tab["n"].values[has_o],
                                prior[has_o], s2, tau2_o)
    grade[~has_o] = lm.eb_shrink(tab["mean"].values[~has_o], tab["n"].values[~has_o],
                                 prior[~has_o], s2, tau2_p)
    tab["grade"] = grade
    return tab, {"b": b, "n_with_other": int(has_o.sum()),
                 "k_other": s2 / tau2_o if tau2_o > 0 else None}


def orientation(G):
    """Mean relative location value by plate region, per platoon: a sign check on the map
    that uses only the map's own xT training, never the criterion."""
    v = G.eval_values(np.ones(len(G.pid)))
    d = G.eval_xzp.assign(v=v)
    x, zz = d["x"], d["z"]
    regions = {"heart": (x.abs() < 0.5) & (zz > 1.9) & (zz < 3.1),
               "low (below 1.5 ft)": zz < 1.5,
               "far low (below 1.0 ft)": zz < 1.0,
               "away edge (x < -0.9)": x < -0.9,
               "inside edge (x > +0.9)": x > 0.9,
               "high (above 3.5 ft)": zz > 3.5}
    lines = []
    for p in sorted(d["p"].unique()):
        lab = {0: "opposite-side batter / catcher frame", 1: "same-side batter"}[int(p)]
        parts = []
        for name, msk in regions.items():
            sel = msk & (d["p"] == p)
            if sel.sum() >= 200:
                parts.append("%s %+.2f (%.0f%%)" % (name, 100 * d.loc[sel, "v"].mean(),
                                                     100 * sel.sum() / (d["p"] == p).sum()))
        lines.append(lab + ": " + "; ".join(parts))
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", choices=["C", "D"], required=True)
    ap.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    ap.add_argument("--dry", action="store_true",
                    help="build maps and pools, print design diagnostics, touch no criterion "
                         "correlation and write nothing")
    args = ap.parse_args()
    spec = CANDIDATES[args.candidate]
    t0 = time.time()

    _refuse_rebuild(DATA, SCORE_WORKDIR, "2024,2025")
    score = fc.load_frame(DATA, SCORE_WORKDIR, "2024,2025")
    if args.pair == "C":
        _refuse_rebuild(DATA_CRIT, CRIT_WORKDIR, "2025,2026")
        crit = fc.load_frame(DATA_CRIT, CRIT_WORKDIR, "2025,2026")
        train_year, eval_year = 2024, 2025
        crit_frame, crit_year = crit, 2025
    else:
        train_year, eval_year = 2024, 2024
        crit_frame, crit_year = score, 2025
    print("  frames loaded in %.0fs" % (time.time() - t0), flush=True)
    tot = score[score["year"] == eval_year].groupby("PitcherId").size().rename("tot")
    ctot = crit_frame[crit_frame["year"] == crit_year].groupby("PitcherId").size().rename("ctot")

    # Fixed pitcher folds for cross-fitting (pair D only); independent of every outcome.
    all_p = np.sort(score["PitcherId"].dropna().unique())
    perm = np.random.default_rng(FOLD_SEED).permutation(len(all_p))
    folds_of = dict(zip(all_p, perm % N_FOLDS))

    need = ALL_GROUPS if spec["shrink"] in NEEDS_ALL else spec["groups"]
    gd = {g: GroupData(score, g, spec, train_year, eval_year, folds_of) for g in need}
    for g in need:
        print("  %s: train rows %d, eval rows %d, m=%s" % (
            g, gd[g].train.sum(), gd[g].eval.sum(), gd[g].m), flush=True)

    out = {"pair": args.pair, "candidate": args.candidate, "spec": spec,
           "share": SHARE, "abs_min": ABS_MIN, "n_boot": N_BOOT, "pass_bar": PASS_BAR,
           "seed": SEED, "n_folds": N_FOLDS if args.pair == "D" else None, "by_pitch": {}}

    for grp in spec["groups"]:
        G = gd[grp]
        c = crit_frame[fc.pitch_mask(crit_frame, grp) & (crit_frame["year"] == crit_year)]
        k = c.groupby("PitcherId").agg(cn=("adjT", "size"), crit=("adjT", "mean")).join(ctot)
        k = k[(k["cn"] >= ABS_MIN) & (k["cn"] / k["ctot"] >= SHARE)]

        # Train-pitcher universe for the bootstrap: the target group's train pitchers, or
        # every group's when the grade borrows from other pitch types.
        uni = [grp] if spec["shrink"] not in NEEDS_ALL else need
        tr_pid = np.concatenate([gd[g].pid[gd[g].train] for g in uni])
        tr_codes, tr_ids = pd.factorize(tr_pid)
        idx_of = {p: i for i, p in enumerate(tr_ids)}
        row_idx = {g: np.array([idx_of.get(p, -1) for p in gd[g].pid]) for g in need}
        ev_ids = G.eval_df["PitcherId"].unique()

        def build(cnt=None, keep=None):
            vals = {}
            for g in (need if spec["shrink"] in NEEDS_ALL else [grp]):
                if cnt is None:
                    w = np.ones(len(gd[g].pid))
                else:
                    ri = row_idx[g]
                    w = np.where(ri >= 0, cnt[np.maximum(ri, 0)], 0.0)
                v = gd[g].eval_values(w)
                p = gd[g].pid[gd[g].eval]
                if keep is not None:
                    sel = np.isin(p, keep)
                    v, p = v[sel], p[sel]
                vals[g] = (v, p)
            v, p = vals[grp]
            tab, diag = grade_table(grp, v, p, spec["shrink"], vals)
            ed = G.eval_df if keep is None else G.eval_df[G.eval_df["PitcherId"].isin(keep)]
            gpool = ed.groupby("PitcherId").agg(sn=("adjT", "size"),
                                                prior=("adjT", "mean")).join(tot)
            gpool = gpool[(gpool["sn"] >= ABS_MIN) & (gpool["sn"] / gpool["tot"] >= SHARE)]
            j = tab[["grade"]].join(gpool, how="inner").join(k, how="inner").dropna(
                subset=["grade", "prior", "crit"])
            return j, diag

        t1 = time.time()
        j, diag = build()
        print("")
        print("=== %s / %s / pair %s  n=%d pitchers  (build %.1fs) ===" % (
            args.candidate, grp, args.pair, len(j), time.time() - t1), flush=True)
        if args.dry:
            print("    [dry] diagnostics: %s" % {kk: (round(vv, 4) if isinstance(vv, float)
                                                      else vv) for kk, vv in diag.items()})
            print("    [dry] map orientation, mean location value x100 on eval rows "
                  "(lower = better for the pitcher):")
            for line in orientation(G):
                print("      " + line)
            continue
        if len(j) < MIN_PITCHERS:
            out["by_pitch"][grp] = {"n": int(len(j)), "skipped": "pool too small"}
            continue
        rs, rp, rb, rsp = _stats(j)
        print("    location r     %+.4f" % rs)
        print("    prior-results  %+.4f" % rp)
        print("    blend 50/50    %+.4f   gain over results %+.4f" % (rb, rb - rp))
        print("    semipartial    %+.4f" % rsp, flush=True)

        rng = np.random.default_rng(SEED)
        gains, sps = [], []
        for bi in range(N_BOOT):
            cnt = np.bincount(rng.integers(0, len(tr_ids), len(tr_ids)),
                              minlength=len(tr_ids)).astype(float)
            keep = np.array(sorted(set(rng.choice(ev_ids, len(ev_ids)))))
            jb, _ = build(cnt, keep)
            if len(jb) < MIN_PITCHERS:
                continue
            s2_ = _stats(jb)
            gains.append(s2_[2] - s2_[1])
            sps.append(s2_[3])
            if (bi + 1) % 50 == 0:
                print("      refit %d/%d  (%.0fs)" % (bi + 1, N_BOOT, time.time() - t1),
                      flush=True)
        gains, sps = np.array(gains), np.array(sps)
        p_gain = float((gains > 0).mean())
        lo, hi = np.percentile(gains, [2.5, 97.5])
        verdict = "PASS" if p_gain >= PASS_BAR else "no"
        print("    blend gain over %d refits: mean %+.4f  CI [%+.4f,%+.4f]  "
              "P(gain>0)=%.3f  -> %s" % (len(gains), gains.mean(), lo, hi, p_gain, verdict),
              flush=True)
        out["by_pitch"][grp] = {
            "n": int(len(j)), "m": G.m, "loc_r": round(rs, 4), "results_r": round(rp, 4),
            "blend_r": round(rb, 4), "blend_gain": round(rb - rp, 4),
            "semipartial": round(rsp, 4), "grade_prior_r": round(float(fc.R(j["grade"],
                                                                          j["prior"])), 4),
            "n_boot_used": int(len(gains)), "gain_mean": round(float(gains.mean()), 4),
            "gain_ci": [round(float(lo), 4), round(float(hi), 4)],
            "p_gain_positive": p_gain,
            "p_semipartial_positive": float((sps > 0).mean()),
            "verdict": verdict,
            "diag": {kk: (round(vv, 5) if isinstance(vv, float) else vv)
                     for kk, vv in diag.items()}}

    if args.dry:
        print("\n  dry run: nothing written   total %.0fs" % (time.time() - t0))
        return 0
    dest = os.path.join(SCORE_WORKDIR, "coach_location_gate_%s_%s.json" % (
        args.pair, args.candidate))
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("")
    print("  wrote %s   total %.0fs" % (os.path.basename(dest), time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
