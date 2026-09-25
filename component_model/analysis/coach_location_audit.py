"""Decomposition audit (read 9 of docs/notes/secondary-research-ledger-2026-09.md): does a
discovered per-type Location+ still add once the type's SHIPPED Stuff+ is in the blend?

The gate (coach_location_gate.py) asks whether Location+ adds to a pitcher's own results.
FRAMEWORK.md rule 2 asks the next question before anything is adopted: does the gain survive
in the combined score, next to the component a coach would already see? For a type that
already has a Stuff+, that is

    gain = r(z(prior) + z(stuff) + z(loc), crit) - r(z(prior) + z(stuff), crit)

with equal z weights (the project's blend rule), on the Location+ gate pool, pair C only.
The bootstrap has the gate's structure: one draw of TRAIN-pitcher frequency weights refits
every location map AND the Stuff+ ridge, and one draw of criterion pitchers re-scores. The
train-pitcher universe is the union of the location and ridge train pitchers, so the random
stream differs from the gate read; this is an audit and carries no bar. P(gain > 0) is
reported for information.

The Stuff+ grade is the shipped one (fair_criterion.ridge_for_group semantics): the pooled
four-seam+sinker ridge on SI_POOLED_TRAIN_FEATS for the sinker, the type's own
FEATS_BY_PITCH ridge otherwise, alpha RIDGE_ALPHA, trained on 2024.

SIGN CONVENTION: every grade, the prior and the criterion are expected run value from the
pitcher's perspective, LOWER = BETTER. All correlations are read in that frame; nothing is
negated.

Data rules: reads the existing D1 caches only (refuses a rebuild); writes one JSON of
aggregates to the score workdir. No pitcher names, per-pitcher output, or absolute paths.
Usage: python coach_location_audit.py --candidate H3 --types SI,SL
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import coach_location_gate as lg
import fair_criterion as fc


def ridge_rows(score, grp):
    """(train frame, eval frame, feats) for the type's shipped Stuff+ ridge."""
    if grp in fc.POOLED_GROUPS:
        feats = list(fc.SI_POOLED_TRAIN_FEATS)
        base = fc.add_derived_feats(score[score["is_ff"] | fc.pitch_mask(score, grp)].copy())
        base = base.dropna(subset=feats + ["Target"])
        ev = base[(base["year"] == 2025) & (base["is_si"] == 1)]
    else:
        feats = fc.feats_for(grp)
        base = fc.add_derived_feats(score[fc.pitch_mask(score, grp)].copy())
        base = base.dropna(subset=feats + ["Target"])
        ev = base[base["year"] == 2025]
    tr = base[base["year"] == 2024]
    return tr, ev, feats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=["H1", "H3"], required=True)
    ap.add_argument("--types", required=True)
    args = ap.parse_args()
    spec = lg.CANDIDATES[args.candidate]
    types = args.types.split(",")
    t0 = time.time()

    lg._refuse_rebuild(lg.DATA, lg.SCORE_WORKDIR, "2024,2025")
    lg._refuse_rebuild(lg.DATA_CRIT, lg.CRIT_WORKDIR, "2025,2026")
    score = fc.load_frame(lg.DATA, lg.SCORE_WORKDIR, "2024,2025")
    crit = fc.load_frame(lg.DATA_CRIT, lg.CRIT_WORKDIR, "2025,2026")
    tot = score[score["year"] == 2025].groupby("PitcherId").size().rename("tot")
    ctot = crit[crit["year"] == 2025].groupby("PitcherId").size().rename("ctot")
    need = lg.ALL_GROUPS if spec["shrink"] in lg.NEEDS_ALL else types
    gd = {g: lg.GroupData(score, g, spec, 2024, 2025, {}) for g in need}
    print("  frames and maps ready in %.0fs" % (time.time() - t0), flush=True)

    out = {"candidate": args.candidate, "pair": "C", "n_boot": lg.N_BOOT, "seed": lg.SEED,
           "note": "audit only, no bar", "by_pitch": {}}
    for grp in types:
        G = gd[grp]
        c = crit[fc.pitch_mask(crit, grp) & (crit["year"] == 2025)]
        k = c.groupby("PitcherId").agg(cn=("adjT", "size"), crit=("adjT", "mean")).join(ctot)
        k = k[(k["cn"] >= lg.ABS_MIN) & (k["cn"] / k["ctot"] >= lg.SHARE)]
        rtr, rev, feats = ridge_rows(score, grp)
        Xtr, ytr = rtr[feats].values, rtr["Target"].values
        Xev, ev_pid = rev[feats].values, rev["PitcherId"].values

        uni = [grp] if spec["shrink"] not in lg.NEEDS_ALL else need
        tr_pid = np.concatenate([gd[g].pid[gd[g].train] for g in uni] + [rtr["PitcherId"].values])
        tr_codes, tr_ids = pd.factorize(tr_pid)
        idx_of = {p: i for i, p in enumerate(tr_ids)}
        row_idx = {g: np.array([idx_of.get(p, -1) for p in gd[g].pid]) for g in need}
        ridge_idx = np.array([idx_of[p] for p in rtr["PitcherId"].values])
        ev_ids = G.eval_df["PitcherId"].unique()

        def build(cnt=None, keep=None):
            vals = {}
            for g in (need if spec["shrink"] in lg.NEEDS_ALL else [grp]):
                ri = row_idx[g]
                w = (np.ones(len(ri)) if cnt is None
                     else np.where(ri >= 0, cnt[np.maximum(ri, 0)], 0.0))
                v = gd[g].eval_values(w)
                p = gd[g].pid[gd[g].eval]
                if keep is not None:
                    sel = np.isin(p, keep)
                    v, p = v[sel], p[sel]
                vals[g] = (v, p)
            tab, _ = lg.grade_table(grp, vals[grp][0], vals[grp][1], spec["shrink"], vals)
            m = make_pipeline(StandardScaler(), Ridge(alpha=fc.RIDGE_ALPHA))
            kw = {} if cnt is None else {"ridge__sample_weight": cnt[ridge_idx]}
            m.fit(Xtr, ytr, **kw)
            st = pd.Series(m.predict(Xev), index=ev_pid).groupby(level=0).mean().rename("stuff")
            ed = G.eval_df if keep is None else G.eval_df[G.eval_df["PitcherId"].isin(keep)]
            gpool = ed.groupby("PitcherId").agg(sn=("adjT", "size"),
                                                prior=("adjT", "mean")).join(tot)
            gpool = gpool[(gpool["sn"] >= lg.ABS_MIN) & (gpool["sn"] / gpool["tot"] >= lg.SHARE)]
            j = (tab[["grade"]].join(st, how="inner").join(gpool, how="inner")
                 .join(k, how="inner").dropna(subset=["grade", "stuff", "prior", "crit"]))
            return j

        def stats(j):
            r2 = float(fc.R(lg._z(j["prior"]) + lg._z(j["stuff"]), j["crit"]))
            r3 = float(fc.R(lg._z(j["prior"]) + lg._z(j["stuff"]) + lg._z(j["grade"]),
                            j["crit"]))
            return r2, r3

        j = build()
        r2, r3 = stats(j)
        rls = float(fc.R(j["grade"], j["stuff"]))
        print("\n=== audit %s / %s  n=%d ===" % (args.candidate, grp, len(j)))
        print("    r(loc, stuff) %+.4f   r(loc, prior) %+.4f   r(stuff, crit) %+.4f"
              % (rls, fc.R(j["grade"], j["prior"]), fc.R(j["stuff"], j["crit"])))
        print("    results+stuff %+.4f   +loc %+.4f   gain %+.4f" % (r2, r3, r3 - r2), flush=True)
        rng = np.random.default_rng(lg.SEED)
        gains = []
        for bi in range(lg.N_BOOT):
            cnt = np.bincount(rng.integers(0, len(tr_ids), len(tr_ids)),
                              minlength=len(tr_ids)).astype(float)
            keep = np.array(sorted(set(rng.choice(ev_ids, len(ev_ids)))))
            jb = build(cnt, keep)
            if len(jb) < lg.MIN_PITCHERS:
                continue
            a, b = stats(jb)
            gains.append(b - a)
            if (bi + 1) % 50 == 0:
                print("      refit %d/%d  (%.0fs)" % (bi + 1, lg.N_BOOT, time.time() - t0),
                      flush=True)
        gains = np.array(gains)
        lo, hi = np.percentile(gains, [2.5, 97.5])
        pg = float((gains > 0).mean())
        print("    3-way gain over %d refits: mean %+.4f  CI [%+.4f,%+.4f]  P(gain>0)=%.3f"
              % (len(gains), gains.mean(), lo, hi, pg), flush=True)
        out["by_pitch"][grp] = {
            "n": int(len(j)), "r_loc_stuff": round(rls, 4),
            "r_loc_prior": round(float(fc.R(j["grade"], j["prior"])), 4),
            "r_stuff_crit": round(float(fc.R(j["stuff"], j["crit"])), 4),
            "r_results_stuff": round(r2, 4), "r_results_stuff_loc": round(r3, 4),
            "gain": round(r3 - r2, 4), "n_boot_used": int(len(gains)),
            "gain_mean": round(float(gains.mean()), 4),
            "gain_ci": [round(float(lo), 4), round(float(hi), 4)], "p_gain_positive": pg}

    dest = os.path.join(lg.SCORE_WORKDIR, "coach_location_audit_C_%s.json" % args.candidate)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print("\n  wrote %s   total %.0fs" % (os.path.basename(dest), time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
