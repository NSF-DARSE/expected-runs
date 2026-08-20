"""What does the pitcher-handedness flag actually buy? (measurement, not a gate)

Asked 2026-08-17: how much would the page change if is_lhp were switched off. Building a
live toggle means running the whole chain twice and shipping both, which is hours. This
answers the underlying question directly instead: refit the SHIPPED feature set with and
without is_lhp and report the validity difference on the fixed next-season criterion.

NOT a gate, and deliberately so. No pre-registered decision rule is attached, because
is_lhp is not a construct problem the way is_lhb was. Batter handedness was opponent
context leaking into a pitch-quality score; PITCHER handedness is a physical attribute of
the pitcher being graded, so there is no principled reason to force it out. This just
prices it.

WHAT TO WATCH FOR IN THE RESULT: most of what is_lhp used to carry has already been moved
into the arm-side mirrors (HorzBreak_arm, RelSide_arm) and the deviation centres, all of
which fold handedness in directly. So a SMALL remaining cost means the flag is mostly
redundant with the mirrors and could be dropped cheaply. A LARGE one means it is carrying a
real residual lefty effect that the geometry does not explain -- worth its own
investigation rather than a toggle.

Criterion, pool, and bootstrap match coach_release_gate.py exactly so the numbers are
comparable to the adoption evidence.

Data rules: reads workdir caches only; writes JSON to the workdir. No pitcher names.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import coach_model_ff_criterion as ffc
import fair_criterion as fc

N_BOOT = 200
FLOOR = 100


def main() -> int:
    t0 = time.time()
    score = ffc._frame(ffc.SCORE_WORKDIR, "2024,2025")
    crit = ffc._frame(ffc.CRIT_WORKDIR, "2025,2026")
    print(f"  frames loaded in {time.time() - t0:.0f}s")

    tr = score[(score["year"] == 2024) & score["Target"].notna()].dropna(subset=fc.FEATS)
    ev = score[score["year"] == 2025].dropna(subset=fc.FEATS).copy()
    k = ev.groupby("PitcherId").size()
    ev = ev[ev["PitcherId"].isin(k[k >= FLOOR].index)]
    c = crit[crit["year"] == 2025].groupby("PitcherId").agg(
        n=("adjT", "size"), adjT=("adjT", "mean"), Target=("Target", "mean"))
    c = c[c["n"] >= FLOOR]

    VAR = {"shipped": list(fc.FEATS),
           "no_lhp": [f for f in fc.FEATS if f != "is_lhp"]}
    print(f"  shipped {len(VAR['shipped'])} features; without is_lhp {len(VAR['no_lhp'])}")

    def validity(cols, train_df, eval_df, crit_col, w=None):
        kw = {"ridge__sample_weight": w} if w is not None else {}
        m = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(
            train_df[cols].values, train_df["Target"].values, **kw)
        p = pd.Series(m.predict(eval_df[cols].values), index=eval_df["PitcherId"].values)
        p = p.groupby(level=0).mean()
        j = pd.DataFrame({"p": p}).join(c[crit_col], how="inner").dropna()
        return float(fc.R(j["p"], j[crit_col])), len(j)

    out = {"n_boot": N_BOOT, "floor": FLOOR, "by_criterion": {}}
    for crit_col in ("adjT", "Target"):
        pt = {k2: validity(v, tr, ev, crit_col) for k2, v in VAR.items()}
        print(f"\n=== criterion 2026 four-seam {crit_col} (n={pt['shipped'][1]}) ===")
        for k2 in VAR:
            print(f"    {k2:<10s} validity r = {pt[k2][0]:+.4f}")

        tr_codes, tr_ids = pd.factorize(tr["PitcherId"].values)
        ev_ids = ev["PitcherId"].unique()
        rng = np.random.default_rng(20260817)
        B = {k2: [] for k2 in VAR}
        for b in range(N_BOOT):
            cnt = np.bincount(rng.integers(0, len(tr_ids), len(tr_ids)),
                              minlength=len(tr_ids)).astype(float)
            w = cnt[tr_codes]
            rs_ev = ev[ev["PitcherId"].isin(set(rng.choice(ev_ids, len(ev_ids))))]
            for k2, cols in VAR.items():
                B[k2].append(validity(cols, tr, rs_ev, crit_col, w=w)[0])
            if (b + 1) % 100 == 0:
                print(f"      refit boot {b + 1}/{N_BOOT}")
        d = np.array(B["no_lhp"]) - np.array(B["shipped"])
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"    cost of removing is_lhp: mean={d.mean():+.4f} SE={d.std():.4f} "
              f"CI=[{lo:+.4f},{hi:+.4f}]  P(no_lhp is worse)={float((d < 0).mean()):.3f}")
        out["by_criterion"][crit_col] = dict(
            n=pt["shipped"][1], shipped=round(pt["shipped"][0], 4),
            no_lhp=round(pt["no_lhp"][0], 4), mean=round(float(d.mean()), 4),
            se=round(float(d.std()), 4), ci=[round(float(lo), 4), round(float(hi), 4)],
            p_worse=float((d < 0).mean()))

    dest = ffc.SCORE_WORKDIR + r"\coach_lhp_cost.json"
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n  wrote {dest}   total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
