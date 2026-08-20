"""Emit the Location+ map for the coach page's Location+ section.

WHY THIS EXISTS AS A SCRIPT: the JSON it writes was originally produced by an ad-hoc
heredoc, which meant section 3 of the page could not be rebuilt from the repo. It does not
depend on FEATS or ridge_pred -- Location+ is a pooled map over plate location only -- so a
Stuff+ retrain does NOT invalidate it, but it still needs to be reproducible.

WHAT IT EMITS
  cells   every 0.25 ft (gx, gz) cell of the fitted PooledLocationMap that cleared the
          50-pitch minimum, as run value per 100 four-seams, plus that cell's sample size.
          This IS the Location+ model: one input, so its "feature importance" is the map.
  bands   five distance bands from the strike zone, with league usage share and value, all
          batter-agnostic by construction (see ORIENTATION below for why that matters).
  by_hand the measured cost of pooling across batter handedness, carried so the page's
          docstring claim is backed by a number in the artifact rather than a memory.

ORIENTATION is established from data, not convention -- the project has shipped one
inverted interpretation before. Positive PlateLocSide is the RIGHT-handed batter's side,
evidenced by hit-by-pitch location (mean +1.73 ft for RHB, -1.79 ft for LHB over 25,157
HBP), and that fact is written into the payload so the renderer never has to re-derive it.
The value map itself is pooled across batter hand, so it cannot distinguish inside from
away; the bands are therefore defined on distance from the zone rather than on in/away
spots.

SIGN CONVENTION: xT is expected runs from the pitcher's perspective, LOWER = BETTER. Values
are emitted in that frame (multiplied by 100 for readability), so a NEGATIVE cell is good
for the pitcher. Nothing is negated here.

Data rules: reads workdir caches only; writes JSON to the workdir. No pitcher names.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import coach_model_ff_criterion as ffc
import fair_criterion as fc

ZX, ZZ0, ZZ1 = 0.83, 1.5, 3.5     # half-plate incl. ball radius; zone bottom/top, feet
STEP = 0.25                        # fine-bin size used by PooledLocationMap
RUNS_PER = 100
BANDS = ["Heart of the zone", "Zone edges", "Just off the plate", "Clearly off",
         "Nowhere near it"]
# Band cuts, in feet of distance OUTSIDE the zone. Fixed here rather than tuned.
CHASE, CLEAR = 0.35, 0.9
ORIENT = dict(pos_side="RHB",
              evidence="HBP mean PlateLocSide RHB +1.73 ft, LHB -1.79 ft (n=25,157)")


def band_of(df: pd.DataFrame) -> pd.Series:
    """Assign each pitch to one of five distance bands. Batter-agnostic by design."""
    ax = df["PlateLocSide"].abs()
    inzone = (ax <= ZX) & df["PlateLocHeight"].between(ZZ0, ZZ1)
    third = (ZZ1 - ZZ0) / 3
    heart = (ax <= ZX / 3) & df["PlateLocHeight"].between(ZZ0 + third, ZZ1 - third)
    dx = (ax - ZX).clip(lower=0)
    dz = np.maximum(ZZ0 - df["PlateLocHeight"], df["PlateLocHeight"] - ZZ1).clip(lower=0)
    out = np.sqrt(dx ** 2 + dz ** 2)
    b = pd.Series(index=df.index, dtype=object)
    b[inzone & heart] = BANDS[0]
    b[inzone & ~heart] = BANDS[1]
    b[~inzone & (out <= CHASE)] = BANDS[2]
    b[~inzone & (out > CHASE) & (out <= CLEAR)] = BANDS[3]
    b[~inzone & (out > CLEAR)] = BANDS[4]
    return b


def main() -> int:
    ff = ffc._frame(ffc.SCORE_WORKDIR, "2024,2025")
    ff = ff[ff["PlateLocSide"].notna() & ff["PlateLocHeight"].notna()]
    tr = ff[(ff["year"] == 2024) & ff["xT"].notna()].copy()
    print(f"  map train rows: {len(tr):,}")

    m = fc.PooledLocationMap(tr)
    print(f"  fine cells (>=50): {len(m.fine)}   coarse: {len(m.coarse)}   "
          f"fallback xT={m.fallback:+.5f}")

    counts = tr.groupby(["gx", "gz"]).size()
    cells = [dict(x=float(k[0]), z=float(k[1]), v=round(float(v) * RUNS_PER, 3),
                  n=int(counts.get(k, 0))) for k, v in m.fine.items()]

    # measured cost of pooling across batter handedness (a candidate improvement, gated
    # separately -- this only records the size of the effect)
    per_hand = {h: fc.PooledLocationMap(g) for h, g in tr.groupby("is_lhb")}
    a, b = per_hand[0].fine, per_hand[1].fine
    common = a.index.intersection(b.index)
    d = (a[common] - b[common]) * RUNS_PER
    by_hand = dict(cells_compared=int(len(common)),
                   corr=round(float(np.corrcoef(a[common], b[common])[0, 1]), 3),
                   mean_abs_diff=round(float(d.abs().mean()), 2),
                   p95_abs_diff=round(float(d.abs().quantile(0.95)), 2))
    print(f"  pooling cost across batter hand: corr={by_hand['corr']:+.3f}, "
          f"mean|diff|={by_hand['mean_abs_diff']:.2f} runs/100")

    tr["band"] = band_of(tr)
    g = tr.groupby("band").agg(n=("xT", "size"), v=("xT", "mean"))
    total = float(g["n"].sum())
    bands = [dict(band=b2, pct=round(100 * float(g.loc[b2, "n"]) / total, 1),
                  v100=round(float(g.loc[b2, "v"]) * RUNS_PER, 2),
                  n=int(g.loc[b2, "n"])) for b2 in BANDS]
    print(f"  {'band':<22}{'usage%':>8}{'runs/100':>11}")
    for r in bands:
        print(f"  {r['band']:<20}{r['pct']:>7.1f}%{r['v100']:>+10.2f}")

    pay = dict(cells=cells, bands=bands, step=STEP,
               zone=dict(x0=-ZX, x1=ZX, z0=ZZ0, z1=ZZ1),
               fallback=round(float(m.fallback) * RUNS_PER, 3),
               n_train=int(len(tr)), orient=ORIENT, by_hand=by_hand)
    dest = os.path.join(ffc.SCORE_WORKDIR, "coach_location_map.json")
    with open(dest, "w") as fh:
        json.dump(pay, fh, indent=1)
    print(f"\n  wrote {dest}  ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
