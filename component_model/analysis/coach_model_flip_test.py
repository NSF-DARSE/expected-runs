"""Is the criterion-dependent sign flip real, or two noisy numbers either side of zero?

Observed (coach_model_ff_criterion.py): the coach-card-minus-our-Stuff+ spread gap is
POSITIVE on raw Target (his card ahead) and NEGATIVE on adjT (ours ahead), at both
sample floors. Those came from separate bootstraps, so neither the flip nor its
decomposition into luck vs opponent had an uncertainty attached.

This tests it properly: ONE resample at a time, all three criteria computed on it, so
the difference-of-differences shares noise and can be tested.

  gap(c)      = spread(coach's card | criterion c) - spread(our Stuff+ | criterion c)
  luck step   = gap(Target) - gap(xT)      effect of removing batted-ball luck
  opp step    = gap(xT) - gap(adjT)        effect of removing league/batter strength
  total       = gap(Target) - gap(adjT)

A positive step means that adjustment moves the comparison in OUR favour (his gap
shrinks). If `total` is not distinguishable from zero, the flip is noise and the honest
conclusion is that criterion choice does not reliably change the answer either.

SIGN CONVENTION: criteria are run values, LOWER = BETTER; spread is worst-third minus
best-third so a LARGER spread means better sorting. Scores are higher-is-better (`_hi`).

Data rules: reads workdir caches; prints only. Never committed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import coach_model_ff_criterion as ffc
import coach_model_paired as cp
import fair_criterion as fc

N_BOOT = 4000
VARIANTS = ["Target", "xT", "adjT"]
PAIR = ("coach_hi", "stuff_hi")


def gaps_for(s: pd.DataFrame) -> dict:
    """All three criterion gaps on one (re)sample. Terciles are recomputed inside the
    resample so bands stay resample-relative, matching the earlier runs."""
    for c in PAIR:
        s[c + "_t"] = pd.qcut(pd.Series(cp.z(s[c].values), index=s.index), 3,
                              labels=ffc.TERCILES)
    out = {}
    for v in VARIANTS:
        crit = f"crit100_{v}"
        out[v] = (ffc.spread(s, PAIR[0], crit) - ffc.spread(s, PAIR[1], crit))
    return out


def main() -> int:
    for floor in (51, 100):
        f = ffc.build(floor, floor)
        obs = gaps_for(f.copy())
        print(f"\n=== floor {floor}, n={len(f)} ===")
        print("  observed gap (coach - ours), positive = his card ahead:")
        for v in VARIANTS:
            print(f"    {v:<8}{obs[v]:+.3f}")

        rng = np.random.default_rng(97)
        idx = f.index.values
        draws = {v: [] for v in VARIANTS}
        for _ in range(N_BOOT):
            g = gaps_for(f.loc[rng.choice(idx, len(idx))])
            for v in VARIANTS:
                draws[v].append(g[v])
        d = {v: np.array(x) for v, x in draws.items()}

        print("  each gap, bootstrapped:")
        for v in VARIANTS:
            fc.boot_report(f"gap on {v}", d[v])
        print("  the flip, decomposed (positive = that adjustment favours OURS):")
        fc.boot_report("luck step   gap(Target) - gap(xT)", d["Target"] - d["xT"])
        fc.boot_report("opp step    gap(xT) - gap(adjT)", d["xT"] - d["adjT"])
        fc.boot_report("TOTAL       gap(Target) - gap(adjT)", d["Target"] - d["adjT"])

        tot = d["Target"] - d["adjT"]
        verdict = ("REAL: the criterion choice does change the answer"
                   if (tot > 0).mean() >= 0.95 or (tot > 0).mean() <= 0.05
                   else "NOT ESTABLISHED: the flip is inside noise")
        print(f"  => {verdict}")
        # And the question that actually matters for the meeting: does EITHER gap clear
        # the project's own bar on its own?
        clears = [v for v in VARIANTS
                  if (d[v] > 0).mean() >= 0.95 or (d[v] > 0).mean() <= 0.05]
        print(f"  gaps clearing P>=0.95 in either direction: {clears or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
