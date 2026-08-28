"""Does Stuff+ pay off only when location is bad? And is the additive blend the problem?

MOTIVATION: on the sample-size curve, Location+ alone catches the equal-weight
Stuff+/Location+ blend by about 200 pitches, which reads as "Stuff+ adds nothing once you
have volume". The 3x3 grid says something different: within the BEST third of Location+
the Stuff+ rows are flat, while within the WORST third they span nearly a run. If that is
real, Stuff+ has conditional value that an ADDITIVE z-blend cannot express, and the flat
curve is a symptom of the combiner rather than of the measure.

TWO TESTS, both on the existing floor-100 fastball pool:

 1. INTERACTION. Stuff+'s tercile spread (worst third minus best third of Stuff+)
    computed separately inside each Location+ tercile, with a paired bootstrap on
    (spread when location is worst) - (spread when location is best). Positive and
    clearing the bar means Stuff+ pays off specifically when location is poor.

 2. THE COMBINER. Paired bootstrap of Location+ alone against the equal-weight blend,
    the comparison the curve script never actually ran. Reported at the pitcher level on
    the same criterion so it is directly comparable to the other panels.

WHY NOT JUST FIT WEIGHTS: this project has overfit fitted blend weights three separate
times (FRAMEWORK.md). Nothing here is fit. Test 1 asks whether an interaction EXISTS;
choosing a functional form is a separate decision that needs its own out-of-sample check.

SIGN CONVENTION: crit100_* are expected runs from the pitcher's perspective, LOWER =
BETTER. Scores are `_hi` (higher = better). A spread is worst-third minus best-third, so
POSITIVE means the score sorts correctly. Nothing is negated twice.

Data rules: reads workdir caches; prints and writes one JSON to the workdir. No names.
Never committed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import coach_model_ff_criterion as ffc
import coach_model_paired as cp
import coach_model_two_blend as tb
import fair_criterion as fc

N_BOOT = 3000
TER = ffc.TERCILES  # ascending on a higher-is-better score: worst, middle, best


def terciles(s: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        s[c + "_t"] = pd.qcut(pd.Series(cp.z(s[c].values), index=s.index), 3, labels=TER)
    return s


def stuff_spread_within(s: pd.DataFrame, loc_band: str, crit: str) -> float:
    """Stuff+'s worst-minus-best spread among pitchers in one Location+ band. Terciles of
    Stuff+ are recomputed WITHIN the band so 'best third of stuff' means best third of
    these pitchers, not of the whole pool."""
    g = s[s["loc_hi_t"] == loc_band]
    q = pd.qcut(pd.Series(cp.z(g["stuff_hi"].values), index=g.index), 3, labels=TER)
    return float(g.loc[q == "worst third", crit].mean()
                 - g.loc[q == "best third", crit].mean())


def spread(s: pd.DataFrame, col: str, crit: str) -> float:
    q = pd.qcut(pd.Series(cp.z(s[col].values), index=s.index), 3, labels=TER)
    return float(s.loc[q == "worst third", crit].mean()
                 - s.loc[q == "best third", crit].mean())


def main() -> int:
    base = tb.add_pitch2(ffc.build(100, 100))
    out = {"n": int(len(base)), "by_criterion": {}}

    for variant in ("Target", "adjT"):
        crit = f"crit100_{variant}"
        s = terciles(base.copy(), ["loc_hi"])
        print(f"\n=== criterion: 2026 four-seam {variant} (runs/100, lower = better), "
              f"n={len(s)} ===")

        print("  Stuff+ spread INSIDE each Location+ band "
              "(bigger = Stuff+ sorts these pitchers):")
        obs = {}
        for band in ["best third", "middle", "worst third"]:
            obs[band] = stuff_spread_within(s, band, crit)
            n_band = int((s["loc_hi_t"] == band).sum())
            print(f"    location {band:<12}{obs[band]:+.2f} runs/100   (n={n_band})")

        loc_alone = spread(s, "loc_hi", crit)
        blend = spread(s, "pitch2_hi", crit)
        print(f"  Location+ alone {loc_alone:+.2f}   equal-weight blend {blend:+.2f}")

        idx = s.index.values
        rng = np.random.default_rng(23)
        d_int, d_comb = [], []
        for _ in range(N_BOOT):
            r = s.loc[rng.choice(idx, len(idx))]
            r = terciles(r, ["loc_hi"])
            d_int.append(stuff_spread_within(r, "worst third", crit)
                         - stuff_spread_within(r, "best third", crit))
            d_comb.append(spread(r, "loc_hi", crit) - spread(r, "pitch2_hi", crit))
        d_int, d_comb = np.array(d_int), np.array(d_comb)

        print("  paired bootstrap:")
        fc.boot_report("INTERACTION  stuff spread when location worst - when best", d_int)
        fc.boot_report("COMBINER     Location+ alone - equal-weight blend", d_comb)

        out["by_criterion"][variant] = dict(
            stuff_spread_by_location_band={k: round(v, 3) for k, v in obs.items()},
            loc_alone=round(loc_alone, 3), blend=round(blend, 3),
            interaction=dict(mean=round(float(d_int.mean()), 3),
                             se=round(float(d_int.std()), 3),
                             p_gt0=round(float((d_int > 0).mean()), 3)),
            combiner=dict(mean=round(float(d_comb.mean()), 3),
                          se=round(float(d_comb.std()), 3),
                          p_gt0=round(float((d_comb > 0).mean()), 3)))

    dest = os.path.join(ffc.SCORE_WORKDIR, "coach_interaction.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
