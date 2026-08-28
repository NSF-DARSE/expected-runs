"""What each model leans on: features ranked by importance, v1 card vs v2 ridge.

Companion to the "where the two grades already agree" panel. The scatter shows THAT they
agree; this shows WHICH measurements each one is built on, and where the feature sets
differ at all.

IMPORTANCE IS WITHIN A MODEL, NOT EVIDENCE EITHER IS RIGHT. A feature can dominate a model
and be weighted backwards -- v1's HorzBreak term is exactly that case (its isolated
contribution correlates +0.056 with next season's run value, i.e. the wrong way). Read
this as a description of each model, never as an endorsement.

TWO NATIVE SCALES, PLUS ONE COMPARABLE ONE:
  v2 ridge  native = |standardised coefficient|, read off the fitted pipeline (the
            StandardScaler means coef_ is ALREADY per-SD; do not multiply by SD again).
  v1 card   native = SD across pitchers of that term's isolated contribution to his score.
  both      comparable = |corr| between the term's isolated contribution and the model's
            OWN total score, at pitcher level. Unit-free, so the two rank orders can sit
            side by side.

DIRECTION is reported in plain words. Both models' contributions are put in
HIGHER = BETTER FOR THE PITCHER orientation first: coach_score already returns
higher-is-better, while ridge_pred is expected run value (LOWER = better), so the ridge
side is negated ONCE. A positive contribution slope then means "this model treats more of
this as better for the pitcher" in both columns.

Data rules: reads workdir caches; writes one JSON to the workdir. No pitcher names.
Never committed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import coach_model_comparison as cm
import coach_model_ff_criterion as ffc
import fair_criterion as fc

FLOOR = 100
PRETTY = {
    # DELIBERATELY NOT merged into one "Velocity" row (2026-08-17). These are different
    # constructs, and the difference is the point: effective velo is computed FROM velo and
    # extension, so v1 carries extension bundled invisibly inside its velocity term, while
    # v2 splits the two into separate inputs. Showing one shared "Velocity" row hid that.
    # Splitting them does NOT make v2 more accurate (the two grades still tie); it makes the
    # grade diagnosable -- "your extension is doing the work" is a different coaching cue
    # from "throw harder". Contrast with HorzBreak/RelSide below, which DO merge: those are
    # one physical metric in two reference frames, not two constructs.
    "EffectiveVelo": "Effective velocity", "RelSpeed": "Velocity",
    "HorzBreak_arm": "Horizontal break", "RelSide_arm": "Release side",  # arm-side = same metric
    "InducedVertBreak": "Vertical break",
    "HorzBreak": "Horizontal break", "SpinRate": "Spin rate", "Extension": "Extension",
    "RelHeight": "Release height", "RelSide": "Release side",
    "velocity_differential": "Velo vs his own fastball",
    "vertbreakdiff": "Vert break vs his fastball",
    "horzbreakdiff": "Horz break vs his fastball",
    "is_lhp": "Pitcher throws left", "is_lhb": "Batter hits left",
    # The coach's construct, adopted into v2 on 2026-08-17. Labelled to match how his card
    # reads it -- distance from a typical release point, either direction -- and kept as
    # rows of their own rather than merged with "Release side"/"Release height", because our
    # monotone terms are still in the model alongside these and mean something different.
    "dev_relheight": "Release height vs typical",
    "dev_relside": "Release side vs typical",
}


def main() -> int:
    terms = cm.load_coach_terms("FourSeamFastBall")
    v1_cols = sorted({t["col"] for t in terms})

    ff = ffc._frame(ffc.SCORE_WORKDIR, "2024,2025")
    ff = ff.dropna(subset=sorted(set(v1_cols) | set(fc.FEATS))).copy()
    ff["v1_total"] = cm.coach_score(ff, terms, 1.0)          # higher = better already

    # refit the reference ridge here so its fitted pipeline is in hand for coefficients
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    train = ff[(ff["year"] == 2024) & ff["Target"].notna()]
    model = make_pipeline(StandardScaler(), Ridge(alpha=10))
    model.fit(train[fc.FEATS].values, train["Target"].values)
    coefs = dict(zip(fc.FEATS, model[-1].coef_))
    scaler = model[0]

    season = ff[ff["year"] == 2025].copy()
    keep = season.groupby("PitcherId").size()
    season = season[season["PitcherId"].isin(keep[keep >= FLOOR].index)]
    print(f"  pool: {season['PitcherId'].nunique()} pitchers, {len(season):,} four-seams")

    # pitcher-level totals
    pid = season["PitcherId"]
    v1_tot = season.groupby("PitcherId")["v1_total"].mean()
    # v2 total in higher-is-better orientation
    v2_tot = -season.groupby("PitcherId")["ridge_pred"].mean()

    rows = []
    # ---- v2: each feature's standardised contribution, negated into higher = better
    means, sds = scaler.mean_, scaler.scale_
    for i, f in enumerate(fc.FEATS):
        contrib = -coefs[f] * (season[f].values - means[i]) / sds[i]
        per_p = pd.Series(contrib, index=pid.values).groupby(level=0).mean()
        rows.append(dict(
            model="v2", feature=f, label=PRETTY.get(f, f),
            native=abs(float(coefs[f])), native_kind="|standardised coefficient|",
            r_with_own_score=round(float(fc.R(per_p, v2_tot.reindex(per_p.index))), 4),
            better_when=("more" if -coefs[f] > 0 else "less")))

    # ---- v1: each term column's isolated contribution (already higher = better)
    for col in v1_cols:
        sub = [t for t in terms if t["col"] == col]
        contrib = cm.coach_score(season, sub, 1.0)
        per_p = pd.Series(np.asarray(contrib), index=pid.values).groupby(level=0).mean()
        slope = float(np.polyfit(season[col].values, np.asarray(contrib), 1)[0])
        # DIRECTION IS NOT DEFINED for his "off average" terms, and reporting one is wrong.
        # A term with relative=False scores |value - hand average| (coach_score's else
        # branch), so its contribution is a symmetric V centred on the hand's typical
        # value: both a very high and a very low release point earn points, and the
        # minimum sits in the middle. Verified on 2025 four-seams -- contribution
        # correlates +1.000 with |RelSide - hand avg|, and the within-hand linear slopes
        # come out OPPOSITE (RHP +0.80, LHP -0.72) precisely because the V is centred at
        # +1.56 for righties and -1.62 for lefties.
        #
        # Fitting one line through that V gives a pooled slope of -0.02 (pooled r = -0.016,
        # i.e. noise), and the sign of that noise was setting a confident down-arrow on the
        # page for both Release side and Release height. Emit "magnitude" so the renderer
        # can show that direction does not apply instead of inventing one. The |r| itself
        # is still meaningful: it is how much this term drives his grade.
        kind = next((t for t in sub if not t["relative"]), None)
        label = PRETTY.get(col, col)
        if kind is not None:
            # His off-average terms score |value - typical for that hand|, which is the SAME
            # CONSTRUCT as our adopted dev_relheight / dev_relside -- his centres are his own
            # per-hand averages, ours are train-year means, and the two differ only in that
            # constant. So label them onto the "vs typical" row where they belong and where
            # they merge with ours, instead of onto the monotone raw row, which measures
            # something genuinely different (feet toward the arm side, not unusualness).
            #
            # Direction IS well defined once measured on the right scale. Earlier this was
            # emitted as "magnitude" (a plus-minus) because the slope was fitted against the
            # RAW column, where a symmetric V has no direction and the pooled slope was
            # noise (r = -0.016). Refitting against the DEVIATION recovers a real sign: more
            # unusual earns more points under the deviation-is-good reading. Derived, not
            # asserted, so if the card is ever rescored under the CONFORM reading
            # (absolute_sign = -1) this flips on its own.
            label = f"{label} vs typical"
            centre = {t["hand"]: t["avg"] for t in sub}
            dev = season[col].sub(season["PitcherThrows"].map(centre)).abs()
            slope = float(np.polyfit(dev.values, np.asarray(contrib), 1)[0])
        rows.append(dict(
            model="v1", feature=col, label=label,
            native=float(per_p.std()), native_kind="SD across pitchers of contribution",
            r_with_own_score=round(float(fc.R(per_p, v1_tot.reindex(per_p.index))), 4),
            better_when=("more" if slope > 0 else "less")))

    both = set(v1_cols) & set(fc.FEATS)
    for r in rows:
        r["shared"] = ("both models" if r["feature"] in both
                       else "only v1" if r["feature"] in v1_cols else "only v2")

    for m in ("v1", "v2"):
        sel = sorted([r for r in rows if r["model"] == m],
                     key=lambda r: -abs(r["r_with_own_score"]))
        print(f"\n  {'v1 Stuff+ (coach card)' if m == 'v1' else 'v2 Stuff+ (ridge)'} "
              f"— ranked by |corr with its own score|")
        print(f"    {'feature':<28}{'|r|':>7}{'better when':>13}   in")
        for r in sel:
            print(f"    {r['label']:<28}{abs(r['r_with_own_score']):>7.3f}"
                  f"{r['better_when']:>13}   {r['shared']}")

    print(f"\n  shared features: {sorted(PRETTY.get(c, c) for c in both)}")
    print(f"  only v1: {sorted(PRETTY.get(c, c) for c in set(v1_cols) - both)}")
    print(f"  only v2: {sorted(PRETTY.get(c, c) for c in set(fc.FEATS) - both)}")

    dest = os.path.join(ffc.SCORE_WORKDIR, "coach_feature_importance.json")
    with open(dest, "w") as fh:
        json.dump(dict(n_pitchers=int(season["PitcherId"].nunique()), floor=FLOOR,
                       rows=rows, shared=sorted(both),
                       only_v1=sorted(set(v1_cols) - both),
                       only_v2=sorted(set(fc.FEATS) - both)), fh, indent=1)
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
