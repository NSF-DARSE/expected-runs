"""Change what the Stuff+ ridge learns FROM, on the unchanged gate.

Every candidate so far (August loop, 2026-09-10 pooled run) varied the feature list of a
ridge that regresses physics directly on per-pitch run value (Target). That label is the
noisiest thing in the extract: a pitch's run value carries count, sequencing, defence and
ball-in-play luck, and physics explains ~1% of it. Public models that work do not learn
from it directly. This harness holds the features fixed at each type's best-known list and
varies the learning target and the learner:

  ridge_target   the shipping method (harness check).
  ridge_xt       same ridge, label = xT: balls in play carry the pooled EV/LA map value
                 instead of what happened after contact. Same information the criterion
                 uses; removes hit-luck from the label only.
  decomp_ridge   decomposed outcomes. Physics predicts the probability of each pitch
                 outcome class (ball, called strike, whiff, foul, in play) with a linear
                 probability ridge per class, and expected run value on contact (xT) with a
                 ridge on in-play rows. Recombined as sum_c P(c) * E[Target | c] for the
                 non-contact classes plus P(in play) * xT_hat. Each label is far cleaner
                 than run value; the grade is still one physics-only expected run value.
  decomp_gbm     the same decomposition with histogram gradient boosting (depth-limited,
                 fixed iterations, no early stopping so refits are deterministic given the
                 bootstrap weights). What Driveline / PitchingBot / tjStuff+ do.
  gbm_target     gradient boosting straight on Target, to separate the learner effect from
                 the target effect.

Groups: FF is the positive control (own model, BASE_FEATS) and must not lose ground; SI is
graded on the pooled four-seam+sinker model with every term of `pooled_all` from
coach_si_pooled_gate.py (Jack lifted the differential exclusion for the sinker on
2026-09-10). Other groups run on their FEATS_BY_PITCH list.

Same statistic, panel, criterion, bootstrap and seed as coach_incremental_gate.py: 50/50
z-blend of grade and prior results, P(blend gain > 0) over 200 cluster-bootstrap refits
resampling train pitchers and criterion pitchers separately, 10%-share / 15-pitch pool,
bar 0.95. Nothing about the gate moves; only the learner does.

Data rules: reads workdir caches only; writes one JSON to the score workdir. No pitcher
names, no per-pitcher output, no absolute paths -- see fair_criterion.workdirs().

Usage (from component_model/analysis, with the four STUFFPLUS_* variables set):
    python coach_target_gate.py --groups SI,FF --candidates ridge_target,ridge_xt,decomp_ridge
    python coach_target_gate.py --groups SI --candidates decomp_gbm,gbm_target --n-boot 200
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import fair_criterion as fc
from coach_si_pooled_gate import add_derived, CANDIDATES as POOLED

SHARE = 0.10
ABS_MIN = 15
MIN_PITCHERS = 60
PASS_BAR = 0.95
SEED = 20260817

DATA, SCORE_WORKDIR, CRIT_WORKDIR = fc.workdirs()
DATA_CRIT = os.environ.get("STUFFPLUS_DATA_CRIT", DATA)

# Outcome classes. HBP and intentional balls are folded into "ball": rare, and their run
# value is a consequence of location, not shape. Misspelled TrackMan tags are mapped too.
CLASS_OF = {
    "BallCalled": "ball", "BallinDirt": "ball", "BallIntentional": "ball", "HitByPitch": "ball",
    "StrikeCalled": "called", "StrikeSwinging": "whiff",
    "FoulBall": "foul", "FoulBallFieldable": "foul", "FoulBallNotFieldable": "foul",
    "FouldBallNotFieldable": "foul",
    "InPlay": "inplay",
}
CLASSES = ["ball", "called", "whiff", "foul", "inplay"]

GBM_KW = dict(max_depth=4, max_iter=150, learning_rate=0.08, min_samples_leaf=200,
              l2_regularization=1.0, early_stopping=False, random_state=0)


# ---------------- learners: fit(train, w) -> predict(eval) ----------------

def _ridge(X, y, w):
    m = make_pipeline(StandardScaler(), Ridge(alpha=fc.RIDGE_ALPHA))
    m.fit(X, y, ridge__sample_weight=w)
    return m


class RidgeTarget:
    label = "Target"

    def fit(self, tr, feats, w):
        self.m = _ridge(tr[feats].values, tr[self.label].values, w)
        return self

    def predict(self, ev, feats):
        return self.m.predict(ev[feats].values)


class RidgeXT(RidgeTarget):
    label = "xT"


class GBMTarget:
    def fit(self, tr, feats, w):
        self.m = HistGradientBoostingRegressor(**GBM_KW)
        self.m.fit(tr[feats].values, tr["Target"].values, sample_weight=w)
        return self

    def predict(self, ev, feats):
        return self.m.predict(ev[feats].values)


class Decomp:
    """P(class | physics) x E[Target | class] + P(in play) x E[xT | physics, in play]."""

    def __init__(self, gbm=False):
        self.gbm = gbm

    def fit(self, tr, feats, w):
        X = tr[feats].values
        cls = tr["cls"].values
        # class run values from the (weighted) training rows, non-contact classes only
        self.rv = {c: float(np.average(tr.loc[tr["cls"] == c, "Target"].values,
                                       weights=w[cls == c]))
                   for c in CLASSES if c != "inplay"}
        ip = cls == "inplay"
        if self.gbm:
            self.clf = HistGradientBoostingClassifier(**GBM_KW)
            self.clf.fit(X, cls, sample_weight=w)
            self.reg = HistGradientBoostingRegressor(**GBM_KW)
            self.reg.fit(X[ip], tr.loc[ip, "xT"].values, sample_weight=w[ip])
        else:
            Y = np.column_stack([(cls == c).astype(float) for c in CLASSES])
            self.clf = _ridge(X, Y, w)
            self.reg = _ridge(X[ip], tr.loc[ip, "xT"].values, w[ip])
        return self

    def predict(self, ev, feats):
        X = ev[feats].values
        if self.gbm:
            P = self.clf.predict_proba(X)
            order = list(self.clf.classes_)
        else:
            P = np.clip(self.clf.predict(X), 0, None)
            P = P / P.sum(axis=1, keepdims=True)
            order = CLASSES
        out = np.zeros(len(X))
        for j, c in enumerate(order):
            if c == "inplay":
                out += P[:, j] * self.reg.predict(X)
            else:
                out += P[:, j] * self.rv[c]
        return out


LEARNERS = {
    "ridge_target": lambda: RidgeTarget(),
    "ridge_xt": lambda: RidgeXT(),
    "decomp_ridge": lambda: Decomp(gbm=False),
    "decomp_gbm": lambda: Decomp(gbm=True),
    "gbm_target": lambda: GBMTarget(),
}


# ---------------- gate statistic (unchanged) ----------------

def _z(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd else s * 0.0


def _stats(j):
    rs = float(fc.R(j["grade"], j["crit"]))
    rp = float(fc.R(j["prior"], j["crit"]))
    rb = float(fc.R(_z(j["grade"]) + _z(j["prior"]), j["crit"]))
    b = np.polyfit(j["prior"].values, j["grade"].values, 1)
    resid = j["grade"].values - np.polyval(b, j["prior"].values)
    rsp = float(fc.R(pd.Series(resid, index=j.index), j["crit"]))
    return rs, rp, rb, rsp


def frame_for(score, grp):
    """(rows, feats, train_groups) for one graded group."""
    if grp == "SI":
        groups, feats = POOLED["pooled_all"]
        fam = score[fc.pitch_mask(score, "FF") | fc.pitch_mask(score, "SI")].copy()
        fam = add_derived(fam)
        return fam, feats, groups
    rows = score[fc.pitch_mask(score, grp)].copy()
    rows["RelSide_arm"] = rows["RelSide"] * (1 - 2 * rows["is_lhp"])
    rows["HorzBreak_arm"] = rows["HorzBreak"] * (1 - 2 * rows["is_lhp"])
    for o, s_ in fc.DEV_SRC.items():
        rows[o] = (rows[s_] - rows["is_lhp"].map(fc.DEV_CENTRES[o])).abs()
    rows["is_si"] = 0.0
    return rows, fc.feats_for(grp), [grp]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="SI,FF")
    ap.add_argument("--candidates", default="ridge_target,ridge_xt,decomp_ridge")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--out", default="coach_target_gate.json")
    args, _ = ap.parse_known_args()
    groups = args.groups.split(",")
    cands = args.candidates.split(",")

    t0 = time.time()
    score = fc.load_frame(DATA, SCORE_WORKDIR, "2024,2025")
    crit = fc.load_frame(DATA_CRIT, CRIT_WORKDIR, "2025,2026")
    print("  frames loaded in %.0fs" % (time.time() - t0), flush=True)
    score["cls"] = score["PitchCall"].map(CLASS_OF)
    tot = score[score["year"] == 2025].groupby("PitcherId").size().rename("tot")
    ctot = crit[crit["year"] == 2025].groupby("PitcherId").size().rename("ctot")

    dest = os.path.join(SCORE_WORKDIR, args.out)
    out = {"share": SHARE, "n_boot": args.n_boot, "pass_bar": PASS_BAR, "seed": SEED,
           "gbm": GBM_KW, "classes": CLASSES, "by_group": {}}
    if os.path.exists(dest):
        out = json.load(open(dest))
        out["n_boot"] = args.n_boot

    for grp in groups:
        rows, feats, train_groups = frame_for(score, grp)
        rows = rows.dropna(subset=feats + ["Target", "xT", "cls"])
        tr = rows[rows["year"] == 2024]
        gm = fc.pitch_mask(rows, grp)
        ev = rows[(rows["year"] == 2025) & gm]
        c = crit[fc.pitch_mask(crit, grp) & (crit["year"] == 2025)]
        k = c.groupby("PitcherId").agg(cn=("adjT", "size"), crit=("adjT", "mean")).join(ctot)
        k = k[(k["cn"] >= ABS_MIN) & (k["cn"] / k["ctot"] >= SHARE)]
        gout = out["by_group"].setdefault(grp, {"feats": feats, "train_groups": train_groups,
                                                "train_rows": int(len(tr))})
        print("\n##### %s  train=%s rows=%d  feats=%d" % (grp, "+".join(train_groups),
                                                          len(tr), len(feats)), flush=True)
        tr_codes, tr_ids = pd.factorize(tr["PitcherId"].values)
        ev_ids = ev["PitcherId"].unique()

        for name in cands:
            def build(train_df, eval_df, w):
                m = LEARNERS[name]().fit(train_df, feats, w)
                p = pd.Series(m.predict(eval_df, feats), index=eval_df["PitcherId"].values)
                s = pd.DataFrame({"grade": p.groupby(level=0).mean()})
                g = eval_df.groupby("PitcherId").agg(sn=("adjT", "size"),
                                                     prior=("adjT", "mean")).join(tot)
                g = g[(g["sn"] >= ABS_MIN) & (g["sn"] / g["tot"] >= SHARE)]
                return s.join(g, how="inner").join(k, how="inner").dropna(
                    subset=["grade", "prior", "crit"])

            t1 = time.time()
            j = build(tr, ev, np.ones(len(tr)))
            print("\n=== %s / %s  n=%d pitchers  (fit %.0fs) ===" % (grp, name, len(j),
                                                                    time.time() - t1), flush=True)
            if len(j) < MIN_PITCHERS:
                gout[name] = {"n": int(len(j)), "skipped": "pool too small"}
                continue
            rs, rp, rb, rsp = _stats(j)
            print("    stuff r        %+.4f" % rs)
            print("    prior-results  %+.4f" % rp)
            print("    blend 50/50    %+.4f   gain over results %+.4f" % (rb, rb - rp))
            print("    semipartial    %+.4f" % rsp, flush=True)
            rec = {"n": int(len(j)), "stuff_r": round(rs, 4), "results_r": round(rp, 4),
                   "blend_r": round(rb, 4), "blend_gain": round(rb - rp, 4),
                   "semipartial": round(rsp, 4)}
            if args.n_boot > 0:
                rng = np.random.default_rng(SEED)
                gains, sps = [], []
                for bi in range(args.n_boot):
                    cnt = np.bincount(rng.integers(0, len(tr_ids), len(tr_ids)),
                                      minlength=len(tr_ids)).astype(float)
                    keep = set(rng.choice(ev_ids, len(ev_ids)))
                    jb = build(tr, ev[ev["PitcherId"].isin(keep)], cnt[tr_codes])
                    if len(jb) < MIN_PITCHERS:
                        continue
                    s2 = _stats(jb)
                    gains.append(s2[2] - s2[1])
                    sps.append(s2[3])
                    if bi % 25 == 24:
                        print("      refit %d/%d  P(gain>0) so far %.3f  (%.0fs)" % (
                            bi + 1, args.n_boot, (np.array(gains) > 0).mean(),
                            time.time() - t1), flush=True)
                gains, sps = np.array(gains), np.array(sps)
                p_gain = float((gains > 0).mean())
                lo, hi = np.percentile(gains, [2.5, 97.5])
                verdict = "PASS" if p_gain >= PASS_BAR else "no"
                print("    blend gain over %d refits: mean %+.4f  CI [%+.4f,%+.4f]  "
                      "P(gain>0)=%.3f  -> %s" % (len(gains), gains.mean(), lo, hi, p_gain,
                                                 verdict), flush=True)
                rec.update({"n_boot_used": int(len(gains)),
                            "gain_mean": round(float(gains.mean()), 4),
                            "gain_ci": [round(float(lo), 4), round(float(hi), 4)],
                            "p_gain_positive": p_gain,
                            "p_semipartial_positive": float((sps > 0).mean()),
                            "verdict": verdict})
            gout[name] = rec
            with open(dest, "w") as fh:
                json.dump(out, fh, indent=1)

    print("\n  wrote %s   total %.0fs" % (dest, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
