"""Candidate SI feature sets against the UNCHANGED shipping gate statistic.

Task zero (coach_crit_reliability_panel.py) established that the sinker criterion is real on
the pair the gate scores against (2025->2026 reliability +0.241 [+0.13,+0.35]), so the SI
failure is a weak grade (stuff_r +0.082 against a ~0.43 ceiling), not a dead criterion. This
harness evaluates candidate SI feature sets on EXACTLY the statistic the gate pre-registered:
equal-weight 50/50 z-blend of grade and prior results, P(blend gain > 0) over 200 cluster-
bootstrap refits resampling train pitchers and criterion pitchers separately, on the
10%-share / 15-pitch pool. Nothing about the bar, panel, criterion or bootstrap moves; only
the ridge's feature list varies. "base" replicates coach_incremental_gate.py's SI row as the
harness check.

Candidate features must be PHYSICS. Anything that needs the pitch's actual location, the
batter, or the pitcher's usage is out of scope for a Stuff+ (one construct per score).

  vaa_flat   approach-angle steepness a sinker would show at a FIXED reference plate height
             (2.0 ft, where sinkers live), from constant-acceleration kinematics on
             RelSpeed, RelHeight, Extension and InducedVertBreak. The fixed reference is
             the point: it is height-adjusted VAA with the location dependence removed by
             construction, so it stays a pure physical composite. Queued from the coach
             meeting (height-adjusted VAA); the raw TrackMan VertApprAngle is not in the
             extract, and would leak location if it were used raw.

Data rules: reads workdir caches only; writes one JSON to the score workdir. No pitcher
names, no per-pitcher output, no absolute paths -- see fair_criterion.workdirs().
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import fair_criterion as fc

GRP = "SI"
SHARE = 0.10
ABS_MIN = 15
MIN_PITCHERS = 60
N_BOOT = 200
PASS_BAR = 0.95

DATA, SCORE_WORKDIR, CRIT_WORKDIR = fc.workdirs()

G_FTPS2 = 32.174
MPH_TO_FTPS = 5280.0 / 3600.0
REF_PLATE_HEIGHT = 2.0   # ft; bottom-of-zone reference where a sinker earns its living
PLATE_SPEED_FRAC = 0.916  # plate speed ~ 91.6% of release speed (drag), standard figure


def add_vaa_flat(df):
    """Approach angle (deg, negative = downward) at the fixed reference height.

    Constant-acceleration kinematics release -> plate: flight distance 60.5ft - Extension,
    time from the average of release and plate speed, vertical acceleration = gravity plus
    the Magnus lift implied by InducedVertBreak (inches over the full flight). Only physics
    columns enter; the pitch's actual PlateLoc* never does.
    """
    v0 = df["RelSpeed"] * MPH_TO_FTPS
    dist = 60.5 - df["Extension"]
    t = dist / (v0 * (1 + PLATE_SPEED_FRAC) / 2.0)
    a = -G_FTPS2 + 2.0 * (df["InducedVertBreak"] / 12.0) / (t ** 2)
    vz0 = (REF_PLATE_HEIGHT - df["RelHeight"] - 0.5 * a * t ** 2) / t
    vzf = vz0 + a * t
    vyf = v0 * PLATE_SPEED_FRAC
    df["vaa_flat"] = np.degrees(np.arctan2(vzf, vyf))
    return df


def add_movement_geometry(df):
    """Break-vector geometry and its deviation from the release slot. Physics only.

    mov_angle   direction of the movement vector in the arm-side frame, degrees from
                pure ride (0 = straight IVB, 90 = pure arm-side run). A sinker's identity
                is WHERE the break points, which the linear IVB + HB terms only encode as
                two independent magnitudes.
    mov_mag     total break, inches.
    slot_dev    movement direction minus release-slot direction (atan2 of RelSide_arm over
                RelHeight, from vertical). Spin axis roughly follows the arm slot, so
                movement that points away from where the slot says it should is the crude
                seam/gyro deviation we CAN compute; the true spin-based SSW residual is not
                computable from this extract (TrackMan SpinAxis is itself movement-inferred).
    """
    mov_angle = np.degrees(np.arctan2(df["HorzBreak_arm"], df["InducedVertBreak"]))
    slot_angle = np.degrees(np.arctan2(df["RelSide_arm"], df["RelHeight"]))
    df["mov_angle"] = mov_angle
    df["mov_mag"] = np.hypot(df["HorzBreak_arm"], df["InducedVertBreak"])
    df["slot_dev"] = mov_angle - slot_angle
    return df


def add_nonlinear_terms(df):
    """Iteration 3: nonlinearity on top of movement geometry, still physics only.

    velo_x_ivb     the "heavy sinker" interaction: hard AND low-ride is a different pitch
                   from either alone, and a linear model credits the two independently.
    velo_x_movang  does break direction matter more (or less) at velocity.
    mov_angle_sq   a break-direction sweet spot: value need not be monotone in direction.
    """
    df["velo_x_ivb"] = df["RelSpeed"] * df["InducedVertBreak"]
    df["velo_x_movang"] = df["RelSpeed"] * df["mov_angle"]
    df["mov_angle_sq"] = df["mov_angle"] ** 2
    # Iteration 4: the full movement-plane quadratic. One hypothesis, stated once: run
    # value is a smooth NON-MONOTONE function of the break vector, and the angle terms
    # above are only one slice of that surface.
    df["ivb_sq"] = df["InducedVertBreak"] ** 2
    df["hb_sq"] = df["HorzBreak_arm"] ** 2
    df["ivb_x_hb"] = df["InducedVertBreak"] * df["HorzBreak_arm"]
    return df


CANDIDATES = {
    "base": [],
    "base+movgeo": ["mov_angle", "mov_mag"],
    "base+movgeo+veloX": ["mov_angle", "mov_mag", "velo_x_ivb", "velo_x_movang"],
    "base+movgeo+angsq": ["mov_angle", "mov_mag", "mov_angle_sq"],
    "base+movgeo+all3": ["mov_angle", "mov_mag", "velo_x_ivb", "velo_x_movang",
                         "mov_angle_sq"],
    "base+movquad": ["ivb_sq", "hb_sq", "ivb_x_hb"],
    "base+movgeo+movquad": ["mov_angle", "mov_mag", "mov_angle_sq",
                            "ivb_sq", "hb_sq", "ivb_x_hb"],
}


def _z(s):
    sd = s.std()
    return (s - s.mean()) / sd if sd else s * 0.0


def _stats(j):
    rs = float(fc.R(j["grade"], j["crit"]))
    rp = float(fc.R(j["prior"], j["crit"]))
    blend = _z(j["grade"]) + _z(j["prior"])
    rb = float(fc.R(blend, j["crit"]))
    b = np.polyfit(j["prior"].values, j["grade"].values, 1)
    resid = j["grade"].values - np.polyval(b, j["prior"].values)
    rsp = float(fc.R(pd.Series(resid, index=j.index), j["crit"]))
    return rs, rp, rb, rsp


def main() -> int:
    t0 = time.time()
    score = fc.load_frame(DATA, SCORE_WORKDIR, "2024,2025")
    crit = fc.load_frame(DATA, CRIT_WORKDIR, "2025,2026")
    print("  frames loaded in %.0fs" % (time.time() - t0), flush=True)
    tot = score[score["year"] == 2025].groupby("PitcherId").size().rename("tot")
    ctot = crit[crit["year"] == 2025].groupby("PitcherId").size().rename("ctot")

    ms = fc.pitch_mask(score, GRP)
    base_feats = fc.feats_for(GRP)
    base = score[ms].copy()
    base["RelSide_arm"] = base["RelSide"] * (1 - 2 * base["is_lhp"])
    base["HorzBreak_arm"] = base["HorzBreak"] * (1 - 2 * base["is_lhp"])
    for o, s_ in fc.DEV_SRC.items():
        base[o] = (base[s_] - base["is_lhp"].map(fc.DEV_CENTRES[o])).abs()
    base = add_vaa_flat(base)
    base = add_movement_geometry(base)
    base = add_nonlinear_terms(base)

    c = crit[fc.pitch_mask(crit, GRP) & (crit["year"] == 2025)]
    k = c.groupby("PitcherId").agg(cn=("adjT", "size"), crit=("adjT", "mean")).join(ctot)
    k = k[(k["cn"] >= ABS_MIN) & (k["cn"] / k["ctot"] >= SHARE)]

    out = {"share": SHARE, "n_boot": N_BOOT, "pass_bar": PASS_BAR,
           "ref_plate_height_ft": REF_PLATE_HEIGHT, "by_candidate": {}}
    for name, extra in CANDIDATES.items():
        feats = base_feats + extra
        b2 = base.dropna(subset=feats + ["Target"])
        tr = b2[(b2["year"] == 2024) & b2["Target"].notna()]
        ev = b2[b2["year"] == 2025]

        def build(train_df, eval_df, w=None):
            kw = {"ridge__sample_weight": w} if w is not None else {}
            m = make_pipeline(StandardScaler(), Ridge(alpha=fc.RIDGE_ALPHA))
            m.fit(train_df[feats].values, train_df["Target"].values, **kw)
            p = pd.Series(m.predict(eval_df[feats].values),
                          index=eval_df["PitcherId"].values)
            s = pd.DataFrame({"grade": p.groupby(level=0).mean()})
            g = eval_df.groupby("PitcherId").agg(sn=("adjT", "size"),
                                                 prior=("adjT", "mean")).join(tot)
            g = g[(g["sn"] >= ABS_MIN) & (g["sn"] / g["tot"] >= SHARE)]
            return s.join(g, how="inner").join(k, how="inner").dropna(
                subset=["grade", "prior", "crit"])

        j = build(tr, ev)
        print("")
        print("=== %s  n=%d pitchers ===" % (name, len(j)))
        if len(j) < MIN_PITCHERS:
            out["by_candidate"][name] = {"n": int(len(j)), "skipped": "pool too small"}
            continue
        rs, rp, rb, rsp = _stats(j)
        print("    stuff r        %+.4f" % rs)
        print("    prior-results  %+.4f" % rp)
        print("    blend 50/50    %+.4f   gain over results %+.4f" % (rb, rb - rp))
        print("    semipartial    %+.4f" % rsp)

        tr_codes, tr_ids = pd.factorize(tr["PitcherId"].values)
        ev_ids = ev["PitcherId"].unique()
        rng = np.random.default_rng(20260817)
        gains, sps = [], []
        for bi in range(N_BOOT):
            cnt = np.bincount(rng.integers(0, len(tr_ids), len(tr_ids)),
                              minlength=len(tr_ids)).astype(float)
            w = cnt[tr_codes]
            keep = set(rng.choice(ev_ids, len(ev_ids)))
            jb = build(tr, ev[ev["PitcherId"].isin(keep)], w=w)
            if len(jb) < MIN_PITCHERS:
                continue
            s2 = _stats(jb)
            gains.append(s2[2] - s2[1])
            sps.append(s2[3])
        gains, sps = np.array(gains), np.array(sps)
        p_gain = float((gains > 0).mean())
        lo, hi = np.percentile(gains, [2.5, 97.5])
        verdict = "PASS" if p_gain >= PASS_BAR else "no"
        print("    blend gain over %d refits: mean %+.4f  CI [%+.4f,%+.4f]  "
              "P(gain>0)=%.3f  -> %s" % (len(gains), gains.mean(), lo, hi, p_gain, verdict))
        out["by_candidate"][name] = {
            "n": int(len(j)), "feats": feats, "stuff_r": round(rs, 4),
            "results_r": round(rp, 4), "blend_r": round(rb, 4),
            "blend_gain": round(rb - rp, 4), "semipartial": round(rsp, 4),
            "n_boot_used": int(len(gains)),
            "gain_mean": round(float(gains.mean()), 4),
            "gain_ci": [round(float(lo), 4), round(float(hi), 4)],
            "p_gain_positive": p_gain,
            "p_semipartial_positive": float((sps > 0).mean()),
            "verdict": verdict}

    dest = os.path.join(SCORE_WORKDIR, "coach_si_feature_gate.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print("")
    print("  wrote %s   total %.0fs" % (dest, time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
