"""The pooled sinker model (fair_criterion.pooled_si_ridge / ridge_for_group).

The property that matters: the collapsed sinker model, which has the same shape as
every other type's (scaler + ridge over its own feature list), predicts EXACTLY what
the pooled four-seam+sinker ridge predicts on sinker rows. If it did not, the pitcher
pages would ship a grade the gate never measured. Synthetic data, real fits.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import arsenal as ar
import fair_criterion as fc
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _frame(seed=11, sinker_only_pitchers=(6, 7)):
    """Two seasons, seven pitchers, four-seams and sinkers. Pitchers in
    sinker_only_pitchers throw no four-seam, so their sinker has no anchor and
    its secondary-sinker differentials must be exactly zero."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in (2024, 2025):
        for pid in range(1, 8):
            lhp = int(pid % 3 == 0)
            plan = [("Sinker", 25)]
            if pid not in sinker_only_pitchers:
                plan.append(("FourSeamFastBall", 25))
            for tag, n in plan:
                for _ in range(n):
                    rows.append({
                        "PitcherId": pid, "Pitcher": f"Test-Pitcher, {pid}", "year": year,
                        "PitcherThrows": "Left" if lhp else "Right", "is_lhp": lhp, "is_lhb": 0,
                        "TaggedPitchType": tag, "is_ff": tag == "FourSeamFastBall",
                        "SpinRate": rng.normal(2200, 120),
                        "Extension": rng.normal(6.2, 0.2),
                        "HorzBreak": rng.normal((14 if tag == "Sinker" else 8) * (-1 if lhp else 1), 3),
                        "InducedVertBreak": rng.normal(8 if tag == "Sinker" else 16, 3),
                        "RelSpeed": rng.normal(91, 1.5),
                        "RelHeight": rng.normal(5.8, 0.25), "RelSide": rng.normal(1.6, 0.3),
                        "PlateLocSide": rng.normal(0, 0.6), "PlateLocHeight": rng.normal(2.4, 0.6),
                        "Target": rng.normal(0, 0.05), "adjT": rng.normal(0, 0.05),
                    })
    return fc.add_fastball_diffs(pd.DataFrame(rows))


def test_collapsed_model_equals_the_pooled_ridge_on_every_sinker():
    df = _frame()
    si, model = fc.pooled_si_ridge(df, return_model=True)
    pred = model.predict(si[fc.SI_FEATS].values)
    np.testing.assert_allclose(pred, si["ridge_pred"].values, rtol=0, atol=1e-9)
    # ...and the pooled prediction is what an independent pooled fit gives.
    both = fc.add_derived_feats(df[df["is_ff"] | fc.pitch_mask(df, "SI")].copy())
    both = both.dropna(subset=fc.SI_POOLED_TRAIN_FEATS + ["Target"])
    tr = both[both["year"] == 2024]
    ref = make_pipeline(StandardScaler(), Ridge(alpha=fc.RIDGE_ALPHA))
    ref.fit(tr[fc.SI_POOLED_TRAIN_FEATS].values, tr["Target"].values)
    si_ref = both[both["is_si"] == 1]
    np.testing.assert_allclose(ref.predict(si_ref[fc.SI_POOLED_TRAIN_FEATS].values),
                               si.loc[si_ref.index, "ridge_pred"].values, atol=1e-9)


def test_collapsed_model_is_not_the_sinkers_own_ridge():
    """The point of pooling: shrinkage toward the four-seam changes the coefficients."""
    df = _frame()
    _, pooled = fc.pooled_si_ridge(df, return_model=True)
    _, own = fc.stuff_ridge(df, pitch_mask=fc.pitch_mask(df, "SI"), feats=fc.SI_FEATS,
                            return_model=True)
    assert not np.allclose(pooled.named_steps["ridge"].coef_, own.named_steps["ridge"].coef_)


def test_secondary_sinker_columns_are_zero_without_a_four_seam_anchor():
    df = _frame(sinker_only_pitchers=(6, 7))
    si = fc.pooled_si_ridge(df)
    lone = si[si["PitcherId"].isin([6, 7])]
    paired = si[~si["PitcherId"].isin([6, 7])]
    assert (lone["is_secondary_si"] == 0).all()
    assert (lone[[f"{c}_sec" for c in fc.DIFF_FEATS]] == 0).all().all()
    assert (paired["is_secondary_si"] == 1).all()
    assert (paired["velocity_differential_sec"] == paired["velocity_differential"]).all()


def test_derived_feats_leave_non_sinker_rows_with_zero_sinker_columns():
    df = _frame()
    out = fc.add_derived_feats(df.copy())
    ff = out[out["is_ff"]]
    for c in ["is_si", "is_secondary_si"] + [f"si_x_{f}" for f in fc.SI_INTERACT]:
        assert (ff[c] == 0).all(), c
    assert set(fc.UNION_FEATS) <= set(out.columns)


def test_ridge_for_group_routes_the_sinker_to_the_pooled_model_and_others_to_their_own():
    df = _frame()
    si, model, feats = fc.ridge_for_group(df, "SI")
    assert feats == fc.SI_FEATS
    assert set(si["TaggedPitchType"]) == {"Sinker"}
    assert len(model.named_steps["ridge"].coef_) == len(fc.SI_FEATS)
    ff, _, ff_feats = fc.ridge_for_group(df, "FF")
    assert ff_feats == fc.feats_for("FF")
    assert ff["is_ff"].all()


def test_fit_type_ships_the_pooled_sinker_on_the_published_order():
    """arsenal.fit_type must carry the pooled list, not FEATS_BY_PITCH["SI"], and the
    artifact must align on UNION_FEATS."""
    df = _frame()
    state = ar.fit_type(df, {"Sinker", "TwoSeamFastBall"}, floor_n=10, fc_module=fc,
                        season_year=2025, group="SI", report_feats=fc.UNION_FEATS)
    assert state["feats"] == fc.SI_FEATS
    assert len(state["coef"]) == len(fc.SI_FEATS)
    assert "mov_angle" in state["feats"] and "is_secondary_si" in state["feats"]
    assert list(state["reference_features"].columns) == fc.UNION_FEATS
    assert set(state["pitches"]["TaggedPitchType"]) == {"Sinker"}


def test_fit_type_refuses_report_feats_that_miss_the_pooled_list():
    df = _frame()
    with pytest.raises(ValueError):
        ar.fit_type(df, {"Sinker"}, 10, fc, 2025, group="SI", report_feats=fc.feats_for("SI"))
