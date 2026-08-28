"""Transform the validated staff_scores.json into the frontend bundle contract.

Input is the dict written by component_model/analysis/08_staff_scores.py.
Scores are already on the 100±15 display scale (higher = better); do not re-flip.
"""
from __future__ import annotations
import math
from typing import Any

try:
    import numpy as np
except ImportError:  # numpy optional for pure-dict inputs
    np = None


def to_native(obj: Any) -> Any:
    if np is not None:
        if isinstance(obj, np.ndarray):
            return [to_native(x) for x in obj.tolist()]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            f = float(obj)
            return None if math.isnan(f) or math.isinf(f) else f
        if isinstance(obj, np.bool_):
            return bool(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(x) for x in obj]
    return obj


def assign_ids(staff: list[dict]) -> dict[str, int]:
    """Stable id per pitcher name: index into the sorted unique-name list, +1."""
    names = sorted({s["name"] for s in staff})
    return {name: i + 1 for i, name in enumerate(names)}


def _res_ladder(s: dict) -> dict | None:
    """Adj Results' ladder (Runs Allowed -> Expected Runs Allowed -> Adj
    Results), if 08_staff_scores.py emitted one. Absent entirely -- not a key
    with nulls in it -- on a staff_scores.json from before the ladder existed,
    the same way locWhere/locBaseline are absent rather than null on an older
    pitcher file. That lets the frontend tell "no ladder shipped for this
    build" from "ladder shipped but a level came back empty" and fall back to
    prose only for the former.
    """
    keys = ("res_runs_allowed", "res_exp_runs_allowed", "res_runs_allowed_raw",
            "res_exp_runs_allowed_raw", "res_adj_results_raw")
    if any(k not in s for k in keys):
        return None
    return {
        "runsAllowed": s["res_runs_allowed"],
        "expRunsAllowed": s["res_exp_runs_allowed"],
        "runsAllowedRaw": s["res_runs_allowed_raw"],
        "expRunsAllowedRaw": s["res_exp_runs_allowed_raw"],
        "adjResultsRaw": s["res_adj_results_raw"],
    }


def _row(s: dict, pid: int) -> dict:
    row = {
        "id": pid,
        "name": s["name"],
        "hand": s["hand"],
        "ff": s["ff"],
        "stuff": s["stuff"],
        "loc": s["loc"],
        "adjres": s["adjres"],
        "pitch": s["pitch"],
        "stuffNoHand": s["stuff_nohand"],
        "pitchNoHand": s["pitch_nohand"],
        "whiff": s["whiff"],
        "zone": s["zone"],
        "heart": s["heart"],
        "meanHeight": s["mean_height"],
        "locFlag": s["loc_flag"],
        "stuffAttr": [[f, v] for f, v in s["stuff_attr"]],
        "stuffAttrNoHand": [[f, v] for f, v in s["stuff_attr_nohand"]],
    }
    ladder = _res_ladder(s)
    if ladder is not None:
        row["resLadder"] = ladder
    return to_native(row)


def build_bundle(staff_scores: dict, *, season: int, data_through: str, built_iso: str) -> dict[str, dict]:
    ids = assign_ids(staff_scores["staff"])
    pitchers = [_row(s, ids[s["name"]]) for s in staff_scores["staff"]]
    pitchers.sort(key=lambda r: r["pitch"], reverse=True)
    return {
        "manifest.json": {
            "built": built_iso, "season": season,
            "dataThrough": data_through, "bundleVersion": built_iso,
        },
        "staff_board.json": {
            "population": to_native(staff_scores["population"]),
            "team": staff_scores["team"],
            "pitchers": pitchers,
        },
    }
