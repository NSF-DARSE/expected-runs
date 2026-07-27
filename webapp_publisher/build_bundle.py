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


def _row(s: dict, pid: int) -> dict:
    return to_native({
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
    })


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
