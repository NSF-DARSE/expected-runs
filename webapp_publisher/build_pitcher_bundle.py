"""Transform 14_pitcher_pages.py's output into the pitcher-page bundle files.

Input is the dict written by component_model/analysis/14_pitcher_pages.py.
Scores are already on the 100±15 display scale (higher = better); do not re-flip.
"""
from __future__ import annotations

# Absolute import, matching publish.py's existing style in this package.
from webapp_publisher.build_bundle import to_native

# Plain-English labels for the coach-facing page. A label that needs a glossary
# entry to be legible is a label defect -- fix the label, not the glossary.
FEATURE_LABELS = {
    "SpinRate": "Spin rate",
    "Extension": "Extension",
    "HorzBreak": "Horizontal break",
    "InducedVertBreak": "Vertical break",
    "EffectiveVelo": "Perceived velo",
    "RelHeight": "Release height",
    "RelSide": "Release side",
    "vertbreakdiff": "Vertical break vs his fastball",
    "horzbreakdiff": "Horizontal break vs his fastball",
    "velocity_differential": "Velo vs his fastball",
    "is_lhp": "Throws left",
    "is_lhb": "Batter hits left",
}

PITCH_TYPE_LABELS = {
    "FF": "Fastball",
    "Slider": "Slider",
    "ChangeUp": "Changeup",
    "Curveball": "Curveball",
    "Sinker": "Sinker",
    "Cutter": "Cutter",
    "Splitter": "Splitter",
}


def pitcher_index(pages: dict) -> list[dict]:
    """Small index for the manifest, so routing does not need every pitcher file."""
    return [{"pitcherId": p["pitcherId"], "name": p["name"], "hand": p["hand"]}
            for p in pages["pitchers"]]


def stamp_pitcher_ids(bundle: dict, pages: dict) -> None:
    """Join the stable TrackMan PitcherId onto each staff-board row, by name.

    The board's own `id` is a positional index into the sorted name list, so it
    shifts whenever the roster changes and cannot name a file or appear in a URL.
    Name is the only column the two sides share -- 08_staff_scores.py is a fixed
    reference and does not emit PitcherId -- so a duplicate name is a hard error
    rather than a coin flip that could route a coach to the wrong player.
    """
    by_name: dict[str, int] = {}
    for p in pages["pitchers"]:
        name = p["name"]
        if name in by_name:
            raise ValueError(f"more than one pitcher file claims the name {name!r}")
        by_name[name] = int(p["pitcherId"])
    for row in bundle["staff_board.json"]["pitchers"]:
        row["pitcherId"] = by_name.get(row["name"])


def build_pitcher_bundle(pages: dict) -> dict[str, dict]:
    model = dict(pages["model"])
    missing = [f for f in model["featureOrder"] if f not in FEATURE_LABELS]
    if missing:
        raise ValueError(f"no plain-English label for features {missing}")
    model["labels"] = {f: FEATURE_LABELS[f] for f in model["featureOrder"]}

    files: dict[str, dict] = {
        "location_maps.json": to_native(pages["grids"]),
        "model_artifacts.json": to_native(model),
    }
    for p in pages["pitchers"]:
        body = to_native({
            "pitcherId": p["pitcherId"],
            "name": p["name"],
            "hand": p["hand"],
            "season": pages["season"],
            "arsenal": [{**a, "label": PITCH_TYPE_LABELS.get(a["type"], a["type"])}
                        for a in p["arsenal"]],
            "outings": p["outings"],
            "pitches": p["pitches"],
        })
        files[f"pitchers/{p['pitcherId']}.json"] = body
    return files
