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


def enrich_stuff_attr_detail(bundle: dict, pages: dict) -> None:
    """Attach each staff-board Stuff+ trait's raw value and percentile, sourced
    from the pitcher's FF arsenal row, so the hover card can show more than bare
    points.

    08_staff_scores.py (upstream of stuffAttr/stuffAttrNoHand) lowercases every
    feature name; 14_pitcher_pages.py (upstream of model.featureOrder and the
    arsenal's typical/percentiles) keeps canonical casing (EffectiveVelo, not
    effectivevelo). A case-sensitive join would null out every trait, so this
    matches case-insensitively.

    Even case-insensitively, a name can still fail to match -- a real naming
    drift between the two scripts, not just casing. That is not a reason to
    guess which feature was meant or to drop the points row that already shipped
    from 08_staff_scores: the row keeps its points, value and percentile come
    back null, and a line is printed so whoever runs publish notices instead of
    the gap sitting quiet on the page forever.

    Requires stamp_pitcher_ids to have already run, since it reads row["pitcherId"].
    """
    order = pages["model"]["featureOrder"]
    index_by_lower = {f.lower(): i for i, f in enumerate(order)}
    pitchers_by_id = {int(p["pitcherId"]): p for p in pages["pitchers"]}

    for row in bundle["staff_board.json"]["pitchers"]:
        names = {f for f, _ in row["stuffAttr"]} | {f for f, _ in row["stuffAttrNoHand"]}
        pitcher = pitchers_by_id.get(row["pitcherId"]) if row["pitcherId"] is not None else None
        ff = next((a for a in pitcher["arsenal"] if a["type"] == "FF"), None) if pitcher else None

        detail: dict[str, dict] = {}
        for name in names:
            idx = index_by_lower.get(name.lower())
            if ff is not None and idx is not None:
                detail[name] = {"value": ff["typical"][idx], "percentile": ff["percentiles"][idx]}
            else:
                if ff is not None and idx is None:
                    print(f"stuffAttr feature {name!r} on {row['name']!r} has no match in "
                          f"model.featureOrder; shipping its points with no value/percentile")
                detail[name] = {"value": None, "percentile": None}
        row["stuffAttrDetail"] = to_native(detail)


def build_type_board(pages: dict) -> dict:
    """Per-pitch-type staff table, regrouped from the pitcher pages.

    No new modeling: 14_pitcher_pages already fits every pitch type with its own
    scale and its own qualified population, so this is the arsenal rows pivoted
    from by-pitcher to by-type.

    Carries Stuff+ and adjusted results, and deliberately not Location+ or
    Pitching+. Location+ is a fastball score by a settled decision (reliable on
    secondaries but with no predictive validity there), and Pitching+ is a blend
    that includes it, so a board showing either for a slider would be inventing
    it. Adjusted results is different in kind: it describes what happened with
    luck, defense and opponent quality removed, and a description does not have
    to predict next season to be true. It is None for a type with too few
    qualifying pitchers to set a scale.

    `nQualified` rides along per type because the scales rest on very different
    populations (four-seam on thousands, splitter on tens), and a grade is not
    readable without it.
    """
    artifacts = pages["model"]["byPitchType"]
    by_type: dict[str, list[dict]] = {}
    for p in pages["pitchers"]:
        for a in p["arsenal"]:
            by_type.setdefault(a["type"], []).append({
                "pitcherId": int(p["pitcherId"]),
                "name": p["name"],
                "hand": p["hand"],
                "n": a["n"],
                "usage": a["usage"],
                "stuff": a["stuff"],
                "adjRes": a.get("adjRes"),
                "avgVelo": a.get("avgVelo"),
                "aboveFloor": a["aboveFloor"],
            })
    return to_native({
        "types": [
            {
                "type": t,
                "label": PITCH_TYPE_LABELS.get(t, t),
                "nQualified": artifacts.get(t, {}).get("nQualified"),
                "sampleFloor": artifacts.get(t, {}).get("sampleFloor"),
                "pitchers": sorted(rows, key=lambda r: r["stuff"], reverse=True),
            }
            for t, rows in sorted(by_type.items(), key=lambda kv: -len(kv[1]))
        ],
    })


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
    files["staff_by_type.json"] = build_type_board(pages)
    return files
