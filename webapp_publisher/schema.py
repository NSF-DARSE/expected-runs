"""Lightweight bundle validation — fail loudly before upload."""
REQUIRED_ROW_KEYS = {"id","name","hand","ff","stuff","loc","adjres","pitch",
                     "whiff","zone","heart","meanHeight","locFlag","stuffAttr",
                     "stuffNoHand","pitchNoHand","stuffAttrNoHand"}

def validate_bundle(bundle: dict) -> None:
    m = bundle["manifest.json"]
    for k in ("built","season","dataThrough","bundleVersion"):
        if k not in m:
            raise ValueError(f"manifest missing {k}")
    board = bundle["staff_board.json"]
    if not board["pitchers"]:
        raise ValueError("staff_board has no pitchers")
    for r in board["pitchers"]:
        missing = REQUIRED_ROW_KEYS - set(r)
        if missing:
            raise ValueError(f"pitcher row {r.get('name')} missing {missing}")
        if r["locFlag"] not in ("", "caution", "small sample"):
            raise ValueError(f"bad locFlag {r['locFlag']}")


REQUIRED_ARSENAL_KEYS = {"type", "label", "n", "usage", "stuff", "loc",
                         "recentChange", "aboveFloor", "typical", "percentiles"}
REQUIRED_PITCH_KEYS = {"d", "t", "x", "z", "c", "g", "f"}


def validate_pitcher_bundle(files: dict) -> None:
    """Fail loudly before upload. Mirrors validate_bundle's style: plain
    ValueErrors naming the offending file and key.
    """
    for name in ("location_maps.json", "model_artifacts.json"):
        if name not in files:
            raise ValueError(f"pitcher bundle missing {name}")

    model = files["model_artifacts.json"]
    order = model["featureOrder"]
    n_feats = len(order)
    for feat in order:
        if model.get("labels", {}).get(feat) in (None, feat):
            raise ValueError(f"feature {feat} has no plain-English label")
    for tname, m in model["byPitchType"].items():
        for key in ("coef", "scalerMean", "scalerScale", "populationMeanZ"):
            if len(m[key]) != n_feats:
                raise ValueError(f"{tname}.{key} feature array is {len(m[key])}, expected {n_feats}")
        if m["displaySd"] <= 0:
            raise ValueError(f"{tname} displaySd must be positive, got {m['displaySd']}")

    pitcher_files = [k for k in files if k.startswith("pitchers/")]
    if not pitcher_files:
        raise ValueError("pitcher bundle has no pitcher files")

    for key in pitcher_files:
        body = files[key]
        if not body.get("arsenal"):
            raise ValueError(f"{key} has no arsenal rows")
        for a in body["arsenal"]:
            missing = REQUIRED_ARSENAL_KEYS - set(a)
            if missing:
                raise ValueError(f"{key} arsenal row missing {missing}")
            if a["type"] != "FF" and a["loc"] is not None:
                raise ValueError(f"{key} emits Location+ for {a['type']}; it is a fastball score only")
            if a["type"] == "FF" and not isinstance(a["loc"], (int, float)):
                raise ValueError(f"{key} is missing a numeric Location+ for its fastball")
            for arr in ("typical", "percentiles"):
                if len(a[arr]) != n_feats:
                    raise ValueError(f"{key} arsenal {arr} feature array is {len(a[arr])}, expected {n_feats}")
            if any(not 0 <= p <= 100 for p in a["percentiles"]):
                raise ValueError(f"{key} has a percentile outside 0-100")
        for p in body["pitches"]:
            missing = REQUIRED_PITCH_KEYS - set(p)
            if missing:
                raise ValueError(f"{key} pitch row missing {missing}")
            if len(p["f"]) != n_feats:
                raise ValueError(f"{key} pitch feature array is {len(p['f'])}, expected {n_feats}")
