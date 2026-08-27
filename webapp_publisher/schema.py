"""Lightweight bundle validation — fail loudly before upload."""
REQUIRED_ROW_KEYS = {"id","name","hand","ff","stuff","loc","adjres","pitch",
                     "whiff","zone","heart","meanHeight","locFlag","stuffAttr",
                     "stuffNoHand","pitchNoHand","stuffAttrNoHand","pitcherId"}

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
                         "recentChange", "trend", "aboveFloor", "typical",
                         "percentiles"}
REQUIRED_PITCH_KEYS = {"d", "t", "x", "z", "c", "g", "f"}

# Plausible-range guard for scores on the 100+/-15 display scale. A raw
# expected-run value (~0.00x, lower = better) or an un-negated score shipped
# once as `loc`, unscaled and with reversed polarity, and bare numeric-ness
# was what let it through. `stuff` (every arsenal row) and `g` (every pitch
# row) sit on the same display scale and are just as exposed to that failure
# mode, so they get the same kind of guard.
#
# Season/pitcher-aggregate scores (loc, stuff) use this tighter band.
DISPLAY_BAND = (40.0, 160.0)
# Per-pitch grades (g) spread wider than season aggregates: measured on a real
# bundle, four-seam per-pitch g ranges ~20-147 with per-pitcher standard
# deviations of 8.9-16.0. DISPLAY_BAND would reject legitimate real data at
# either edge, so PITCH_GRADE_BAND is deliberately wider.
#
# On how wide: be honest about what this band can and cannot detect. A raw
# ridge_pred is |v| < ~0.2, so ANY lower bound above ~0.5 catches an unscaled
# value just as decisively as a tight one buys nothing extra. What a band cannot
# catch on `g` or `stuff` is a POLARITY flip -- 100 + 15z lands in 40-160 the
# same as 100 - 15z does, which is why the original loc guard worked only because
# loc was arriving raw. So the band's whole job is "unscaled value", and every
# point of tightness beyond that is pure downside: one real team's worst pitch
# already grades 19.9, leaving under 10 points of headroom, and a genuinely awful
# pitch on some other staff would abort a publish for no diagnostic gain.
PITCH_GRADE_BAND = (1.0, 250.0)


def _check_display_band(value, band, *, key, field, ptype):
    """Raise if `value` (a score on the 100+/-15 display scale) falls outside
    `band`. Callers pass a file key, a field description, and the pitch type
    so the message can name all three plus the offending value.
    """
    low, high = band
    if not isinstance(value, (int, float)):
        raise ValueError(f"{key} {field} for {ptype} is not numeric: {value!r}")
    if not low <= value <= high:
        raise ValueError(
            f"{key} {field} for {ptype} is {value}, outside the plausible "
            f"{low:g}-{high:g} display range; it is probably not on the 100+/-15 scale"
        )


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
        label = model.get("labels", {}).get(feat)
        # Presence and non-emptiness only. Do NOT require label != feat: some
        # features are already plain English ("Extension"), and requiring a
        # difference rejects a correct bundle. build_pitcher_bundle owns the
        # label map and raises when an entry is genuinely absent.
        if not isinstance(label, str) or not label.strip():
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
            if a["type"] == "FF":
                if not isinstance(a["loc"], (int, float)):
                    raise ValueError(f"{key} is missing a numeric Location+ for its fastball")
                _check_display_band(a["loc"], DISPLAY_BAND, key=key, field="fastball Location+", ptype=a["type"])
            _check_display_band(a["stuff"], DISPLAY_BAND, key=key, field="arsenal Stuff+", ptype=a["type"])
            if a["type"] not in model["byPitchType"]:
                raise ValueError(
                    f"{key} arsenal has pitch type {a['type']!r} with no matching entry "
                    f"in model_artifacts.json byPitchType"
                )
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
            _check_display_band(p["g"], PITCH_GRADE_BAND, key=key, field="pitch grade (g)", ptype=p["t"])
