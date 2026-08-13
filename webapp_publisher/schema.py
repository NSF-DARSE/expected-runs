"""Lightweight bundle validation — fail loudly before upload."""
REQUIRED_ROW_KEYS = {"id","name","hand","ff","stuff","loc","adjres","pitch",
                     "whiff","zone","heart","meanHeight","locFlag","stuffAttr",
                     "stuffNoHand","pitchNoHand","stuffAttrNoHand","pitcherId",
                     "stuffAttrDetail"}


def _check_stuff_attr_detail(row: dict) -> None:
    """stuffAttrDetail carries one value/percentile pair per feature named in
    stuffAttr or stuffAttrNoHand. A feature in one of those lists with no entry
    here would mean its points shipped with nothing to back them up; a value
    present without its percentile (or the reverse) means the join that built
    the pair only half-ran, which is a bug in the enrichment step, not a real
    "no data" case -- the real "no data" case (no pitcher file, or the feature
    name did not match model.featureOrder) always nulls both together. A
    percentile outside 0-100 is the same reference-population failure the
    arsenal percentile check above guards against.
    """
    detail = row["stuffAttrDetail"]
    names = {f for f, _ in row["stuffAttr"]} | {f for f, _ in row["stuffAttrNoHand"]}
    missing = names - set(detail)
    if missing:
        raise ValueError(f"pitcher row {row.get('name')} stuffAttrDetail missing {missing}")
    for name, d in detail.items():
        has_value = d["value"] is not None
        has_pct = d["percentile"] is not None
        if has_value != has_pct:
            raise ValueError(
                f"pitcher row {row.get('name')} stuffAttrDetail[{name!r}] has a value with no "
                f"percentile or a percentile with no value; they should always be null together"
            )
        if has_pct and not 0 <= d["percentile"] <= 100:
            raise ValueError(
                f"pitcher row {row.get('name')} stuffAttrDetail[{name!r}] percentile "
                f"{d['percentile']} is outside 0-100"
            )


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
        _check_stuff_attr_detail(r)


REQUIRED_ARSENAL_KEYS = {"type", "label", "n", "usage", "stuff", "loc",
                         "recentChange", "avgVelo", "locWhere", "aboveFloor",
                         "typical", "percentiles"}

# How far the location decomposition may drift from the score it explains before
# the publish aborts. The rows are an exact algebraic split of the same mean, so
# the only legitimate gap is float error: rare cells are pooled into an
# "Everywhere else" row rather than discarded. A loose tolerance here would have
# hidden exactly the bug this caught, where dropped cells cost 1.5 points.
LOC_DECOMP_TOLERANCE = 0.01
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

# avgVelo is the one number on the page in real units rather than on the display
# scale, so the band is a units check, not a quality check. A mean RelSpeed that
# lands outside this is a join that dropped rows, a NaN mean, or someone handing
# it m/s (a 93 mph fastball reads 41.6). College arms sit roughly 68-98; the band
# is wide enough that no real pitch aborts a publish and narrow enough that none
# of those three failures survives it.
VELO_BAND = (55.0, 110.0)

# Per-type adjusted results spread far wider than the fastball season figure
# DISPLAY_BAND was drawn for, and for a structural reason rather than a defect:
# the fastball board averages over 100+ pitches per qualified pitcher, while a
# pitch-type results number can rest on under 30. Measured on a real bundle the
# range is 28.7 to 140.1 across types, with per-type medians a sane 74.7 to
# 103.3, so DISPLAY_BAND would abort a publish over a legitimately bad changeup.
# The band's real job here is the same as PITCH_GRADE_BAND's: catch a raw run
# value (|v| < ~0.2) that never went through to_display. Anything above ~0.5
# does that, and tightness beyond it only costs real publishes.
ADJRES_BAND = (1.0, 250.0)


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


def _check_velo(value, *, key, ptype):
    """Raise if `value` is not a plausible average release speed in mph.

    None is allowed and means the source extract carried no RelSpeed column; the
    page renders the row without a velocity rather than inventing one. What is
    not allowed is a present-but-wrong value. NaN is called out separately: a
    mean over an all-null slice comes back NaN, which fails every comparison
    silently and would otherwise reach the page as "NaN mph".
    """
    if value is None:
        return
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{key} avgVelo for {ptype} is not numeric: {value!r}")
    if value != value:
        raise ValueError(f"{key} avgVelo for {ptype} is NaN; RelSpeed is missing for that slice")
    low, high = VELO_BAND
    if not low <= value <= high:
        raise ValueError(
            f"{key} avgVelo for {ptype} is {value}, outside the plausible "
            f"{low:g}-{high:g} mph range; check units and the RelSpeed join"
        )


def validate_pitcher_bundle(files: dict) -> None:
    """Fail loudly before upload. Mirrors validate_bundle's style: plain
    ValueErrors naming the offending file and key.
    """
    for name in ("location_maps.json", "model_artifacts.json", "staff_by_type.json"):
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

    for t in files["staff_by_type.json"]["types"]:
        if not t["pitchers"]:
            raise ValueError(f"staff_by_type has no pitchers for {t['type']}")
        if t["type"] not in files["model_artifacts.json"]["byPitchType"]:
            raise ValueError(f"staff_by_type has pitch type {t['type']!r} with no model artifact")
        for r in t["pitchers"]:
            _check_display_band(r["stuff"], DISPLAY_BAND, key="staff_by_type.json",
                                field="staff Stuff+", ptype=t["type"])
            # None is legitimate: a type can lack the qualifying pitchers needed
            # to set a results scale. A present value still has to be on the
            # display scale like every other score.
            if r.get("adjRes") is not None:
                _check_display_band(r["adjRes"], ADJRES_BAND, key="staff_by_type.json",
                                    field="staff Adj Results", ptype=t["type"])

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
            _check_velo(a["avgVelo"], key=key, ptype=a["type"])
            if a.get("adjRes") is not None:
                _check_display_band(a["adjRes"], ADJRES_BAND, key=key,
                                    field="arsenal Adj Results", ptype=a["type"])
            if a["type"] == "FF":
                if not a["locWhere"]:
                    raise ValueError(f"{key} fastball row has no Location+ decomposition")
                total = sum(r["points"] for r in a["locWhere"])
                if abs(total - (a["loc"] - 100.0)) > LOC_DECOMP_TOLERANCE:
                    raise ValueError(
                        f"{key} Location+ decomposition sums to {total:.2f} but the score is "
                        f"{a['loc'] - 100.0:.2f} off 100; the rows do not explain the number "
                        f"they sit under"
                    )
                for r in a["locWhere"]:
                    if not 0.0 <= r["share"] <= 1.0 or not 0.0 <= r["leagueShare"] <= 1.0:
                        raise ValueError(f"{key} Location+ row {r['region']!r} has a share outside 0-1")
            elif a["locWhere"] is not None:
                raise ValueError(
                    f"{key} emits a Location+ decomposition for {a['type']}; Location+ is a "
                    f"fastball score only"
                )
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
