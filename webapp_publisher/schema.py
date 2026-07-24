"""Lightweight bundle validation — fail loudly before upload."""
REQUIRED_ROW_KEYS = {"id","name","hand","ff","stuff","loc","adjres","pitch",
                     "whiff","zone","heart","meanHeight","locFlag","stuffAttr"}

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
