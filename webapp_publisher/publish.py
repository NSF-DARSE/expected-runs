"""Local refresh job: score current season -> build bundle -> upload to Blob.

Runs the validated scorer (08_staff_scores.py) as a subprocess so its logic is
never restructured, then transforms + uploads. Fails loudly on any step.
"""
import argparse, json, os, pathlib, subprocess, sys
from datetime import datetime, timezone

import pandas as pd

from webapp_publisher.build_bundle import build_bundle, to_native
from webapp_publisher.build_pitcher_bundle import (
    build_pitcher_bundle, enrich_stuff_attr_detail, pitcher_index, stamp_pitcher_ids,
)
from webapp_publisher.schema import validate_bundle, validate_pitcher_bundle
from webapp_publisher.upload import upload_bundle

REPO = pathlib.Path(__file__).resolve().parents[1]
SCORER = REPO / "component_model" / "analysis" / "08_staff_scores.py"
PITCHER_SCORER = REPO / "component_model" / "analysis" / "14_pitcher_pages.py"


def _load_env_file() -> None:
    """Load webapp_publisher/.env if python-dotenv is available.

    Best-effort only: the scheduler (run_refresh.ps1) may already populate
    os.environ directly, so a missing python-dotenv install or a missing
    .env file should never be fatal here.
    """
    env_path = pathlib.Path(__file__).parent / ".env"
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("[publish] python-dotenv not installed; skipping .env load "
              "(relying on process environment variables instead)")
        return
    load_dotenv(env_path)


def default_season() -> int:
    """The season the scorer actually grades: the later year in STUFFPLUS_YEARS.

    08_staff_scores.py's population is always the LATER year of the
    train,eval pair (relabeled internally to the "2025" role regardless of
    the literal year) -- see fair_criterion.py's --years/STUFFPLUS_YEARS
    docstring. Labels must follow that, not a free-text default.
    """
    years_env = os.environ.get("STUFFPLUS_YEARS", "2024,2025")
    years = [int(y.strip()) for y in years_env.split(",")]
    if len(years) != 2:
        raise ValueError(f"STUFFPLUS_YEARS must be two comma-separated years, got {years_env!r}")
    return max(years)


def derive_data_through(data_path: str, season: int, team: str) -> str:
    """Latest game date (YYYY-MM-DD) `team` threw a pitch in `data_path`, within `season`.

    `data_path` is the full population source (every D1 team), so the date
    must be scoped to `team` -- otherwise it describes the whole population's
    latest game, not the bundle's own team, which is a correctness defect
    (the frontend renders this value next to that team's rows only).

    Reads only the Date and PitcherTeam columns. Dates may be strings
    ("2025-05-16") or yyyymmdd integers/strings (20250516); both are handled.
    Fails loudly (rather than silently falling back to "today" or to the
    population-wide maximum) if the Date column is missing or no rows match
    both `team` and the season year -- a wrong-but-quiet label is worse than
    a crash here.
    """
    df = pd.read_csv(data_path, usecols=["Date", "PitcherTeam"])
    as_str = df["Date"].astype(str)
    parsed = pd.to_datetime(as_str, errors="coerce")
    still_missing = parsed.isna()
    if still_missing.any():
        yyyymmdd = pd.to_datetime(as_str[still_missing], format="%Y%m%d", errors="coerce")
        parsed.loc[still_missing] = yyyymmdd
    in_season = parsed[(parsed.dt.year == season) & (df["PitcherTeam"] == team)]
    if in_season.empty:
        raise ValueError(f"No rows in {data_path} with PitcherTeam == {team!r} and Date in "
                         f"season {season}; cannot derive --data-through. Pass it explicitly "
                         f"if this is expected.")
    return in_season.max().strftime("%Y-%m-%d")


def run_scorer(data: str, workdir: str, team: str) -> dict:
    workdir_p = pathlib.Path(workdir)
    workdir_p.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCORER), "--data", data, "--workdir", workdir, "--team", team]
    subprocess.run(cmd, check=True)  # raises CalledProcessError -> loud failure
    out = workdir_p / "staff_scores.json"
    if not out.exists():
        raise FileNotFoundError(f"scorer did not produce {out}")
    return json.loads(out.read_text())


def run_pitcher_scorer(data: str, workdir: str, team: str) -> dict:
    workdir_p = pathlib.Path(workdir)
    cmd = [sys.executable, str(PITCHER_SCORER), "--data", data, "--workdir", workdir, "--team", team]
    subprocess.run(cmd, check=True)  # raises CalledProcessError -> loud failure
    out = workdir_p / "pitcher_pages.json"
    if not out.exists():
        raise FileNotFoundError(f"pitcher scorer did not produce {out}")
    return json.loads(out.read_text())


def main() -> int:
    _load_env_file()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("STUFFPLUS_DATA"))
    ap.add_argument("--workdir", default=os.environ.get("STUFFPLUS_WORKDIR"))
    ap.add_argument("--team", default="DEL_BLU")
    ap.add_argument("--season", type=int, default=None,
                    help="Override the graded population's season. Default: the later "
                         "year in STUFFPLUS_YEARS (the year the scorer actually populates).")
    ap.add_argument("--data-through", default=None,
                    help="Override YYYY-MM-DD latest game date. Default: derived from the "
                         "max Date in --data restricted to the season year.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.data or not args.workdir:
        ap.error("--data and --workdir (or STUFFPLUS_DATA/STUFFPLUS_WORKDIR) required")

    season = args.season if args.season is not None else default_season()
    data_through = args.data_through if args.data_through is not None else derive_data_through(args.data, season, args.team)

    staff_scores = run_scorer(args.data, args.workdir, args.team)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    bundle = build_bundle(staff_scores, season=season, data_through=data_through, built_iso=built)
    pages = run_pitcher_scorer(args.data, args.workdir, args.team)
    pitcher_files = build_pitcher_bundle(pages)
    validate_pitcher_bundle(pitcher_files)
    bundle["manifest.json"]["pitchers"] = pitcher_index(pages)
    stamp_pitcher_ids(bundle, pages)
    enrich_stuff_attr_detail(bundle, pages)
    bundle.update(pitcher_files)
    validate_bundle(bundle)

    if args.dry_run:
        out = pathlib.Path(args.workdir) / "bundle"
        out.mkdir(parents=True, exist_ok=True)
        for name, payload in bundle.items():
            # Bundle keys can be nested (e.g. "pitchers/1000123.json"), so the
            # per-file parent has to exist before the write, not just <workdir>/bundle.
            dest = out / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(to_native(payload), indent=2, allow_nan=False))
        print(f"[dry-run] wrote {len(bundle)} files to {out}")
        return 0

    conn = os.environ["WEBAPP_STORAGE_CONNECTION_STRING"]
    container = os.environ.get("WEBAPP_BUNDLE_CONTAINER", "bundles")
    names = upload_bundle(bundle, connection_string=conn, container=container)
    print(f"uploaded {len(names)} files to {container}: {names} (season {season}, data through {data_through})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
