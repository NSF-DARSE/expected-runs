"""Local refresh job: score current season -> build bundle -> upload to Blob.

Runs the validated scorer (08_staff_scores.py) as a subprocess so its logic is
never restructured, then transforms + uploads. Fails loudly on any step.
"""
import argparse, json, os, pathlib, subprocess, sys
from datetime import datetime, timezone
from webapp_publisher.build_bundle import build_bundle
from webapp_publisher.schema import validate_bundle
from webapp_publisher.upload import upload_bundle

REPO = pathlib.Path(__file__).resolve().parents[1]
SCORER = REPO / "component_model" / "analysis" / "08_staff_scores.py"


def run_scorer(data: str, workdir: str, team: str) -> dict:
    workdir_p = pathlib.Path(workdir)
    workdir_p.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(SCORER), "--data", data, "--workdir", workdir, "--team", team]
    subprocess.run(cmd, check=True)  # raises CalledProcessError -> loud failure
    out = workdir_p / "staff_scores.json"
    if not out.exists():
        raise FileNotFoundError(f"scorer did not produce {out}")
    return json.loads(out.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("STUFFPLUS_DATA"))
    ap.add_argument("--workdir", default=os.environ.get("STUFFPLUS_WORKDIR"))
    ap.add_argument("--team", default="DEL_BLU")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--data-through", required=True, help="YYYY-MM-DD latest game date in the data")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if not args.data or not args.workdir:
        ap.error("--data and --workdir (or STUFFPLUS_DATA/STUFFPLUS_WORKDIR) required")

    staff_scores = run_scorer(args.data, args.workdir, args.team)
    built = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    bundle = build_bundle(staff_scores, season=args.season, data_through=args.data_through, built_iso=built)
    validate_bundle(bundle)

    if args.dry_run:
        out = pathlib.Path(args.workdir) / "bundle"
        out.mkdir(parents=True, exist_ok=True)
        for name, payload in bundle.items():
            (out / name).write_text(json.dumps(payload, indent=2, allow_nan=False))
        print(f"[dry-run] wrote {len(bundle)} files to {out}")
        return 0

    conn = os.environ["WEBAPP_STORAGE_CONNECTION_STRING"]
    container = os.environ.get("WEBAPP_BUNDLE_CONTAINER", "bundles")
    names = upload_bundle(bundle, connection_string=conn, container=container)
    print(f"uploaded {len(names)} files to {container}: {names} (data through {args.data_through})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
