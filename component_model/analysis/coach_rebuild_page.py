"""Rebuild every input the coach page needs, then render it, in dependency order.

WHY THIS EXISTS: changing fair_criterion.FEATS changes ridge_pred, which invalidates every
section of the page at once. Rebuilding by hand invites exactly the failure this project
already hit -- a stale artifact surviving next to fresh ones and being read as a real
result. One command, fixed order, and a hard stop on the first failure.

The location map is rebuilt too even though it does NOT depend on FEATS (Location+ is a
pooled map over plate location only). It is cheap, and having one entry point that
reproduces the whole page beats remembering which artifacts a given change spares.

Run with -u so progress is visible while it works; Python buffers stdout when piped.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import coach_model_ff_criterion as ffc

DATA = ffc.DATA
WORKDIR = ffc.SCORE_WORKDIR

# LEVEL matters and is not optional here: the page's pool is D1 (its cache is
# pitches_cache_D1.parquet). fc.paths() defaults --level to None, i.e. ALL levels, so
# omitting it would quietly rebuild a different population than the one already on the
# page -- fresh artifacts describing a pool the rest of the page does not use.
LEVEL = "D1"
IO = ["--data", DATA, "--workdir", WORKDIR, "--level", LEVEL]

# (script, extra args). Order is a dependency order, not a preference: every analysis step
# writes a JSON the renderer reads, so the renderer runs last. Scripts that resolve paths
# through fc.paths() need IO; the rest read ffc's module constants and take no arguments.
STEPS = [
    ("coach_model_disagreement.py", IO),
    ("coach_model_paired.py", IO),
    ("coach_page_data.py", IO),
    ("coach_sample_curve.py", []),
    ("coach_location_map.py", []),
    ("coach_feature_importance.py", []),
    ("coach_model_ff_criterion.py", []),
    ("coach_page_render.py", IO),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="run just these scripts (still in dependency order)")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="skip these scripts (use only when you know they are current)")
    args = ap.parse_args()

    steps = [s for s in STEPS if s[0] not in args.skip
             and (args.only is None or s[0] in args.only)]
    print(f"rebuilding {len(steps)} step(s) into {WORKDIR}\n")
    t0 = time.time()
    for i, (script, extra) in enumerate(steps, 1):
        t1 = time.time()
        print(f"[{i}/{len(steps)}] {script} {' '.join(extra)}", flush=True)
        r = subprocess.run([sys.executable, "-u", script, *extra])
        if r.returncode != 0:
            print(f"\nFAILED at {script} (exit {r.returncode}). Stopping so a stale "
                  f"artifact cannot be read next to fresh ones.")
            return r.returncode
        print(f"      done in {time.time() - t1:.0f}s\n", flush=True)
    print(f"ALL DONE in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
