"""The Command+ section of the bundle: `command_pairs.json`.

The pairs come from component_model/analysis/command_pairs.py; this module only
decides where its inputs come from and refuses a half-configured run.

Inputs (all optional, so a publish without Command+ inputs still ships):
  COMMAND_TAGS_DIR        local mirror of the api's `session-tags` container
                          (<date>/<pitcherId>/<sessionId>.json)
  COMMAND_PULL_TAGS=1     mirror that container into COMMAND_TAGS_DIR (or
                          <workdir>/session_tags) first, read-only, using
                          WEBAPP_STORAGE_CONNECTION_STRING
  COMMAND_TAGS_CONTAINER  default "session-tags"
  COMMAND_PITCH_ROOTS     os.pathsep-separated roots of <year>/<month>/<day> TrackMan
                          practice CSVs carrying UTCDateTime

With no tag source the file still ships, empty and with a note, so the app can tell
"no tagged pens yet" apart from "a bundle published before Command+ existed".
Tags without a pitch source is an error: shipping every session as unjoined would
read in the app as a tagging failure when it is a publisher setting.
"""
from __future__ import annotations

import os
import pathlib
import sys

_ANALYSIS = pathlib.Path(__file__).resolve().parents[1] / "component_model" / "analysis"
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

import command_pairs as cp  # noqa: E402


def resolve_tags_dir(tags_dir: str | None, workdir: str, pull: bool) -> str | None:
    if tags_dir:
        return tags_dir
    if pull:
        return str(pathlib.Path(workdir) / "session_tags")
    return None


def build_command_section(*, tags_dir: str | None, pitch_roots: str | None,
                          pull: bool = False, connection_string: str | None = None,
                          container: str = "session-tags") -> dict:
    if pull:
        if not connection_string:
            raise ValueError("COMMAND_PULL_TAGS is set but WEBAPP_STORAGE_CONNECTION_STRING is not")
        if not tags_dir:
            raise ValueError("COMMAND_PULL_TAGS needs a destination directory")
        os.makedirs(tags_dir, exist_ok=True)
        n = cp.pull_tag_sessions(connection_string, container, tags_dir)
        print(f"[command] mirrored {n} tag sessions from container {container!r}")

    if not tags_dir:
        print("[command] no tag source configured (COMMAND_TAGS_DIR / COMMAND_PULL_TAGS); "
              "shipping an empty command_pairs.json")
        return cp.empty_payload("no tag source configured for this publish")

    sessions = cp.read_tag_sessions(tags_dir)
    if not sessions:
        print(f"[command] no tagged sessions under the tag source; shipping an empty section")
        return cp.empty_payload("no tagged bullpen sessions found")
    if not pitch_roots:
        raise ValueError(f"{len(sessions)} tagged sessions found but COMMAND_PITCH_ROOTS is not "
                         "set, so none of them can be joined to TrackMan pitches")

    payload = cp.build_command_pairs(sessions, pitch_roots.split(os.pathsep))
    for s in payload["sessions"]:
        if s["status"] != "joined":
            # Session id carries a date and a TrackMan id, never a name.
            print(f"[command] unjoined {s['sessionId']}: {s['reason']}")
    print(f"[command] {payload['nJoined']}/{payload['nSessions']} sessions joined, "
          f"{sum(len(s['pairs']) for s in payload['sessions'])} scored pitches")
    return payload
