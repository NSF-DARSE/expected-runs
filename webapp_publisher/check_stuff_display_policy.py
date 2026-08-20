"""Does a bundle ship Stuff+ only for the pitch types confirmed to earn one?

THE POLICY THIS ENFORCES. Jack ruled on 2026-08-20 that a per-pitch-type Stuff+ reaches a
coach only where the incremental-validity gate confirms Stuff+ adds something the pitcher's own
recent results do not already say. Confirmed today: FF, Slider, Curveball, ChangeUp. Withheld:
Sinker (29% confident against a 95% bar), Cutter (78%), Splitter (never tested, no contract
entry at all). The confirmed set is NOT hardcoded here -- it is read from
coach_pitching_plus_weights.json's composite_eligible, so re-running the gate moves this
check's target the same way it moves the app's.

WHAT IT IS FOR, and why it does not go green. The live app enforces this at DISPLAY time
(isStuffPlusConfirmed in src/lib/pitchingPlusMix.ts, app repo). The publisher does not: the
bundle carries every type's Stuff+, so the withheld grades are still in the JSON the browser
downloads. Jack was asked on 2026-08-20 whether to gate the publisher too and declined: a coach
is not going to open devtools to dig out a grade, and these are not confidential numbers. THE
GAP IS AN ACCEPTED DECISION, NOT A BACKLOG ITEM. Do not build a publisher-side gate off the
back of this file.

That leaves this check with no green target against a real bundle, by design. Run it to see WHAT
is exposed and to confirm the set has not drifted past the three known types (Sinker, Cutter,
Splitter); do not run it as a pass/fail build step, because it will always fail and a check that
always fails gets switched off or, worse, gets "fixed" by deleting the confirmed types -- which
would suppress exactly what Jack decided to keep. The display gate in the app is the enforcement
point. This is instrumentation.

HISTORY WORTH KEEPING, because it is why this file exists at all. An earlier version of this
check asserted fastball-only, which was the right policy for about four hours on 2026-08-20 and
then was not. It was written while the coach's hold on off-speed Stuff+ was still in force;
Jack replaced that blanket hold with the confirmed-set rule the same day. A check that encodes
a policy rather than reading it goes stale silently and then argues with the product.

TWO FAILURE MODES IT IS BUILT AROUND, both of which look like success:

  1. THE SILENT NO-OP. upload.py's only write is upload_blob(overwrite=True). No delete, no
     prune, no container sync. A publisher that merely STOPS emitting staff_by_type.json
     leaves the old blob serving the old grades, and the publish reports success. Any
     suppression therefore has to OVERWRITE, and the only way to know it happened is to read
     back what is actually there -- which is what container mode does.
  2. THE SECOND SURFACE. The staff board is not the only place these numbers appear. Each
     pitchers/<id>.json arsenal row carries its own "stuff". Fixing the staff file alone moves
     grades out of one view and leaves them in eighteen others.

FAILS CLOSED throughout. An unreadable bundle, an unreadable contract, a missing pitchers
directory, an arsenal row whose type it cannot classify, its own exception -- every one is a
FAILURE. In particular, an unreadable contract confirms NOTHING, so every graded type counts
against the policy. A check that goes quiet when it cannot see is worse than no check, because
someone reads the silence as an all-clear.

Exit 0 = bundle matches the policy. Exit 1 = it does not, or the check could not be sure.

Data rules: Level II. Prints pitch types and COUNTS only -- never a pitcher name and never a
grade value, so the output of a failing run is safe to paste into a ticket or a chat.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# Bundle pitch-type keys mapped onto the contract's keys. This duplicates BUNDLE_TO_CONTRACT in
# the app's src/lib/pitchingPlusMix.ts on purpose: the two live in different languages and
# different repos, and a shared source would couple the publisher's checks to a frontend build.
# If a pitch type is added, both maps need it -- a bundle type absent from this map is treated
# as unconfirmed, so the failure is loud rather than silent.
BUNDLE_TO_CONTRACT = {
    "FF": "FF", "Sinker": "SI", "Cutter": "FC", "Slider": "SL",
    "Curveball": "CB", "ChangeUp": "CH", "Splitter": None,
}


def fail(msg):
    print("  FAIL  %s" % msg)
    return 1


def confirmed_types(contract_path):
    """Bundle pitch types whose Stuff+ may be shown, read from the contract.

    Returns (set_of_bundle_types, problems). Fail-closed: any trouble reading the contract
    yields an EMPTY set, which makes every graded type a violation. That is deliberate -- the
    alternative is a check that quietly clears a bundle because it could not find the rules.
    """
    try:
        with open(contract_path, encoding="utf-8") as fh:
            by_pitch = json.load(fh)["by_pitch"]
    except Exception as e:
        return set(), fail("contract unreadable at %s (%s: %s). Nothing can be confirmed, so "
                           "every graded pitch type below is reported."
                           % (contract_path, type(e).__name__, e))
    if not isinstance(by_pitch, dict):
        # Same guard the generator carries. Without it a malformed contract raises out of here
        # and the reason a coach-facing check failed reads as a Python bug, not a bad contract.
        return set(), fail("contract at %s has by_pitch as %s, expected an object. Nothing can "
                           "be confirmed." % (contract_path, type(by_pitch).__name__))
    ok = set()
    for bundle_type, contract_type in BUNDLE_TO_CONTRACT.items():
        if contract_type is None:
            continue
        entry = by_pitch.get(contract_type)
        # `is True`, not truthiness: a corrupted contract shipping a string or a 1 must read as
        # unconfirmed. Same rule the app's isCompositeEligible uses.
        if isinstance(entry, dict) and entry.get("composite_eligible") is True:
            ok.add(bundle_type)
    return ok, 0


def check_staff_by_type(text, where, allowed):
    problems = 0
    try:
        types = json.loads(text)["types"]
    except Exception as e:
        return fail("%s is unreadable (%s: %s)." % (where, type(e).__name__, e))
    bad = []
    for t in types:
        name = t.get("type")
        if name is None:
            problems += fail("%s has an entry with no 'type'; cannot classify it." % where)
            continue
        rows = t.get("pitchers") or t.get("rows") or []
        graded = sum(1 for r in rows if r.get("stuff") is not None)
        if name not in allowed and graded:
            bad.append((name, graded))
    if bad:
        problems += fail("%s ships Stuff+ for unconfirmed types: %s  (total %d)"
                         % (where, ", ".join("%s=%d" % x for x in bad),
                            sum(n for _, n in bad)))
    else:
        print("  ok    %s ships Stuff+ only for confirmed types (%d entries)"
              % (where, len(types)))
    return problems


def check_pitcher_page(text, where, allowed):
    try:
        arsenal = json.loads(text).get("arsenal", [])
    except Exception as e:
        return [], fail("%s is unreadable (%s: %s)." % (where, type(e).__name__, e))
    found = []
    for a in arsenal:
        name = a.get("type")
        if name is None:
            found.append("UNTYPED")          # unclassifiable cannot be cleared
        elif name not in allowed and a.get("stuff") is not None:
            found.append(name)
    return found, 0


def tally(items, expected, allowed):
    counts, problems, seen = {}, 0, 0
    for where, text in items:
        found, p = check_pitcher_page(text, where, allowed)
        problems += p
        seen += 1
        for name in found:
            counts[name] = counts.get(name, 0) + 1
    if seen != expected:
        problems += fail("read %d of %d pitcher pages; an unread page could hide a grade."
                         % (seen, expected))
    if counts:
        problems += fail("pitcher pages ship Stuff+ for unconfirmed types: %s  "
                         "(total %d across %d pages)"
                         % (", ".join("%s=%d" % (k, v) for k, v in sorted(counts.items())),
                            sum(counts.values()), seen))
    else:
        print("  ok    %d pitcher pages ship Stuff+ only for confirmed types" % seen)
    return problems


def read_files(paths):
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            yield os.path.basename(p), fh.read()


def from_dir(path, allowed):
    problems = 0
    sbt = os.path.join(path, "staff_by_type.json")
    if os.path.exists(sbt):
        with open(sbt, encoding="utf-8") as fh:
            problems += check_staff_by_type(fh.read(), "staff_by_type.json", allowed)
    else:
        # Absent locally proves nothing about what is deployed -- see failure mode 1.
        print("  --    staff_by_type.json not emitted (container check still required)")
    files = sorted(glob.glob(os.path.join(path, "pitchers", "*.json")))
    if not files:
        return problems + fail("no pitchers/ pages under %s; the second surface is unchecked."
                               % path)
    return problems + tally(read_files(files), len(files), allowed)


def container_client(account, container):
    """Prefer the same connection string the publisher uses.

    Not a convenience. The check has to run in the environment that publishes, and that
    environment authenticates with WEBAPP_STORAGE_CONNECTION_STRING. DefaultAzureCredential
    needs the az CLI on PATH, which it is not here (the install is a venv shim called by full
    path), so leaning on it alone makes the check fail closed every time -- and a check that
    always fails gets switched off.
    """
    from azure.storage.blob import ContainerClient
    conn = os.environ.get("WEBAPP_STORAGE_CONNECTION_STRING")
    if conn:
        print("  auth: WEBAPP_STORAGE_CONNECTION_STRING")
        return ContainerClient.from_connection_string(conn, container)
    if not account:
        raise RuntimeError("no WEBAPP_STORAGE_CONNECTION_STRING and no --account given.")
    from azure.identity import DefaultAzureCredential
    print("  auth: Azure AD (needs Storage Blob Data Reader)")
    return ContainerClient("https://%s.blob.core.windows.net" % account, container,
                           credential=DefaultAzureCredential())


def read_blobs(cc, names):
    for n in names:
        yield n, cc.get_blob_client(n).download_blob().readall().decode("utf-8")


def from_container(account, container, allowed):
    cc = container_client(account, container)
    names = [b.name for b in cc.list_blobs()]
    problems = 0
    if "staff_by_type.json" in names:
        blob = cc.get_blob_client("staff_by_type.json")
        # Printed because failure mode 1 is invisible otherwise: a lastModified that did not
        # move means the publish never overwrote this file.
        print("  staff_by_type.json present, lastModified %s"
              % blob.get_blob_properties().last_modified)
        problems += check_staff_by_type(blob.download_blob().readall().decode("utf-8"),
                                        "staff_by_type.json (deployed)", allowed)
    else:
        print("  --    staff_by_type.json is not in the container")
    pages = [n for n in names if n.startswith("pitchers/") and n.endswith(".json")]
    if not pages:
        return problems + fail("container has no pitchers/ pages; second surface unchecked.")
    return problems + tally(read_blobs(cc, pages), len(pages), allowed)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bundle", help="Local bundle directory to check before upload.")
    g.add_argument("--account", nargs="?", const="", default=None,
                   help="Check the deployed container. Uses "
                        "WEBAPP_STORAGE_CONNECTION_STRING when set; otherwise pass the "
                        "storage account name and hold Storage Blob Data Reader.")
    ap.add_argument("--container",
                    help="Blob container to read in --account mode. No default on purpose: a "
                         "wrong container name lists zero blobs and reads as 'nothing "
                         "deployed', which is the one failure this check must never report as "
                         "clean.")
    ap.add_argument("--contract", required=True,
                    help="coach_pitching_plus_weights.json. Its composite_eligible flags ARE "
                         "the policy; an unreadable one confirms nothing.")
    a = ap.parse_args()
    try:
        allowed, problems = confirmed_types(a.contract)
        print("  confirmed types: %s" % (", ".join(sorted(allowed)) or "(none)"))
        if a.bundle:
            problems += from_dir(a.bundle, allowed)
        else:
            if not a.container:
                return fail("--container is required with --account.")
            problems += from_container(a.account or None, a.container, allowed)
    except Exception as e:
        # Fail closed: an exception means the check did not run, which is not a pass.
        return fail("the check itself could not complete (%s: %s)." % (type(e).__name__, e))
    print("")
    if problems:
        print("  EXPOSED IN THE BUNDLE (%d finding%s). The publisher deliberately does not "
              "gate Stuff+ (Jack, 2026-08-20), so this is the accepted state, not a "
              "regression. Enforcement is the app's display gate."
              % (problems, "" if problems == 1 else "s"))
        return 1
    print("  policy met: Stuff+ ships only for confirmed pitch types.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
