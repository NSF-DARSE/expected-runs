"""Is off-speed Stuff+ absent from a bundle -- checked, not assumed.

WHY THIS EXISTS. On 2026-08-17 the coach asked to hold off-speed Stuff+ until the refinement
pass finished. It shipped anyway: staff_by_type.json reached the deployed container the same
afternoon, and every per-pitcher page carried the same grades in its arsenal rows. Nobody
decided that; it rode along with a publish. This script is the check that would have caught it.

TWO FAILURE MODES IT IS BUILT AROUND, both of which look like success:

  1. THE SILENT NO-OP. upload.py's only write is upload_blob(overwrite=True). There is no
     delete, no prune, no container sync. So a publisher that merely STOPS emitting
     staff_by_type.json leaves the old blob serving the old grades, and the publish reports
     success. Suppression therefore has to OVERWRITE the file with a fastball-only one, and
     the only way to know it happened is to read back what is actually there.
  2. THE SECOND SURFACE. The staff board is not the only place these numbers appear. Each
     pitchers/<id>.json arsenal row carries its own "stuff", rendered by ArsenalTable with no
     fastball gate. Fixing the staff file alone moves 45 grades out of one view and leaves
     them in eighteen others.

FAILS CLOSED, and that is the whole design. An unreadable bundle, a missing pitchers
directory, an arsenal row whose pitch type it cannot classify -- every one of those is a
FAILURE, not a pass. A check that goes quiet when it cannot see is worse than no check,
because someone will read the silence as an all-clear.

Reads a local bundle directory (before upload, which is the point) or the deployed container
(after, to confirm the write landed). Container mode needs only read access, which is what the
publishing identity has; deletes require Storage Blob Data Contributor and are deliberately
not attempted here.

Exit 0 = no off-speed Stuff+ found. Exit 1 = found, or could not be sure.

Data rules: Level II. Prints pitch types and COUNTS only -- never a pitcher name and never a
grade value, so the output of a failing run is safe to paste into a ticket or a chat.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

FASTBALL = "FF"


def fail(msg):
    print("  FAIL  %s" % msg)
    return 1


def check_staff_by_type(text, where):
    """staff_by_type.json must carry the fastball and nothing else."""
    problems = 0
    try:
        types = json.loads(text)["types"]
    except Exception as e:
        return fail("%s is unreadable (%s: %s). Cannot confirm the hold."
                    % (where, type(e).__name__, e))
    offspeed = []
    for t in types:
        name = t.get("type")
        if name is None:
            problems += fail("%s has an entry with no 'type'; cannot classify it." % where)
            continue
        rows = t.get("pitchers") or t.get("rows") or []
        graded = sum(1 for r in rows if r.get("stuff") is not None)
        if name != FASTBALL and graded:
            offspeed.append((name, graded))
    if offspeed:
        problems += fail("%s exposes off-speed Stuff+: %s  (total %d)"
                         % (where, ", ".join("%s=%d" % x for x in offspeed),
                            sum(n for _, n in offspeed)))
    else:
        print("  ok    %s carries no off-speed Stuff+ (%d type entries)" % (where, len(types)))
    return problems


def check_pitcher_page(text, where):
    """No non-fastball arsenal row may carry a Stuff+. Returns (findings, problems)."""
    try:
        arsenal = json.loads(text).get("arsenal", [])
    except Exception as e:
        return [], fail("%s is unreadable (%s: %s)." % (where, type(e).__name__, e))
    found = []
    for a in arsenal:
        name = a.get("type")
        if name is None:
            # An unclassifiable row cannot be cleared, so it counts against the hold.
            found.append("UNTYPED")
        elif name != FASTBALL and a.get("stuff") is not None:
            found.append(name)
    return found, 0


def tally(items, expected):
    """Aggregate the per-page results. Counts only -- no names, no values."""
    counts, problems, seen = {}, 0, 0
    for where, text in items:
        found, p = check_pitcher_page(text, where)
        problems += p
        seen += 1
        for name in found:
            counts[name] = counts.get(name, 0) + 1
    if seen != expected:
        problems += fail("read %d of %d pitcher pages; an unread page could hide a grade."
                         % (seen, expected))
    if counts:
        problems += fail("pitcher pages expose off-speed Stuff+: %s  (total %d across %d pages)"
                         % (", ".join("%s=%d" % (k, v) for k, v in sorted(counts.items())),
                            sum(counts.values()), seen))
    else:
        print("  ok    %d pitcher pages carry no off-speed Stuff+" % seen)
    return problems


def read_files(paths):
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            yield os.path.basename(p), fh.read()


def from_dir(path):
    problems = 0
    sbt = os.path.join(path, "staff_by_type.json")
    if os.path.exists(sbt):
        with open(sbt, encoding="utf-8") as fh:
            problems += check_staff_by_type(fh.read(), "staff_by_type.json")
    else:
        # Absent locally is fine ONLY because a local bundle is pre-upload. The container
        # check is what catches the stale-blob case, which this cannot see.
        print("  ok    staff_by_type.json not emitted (container check still required)")
    pdir = os.path.join(path, "pitchers")
    files = sorted(glob.glob(os.path.join(pdir, "*.json")))
    if not files:
        return problems + fail("no pitchers/ pages found under %s. If this bundle is meant to "
                               "have them, the second surface is unchecked." % path)
    return problems + tally(read_files(files), len(files))


def read_blobs(cc, names):
    for n in names:
        yield n, cc.get_blob_client(n).download_blob().readall().decode("utf-8")


def container_client(account, container):
    """Prefer the same connection string the publisher uses.

    Not a convenience. The check has to be runnable in the environment that does the
    publishing, and that environment authenticates with WEBAPP_STORAGE_CONNECTION_STRING.
    DefaultAzureCredential needs the Azure CLI on PATH, which it is not here (the az install
    is a venv shim called by full path), so relying on it alone makes the check fail closed
    every time and a check that always fails gets switched off. --account stays supported for
    an RBAC session; the connection string wins when both are available.
    """
    from azure.storage.blob import ContainerClient
    conn = os.environ.get("WEBAPP_STORAGE_CONNECTION_STRING")
    if conn:
        print("  auth: WEBAPP_STORAGE_CONNECTION_STRING")
        return ContainerClient.from_connection_string(conn, container)
    if not account:
        raise RuntimeError("no WEBAPP_STORAGE_CONNECTION_STRING and no --account given.")
    from azure.identity import AzureCliCredential, DefaultAzureCredential
    print("  auth: Azure AD (needs Storage Blob Data Reader)")
    cred = AzureCliCredential() if os.environ.get("AZURE_CLI_PATH") else DefaultAzureCredential()
    return ContainerClient("https://%s.blob.core.windows.net" % account, container,
                           credential=cred)


def from_container(account, container):
    cc = container_client(account, container)
    names = [b.name for b in cc.list_blobs()]
    problems = 0
    if "staff_by_type.json" in names:
        blob = cc.get_blob_client("staff_by_type.json")
        # lastModified is printed because the silent no-op above is invisible otherwise: a
        # timestamp that did not move means the publish never overwrote this file.
        print("  staff_by_type.json present, lastModified %s"
              % blob.get_blob_properties().last_modified)
        problems += check_staff_by_type(blob.download_blob().readall().decode("utf-8"),
                                        "staff_by_type.json (deployed)")
    else:
        print("  ok    staff_by_type.json is not in the container")
    pages = [n for n in names if n.startswith("pitchers/") and n.endswith(".json")]
    if not pages:
        return problems + fail("container has no pitchers/ pages; the second surface is "
                               "unchecked.")
    return problems + tally(read_blobs(cc, pages), len(pages))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--bundle", help="Local bundle directory to check before upload.")
    g.add_argument("--account", nargs="?", const="", default=None,
                   help="Check the deployed container after upload. Uses "
                        "WEBAPP_STORAGE_CONNECTION_STRING when set; otherwise pass the "
                        "storage account name and hold Storage Blob Data Reader.")
    ap.add_argument("--container", default="bundles")
    a = ap.parse_args()
    try:
        problems = (from_dir(a.bundle) if a.bundle
                    else from_container(a.account or None, a.container))
    except Exception as e:
        # Fail closed: an exception means the check did not run, which is not a pass.
        return fail("the check itself could not complete (%s: %s)." % (type(e).__name__, e))
    print("")
    if problems:
        print("  HOLD IS BREACHED (%d problem%s). Do not publish, and do not report it clean."
              % (problems, "" if problems == 1 else "s"))
        return 1
    print("  hold intact: no off-speed Stuff+ on either surface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
