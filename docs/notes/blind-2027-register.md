# Blind 2027 read: register of locked candidates

Every model candidate that stalled on the 2025→2026 pair was locked for one blind read on the
2026→2027 pair. This page lists them in one place so that read cannot be widened, tuned, or read
selectively. Each entry's full registration lives in its own ledger. This page adds no
candidate and changes no rule.

Sign convention: every score and criterion below is expected run value from the pitcher's
perspective, LOWER = BETTER. A working grade correlates POSITIVELY with the criterion.

## Locked

| # | candidate | ledger (PR) | pool and stratum | rule | if it passes / fails |
|---|---|---|---|---|---|
| 1 | Sinker Stuff+, `pooled_all` ridge (raw run value, alpha 10, four-seams + sinkers) | `sinker-cutter-loop-ledger.md` (main) | frozen gate pool, sinkers | frozen gate: P(blend gain > 0) >= 0.95 | draft note comes off / drafted grade comes down (Jack's 2026-09-11 ruling) |
| 2 | Two-season forecast R1: results pooled over two prior seasons, lambda 1 | `two-season-forecast-ledger-2026-09.md` (#34) | returning pitchers, FF, SL, CH; plus the three pooled by inverse variance | P(Delta > 0) >= 0.95 and >= 1 SE, per type and pooled | recommend pooled results in the forecast / nothing changes |
| 3 | NAWARE four-seam Pitching+: z scaled by min(1, rel(n)/rel(200)) | `pitching-plus-naware-ledger-2026-09.md` (#35) | frozen gate pool, under 200 four-seams | P(Delta > 0) >= 0.95 and >= 1 SE | recommend it replace script 08's blend; floor is Jack's call / nothing changes |

## Proposed, awaiting Jack

| # | candidate | ledger (PR) | note |
|---|---|---|---|
| 4 | H3 Location+ for sinker and slider (curveball and changeup read alongside) | `secondary-research-ledger-2026-09.md` (#33) | locks only if Jack says yes; construction frozen as in `coach_location_gate.py` |

## Rules for the read

- **One read per candidate**, on the first 2026→2027 pair built with the same filters as pair C
  (D1, bullpen rows excluded, role relabel). No new candidate joins after 2027 data exists.
- **The only permitted code change is the year shift.** Each harness hard-codes its seasons.
  Before any harness is pointed at 2027, the year shift must reproduce that harness's read-1
  numbers exactly on the 2025→2026 pair. That run is a reproduction check, not a read.
- **Report every locked entry together**, pass or fail, in one results section appended here.
- **Power before the read.** Each entry's minimum detectable effect is recomputed on the 2027
  dry run (simulation, not the analytic approximation, per the NAWARE ledger) and written here
  before any criterion correlation.

## Family-wise error, stated now

Entries 1-3 carry about six confirmatory tests (one, four correlated, one). If all of them were
independent true nulls, the chance that at least one clears P >= 0.95 is about 26%. The four R1
tests are correlated, so the real figure is lower. Entry 4 would add two to four more. The
registered rules are not changed to correct for this; each was fixed in its own ledger before
its read. A single pass among many is reported as a single pass among many. Replication on a
later pair is the correction.
