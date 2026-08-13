# Portal buy-low analysis

Scripts behind the transfer-portal "buy-low board" (2026-07-23) and the
supporting analyses recorded in `../RESULTS.md`. Run order and findings are
documented there under the 2026-07-23 addenda.

- `waste_and_feature_stability.py` — waste-type decomposition (horizontal /
  low / high misses) and year-over-year reliability + validity of each
  location feature inside Location+.
- `build_portal_data.py` — the board data build: coach stats (RA9, K%, BB%,
  whiff%) per pitcher-year for 2025 and 2026 D1, arsenal Pitching+ grades
  (7 pitch types), gap ranking, cohort/matched-pairs/regression proof
  blocks, and the per-pitcher tooltip `detail` payload (within-type
  Stuff+/Location+ plus exact ridge feature contributions). One run
  regenerates `portal_board.json` end to end.
- `arsenal_grade_test.py` / `arsenal_grade_test_2425.py` — arsenal grade vs
  FF-only grade on the 2025->2026 and 2024->2025 D1 pairs (arsenal subsumes
  FF-only in both).
- `extended_types_test.py` — 7-type arsenal (adds Sinker/Cutter/Splitter)
  vs the 4-type baseline; adopted (paired diff +0.03/+0.04, never worse).
- `arsenal_weighting_test.py` — four weighting schemes; mix-neutral
  (quality vs pitch-type average) adopted; learned per-type weights overfit.

## Data stays out of the repo

These scripts read from a local analysis workdir
(`C:\Users\jackdav\stuffplus_replication\`) holding the Level II pitch-level
caches and season CSVs, and they write `portal_board.json` there. The JSON
and the rendered board HTML contain named-pitcher aggregates derived from
licensed TrackMan data, so neither is committed to this public repo — only
the code is. Paths are currently hardcoded to that workdir; parameterize
before running elsewhere.

The published board (Claude artifact, private): the URL and durable HTML
source location are recorded in the project session notes.
