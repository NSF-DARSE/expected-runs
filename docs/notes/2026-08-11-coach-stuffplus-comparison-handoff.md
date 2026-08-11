# Handoff: coach's ChatGPT Stuff+ workbook, and prep for the meeting with him

Written 2026-08-11 from a VS Code session. Task was analysis and meeting prep, not code. **No source files were changed by this session.** The only commit is this note.

## Odd state to know about before you touch anything

None of these are bugs. Do not "clean them up."

- **A stray untracked file at repo root literally named `-`.** Probably a shell redirect typo from some earlier session. Harmless, but never run `git add .` here; it will get committed along with licensed documents.
- **`CLAUDE.md` is deliberately untracked** and stays that way.
- **Licensed / Level II files sit untracked at repo root on purpose**: `Coach_Linear_Regression_Model.xlsx`, the TrackMan license and renewal `.docx` files, `meeting-2026.07.30.vtt`, `share/`. The repo (NSF-DARSE/expected-runs) is public. These must not be staged.
- **`webapp_publisher/.env.example` is untracked because a permission deny rule blocks agents from staging anything env-shaped.** Jack stages that file himself. Do not work around the rule.
- **`trackman_api/2026/` holds ~6.9 GB of raw TrackMan CSVs inside the repo tree**, guarded by gitignore rules. Do not remove the guard.

## State

- Branch `component-model-framework`, **4 commits ahead of origin** (`c6b2c33`, `c7435c3`, `ef3ef0f`, `1a38113` — none of them mine; a concurrent session produced them). Nothing pushed by me.
- The tree moved during this session: HEAD was `80b7176` at the start and `c6b2c33` by the end. **Another session is actively working on the pitcher page** (`webapp_publisher/schema.py`, `component_model/analysis/15_recent_change_floor.py`, `docs/notes/2026-08-11-pitcher-page-coach-review-handoff.md`). Assume it is still running.
- Uncommitted work of mine: none. The untracked list above predates this session.
- Analysis scratch (throwaway, Level II, not in repo): `…\Temp\claude\c--Users-jackdav-repos-baseball-stuff-plus\e88660f9-…\scratchpad\` holds `coach_games.pkl`, `coach_all.pkl`, `his_ids.txt`.

## Hands off

`webapp_publisher/`, `component_model/analysis/1[45]_*`, and `docs/notes/2026-08-11-pitcher-page-coach-review-handoff.md` belong to the concurrent pitcher-page session. Leave them alone.

## What was established

The pitching coach emailed a ChatGPT transcript proposing a full Stuff+/Whiff+/Strike+/Run Value+/Command+/Pitching+ system, plus his workbook: `C:\Users\jackdav\Downloads\Delaware 2026 Trackman and Bullpen Data 1.xlsx`.

**Profile of that workbook** (verified, not estimated):

- Sheet `Trackman Games`: 19,728 rows, but 4,169 are League `Team` (intrasquad). Real games: **15,559 pitches, 51 games, all Feb 13 – May 16 2026**. The Sept/Oct 2025 dates belong to the intrasquad rows.
- **Delaware is 51.7% of the game pitches.** DEL_BLU 12,211; next team 604. So "100 = D1 average" would mean "100 = half Delaware." ChatGPT raised this limitation itself, then waived it on the false premise that including opponents fixes it.
- 175 opposing pitchers, **median 35 total pitches each**.
- 787 pitcher × pitch-type combinations; **55 clear 50 pitches** (our measured reliability floor), and **only 5 of those 55 are opponents**.
- Whole file contains **1,511 swinging strikes and 2,659 balls in play** — not enough to fit Whiff+ split by type × count × handedness, or Run Value+ at all.
- `TaggedPitchType` is dirty: **`FourSeamFastBall` (4,178) and `Fastball` (1,831) are separate labels**, plus `Splittler`, `Sweeoer`, `SInker`, `Changeup`.
- One season only, so nothing in it can be validated out-of-sample. This is the decisive objection.
- Other sheets are not usable for modeling: `Trackman Bullpens` (2,735 pitches, no hitter, a third `PitchCall = Undefined`), `Trackman Camp Data` (96), `Old Trackman Games` (3,612, Jan–Feb 2025, 3 teams).

**Overlap with our data** (checked against `C:\Users\jackdav\stuffplus_replication\source_2025_2026.csv`):

- **49 of his 51 games are already in our data**, = 14,003 pitches (matches the DEL_BLU 2026 figure already in project memory).
- **His 18 Delaware game pitchers match ours exactly**, both directions, zero name reconciliation needed. (An earlier count of 28 was wrong — it included bullpen/intrasquad rows.)
- The 2 absent games, `20260318-BobHannahStadium-1` and `20260514-BobHannahStadium-2`, exist in the national FTP feed **only as `_unverified` captures**, which the pipeline skips by design across all three seasons. His own export doesn't filter that way. Nothing is missing from our side.

Conclusion: there is nothing to import. His workbook is a 51-game local slice of a feed we hold 7,943 games of.

## Decisions made — do not relitigate

- **Lead the meeting with his file, not with our work.** Open by telling him his 18 pitchers are all gradeable and already loaded, and that the only thing his data lacked was a yardstick. Do not frame it as correcting him; he was apologetic in the email and this has to land as building on what he did.
- **Show exactly two things: the Staff Board (live app) and the portal buy-low board.** The portal board is the proof the grades do something (matched pairs, ~1-run RA9 gap).
- **Hold the Usage Gap Board in reserve.** It reads as the model second-guessing his usage; wrong for a first meeting unless he asks.
- **Do not show or promise Command+, secondary-pitch Location+, Deployment+, or auto-generated development prescriptions.** Reasons are already settled and recorded: intended target is not in the data so Command+ cannot be built; Location+ has no predictive value on secondaries; Deployment+ failed 2026 replication and was withdrawn; within-pitcher tests say these traits are not coaching levers.
- **Fitted weights are settled.** Equal-weight z-blend for Pitching+. Weight fitting has been tried and overfit three separate times. ChatGPT's proposed 45/35/20 "tuned after validation" is exactly the thing that failed.
- **Agreed next deliverable:** hand his workbook back with a sheet appended — his 18 pitchers, per pitch type, with Stuff+ / Location+ / Pitching+ and the sample-size flag, graded against the national D1 population. Now trivial since rosters match. Not yet built.

## Needs Jack, not code

- **Confirm the coach's `@udel.edu` is assigned to the Entra app** for `https://jolly-forest-0894cd60f.7.azurestaticapps.net`, and plan to walk him through sign-in live rather than beforehand.
- **Refresh the bundle** before the meeting so the board is current (publisher env vars are in project memory).
- **Decide the unverified-game rule.** He will eventually notice a game he remembers is missing from a count. Current behavior excludes unverified captures across all seasons. Including them is possible but should be a deliberate call, not a silent one.
- **Meeting is not yet scheduled.** No date set as of this note.
- Optional: a half-page glossary for him (100 = average D1 pitch of that type, higher is better, what the sample-size flag means, and the one line about what we deliberately do not claim).
