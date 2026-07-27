# Reliability Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the year-over-year unpredictability of pitcher success against the fair criterion into measurement noise, true year-to-year change, and skill that Pitching+ misses — and report the missing-skill share as a number.

**Architecture:** A pure-function estimator module (`variance_components.py`) holding the method-of-moments math, unit-tested against synthetic data with known variance components; a driver script (`12_reliability_decomposition.py`) that assembles a three-season D1 panel from existing pitch caches, computes Part A (within-season game-parity split-half reliability) and Part B (three-bucket variance decomposition plus the missing-skill residual), and cross-checks A against B. No new dependencies: `statsmodels` is **not** installed and is **not** needed — the method-of-moments decomposition is closed-form in numpy and more transparent than a mixed-model fit.

**Tech Stack:** Python, pandas, numpy, scipy, scikit-learn (via `fair_criterion.py`), pytest 9.1.1.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-27-reliability-decomposition-design.md`. Scope is Part A + Part B only; Part C (incremental-validity horse race) is out of scope.
- **Success metric:** the fair criterion `xT` (defense/luck-stripped expected run value). `adjT` reported as a secondary robustness row only.
- **Sign conventions** (`fair_criterion.py` docstring; an inverted narrative shipped once already): `Target`, `xT`, `adjT`, `ridge_pred`, and all location-map values are expected runs from the pitcher's perspective, **LOWER = BETTER**. A predictor is consistently oriented when it correlates **POSITIVELY** with a future run-value criterion. Variance decomposition is orientation-free, but the missing-skill regression coefficients are not — the script must assert their sign.
- **Never modify `fair_criterion.py`.** It is a fixed reference; changing it invalidates every committed comparison. Compose its functions only.
- **Level filter: D1 only.** The feed's D2 share doubled from 2024 to 2026, so all-levels numbers are composition-diluted. All-levels is unavailable anyway (see Task 2) — state that as a scope note, do not fake it.
- **Four-seam fastballs only.** Per-type repetition is out of scope.
- **Data rules (licensed TrackMan, Level II):** read from the caches / `STUFFPLUS_DATA`, write only under `STUFFPLUS_WORKDIR`. **Never commit derived TrackMan values, per-pitcher output, or cache contents.** Aggregates only in `RESULTS.md`.
- **Minimum sample:** 100+ qualifying FF per pitcher-season (mirrors `fc.PANEL_MIN_FF`); 25+ FF per half in Part A.
- **Commit style:** match recent history (imperative, lowercase-after-prefix not required; e.g. `Add reliability decomposition: noise vs missing skill`). No AI attribution footers, no em dashes in commit messages.

---

### Task 1: Variance-components estimator module

The math lives in its own module with no I/O so it can be tested against synthetic data where the true components are known. This is the load-bearing piece; if it is wrong, every number downstream is wrong and there is no way to notice from real data alone.

**Files:**
- Create: `component_model/analysis/variance_components.py`
- Create: `component_model/analysis/tests/test_variance_components.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (numpy/pandas only).
- Produces, relied on by Tasks 2 and 3:
  - `pooled_within_variance(values: pd.Series, groups: pd.Series) -> float`
  - `variance_components(tab: pd.DataFrame, sigma2_w: float, value_col: str = "mean") -> dict`
    - `tab` has one row per pitcher-season with columns `pitcher`, `season`, `value_col`, `n`.
    - returns dict with keys: `s2_stable`, `s2_drift`, `s2_noise`, `total`,
      `share_stable`, `share_drift`, `share_noise`, `rho_within_pred`,
      `r_across_pred`, `persistence`, `pairs` (dict keyed by `(season_a, season_b)` -> `(cov, n)`), `n_pitcher_seasons`, `n_pitchers`.
  - `spearman_brown(r: float) -> float`

- [ ] **Step 1: Write the failing tests**

Create `component_model/analysis/tests/test_variance_components.py`:

```python
"""Synthetic-recovery tests for the variance-components estimator.

The estimator's job is to split observed pitcher-season variance into stable
skill, true drift, and sampling noise. On real data there is no way to check
that it did so correctly, so it is verified here against simulated panels whose
true components are known by construction.
"""
import numpy as np
import pandas as pd
import pytest

import variance_components as vc


def simulate_panel(n_pitchers=1200, seasons=(2024, 2025, 2026), s2_stable=4e-4,
                   s2_drift=2e-4, sigma2_w=0.105, n_lo=100, n_hi=400, seed=7):
    """Panel with known components. Returns (pitcher-season table, truth dict).

    y_it = a_i + b_it + e_it, var(e_it) = sigma2_w / n_it. Scales mimic real xT:
    pitch-level sd ~0.32 (var ~0.105), so with ~200 FF the noise term is the
    same order as stable skill -- the regime that makes this question hard.
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0, np.sqrt(s2_stable), n_pitchers)
    rows = []
    for i in range(n_pitchers):
        for s in seasons:
            n = int(rng.integers(n_lo, n_hi))
            b = rng.normal(0, np.sqrt(s2_drift))
            e = rng.normal(0, np.sqrt(sigma2_w / n))
            rows.append({"pitcher": i, "season": s, "mean": a[i] + b + e, "n": n})
    truth = {"s2_stable": s2_stable, "s2_drift": s2_drift, "sigma2_w": sigma2_w}
    return pd.DataFrame(rows), truth


def test_pooled_within_variance_recovers_pitch_level_variance():
    rng = np.random.default_rng(3)
    frames = []
    for g in range(400):
        n = int(rng.integers(100, 400))
        frames.append(pd.DataFrame({"v": rng.normal(rng.normal(0, 0.02), 0.32, n),
                                    "g": g}))
    df = pd.concat(frames, ignore_index=True)
    got = vc.pooled_within_variance(df["v"], df["g"])
    assert got == pytest.approx(0.32 ** 2, rel=0.05)


def test_variance_components_recovers_known_truth():
    tab, truth = simulate_panel()
    out = vc.variance_components(tab, truth["sigma2_w"])
    assert out["s2_stable"] == pytest.approx(truth["s2_stable"], rel=0.15)
    assert out["s2_drift"] == pytest.approx(truth["s2_drift"], rel=0.20)
    assert out["s2_noise"] > 0


def test_shares_sum_to_one():
    tab, truth = simulate_panel()
    out = vc.variance_components(tab, truth["sigma2_w"])
    total = out["share_stable"] + out["share_drift"] + out["share_noise"]
    assert total == pytest.approx(1.0, abs=1e-9)


def test_zero_drift_is_detected():
    """With no true season-to-season change, drift must estimate near zero."""
    tab, truth = simulate_panel(s2_drift=0.0, seed=11)
    out = vc.variance_components(tab, truth["sigma2_w"])
    assert abs(out["s2_drift"]) < 0.25 * out["s2_stable"]


def test_all_noise_panel_has_no_stable_skill():
    """Pure sampling noise: stable and drift both estimate near zero."""
    tab, truth = simulate_panel(s2_stable=0.0, s2_drift=0.0, seed=13)
    out = vc.variance_components(tab, truth["sigma2_w"])
    assert abs(out["s2_stable"]) < 0.15 * out["s2_noise"]
    assert out["share_noise"] > 0.8


def test_derived_ratios_match_definitions():
    tab, truth = simulate_panel()
    out = vc.variance_components(tab, truth["sigma2_w"])
    assert out["rho_within_pred"] == pytest.approx(
        (out["s2_stable"] + out["s2_drift"]) / out["total"], abs=1e-9)
    assert out["r_across_pred"] == pytest.approx(out["s2_stable"] / out["total"], abs=1e-9)
    assert out["persistence"] == pytest.approx(
        out["s2_stable"] / (out["s2_stable"] + out["s2_drift"]), abs=1e-9)


def test_pairwise_covariances_reported_per_lag():
    tab, truth = simulate_panel()
    out = vc.variance_components(tab, truth["sigma2_w"])
    assert set(out["pairs"]) == {(2024, 2025), (2024, 2026), (2025, 2026)}
    for cov, n in out["pairs"].values():
        assert n > 1000
        assert cov == pytest.approx(truth["s2_stable"], rel=0.30)


def test_two_season_panel_still_identified():
    tab, truth = simulate_panel(seasons=(2024, 2025))
    out = vc.variance_components(tab, truth["sigma2_w"])
    assert out["s2_stable"] == pytest.approx(truth["s2_stable"], rel=0.20)


def test_spearman_brown_doubles_length():
    assert vc.spearman_brown(0.5) == pytest.approx(2 / 3)
    assert vc.spearman_brown(0.0) == pytest.approx(0.0)
    assert vc.spearman_brown(1.0) == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd component_model/analysis && python -m pytest tests/test_variance_components.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'variance_components'`

- [ ] **Step 3: Write the implementation**

Create `component_model/analysis/variance_components.py`:

```python
"""Method-of-moments decomposition of pitcher-season variance.

Splits the observed spread in a pitcher-season mean into three buckets:

    y_it = mu_t + a_i + b_it + e_it

    a_i    stable skill            var = s2_stable   (persists across seasons)
    b_it   true year-to-year change var = s2_drift   (real, but unknowable in advance)
    e_it   sampling noise           var = sigma2_w / n_it

Identification: a_i is the only term shared across a pitcher's seasons, so the
cross-season covariance of centered season means estimates s2_stable directly.
sigma2_w comes from pitch-level scatter within a pitcher-season, so the noise
term is known rather than assumed. s2_drift is what is left over.

Closed-form in numpy on purpose: a mixed-model fit would need statsmodels (not a
project dependency) and would hide the one step worth seeing, which is that
stable skill IS the cross-season covariance.

ORIENTATION: this module operates on variances and covariances, which are
sign-free, so the lower-is-better run-value convention does not apply to
anything returned here. Callers regressing predictors against the criterion must
check coefficient signs themselves (see fair_criterion.py docstring).
"""
import numpy as np
import pandas as pd

MIN_PAIR_N = 30


def pooled_within_variance(values, groups):
    """sigma2_w: pooled within-group variance of pitch-level values.

    Pooled across pitcher-seasons rather than averaged, so that groups with more
    pitches carry more weight in the estimate.
    """
    df = pd.DataFrame({"v": np.asarray(values, dtype=float),
                       "g": np.asarray(groups)}).dropna()
    grp = df.groupby("g")["v"]
    means = grp.transform("mean")
    ss = float(((df["v"] - means) ** 2).sum())
    dof = float(len(df) - grp.ngroups)
    if dof <= 0:
        return float("nan")
    return ss / dof


def spearman_brown(r):
    """Reliability of a full-length measure from its split-half correlation."""
    return 2.0 * r / (1.0 + r) if (1.0 + r) != 0 else float("nan")


def variance_components(tab, sigma2_w, value_col="mean"):
    """Split pitcher-season variance into stable / drift / noise.

    tab: one row per pitcher-season, columns 'pitcher', 'season', value_col, 'n'.
    sigma2_w: pooled within-pitcher-season pitch-level variance.

    Season means are removed first, so environment-level shifts (feed growth,
    level-mix drift, scoring environment) never land in any bucket.

    s2_drift is returned RAW and may come out slightly negative when the true
    value is near zero; that is honest sampling error in the estimator, not a
    bug. Do not clamp it silently -- callers print it as-is and note it.
    """
    t = tab[["pitcher", "season", value_col, "n"]].dropna().copy()
    t["c"] = t[value_col] - t.groupby("season")[value_col].transform("mean")

    wide = t.pivot_table(index="pitcher", columns="season", values="c")
    seasons = sorted(wide.columns)
    pairs = {}
    for i, s1 in enumerate(seasons):
        for s2 in seasons[i + 1:]:
            both = wide[[s1, s2]].dropna()
            if len(both) < MIN_PAIR_N:
                continue
            cov = float(np.cov(both[s1].values, both[s2].values, ddof=1)[0, 1])
            pairs[(s1, s2)] = (cov, int(len(both)))
    if not pairs:
        raise ValueError("no season pair has enough shared pitchers to identify "
                         "stable skill; need %d+ pitchers in two seasons" % MIN_PAIR_N)

    # Precision-weight the pairwise covariances by the number of shared pitchers.
    den = sum(n for _, n in pairs.values())
    s2_stable = sum(cov * n for cov, n in pairs.values()) / den

    s2_noise = float(sigma2_w * (1.0 / t["n"]).mean())
    total_obs = float(t["c"].var(ddof=1))
    s2_drift = total_obs - s2_stable - s2_noise

    total = s2_stable + s2_drift + s2_noise
    return {
        "s2_stable": s2_stable,
        "s2_drift": s2_drift,
        "s2_noise": s2_noise,
        "total": total,
        "total_observed": total_obs,
        "share_stable": s2_stable / total,
        "share_drift": s2_drift / total,
        "share_noise": s2_noise / total,
        "rho_within_pred": (s2_stable + s2_drift) / total,
        "r_across_pred": s2_stable / total,
        "persistence": s2_stable / (s2_stable + s2_drift),
        "pairs": pairs,
        "n_pitcher_seasons": int(len(t)),
        "n_pitchers": int(t["pitcher"].nunique()),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd component_model/analysis && python -m pytest tests/test_variance_components.py -v`
Expected: PASS, 9 passed. If `test_variance_components_recovers_known_truth` fails on tolerance rather than sign/scale, do **not** loosen the tolerance — re-derive the moment equations first.

- [ ] **Step 5: Commit**

```bash
git add component_model/analysis/variance_components.py component_model/analysis/tests/test_variance_components.py
git commit -m "Add variance-components estimator with synthetic-recovery tests"
```

---

### Task 2: Three-season panel + Part A within-season reliability

Assembles the panel and answers Part A: is the criterion noisy (bucket 1), or are pitchers stable within a year and drifting across years (bucket 2)?

**Data sourcing — read this before writing code.** Verified on disk 2026-07-27:

- `Final_Target_Calc_2109.csv` in the repo root is **STALE AND UNUSABLE**: 52 columns, no `PitchUID` (dedup), no `ExitSpeed`/`Angle` (xT), no `GameID` (Part A). It predates the regeneration. Do not use it.
- The G: Drive Project Share (`Final_Target_Calc_1535.csv`) is **not mounted**.
- Usable source: the D1 pitch caches from the 2026 replication run, which carry all 36 `fc.USECOLS` plus `year`/`is_ff`/`is_lhp`/`is_lhb`/`is_inplay`:
  - `C:\Users\jackdav\stuffplus_replication\workdir_2425_d1\pitches_cache_D1.parquet` — real 2024 (1,265,378 rows) + 2025 (1,550,074), all D1
  - `C:\Users\jackdav\stuffplus_replication\workdir_2526_d1\pitches_cache_2025_2026_D1.parquet` — real 2025 (1,550,304) + 2026 (1,749,078), all D1
- **CRITICAL: the second cache has ROLE-RELABELED years.** Its `year` column reads 2024 for real 2025 and 2025 for real 2026 (`fc.load_pitches` relabels before caching). `Date` preserves the true year. The panel builder MUST recompute `year` from `Date`. Skipping this silently corrupts every number.
- The two caches disagree on 2025 by 230 rows (1,550,074 vs 1,550,304) because each deduped `PitchUID` within its own file. Concat-then-dedup across both resolves it deterministically; print the final per-season counts so the resolution is visible.
- **All-levels is unavailable**: the only all-levels 2024 frame (`workdir_2425_check/pitches_cache.parquet`) lacks the `Level` column. D1 is the headline call regardless. Record this as a scope note; do not substitute an unverified frame.

**Files:**
- Create: `component_model/analysis/12_reliability_decomposition.py`

**Interfaces:**
- Consumes: `variance_components` module from Task 1 (`spearman_brown`); `fair_criterion` as `fc` (`add_xt`, `add_adjusted`, `stuff_ridge`, `add_loc_bins`, `PooledLocationMap`, `USECOLS`, `PANEL_MIN_FF`, `R`).
- Produces, relied on by Task 3:
  - `cli() -> argparse.Namespace` with `.caches` (list of parquet paths), `.workdir`, `.min_ff`, `.min_half`, `.boot`
  - `load_seasons(paths: list[str]) -> pd.DataFrame` — deduped 3-season pitch frame with TRUE `year`
  - `build_panel(args) -> tuple[pd.DataFrame, pd.DataFrame]` — `(ff, tab)` where `ff` is qualifying FF pitches carrying `xT`, `adjT`, `ridge_pred`, `loc`, `GameID`, `half`; `tab` is the pitcher-season table with columns `pitcher`, `season`, `mean` (mean xT), `mean_adjT`, `n`, `stuff`, `loc`
  - `add_game_parity(ff: pd.DataFrame) -> pd.DataFrame` — adds `half` (0/1) by game parity within pitcher-season
  - `part_a(ff, args) -> dict[int, dict]` — per-season `{"rho_half", "rho_full", "n"}` for xT, plus a `ridge_pred` anchor row

- [ ] **Step 1: Write the script through Part A**

Create `component_model/analysis/12_reliability_decomposition.py`:

```python
"""12: How much of year-over-year unpredictability is noise vs missing skill?

Splits the spread in pitcher-season fair-criterion performance three ways:
  bucket 1  measurement noise  -- a college season is too few pitches
  bucket 2  true drift         -- pitchers really change year to year
  bucket 3  missing skill      -- persistent talent Pitching+ does not encode
Only bucket 3 is a Pitching+ problem. Buckets 1 and 2 set the ceiling that no
static physical model can beat.

Part A  within-season split-half reliability of xT, split by GAME parity and
        Spearman-Brown corrected. Separates bucket 1 from bucket 2, which the
        across-season number alone cannot do.
Part B  method-of-moments variance decomposition over the three-season panel,
        then the same fit with physical Stuff+/Location+ added; the stable
        variance that survives is missing skill.

Game parity, not pitch parity: consecutive pitches in one game share batter,
park, umpire, and day effects, so a pitch-parity split (script 06) leaves that
shared variance in BOTH halves and overstates reliability. Expect a lower and
more honest number here.

Criterion is xT (defense/luck-stripped expected runs, LOWER = BETTER); adjT
prints as a robustness row. Four-seam, D1, 2024-2026.

Input: pitch-cache parquets written by fc.load_pitches (--caches). Regenerate
them by running script 01 once per year pair with --level D1 if absent. Years
are re-derived from Date, so role-relabeled caches are safe to pass.
Writes nothing outside --workdir. Never commit output: Level II per-pitcher data.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fair_criterion as fc
import variance_components as vc

SEASONS = [2024, 2025, 2026]
TRAIN_YEAR = 2024


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", default=os.environ.get("STUFFPLUS_CACHES"),
                    help="Comma-separated pitch-cache parquet paths. Years are "
                         "re-derived from Date, so relabeled caches are fine.")
    ap.add_argument("--workdir", default=os.environ.get("STUFFPLUS_WORKDIR"))
    ap.add_argument("--min-ff", type=int, default=fc.PANEL_MIN_FF)
    ap.add_argument("--min-half", type=int, default=25)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    if not args.caches or not args.workdir:
        sys.exit("Set --caches (comma-separated pitch-cache parquets) and "
                 "--workdir (outside the repo), or STUFFPLUS_CACHES / "
                 "STUFFPLUS_WORKDIR.")
    args.caches = [p.strip() for p in args.caches.split(",") if p.strip()]
    os.makedirs(args.workdir, exist_ok=True)
    return args


def load_seasons(paths):
    """Concat pitch caches -> deduped frame with TRUE calendar years.

    fc.load_pitches ROLE-RELABELS year for non-default pairs (a 2025-2026 cache
    stores year=2024 for real 2025), so year is recomputed from Date here. Dedup
    runs across the concatenation because each cache deduped only within itself.
    """
    frames = []
    for p in paths:
        d = pd.read_parquet(p)
        print(f"  read {os.path.basename(p)}: {len(d):,} rows")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["year"] = pd.to_datetime(df["Date"], errors="coerce").dt.year
    before = len(df)
    df = df.dropna(subset=["PitchUID"]).drop_duplicates(subset="PitchUID", keep="first")
    df = df[df["year"].isin(SEASONS)].copy()
    df["year"] = df["year"].astype(int)
    print(f"  concat {before:,} -> deduped, in-scope {len(df):,} rows")
    print("  rows by TRUE year: %s" % df["year"].value_counts().sort_index().to_dict())
    if "Level" in df.columns:
        print("  levels present: %s" % df["Level"].value_counts().to_dict())
    return df


def add_game_parity(ff):
    """half = parity of the game's chronological index within pitcher-season.

    GameID begins yyyymmdd, so sorting it orders games in time. Alternating
    whole games keeps the halves balanced while making them independent of
    within-game shared effects.
    """
    keys = ff[["PitcherId", "year", "GameID"]].drop_duplicates()
    keys = keys.sort_values(["PitcherId", "year", "GameID"])
    keys["half"] = keys.groupby(["PitcherId", "year"]).cumcount() % 2
    return ff.merge(keys, on=["PitcherId", "year", "GameID"], how="left")


def build_panel(args):
    """Three-season qualifying-FF frame plus the pitcher-season table.

    xT is fit ONCE on all three seasons pooled, so a given EV/LA outcome maps to
    the same run value in every season -- required for cross-season comparability
    (the spec's frozen reference vintage). The Stuff+ ridge and the location map
    are trained on TRAIN_YEAR only, matching the suite's fixed references.
    """
    df = load_seasons(args.caches)
    fc.add_xt(df)
    fc.add_adjusted(df)
    ff = fc.stuff_ridge(df)

    ff = ff[ff["xT"].notna()].copy()
    fc.add_loc_bins(ff)
    lmap = fc.PooledLocationMap(ff[(ff["year"] == TRAIN_YEAR) & ff["xT"].notna()])
    ff["loc"] = lmap.apply(ff)

    n_by = ff.groupby(["PitcherId", "year"]).size().rename("n_ff")
    ok = n_by[n_by >= args.min_ff].reset_index()[["PitcherId", "year"]]
    ff = ff.merge(ok, on=["PitcherId", "year"], how="inner")
    # A pitcher needs two qualified seasons for cross-season covariance to exist.
    seasons_per = ff.groupby("PitcherId")["year"].nunique()
    ff = ff[ff["PitcherId"].isin(seasons_per[seasons_per >= 2].index)].copy()
    ff = add_game_parity(ff)

    g = ff.groupby(["PitcherId", "year"])
    tab = pd.DataFrame({
        "mean": g["xT"].mean(),
        "mean_adjT": g["adjT"].mean(),
        "n": g["xT"].size(),
        "stuff": g["ridge_pred"].mean(),
        "loc": g["loc"].mean(),
    }).reset_index().rename(columns={"PitcherId": "pitcher", "year": "season"})

    print(f"\nPANEL: {tab['pitcher'].nunique()} pitchers, {len(tab)} pitcher-seasons "
          f"({args.min_ff}+ FF, 2+ qualified seasons, D1)")
    print("  qualified pitcher-seasons by year: %s"
          % tab["season"].value_counts().sort_index().to_dict())
    print("  median FF per pitcher-season: %.0f" % tab["n"].median())
    return ff, tab


def part_a(ff, args):
    """Within-season game-parity split-half reliability, Spearman-Brown corrected."""
    print("\n" + "=" * 78)
    print("PART A -- WITHIN-SEASON RELIABILITY (game parity, Spearman-Brown)")
    print("=" * 78)
    print("Measures bucket 1 alone: same pitcher, same season, different games.")
    print(f"{'season':<8}{'metric':<12}{'pitchers':>9}{'mean FF':>9}"
          f"{'half r':>9}{'SB full':>9}")
    out = {}
    for col, label in [("xT", "xT"), ("ridge_pred", "Stuff+ anchor")]:
        for year in SEASONS:
            sub = ff[ff["year"] == year]
            piv = sub.groupby(["PitcherId", "half"])[col].agg(["mean", "size"]).unstack("half")
            piv = piv.dropna()
            if piv.empty:
                continue
            piv.columns = ["mA", "mB", "nA", "nB"]
            piv = piv[(piv["nA"] >= args.min_half) & (piv["nB"] >= args.min_half)]
            if len(piv) < 30:
                print(f"{year:<8}{label:<12}{len(piv):>9}  too few pitchers, skipped")
                continue
            r = float(pearsonr(piv["mA"], piv["mB"])[0])
            sb = vc.spearman_brown(r)
            n_mean = float((piv["nA"] + piv["nB"]).mean())
            print(f"{year:<8}{label:<12}{len(piv):>9}{n_mean:>9.0f}{r:>9.3f}{sb:>9.3f}")
            if col == "xT":
                out[year] = {"rho_half": r, "rho_full": sb, "n": len(piv),
                             "mean_ff": n_mean}
    print("\nNote: script 06 splits by PITCH parity, which shares within-game")
    print("effects across both halves and reads higher. Game parity is the")
    print("honest unit for a noise estimate.")
    return out


if __name__ == "__main__":
    args = cli()
    ff, tab = build_panel(args)
    a_out = part_a(ff, args)
```

- [ ] **Step 2: Verify the panel builds and Part A runs**

```bash
cd component_model/analysis
python 12_reliability_decomposition.py \
  --caches "C:/Users/jackdav/stuffplus_replication/workdir_2425_d1/pitches_cache_D1.parquet,C:/Users/jackdav/stuffplus_replication/workdir_2526_d1/pitches_cache_2025_2026_D1.parquet" \
  --workdir "C:/Users/jackdav/stuffplus_replication/workdir_decomp"
```

Expected (takes several minutes; ~4.5M rows):
- `rows by TRUE year` shows all three of 2024, 2025, 2026 with roughly 1.26M / 1.55M / 1.75M. **If only two years appear, the un-relabeling is broken — stop and fix.**
- `levels present` shows `D1` only.
- Panel is a few hundred pitchers with 2+ qualified seasons.
- Part A prints three `xT` rows and three `Stuff+ anchor` rows. The Stuff+ anchor `SB full` must be high (~0.9); it is nearly pure physics, so a low value means the split or the panel is wrong.

- [ ] **Step 3: Sanity-check Part A against the known ladder**

`xT` `SB full` should land above the across-season figure (~0.27 to 0.30 from RESULTS.md) but well below the Stuff+ anchor. Two readings, both valid outcomes to report:
- close to ~0.30 → bucket 1 dominates; the metric is just noisy
- much higher → bucket 2 dominates; pitchers drift and no static model beats that ceiling

Record the actual numbers; do not tune anything to land in either regime.

- [ ] **Step 4: Commit**

```bash
git add component_model/analysis/12_reliability_decomposition.py
git commit -m "Add three-season panel and within-season reliability for the decomposition"
```

---

### Task 3: Part B — buckets, missing skill, bootstrap, A/B consistency

**Files:**
- Modify: `component_model/analysis/12_reliability_decomposition.py` (append `part_b`, `missing_skill`, `cluster_bootstrap`, `consistency`; extend `__main__`)

**Interfaces:**
- Consumes: `vc.pooled_within_variance`, `vc.variance_components` (Task 1); `build_panel`, `part_a` (Task 2).
- Produces: printed report only. No new callable relied on by later tasks.

- [ ] **Step 1: Append Part B to the script**

Add to `component_model/analysis/12_reliability_decomposition.py`, above `if __name__ == "__main__":`:

```python
def residualize(tab, cols):
    """Season-center the criterion, then project out physical predictors.

    Returns (residual table shaped for vc.variance_components, coefficients).
    Predictors are z-scored so coefficients are comparable and sign-readable.

    ORIENTATION GATE: stuff/loc are expected runs (lower = better) and so is the
    criterion, so a correctly oriented predictor gets a POSITIVE coefficient. An
    inverted trait narrative shipped once in this project before review caught
    it; the caller prints these signs.
    """
    t = tab.copy()
    t["c"] = t["mean"] - t.groupby("season")["mean"].transform("mean")
    X = np.column_stack([fc.z(t[c]).values for c in cols] + [np.ones(len(t))])
    beta, *_ = np.linalg.lstsq(X, t["c"].values, rcond=None)
    t["resid"] = t["c"].values - X @ beta
    out = t[["pitcher", "season", "n"]].copy()
    out["mean"] = t["resid"].values
    return out, dict(zip(cols, beta[:len(cols)]))


def part_b(ff, tab, sigma2_w, args):
    """Three-bucket decomposition, then the same fit net of physical Pitching+."""
    print("\n" + "=" * 78)
    print("PART B -- VARIANCE DECOMPOSITION")
    print("=" * 78)
    print(f"pitch-level within-pitcher-season variance of xT (sigma2_w) = {sigma2_w:.5f}"
          f"   (sd = {np.sqrt(sigma2_w):.3f})")

    base = vc.variance_components(tab, sigma2_w)
    print(f"\nBUCKET SHARES of single-season observed variance in mean xT "
          f"(n={base['n_pitchers']} pitchers, {base['n_pitcher_seasons']} pitcher-seasons):")
    print(f"  bucket 1  measurement noise   {base['share_noise']:>7.1%}   "
          f"(var {base['s2_noise']:.6f})")
    print(f"  bucket 2  true drift          {base['share_drift']:>7.1%}   "
          f"(var {base['s2_drift']:.6f})")
    print(f"  stable skill                  {base['share_stable']:>7.1%}   "
          f"(var {base['s2_stable']:.6f})")
    if base["s2_drift"] < 0:
        print("  NOTE: drift estimated slightly negative -- consistent with a true")
        print("        value near zero. Reported raw, not clamped.")
    print("\n  stable-skill covariance by season pair (lag diagnostic):")
    for (s1, s2), (cov, n) in sorted(base["pairs"].items()):
        lag = s2 - s1
        print(f"    {s1}-{s2}  lag {lag}   cov={cov:.6f}   n={n}")
    print("    If lag-2 covariance sits below lag-1, part of what looks stable")
    print("    at one year of separation is slow drift, not permanent talent.")

    print("\nMISSING SKILL -- stable variance surviving physical Pitching+:")
    print("  Results-based predictors are deliberately EXCLUDED: explaining a")
    print("  results-based criterion with a results-based predictor would be")
    print("  circular. Physical traits only (Stuff+ ridge, pooled Location+).")
    print(f"{'model':<28}{'stable var':>12}{'captured':>10}{'MISSING':>10}"
          f"{'of total':>10}")
    print(f"{'none (raw criterion)':<28}{base['s2_stable']:>12.6f}"
          f"{0.0:>10.1%}{1.0:>10.1%}{base['share_stable']:>10.1%}")
    variants = [("Stuff+ only", ["stuff"]),
                ("Location+ only", ["loc"]),
                ("Stuff+ and Location+", ["stuff", "loc"])]
    results = {}
    for label, cols in variants:
        rtab, coefs = residualize(tab, cols)
        out = vc.variance_components(rtab, sigma2_w)
        captured = 1.0 - out["s2_stable"] / base["s2_stable"]
        print(f"{label:<28}{out['s2_stable']:>12.6f}{captured:>10.1%}"
              f"{1 - captured:>10.1%}{out['s2_stable'] / base['total']:>10.1%}")
        results[label] = {"out": out, "captured": captured, "coefs": coefs}

    print("\n  coefficient signs (POSITIVE = correctly oriented; both predictor")
    print("  and criterion are expected runs, lower = better):")
    for label, res in results.items():
        sig = "  ".join(f"{k}={v:+.4f}" for k, v in res["coefs"].items())
        bad = [k for k, v in res["coefs"].items() if v < 0]
        flag = f"   <-- CHECK: {','.join(bad)} negative" if bad else ""
        print(f"    {label:<26}{sig}{flag}")

    full = results["Stuff+ and Location+"]
    print(f"\n  HEADLINE: physical Pitching+ captures {full['captured']:.1%} of stable "
          f"skill;\n            {1 - full['captured']:.1%} of stable skill is MISSING "
          f"(= {full['out']['s2_stable'] / base['total']:.1%} of total\n            "
          f"single-season variance in mean xT).")
    return base, results


def cluster_bootstrap(tab, sigma2_w, reps, seed=17):
    """Resample PITCHERS with replacement; whole careers move together.

    sigma2_w is held fixed: it is estimated from millions of pitches, so its
    sampling error is negligible next to the between-pitcher terms.
    """
    rng = np.random.default_rng(seed)
    pids = tab["pitcher"].unique()
    idx = {p: g for p, g in tab.groupby("pitcher")}
    keys = ["share_noise", "share_drift", "share_stable", "persistence"]
    acc = {k: [] for k in keys}
    acc["captured"] = []
    for _ in range(reps):
        draw = rng.choice(pids, size=len(pids), replace=True)
        parts = []
        for j, p in enumerate(draw):
            g = idx[p].copy()
            g["pitcher"] = j  # relabel so a pitcher drawn twice counts twice
            parts.append(g)
        bt = pd.concat(parts, ignore_index=True)
        try:
            b = vc.variance_components(bt, sigma2_w)
            rtab, _ = residualize(bt, ["stuff", "loc"])
            r = vc.variance_components(rtab, sigma2_w)
        except (ValueError, np.linalg.LinAlgError):
            continue
        for k in keys:
            acc[k].append(b[k])
        acc["captured"].append(1.0 - r["s2_stable"] / b["s2_stable"])
    print("\nCLUSTER BOOTSTRAP over pitchers "
          f"({len(acc['share_stable'])} usable of {reps} reps):")
    for k in keys + ["captured"]:
        fc.boot_report(k, acc[k])
    return acc


def consistency(a_out, base):
    """Do Part A and Part B agree? They measure overlapping quantities."""
    print("\n" + "=" * 78)
    print("A vs B CONSISTENCY CHECK")
    print("=" * 78)
    obs = [v["rho_full"] for v in a_out.values()]
    rho_obs = float(np.mean(obs)) if obs else float("nan")
    print(f"{'quantity':<42}{'observed':>10}{'B predicts':>12}")
    print(f"{'within-season reliability (A, mean of seasons)':<42}"
          f"{rho_obs:>10.3f}{base['rho_within_pred']:>12.3f}")
    print("  (A measures stable+drift over stable+drift+noise; B predicts the same)")
    print(f"\n{'persistence = stable / (stable + drift)':<42}"
          f"{'':>10}{base['persistence']:>12.3f}")
    print("  Fraction of what repeats within a season that also survives to the")
    print("  next one. This, not the model, is what caps year-over-year forecasting.")
    gap = abs(rho_obs - base["rho_within_pred"])
    verdict = "AGREE" if gap < 0.05 else "DISAGREE -- reconcile before reporting"
    print(f"\n  gap = {gap:.3f}  ->  {verdict}")
```

Then replace the `__main__` block with:

```python
if __name__ == "__main__":
    args = cli()
    ff, tab = build_panel(args)
    a_out = part_a(ff, args)
    sigma2_w = vc.pooled_within_variance(
        ff["xT"], ff["PitcherId"].astype(str) + "|" + ff["year"].astype(str))
    base, results = part_b(ff, tab, sigma2_w, args)
    cluster_bootstrap(tab, sigma2_w, args.boot)
    consistency(a_out, base)

    print("\n" + "=" * 78)
    print("ROBUSTNESS: same decomposition on adjT (opponent-adjusted criterion)")
    print("=" * 78)
    alt = tab.drop(columns=["mean"]).rename(columns={"mean_adjT": "mean"})
    s2w_alt = vc.pooled_within_variance(
        ff["adjT"], ff["PitcherId"].astype(str) + "|" + ff["year"].astype(str))
    b_alt = vc.variance_components(alt, s2w_alt)
    r_alt, _ = residualize(alt, ["stuff", "loc"])
    o_alt = vc.variance_components(r_alt, s2w_alt)
    print(f"  noise {b_alt['share_noise']:.1%}  drift {b_alt['share_drift']:.1%}  "
          f"stable {b_alt['share_stable']:.1%}  "
          f"captured {1 - o_alt['s2_stable'] / b_alt['s2_stable']:.1%}")
    print("  xT and adjT correlate ~0.985, so these should track the headline")
    print("  closely; a large divergence means something is wrong.")
```

- [ ] **Step 2: Run the full script**

```bash
cd component_model/analysis
python 12_reliability_decomposition.py \
  --caches "C:/Users/jackdav/stuffplus_replication/workdir_2425_d1/pitches_cache_D1.parquet,C:/Users/jackdav/stuffplus_replication/workdir_2526_d1/pitches_cache_2025_2026_D1.parquet" \
  --workdir "C:/Users/jackdav/stuffplus_replication/workdir_decomp" \
  --boot 300 2>&1 | tee "C:/Users/jackdav/stuffplus_replication/log_12_decomposition.txt"
```

Use `--boot 300` for the first pass to check it runs, then `--boot 1000` for the numbers that get reported.

- [ ] **Step 3: Check the four things that would invalidate the result**

1. **Bucket shares sum to 100%** and none is wildly negative. Small negative drift is acceptable and self-documenting; a large negative anything means the moment equations or `sigma2_w` are wrong.
2. **Coefficient signs POSITIVE** for `stuff` and `loc`. A negative sign means the orientation is inverted — stop and reconcile against the `fair_criterion.py` docstring before writing any interpretation.
3. **A/B consistency prints AGREE.** If it disagrees, the two parts are measuring different things and neither number is reportable yet. Most likely cause: `sigma2_w` computed over a different row set than the panel means.
4. **`captured` is between 0 and 1** and roughly consistent with the standalone-validity framing (physical Pitching+ correlates ~0.21 to 0.28 with next-year criterion against a reliability ceiling near 0.52 to 0.55, so a captured share in the low tens of percent of *total* variance is expected; as a share of *stable* skill it will be much higher).

- [ ] **Step 4: Commit**

```bash
git add component_model/analysis/12_reliability_decomposition.py
git commit -m "Add variance decomposition and missing-skill estimate to script 12"
```

---

### Task 4: Record the verdict

**Files:**
- Modify: `component_model/RESULTS.md` (append a verdict section)
- Modify: `component_model/analysis/README.md` (add script 12 to the run-order table; note the tests)

**Interfaces:**
- Consumes: printed output from Task 3's run (`log_12_decomposition.txt`).
- Produces: documentation only.

- [ ] **Step 1: Append the verdict section to `component_model/RESULTS.md`**

Use the real numbers from the log. Aggregates only — no pitcher names, no derived TrackMan values. Structure:

```markdown
## Reliability decomposition: noise vs missing skill (2026-07-27, script 12)

Question: of the year-over-year unpredictability in pitcher fair-criterion
performance, how much is irreducible and how much is skill Pitching+ misses?

Panel: <N> D1 pitchers, <M> pitcher-seasons, 2024-2026, four-seam, 100+ FF per
season and 2+ qualified seasons. Criterion xT; adjT as a robustness row.

**Part A -- within-season reliability (game parity, Spearman-Brown):**
<per-season table>. Stuff+ anchor <value> confirms the split. Game parity reads
below script 06's pitch-parity number by construction: pitch parity leaves
within-game shared effects in both halves.

**Part B -- bucket shares of single-season variance in mean xT:**
| bucket | share |
|---|---|
| measurement noise | <x>% |
| true drift | <x>% |
| stable skill | <x>% |

**Missing skill:** physical Pitching+ (Stuff+ and Location+, results excluded as
circular) captures <x>% of stable skill; <x>% is missing, which is <x>% of total
single-season variance. Stuff+ alone <x>%, Location+ alone <x>%.

**Consistency:** A's within-season reliability <x> vs B's predicted <x> (gap
<x>). Persistence (stable / stable+drift) = <x>.

**Verdict:** <the actual read: which bucket dominates, and whether the missing
share justifies running Part C>.

Caveats: D1 only (all-levels 2024 frame lacks a Level column; the feed's D2 share
doubled, so all-levels would be composition-diluted anyway). Four-seam only.
Three seasons, so drift and stable skill are separated but not precisely.
Cluster-bootstrap CIs over pitchers; sigma2_w treated as known.
```

- [ ] **Step 2: Add script 12 to the `README.md` run-order table**

Add after the script 11 row:

```markdown
| `12_reliability_decomposition.py` | How much year-over-year unpredictability is noise vs skill Pitching+ misses? | pitch caches (`--caches`) |
```

And under Rules, add:

```markdown
- `variance_components.py` is unit-tested: `cd component_model/analysis && python -m pytest tests/ -v`.
  Run the tests after touching the estimator; the synthetic-recovery test is the
  only check that the decomposition is arithmetically correct.
```

- [ ] **Step 3: Verify the docs match the log**

Re-read the appended section against `log_12_decomposition.txt` and confirm every number transcribed matches, that no pitcher names or per-pitch values leaked in, and that the verdict states which bucket dominates rather than hedging.

- [ ] **Step 4: Run the full test suite once more**

Run: `cd component_model/analysis && python -m pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add component_model/RESULTS.md component_model/analysis/README.md
git commit -m "Record the reliability decomposition verdict"
```

---

## Self-Review

**Spec coverage:**
- Question / three-bucket framing → Task 3 `part_b` prints all three buckets.
- Target xT, lower=better, sign gate → Global Constraints; `residualize` docstring; Task 3 Step 3 check 2.
- D1-only, four-seam → Global Constraints; Task 2 data-sourcing note documents why all-levels is unavailable.
- Reuse `fair_criterion.py`, don't modify → Global Constraints; script imports `fc` and trains nothing new.
- Part A odd/even GameID + Spearman-Brown, per season → Task 2 `add_game_parity`, `part_a`.
- Part A composes `load_pitches`-style frames rather than the slim panel (needs `GameID`) → Task 2 `load_seasons` reads caches carrying `GameID`.
- Part A interpretation gate → Task 2 Step 3.
- Three-season panel, frozen xT vintage → Task 2 `build_panel` (single pooled `add_xt`).
- Variance decomposition into stable / drift / noise with season fixed effects → Task 1 `variance_components` (season-centering); Task 3.
- Cheaper aggregate route with known measurement error → adopted as the primary method (no statsmodels), documented in Architecture and the module docstring.
- Missing-skill step with results excluded as circular → Task 3 `part_b`.
- Two denominators (share of stable, share of total) → Task 3 table columns.
- Standalone-validity cross-check → Task 3 Step 3 check 4.
- Deliverable script 12 + RESULTS verdict + README row → Tasks 2, 3, 4.
- Replication caveat → Task 4 Step 1 caveats.
- Out of scope (Part C, per-type, criterion changes) → Global Constraints.

**Deviations from the spec, both deliberate:**
1. Spec named a mixed-model formula (`(1|pitcher) + (1|pitcher:season)`). `statsmodels` is not installed and its `MixedLM` does not fit crossed/nested random effects of this shape well anyway. The method-of-moments estimator recovers the identical three-bucket split, adds no dependency, and is unit-testable against known truth. Same estimand, better verification.
2. All-levels footnote dropped: the only all-levels 2024 frame lacks `Level`. Recorded as a caveat rather than faked.

**Placeholder scan:** No TBDs. The `<...>` markers in Task 4 Step 1 are transcription slots for measured values that cannot exist before the run, not unspecified work.

**Type consistency:** `tab` carries `pitcher`/`season`/`mean`/`n` everywhere `vc.variance_components` is called — `build_panel` renames `PitcherId`/`year` at creation, `residualize` returns those four columns, and the adjT block renames `mean_adjT` to `mean` before use. `spearman_brown` is defined in Task 1 and called as `vc.spearman_brown` in Task 2. `fc.boot_report` and `fc.z` match the existing signatures in `fair_criterion.py`.
