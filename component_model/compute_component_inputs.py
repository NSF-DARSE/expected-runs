"""
Generate the empirical input tables for the component model.

Reads the final target/calculated dataset and produces three tables:
    1. swing_branch_probs.csv    P(whiff / foul / in play | swing) per ball-strike count
    2. branch_run_values.csv     mean run value (Target) per branch per count
    3. contact_values_by_hittype.csv   mean run value of in-play contact by TaggedHitType
                                       (interim contact-quality labels until ExitSpeed and
                                       Angle flow through the pipeline)

Each table is computed twice: for all pitches and for the four-seam family.

This repository is public and the TrackMan data is licensed, so generated tables must not
be committed. Point --out somewhere outside the repository.

Usage:
    python compute_component_inputs.py --data <path to Final_Target_Calc CSV> --out <dir>
"""

import argparse
import os

import pandas as pd


WHIFF_CALLS = {"StrikeSwinging"}
FOUL_CALLS = {"FoulBall", "FoulBallFieldable", "FoulBallNotFieldable"}
INPLAY_CALLS = {"InPlay"}
SWING_CALLS = WHIFF_CALLS | FOUL_CALLS | INPLAY_CALLS

FOURSEAM_TYPES = {"Fastball", "FourSeamFastBall", "FourSeamFastball"}

# Known casing variants in TaggedHitType (e.g. Popup vs PopUp).
HIT_TYPE_FIXES = {"PopUp": "Popup"}

USECOLS = [
    "TaggedPitchType", "PitchCall", "TaggedHitType",
    "Balls", "Strikes", "Target",
]


def load_swings(data_path):
    df = pd.read_csv(data_path, usecols=USECOLS)
    df = df.dropna(subset=["Target", "PitchCall", "Balls", "Strikes"])
    df = df[df["PitchCall"].isin(SWING_CALLS)].copy()

    df["branch"] = "foul"
    df.loc[df["PitchCall"].isin(WHIFF_CALLS), "branch"] = "whiff"
    df.loc[df["PitchCall"].isin(INPLAY_CALLS), "branch"] = "inplay"

    df["TaggedHitType"] = df["TaggedHitType"].replace(HIT_TYPE_FIXES)
    return df


def branch_probabilities(swings, subset_name):
    counts = (
        swings.groupby(["Balls", "Strikes", "branch"]).size().unstack(fill_value=0)
    )
    n = counts.sum(axis=1)
    out = pd.DataFrame({
        "subset": subset_name,
        "n_swings": n,
        "p_whiff": counts.get("whiff", 0) / n,
        "p_foul": counts.get("foul", 0) / n,
        "p_inplay": counts.get("inplay", 0) / n,
    }).reset_index()
    return out


def branch_run_values(swings, subset_name):
    g = (
        swings.groupby(["Balls", "Strikes", "branch"])["Target"]
        .agg(n="size", mean_target="mean")
        .reset_index()
    )
    g.insert(0, "subset", subset_name)
    return g


def contact_values(swings, subset_name):
    inplay = swings[swings["branch"] == "inplay"]
    inplay = inplay.dropna(subset=["TaggedHitType"])
    g = (
        inplay.groupby("TaggedHitType")["Target"]
        .agg(n="size", mean_target="mean", median_target="median")
        .reset_index()
        .rename(columns={"TaggedHitType": "hit_type"})
    )
    g["pct_of_inplay"] = 100 * g["n"] / g["n"].sum()
    g.insert(0, "subset", subset_name)
    return g


def sanity_checks(swings, subset_name):
    """A whiff with two strikes is a strikeout, so its run value should be much more
    negative than early-count whiffs. A two-strike foul leaves the count unchanged,
    so its run value should be near zero."""
    whiff = swings[swings["branch"] == "whiff"]
    foul = swings[swings["branch"] == "foul"]
    early = whiff.loc[whiff["Strikes"] < 2, "Target"].mean()
    late = whiff.loc[whiff["Strikes"] == 2, "Target"].mean()
    foul2 = foul.loc[foul["Strikes"] == 2, "Target"].mean()
    print(f"[{subset_name}] whiff mean RV: {early:.4f} (<2 strikes) vs {late:.4f} (2 strikes)")
    print(f"[{subset_name}] foul mean RV at 2 strikes: {foul2:.4f} (expect ~0)")
    if not (late < early):
        print(f"[{subset_name}] WARNING: 2-strike whiffs are not more valuable; check the data")
    if abs(foul2) > 0.02:
        print(f"[{subset_name}] WARNING: 2-strike fouls are not ~0; check the data")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to Final_Target_Calc CSV")
    parser.add_argument("--out", required=True,
                        help="Output directory (keep it outside the repository)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    swings = load_swings(args.data)
    fourseam = swings[swings["TaggedPitchType"].isin(FOURSEAM_TYPES)]

    subsets = [("all", swings), ("fourseam", fourseam)]

    probs = pd.concat([branch_probabilities(s, name) for name, s in subsets])
    rvs = pd.concat([branch_run_values(s, name) for name, s in subsets])
    contact = pd.concat([contact_values(s, name) for name, s in subsets])

    probs.to_csv(os.path.join(args.out, "swing_branch_probs.csv"), index=False)
    rvs.to_csv(os.path.join(args.out, "branch_run_values.csv"), index=False)
    contact.to_csv(os.path.join(args.out, "contact_values_by_hittype.csv"), index=False)

    for name, s in subsets:
        sanity_checks(s, name)
    print(f"\nTables written to {args.out}")


if __name__ == "__main__":
    main()
