"""
Brute-force pit strategy simulator: given fitted degradation curves for a
race, try every (compound, compound, pit lap) combination and find the
fastest total race time. This is the Day 4 deliverable -- the core result.

Deliberately not a Monte Carlo simulation yet (no safety car probability,
no traffic model). That's a stretch goal once the base version works
end-to-end -- see README roadmap.

Usage:
    python -m models.simulate --year 2025 --race Bahrain --total-laps 57
"""

import argparse
import itertools
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

# Rough average pit stop time loss (in-lap + stationary + out-lap delta vs.
# a green-flag lap). Replace with a race-specific figure once you've
# measured it from in/out lap times -- see README known limitations.
PIT_LOSS_SECONDS = 22.0


def load_degradation(race: str, year: int, min_pct_positive: float = 0.5) -> dict:
    """
    Pulls the most recent fit for this race from the degradation log.
    Drops any compound whose pct_stints_positive_slope is below
    min_pct_positive -- a low value means most individual stints didn't
    even show degradation in the expected direction, so the fit isn't
    trustworthy enough to steer a strategy decision.

    Known case: Bahrain 2025 HARD is excluded by this check. Every HARD
    stint in that race was either a car's final stint (drivers running a
    conserve-early/push-late pace once no stops were left -- real
    race-craft, not tyre physics) or started right at the lap-32 safety
    car restart. Confirmed via stint-start-lap inspection: all 14 HARD
    stints clustered at exactly those two situations, so there's no clean
    subsample left in this race to fit HARD degradation from. Revisit once
    a race is added (Day 6) where HARD is used mid-race under green flag.
    """
    log_path = EXPERIMENTS_DIR / "degradation_results.jsonl"
    latest = None
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["race"] == race and entry["year"] == year:
                latest = entry
    if latest is None:
        raise ValueError(f"No degradation fit found for {race} {year}. Run models.degradation first.")

    compounds = {}
    for name, params in latest["compounds"].items():
        pct_positive = params.get("pct_stints_positive_slope", 1.0)
        if pct_positive < min_pct_positive:
            logger.warning(
                "Excluding %s from simulation: only %.0f%% of stints showed positive "
                "degradation (below %.0f%% threshold). See load_degradation docstring.",
                name, 100 * pct_positive, 100 * min_pct_positive,
            )
            continue
        compounds[name] = params

    if len(compounds) < 2:
        raise ValueError(
            f"Fewer than 2 trustworthy compounds for {race} {year} -- can't simulate a "
            "two-stop strategy. Lower min_pct_positive or investigate the excluded compound(s)."
        )
    return compounds


def lap_time(compound_params: dict, tyre_age: int) -> float:
    return compound_params["intercept_s"] + compound_params["slope_s_per_lap"] * tyre_age


def simulate_strategy(compounds: dict, total_laps: int, pit_lap: int,
                       compound_1: str, compound_2: str) -> float:
    total = PIT_LOSS_SECONDS
    for lap in range(1, total_laps + 1):
        if lap <= pit_lap:
            total += lap_time(compounds[compound_1], tyre_age=lap)
        else:
            total += lap_time(compounds[compound_2], tyre_age=lap - pit_lap)
    return total


def find_best_strategy(compounds: dict, total_laps: int) -> dict:
    """
    Best strategy is found across ALL pit laps (the model needs the full
    picture to compare options fairly), but flagged if either resulting
    stint length exceeds the longest stint actually observed for that
    compound -- past that point the linear degradation fit is
    extrapolating, and a real tyre could behave very differently (a
    cliff-off is common and this model can't see it).
    """
    available = list(compounds.keys())
    best = {"time_s": float("inf")}

    for c1, c2 in itertools.permutations(available, 2):
        for pit_lap in range(3, total_laps - 2):
            t = simulate_strategy(compounds, total_laps, pit_lap, c1, c2)
            if t < best["time_s"]:
                stint_2_length = total_laps - pit_lap
                best = {
                    "time_s": round(t, 1),
                    "compound_1": c1,
                    "compound_2": c2,
                    "pit_lap": pit_lap,
                    "stint_1_length": pit_lap,
                    "stint_2_length": stint_2_length,
                }

    max_1 = compounds[best["compound_1"]].get("max_observed_tyre_life")
    max_2 = compounds[best["compound_2"]].get("max_observed_tyre_life")
    extrapolated = []
    if max_1 is not None and best["stint_1_length"] > max_1:
        extrapolated.append(f"{best['compound_1']} stint ({best['stint_1_length']} laps > {max_1} observed)")
    if max_2 is not None and best["stint_2_length"] > max_2:
        extrapolated.append(f"{best['compound_2']} stint ({best['stint_2_length']} laps > {max_2} observed)")
    best["extrapolated_beyond_data"] = extrapolated
    if extrapolated:
        logger.warning(
            "Best strategy extrapolates beyond observed data: %s. Treat this "
            "recommendation with caution -- the linear model has no evidence "
            "for how the tyre actually behaves that far into a stint.",
            "; ".join(extrapolated),
        )
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--race", type=str, required=True)
    parser.add_argument("--total-laps", type=int, required=True)
    parser.add_argument(
        "--min-pct-positive", type=float, default=0.4,
        help="Minimum fraction of stints that must show positive degradation "
             "for a compound to be trusted in the simulation. Default 0.4 -- "
             "lowered from a stricter 0.5 because Bahrain 2025 showed weak "
             "degradation overall; tighten this for higher-deg circuits.",
    )
    args = parser.parse_args()

    compounds = load_degradation(args.race, args.year, args.min_pct_positive)
    logger.info("Simulating with compounds: %s", list(compounds.keys()))

    best = find_best_strategy(compounds, args.total_laps)

    logger.info("Best strategy for %s %s: %s", args.year, args.race, best)