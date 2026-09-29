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


def load_degradation(race: str, year: int) -> dict:
    """Pulls the most recent fit for this race from the degradation log."""
    log_path = EXPERIMENTS_DIR / "degradation_results.jsonl"
    latest = None
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["race"] == race and entry["year"] == year:
                latest = entry
    if latest is None:
        raise ValueError(f"No degradation fit found for {race} {year}. Run models.degradation first.")
    return latest["compounds"]


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
    available = list(compounds.keys())
    best = {"time_s": float("inf")}

    for c1, c2 in itertools.permutations(available, 2):
        for pit_lap in range(3, total_laps - 2):
            t = simulate_strategy(compounds, total_laps, pit_lap, c1, c2)
            if t < best["time_s"]:
                best = {
                    "time_s": round(t, 1),
                    "compound_1": c1,
                    "compound_2": c2,
                    "pit_lap": pit_lap,
                }
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--race", type=str, required=True)
    parser.add_argument("--total-laps", type=int, required=True)
    args = parser.parse_args()

    compounds = load_degradation(args.race, args.year)
    best = find_best_strategy(compounds, args.total_laps)

    logger.info("Best strategy for %s %s: %s", args.year, args.race, best)
