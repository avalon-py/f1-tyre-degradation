"""
Runs the full pipeline for one race, end to end: load -> clean -> fit
degradation -> simulate best strategy. This is what ties Days 1-4 together
into one command.

Usage:
    python -m pipeline.run_pipeline --year 2025 --race Bahrain --total-laps 57
"""

import argparse
import logging

from features.clean import clean_laps
from ingestion.load_session import check_race_quality, load_race_laps, save_raw
from models.degradation import fit_degradation, log_result
from models.simulate import find_best_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(year: int, race: str, total_laps: int, session_type: str = "R") -> dict:
    laps = load_race_laps(year, race, session_type)
    save_raw(laps, year, race)

    quality = check_race_quality(laps)
    logger.info("Quality check: %s", quality)
    if quality["pct_green_flag"] < 0.7:
        logger.warning(
            "Only %.0f%% green-flag laps -- degradation fit may be noisy. "
            "Consider a different race for the core analysis.",
            100 * quality["pct_green_flag"],
        )

    cleaned = clean_laps(laps)
    degradation = fit_degradation(cleaned, race_label=f"{year} {race}")
    log_result(race, year, degradation)

    best = find_best_strategy(degradation, total_laps)
    logger.info("Best strategy: %s", best)
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--race", type=str, required=True)
    parser.add_argument("--total-laps", type=int, required=True)
    parser.add_argument("--session", type=str, default="R")
    args = parser.parse_args()

    run(args.year, args.race, args.total_laps, args.session)
