"""
Runs ingestion -> clean -> degradation (NOT simulate -- that needs a
total-lap count and real validation, kept manual on purpose) across every
round of the season, so you get breadth for free from the pipeline you
already built.

Intentionally cheap: no per-race judgment calls, no hand validation. A
race that errors out (cancelled session, wet-only data, whatever) is
logged and skipped rather than stopping the whole run -- check
experiments/batch_run_log.jsonl afterward for anything that failed.

FastF1's first download per race is slow (30-90s), so a full season run
can take 20-40 minutes on a first pass. Cached after that.

Usage:
    python -m pipeline.run_all_races --year 2025
    python -m pipeline.run_all_races --year 2025 --skip Monaco  # repeatable
"""

import argparse
import json
import logging
from pathlib import Path

import fastf1

from features.clean import clean_laps
from ingestion.load_session import check_race_quality, load_race_laps, save_raw
from models.degradation import fit_degradation, log_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


def get_race_names(year: int) -> list[str]:
    """Pulls the season calendar so we don't hardcode 24 race names by hand."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return schedule["EventName"].tolist()


def run_one_race(year: int, race: str) -> dict:
    """Same as pipeline.run_pipeline, minus the simulate step -- this is
    the cheap, no-judgment-calls part of the pipeline only."""
    laps = load_race_laps(year, race, "R")
    save_raw(laps, year, race)
    quality = check_race_quality(laps)

    cleaned = clean_laps(laps)
    degradation = fit_degradation(cleaned, race_label=f"{year} {race}")
    log_result(race, year, degradation)

    return {"race": race, "status": "ok", "quality": quality, "compounds_fit": list(degradation.keys())}


def run_all(year: int, skip: list[str]) -> None:
    races = get_race_names(year)
    logger.info("Found %d races for %d season", len(races), year)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_DIR / "batch_run_log.jsonl"

    for i, race in enumerate(races, 1):
        if race in skip:
            logger.info("[%d/%d] Skipping %s (in --skip list)", i, len(races), race)
            continue

        logger.info("[%d/%d] Running %s...", i, len(races), race)
        try:
            result = run_one_race(year, race)
        except Exception as e:
            logger.error("[%d/%d] FAILED on %s: %s", i, len(races), race, e)
            result = {"race": race, "status": "failed", "error": str(e)}

        with open(log_path, "a") as f:
            f.write(json.dumps({"year": year, **result}) + "\n")

    logger.info("Batch run complete. See %s for per-race status.", log_path)
    logger.info(
        "Any 'failed' entries in that log are races worth checking by hand -- "
        "don't assume they're fine just because the loop kept going."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--skip", type=str, nargs="*", default=["Monaco Grand Prix"],
        help="Race names (as they appear in FastF1's schedule) to skip. "
             "Monaco is skipped by default -- special two-stop rule, not "
             "comparable strategy dynamics (see config/races.json).",
    )
    args = parser.parse_args()

    run_all(args.year, args.skip)
