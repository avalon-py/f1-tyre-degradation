"""
Load a race session's lap data via FastF1 and cache it locally.

FastF1 caches raw API responses to disk on first load (slow, ~30-90s per
session) and reads from cache on every subsequent call. `cache/` is
gitignored on purpose -- it's large and machine-specific, not something to
commit.

Usage:
    python -m ingestion.load_session --year 2025 --race Bahrain
"""

import argparse
import logging
from pathlib import Path

import fastf1
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def load_race_laps(year: int, race: str, session_type: str = "R") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load lap-level data for one session.

    Returns (laps, track_status). track_status holds the SC / VSC / red-flag
    start and end times, needed to drop restart laps for every driver.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    session = fastf1.get_session(year, race, session_type)
    session.load(telemetry=False, weather=True, messages=False)

    laps = session.laps.copy()
    laps["Year"] = year
    laps["Race"] = race
    track_status = session.track_status.copy()

    logger.info("Loaded %d laps for %s %s %s", len(laps), year, race, session_type)
    return laps, track_status


def save_raw(laps: pd.DataFrame, track_status: pd.DataFrame, year: int, race: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{year}_{race.replace(' ', '_')}"
    laps_path = RAW_DIR / f"{stem}_laps.parquet"
    status_path = RAW_DIR / f"{stem}_track_status.parquet"

    laps.to_parquet(laps_path, index=False)
    track_status.to_parquet(status_path, index=False)
    logger.info("Saved raw laps to %s and track status to %s", laps_path, status_path)
    return laps_path


def check_race_quality(laps: pd.DataFrame) -> dict:
    """
    Quick dry/green-flag sanity check before committing a race to the
    project -- run this BEFORE spending a day analyzing a messy race.
    """
    track_status = laps["TrackStatus"].astype(str)
    # Per-lap TrackStatus concatenates codes ("12", "124"): green means ONLY "1".
    pct_green = (track_status == "1").mean()
    compounds_used = laps["Compound"].dropna().unique().tolist()
    rainfall = laps.get("Rainfall")

    return {
        "n_laps": len(laps),
        "pct_green_flag": round(float(pct_green), 3),
        "compounds_used": compounds_used,
        "any_rain_flag": bool(rainfall.any()) if rainfall is not None else "unknown",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--race", type=str, required=True)
    parser.add_argument("--session", type=str, default="R")
    args = parser.parse_args()

    laps, track_status = load_race_laps(args.year, args.race, args.session)
    save_raw(laps, track_status, args.year, args.race)

    quality = check_race_quality(laps)
    logger.info("Quality check: %s", quality)