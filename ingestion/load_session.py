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


def load_race(year: int, race: str, session_type: str = "R") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load lap-level data for one session. Returns (laps, track_status).

    laps is FastF1's laps DataFrame with driver, lap time, compound, tyre
    life, pit flags, and per-lap track status attached -- no telemetry
    pulled, this stays lap-level by design. track_status is FastF1's
    session.track_status (SC / VSC / red-flag start and end times), needed
    to drop restart laps. It's returned separately on purpose: stashing a
    DataFrame in laps.attrs breaks laps.to_parquet (pyarrow JSON-encodes attrs).
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


def load_race_laps(year: int, race: str, session_type: str = "R") -> pd.DataFrame:
    """Laps only (kept for callers that don't need track status)."""
    return load_race(year, race, session_type)[0]


def save_raw(laps: pd.DataFrame, year: int, race: str,
             track_status: pd.DataFrame | None = None) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{year}_{race.replace(' ', '_')}_laps.parquet"
    laps.to_parquet(out_path, index=False)
    logger.info("Saved raw laps to %s", out_path)
    if track_status is not None:
        track_status.to_parquet(
            out_path.with_name(out_path.name.replace("_laps.", "_track_status.")), index=False)
    return out_path


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

    laps, track_status = load_race(args.year, args.race, args.session)
    save_raw(laps, args.year, args.race, track_status)

    quality = check_race_quality(laps)
    logger.info("Quality check: %s", quality)