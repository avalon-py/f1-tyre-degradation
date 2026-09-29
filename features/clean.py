"""
Clean raw lap data into something a degradation model can trust.

Drops: in-laps, out-laps, laps under safety car / VSC / red flag, and laps
with no recorded compound. Applies a linear fuel-load correction so laps
aren't confounded by the car getting lighter over the race.

Usage:
    python -m features.clean --year 2025 --race Bahrain
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Seconds gained per lap from fuel burn-off. Rough estimate -- ~110kg of fuel
# burned over a race at roughly 0.03s/lap/10kg. Treat as a starting
# assumption, not ground truth; revisit once you have degradation fits and
# can sanity-check the slope.
FUEL_CORRECTION_S_PER_LAP = 0.03

# Track status codes that mean "not a clean racing lap" per FastF1's docs.
# 1 = green flag. Everything else gets dropped for degradation fitting.
NON_GREEN_STATUS_CODES = {"2", "4", "5", "6", "7"}  # yellow, SC, red, VSC, VSC ending


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    df = laps.copy()
    n_start = len(df)

    df = df[df["Compound"].notna()]
    df = df[df["PitInTime"].isna() & df["PitOutTime"].isna()]
    df = df[~df["TrackStatus"].astype(str).isin(NON_GREEN_STATUS_CODES)]
    df = df[df["LapTime"].notna()]

    # Lap 1 is a standing start -- cold tyres, first-corner bunching, not
    # representative tyre wear. It's not a pit lap and not flagged by track
    # status, so it needs its own filter. Confirmed via EDA: lap 1 averaged
    # ~106.5s at Bahrain 2025 vs ~100.5-100.9s for laps 2-5.
    df = df[df["LapNumber"] > 1]

    df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()

    # Fuel correction: lap 1 is the heaviest, so later laps get time added
    # back to make them comparable to a full-fuel lap.
    max_lap = df["LapNumber"].max()
    df["LapTimeCorrected"] = df["LapTimeSeconds"] + FUEL_CORRECTION_S_PER_LAP * (
        max_lap - df["LapNumber"]
    )

    logger.info("Cleaned %d -> %d laps (%.0f%% kept)", n_start, len(df), 100 * len(df) / n_start)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--race", type=str, required=True)
    args = parser.parse_args()

    raw_path = RAW_DIR / f"{args.year}_{args.race.replace(' ', '_')}_laps.parquet"
    laps = pd.read_parquet(raw_path)

    cleaned = clean_laps(laps)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{args.year}_{args.race.replace(' ', '_')}_clean.parquet"
    cleaned.to_parquet(out_path, index=False)
    logger.info("Saved cleaned laps to %s", out_path)