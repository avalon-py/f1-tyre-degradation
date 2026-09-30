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

    # Drop the first two laps of every stint (TyreLife <= 2), not just the
    # race's opening lap. FastF1's TyreLife counts the out-lap itself as
    # TyreLife==1 -- that lap is already excluded by the pit filter above,
    # so it alone isn't enough. The first FLYING lap on a fresh set is
    # TyreLife==2, and that's the one that's unrepresentative: cold tyres
    # and first-corner bunching at a standing start (confirmed via EDA --
    # lap 1 averaged ~106.5s at Bahrain 2025 vs ~100.5-100.9s for laps
    # 2-5), or a bunched, sub-racing-pace restart lap if the stint began
    # right after a safety car. Confirmed via Bahrain 2025: the SC on lap
    # 32 (track debris) triggered most front-runners' final stops, and
    # every one of those stints showed a spurious NEGATIVE degradation
    # slope before this fix -- restart bunching inflated the early laps of
    # the stint, which then looked like decay as the field spread back
    # out. Track status alone doesn't catch this, since the restart lap is
    # already logged as green by the time it's recorded.
    df = df[df["TyreLife"] > 2]

    # Force a real copy before adding columns. Every `df = df[condition]`
    # above returns a filtered slice that pandas can't always guarantee is
    # independent of the original -- writing new columns onto it can raise
    # SettingWithCopyWarning. The result was still correct here, but this
    # removes the ambiguity instead of relying on that being true by luck.
    df = df.copy()

    df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()

    # Fuel correction: the car is heaviest (slowest, fuel-wise) on lap 1 and
    # lightest (fastest, fuel-wise) on the final lap. To isolate tyre wear,
    # we need to undo that fuel-driven speedup -- so LATER laps get MORE
    # time added back, bringing them up to what they'd have run on a full
    # tank. Anchored so lap 1 gets ~0 correction.
    df["LapTimeCorrected"] = df["LapTimeSeconds"] + FUEL_CORRECTION_S_PER_LAP * (
        df["LapNumber"] - 1
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
