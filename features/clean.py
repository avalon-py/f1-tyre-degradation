"""
Clean raw lap data into something a degradation model can trust.

Drops: in-laps, out-laps, laps under safety car / VSC / red flag, and laps
not on a dry compound (SOFT / MEDIUM / HARD). Applies a linear fuel-load correction so laps
aren't confounded by the car getting lighter over the race.

Usage:
    python -m features.clean --year 2025 --race Bahrain [--total-laps 57]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Fuel effect, per race. Fuel burned per lap is (start fuel / race laps), so a
# race with fewer laps (Spa, 44) burns more per lap than one with more
# (Monaco, 78). Race distance is ~305 km everywhere, so laps is effectively
# the inverse of circuit length.
#
# Both constants are educated guesses, not measured values:
#   - 110 kg is the regulation maximum; teams often start a little under it.
#   - ~0.03 s/kg is the commonly quoted lap-time cost of fuel mass.
# Assumes linear burn and ignores per-circuit consumption differences.
START_FUEL_KG = 110.0
FUEL_S_PER_KG = 0.03


def fuel_s_per_lap(total_laps: int) -> float:
    """Seconds of lap time gained per lap from fuel burn-off in this race."""
    return START_FUEL_KG / total_laps * FUEL_S_PER_KG


# Only dry compounds are modelled. This is an allow-list rather than a
# notna() check because FastF1 sometimes gives the strings "nan" / "None" /
# "UNKNOWN", which notna() lets straight through.
DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}

# Track status codes that mean "not a clean racing lap" per FastF1's docs.
# 1 = green flag. Everything else gets dropped for degradation fitting.
NON_GREEN_STATUS_CODES = {"2", "4", "5", "6", "7"}  # yellow, SC, red, VSC, VSC ending

# FastF1's per-lap TrackStatus is a STRING OF CONCATENATED CODES for every
# status that was active at any point during the lap: "1", "12", "124",
# "267"... An exact-match test against single codes misses every lap that
# mixed green with yellow/SC/VSC, so test for *any* bad character instead.
_NON_GREEN_RE = "[" + "".join(sorted(NON_GREEN_STATUS_CODES)) + "]"

# Status codes that, when followed by green ("1"), mean "the field just got
# released": SC, red flag, VSC, VSC ending.
_RESTART_FROM = {"4", "5", "6", "7"}

# Laps dropped per driver after green is shown: the lap that contains the
# restart itself, plus this many after it (bunched field, cold tyres and
# brakes, fuel-saving, gaps being closed up).
RESTART_LAPS_AFTER = 1

# Slow-lap filter: traffic, lapped cars, off-track moments and leftover
# yellows aren't tyre wear. Same idea as FastF1's pick_quicklaps: drop laps
# slower than this multiple of the race's fastest remaining lap.
SLOW_LAP_FACTOR = 1.07


def restart_lap_mask(laps: pd.DataFrame, track_status: pd.DataFrame,
                     n_after: int = RESTART_LAPS_AFTER) -> pd.Series:
    """
    True for laps that should be dropped because they contain, or directly
    follow, a restart to green after an SC / VSC / red flag -- for EVERY
    driver on track at that moment, including those who stayed out (which
    a per-driver TyreLife filter can't catch).

    `track_status` is FastF1's `session.track_status` (columns Time, Status)
    saved by ingestion. Needs laps' LapStartTime, Time, Driver, LapNumber.
    """
    bad = pd.Series(False, index=laps.index)
    ts = track_status.sort_values("Time")
    status = ts["Status"].astype(str).tolist()
    times = ts["Time"].tolist()
    restarts = [t for prev, cur, t in zip(status, status[1:], times[1:])
                if cur == "1" and prev in _RESTART_FROM]
    if not restarts or "LapStartTime" not in laps or "Time" not in laps:
        return bad
    for t in restarts:
        containing = laps[(laps["LapStartTime"] <= t) & (laps["Time"] > t)]
        for driver, g in containing.groupby("Driver"):
            n = int(g["LapNumber"].iloc[0])
            bad |= (laps["Driver"] == driver) & laps["LapNumber"].between(n, n + n_after)
    return bad


def clean_laps(laps: pd.DataFrame, total_laps: int,
               track_status: pd.DataFrame | None = None,
               slow_lap_factor: float | None = SLOW_LAP_FACTOR,
               restart_laps_after: int = RESTART_LAPS_AFTER) -> pd.DataFrame:
    df = laps.copy()
    n_start = len(df)

    # Restart laps have to be flagged on the FULL lap table, before the other
    # filters remove rows, because they're defined by lap number per driver.
    if track_status is not None and len(track_status):
        df = df[~restart_lap_mask(df, track_status, restart_laps_after)]
    if "Deleted" in df.columns:
        df = df[~df["Deleted"].fillna(False).astype(bool)]   # stewards deleted the time

    df = df[df["Compound"].isin(DRY_COMPOUNDS)]
    df = df[df["PitInTime"].isna() & df["PitOutTime"].isna()]
    df = df[~df["TrackStatus"].astype(str).str.contains(_NON_GREEN_RE, regex=True)]
    df = df[df["LapTime"].notna()]
    if slow_lap_factor:
        secs = df["LapTime"].dt.total_seconds()
        df = df[secs <= slow_lap_factor * secs.min()]

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
    df["LapTimeCorrected"] = df["LapTimeSeconds"] + fuel_s_per_lap(total_laps) * (
        df["LapNumber"] - 1
    )

    logger.info(
        "Cleaned %d -> %d laps (%.0f%% kept); fuel effect %.4f s/lap over %d laps",
        n_start, len(df), 100 * len(df) / n_start, fuel_s_per_lap(total_laps), total_laps,
    )
    return df.reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--race", type=str, required=True)
    parser.add_argument("--total-laps", type=int, default=None,
                        help="Race laps (default: winner's lap count from the data)")
    args = parser.parse_args()

    raw_path = RAW_DIR / f"{args.year}_{args.race.replace(' ', '_')}_laps.parquet"
    laps = pd.read_parquet(raw_path)
    ts_path = raw_path.with_name(raw_path.name.replace("_laps.", "_track_status."))
    track_status = pd.read_parquet(ts_path) if ts_path.exists() else None
    if track_status is None:
        logger.warning("No %s -- SC/VSC restart laps will NOT be dropped. "
                       "Re-run ingestion.load_session to save it.", ts_path.name)

    total_laps = args.total_laps or int(laps["LapNumber"].max())
    cleaned = clean_laps(laps, total_laps, track_status)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{args.year}_{args.race.replace(' ', '_')}_clean.parquet"
    cleaned.to_parquet(out_path, index=False)
    logger.info("Saved cleaned laps to %s", out_path)