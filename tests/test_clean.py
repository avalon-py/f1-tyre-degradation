"""
Basic sanity tests for the cleaning pipeline. Run with: pytest
"""

import pandas as pd
import pytest

from features.clean import clean_laps


def make_fake_laps():
    return pd.DataFrame({
        "Compound": ["MEDIUM", "MEDIUM", None, "HARD", "MEDIUM", "MEDIUM"],
        "PitInTime": [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.Timestamp("2025-01-01"), pd.NaT],
        "PitOutTime": [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        "TrackStatus": ["1", "1", "1", "2", "1", "1"],
        "LapTime": pd.to_timedelta(["0:01:46", "0:01:31", "0:01:29", "0:01:35", "0:01:32", "0:01:30"]),
        "LapNumber": [1, 2, 3, 4, 5, 6],
        "TyreLife": [1, 2, 1, 3, 1, 3],
    })


def test_drops_missing_compound():
    laps = make_fake_laps()
    cleaned = clean_laps(laps)
    assert cleaned["Compound"].notna().all()


def test_drops_non_green_flag_laps():
    laps = make_fake_laps()
    cleaned = clean_laps(laps)
    assert (cleaned["TrackStatus"].astype(str) == "2").sum() == 0


def test_drops_pit_laps():
    laps = make_fake_laps()
    cleaned = clean_laps(laps)
    assert cleaned["PitInTime"].isna().all()


def test_drops_lap_one():
    laps = make_fake_laps()
    cleaned = clean_laps(laps)
    assert (cleaned["LapNumber"] == 1).sum() == 0


def test_fuel_correction_increases_earlier_laps_more():
    laps = make_fake_laps()
    cleaned = clean_laps(laps)
    # Lap 2 should get more fuel correction added than lap 6 (earlier = heavier = more correction)
    lap2 = cleaned[cleaned["LapNumber"] == 2].iloc[0]
    lap6 = cleaned[cleaned["LapNumber"] == 6].iloc[0]
    correction_lap2 = lap2["LapTimeCorrected"] - lap2["LapTimeSeconds"]
    correction_lap6 = lap6["LapTimeCorrected"] - lap6["LapTimeSeconds"]
    assert correction_lap2 > correction_lap6