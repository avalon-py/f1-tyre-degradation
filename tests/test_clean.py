"""
Basic sanity tests for the cleaning pipeline. Run with: pytest
"""

import pandas as pd
import pytest

from features.clean import clean_laps


def make_fake_laps():
    return pd.DataFrame({
        "Compound": ["MEDIUM", "MEDIUM", None, "HARD", "MEDIUM", "MEDIUM", "MEDIUM"],
        "PitInTime": [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.Timestamp("2025-01-01"), pd.NaT, pd.NaT],
        "PitOutTime": [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        "TrackStatus": ["1", "1", "1", "2", "1", "1", "1"],
        "LapTime": pd.to_timedelta(
            ["0:01:46", "0:01:38", "0:01:29", "0:01:35", "0:01:32", "0:01:30", "0:01:31"]
        ),
        "LapNumber": [1, 2, 3, 4, 5, 6, 7],
        "TyreLife": [1, 2, 1, 3, 1, 3, 4],
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


def test_drops_tyre_life_two_as_well():
    # TyreLife==1 is the out-lap itself, already caught by the pit filter.
    # The real "unrepresentative" lap -- cold start OR a bunched safety-car
    # restart -- is the first FLYING lap, TyreLife==2. That must be dropped
    # explicitly, since it isn't a pit lap and isn't flagged by track status.
    laps = make_fake_laps()
    cleaned = clean_laps(laps)
    assert (cleaned["TyreLife"] <= 2).sum() == 0


def test_drops_restart_flying_lap_mid_race():
    # A stint that starts right after a safety car restart (not a literal
    # pit-out lap, not flagged by track status) still has TyreLife==2 on
    # its first flying lap and must be dropped the same way.
    laps = make_fake_laps()
    restart_row = pd.DataFrame([{
        "Compound": "MEDIUM", "PitInTime": pd.NaT, "PitOutTime": pd.NaT,
        "TrackStatus": "1", "LapTime": pd.to_timedelta("0:01:40"),
        "LapNumber": 35, "TyreLife": 2,
    }])
    laps = pd.concat([laps, restart_row], ignore_index=True)
    cleaned = clean_laps(laps)
    assert (cleaned["LapNumber"] == 35).sum() == 0


def test_fuel_correction_increases_later_laps_more():
    laps = make_fake_laps()
    cleaned = clean_laps(laps)
    # Lap 7 (lighter on fuel, ran unfairly fast) should get MORE time added
    # back than lap 6, to bring it up to full-tank-equivalent pace.
    lap6 = cleaned[cleaned["LapNumber"] == 6].iloc[0]
    lap7 = cleaned[cleaned["LapNumber"] == 7].iloc[0]
    correction_lap6 = lap6["LapTimeCorrected"] - lap6["LapTimeSeconds"]
    correction_lap7 = lap7["LapTimeCorrected"] - lap7["LapTimeSeconds"]
    assert correction_lap7 > correction_lap6