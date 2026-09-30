import pandas as pd

from features.clean import clean_laps, restart_lap_mask


def laps_df(rows):
    """rows: (driver, lap, status, seconds). Each lap is 100 s of session time."""
    out = []
    for d, n, st, sec in rows:
        out.append({"Driver": d, "LapNumber": n, "TrackStatus": st, "Compound": "MEDIUM",
                    "PitInTime": pd.NaT, "PitOutTime": pd.NaT, "TyreLife": n + 10,
                    "LapTime": pd.to_timedelta(sec, unit="s"),
                    "LapStartTime": pd.to_timedelta((n - 1) * 100, unit="s"),
                    "Time": pd.to_timedelta(n * 100, unit="s")})
    return pd.DataFrame(out)


def test_mixed_status_codes_are_dropped():
    df = laps_df([("A", n, st, 90) for n, st in enumerate(["1", "12", "124", "267", "1", "1"], 1)])
    kept = clean_laps(df, 57, slow_lap_factor=None)
    assert sorted(kept["LapNumber"]) == [1, 5, 6]


def test_restart_drops_all_drivers_including_those_who_stayed_out():
    ts = pd.DataFrame({"Time": pd.to_timedelta([0, 250, 420], unit="s"), "Status": ["1", "4", "1"]})
    # green comes back at t=420 -> inside lap 5 (400..500), for both drivers
    rows = [(d, n, "1", 90) for d in "AB" for n in range(1, 10)]
    df = laps_df(rows)
    mask = restart_lap_mask(df, ts, n_after=1)
    assert sorted(df[mask]["LapNumber"].unique()) == [5, 6]
    assert set(df[mask]["Driver"]) == {"A", "B"}
    kept = clean_laps(df, 57, ts, slow_lap_factor=None)
    assert not kept["LapNumber"].isin([5, 6]).any()


def test_no_restart_when_no_sc_before_green():
    ts = pd.DataFrame({"Time": pd.to_timedelta([0, 300], unit="s"), "Status": ["1", "2"]})
    df = laps_df([("A", n, "1", 90) for n in range(1, 8)])
    assert not restart_lap_mask(df, ts).any()


def test_slow_laps_filtered():
    df = laps_df([("A", n, "1", s) for n, s in enumerate([90, 91, 97, 90.5, 120], 1)])
    kept = clean_laps(df, 57)            # cutoff = 1.07 * 90 = 96.3 s
    assert sorted(kept["LapNumber"]) == [1, 2, 4]