from pipeline.run_all_races import borrow_missing


def comp(intercept, slope, pct=1.0):
    return {"intercept_s": intercept, "slope_s_per_lap": slope,
            "pct_stints_positive_slope": pct}


def race(name, compounds, status="ok", laps=57):
    return {"year": 2025, "race": name, "status": status, "total_laps": laps,
            "compounds": compounds, **({"reason": "only 1 ..."} if status == "skipped" else {})}


def full_race(name, offset=0.0):
    return race(name, {"SOFT": comp(90 + offset, 0.09), "MEDIUM": comp(90.5 + offset, 0.06),
                       "HARD": comp(91 + offset, 0.03)})


def test_borrows_second_compound_for_one_compound_race():
    records = [full_race(f"R{i}", offset=i) for i in range(4)]
    # Miami-like race: only HARD is trustworthy (MEDIUM's stints mostly went negative).
    records.append(race("Miami", {"MEDIUM": comp(95, -0.3, pct=0.1), "HARD": comp(96, 0.04)},
                        status="skipped"))
    new = borrow_missing(records)
    assert [r["race"] for r in new] == ["Miami"]
    miami = new[0]
    assert miami["status"] == "ok" and "reason" not in miami
    # Borrowed compounds use the season's median slope, and pace offset from HARD.
    assert miami["borrowed"]["MEDIUM"]["slope_s_per_lap"] == 0.06
    assert miami["borrowed"]["SOFT"]["intercept_s"] == 96 - 1.0
    assert miami["best_strategy"]["compound_1"] != miami["best_strategy"]["compound_2"]


def test_not_enough_season_data_to_borrow():
    records = [full_race("R0"), race("Miami", {"HARD": comp(96, 0.04)}, status="skipped")]
    assert borrow_missing(records) == []


def test_races_with_two_good_compounds_are_left_alone():
    records = [full_race(f"R{i}") for i in range(4)]
    assert borrow_missing(records) == []

def test_borrowed_compounds_get_season_median_stint_cap():
    def capped(offset, caps):
        r = full_race("x", offset)
        for name, cap in caps.items():
            r["compounds"][name]["max_observed_tyre_life"] = cap
        return r
    records = [{**capped(i, {"SOFT": 20 + i, "MEDIUM": 30, "HARD": 40}), "race": f"R{i}"} for i in range(4)]
    records.append(race("Miami", {"MEDIUM": comp(95, -0.3, pct=0.1),
                                  "HARD": {**comp(96, 0.04), "max_observed_tyre_life": 35}},
                        status="skipped"))
    miami = borrow_missing(records)[0]
    assert miami["borrowed"]["SOFT"]["max_observed_tyre_life"] == 22   # median of 20..23 = 21.5 -> 22
    assert miami["borrowed"]["MEDIUM"]["max_observed_tyre_life"] == 30