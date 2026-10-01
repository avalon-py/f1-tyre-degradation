import numpy as np
import pandas as pd

from models.degradation import fit_degradation, pooled_slope
from models.simulate import find_best_strategy


def stints(slope=0.08, n=14, seed=0, noise=0.2):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n):
        base = 90 + rng.normal(0, 1.5)
        length = int(rng.integers(8, 28))
        for age in range(3, 3 + length):
            rows.append({"Driver": f"D{s}", "Stint": 1, "Compound": "MEDIUM", "TyreLife": age,
                         "LapTimeCorrected": base + slope * age + rng.normal(0, noise)})
    return pd.DataFrame(rows)


def test_pooled_slope_recovers_truth_despite_baseline_spread():
    slope, _ = pooled_slope(stints())
    assert abs(slope - 0.08) < 0.01


def test_fit_degradation_reports_pooled_and_per_stint():
    res = fit_degradation(stints())["MEDIUM"]
    assert abs(res["slope_s_per_lap"] - 0.08) < 0.01
    assert "slope_median_per_stint" in res and res["pct_stints_positive_slope"] > 0.8


def comp(intercept, slope, cap):
    return {"intercept_s": intercept, "slope_s_per_lap": slope, "max_observed_tyre_life": cap}


def test_cap_blocks_overlong_stints():
    # SOFT is far faster fresh but wears faster; MEDIUM is the slower, longer-lasting tyre.
    cs = {"SOFT": comp(88, 0.02, 30), "MEDIUM": comp(89, 0.01, 40)}
    capped = find_best_strategy(cs, 57)
    assert not capped["cap_infeasible"]
    assert capped["stint_1_length"] <= cs[capped["compound_1"]]["max_observed_tyre_life"]
    assert capped["stint_2_length"] <= cs[capped["compound_2"]]["max_observed_tyre_life"]
    assert capped["extrapolated_beyond_data"] == []


def test_uncapped_search_would_run_too_long():
    # near-zero slope on MEDIUM and a big pit loss: linear model says never really stop.
    cs = {"SOFT": comp(88, 0.02, 20), "MEDIUM": comp(89, 0.0, 22)}
    free = find_best_strategy(cs, 57, enforce_cap=False)
    assert free["stint_1_length"] > 22 or free["stint_2_length"] > 22


def test_infeasible_cap_falls_back_and_says_so():
    cs = {"SOFT": comp(88, 0.05, 20), "MEDIUM": comp(89, 0.04, 22)}   # 20+22 < 57 laps
    best = find_best_strategy(cs, 57)
    assert best["cap_infeasible"] is True
    assert best["extrapolated_beyond_data"]