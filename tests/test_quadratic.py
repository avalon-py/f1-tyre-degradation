import numpy as np
import pandas as pd

from models.degradation_quadratic import fit_pooled, holdout_compare


def make(c_true, b_true=0.02, n_stints=14, noise=0.25, seed=1, cliff=None):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_stints):
        base = 90 + rng.normal(0, 1.0)            # big driver/car baseline spread
        start = 3
        n = rng.integers(14, 26)
        for age in range(start, start + n):
            y = base + b_true * age + c_true * age ** 2 + rng.normal(0, noise)
            rows.append({"Driver": f"D{s}", "Stint": 1, "Compound": "MEDIUM",
                         "TyreLife": age, "LapTimeCorrected": y})
    return pd.DataFrame(rows)


def test_recovers_curvature_despite_baseline_spread():
    fit = fit_pooled(make(c_true=0.004), quadratic=True, n_boot=100)
    assert abs(fit["quad_s_per_lap2"] - 0.004) < 0.0015
    assert fit["c_ci_low"] > 0 and not fit["c_clamped"]


def test_c_clamped_when_truth_is_linear_or_concave():
    fit = fit_pooled(make(c_true=-0.003), quadratic=True, n_boot=50)
    assert fit["quad_s_per_lap2"] == 0.0 and fit["c_clamped"]
    assert fit["slope_s_per_lap"] > -0.1


def test_too_few_stints_returns_none():
    assert fit_pooled(make(0.004, n_stints=2)) is None


def test_quadratic_beats_linear_on_heldout_stints():
    df = make(c_true=0.006, b_true=0.0, n_stints=16)
    res = holdout_compare(df).set_index("method")
    assert res.loc["pooled_quadratic", "late_rmse_s"] < res.loc["per_stint_median_linear", "late_rmse_s"]
    assert res.loc["pooled_quadratic", "rmse_s"] < res.loc["per_stint_median_linear", "rmse_s"]