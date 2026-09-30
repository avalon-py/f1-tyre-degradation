"""
Pooled per-race, per-compound tyre degradation fit with optional curvature:

    lap_time = a_stint + b * age + c * age^2      (c >= 0)

One (b, c) pair per compound per race, fit from ALL stints at once, with a
separate baseline a_stint for every driver-stint. The stint baselines are
handled by demeaning within each stint, so "faster driver" / "car set-up" /
"fuel+track state at stint start" never leak into the wear terms -- this
keeps the protection the per-stint-slope approach gave us, without fitting
a noisy c on each ~15-lap stint separately.

Not wired into models/simulate.py on purpose: compare first
(pipeline/compare_curves.py), change the simulator only if c is clearly
positive and the held-out late-lap error improves.
"""

import numpy as np
import pandas as pd

from models.degradation import MIN_LAPS_PER_STINT, _fit_one_stint

MIN_STINTS = 3


def _stint_arrays(group: pd.DataFrame, min_laps: int = MIN_LAPS_PER_STINT):
    """List of (age, y) numpy pairs, one per driver-stint with enough laps."""
    out = []
    for _, g in group.groupby(["Driver", "Stint"]):
        if len(g) >= min_laps:
            out.append((g["TyreLife"].to_numpy(float), g["LapTimeCorrected"].to_numpy(float)))
    return out


def _solve(stints, quadratic: bool):
    """Within-stint least squares. Returns (b, c, baselines, clamped)."""
    def design(age):
        return np.column_stack([age, age ** 2]) if quadratic else age[:, None]

    X = np.vstack([design(a) - design(a).mean(axis=0) for a, _ in stints])
    y = np.concatenate([yy - yy.mean() for _, yy in stints])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    b = float(beta[0])
    c = float(beta[1]) if quadratic else 0.0
    clamped = False
    if quadratic and c < 0:          # no tyre gets *better* at an accelerating rate
        clamped = True
        c = 0.0
        Xl = np.vstack([a[:, None] - a.mean() for a, _ in stints])
        b = float(np.linalg.lstsq(Xl, y, rcond=None)[0][0])
    baselines = np.array([yy.mean() - b * a.mean() - c * (a ** 2).mean() for a, yy in stints])
    return b, c, baselines, clamped


def fit_pooled(group: pd.DataFrame, quadratic: bool = True, n_boot: int = 200,
               seed: int = 0) -> dict | None:
    """Fit one compound's laps. Returns None if too few stints.

    c_ci_low/high: 90% cluster bootstrap over stints (resample whole stints,
    not laps -- laps within a stint aren't independent). If c_ci_low is
    ~0 and c_clamped is True in many resamples, curvature isn't identified.
    """
    stints = _stint_arrays(group)
    if len(stints) < MIN_STINTS:
        return None
    b, c, base, clamped = _solve(stints, quadratic)
    res = {
        "intercept_s": round(float(np.median(base)), 3),   # value at age 0, like the linear fit
        "slope_s_per_lap": round(b, 4),
        "quad_s_per_lap2": round(c, 5),
        "c_clamped": clamped,
        "n_stints_used": len(stints),
        "n_laps": int(sum(len(a) for a, _ in stints)),
        "max_observed_tyre_life": int(group["TyreLife"].max()),
    }
    if quadratic and n_boot:
        rng = np.random.default_rng(seed)
        cs = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(stints), len(stints))
            cs.append(_solve([stints[i] for i in idx], True)[1])
        cs = np.array(cs)
        res["c_ci_low"] = round(float(np.percentile(cs, 5)), 5)
        res["c_ci_high"] = round(float(np.percentile(cs, 95)), 5)
        res["pct_boot_c_clamped"] = round(float((cs == 0).mean()), 3)
    return res


def fit_race(cleaned: pd.DataFrame, n_boot: int = 200) -> dict:
    """{compound: {"linear": ..., "quadratic": ...}} for one cleaned race."""
    out = {}
    for compound, g in cleaned.groupby("Compound"):
        lin = fit_pooled(g, quadratic=False, n_boot=0)
        quad = fit_pooled(g, quadratic=True, n_boot=n_boot)
        if lin and quad:
            out[compound] = {"linear": lin, "quadratic": quad}
    return out


# ---------------------------------------------------------------- held-out test

def _predict_shape(params, age):
    return params["slope_s_per_lap"] * age + params.get("quad_s_per_lap2", 0.0) * age ** 2


def holdout_compare(group: pd.DataFrame, k_calib: int = 5, min_laps: int = 9) -> pd.DataFrame:
    """
    Leave-one-stint-out. For each held-out stint the model never saw:
      1. fit on all OTHER stints of this compound,
      2. set the stint's level from its first `k_calib` laps (the unknown
         baseline has to come from somewhere),
      3. predict the REST of the stint and compare.
    This is exactly the question the simulator cares about: is the model
    right about how lap time moves later in a stint? `late_bias_s` > 0
    means actual laps were slower than predicted, i.e. the model underprices
    late laps. Methods: per-stint-median linear (current pipeline), pooled
    linear, pooled quadratic.
    """
    keyed = [(k, g) for k, g in group.groupby(["Driver", "Stint"]) if len(g) >= min_laps]
    if len(keyed) < MIN_STINTS + 1:
        return pd.DataFrame()
    rows = []
    for i, (_, held) in enumerate(keyed):
        train_groups = [g for j, (_, g) in enumerate(keyed) if j != i]
        train = pd.concat(train_groups)
        slopes = [f["slope_s_per_lap"] for g in train_groups if (f := _fit_one_stint(g))]
        models = {
            "per_stint_median_linear": {"slope_s_per_lap": float(np.median(slopes))} if slopes else None,
            "pooled_linear": fit_pooled(train, quadratic=False, n_boot=0),
            "pooled_quadratic": fit_pooled(train, quadratic=True, n_boot=0),
        }
        age = held["TyreLife"].to_numpy(float)
        y = held["LapTimeCorrected"].to_numpy(float)
        order = np.argsort(age)
        age, y = age[order], y[order]
        cal, test = slice(0, k_calib), slice(k_calib, None)
        late = np.arange(len(age))[test]
        late = late[len(late) * 2 // 3:]                      # last third of the test laps
        for name, p in models.items():
            if p is None:
                continue
            shape = _predict_shape(p, age)
            level = float(np.mean(y[cal] - shape[cal]))
            err = y - (level + shape)
            rows.append({"method": name, "stint_laps": len(age),
                         "sq_err": float(np.mean(err[test] ** 2)),
                         "late_err": float(np.mean(err[late])),
                         "late_sq_err": float(np.mean(err[late] ** 2))})
    df = pd.DataFrame(rows)
    return (df.groupby("method")
              .agg(n_stints=("sq_err", "size"),
                   rmse_s=("sq_err", lambda s: float(np.sqrt(s.mean()))),
                   late_rmse_s=("late_sq_err", lambda s: float(np.sqrt(s.mean()))),
                   late_bias_s=("late_err", "mean"))
              .round(3).reset_index())