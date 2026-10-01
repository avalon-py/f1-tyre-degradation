"""
Fit tyre degradation per compound: how much slower (seconds/lap) as tyre
age increases. Deliberately simple -- one linear regression per compound,
per race, on fuel-corrected lap times. This is the Day 3 deliverable.

Usage:
    python -m models.degradation --year 2025 --race Bahrain
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


MIN_LAPS_PER_STINT = 4


def _fit_one_stint(group: pd.DataFrame) -> dict | None:
    """Fits TyreLife -> LapTimeCorrected for a single driver's single stint."""
    if len(group) < MIN_LAPS_PER_STINT:
        return None
    X = group[["TyreLife"]].values
    y = group["LapTimeCorrected"].values
    model = LinearRegression().fit(X, y)
    return {
        "intercept_s": float(model.intercept_),
        "slope_s_per_lap": float(model.coef_[0]),
        "n_laps": int(len(group)),
        "r_squared": float(model.score(X, y)),
    }


def _stint_demeaned(compound_group: pd.DataFrame):
    """(age - stint mean age, lap time - stint mean lap time, stint mean age,
    stint mean lap time) for every driver-stint long enough to use."""
    out = []
    for _, g in compound_group.groupby(["Driver", "Stint"]):
        if len(g) >= MIN_LAPS_PER_STINT:
            x = g["TyreLife"].to_numpy(float)
            y = g["LapTimeCorrected"].to_numpy(float)
            out.append((x - x.mean(), y - y.mean(), x.mean(), y.mean()))
    return out


def pooled_slope(compound_group: pd.DataFrame) -> tuple[float, float] | None:
    """
    One slope for the whole compound, fit from ALL stints at once with a
    separate baseline per driver-stint (via within-stint demeaning).
    Returns (slope, intercept_at_age_0) or None if no usable stints.

    Same protection against "faster driver vs less wear" as fitting each
    stint separately, but a long stint counts for more than a short one
    and one noisy 5-lap stint can't swing the answer. On the cleaned
    Bahrain/Qatar 2025 data it matched the per-stint median on held-out
    stints or beat it (see experiments/curve_comparison.jsonl).
    """
    stints = _stint_demeaned(compound_group)
    if not stints:
        return None
    num = sum(float(x @ y) for x, y, _, _ in stints)
    den = sum(float(x @ x) for x, _, _, _ in stints)
    if den == 0:
        return None
    slope = num / den
    intercept = float(np.median([ym - slope * xm for _, _, xm, ym in stints]))
    return slope, intercept


def stint_slopes_for_compound(cleaned: pd.DataFrame, compound: str) -> pd.DataFrame:
    """Debug helper: one row per driver-stint with its fitted slope, so you
    can spot which stints are dragging an average around before trusting it."""
    rows = []
    subset = cleaned[cleaned["Compound"] == compound]
    for (driver, stint), group in subset.groupby(["Driver", "Stint"]):
        fit = _fit_one_stint(group)
        if fit is not None:
            rows.append({"Driver": driver, "Stint": stint, **fit})
    return pd.DataFrame(rows).sort_values("slope_s_per_lap")


def fit_degradation(cleaned: pd.DataFrame, race_label: str = "") -> dict:
    """
    Returns, per compound: slope from ONE pooled fit with a separate baseline
    per driver-stint (see pooled_slope). A naive pooled regression with a
    single shared baseline confounds "faster driver" with "less tyre wear";
    stint baselines remove that. The per-stint median/mean slopes are still
    reported (slope_median_per_stint, slope_mean, slope_std,
    pct_stints_positive_slope) -- the last one is what load_degradation
    uses as its reliability check. n_stints_used tells you how many stints had enough
    laps (>=4) to fit; n_laps is the total laps behind those stints.

    race_label is purely for logging -- makes log lines identifiable when
    running across many races in a batch (see pipeline.run_all_races),
    where a bare "HARD: 90.06s base..." with no race name is unreadable
    once you're 15 races into a season-wide run.
    """
    prefix = f"[{race_label}] " if race_label else ""
    results = {}
    for compound, compound_group in cleaned.groupby("Compound"):
        stint_fits = []
        for (_driver, _stint), stint_group in compound_group.groupby(["Driver", "Stint"]):
            fit = _fit_one_stint(stint_group)
            if fit is not None:
                stint_fits.append(fit)

        if len(stint_fits) < 3:
            logger.warning(
                "%sSkipping %s: only %d fittable stints, too few to average",
                prefix, compound, len(stint_fits)
            )
            continue

        slopes = np.array([f["slope_s_per_lap"] for f in stint_fits])
        intercepts = np.array([f["intercept_s"] for f in stint_fits])

        max_tyre_life = int(compound_group["TyreLife"].max())
        pooled = pooled_slope(compound_group)
        pooled_slope_s, pooled_intercept_s = pooled

        results[compound] = {
            "intercept_s": round(pooled_intercept_s, 3),
            "slope_s_per_lap": round(pooled_slope_s, 4),
            "slope_median_per_stint": round(float(np.median(slopes)), 4),
            "intercept_median_per_stint": round(float(np.median(intercepts)), 3),
            "slope_mean": round(float(np.mean(slopes)), 4),
            "slope_std": round(float(np.std(slopes)), 4),
            "n_stints_used": len(stint_fits),
            "n_laps": int(sum(f["n_laps"] for f in stint_fits)),
            "pct_stints_positive_slope": round(float((slopes > 0).mean()), 3),
            "max_observed_tyre_life": max_tyre_life,
        }
        logger.info(
            "%s%s: %.3fs base + %.4fs/lap degradation [pooled] (per-stint median=%.4f, mean=%.4f, std=%.4f, "
            "stints=%d, n_laps=%d, %.0f%% of stints positive)",
            prefix, compound, results[compound]["intercept_s"], results[compound]["slope_s_per_lap"],
            results[compound]["slope_median_per_stint"], results[compound]["slope_mean"], results[compound]["slope_std"],
            results[compound]["n_stints_used"],
            results[compound]["n_laps"], 100 * results[compound]["pct_stints_positive_slope"],
        )
    return results


def log_result(race: str, year: int, results: dict) -> None:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_DIR / "degradation_results.jsonl"
    entry = {"year": year, "race": race, "compounds": results}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info("Appended result to %s", log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--race", type=str, required=True)
    args = parser.parse_args()

    path = PROCESSED_DIR / f"{args.year}_{args.race.replace(' ', '_')}_clean.parquet"
    cleaned = pd.read_parquet(path)

    results = fit_degradation(cleaned, race_label=f"{args.year} {args.race}")
    log_result(args.race, args.year, results)