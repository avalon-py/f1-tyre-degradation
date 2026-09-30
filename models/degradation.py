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


MIN_LAPS_PER_STINT = 6  # raised from 4 -- a slope fit on 4-5 laps is too easily
                          # swung by a single noisy lap (traffic, small error)


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
    Returns, per compound: mean intercept and slope across individual
    driver-stints, not one pooled regression across all drivers. Pooling
    everyone together confounds "faster driver" with "less tyre wear" --
    e.g. if quicker cars happen to run longer stints, the pooled fit can
    end up flat or even negative even when every individual stint shows
    clean positive degradation. Fitting per stint and averaging avoids
    that confound. n_stints_used tells you how many stints had enough
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

        # Median, not mean -- per-stint slopes are heavy-tailed (a single
        # messy stint can swing a mean far more than it should). Mean is
        # still reported for comparison/transparency, since a mean vs.
        # median gap is itself a useful "how skewed is this" signal.
        max_tyre_life = int(compound_group["TyreLife"].max())

        results[compound] = {
            "intercept_s": round(float(np.median(intercepts)), 3),
            "slope_s_per_lap": round(float(np.median(slopes)), 4),
            "slope_mean": round(float(np.mean(slopes)), 4),
            "slope_std": round(float(np.std(slopes)), 4),
            "n_stints_used": len(stint_fits),
            "n_laps": int(sum(f["n_laps"] for f in stint_fits)),
            "pct_stints_positive_slope": round(float((slopes > 0).mean()), 3),
            # Longest stint actually observed on this compound. The linear
            # fit is only trustworthy up to here -- beyond it, the
            # simulator is extrapolating past any real data (tyres can
            # fall off a cliff well before a linear model would predict).
            "max_observed_tyre_life": max_tyre_life,
        }
        logger.info(
            "%s%s: %.3fs base + %.4fs/lap degradation [median] (mean=%.4f, std=%.4f, "
            "stints=%d, n_laps=%d, %.0f%% of stints positive)",
            prefix, compound, results[compound]["intercept_s"], results[compound]["slope_s_per_lap"],
            results[compound]["slope_mean"], results[compound]["slope_std"],
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
