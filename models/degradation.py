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


def fit_degradation(cleaned: pd.DataFrame) -> dict:
    """
    Returns, per compound: intercept (s), slope (s/lap of tyre age), and
    n_laps used. TyreLife is FastF1's tyre-age-in-laps column.
    """
    results = {}
    for compound, group in cleaned.groupby("Compound"):
        if len(group) < 5:
            logger.warning("Skipping %s: only %d laps, too few to fit", compound, len(group))
            continue

        X = group[["TyreLife"]].values
        y = group["LapTimeCorrected"].values

        model = LinearRegression().fit(X, y)
        results[compound] = {
            "intercept_s": round(float(model.intercept_), 3),
            "slope_s_per_lap": round(float(model.coef_[0]), 4),
            "n_laps": int(len(group)),
            "r_squared": round(float(model.score(X, y)), 3),
        }
        logger.info(
            "%s: %.3fs base + %.4fs/lap degradation (n=%d, R2=%.3f)",
            compound, results[compound]["intercept_s"], results[compound]["slope_s_per_lap"],
            results[compound]["n_laps"], results[compound]["r_squared"],
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

    results = fit_degradation(cleaned)
    log_result(args.race, args.year, results)
