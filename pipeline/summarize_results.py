"""
Reads experiments/degradation_results.jsonl (built up by run_all_races.py)
and prints one row per race/compound -- the season-wide view. Run this
after a batch run to see whether Bahrain's findings (weak SOFT signal,
track-evolution confound) are circuit-specific or a general pattern.

Usage:
    python -m pipeline.summarize_results
"""

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


def load_all_results() -> pd.DataFrame:
    log_path = EXPERIMENTS_DIR / "degradation_results.jsonl"
    rows = []
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line)
            for compound, params in entry["compounds"].items():
                rows.append({"year": entry["year"], "race": entry["race"], "compound": compound, **params})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_all_results()

    # keep only the latest fit per (race, compound) in case a race was re-run
    df = df.drop_duplicates(subset=["year", "race", "compound"], keep="last")

    summary = df[[
        "race", "compound", "slope_s_per_lap", "pct_stints_positive_slope",
        "n_stints_used", "max_observed_tyre_life",
    ]].sort_values(["race", "compound"])

    pd.set_option("display.max_rows", None)
    print(summary.to_string(index=False))

    unreliable = summary[summary["pct_stints_positive_slope"] < 0.4]
    logger.info(
        "%d of %d race/compound fits fall below the 40%% reliability threshold "
        "used by models.simulate -- these are the ones worth a closer look, same "
        "way Bahrain HARD was diagnosed.",
        len(unreliable), len(summary),
    )
