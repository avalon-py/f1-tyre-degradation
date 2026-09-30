"""
Reads experiments/race_results.jsonl (built by run_all_races.py) and prints
one row per race: its lap count, fuel effect, tyre slopes per compound, and
the simulated best strategy (or why the race was skipped).

Usage:
    python -m pipeline.summarize_results
"""

import json
from pathlib import Path

import pandas as pd

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


def load_latest_per_race() -> pd.DataFrame:
    rows = {}
    with open(EXPERIMENTS_DIR / "race_results.jsonl") as f:
        for line in f:
            r = json.loads(line)
            rows[(r["year"], r["race"])] = r  # a re-run replaces the earlier row

    table = []
    for r in rows.values():
        compounds = r.get("compounds", {})
        best = r.get("best_strategy")
        table.append({
            "race": r["race"],
            "laps": r.get("total_laps"),
            "fuel_s/lap": r.get("fuel_s_per_lap"),
            **{f"{c[0]}_slope": compounds[c]["slope_s_per_lap"] if c in compounds else None
               for c in ("SOFT", "MEDIUM", "HARD")},
            "best": (f"{best['compound_1']}->{best['compound_2']} @ lap {best['pit_lap']}"
                     if best else f"[{r['status']}] {r.get('reason', '')}"),
        })
    return pd.DataFrame(table)


if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    print(load_latest_per_race().to_string(index=False))