"""
Reads experiments/race_results.jsonl (built by run_all_races.py) and prints
one row per race: its lap count, fuel effect, tyre slopes per compound, and
the simulated best strategy (or why the race was skipped). `stints` is the two
stint lengths; `flag` is NO-FIT-IN-DATA when no one-stop strategy fits inside
the observed stint lengths (the answer is pure extrapolation -- ignore it),
EXTRAP when a stint still exceeds the observed max, else ok.

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
            rows[(r["year"], r["race"])] = r

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
            "stints": f"{best['stint_1_length']}/{best['stint_2_length']}" if best else "",
            "flag": ("NO-FIT-IN-DATA" if best and best.get("cap_infeasible")
                     else "EXTRAP" if best and best.get("extrapolated_beyond_data") else
                     "ok" if best else ""),
            "best": (f"{best['compound_1']}->{best['compound_2']} @ lap {best['pit_lap']}"
                     + (f" *borrowed {sorted(r['borrowed'])}" if r.get("borrowed") else "")
                     if best else f"[{r['status']}] {r.get('reason', '')}"),
        })
    return pd.DataFrame(table)


if __name__ == "__main__":
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    print(load_latest_per_race().to_string(index=False))