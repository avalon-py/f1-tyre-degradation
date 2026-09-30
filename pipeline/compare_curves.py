"""
Compare linear vs quadratic degradation on races you've already cleaned
(needs data/processed/<year>_<race>_clean.parquet from `features.clean`).

    python -m pipeline.compare_curves --year 2025 --race Bahrain Qatar
    python -m pipeline.compare_curves --year 2025            # every cleaned race on disk

Prints, per race and compound: c with its bootstrap interval (and whether it
was clamped at 0), plus leave-one-stint-out error on held-out stints.
Appends to experiments/curve_comparison.jsonl. Send the printed output back.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from models.degradation_quadratic import fit_race, holdout_compare

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
LOG = ROOT / "experiments" / "curve_comparison.jsonl"


def main(year: int, races: list[str] | None) -> None:
    if races:
        paths = [(r, PROCESSED / f"{year}_{r.replace(' ', '_')}_clean.parquet") for r in races]
    else:
        paths = [(p.stem[len(str(year)) + 1:-len("_clean")].replace("_", " "), p)
                 for p in sorted(PROCESSED.glob(f"{year}_*_clean.parquet"))]
    LOG.parent.mkdir(exist_ok=True)
    for race, path in paths:
        if not path.exists():
            print(f"\n== {race}: {path.name} not found, skipped")
            continue
        cleaned = pd.read_parquet(path)
        fits = fit_race(cleaned)
        print(f"\n== {race} {year}")
        record = {"year": year, "race": race, "compounds": {}}
        for compound, f in fits.items():
            q, l = f["quadratic"], f["linear"]
            hold = holdout_compare(cleaned[cleaned["Compound"] == compound])
            print(f"\n{compound}: {q['n_stints_used']} stints, {q['n_laps']} laps, "
                  f"max age {q['max_observed_tyre_life']}")
            print(f"  linear   b={l['slope_s_per_lap']:+.4f} s/lap")
            print(f"  quadratic b={q['slope_s_per_lap']:+.4f}  c={q['quad_s_per_lap2']:+.5f} "
                  f"(90% CI {q['c_ci_low']:+.5f}..{q['c_ci_high']:+.5f}, "
                  f"clamped={q['c_clamped']}, clamped in {q['pct_boot_c_clamped']:.0%} of resamples)")
            if not hold.empty:
                print(hold.to_string(index=False))
            record["compounds"][compound] = {**f, "holdout": hold.to_dict("records")}
        with open(LOG, "a") as fh:
            fh.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--race", nargs="*", default=None)
    a = ap.parse_args()
    main(a.year, a.race)