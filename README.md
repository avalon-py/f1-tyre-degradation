# f1-pit-strategy

Fits real tyre degradation curves from 2025 F1 race data, then simulates
every possible pit strategy to find the fastest one -- and checks it
against what teams actually did.

## Status

| Area | State |
|---|---|
| Ingestion (`ingestion/load_session.py`) | Built. Pulls lap-level data via FastF1, caches to disk, flags race quality (green-flag %, compounds used) before committing to a race. |
| Cleaning (`features/clean.py`) | Built. Drops in/out laps, non-green-flag laps, missing-compound laps; applies a flat fuel correction. |
| Degradation fit (`models/degradation.py`) | Built. One linear regression per compound per race (tyre age -> lap time). Logged to `experiments/degradation_results.jsonl`. |
| Strategy simulator (`models/simulate.py`) | Built. Brute-force search over compound pairs and pit lap. No safety car / traffic model yet -- see Roadmap. |
| Validation against real races | Not yet built -- Day 5 (see Roadmap). |
| Multi-race generalization | Not yet built -- Day 6. |

## What it does

```
target: for a given race, which (compound, compound, pit lap) combination
        minimizes total race time, given how each compound actually
        degraded that day
```

## Pipeline

```
FastF1 API ──► ingestion/load_session.py ──► data/raw/*.parquet
                                                    │
                                                    ▼
                                        features/clean.py
                          (drop in/out laps, SC/VSC, fuel correction)
                                                    │
                                                    ▼
                                     models/degradation.py
                    (linear fit per compound, logs to experiments/*.jsonl)
                                                    │
                                                    ▼
                                       models/simulate.py
                        (brute-force search, fastest pit lap + compounds)
```

`pipeline/run_pipeline.py` chains all four steps for one race.

## Data

2025 F1 season only, via [FastF1](https://docs.fastf1.dev/). Lap-level data
(one row per driver per lap) is enough here -- tyre degradation is a
lap-to-lap trend, not something that needs 10Hz telemetry. See
`config/races.json` for which races are in scope and why.

Compounds are modeled as relative labels (SOFT/MEDIUM/HARD) per race, never
Pirelli's absolute C1-C5 grades, since the compound mapping changes by
circuit -- a "medium" at one track isn't the same physical tyre as a
"medium" at another.

## Known limitations

- **Fuel correction is a flat 0.03s/lap estimate**, not measured from this
  data. It's a starting assumption -- worth revisiting once degradation
  fits look stable, since an overcorrected or undercorrected fuel term
  would bias the slope.
- **Pit loss (22s) is a rough constant**, not race-specific. Should be
  measured from actual in-lap/out-lap deltas per circuit.
- **No safety car or traffic modeling.** The simulator assumes a clean,
  uninterrupted race, which real races rarely are. This is the first thing
  to add if there's time left (see Roadmap).
- **Linear degradation only.** Real tyre wear can have a "cliff" past a
  certain age. Worth checking residuals before trusting the model near the
  end of a stint.

## Setup

```bash
pip install -r requirements.txt
```

## Running it

```bash
python -m ingestion.load_session --year 2025 --race Bahrain      # 1. pull + cache raw laps
python -m features.clean --year 2025 --race Bahrain              # 2. clean + fuel-correct
python -m models.degradation --year 2025 --race Bahrain          # 3. fit degradation curves
python -m models.simulate --year 2025 --race Bahrain --total-laps 57   # 4. find best strategy

# or run all four in one go:
python -m pipeline.run_pipeline --year 2025 --race Bahrain --total-laps 57
```

Run `pytest` for the cleaning-logic tests.

## Roadmap

1. ~~Ingestion + quality check~~ -- done.
2. ~~Cleaning + fuel correction~~ -- done.
3. ~~Degradation fit per compound~~ -- done.
4. ~~Brute-force strategy simulator~~ -- done.
5. **Validate against a real race** -- compare the model's recommended
   strategy to what the top finishers actually ran; write up where it
   agrees and where it doesn't, and why.
6. **Repeat on 2-3 more races** (see `config/races.json`) to check the
   approach generalizes across circuit types.
7. Stretch, only if time allows: measure pit loss per-race instead of using
   a constant; add a basic safety-car probability to the simulator; wrap
   `models.simulate` in a small Streamlit dashboard.

## Repo structure

```
ingestion/    load_session.py -- FastF1 pull + cache + quality check
features/     clean.py -- filtering + fuel correction
models/       degradation.py (per-compound fit), simulate.py (strategy search)
pipeline/     run_pipeline.py -- chains ingestion -> clean -> degradation -> simulate
config/       races.json -- season, race list, compound-labeling notes
experiments/  degradation_results.jsonl -- one line per race fit
tests/        test_clean.py
notebooks/    EDA and validation writeups go here
data/         raw/ and processed/ parquet files (gitignored)
cache/        FastF1's own on-disk cache (gitignored, large)
```
