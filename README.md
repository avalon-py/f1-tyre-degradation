# f1-pit-strategy

Fits real tyre degradation curves from 2025 F1 race data, then simulates
every possible pit strategy to find the fastest one -- and checks it
against what teams actually did.

## Status

| Area | State |
|---|---|
| Ingestion (`ingestion/load_session.py`) | Built. Pulls lap-level data via FastF1, caches to disk, flags race quality (green-flag %, compounds used) before committing to a race. |
| Cleaning (`features/clean.py`) | Built. Drops in/out laps, non-green-flag laps, non-dry-compound laps; applies a per-race fuel correction (from the race's lap count). |
| Degradation fit (`models/degradation.py`) | Built. One linear regression per driver-stint, averaged (median) per compound per race. Logged to `experiments/degradation_results.jsonl`. |
| Strategy simulator (`models/simulate.py`) | Built. Brute-force search over compound pairs and pit lap; auto-excludes compounds with unreliable degradation fits. No safety car / traffic model yet -- see Roadmap. |
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

`pipeline/run_all_races.py` chains all four steps for every race in a season (or one race with `--race`), giving each race its own fuel effect and tyre parameters.

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

- **Fuel effect is an estimate, not measured from this data.** It is
  `110 kg / race laps * 0.03 s/kg` per race (~0.058 s/lap at Bahrain's 57
  laps, ~0.075 at Spa's 44). Both constants are educated guesses -- worth
  revisiting once degradation fits look stable, since an overcorrected or
  undercorrected fuel term would bias the slope.
- **Pit loss (22s) is a rough constant**, not race-specific. Should be
  measured from actual in-lap/out-lap deltas per circuit.
- **No safety car or traffic modeling.** The simulator assumes a clean,
  uninterrupted race, which real races rarely are. This is the first thing
  to add if there's time left (see Roadmap).
- **Linear degradation only.** Real tyre wear can have a "cliff" past a
  certain age. Worth checking residuals before trusting the model near the
  end of a stint.
- **Degradation is fit per driver-stint and averaged (median, not mean)**,
  not from one pooled regression across all drivers. Pooling confounds
  "faster driver" with "less tyre wear" -- e.g. if quicker cars happen to
  run longer stints, a pooled fit can end up flat or negative even when
  every individual stint shows clean positive degradation. Median is used
  over mean because per-stint slopes are heavy-tailed; one noisy short
  stint can swing a mean far more than it should.
- **The safety-car filter only catches drivers who pitted into the
  restart, not everyone else on track.** `features/clean.py` drops
  `TyreLife <= 2`, which removes a fresh-tyre stint's unrepresentative
  opening laps -- including a stint that happens to start right at an SC
  restart. But a driver who *stayed out* through the safety car on old
  tyres has a high `TyreLife` at that same moment, so this filter doesn't
  touch their lap at all, even though it's just as bunched and
  unrepresentative as the pitted drivers' laps. The correct fix is to drop
  by *lap number relative to when the SC/VSC was lifted* (available via
  FastF1's track status transitions or race control messages), for every
  driver on track at that moment, not by each driver's own `TyreLife`.
  Not implemented -- this is a real, specific, remaining gap, not
  something the current filter accidentally handles.
- ~~Literal `"nan"`/`"None"` compound strings (Miami, Belgian GP)~~ --
  fixed: `clean.py` now keeps only SOFT/MEDIUM/HARD laps (an allow-list),
  so stray strings and intermediate/wet tyres are dropped.
- **Bahrain 2025 HARD compound is excluded from the simulation** (see
  `models/simulate.py::load_degradation`). All 14 HARD stints in this race
  were confounded: some started right at the lap-32 safety car restart
  (bunched field, not at racing pace), and the rest were simply every
  driver's *final* stint with no more stops to make -- which invites a
  conserve-early/push-late pace that has nothing to do with tyre physics.
  Diagnosed by checking each stint's starting lap number and cross-
  referencing against the actual race (a real SC was called for track
  debris on lap 32; most front-runners took their last stop under it).
  There wasn't a clean subsample of HARD left to fit from in this
  particular race -- not a bug to keep patching, a real property of this
  race's strategy shape. `models/simulate.py` auto-excludes any compound
  whose stints weren't majority-positive on degradation
  (`--min-pct-positive`, default 0.4) rather than silently trusting a bad
  fit. Revisit once a race is added (Day 6) where HARD is used mid-race
  under green flag.

## Setup

```bash
pip install -r requirements.txt
```

## Running it

```bash
python -m ingestion.load_session --year 2025 --race Bahrain      # 1. pull + cache raw laps
python -m features.clean --year 2025 --race Bahrain              # 2. clean + fuel-correct (per-race, from lap count)
python -m models.degradation --year 2025 --race Bahrain          # 3. fit degradation curves
python -m models.simulate --year 2025 --race Bahrain --total-laps 57   # 4. find best strategy

# or run all four in one go, for one race or the whole 2025 calendar
# (total laps are read from the data; wet/too-messy races are skipped with a reason):
python -m pipeline.run_all_races --year 2025 --race Bahrain
python -m pipeline.run_all_races --year 2025
python -m pipeline.summarize_results
```

By default, `models.simulate` auto-selects whichever compounds passed the
reliability check and finds the best strategy among them -- it won't tell
you what a compound it silently excluded (or never even fit) would have
done. To explicitly ask "what if I use SOFT here," force it with
`--compounds`:

```bash
python -m models.simulate --year 2025 --race Bahrain --total-laps 57 --compounds SOFT MEDIUM
```

If SOFT isn't usable, this fails loudly with one of two specific reasons
instead of just quietly leaving it out: **never fit** (not enough long
stints existed to model it at all -- no threshold will fix this) or
**below reliability threshold** (it was fit, but most stints disagreed on
direction -- data exists, just noisy; overridable via
`--min-pct-positive`).

Run `pytest` for the test suite. Coverage is currently uneven: thorough
for `features/clean.py` (9 tests covering the lap-filtering and fuel-
correction logic), but `models/degradation.py` and `models/simulate.py`
have real branching logic (median-vs-mean, the reliability threshold, the
`--compounds` error paths) with no tests yet -- a known gap, not an
oversight to assume is covered.

## Validation: model vs. real strategy (Bahrain 2025)

The simulator's recommended strategy was compared against the three
podium finishers' actual stints (via FastF1's `session.laps`).

**Model recommendation:** MEDIUM (6 laps) -> SOFT (51 laps), flagged
`extrapolated_beyond_data` because no SOFT stint in the race ever ran
anywhere close to 51 laps (28 was the longest observed).

**Actual strategies:**

| Driver | Stint 1 | Stint 2 | Stint 3 |
|---|---|---|---|
| PIA (P1) | SOFT, 14 laps | MEDIUM, 18 laps | MEDIUM, 25 laps |
| RUS (P2) | SOFT, 13 laps | MEDIUM, 19 laps | SOFT, 25 laps |
| NOR (P3) | SOFT, 10 laps | MEDIUM, 22 laps | MEDIUM, 25 laps |

Three findings came out of this comparison, each pointing at a different
part of the pipeline:

1. **The extrapolation guard worked as intended.** All three drivers ran
   10-25 lap stints; nobody came close to 51 laps on one set. The model's
   own `max_observed_tyre_life` check flagged its own recommendation as
   untrustworthy before it was taken at face value -- the system caught
   the problem it was designed to catch.

2. **Compound order is backwards from reality, and it's a genuine
   confound, not a repeat of an earlier bug.** All three drivers started
   on SOFT and moved to MEDIUM. The model recommends the opposite, because
   its fitted SOFT intercept (99.35s) came out slower than MEDIUM's
   (98.74s) -- physically backwards, since softs are faster than mediums
   at equal tyre age by definition. The likely cause: SOFT only appears
   early in this race (laps 1-14ish, on a green, low-grip track) while
   MEDIUM mostly appears later (once the track has rubbered in). The model
   has no way to separate "track evolution made this lap faster" from
   "this compound is faster" -- it's the fuel-correction confound's mirror
   image (there: car weight changing over the race; here: track grip
   changing over the race), and it wasn't caught earlier because nothing
   in Days 2-4 isolates track evolution from compound choice.

3. **The simulator can't represent a safety-car-created extra stop.** All
   three actual strategies are 3-stint (2 pit stops), because the lap-32
   SC gave the field a free pit window. The simulator only ever searches
   one pit lap / two compounds -- it has no concept of a bonus stop, so it
   isn't just wrong here, it's structurally unable to consider the
   strategy the real teams actually ran. This is the same "no safety car
   modeling" limitation noted above, now with a concrete example of how
   much it can change the optimal answer.

None of these three were fixed as part of this project (out of time
budget), but each is diagnosed to a specific, named cause rather than left
as an unexplained mismatch -- see Roadmap for what fixing each would
involve.

## Roadmap

1. ~~Ingestion + quality check~~ -- done.
2. ~~Cleaning + fuel correction~~ -- done.
3. ~~Degradation fit per compound~~ -- done.
4. ~~Brute-force strategy simulator~~ -- done.
5. ~~Validate against a real race~~ -- done, see Validation section above.
   Found three concrete gaps: extrapolation beyond observed stint length
   (now auto-flagged), a track-evolution/compound confound in the SOFT vs
   MEDIUM comparison, and no support for safety-car-created extra stops.
6. **Run the full chain across the 2025 calendar**
   (`pipeline/run_all_races.py`, then `pipeline/summarize_results.py` for
   the season table) to check whether the track-evolution confound and
   weak degradation signal seen at Bahrain are Bahrain-specific or
   general. This is the cheap, automatable half of generalizing -- no new
   judgment calls, just the existing pipeline run across every round.
   Monaco excluded by default (special two-stop rule). Hand validation
   (the Day 5 real-vs-model comparison) stays scoped to 2-3 races, since
   that needs real thinking time per race and doesn't scale to a full
   season in the time remaining. Deliberately staying within 2025 rather
   than adding more seasons, to avoid reopening the regulation-consistency
   problem Day 1 was built to avoid.
7. Stretch, only if time allows, roughly in order of expected value: (a)
   cap simulated stint length at each compound's `max_observed_tyre_life`
   instead of just warning about it, so the simulator can't recommend an
   extrapolated strategy at all; (b) allow the simulator to search 3-stint
   strategies, since real races often use them; (c) measure pit loss
   per-race instead of a constant; (d) add a basic safety-car probability;
   (e) wrap `models.simulate` in a small Streamlit dashboard.

## Repo structure

```
ingestion/    load_session.py -- FastF1 pull + cache + quality check
features/     clean.py -- filtering + fuel correction
models/       degradation.py (per-compound fit), simulate.py (strategy search)
pipeline/     run_all_races.py (full chain incl. simulate, per race; --race for one)
              summarize_results.py (season-wide table from race_results.jsonl)
config/       races.json -- season, race list, compound-labeling notes
experiments/  degradation_results.jsonl -- one line per race fit
tests/        test_clean.py
notebooks/    EDA and validation writeups go here
data/         raw/ and processed/ parquet files (gitignored)
cache/        FastF1's own on-disk cache (gitignored, large)
```