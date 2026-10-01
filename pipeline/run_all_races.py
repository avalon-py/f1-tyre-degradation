"""
Runs the whole chain -- load -> clean -> fit tyre degradation -> simulate
best strategy -- for every race in a season, and writes one record per race
to experiments/race_results.jsonl.

Each race gets its own fuel effect (from its lap count) and its own tyre
parameters per compound; the simulator only ever sees that one race's
numbers. A race that can't be modelled (wet, too many interruptions) is
recorded as "skipped" with a reason, and one that errors out is recorded as
"failed" -- the loop never stops.

A strategy needs two different compounds (F1 rule). A race with only one
trustworthy compound gets a second one *borrowed* from the season: the
median slope for that compound across races where it was trustworthy, at a
pace offset from the race's real compound. Those results are flagged.

FastF1's first download per race is slow (30-90s), so a full season on a
first pass can take 20-40 minutes. Cached after that.

Usage:
    python -m pipeline.run_all_races --year 2025
    python -m pipeline.run_all_races --year 2025 --race Bahrain    # one race
    python -m pipeline.run_all_races --year 2025 --skip "Monaco Grand Prix"
    python -m pipeline.run_all_races --year 2025 --borrow-only     # no re-download
"""

import argparse
import json
import logging
from pathlib import Path

import fastf1
import numpy as np

from features.clean import DRY_COMPOUNDS, clean_laps, fuel_s_per_lap
from ingestion.load_session import check_race_quality, load_race, save_raw
from models.degradation import fit_degradation, log_result
from models.simulate import find_best_strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

MIN_GREEN_FLAG_PCT = 0.7   # below this the degradation fit is too noisy to trust
MIN_PCT_POSITIVE = 0.4     # a compound is only simulated if this share of its
                           # stints actually showed degradation (see models.simulate)
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
MIN_RACES_TO_BORROW = 3    # need this many races of evidence for a season median


def get_race_names(year: int) -> list[str]:
    """Pulls the season calendar so we don't hardcode 24 race names by hand."""
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return schedule["EventName"].tolist()


def skip_reason(quality: dict) -> str | None:
    """Why this race shouldn't be modelled at all, or None if it's fine."""
    if WET_COMPOUNDS & set(quality["compounds_used"]):
        return "wet race (intermediate/wet tyres used)"
    if quality["pct_green_flag"] < MIN_GREEN_FLAG_PCT:
        return f"only {100 * quality['pct_green_flag']:.0f}% green-flag laps"
    return None


def run_one_race(year: int, race: str) -> dict:
    laps, track_status = load_race(year, race, "R")
    save_raw(laps, year, race, track_status)
    quality = check_race_quality(laps)

    # Winner's lap count = highest LapNumber in the data. Would be too low
    # for a race abandoned early, but those are skipped as wet/interrupted.
    total_laps = int(laps["LapNumber"].max())
    record = {
        "race": race,
        "total_laps": total_laps,
        "fuel_s_per_lap": round(fuel_s_per_lap(total_laps), 4),
        "pct_green_flag": quality["pct_green_flag"],
    }

    reason = skip_reason(quality)
    if reason:
        return {**record, "status": "skipped", "reason": reason}

    cleaned = clean_laps(laps, total_laps, track_status)
    degradation = fit_degradation(cleaned, race_label=f"{year} {race}")
    log_result(race, year, degradation)
    record["compounds"] = degradation

    usable = {
        name: params for name, params in degradation.items()
        if params["pct_stints_positive_slope"] >= MIN_PCT_POSITIVE
    }
    if len(usable) < 2:
        return {**record, "status": "skipped",
                "reason": f"only {len(usable)} trustworthy compound(s) fit, need 2"}

    best = find_best_strategy(usable, total_laps)
    return {**record, "status": "ok", "best_strategy": best}


def trustworthy(compounds: dict) -> dict:
    return {n: p for n, p in compounds.items()
            if p["pct_stints_positive_slope"] >= MIN_PCT_POSITIVE}


def season_medians(records: list[dict]) -> tuple[dict, dict, dict]:
    """Season-wide (slope, pace offset vs MEDIUM, longest observed stint) per
    compound, from races where the compound was trustworthy. Compounds with
    too little evidence are left out."""
    slopes, offsets, caps = {}, {}, {}
    for r in records:
        good = trustworthy(r.get("compounds", {}))
        for name, p in good.items():
            slopes.setdefault(name, []).append(p["slope_s_per_lap"])
            if "max_observed_tyre_life" in p:
                caps.setdefault(name, []).append(p["max_observed_tyre_life"])
            if "MEDIUM" in good:
                offsets.setdefault(name, []).append(
                    p["intercept_s"] - good["MEDIUM"]["intercept_s"])
    med = lambda d: {n: float(np.median(v)) for n, v in d.items() if len(v) >= MIN_RACES_TO_BORROW}
    return med(slopes), med(offsets), med(caps)


def borrow_missing(records: list[dict]) -> list[dict]:
    """For each race with exactly one trustworthy compound, borrow the other
    dry compounds from the season and simulate. Returns new records (the
    originals are not modified)."""
    slopes, offsets, caps = season_medians(records)
    new = []
    for r in records:
        good = trustworthy(r.get("compounds", {}))
        if r["status"] != "skipped" or len(good) != 1:
            continue
        anchor, own = next(iter(good.items()))
        if anchor not in offsets:
            logger.warning("%s: can't borrow, no season pace offset for %s", r["race"], anchor)
            continue
        borrowed = {
            c: {"intercept_s": round(own["intercept_s"] - offsets[anchor] + offsets[c], 3),
                "slope_s_per_lap": round(slopes[c], 4),
                # season median of the longest observed stint -> stint-length cap
                **({"max_observed_tyre_life": int(round(caps[c]))} if c in caps else {})}
            for c in DRY_COMPOUNDS - {anchor} if c in slopes and c in offsets
        }
        if not borrowed:
            logger.warning("%s: can't borrow, not enough season data", r["race"])
            continue
        best = find_best_strategy({anchor: own, **borrowed}, r["total_laps"])
        logger.info("%s: borrowed %s -> %s->%s @ lap %d", r["race"], sorted(borrowed),
                    best["compound_1"], best["compound_2"], best["pit_lap"])
        rec = {k: v for k, v in r.items() if k != "reason"}
        new.append({**rec, "status": "ok", "borrowed": borrowed, "best_strategy": best})
    return new


def fill_borrowed(year: int) -> None:
    """Reads the results log, and appends a borrowed-compound result for every
    race that only had one trustworthy compound."""
    log_path = EXPERIMENTS_DIR / "race_results.jsonl"
    latest = {}
    with open(log_path) as f:
        for line in f:
            r = json.loads(line)
            if r["year"] == year:
                latest[r["race"]] = r  # a later line replaces an earlier one
    with open(log_path, "a") as f:
        for rec in borrow_missing(list(latest.values())):
            f.write(json.dumps(rec) + "\n")


def run_all(year: int, only: list[str], skip: list[str]) -> None:
    races = get_race_names(year)
    if only:
        races = [r for r in races if any(o.lower() in r.lower() for o in only)]
    logger.info("Running %d races for %d", len(races), year)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENTS_DIR / "race_results.jsonl"

    for i, race in enumerate(races, 1):
        if race in skip:
            logger.info("[%d/%d] Skipping %s (in --skip list)", i, len(races), race)
            continue

        logger.info("[%d/%d] Running %s...", i, len(races), race)
        try:
            result = run_one_race(year, race)
        except Exception as e:
            logger.error("[%d/%d] FAILED on %s: %s", i, len(races), race, e)
            result = {"race": race, "status": "failed", "reason": str(e)}

        if result["status"] != "ok":
            logger.warning("[%d/%d] %s: %s -- %s", i, len(races), race,
                           result["status"], result["reason"])

        with open(log_path, "a") as f:
            f.write(json.dumps({"year": year, **result}) + "\n")

    fill_borrowed(year)
    logger.info("Done. Run `python -m pipeline.summarize_results` for the season table.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--race", type=str, nargs="*", default=[],
        help="Only run races whose name contains one of these (e.g. Bahrain).",
    )
    parser.add_argument(
        "--skip", type=str, nargs="*", default=["Monaco Grand Prix"],
        help="Exact FastF1 race names to skip. Monaco is skipped by default -- "
             "special two-stop rule, not comparable strategy dynamics.",
    )
    parser.add_argument(
        "--borrow-only", action="store_true",
        help="Don't run any races; just fill in borrowed-compound results from "
             "the existing race_results.jsonl.",
    )
    args = parser.parse_args()

    if args.borrow_only:
        fill_borrowed(args.year)
    else:
        run_all(args.year, args.race, args.skip)