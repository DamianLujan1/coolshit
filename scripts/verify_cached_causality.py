"""Offline causal regressions against the complete cached nflverse history.

Run from the repository root:
    .venv/bin/python scripts/verify_cached_causality.py

Only the requested JSON report is written. PBP mutations remain in memory;
the script does not fetch, train, edit pick histories, or publish anything.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from gridiron import config, data, features


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_hash(frame: pl.DataFrame) -> str:
    """Hash ordered inputs/features, with numerical noise rounded to 10 decimals."""
    def canonical(value):
        if isinstance(value, dt.datetime):
            return value.isoformat()
        if isinstance(value, float):
            if not math.isfinite(value):
                raise AssertionError("nonfinite feature in causal verification")
            return format(value, ".10f")
        return value

    rows = [{key: canonical(value) for key, value in row.items()}
            for row in frame.sort("game_id").to_dicts()]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=2023)
    parser.add_argument("--week", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("reports/causality-check.json"))
    args = parser.parse_args()

    schedule = data.load_schedules(offline=True)
    pbp = data.load_pbp(offline=True)
    target = schedule.filter((pl.col("season") == args.season) & (pl.col("week") == args.week))
    if target.is_empty() or not target["game_completed"].all():
        raise ValueError("choose a fully completed cached target week")
    first_kickoff = target["kickoff_utc"].min()
    target_ids = target["game_id"].to_list()
    affected_ids = schedule.filter(pl.col("kickoff_utc") >= first_kickoff)["game_id"].to_list()
    affected = pl.col("game_id").is_in(affected_ids)
    compared = ["game_id", *features.STAGE2_FEATURES, "prediction_time",
                *[f"{side}_{name}" for side in ("home", "away")
                  for name in [*features.QB_PROVENANCE, "qb_prior_games"]]]

    def snapshot(sched: pl.DataFrame, plays: pl.DataFrame) -> pl.DataFrame:
        return (features.build_game_features(sched, plays)
                .filter(pl.col("game_id").is_in(target_ids)).select(compared).sort("game_id"))

    qb_stats = features.qb_game_stats(pbp.filter(affected))
    dominant = features.game_starters(qb_stats).rename({"team": "posteam", "starter_id": "_dominant_qb"})

    def mutate(name: str) -> pl.DataFrame:
        if name == "passer_identities":
            return pbp.with_columns(
                pl.when(affected & pl.col("passer_player_id").is_not_null())
                  .then(pl.lit("FUTURE_ONLY_") + pl.col("posteam"))
                  .otherwise(pl.col("passer_player_id")).alias("passer_player_id"),
                pl.when(affected & (pl.col("qb_scramble") == 1))
                  .then(pl.lit("FUTURE_ONLY_") + pl.col("posteam"))
                  .otherwise(pl.col("rusher_player_id")).alias("rusher_player_id"),
            )
        if name == "dominant_passer_dropback_counts":
            identity = (pl.when(pl.col("passer_player_id").is_not_null()).then(pl.col("passer_player_id"))
                        .when(pl.col("qb_scramble") == 1).then(pl.col("rusher_player_id")).otherwise(None))
            return (pbp.join(dominant, on=["game_id", "posteam"], how="left")
                    .with_columns(pl.when(affected & (identity == pl.col("_dominant_qb")))
                                  .then(0).otherwise(pl.col("qb_dropback")).alias("qb_dropback"))
                    .drop("_dominant_qb"))
        if name == "truncate_target_and_future":
            return pbp.filter(~affected)
        raise ValueError(name)

    report = {
        "status": "running",
        "run_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "command": f".venv/bin/python scripts/verify_cached_causality.py --season {args.season} --week {args.week} --output {args.output}",
        "offline": True,
        "implementation_sha256": {str(path.relative_to(ROOT)): file_hash(path)
                                  for path in (ROOT / "gridiron/features.py", ROOT / "gridiron/data.py", Path(__file__))},
        "cache_sha256": {path.name: file_hash(path)
                         for path in sorted([config.CACHE_DIR / "schedules.parquet", *config.CACHE_DIR.glob("pbp_*.parquet")])},
        "input_counts": {"schedule_games": schedule.height, "pbp_rows": pbp.height,
                         "pbp_seasons": sorted(pbp["season"].unique().to_list()),
                         "affected_pbp_rows": pbp.filter(affected).height,
                         "affected_qb_dropbacks": int(pbp.filter(affected)["qb_dropback"].sum())},
        "target": {"season": args.season, "week": args.week, "games": target.height,
                   "game_ids": sorted(target_ids), "first_kickoff_utc": first_kickoff.isoformat()},
        "comparison": {"columns": compared, "absolute_tolerance": 1e-12, "relative_tolerance": 1e-12,
                       "hash_encoding": "ordered JSON; numeric values formatted to 10 decimal places; other inputs exact"},
        "checks": [],
        "limitations": "Rebuilt current nflverse files are not archived prediction-time data vintages. Historical completion/availability uses the documented kickoff + 12-hour proxy.",
    }
    reference = None
    for present in (True, False):
        sched = schedule if present else schedule.with_columns(
            pl.lit(None, dtype=pl.String).alias("home_qb_id"),
            pl.lit(None, dtype=pl.String).alias("away_qb_id"))
        base = snapshot(sched, pbp)
        if reference is not None:
            assert_frame_equal(reference, base, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)
        else:
            reference = base
            report["target"]["prediction_times_utc"] = sorted({t.isoformat() for t in base["prediction_time"]})
        base_hash = snapshot_hash(base)
        for name in ("passer_identities", "dominant_passer_dropback_counts", "truncate_target_and_future"):
            changed = mutate(name)
            got = snapshot(sched, changed)
            assert_frame_equal(base, got, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)
            difference = np.abs(base.select(features.STAGE2_FEATURES).to_numpy().astype(float)
                                - got.select(features.STAGE2_FEATURES).to_numpy().astype(float))
            changed_qbs = features.game_starters(features.qb_game_stats(changed.filter(affected)))
            before_qbs = features.game_starters(qb_stats)
            joined = before_qbs.join(changed_qbs, on=["game_id", "team"], how="left", suffix="_after")
            changed_dominance = joined.filter((pl.col("starter_id") != pl.col("starter_id_after"))
                                             | pl.col("starter_id_after").is_null()).height
            if changed_dominance == 0:
                raise AssertionError("mutation did not exercise any target/future QB identity/dominance")
            outcome = {"schedule_starters": "present" if present else "missing", "mutation": name,
                       "status": "passed", "target_games_compared": got.height, "mutated_pbp_rows": changed.height,
                       "affected_dropbacks_after": int(changed.filter(affected)["qb_dropback"].sum()),
                       "changed_or_removed_dominant_passers": changed_dominance,
                       "max_absolute_feature_difference": float(difference.max()),
                       "baseline_snapshot_sha256": base_hash, "mutated_snapshot_sha256": snapshot_hash(got)}
            outcome["snapshot_hash_matches"] = outcome["baseline_snapshot_sha256"] == outcome["mutated_snapshot_sha256"]
            if not outcome["snapshot_hash_matches"]:
                raise AssertionError("rounded causal snapshot hashes differ")
            report["checks"].append(outcome)
            print(f"PASS: {outcome['schedule_starters']} starters / {name}: {got.height} games; "
                  f"{changed_dominance} dominant passers altered/removed; max delta {difference.max():.3g}", flush=True)
    report["status"] = "passed"
    report["summary"] = {"checks_passed": len(report["checks"]), "target_games_per_check": target.height,
                         "all_snapshot_hashes_match": all(row["snapshot_hash_matches"] for row in report["checks"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
