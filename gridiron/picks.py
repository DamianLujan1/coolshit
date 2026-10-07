"""Pick generation, the append-only pick log, grading and the running record.

The pick log (data/picks/picks.csv) is the public proof. Rules:
- a game is picked once; later runs never modify or re-pick it;
- a pick is refused if the game has already kicked off;
- the file is only ever opened in append mode, and `append_picks` checks that the existing
  bytes are unchanged before adding rows.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

import polars as pl

from . import __version__, config
from .evaluate import market_prob_home, spread_prob_home

PICK_COLUMNS = [
    "picked_at_utc", "season", "week", "game_id", "kickoff_utc", "home_team", "away_team",
    "p_home", "pick", "pick_prob", "confidence", "model_stage", "model_version", "n_train",
    "spread_line_at_pick", "home_moneyline_at_pick", "away_moneyline_at_pick",
]


def confidence_label(pick_prob: float) -> str:
    for threshold, label in config.CONFIDENCE_LABELS:
        if pick_prob >= threshold:
            return label
    return config.CONFIDENCE_LABELS[-1][1]


def make_picks(features: pl.DataFrame, p_home, season: int, week: int, stage: str, n_train: int,
               now: dt.datetime | None = None) -> pl.DataFrame:
    """Rows of PICK_COLUMNS for the games of (season, week) in `features`, with p_home aligned."""
    now = now or dt.datetime.now(dt.timezone.utc)
    df = features.with_columns(pl.Series("p_home", list(map(float, p_home))))
    rows = []
    for r in df.iter_rows(named=True):
        p = r["p_home"]
        pick = r["home_team"] if p >= 0.5 else r["away_team"]
        pick_prob = p if p >= 0.5 else 1 - p
        rows.append({
            "picked_at_utc": now.isoformat(timespec="seconds"),
            "season": season, "week": week, "game_id": r["game_id"],
            "kickoff_utc": r["kickoff_utc"].isoformat(timespec="seconds"),
            "home_team": r["home_team"], "away_team": r["away_team"],
            "p_home": round(p, 4), "pick": pick, "pick_prob": round(pick_prob, 4),
            "confidence": confidence_label(pick_prob),
            "model_stage": stage, "model_version": __version__, "n_train": n_train,
            "spread_line_at_pick": r.get("spread_line"),
            "home_moneyline_at_pick": r.get("home_moneyline"),
            "away_moneyline_at_pick": r.get("away_moneyline"),
        })
    return pl.DataFrame(rows, schema=PICK_SCHEMA, orient="row") if rows else empty_picks()


PICK_SCHEMA = {
    "picked_at_utc": pl.Utf8, "season": pl.Int64, "week": pl.Int64, "game_id": pl.Utf8, "kickoff_utc": pl.Utf8,
    "home_team": pl.Utf8, "away_team": pl.Utf8, "p_home": pl.Float64, "pick": pl.Utf8, "pick_prob": pl.Float64,
    "confidence": pl.Utf8, "model_stage": pl.Utf8, "model_version": pl.Utf8, "n_train": pl.Int64,
    "spread_line_at_pick": pl.Float64, "home_moneyline_at_pick": pl.Int64, "away_moneyline_at_pick": pl.Int64,
}


def empty_picks() -> pl.DataFrame:
    return pl.DataFrame(schema=PICK_SCHEMA)


def load_picks(path: Path = config.PICKS_FILE) -> pl.DataFrame:
    if not path.exists():
        return empty_picks()
    return pl.read_csv(path, schema_overrides=PICK_SCHEMA)


class PickLogError(RuntimeError):
    pass


def append_picks(new: pl.DataFrame, path: Path = config.PICKS_FILE, now: dt.datetime | None = None) -> int:
    """Append rows for games not already in the log. Returns the number of rows added.

    Refuses rows whose kickoff has passed, and never rewrites existing bytes.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_picks(path)
    already = set(existing["game_id"].to_list())
    before = _digest(path)

    rows = []
    for r in new.iter_rows(named=True):
        if r["game_id"] in already:
            continue
        kickoff = dt.datetime.fromisoformat(r["kickoff_utc"])
        if kickoff <= now:
            raise PickLogError(f"refusing to log a pick for {r['game_id']}: kickoff {kickoff} is not in the future")
        rows.append({c: r.get(c) for c in PICK_COLUMNS})
    if not rows:
        return 0
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PICK_COLUMNS)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    after = _digest(path, limit=before[1])
    if before[0] != after[0]:
        raise PickLogError("pick log prefix changed during append; investigate before trusting the log")
    return len(rows)


def _digest(path: Path, limit: int | None = None) -> tuple[str, int]:
    if not path.exists():
        return hashlib.sha256(b"").hexdigest(), 0
    data = path.read_bytes()
    if limit is not None:
        data = data[:limit]
    return hashlib.sha256(data).hexdigest(), len(data)


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #

def grade(picks: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """Join picks to final scores. Baselines are graded on the closing line in the schedules file.

    Columns added: result, winner, graded (bool), model_hit, home_hit, spread_hit, market_hit
    (1 hit, 0 miss, null push/ungraded), and Brier contributions for the model and market.
    """
    if picks.height == 0:
        return picks.with_columns(pl.lit(None).alias("result"))
    sched = schedules.select(
        "game_id", "result", "home_score", "away_score",
        pl.col("spread_line").alias("closing_spread"),
        pl.col("home_moneyline").alias("closing_home_ml"),
        pl.col("away_moneyline").alias("closing_away_ml"),
    )
    df = picks.with_columns(pl.col("p_home").cast(pl.Float64)).join(sched, on="game_id", how="left")
    winner = (pl.when(pl.col("result") > 0).then(pl.col("home_team"))
              .when(pl.col("result") < 0).then(pl.col("away_team"))
              .otherwise(None))
    spread_fav = pl.when(pl.col("closing_spread") < 0).then(pl.col("away_team")).otherwise(pl.col("home_team"))
    market_fav = (pl.when(pl.col("closing_away_ml") < pl.col("closing_home_ml")).then(pl.col("away_team"))
                  .otherwise(pl.col("home_team")))
    market_p = [market_prob_home(h, a) for h, a in zip(df["closing_home_ml"].to_list(), df["closing_away_ml"].to_list())]
    spread_p = [spread_prob_home(s) for s in df["closing_spread"].to_list()]
    df = df.with_columns(
        winner.alias("winner"),
        pl.col("result").is_not_null().alias("graded"),
        pl.Series("p_market_home", market_p, dtype=pl.Float64),
        pl.Series("p_spread_home", spread_p, dtype=pl.Float64),
        spread_fav.alias("spread_pick"), market_fav.alias("market_pick"),
    )
    home_win = pl.when(pl.col("result") > 0).then(1.0).when(pl.col("result") < 0).then(0.0).otherwise(None)

    def hit(col):
        return pl.when(pl.col("winner").is_null()).then(None).otherwise((pl.col(col) == pl.col("winner")).cast(pl.Int8))

    df = df.with_columns(
        hit("pick").alias("model_hit"),
        pl.when(pl.col("winner").is_null()).then(None).otherwise((pl.col("home_team") == pl.col("winner")).cast(pl.Int8)).alias("home_hit"),
        hit("spread_pick").alias("spread_hit"),
        hit("market_pick").alias("market_hit"),
        ((pl.col("p_home") - home_win) ** 2).alias("model_brier"),
        ((pl.col("p_market_home") - home_win) ** 2).alias("market_brier"),
        ((pl.col("p_spread_home") - home_win) ** 2).alias("spread_brier"),
    )
    return df


BASELINES = {"model": "Gridiron model", "home": "Always home", "spread": "Closing-spread favorite", "market": "Moneyline favorite"}


def _tally(g: pl.DataFrame, who: str) -> dict:
    col = f"{who}_hit"
    wins = int(g[col].sum() or 0) if g.height else 0
    graded = g.filter(pl.col(col).is_not_null())
    losses = graded.height - wins
    pushes = g.filter(pl.col("graded") & pl.col(col).is_null()).height
    out = {"wins": wins, "losses": losses, "pushes": pushes,
           "pct": round(wins / graded.height, 3) if graded.height else None}
    bcol = f"{who}_brier"
    if bcol in g.columns and graded.height:
        out["brier"] = round(float(graded[bcol].mean()), 4)
    return out


def running_record(graded: pl.DataFrame) -> dict:
    """Record for the model and each baseline, overall, per season and per week."""
    if graded.height == 0 or "graded" not in graded.columns:
        return {"overall": {k: _tally(empty_picks().with_columns(pl.lit(False).alias("graded")), "model") for k in BASELINES},
                "seasons": {}, "weeks": [], "n_graded": 0, "n_pending": graded.height}
    done = graded.filter(pl.col("graded"))
    rec = {
        "overall": {k: _tally(done, k) for k in BASELINES},
        "seasons": {},
        "weeks": [],
        "n_graded": done.height,
        "n_pending": graded.height - done.height,
        "labels": BASELINES,
    }
    for season in sorted(done["season"].unique().to_list()):
        g = done.filter(pl.col("season") == season)
        rec["seasons"][str(int(season))] = {k: _tally(g, k) for k in BASELINES}
        for week in sorted(g["week"].unique().to_list()):
            gw = g.filter(pl.col("week") == week)
            rec["weeks"].append({"season": int(season), "week": int(week), **{k: _tally(gw, k) for k in BASELINES}})
    return rec


def save_record(rec: dict, path: Path = config.RECORD_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2))
