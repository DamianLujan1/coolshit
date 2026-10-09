"""Fetching, caching, normalising and validating nflverse data via nflreadpy.

Cache policy: one parquet per (dataset, season) under data/cache. Completed
seasons are cached forever. The current season and the schedules file are
re-downloaded when `refresh=True` (the weekly command does this); if the
download fails and a cached copy exists, we fall back to it with a warning so
the pipeline still works offline.

Only schedules, PBP, team stats and player stats are allowed. FTN charting and
participation data (including their share-alike inputs) are never requested.
Attribution lives in the README and on the site. Weekly stats are downloaded
for coverage/validation, not added to the feature matrix or used as QB news.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import logging
import uuid
import zoneinfo
from pathlib import Path

import polars as pl

from . import config

log = logging.getLogger(__name__)
ALLOWED_DATASETS = frozenset({"schedules", "pbp", "team_stats", "player_stats"})


class DataValidationError(RuntimeError):
    """Raised when the data is not in a state we are willing to model on."""


# --------------------------------------------------------------------------- #
# Fetch + cache
# --------------------------------------------------------------------------- #

def _cache_path(name: str, season: int | None = None) -> Path:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.CACHE_DIR / (f"{name}_{season}.parquet" if season else f"{name}.parquet")


def _download(name: str, season: int | None) -> pl.DataFrame:
    if name not in ALLOWED_DATASETS:
        raise ValueError(f"dataset is not allowed: {name}")
    import nflreadpy as nfl
    from nflreadpy.config import update_config

    update_config(cache_mode="off", verbose=False)
    if name == "schedules":
        return nfl.load_schedules(True)  # every season; it's a 500 KB file
    if name == "pbp":
        return nfl.load_pbp(season)
    if name == "team_stats":
        return nfl.load_team_stats(season, summary_level="week")
    if name == "player_stats":
        return nfl.load_player_stats(season, summary_level="week")
    raise ValueError(name)


def _load_cached(name: str, season: int | None, refresh: bool, *, offline: bool = False) -> pl.DataFrame:
    if name not in ALLOWED_DATASETS:
        raise ValueError(f"dataset is not allowed: {name}")
    path = _cache_path(name, season)
    if path.exists() and (offline or not refresh):
        return pl.read_parquet(path)
    if offline:
        raise FileNotFoundError(f"offline cache miss: {path}; run `gridiron fetch` while online")
    try:
        df = _download(name, season)
    except Exception as exc:  # network down, GitHub hiccup, etc.
        if path.exists():
            log.warning("Download of %s %s failed (%s); using cached copy from %s",
                        name, season or "", exc, dt.datetime.fromtimestamp(path.stat().st_mtime))
            return pl.read_parquet(path)
        raise
    # Validate before replacing a usable cache; publish each file atomically.
    _validate_download_schema(name, df)
    tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        df.write_parquet(tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    metadata = {
        "dataset": name, "season": season,
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": f"nflreadpy.load_{name}",
        "nflreadpy_version": importlib.metadata.version("nflreadpy"),
        "rows": df.height, "sha256": _sha256(path),
        "historical_snapshot": False,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return df


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_manifest() -> list[dict]:
    """Current cache provenance; an old cache mtime is never a claimed fetch time."""
    rows = []
    for path in sorted(config.CACHE_DIR.glob("*.parquet")):
        metadata_path = path.with_suffix(".json")
        dataset, separator, season_text = path.stem.rpartition("_")
        season = int(season_text) if separator and season_text.isdigit() else None
        dataset = dataset if season is not None else path.stem
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {
            "dataset": dataset, "season": season, "fetched_at": None, "historical_snapshot": False,
            "sha256": _sha256(path), "rows": pl.scan_parquet(path).select(pl.len()).collect().item(),
            "provenance_note": "Legacy cache; original retrieval time is unknown.",
        }
        rows.append({"file": path.name, **metadata})
    return rows


def load_schedules(refresh: bool = False, *, offline: bool = False) -> pl.DataFrame:
    """Schedules/games for every season, with normalised team codes and kickoff datetimes."""
    raw = _load_cached("schedules", None, refresh, offline=offline)
    df = raw.filter(pl.col("season").is_between(config.PRIOR_SEED_SEASON, config.CURRENT_SEASON))
    return normalise_schedules(df)


def load_pbp(seasons: list[int] | None = None, refresh_current: bool = False, *, offline: bool = False) -> pl.DataFrame:
    """Play-by-play for the given seasons, reduced to the columns we use."""
    seasons = seasons or [config.PRIOR_SEED_SEASON, *config.SEASONS]
    frames = []
    for s in seasons:
        refresh = refresh_current and s == config.CURRENT_SEASON
        df = _load_cached("pbp", s, refresh, offline=offline)
        frames.append(select_pbp_columns(df))
    return pl.concat(frames, how="diagonal_relaxed")


def _load_stats(name: str, seasons: list[int] | None, refresh_current: bool, offline: bool) -> pl.DataFrame:
    seasons = seasons or [config.PRIOR_SEED_SEASON, *config.SEASONS]
    frames = [normalise_stats(_load_cached(name, s, refresh_current and s == config.CURRENT_SEASON,
                                          offline=offline), name)
              for s in seasons]
    return pl.concat(frames, how="diagonal_relaxed")


def load_team_stats(seasons: list[int] | None = None, refresh_current: bool = False, *, offline: bool = False) -> pl.DataFrame:
    """Weekly team stats, retained for source coverage and validation only."""
    return _load_stats("team_stats", seasons, refresh_current, offline)


def load_player_stats(seasons: list[int] | None = None, refresh_current: bool = False, *, offline: bool = False) -> pl.DataFrame:
    """Weekly player stats; never treated as timestamped pregame starter news."""
    return _load_stats("player_stats", seasons, refresh_current, offline)


PBP_COLUMNS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team",
    "play_type", "pass", "rush", "qb_dropback", "qb_scramble", "down", "epa", "qb_epa", "success",
    "passer_player_id", "rusher_player_id",
]
SCHEDULE_COLUMNS = [
    "game_id", "season", "week", "game_type", "gameday", "gametime", "home_team", "away_team",
    "home_score", "away_score", "result", "location", "home_rest", "away_rest", "div_game",
]
STATS_COLUMNS = ["season", "week", "season_type"]


def _require_columns(df: pl.DataFrame, columns: list[str], name: str) -> None:
    if missing := sorted(set(columns) - set(df.columns)):
        raise DataValidationError(f"{name} is missing expected columns: {missing}")


def _validate_download_schema(name: str, df: pl.DataFrame) -> None:
    if name == "schedules":
        _require_columns(df, SCHEDULE_COLUMNS, name)
    elif name == "pbp":
        _require_columns(df, PBP_COLUMNS, name)
    else:
        normalise_stats(df, name)
    if df.is_empty():
        raise DataValidationError(f"{name} download is empty")


def normalise_stats(df: pl.DataFrame, name: str) -> pl.DataFrame:
    if "team" not in df.columns and "recent_team" in df.columns:
        df = df.rename({"recent_team": "team"})
    required = [*STATS_COLUMNS, "team"] + (["player_id"] if name == "player_stats" else [])
    _require_columns(df, required, name)
    team_columns = [c for c in ("team", "opponent_team") if c in df.columns]
    return df.with_columns([pl.col(c).replace(config.TEAM_RENAMES) for c in team_columns])


def select_pbp_columns(df: pl.DataFrame) -> pl.DataFrame:
    _require_columns(df, PBP_COLUMNS, "play-by-play")
    # Keep true play identifiers when present to detect duplicate cached records.
    out = df.select([*PBP_COLUMNS, *[c for c in ("play_id", "game_end") if c in df.columns]])
    for col in ("posteam", "defteam", "home_team", "away_team"):
        out = out.with_columns(pl.col(col).replace(config.TEAM_RENAMES))
    return out


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

def normalise_schedules(df: pl.DataFrame) -> pl.DataFrame:
    """Rename relocated franchises, add kickoff (UTC) and the home-win target."""
    _require_columns(df, SCHEDULE_COLUMNS, "schedules")
    # These optional inputs must have a stable schema for downstream benchmark
    # reporting. Missing market prices remain null, never substitute features.
    optional = {"spread_line": pl.Float64, "home_moneyline": pl.Float64, "away_moneyline": pl.Float64,
                "home_qb_id": pl.String, "away_qb_id": pl.String}
    df = df.with_columns([pl.lit(None, dtype=dtype).alias(c)
                          for c, dtype in optional.items() if c not in df.columns])
    missing_kickoff = df.filter(pl.col("gameday").is_null() | pl.col("gametime").is_null()
                               | (pl.col("gameday") == "") | (pl.col("gametime") == ""))
    if missing_kickoff.height:
        raise DataValidationError(
            "schedules have unknown kickoff times; refresh after kickoff is confirmed before making picks: "
            f"{missing_kickoff['game_id'].head(6).to_list()}")
    df = df.with_columns(
        pl.col("home_team").replace(config.TEAM_RENAMES),
        pl.col("away_team").replace(config.TEAM_RENAMES),
    )
    kickoff = [
        kickoff_utc(d, t) for d, t in zip(df["gameday"].to_list(), df["gametime"].to_list())
    ]
    df = df.with_columns(
        pl.Series("kickoff_utc", kickoff, dtype=pl.Datetime("us", "UTC")),
        (pl.col("location") == "Neutral").cast(pl.Int8).alias("neutral"),
        (pl.col("game_type") != "REG").cast(pl.Int8).alias("playoff"),
        pl.when(pl.col("result").is_null()).then(None)
          .when(pl.col("result") > 0).then(1)
          .when(pl.col("result") < 0).then(0)
          .otherwise(None)  # ties carry no label
          .cast(pl.Int8).alias("home_win"),
        (pl.col("result") == 0).cast(pl.Int8).alias("tie"),
    )
    # nflverse schedules do not provide historical final-whistle/ingestion
    # timestamps. This is a conservative reconstructed availability proxy, not
    # proof that a Tuesday forecast could have downloaded these exact values.
    completed = pl.all_horizontal([pl.col(c).is_not_null() for c in ("result", "home_score", "away_score")])
    for status_col in ("game_status", "status"):
        if status_col in df.columns:
            status = pl.col(status_col).cast(pl.String).str.to_lowercase()
            completed = completed & (status.is_null() | status.is_in(["final", "completed", "complete", "closed", "f", "final/ot"]))
    df = df.with_columns(completed.alias("game_completed"))
    df = df.with_columns(
        pl.when(pl.col("game_completed")).then(pl.col("kickoff_utc") + pl.duration(hours=12))
          .otherwise(None).alias("history_available_at"),
        pl.when(pl.col("game_completed")).then(pl.col("home_win")).otherwise(None).alias("home_win"),
        pl.when(pl.col("game_completed")).then(pl.col("tie")).otherwise(None).alias("tie"),
    )
    return df.sort(["kickoff_utc", "game_id"])


def kickoff_utc(gameday: str, gametime: str) -> dt.datetime:
    tz = zoneinfo.ZoneInfo(config.KICKOFF_TZ)
    local = dt.datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
    return local.astimezone(dt.timezone.utc)


# --------------------------------------------------------------------------- #
# Validation: fail loudly
# --------------------------------------------------------------------------- #

def validate(schedules: pl.DataFrame, pbp: pl.DataFrame, now: dt.datetime | None = None, *,
             team_stats: pl.DataFrame | None = None, player_stats: pl.DataFrame | None = None) -> dict:
    """Raise DataValidationError on anything that would silently corrupt a run.

    Validate coverage, identity keys, score consistency and schemas. Odds are
    optional benchmark-only data: absent lines must never block a prediction.
    A just-started or explicitly in-progress game is not completed history.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        raise ValueError("validation now must include a timezone")
    problems: list[str] = []
    expected = set(range(config.PRIOR_SEED_SEASON, config.CURRENT_SEASON + 1))
    _require_columns(schedules, [*SCHEDULE_COLUMNS, "kickoff_utc"], "schedules")
    _require_columns(pbp, PBP_COLUMNS, "play-by-play")
    for c in ("epa", "qb_epa", "success"):
        if not pbp.schema[c].is_numeric() and pbp.schema[c] != pl.Null:
            raise DataValidationError(f"play-by-play {c} must be numeric, got {pbp.schema[c]}")

    def check_keys(frame, keys, label, nullable_keys=()):
        required_keys = [c for c in keys if c not in nullable_keys]
        if frame.filter(pl.any_horizontal([pl.col(c).is_null() for c in required_keys])).height:
            problems.append(f"{label} has null identity keys {required_keys}")
        n = frame.select(keys).is_duplicated().sum()
        if n:
            problems.append(f"{label} has {n} rows with duplicated identity keys {keys}")

    check_keys(schedules, ["game_id"], "schedules")
    if "play_id" in pbp.columns:
        check_keys(pbp, ["game_id", "play_id"], "play-by-play")
    elif pbp.filter(pl.col("game_id").is_null()).height:
        problems.append("play-by-play has null game_id")

    have_sched = set(schedules["season"].unique().to_list())
    have_pbp = set(pbp["season"].unique().to_list())
    if missing := sorted(expected - have_sched):
        problems.append(f"schedules missing seasons {missing}")
    completed = schedules.filter(pl.col("game_completed")) if "game_completed" in schedules.columns else schedules.filter(pl.col("result").is_not_null())
    available = completed.filter(
        (pl.col("history_available_at") <= now) if "history_available_at" in completed.columns
        else (pl.col("kickoff_utc") <= now - dt.timedelta(hours=12)))
    # PBP/stats do not exist for a season before its first completed game.
    expected_played = set(available["season"].unique().to_list())
    if missing := sorted(expected_played - have_pbp):
        problems.append(f"play-by-play missing seasons {missing}")

    known = set(config.TEAMS)
    for frame, cols, label in ((schedules, ("home_team", "away_team"), "schedules"),
                               (pbp, ("posteam", "defteam", "home_team", "away_team"), "pbp")):
        seen = set()
        for c in cols:
            seen |= set(frame[c].drop_nulls().unique().to_list())
        if unmapped := sorted(seen - known):
            problems.append(f"{label} has unmapped team abbreviations {unmapped}")

    # A game that kicked off more than 12 hours ago must have a score.
    stale = schedules.filter(
        pl.col("result").is_null() & (pl.col("kickoff_utc") < (now - dt.timedelta(hours=12)))
    )
    if stale.height:
        ids = stale["game_id"].to_list()
        problems.append(
            f"{len(ids)} game(s) have kicked off but are not scored yet: {ids[:6]}{'...' if len(ids) > 6 else ''}"
        )

    inconsistent = schedules.filter(
        pl.col("result").is_not_null() & (
            pl.col("home_score").is_null() | pl.col("away_score").is_null()
            | (pl.col("home_score") < 0) | (pl.col("away_score") < 0)
            | ((pl.col("home_score") - pl.col("away_score") - pl.col("result")).abs() > 1e-8)))
    if inconsistent.height:
        problems.append(f"schedules have inconsistent final scores: {inconsistent['game_id'].head(6).to_list()}")

    # Every completed and conservatively available game must have usable plays.
    scored_ids = set(available["game_id"].to_list())
    pbp_ids = set(pbp["game_id"].unique().to_list())
    if missing := sorted(scored_ids - pbp_ids):
        problems.append(f"{len(missing)} scored game(s) have no play-by-play: {missing[:6]}")
    usable_ids = set(pbp.filter(
        ((pl.col("pass") == 1) | (pl.col("rush") == 1)) & pl.col("epa").is_not_null()
        & pl.col("posteam").is_not_null())["game_id"].unique().to_list())
    if unusable := sorted(scored_ids.intersection(pbp_ids) - usable_ids):
        problems.append(f"completed game(s) have no usable EPA plays: {unusable[:6]}")
    if pbp.filter(pl.any_horizontal([
        pl.col(c).cast(pl.Float64).is_infinite() | pl.col(c).cast(pl.Float64).is_nan()
        for c in ("epa", "qb_epa")])).height:
        problems.append("play-by-play contains non-finite EPA values (NaN or infinity)")
    if pbp.filter(pl.col("success").is_not_null() & ~((pl.col("success") == 0) | (pl.col("success") == 1))).height:
        problems.append("play-by-play contains success outside {0, 1}")
    # PBP must agree with its schedule game identity. Unscheduled plays (e.g.
    # an abandoned fixture) are excluded by schedule joins and counted below.
    game_meta = pbp.select("game_id", "season", "week", "home_team", "away_team").unique()
    joined = game_meta.join(schedules.select("game_id", "season", "week", "home_team", "away_team"),
                            on="game_id", how="inner", suffix="_schedule")
    if joined.filter(pl.any_horizontal([pl.col(c) != pl.col(c + "_schedule")
                                        for c in ("season", "week", "home_team", "away_team")])).height:
        problems.append("play-by-play game identity disagrees with schedules")

    stats_counts = {}
    for frame, label in ((team_stats, "team_stats"), (player_stats, "player_stats")):
        if frame is None:
            continue
        frame = normalise_stats(frame, label)
        keys = [*STATS_COLUMNS, "team"] + (["player_id"] if label == "player_stats" else [])
        # nflverse includes Team/unattributed stat lines without a player_id.
        # Retain them in the source table and report the limitation; they are
        # never individual QB evidence or model features. Other key fields
        # remain mandatory, and duplicate complete keys still fail.
        check_keys(frame, keys, label, nullable_keys=["player_id"] if label == "player_stats" else [])
        if label == "player_stats":
            stats_counts["player_stats_unattributed_rows"] = frame["player_id"].null_count()
        if missing := sorted(expected_played - set(frame["season"].unique().to_list())):
            problems.append(f"{label} missing seasons {missing}")
        if unknown := sorted(set(frame["team"].drop_nulls().unique().to_list()) - known):
            problems.append(f"{label} has unmapped team abbreviations {unknown}")
        if frame.filter(~pl.col("season_type").is_in(["REG", "POST"])).height:
            problems.append(f"{label} has unsupported season_type (expected weekly REG/POST)")
        expected_team_games = pl.concat([
            available.select("season", "week", "game_type", pl.col("home_team").alias("team")),
            available.select("season", "week", "game_type", pl.col("away_team").alias("team")),
        ]).with_columns(pl.when(pl.col("game_type") == "REG").then(pl.lit("REG"))
                        .otherwise(pl.lit("POST")).alias("season_type")).drop("game_type")
        missing_stats = expected_team_games.join(frame.select([*STATS_COLUMNS, "team"]).unique(),
                                                on=[*STATS_COLUMNS, "team"], how="anti")
        if missing_stats.height:
            problems.append(f"{label} missing {missing_stats.height} completed team-game(s): {missing_stats.head(3).to_dicts()}")
        stats_counts[label] = frame.height

    # Full seasons must have the right number of regular-season games.
    for season, n in schedules.filter(pl.col("game_type") == "REG").group_by("season").len().iter_rows():
        want = 256 if season < 2021 else 272
        if season < config.CURRENT_SEASON and n < want - 1:
            problems.append(f"season {season} has {n} regular-season games, expected {want}")
        elif season < config.CURRENT_SEASON and n == want - 1:
            log.warning("season %s has %s regular-season games: one fewer than nominal; verify cancellation", season, n)

    if problems:
        raise DataValidationError("Data validation failed:\n  - " + "\n  - ".join(problems))

    upcoming = upcoming_week(schedules, now)
    return {"seasons": sorted(have_sched), "games": schedules.height, "plays": pbp.height,
            "upcoming": upcoming, "completed_available_games": available.height,
            "unscheduled_pbp_games_excluded": len(pbp_ids - set(schedules["game_id"].to_list())),
            "history_availability": "Reconstructed final score plus kickoff + 12h; not archived publication timestamps.",
            **stats_counts, "cache_manifest": cache_manifest()}


def upcoming_week(schedules: pl.DataFrame, now: dt.datetime | None = None) -> tuple[int, int] | None:
    """(season, week) of the earliest week with an unplayed game, or None if the season is over."""
    now = now or dt.datetime.now(dt.timezone.utc)
    future = schedules.filter(pl.col("result").is_null() & (pl.col("kickoff_utc") > now))
    if future.height == 0:
        return None
    row = future.sort(["season", "week"]).row(0, named=True)
    return int(row["season"]), int(row["week"])


def last_completed_week(schedules: pl.DataFrame, season: int) -> int | None:
    done = schedules.filter((pl.col("season") == season) & pl.col("result").is_not_null())
    return int(done["week"].max()) if done.height else None
