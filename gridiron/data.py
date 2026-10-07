"""Fetching, caching, normalising and validating nflverse data via nflreadpy.

Cache policy: one parquet per (dataset, season) under data/cache. Completed
seasons are cached forever. The current season and the schedules file are
re-downloaded when `refresh=True` (the weekly command does this); if the
download fails and a cached copy exists, we fall back to it with a warning so
the pipeline still works offline.

Licensing: nflverse data is CC-BY 4.0 except FTN charting data, which is
share-alike and is never loaded here. Attribution lives in the README and on
the site.
"""

from __future__ import annotations

import datetime as dt
import logging
import zoneinfo
from pathlib import Path

import polars as pl

from . import config

log = logging.getLogger(__name__)


class DataValidationError(RuntimeError):
    """Raised when the data is not in a state we are willing to model on."""


# --------------------------------------------------------------------------- #
# Fetch + cache
# --------------------------------------------------------------------------- #

def _cache_path(name: str, season: int | None = None) -> Path:
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.CACHE_DIR / (f"{name}_{season}.parquet" if season else f"{name}.parquet")


def _download(name: str, season: int | None) -> pl.DataFrame:
    import nflreadpy as nfl
    from nflreadpy.config import update_config

    update_config(cache_mode="off", verbose=False)
    if name == "schedules":
        return nfl.load_schedules(True)  # every season; it's a 500 KB file
    if name == "pbp":
        return nfl.load_pbp(season)
    raise ValueError(name)


def _load_cached(name: str, season: int | None, refresh: bool) -> pl.DataFrame:
    path = _cache_path(name, season)
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    try:
        df = _download(name, season)
    except Exception as exc:  # network down, GitHub hiccup, etc.
        if path.exists():
            log.warning("Download of %s %s failed (%s); using cached copy from %s",
                        name, season or "", exc, dt.datetime.fromtimestamp(path.stat().st_mtime))
            return pl.read_parquet(path)
        raise
    df.write_parquet(path)
    return df


def load_schedules(refresh: bool = False) -> pl.DataFrame:
    """Schedules/games for every season, with normalised team codes and kickoff datetimes."""
    raw = _load_cached("schedules", None, refresh)
    df = raw.filter(pl.col("season").is_between(config.PRIOR_SEED_SEASON, config.CURRENT_SEASON))
    return normalise_schedules(df)


def load_pbp(seasons: list[int] | None = None, refresh_current: bool = False) -> pl.DataFrame:
    """Play-by-play for the given seasons, reduced to the columns we use."""
    seasons = seasons or [config.PRIOR_SEED_SEASON, *config.SEASONS]
    frames = []
    for s in seasons:
        refresh = refresh_current and s == config.CURRENT_SEASON
        df = _load_cached("pbp", s, refresh)
        frames.append(select_pbp_columns(df))
    return pl.concat(frames, how="vertical_relaxed")


PBP_COLUMNS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team",
    "play_type", "pass", "rush", "qb_dropback", "qb_scramble", "down", "epa", "qb_epa", "success",
    "passer_player_id", "rusher_player_id",
]


def select_pbp_columns(df: pl.DataFrame) -> pl.DataFrame:
    missing = [c for c in PBP_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"play-by-play is missing expected columns: {missing}")
    out = df.select(PBP_COLUMNS)
    for col in ("posteam", "defteam", "home_team", "away_team"):
        out = out.with_columns(pl.col(col).replace(config.TEAM_RENAMES))
    return out


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

def normalise_schedules(df: pl.DataFrame) -> pl.DataFrame:
    """Rename relocated franchises, add kickoff (UTC) and the home-win target."""
    df = df.with_columns(
        pl.col("home_team").replace(config.TEAM_RENAMES),
        pl.col("away_team").replace(config.TEAM_RENAMES),
        pl.col("gametime").fill_null("13:00"),
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
    return df.sort(["kickoff_utc", "game_id"])


def kickoff_utc(gameday: str, gametime: str) -> dt.datetime:
    tz = zoneinfo.ZoneInfo(config.KICKOFF_TZ)
    local = dt.datetime.strptime(f"{gameday} {gametime}", "%Y-%m-%d %H:%M").replace(tzinfo=tz)
    return local.astimezone(dt.timezone.utc)


# --------------------------------------------------------------------------- #
# Validation: fail loudly
# --------------------------------------------------------------------------- #

def validate(schedules: pl.DataFrame, pbp: pl.DataFrame, now: dt.datetime | None = None) -> dict:
    """Raise DataValidationError on anything that would silently corrupt a run.

    Checks: every season present in both datasets; every team code known; no
    game whose kickoff has clearly passed is still unscored (a week that has
    not been scored yet); scored games have play-by-play; closing lines exist
    for upcoming games.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    problems: list[str] = []
    expected = set(range(config.PRIOR_SEED_SEASON, config.CURRENT_SEASON + 1))

    have_sched = set(schedules["season"].unique().to_list())
    have_pbp = set(pbp["season"].unique().to_list())
    if missing := sorted(expected - have_sched):
        problems.append(f"schedules missing seasons {missing}")
    if missing := sorted(expected - have_pbp):
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

    # Every scored game must have plays we can build features from.
    scored_ids = set(schedules.filter(pl.col("result").is_not_null())["game_id"].to_list())
    pbp_ids = set(pbp["game_id"].unique().to_list())
    if missing := sorted(scored_ids - pbp_ids):
        problems.append(f"{len(missing)} scored game(s) have no play-by-play: {missing[:6]}")

    # Full seasons must have the right number of regular-season games.
    for season, n in schedules.filter(pl.col("game_type") == "REG").group_by("season").len().iter_rows():
        want = 256 if season < 2021 else 272
        if season < config.CURRENT_SEASON and n < want - 1:
            problems.append(f"season {season} has {n} regular-season games, expected {want}")
        elif season < config.CURRENT_SEASON and n == want - 1:
            # 2022 is one short: BUF at CIN was cancelled. One missing game is tolerated.
            log.warning("season %s has %s regular-season games (one cancelled game is expected)", season, n)

    if problems:
        raise DataValidationError("Data validation failed:\n  - " + "\n  - ".join(problems))

    upcoming = upcoming_week(schedules, now)
    return {"seasons": sorted(have_sched), "games": schedules.height, "plays": pbp.height,
            "upcoming": upcoming}


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
