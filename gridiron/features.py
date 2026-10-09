"""Feature pipeline using information available before each prediction cutoff.

Historical cutoffs are Tuesday noon America/New_York before the week's first
kickoff. Rebuilt nflverse files are not historical data vintages: a scored game
becomes eligible at kickoff + 12 hours, a conservative completion/availability
proxy, not a verified publication timestamp. Latest schedule QB IDs are actual
starters and provide no evidence of what was known at a historical cutoff.

Pipeline
--------
1. `team_game_stats`: per (game, team) offensive and defensive rates from play-by-play.
2. `rolling_team_features`: for each team-game, an exponentially weighted average of the
   team's PRIOR games this season, blended with a preseason prior:

       est = (sum_i w^age_i * x_i + K * prior) / (sum_i w^age_i + K)

   where age_i counts games back (0 = most recent prior game), w = EWMA_DECAY,
   K = PRIOR_WEIGHT_GAMES, and prior = PRIOR_SHRINK * last_season_team_mean
   + (1 - PRIOR_SHRINK) * last_season_league_mean. With no prior games, est = prior.
3. `qb_game_stats` + `rolling_qb_features`: the same idea for the starting quarterback,
   weighted by dropbacks and carried across seasons, with a replacement-level prior.
4. `build_game_features`: joins home and away into one row per game with differences and flags.
"""

from __future__ import annotations

import datetime as dt
import logging
import zoneinfo

import numpy as np
import polars as pl

from . import config

log = logging.getLogger(__name__)

TEAM_STATS = [
    "off_pass_epa", "off_rush_epa", "off_sr", "off_early_epa",
    "def_pass_epa", "def_rush_epa", "def_sr", "def_early_epa",
]
DIFF_FEATURES = [f"d_{s}" for s in TEAM_STATS] + ["d_qb_epa"]
FLAG_FEATURES = ["rest_diff", "home_bye", "away_bye", "div_game", "neutral", "playoff"]
STAGE1_FEATURES = DIFF_FEATURES + FLAG_FEATURES
STAGE2_FEATURES = (
    STAGE1_FEATURES
    + [f"home_{s}" for s in TEAM_STATS] + [f"away_{s}" for s in TEAM_STATS]
    + ["home_qb_epa", "away_qb_epa"]
)

QB_PROVENANCE = [
    "qb_id", "qb_source", "qb_source_detail", "qb_available_at",
    "qb_observed_at", "qb_source_game_id",
]


def prediction_schedule(schedule: pl.DataFrame,
                        prediction_time: dt.datetime | None = None) -> pl.DataFrame:
    """Attach deterministic historical cutoffs and reconstructed availability.

    A live timestamp overrides only still-unscored games that have not kicked
    off. It never changes historical training features. The Tuesday cutoff is
    shared by an entire NFL week, including postponed games in that week.
    """
    if prediction_time is not None:
        if prediction_time.tzinfo is None or prediction_time.utcoffset() is None:
            raise ValueError("prediction_time must be timezone-aware")
        prediction_time = prediction_time.astimezone(dt.timezone.utc)
    tz = zoneinfo.ZoneInfo(config.KICKOFF_TZ)
    first = schedule.group_by(["season", "week"]).agg(pl.col("kickoff_utc").min())
    cutoffs = {}
    for season, week, kickoff in first.iter_rows():
        local = kickoff.astimezone(tz)
        day = local.date() - dt.timedelta(days=(local.weekday() - 1) % 7)
        cutoff = dt.datetime.combine(day, dt.time(12), tzinfo=tz)
        if cutoff >= local:
            cutoff -= dt.timedelta(days=7)
        cutoffs[(season, week)] = cutoff.astimezone(dt.timezone.utc)
    times, available = [], []
    for r in schedule.iter_rows(named=True):
        cutoff = cutoffs[(r["season"], r["week"])]
        if prediction_time is not None and r["result"] is None and r["kickoff_utc"] > prediction_time:
            cutoff = prediction_time
        times.append(cutoff)
        if r["result"] is None:
            available.append(None)
        elif "history_available_at" in r:
            available.append(r["history_available_at"])
        else:
            available.append(r["kickoff_utc"] + dt.timedelta(hours=12))
    return schedule.with_columns(
        pl.Series("prediction_time", times, dtype=pl.Datetime("us", "UTC")),
        pl.Series("history_available_at", available, dtype=pl.Datetime("us", "UTC")),
    )


def _completed_stats(stats: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """Join stats only to scored schedule games with a completion proxy."""
    return stats.join(
        schedule.select("game_id", "history_available_at")
        .filter(pl.col("history_available_at").is_not_null()),
        on="game_id", how="inner",
    )


# --------------------------------------------------------------------------- #
# 1. Per-game team stats from play-by-play
# --------------------------------------------------------------------------- #

def _real_plays(pbp: pl.DataFrame) -> pl.DataFrame:
    return pbp.filter(
        ((pl.col("pass") == 1) | (pl.col("rush") == 1))
        & pl.col("epa").is_not_null()
        & pl.col("posteam").is_not_null()
        & ~pl.col("play_type").is_in(["qb_kneel", "qb_spike"])
    )


def team_game_stats(pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per (game_id, team) with offensive and defensive per-play rates."""
    plays = _real_plays(pbp)

    def agg(side_col: str, prefix: str) -> pl.DataFrame:
        return plays.group_by(["game_id", "season", "week", side_col]).agg(
            pl.col("epa").filter(pl.col("pass") == 1).mean().alias(f"{prefix}_pass_epa"),
            pl.col("epa").filter(pl.col("rush") == 1).mean().alias(f"{prefix}_rush_epa"),
            pl.col("success").mean().alias(f"{prefix}_sr"),
            pl.col("epa").filter(pl.col("down").cast(pl.Int64).is_in([1, 2])).mean().alias(f"{prefix}_early_epa"),
            pl.len().alias(f"{prefix}_plays"),
        ).rename({side_col: "team"})

    off = agg("posteam", "off")
    de = agg("defteam", "def")
    out = off.join(de, on=["game_id", "season", "week", "team"], how="full", coalesce=True)
    return out.sort(["season", "week", "game_id", "team"])


# --------------------------------------------------------------------------- #
# 2. Rolling team features with a preseason prior
# --------------------------------------------------------------------------- #

def _ewma_with_prior(values: np.ndarray, prior: float, decay: float, prior_weight: float) -> np.ndarray:
    """Pre-game estimates. Position g uses values[:g] only; position 0 is the prior.

    Returns len(values) + 1 entries: the last one is the estimate after every value.
    """
    out = np.empty(len(values) + 1)
    num, den = 0.0, 0.0
    for g, v in enumerate(values):
        out[g] = (num + prior_weight * prior) / (den + prior_weight)
        num = num * decay + v
        den = den * decay + 1.0
    out[len(values)] = (num + prior_weight * prior) / (den + prior_weight)
    return out


def _season_means(stats: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    team = stats.group_by(["season", "team"]).agg([pl.col(s).mean() for s in TEAM_STATS])
    league = stats.group_by("season").agg([pl.col(s).mean() for s in TEAM_STATS])
    return team, league


def rolling_team_features(stats: pl.DataFrame, schedule: pl.DataFrame,
                          history_schedule: pl.DataFrame | None = None) -> pl.DataFrame:
    """Rolling team estimates from completed games available at prediction time.

    The full history schedule must include the seed year. A postponed earlier
    game contributes only after its completion proxy passes the cutoff.
    """
    if "prediction_time" not in schedule.columns:
        schedule = prediction_schedule(schedule)
    history_schedule = history_schedule if history_schedule is not None else schedule
    if "history_available_at" not in history_schedule.columns:
        history_schedule = prediction_schedule(history_schedule)
    completed = _completed_stats(stats, history_schedule)
    history: dict[tuple[str, int], list[dict]] = {}
    for r in completed.sort(["history_available_at", "game_id"]).iter_rows(named=True):
        history.setdefault((r["team"], r["season"]), []).append(r)
    prior_cache: dict[tuple[int, dt.datetime], tuple[dict, dict]] = {}
    out = []
    for game in schedule.iter_rows(named=True):
        season, cutoff = game["season"], game["prediction_time"]
        key = (season, cutoff)
        if key not in prior_cache:
            prev = completed.filter((pl.col("season") == season - 1)
                                    & (pl.col("history_available_at") <= cutoff))
            if prev.is_empty():
                raise ValueError(f"no league prior available for season {season}; fetch {season - 1}")
            team_means, league_means = _season_means(prev)
            prior_cache[key] = ({r["team"]: r for r in team_means.iter_rows(named=True)},
                                league_means.row(0, named=True))
        tm, league = prior_cache[key]
        for team in (game["home_team"], game["away_team"]):
            games = [r for r in history.get((team, season), [])
                     if r["history_available_at"] <= cutoff and r["game_id"] != game["game_id"]]
            values = []
            for stat in TEAM_STATS:
                prior = league[stat]
                if team in tm and tm[team][stat] is not None:
                    prior = config.PRIOR_SHRINK * tm[team][stat] + (1 - config.PRIOR_SHRINK) * prior
                previous = np.array([r[stat] for r in games if r[stat] is not None], dtype=float)
                values.append(_ewma_with_prior(previous, prior, config.EWMA_DECAY,
                                                config.PRIOR_WEIGHT_GAMES)[-1])
            out.append((game["game_id"], team, *values))
    return pl.DataFrame(out, schema=["game_id", "team", *TEAM_STATS], orient="row")


# --------------------------------------------------------------------------- #
# 3. Quarterback features
# --------------------------------------------------------------------------- #

def qb_game_stats(pbp: pl.DataFrame) -> pl.DataFrame:
    """Dropbacks and total qb_epa per (game, team, quarterback)."""
    drop = pbp.filter((pl.col("qb_dropback") == 1) & pl.col("qb_epa").is_not_null() & pl.col("posteam").is_not_null())
    drop = drop.with_columns(
        pl.when(pl.col("passer_player_id").is_not_null()).then(pl.col("passer_player_id"))
          .when(pl.col("qb_scramble") == 1).then(pl.col("rusher_player_id"))
          .otherwise(None).alias("qb_id")
    ).filter(pl.col("qb_id").is_not_null())
    return drop.group_by(["game_id", "season", "week", "posteam", "qb_id"]).agg(
        pl.len().alias("dropbacks"), pl.col("qb_epa").sum().alias("qb_epa_sum")
    ).rename({"posteam": "team"}).sort(["season", "week", "game_id", "team"])


def game_starters(qb_stats: pl.DataFrame) -> pl.DataFrame:
    """Retrospective dominant passer, usable only as a lagged completed-game proxy.

    This is deliberately not claimed to identify an announced or actual starter.
    An injury replacement can have the most dropbacks in a completed game.
    """
    return (qb_stats.sort(["game_id", "team", "dropbacks", "qb_id"], descending=[False, False, True, False])
            .group_by(["game_id", "team"], maintain_order=True).first()
            .select("game_id", "team", pl.col("qb_id").alias("starter_id")))


def _qb_evidence(qb_inputs: pl.DataFrame | None) -> dict[tuple[str, str], list[dict]]:
    """Validate separately captured evidence; latest schedule QB IDs never enter."""
    if qb_inputs is None:
        return {}
    required = {"game_id", "team", "qb_id", "available_at", "observed_at", "source"}
    if missing := required - set(qb_inputs.columns):
        raise ValueError(f"QB evidence is missing required provenance: {sorted(missing)}")
    result: dict[tuple[str, str], list[dict]] = {}
    seen = {}
    for r in qb_inputs.iter_rows(named=True):
        for name in ("game_id", "team", "qb_id", "source"):
            if not isinstance(r[name], str) or not r[name].strip():
                raise ValueError(f"QB evidence requires a nonempty {name}")
        for name in ("available_at", "observed_at"):
            if not isinstance(r[name], dt.datetime) or r[name].tzinfo is None or r[name].utcoffset() is None:
                raise ValueError(f"QB evidence {name} must be a timezone-aware timestamp")
            r[name] = r[name].astimezone(dt.timezone.utc)
        if r["available_at"] > r["observed_at"]:
            raise ValueError("QB evidence cannot be observed before it became available")
        key = (r["game_id"], r["team"], r["observed_at"])
        if key in seen and seen[key] != r["qb_id"]:
            raise ValueError("conflicting QB evidence at the same observation timestamp")
        seen[key] = r["qb_id"]
        result.setdefault((r["game_id"], r["team"]), []).append(r)
    for entries in result.values():
        entries.sort(key=lambda r: (r["observed_at"], r["available_at"], r["source"]))
    return result


def resolve_starters(schedule: pl.DataFrame, qb_stats: pl.DataFrame, *,
                     qb_inputs: pl.DataFrame | None = None) -> pl.DataFrame:
    """Resolve each QB solely from pre-prediction evidence or completed history.

    Supply the full schedule, including the seed season. Explicit evidence must
    have both an availability timestamp and a capture timestamp no later than
    the prediction cutoff. Otherwise use the team's dominant passer from its
    most recent eligible completed game, or a replacement prior when unknown.
    Target-game PBP and unversioned schedule QB IDs never resolve that game's QB.
    """
    if "prediction_time" not in schedule.columns:
        schedule = prediction_schedule(schedule)
    evidence = _qb_evidence(qb_inputs)
    actual = game_starters(_completed_stats(qb_stats, schedule))
    completed = actual.join(schedule.select("game_id", "history_available_at"), on="game_id")
    history: dict[str, list[dict]] = {}
    for r in completed.sort(["history_available_at", "game_id"]).iter_rows(named=True):
        history.setdefault(r["team"], []).append(r)
    rows = []
    for game in schedule.iter_rows(named=True):
        cutoff = game["prediction_time"]
        for team in (game["home_team"], game["away_team"]):
            explicit = [r for r in evidence.get((game["game_id"], team), [])
                        if r["available_at"] <= cutoff and r["observed_at"] <= cutoff]
            prior_games = [r for r in history.get(team, [])
                           if r["history_available_at"] <= cutoff and r["game_id"] != game["game_id"]]
            if explicit:
                r = explicit[-1]
                values = (r["qb_id"], "timestamped_pre_prediction", r["source"],
                          r["available_at"], r["observed_at"], None)
            elif prior_games:
                r = prior_games[-1]
                values = (r["starter_id"], "lagged_completed_game", "prior-game dominant passer; kickoff + 12h availability proxy",
                          r["history_available_at"], None, r["game_id"])
            else:
                values = (None, "replacement_no_history", "no eligible QB evidence or completed-game passer", None, None, None)
            rows.append((game["game_id"], team, *values))
    schema = {"game_id": pl.String, "team": pl.String, **{name: pl.String for name in QB_PROVENANCE}}
    schema.update({"qb_available_at": pl.Datetime("us", "UTC"), "qb_observed_at": pl.Datetime("us", "UTC")})
    return pl.DataFrame(rows, schema=schema, orient="row")


def rolling_qb_features(qb_stats: pl.DataFrame, starters: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """EPA/dropback estimate from the resolved QB's completed prior games.

    Historical games and the previous-season league replacement prior must be
    available at the prediction cutoff, not merely before the target kickoff.
    """
    if "prediction_time" not in schedule.columns:
        schedule = prediction_schedule(schedule)
    completed = _completed_stats(qb_stats, schedule)
    hist = (completed.group_by(["qb_id", "game_id", "history_available_at"])
            .agg(pl.col("dropbacks").sum(), pl.col("qb_epa_sum").sum())
            .sort(["qb_id", "history_available_at", "game_id"]))
    history: dict[str, list[dict]] = {}
    for r in hist.iter_rows(named=True):
        history.setdefault(r["qb_id"], []).append(r)
    prior_cache = {}
    rows = starters.join(schedule.select("game_id", "season", "prediction_time"), on="game_id", how="inner")
    estimates = []
    for r in rows.iter_rows(named=True):
        season, cutoff = r["season"], r["prediction_time"]
        key = (season, cutoff)
        if key not in prior_cache:
            previous = completed.filter((pl.col("season") == season - 1)
                                        & (pl.col("history_available_at") <= cutoff))
            if previous.is_empty():
                raise ValueError(f"no league QB prior for season {season}")
            prior_cache[key] = previous["qb_epa_sum"].sum() / previous["dropbacks"].sum() - config.QB_REPLACEMENT_PENALTY
        prior = prior_cache[key]
        num, den = config.QB_PRIOR_DROPBACKS * prior, config.QB_PRIOR_DROPBACKS
        games = [g for g in history.get(r["qb_id"], [])
                 if g["history_available_at"] <= cutoff and g["game_id"] != r["game_id"]]
        for age, game in enumerate(reversed(games)):
            w = config.QB_DECAY ** age
            num += w * game["qb_epa_sum"]
            den += w * game["dropbacks"]
        estimates.append((r["game_id"], r["team"], num / den, len(games)))
    values = pl.DataFrame(estimates, schema=["game_id", "team", "qb_epa", "qb_prior_games"], orient="row")
    return starters.join(values, on=["game_id", "team"], how="inner")


# --------------------------------------------------------------------------- #
# 4. Game matrix
# --------------------------------------------------------------------------- #

def build_game_features(schedule: pl.DataFrame, pbp: pl.DataFrame, *,
                        prediction_time: dt.datetime | None = None,
                        qb_inputs: pl.DataFrame | None = None) -> pl.DataFrame:
    """One row per game with every model feature and the label.

    `schedule` and `pbp` should include the seed season: it supplies priors and quarterback
    history but gets no rows of its own. Historical rows use the Tuesday cutoff;
    an explicit live prediction_time affects only future unplayed games.
    """
    schedule = prediction_schedule(schedule, prediction_time)
    stats = team_game_stats(pbp)
    have_prior = set(stats["season"].unique().to_list())
    target = schedule.filter((pl.col("season") - 1).is_in(sorted(have_prior)))
    team_feats = rolling_team_features(stats, target, schedule)
    qbs = qb_game_stats(pbp)
    starters = resolve_starters(schedule, qbs, qb_inputs=qb_inputs).filter(pl.col("game_id").is_in(target["game_id"].implode()))
    qb_feats = rolling_qb_features(qbs, starters, schedule)
    # Expose the QB actually used by the model, not the latest actual starter
    # from nflverse. This also prevents misleading *_qb_id_right join columns.
    schedule = target.drop([c for c in ("home_qb_id", "away_qb_id") if c in target.columns])

    side = team_feats.join(qb_feats, on=["game_id", "team"], how="left")

    def prefixed(prefix: str) -> pl.DataFrame:
        return side.rename({c: f"{prefix}_{c}" for c in side.columns if c not in ("game_id",)})

    home = prefixed("home").rename({"home_team": "home_team"})
    away = prefixed("away").rename({"away_team": "away_team"})

    df = (schedule.join(home, on=["game_id", "home_team"], how="left")
                  .join(away, on=["game_id", "away_team"], how="left"))
    df = df.with_columns(
        [(pl.col(f"home_{s}") - pl.col(f"away_{s}")).alias(f"d_{s}") for s in TEAM_STATS]
        + [
            (pl.col("home_qb_epa") - pl.col("away_qb_epa")).alias("d_qb_epa"),
            (pl.col("home_rest") - pl.col("away_rest")).cast(pl.Float64).alias("rest_diff"),
            (pl.col("home_rest") >= config.BYE_REST_DAYS).cast(pl.Int8).alias("home_bye"),
            (pl.col("away_rest") >= config.BYE_REST_DAYS).cast(pl.Int8).alias("away_bye"),
            pl.col("div_game").cast(pl.Int8),
        ]
    )
    nulls = df.filter(pl.any_horizontal([pl.col(c).is_null() for c in STAGE1_FEATURES]))
    if nulls.height:
        log.warning("%d games have null features (first: %s)", nulls.height, nulls["game_id"][:3].to_list())
    return df


def modeling_frame(features: pl.DataFrame) -> pl.DataFrame:
    """Played, non-tied games with complete features: the training universe."""
    return features.filter(
        pl.col("home_win").is_not_null()
        & pl.all_horizontal([pl.col(c).is_not_null() for c in STAGE2_FEATURES])
    )
