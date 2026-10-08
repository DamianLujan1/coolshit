"""Feature pipeline. Every feature for a game uses only games that kicked off strictly before it.

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


def rolling_team_features(stats: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """One row per (game_id, team) in `schedule` with pre-game rolling estimates.

    `schedule` supplies the ordering (kickoff_utc) and includes unplayed games, which get
    features from the games before them. `stats` only has played games.
    """
    long = pl.concat([
        schedule.select("game_id", "season", "week", "kickoff_utc", pl.col("home_team").alias("team")),
        schedule.select("game_id", "season", "week", "kickoff_utc", pl.col("away_team").alias("team")),
    ]).join(stats.select(["game_id", "team", *TEAM_STATS]), on=["game_id", "team"], how="left")
    long = long.sort(["team", "kickoff_utc", "game_id"])

    team_means, league_means = _season_means(stats)
    tm = {(r["season"], r["team"]): r for r in team_means.iter_rows(named=True)}
    lm = {r["season"]: r for r in league_means.iter_rows(named=True)}

    pieces = []
    for (team, season), grp in long.group_by(["team", "season"], maintain_order=True):
        prev_team = tm.get((season - 1, team))
        prev_league = lm.get(season - 1)
        cols = {}
        for stat in TEAM_STATS:
            if prev_league is None:
                raise ValueError(f"no league prior available for season {season}; fetch {season - 1}")
            prior = prev_league[stat]
            if prev_team is not None and prev_team[stat] is not None:
                prior = config.PRIOR_SHRINK * prev_team[stat] + (1 - config.PRIOR_SHRINK) * prev_league[stat]
            vals = grp[stat].to_numpy().astype(float)
            # unplayed games (NaN) contribute nothing and do not advance the series
            played = ~np.isnan(vals)
            series = _ewma_with_prior(vals[played], prior, config.EWMA_DECAY, config.PRIOR_WEIGHT_GAMES)
            # each row's estimate = series at (number of played games strictly before it)
            idx = np.cumsum(played) - played.astype(int)
            cols[stat] = series[idx]
        pieces.append(grp.select(["game_id", "team"]).with_columns(
            [pl.Series(stat, cols[stat]) for stat in TEAM_STATS]
        ))
    feats = pl.concat(pieces)
    return feats


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
    """The quarterback with the most dropbacks for each (game, team): the de facto starter."""
    return (qb_stats.sort(["game_id", "team", "dropbacks", "qb_id"], descending=[False, False, True, False])
            .group_by(["game_id", "team"], maintain_order=True).first()
            .select("game_id", "team", pl.col("qb_id").alias("starter_id")))


def resolve_starters(schedule: pl.DataFrame, qb_stats: pl.DataFrame) -> pl.DataFrame:
    """Starting QB id for every (game, team).

    Priority: the schedule's listed starter (nflverse records the actual starter for played
    games and the projected starter for the next week), else the team's starter from its most
    recent played game. Returns game_id, team, qb_id.
    """
    listed = pl.concat([
        schedule.select("game_id", "season", "kickoff_utc", pl.col("home_team").alias("team"), pl.col("home_qb_id").alias("qb_id")),
        schedule.select("game_id", "season", "kickoff_utc", pl.col("away_team").alias("team"), pl.col("away_qb_id").alias("qb_id")),
    ])
    actual = game_starters(qb_stats)
    long = (listed.join(actual, on=["game_id", "team"], how="left")
            .with_columns(pl.coalesce(["starter_id", "qb_id"]).alias("qb_id"))
            .drop("starter_id")
            .sort(["team", "kickoff_utc", "game_id"]))
    # forward-fill within team so an upcoming game with no listed starter inherits the last one
    long = long.with_columns(pl.col("qb_id").forward_fill().over("team", order_by="kickoff_utc"))
    return long.select("game_id", "team", "qb_id")


def rolling_qb_features(qb_stats: pl.DataFrame, starters: pl.DataFrame, schedule: pl.DataFrame) -> pl.DataFrame:
    """Pre-game EPA per dropback estimate for each (game, team)'s starter.

    est = (sum w^age * epa_sum + K * prior) / (sum w^age * dropbacks + K), over the QB's prior
    games in any season. prior = last season's league EPA/dropback minus a replacement penalty.
    """
    league = (qb_stats.group_by("season")
              .agg((pl.col("qb_epa_sum").sum() / pl.col("dropbacks").sum()).alias("lg"))
              .to_dict(as_series=False))
    league_rate = dict(zip(league["season"], league["lg"]))

    # the QB's history, one row per game (a data glitch can list a passer for both teams), ordered by kickoff
    hist = (qb_stats.join(schedule.select("game_id", "kickoff_utc"), on="game_id", how="inner")
            .group_by(["qb_id", "game_id", "kickoff_utc"]).agg(pl.col("dropbacks").sum(), pl.col("qb_epa_sum").sum())
            .sort(["qb_id", "kickoff_utc", "game_id"]))
    hist_by_qb: dict[str, list[tuple[dt.datetime, int, float]]] = {}
    for r in hist.iter_rows(named=True):
        hist_by_qb.setdefault(r["qb_id"], []).append((r["kickoff_utc"], r["dropbacks"], r["qb_epa_sum"]))

    rows = starters.join(schedule.select("game_id", "season", "kickoff_utc"), on="game_id", how="inner")
    out = []
    for r in rows.iter_rows(named=True):
        season = r["season"]
        if season - 1 not in league_rate:
            raise ValueError(f"no league QB prior for season {season}")
        prior = league_rate[season - 1] - config.QB_REPLACEMENT_PENALTY
        num, den = config.QB_PRIOR_DROPBACKS * prior, config.QB_PRIOR_DROPBACKS
        games = [g for g in hist_by_qb.get(r["qb_id"], []) if g[0] < r["kickoff_utc"]]
        # most recent first so age 0 is the last game played
        for age, (_, n, s) in enumerate(reversed(games)):
            w = config.QB_DECAY ** age
            num += w * s
            den += w * n
        out.append((r["game_id"], r["team"], r["qb_id"], num / den, len(games)))
    return pl.DataFrame(out, schema=["game_id", "team", "qb_id", "qb_epa", "qb_prior_games"], orient="row")


# --------------------------------------------------------------------------- #
# 4. Game matrix
# --------------------------------------------------------------------------- #

def build_game_features(schedule: pl.DataFrame, pbp: pl.DataFrame) -> pl.DataFrame:
    """One row per game with every model feature and the label.

    `schedule` and `pbp` should include the seed season: it supplies priors and quarterback
    history but gets no rows of its own. Only games that kicked off before each game influence
    its row.
    """
    stats = team_game_stats(pbp)
    have_prior = set(stats["season"].unique().to_list())
    target = schedule.filter((pl.col("season") - 1).is_in(sorted(have_prior)))
    team_feats = rolling_team_features(stats, target)
    qbs = qb_game_stats(pbp)
    starters = resolve_starters(target, qbs)
    qb_feats = rolling_qb_features(qbs, starters, schedule)
    schedule = target

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
