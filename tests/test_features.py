import numpy as np
import pytest
import polars as pl

from gridiron import config, features as F


def test_ewma_with_prior_starts_at_prior_and_moves_toward_data():
    out = F._ewma_with_prior(np.array([1.0, 1.0, 1.0]), prior=0.0, decay=0.85, prior_weight=4.0)
    assert out[0] == 0.0
    assert np.all(np.diff(out) > 0)
    assert out[1] == 1.0 / (1.0 + 4.0)
    # exact closed form for position 2: (0.85*1 + 1) / (0.85 + 1 + 4)
    assert np.isclose(out[2], (0.85 + 1) / (0.85 + 1 + 4))
    assert len(out) == 4


def test_team_game_stats_shapes_and_sides(pbp):
    stats = F.team_game_stats(pbp)
    played_games = pbp["game_id"].n_unique()
    assert stats.height == played_games * 2
    # a team's offense in a game equals the opponent's defense in that game
    g = stats.filter(pl.col("game_id") == stats["game_id"][0])
    a, b = g.row(0, named=True), g.row(1, named=True)
    assert np.isclose(a["off_pass_epa"], b["def_pass_epa"])
    assert np.isclose(b["off_sr"], a["def_sr"])
    assert set(F.TEAM_STATS) <= set(stats.columns)


def test_rolling_features_reflect_skill_and_prior(schedule, pbp):
    feats = F.build_game_features(schedule, pbp)
    # by week 3 of 2015 the good offense should be rated above the bad one
    wk3 = feats.filter((pl.col("season") == 2015) & (pl.col("week") == 3))
    side = pl.concat([
        wk3.select(pl.col("home_team").alias("team"), pl.col("home_off_pass_epa").alias("x")),
        wk3.select(pl.col("away_team").alias("team"), pl.col("away_off_pass_epa").alias("x")),
    ])
    rating = dict(zip(side["team"].to_list(), side["x"].to_list()))
    assert rating["AAA"] > rating["BBB"] > rating["CCC"] > rating["DDD"]
    # week 1 of 2015 is pure prior: last season's team mean shrunk to the league mean
    wk1 = feats.filter((pl.col("season") == 2015) & (pl.col("week") == 1))
    stats = F.team_game_stats(pbp)
    s14 = stats.filter(pl.col("season") == 2014)
    league = s14["off_pass_epa"].mean()
    for r in wk1.iter_rows(named=True):
        team_mean = s14.filter(pl.col("team") == r["home_team"])["off_pass_epa"].mean()
        expected = config.PRIOR_SHRINK * team_mean + (1 - config.PRIOR_SHRINK) * league
        assert np.isclose(r["home_off_pass_epa"], expected)


def test_flags_and_unplayed_games_get_features(schedule, pbp):
    feats = F.build_game_features(schedule, pbp)
    wk3 = feats.filter((pl.col("season") == 2015) & (pl.col("week") == 3))
    assert wk3["away_bye"].to_list() == [1, 1]
    assert wk3["rest_diff"].to_list() == [-7.0, -7.0]
    wk4 = feats.filter((pl.col("season") == 2015) & (pl.col("week") == 4))  # unplayed
    assert wk4["home_win"].is_null().all()
    assert wk4.select(F.STAGE1_FEATURES).null_count().sum_horizontal().item() == 0
    assert F.modeling_frame(feats).filter(pl.col("week") == 4).height == 0


def test_qb_feature_uses_only_prior_games_and_falls_back(schedule, pbp):
    qbs = F.qb_game_stats(pbp)
    starters = F.resolve_starters(schedule, qbs)
    # the seed season has no prior of its own, so asking for its rows must refuse, not guess
    with pytest.raises(ValueError):
        F.rolling_qb_features(qbs, starters, schedule)
    starters = starters.join(schedule.select("game_id", "season"), on="game_id").filter(pl.col("season") >= 2015).drop("season")
    qf = F.rolling_qb_features(qbs, starters, schedule)
    f15 = qf.join(schedule.select("game_id", "season", "week"), on="game_id").filter(pl.col("season") == 2015)
    assert f15.filter(pl.col("week") == 1)["qb_prior_games"].to_list() == [4, 4, 4, 4]
    # a brand-new quarterback (no history) gets the replacement-level prior exactly
    sched2 = schedule.with_columns(
        pl.when(pl.col("game_id") == "2015_04_BBB_AAA").then(pl.lit("QB_NEW")).otherwise(pl.col("home_qb_id")).alias("home_qb_id"))
    starters2 = F.resolve_starters(sched2.filter(pl.col("season") >= 2015), qbs)
    qf2 = F.rolling_qb_features(qbs, starters2, sched2)
    row = qf2.filter((pl.col("game_id") == "2015_04_BBB_AAA") & (pl.col("team") == "AAA")).row(0, named=True)
    assert row["qb_id"] == "QB_NEW" and row["qb_prior_games"] == 0
    league14 = qbs.filter(pl.col("season") == 2014)
    prior = league14["qb_epa_sum"].sum() / league14["dropbacks"].sum() - config.QB_REPLACEMENT_PENALTY
    assert np.isclose(row["qb_epa"], prior)
