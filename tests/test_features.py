import datetime as dt

import numpy as np
import pytest
import polars as pl
from polars.testing import assert_frame_equal

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
    # A genuinely timestamped new QB gets replacement-level history. Merely
    # changing the latest schedule actual starter cannot establish availability.
    cutoff = F.prediction_schedule(schedule).filter(pl.col("game_id") == "2015_04_BBB_AAA")["prediction_time"][0]
    evidence = pl.DataFrame([{"game_id": "2015_04_BBB_AAA", "team": "AAA", "qb_id": "QB_NEW",
                             "available_at": cutoff - dt.timedelta(hours=2),
                             "observed_at": cutoff - dt.timedelta(hours=1), "source": "saved pregame announcement"}])
    starters2 = F.resolve_starters(schedule, qbs, qb_inputs=evidence)
    starters2 = starters2.filter(pl.col("game_id").str.starts_with("2015_"))
    qf2 = F.rolling_qb_features(qbs, starters2, schedule)
    row = qf2.filter((pl.col("game_id") == "2015_04_BBB_AAA") & (pl.col("team") == "AAA")).row(0, named=True)
    assert row["qb_id"] == "QB_NEW" and row["qb_prior_games"] == 0
    league14 = qbs.filter(pl.col("season") == 2014)
    prior = league14["qb_epa_sum"].sum() / league14["dropbacks"].sum() - config.QB_REPLACEMENT_PENALTY
    assert np.isclose(row["qb_epa"], prior)
    assert row["qb_source"] == "timestamped_pre_prediction"
    assert row["qb_source_detail"] == "saved pregame announcement"
    assert row["qb_observed_at"] <= cutoff


def test_first_modeled_season_keeps_seed_qb_identity_and_provenance(schedule, pbp):
    feats = F.build_game_features(schedule, pbp)
    for r in feats.filter((pl.col("season") == 2015) & (pl.col("week") == 1)).iter_rows(named=True):
        for side in ("home", "away"):
            assert r[f"{side}_qb_id"] == f"QB_{r[f'{side}_team']}"
            assert r[f"{side}_qb_source"] == "lagged_completed_game"
            assert r[f"{side}_qb_source_game_id"].startswith("2014_")
            assert r[f"{side}_qb_prior_games"] == 4
            assert r[f"{side}_qb_available_at"] <= r["prediction_time"]
            assert r[f"{side}_qb_observed_at"] is None  # reconstructed, not a saved vintage


@pytest.mark.parametrize("late_field", ["available_at", "observed_at"])
def test_late_qb_evidence_is_ignored(schedule, pbp, late_field):
    qbs = F.qb_game_stats(pbp)
    game_id = "2015_03_CCC_BBB"
    cutoff = F.prediction_schedule(schedule).filter(pl.col("game_id") == game_id)["prediction_time"][0]
    item = {"game_id": game_id, "team": "BBB", "qb_id": "QB_FUTURE",
            "available_at": cutoff - dt.timedelta(hours=2),
            "observed_at": cutoff + dt.timedelta(hours=2), "source": "later capture"}
    if late_field == "available_at":
        item["available_at"] = cutoff + dt.timedelta(hours=1)
    got = F.resolve_starters(schedule, qbs, qb_inputs=pl.DataFrame([item]))
    row = got.filter((pl.col("game_id") == game_id) & (pl.col("team") == "BBB")).row(0, named=True)
    assert row["qb_id"] == "QB_BBB"
    assert row["qb_source"] == "lagged_completed_game"


def test_timestamped_qb_evidence_requires_both_capture_and_availability(schedule, pbp):
    evidence = pl.DataFrame([{"game_id": "2015_03_CCC_BBB", "team": "BBB", "qb_id": "QB_NEW",
                             "source": "latest schedules are not a timestamp"}])
    with pytest.raises(ValueError, match="required provenance"):
        F.resolve_starters(schedule, F.qb_game_stats(pbp), qb_inputs=evidence)


def test_missing_prior_passer_uses_replacement_even_with_latest_schedule_qb(schedule, pbp):
    # Unknown prior identity must never be filled from the target's actual starter.
    reduced = pbp.with_columns(pl.when(pl.col("posteam") == "AAA").then(None)
                              .otherwise(pl.col("passer_player_id")).alias("passer_player_id"))
    feats = F.build_game_features(schedule, reduced)
    row = feats.filter(pl.col("game_id") == "2015_04_BBB_AAA").row(0, named=True)
    assert row["home_qb_id"] is None
    assert row["home_qb_source"] == "replacement_no_history"
    assert row["home_qb_prior_games"] == 0
    assert np.isfinite(row["home_qb_epa"])


def test_completed_game_replacement_is_lagged_into_the_next_prediction(schedule, pbp):
    previous = "2015_02_DDD_CCC"
    changed = pbp.with_columns(
        pl.when((pl.col("game_id") == previous) & (pl.col("posteam") == "CCC")
                & pl.col("passer_player_id").is_not_null())
          .then(pl.lit("QB_REPLACEMENT_CCC")).otherwise(pl.col("passer_player_id")).alias("passer_player_id"))
    got = F.build_game_features(schedule, changed)
    current = got.filter(pl.col("game_id") == previous).row(0, named=True)
    following = got.filter(pl.col("game_id") == "2015_03_CCC_BBB").row(0, named=True)
    assert current["home_qb_id"] == "QB_CCC"  # cannot learn about its own replacement yet
    assert following["away_qb_id"] == "QB_REPLACEMENT_CCC"
    assert following["away_qb_source_game_id"] == previous
    assert following["away_qb_prior_games"] == 1


def test_live_cutoff_changes_only_unplayed_rows(schedule, pbp):
    base = F.build_game_features(schedule, pbp)
    now = dt.datetime(2015, 9, 24, 16, tzinfo=dt.timezone.utc)
    live = F.build_game_features(schedule, pbp, prediction_time=now)
    historical = base.filter(pl.col("result").is_not_null()).sort("game_id")
    rebuilt = live.filter(pl.col("result").is_not_null()).sort("game_id")
    assert_frame_equal(historical, rebuilt, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)
    assert live.filter(pl.col("result").is_null())["prediction_time"].to_list() == [now, now]
    with pytest.raises(ValueError, match="timezone-aware"):
        F.build_game_features(schedule, pbp, prediction_time=now.replace(tzinfo=None))
