"""Leakage is a critical bug. These tests prove that no feature for a game depends on that game or later ones."""

import datetime as dt

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from gridiron import features as F
from tests.conftest import make_pbp, make_schedule


def _features(schedule, pbp):
    return F.build_game_features(schedule, pbp)


def test_scrambling_a_game_and_everything_after_it_does_not_change_its_features():
    from gridiron.data import normalise_schedules

    sched = normalise_schedules(make_schedule(scored_through=(2015, 4)))
    pbp = make_pbp(sched)
    base = _features(sched, pbp)
    compare = F.STAGE2_FEATURES

    for target in sched.filter(pl.col("season") == 2015)["game_id"].to_list():
        kickoff = sched.filter(pl.col("game_id") == target)["kickoff_utc"][0]
        later_ids = sched.filter(pl.col("kickoff_utc") >= kickoff)["game_id"].to_list()
        # wreck every play from the target game onward: flip sign, add noise, change success
        wrecked = pbp.with_columns(
            pl.when(pl.col("game_id").is_in(later_ids)).then(-pl.col("epa") * 3 + 1.0).otherwise(pl.col("epa")).alias("epa"),
            pl.when(pl.col("game_id").is_in(later_ids)).then(-pl.col("qb_epa") * 3 + 1.0).otherwise(pl.col("qb_epa")).alias("qb_epa"),
            pl.when(pl.col("game_id").is_in(later_ids)).then(1 - pl.col("success")).otherwise(pl.col("success")).alias("success"),
        )
        # also wreck the scores of those games (they feed the label and the schedule-side priors)
        wrecked_sched = sched.with_columns(
            pl.when(pl.col("game_id").is_in(later_ids)).then(pl.col("result") * -1).otherwise(pl.col("result")).alias("result"))
        got = _features(wrecked_sched, wrecked)
        a = base.filter(pl.col("game_id") == target).select(compare).to_numpy()
        b = got.filter(pl.col("game_id") == target).select(compare).to_numpy()
        assert np.allclose(a.astype(float), b.astype(float), equal_nan=True), f"features for {target} changed when later data changed"


def test_features_do_change_when_an_earlier_game_changes():
    """Sanity check that the test above has teeth: earlier data must matter."""
    from gridiron.data import normalise_schedules

    sched = normalise_schedules(make_schedule(scored_through=(2015, 4)))
    pbp = make_pbp(sched)
    base = _features(sched, pbp)
    target = "2015_03_CCC_BBB"
    earlier = ["2015_02_DDD_CCC"]
    assert sched.filter(pl.col("game_id").is_in([target, *earlier])).height == 2
    wrecked = pbp.with_columns(pl.when(pl.col("game_id").is_in(earlier)).then(pl.col("epa") + 5.0).otherwise(pl.col("epa")).alias("epa"))
    got = _features(sched, wrecked)
    a = base.filter(pl.col("game_id") == target).select(F.STAGE1_FEATURES).to_numpy().astype(float)
    b = got.filter(pl.col("game_id") == target).select(F.STAGE1_FEATURES).to_numpy().astype(float)
    assert not np.allclose(a, b)


@pytest.mark.parametrize("listed_starters", [True, False])
@pytest.mark.parametrize("mutation", ["passer_identity", "dropback_counts", "truncation"])
def test_target_and_future_pbp_cannot_resolve_target_qb(listed_starters, mutation):
    """Exercise the identity leak, which EPA-only scrambling never detected."""
    from gridiron.data import normalise_schedules

    sched = normalise_schedules(make_schedule(scored_through=(2015, 4)))
    if not listed_starters:
        sched = sched.with_columns(pl.lit(None, dtype=pl.String).alias("home_qb_id"),
                                   pl.lit(None, dtype=pl.String).alias("away_qb_id"))
    pbp = make_pbp(sched)
    # One backup dropback per team/game lets a count-only mutation change the
    # target's dominant passer while keeping its set of passer identities fixed.
    backup = (pbp.filter(pl.col("qb_dropback") == 1)
              .group_by(["game_id", "posteam"], maintain_order=True).first()
              .with_columns((pl.lit("BACKUP_") + pl.col("posteam")).alias("passer_player_id")))
    pbp = pl.concat([pbp, backup.select(pbp.columns)])
    base = _features(sched, pbp)
    compare = F.STAGE2_FEATURES + ["prediction_time"] + [f"{side}_{c}" for side in ("home", "away") for c in F.QB_PROVENANCE]

    for target in sched.filter(pl.col("season") == 2015)["game_id"].to_list():
        kickoff = sched.filter(pl.col("game_id") == target)["kickoff_utc"][0]
        later = sched.filter(pl.col("kickoff_utc") >= kickoff)["game_id"].to_list()
        mask = pl.col("game_id").is_in(later)
        if mutation == "passer_identity":
            changed = pbp.with_columns(pl.when(mask & pl.col("passer_player_id").is_not_null())
                                       .then(pl.lit("TARGET_ONLY_QB"))
                                       .otherwise(pl.col("passer_player_id")).alias("passer_player_id"))
        elif mutation == "dropback_counts":
            changed = pbp.with_columns(pl.when(mask & ~pl.col("passer_player_id").str.starts_with("BACKUP_").fill_null(False))
                                       .then(0).otherwise(pl.col("qb_dropback")).alias("qb_dropback"))
        else:
            changed = pbp.filter(~mask)
        got = _features(sched, changed)
        a = base.filter(pl.col("game_id") == target).select(compare)
        b = got.filter(pl.col("game_id") == target).select(compare)
        assert_frame_equal(a, b, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)


@pytest.mark.parametrize("listed_starters", [True, False])
def test_latest_actual_schedule_starter_is_not_historical_evidence(listed_starters):
    from gridiron.data import normalise_schedules

    sched = normalise_schedules(make_schedule(scored_through=(2015, 4)))
    pbp = make_pbp(sched)
    base = _features(sched, pbp).sort("game_id")
    changed = sched.with_columns(pl.lit("SCHEDULE_FUTURE_QB" if listed_starters else None, dtype=pl.String).alias("home_qb_id"),
                                 pl.lit("SCHEDULE_FUTURE_QB" if listed_starters else None, dtype=pl.String).alias("away_qb_id"))
    got = _features(changed, pbp).sort("game_id")
    assert_frame_equal(base, got, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)


def test_game_played_after_tuesday_cutoff_cannot_enter_team_or_qb_history():
    from gridiron.data import normalise_schedules

    sched = normalise_schedules(make_schedule(scored_through=(2015, 4)))
    delayed = "2015_02_DDD_CCC"
    # Week 2's game occurs after the week-3 Tuesday cutoff but before kickoff.
    delayed_kickoff = dt.datetime(2015, 9, 16, 17, tzinfo=dt.timezone.utc)
    sched = sched.with_columns(
        pl.when(pl.col("game_id") == delayed).then(pl.lit(delayed_kickoff)).otherwise(pl.col("kickoff_utc")).alias("kickoff_utc"),
        pl.when(pl.col("game_id") == delayed).then(pl.lit(delayed_kickoff + dt.timedelta(hours=12)))
          .otherwise(pl.col("history_available_at")).alias("history_available_at"),
    )
    pbp = make_pbp(sched)
    base = _features(sched, pbp)
    changed = pbp.with_columns(
        pl.when(pl.col("game_id") == delayed).then(pl.lit("DELAYED_QB")).otherwise(pl.col("passer_player_id")).alias("passer_player_id"),
        pl.when(pl.col("game_id") == delayed).then(pl.lit(20.0)).otherwise(pl.col("epa")).alias("epa"),
        pl.when(pl.col("game_id") == delayed).then(pl.lit(20.0)).otherwise(pl.col("qb_epa")).alias("qb_epa"),
    )
    got = _features(sched, changed)
    target = "2015_03_CCC_BBB"
    assert_frame_equal(base.filter(pl.col("game_id") == target), got.filter(pl.col("game_id") == target),
                       check_exact=False, rel_tol=1e-12, abs_tol=1e-12)
    row = got.filter(pl.col("game_id") == target).row(0, named=True)
    assert row["prediction_time"] < delayed_kickoff < row["kickoff_utc"]
    assert row["away_qb_source_game_id"] != delayed


def test_partial_scored_game_never_counts_as_completed_history():
    from gridiron.data import normalise_schedules

    raw = make_schedule(scored_through=(2015, 4)).with_columns(
        pl.when(pl.col("game_id") == "2015_02_DDD_CCC").then(pl.lit("IN_PROGRESS"))
          .otherwise(pl.lit("FINAL")).alias("game_status"))
    sched = normalise_schedules(raw)
    pbp = make_pbp(sched)
    base = _features(sched, pbp)
    removed = _features(sched, pbp.filter(pl.col("game_id") != "2015_02_DDD_CCC"))
    assert_frame_equal(base, removed, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)
    assert base.filter(pl.col("game_id") == "2015_02_DDD_CCC")["history_available_at"][0] is None


def test_walk_forward_trains_only_on_earlier_weeks(monkeypatch):
    from gridiron import models
    from gridiron.data import normalise_schedules

    sched = normalise_schedules(make_schedule(scored_through=(2015, 4)))
    frame = F.modeling_frame(_features(sched, make_pbp(sched)))
    seen = []
    real_fit = models.fit

    def spy(train, stage):
        seen.append((train["season"].max(), train["week"].max(), train.height))
        return real_fit(train, stage)

    monkeypatch.setattr(models, "fit", spy)
    out = models.walk_forward(frame, "stage1", [2015], min_train_games=2)
    # week 1 has nothing earlier to train on and is skipped; weeks 2-4 are predicted
    assert out.height == frame.filter((pl.col("season") == 2015) & (pl.col("week") > 1)).height
    weeks = sorted(out["week"].unique().to_list())
    assert weeks == [2, 3, 4]
    for (max_season, max_week, _), w in zip(seen, weeks):
        assert max_season == 2015 and max_week < w
