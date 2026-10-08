"""Leakage is a critical bug. These tests prove that no feature for a game depends on that game or later ones."""

import numpy as np
import polars as pl

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
    out = models.walk_forward(frame, "stage1", [2015], min_train_games=2, train_first_season=2015)
    # week 1 has nothing earlier to train on and is skipped; weeks 2-4 are predicted
    assert out.height == frame.filter((pl.col("season") == 2015) & (pl.col("week") > 1)).height
    weeks = sorted(out["week"].unique().to_list())
    assert weeks == [2, 3, 4]
    for (max_season, max_week, _), w in zip(seen, weeks):
        assert max_season == 2015 and max_week < w


def test_training_set_respects_start_season_and_never_includes_target_week():
    import polars as pl
    from gridiron import config, models

    frame = pl.DataFrame({
        "season": [2018, 2019, 2020, 2021, 2026, 2026, 2026, 2026],
        "week":   [5,    5,    5,    5,    3,    4,    5,    6],
    })
    got = models.training_set(frame, 2026, 5, train_first_season=2020)
    assert got.select("season", "week").rows() == [(2020, 5), (2021, 5), (2026, 3), (2026, 4)]
    # default start comes from config
    default = models.training_set(frame, 2026, 5)
    assert default["season"].min() == config.TRAIN_FIRST_SEASON
    assert default.filter((pl.col("season") == 2026) & (pl.col("week") >= 5)).height == 0
