import datetime as dt
import json

import polars as pl
import pytest

from gridiron import features as F, picks as P


@pytest.mark.parametrize("changed_home,changed_away", [("Different latest starter", "Other latest starter"), (None, None)])
def test_schedule_qb_names_cannot_change_saved_causal_provenance(schedule, pbp, tmp_path,
                                                               changed_home, changed_away):
    now = dt.datetime(2015, 9, 26, 12, tzinfo=dt.timezone.utc)
    original = schedule.with_columns(
        pl.lit("Unversioned home name").alias("home_qb_name"),
        pl.lit("Unversioned away name").alias("away_qb_name"),
        pl.lit("unverified schedule metadata").alias("home_qb_future_note"),
    )
    changed = original.with_columns(
        pl.lit(changed_home, dtype=pl.String).alias("home_qb_name"),
        pl.lit(changed_away, dtype=pl.String).alias("away_qb_name"),
        pl.lit("different unverified metadata").alias("home_qb_future_note"),
    )
    stored = []
    for label, source in (("original", original), ("changed", changed)):
        target = F.build_game_features(source, pbp, prediction_time=now).filter(
            (pl.col("season") == 2015) & (pl.col("week") == 4)).sort("game_id")
        # Demonstrate that the undesired schedule fields reach make_picks; the
        # persisted evidence must ignore them while retaining actual model inputs.
        assert "home_qb_name" in target.columns and "home_qb_future_note" in target.columns
        generated = P.make_picks(target, [.6, .4], 2015, 4, "stage1", 10, now=now)
        path = tmp_path / label / "picks.csv"
        assert P.append_picks(generated, path=path, now=now) == 2
        sidecar = path.with_name("qb_inputs.jsonl")
        stored.append(sidecar.read_bytes())
        for line, game in zip(sidecar.read_text().splitlines(), target.iter_rows(named=True)):
            evidence = json.loads(line)["inputs"]
            expected = {"prediction_time"} | {
                f"{side}_{field}" for side in ("home", "away")
                for field in [*F.QB_PROVENANCE, "qb_epa", "qb_prior_games"]
            }
            assert set(evidence) == expected
            assert evidence["prediction_time"] == now.isoformat()
            for side in ("home", "away"):
                assert evidence[f"{side}_qb_id"] == game[f"{side}_qb_id"]
                assert evidence[f"{side}_qb_source"] == "lagged_completed_game"
                assert evidence[f"{side}_qb_source_game_id"] == game[f"{side}_qb_source_game_id"]
                assert evidence[f"{side}_qb_epa"] == game[f"{side}_qb_epa"]
                assert evidence[f"{side}_qb_prior_games"] == game[f"{side}_qb_prior_games"]
    assert stored[0] == stored[1]
