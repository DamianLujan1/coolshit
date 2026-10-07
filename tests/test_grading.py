import datetime as dt

import polars as pl
import pytest

from gridiron import picks as P
from gridiron.evaluate import market_prob_home, spread_prob_home, metrics, calibration_table


def _pick_rows():
    base = {"picked_at_utc": "2026-10-06T23:00:00+00:00", "season": 2026, "week": 5,
            "model_stage": "stage1", "model_version": "0.1.0", "n_train": 3000,
            "spread_line_at_pick": 3.0, "home_moneyline_at_pick": -150, "away_moneyline_at_pick": 130}
    rows = [
        # pick home, home wins -> hit
        {**base, "game_id": "g1", "kickoff_utc": "2026-10-08T00:15:00+00:00", "home_team": "DAL", "away_team": "TB", "p_home": 0.7, "pick": "DAL", "pick_prob": 0.7, "confidence": "confident"},
        # pick away, home wins -> miss
        {**base, "game_id": "g2", "kickoff_utc": "2026-10-11T17:00:00+00:00", "home_team": "GB", "away_team": "CHI", "p_home": 0.4, "pick": "CHI", "pick_prob": 0.6, "confidence": "lean"},
        # tie -> push
        {**base, "game_id": "g3", "kickoff_utc": "2026-10-11T17:00:00+00:00", "home_team": "MIA", "away_team": "CIN", "p_home": 0.55, "pick": "MIA", "pick_prob": 0.55, "confidence": "lean"},
        # not played yet -> pending
        {**base, "game_id": "g4", "kickoff_utc": "2026-10-12T00:20:00+00:00", "home_team": "SEA", "away_team": "NE", "p_home": 0.8, "pick": "SEA", "pick_prob": 0.8, "confidence": "strong"},
    ]
    return pl.DataFrame(rows)


def _schedule():
    return pl.DataFrame({
        "game_id": ["g1", "g2", "g3", "g4"],
        "result": [7, 3, 0, None], "home_score": [27, 20, 17, None], "away_score": [20, 17, 17, None],
        "spread_line": [3.0, -2.5, 0.0, 6.5], "home_moneyline": [-150, 120, -105, -300], "away_moneyline": [130, -140, -115, 240],
    })


def test_grade_hits_misses_pushes_and_pending():
    g = P.grade(_pick_rows(), _schedule())
    by = {r["game_id"]: r for r in g.iter_rows(named=True)}
    assert by["g1"]["model_hit"] == 1 and by["g1"]["graded"]
    assert by["g2"]["model_hit"] == 0
    assert by["g3"]["model_hit"] is None and by["g3"]["graded"]
    assert by["g4"]["model_hit"] is None and not by["g4"]["graded"]
    # baselines graded on the closing line: g2 spread is -2.5 so the favorite is the away team (CHI), who lost
    assert by["g2"]["spread_pick"] == "CHI" and by["g2"]["spread_hit"] == 0
    assert by["g2"]["market_pick"] == "CHI" and by["g2"]["market_hit"] == 0
    assert by["g1"]["home_hit"] == 1
    assert abs(by["g1"]["model_brier"] - (0.7 - 1) ** 2) < 1e-9


def test_running_record_counts():
    rec = P.running_record(P.grade(_pick_rows(), _schedule()))
    assert rec["n_graded"] == 3 and rec["n_pending"] == 1
    m = rec["overall"]["model"]
    assert (m["wins"], m["losses"], m["pushes"]) == (1, 1, 1)
    assert m["pct"] == 0.5
    assert rec["overall"]["home"]["wins"] == 2
    assert rec["seasons"]["2026"]["model"]["wins"] == 1
    assert rec["weeks"][0]["week"] == 5


def test_confidence_labels():
    assert P.confidence_label(0.5) == "coin flip"
    assert P.confidence_label(0.55) == "lean"
    assert P.confidence_label(0.66) == "confident"
    assert P.confidence_label(0.9) == "strong"


def test_market_and_spread_probabilities():
    p = market_prob_home(-150, 130)
    assert 0.5 < p < 0.6
    assert abs(market_prob_home(-110, -110) - 0.5) < 1e-12
    assert spread_prob_home(0.0) == 0.5
    assert spread_prob_home(7.0) > 0.65 > spread_prob_home(3.0) > 0.5 > spread_prob_home(-3.0)
    assert market_prob_home(None, 130) is None


def test_metrics_and_calibration():
    m = metrics([0.9, 0.1, 0.6, 0.4], [1, 0, 0, 1])
    assert m["accuracy"] == 0.5 and m["n"] == 4
    assert abs(m["brier"] - ((0.1 ** 2) * 2 + (0.6 ** 2) * 2) / 4) < 1e-9
    rows = calibration_table([0.05, 0.95, 0.95, 0.55], [0, 1, 0, 1])
    assert rows[0]["n"] == 1 and rows[9]["n"] == 2 and rows[9]["observed"] == 0.5


def test_append_only_pick_log(tmp_path):
    path = tmp_path / "picks.csv"
    now = dt.datetime(2026, 10, 6, 23, 0, tzinfo=dt.timezone.utc)
    rows = _pick_rows()
    assert P.append_picks(rows, path, now=now) == 4
    first_bytes = path.read_bytes()
    # re-appending the same games adds nothing and changes nothing
    assert P.append_picks(rows, path, now=now) == 0
    assert path.read_bytes() == first_bytes
    # a changed probability for an already-picked game is ignored: the first pick stands
    changed = rows.with_columns(pl.lit(0.99).alias("p_home"))
    assert P.append_picks(changed, path, now=now) == 0
    assert path.read_bytes() == first_bytes
    # new games are appended after the existing bytes, which are untouched
    more = rows.with_columns((pl.col("game_id") + "_b").alias("game_id"))
    assert P.append_picks(more, path, now=now) == 4
    assert path.read_bytes().startswith(first_bytes)
    assert P.load_picks(path).height == 8
    # a pick after kickoff is refused outright
    late = rows.with_columns((pl.col("game_id") + "_c").alias("game_id"))
    with pytest.raises(P.PickLogError):
        P.append_picks(late, path, now=dt.datetime(2026, 10, 13, tzinfo=dt.timezone.utc))
