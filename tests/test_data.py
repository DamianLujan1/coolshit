"""Network-free validation and cache contracts for all four permitted datasets."""

import datetime as dt
import json

import polars as pl
import pytest

from gridiron import config, data
from tests.conftest import TEAMS, make_pbp, make_schedule


@pytest.fixture
def source_frames(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PRIOR_SEED_SEASON", 2015)
    monkeypatch.setattr(config, "CURRENT_SEASON", 2015)
    monkeypatch.setattr(config, "TEAMS", TEAMS)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    sched = data.normalise_schedules(make_schedule(seasons=(2015,)))
    pbp = make_pbp(sched).with_row_index("play_id")
    rows = []
    for game in sched.filter(pl.col("result").is_not_null()).iter_rows(named=True):
        for side in ("home_team", "away_team"):
            rows.append({"season": game["season"], "week": game["week"], "season_type": "REG",
                         "team": game[side], "player_id": "QB_" + game[side]})
    players = pl.DataFrame(rows)
    return sched, pbp, players.drop("player_id"), players


def validate_frames(frames, **updates):
    sched, pbp, teams, players = frames
    return data.validate(updates.get("sched", sched), updates.get("pbp", pbp),
                         now=dt.datetime(2015, 9, 23, tzinfo=dt.timezone.utc),
                         team_stats=updates.get("teams", teams), player_stats=updates.get("players", players))


def test_all_four_sources_validate_without_odds(source_frames):
    sched = source_frames[0].drop("home_moneyline", "away_moneyline", "spread_line")
    result = validate_frames(source_frames, sched=sched)
    assert result["completed_available_games"] == 6
    assert result["team_stats"] == result["player_stats"] == 12


def test_missing_optional_odds_and_starters_normalise_to_null():
    optional = ["spread_line", "home_moneyline", "away_moneyline", "home_qb_id", "away_qb_id"]
    sched = data.normalise_schedules(make_schedule().drop(optional))
    assert all(sched[c].is_null().all() for c in optional)
    assert sched.schema["home_moneyline"] == pl.Float64
    assert sched.schema["home_qb_id"] == pl.String


@pytest.mark.parametrize("value", [None, ""])
def test_unknown_kickoff_fails_closed(value):
    raw = make_schedule().with_columns(pl.lit(value, dtype=pl.String).alias("gametime"))
    with pytest.raises(data.DataValidationError, match="unknown kickoff times"):
        data.normalise_schedules(raw)


def test_float_success_flags_and_unattributed_stat_rows_are_reported(source_frames):
    pbp = source_frames[1].with_columns(pl.col("success").cast(pl.Float64))
    players = source_frames[3].with_columns(pl.lit(None, dtype=pl.String).alias("player_id"))
    result = validate_frames(source_frames, pbp=pbp, players=players)
    assert result["player_stats_unattributed_rows"] == players.height


@pytest.mark.parametrize("dataset,keys", [("sched", ["game_id"]), ("pbp", ["game_id", "play_id"]),
                                          ("teams", ["season", "week", "team"]), ("players", ["player_id"])])
def test_duplicate_identifiers_fail(source_frames, dataset, keys):
    original = source_frames[{"sched": 0, "pbp": 1, "teams": 2, "players": 3}[dataset]]
    bad = pl.concat([original, original.head(1)])
    with pytest.raises(data.DataValidationError, match="duplicated identity keys"):
        validate_frames(source_frames, **{dataset: bad})


def test_missing_team_or_player_coverage_fails(source_frames):
    for name, i in (("teams", 2), ("players", 3)):
        with pytest.raises(data.DataValidationError, match="missing 1 completed team-game"):
            validate_frames(source_frames, **{name: source_frames[i].slice(1)})


def test_bad_score_or_pbp_identity_fails(source_frames):
    with pytest.raises(data.DataValidationError, match="inconsistent final scores"):
        validate_frames(source_frames, sched=source_frames[0].with_columns((pl.col("result") + 1).alias("result")))
    with pytest.raises(data.DataValidationError, match="identity disagrees"):
        validate_frames(source_frames, pbp=source_frames[1].with_columns((pl.col("week") + 1).alias("week")))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_epa_fails(source_frames, value):
    with pytest.raises(data.DataValidationError, match="non-finite EPA"):
        validate_frames(source_frames, pbp=source_frames[1].with_columns(pl.lit(value).alias("epa")))


def test_missing_required_schema_is_clear(source_frames):
    with pytest.raises(data.DataValidationError, match="missing expected columns.*epa"):
        validate_frames(source_frames, pbp=source_frames[1].drop("epa"))
    with pytest.raises(data.DataValidationError, match="missing expected columns.*player_id"):
        validate_frames(source_frames, players=source_frames[3].drop("player_id"))


def test_completion_proxy_is_not_kickoff_or_partial_score():
    raw = make_schedule(seasons=(2015,)).with_columns(pl.lit("final").alias("game_status"))
    raw = raw.with_columns(pl.when(pl.col("week") == 3).then(pl.lit("in_progress"))
                           .otherwise(pl.col("game_status")).alias("game_status"))
    sched = data.normalise_schedules(raw)
    final = sched.filter(pl.col("week") == 1).row(0, named=True)
    assert final["game_completed"]
    assert final["history_available_at"] == final["kickoff_utc"] + dt.timedelta(hours=12)
    unfinished = sched.filter(pl.col("week") >= 3)
    assert unfinished["history_available_at"].is_null().all()
    assert unfinished["home_win"].is_null().all()


def test_just_started_game_does_not_require_completed_pbp(source_frames):
    sched, pbp, teams, players = source_frames
    kickoff = sched.filter(pl.col("week") == 3)["kickoff_utc"][0]
    result = data.validate(sched, pbp.filter(pl.col("week") < 3), now=kickoff + dt.timedelta(minutes=30),
                           team_stats=teams.filter(pl.col("week") < 3),
                           player_stats=players.filter(pl.col("week") < 3))
    assert result["completed_available_games"] == 4


@pytest.mark.parametrize("name", ["schedules", "pbp", "team_stats", "player_stats"])
def test_offline_cache_miss_never_downloads(monkeypatch, tmp_path, name):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(data, "_download", lambda *_: pytest.fail("offline mode attempted a download"))
    with pytest.raises(FileNotFoundError, match="offline cache miss"):
        data._load_cached(name, None if name == "schedules" else 2015, refresh=True, offline=True)


def test_cache_refresh_failure_falls_back_but_does_not_invent_freshness(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    raw = make_schedule()
    raw.write_parquet(tmp_path / "schedules.parquet")

    def fail(*_):
        raise ConnectionError("offline")

    monkeypatch.setattr(data, "_download", fail)
    assert data._load_cached("schedules", None, refresh=True).equals(raw)
    meta = data.cache_manifest()[0]
    assert meta["fetched_at"] is None and meta["rows"] == raw.height
    assert len(meta["sha256"]) == 64


def test_successful_cache_fetch_has_verifiable_provenance(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    raw = make_schedule()
    monkeypatch.setattr(data, "_download", lambda *_: raw)
    data._load_cached("schedules", None, refresh=True)
    metadata = json.loads((tmp_path / "schedules.json").read_text())
    assert dt.datetime.fromisoformat(metadata["fetched_at"]).tzinfo is not None
    assert metadata["rows"] == raw.height
    assert metadata["sha256"] == data._sha256(tmp_path / "schedules.parquet")
    assert metadata["historical_snapshot"] is False
    # Even refresh=True obeys offline mode and returns the existing bytes.
    monkeypatch.setattr(data, "_download", lambda *_: pytest.fail("offline download"))
    assert data._load_cached("schedules", None, refresh=True, offline=True).equals(raw)


def test_bad_download_does_not_replace_existing_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    path = tmp_path / "schedules.parquet"
    make_schedule().write_parquet(path)
    original = path.read_bytes()
    monkeypatch.setattr(data, "_download", lambda *_: pl.DataFrame({"season": [2015]}))
    with pytest.raises(data.DataValidationError):
        data._load_cached("schedules", None, refresh=True)
    assert path.read_bytes() == original


@pytest.mark.parametrize("name", ["ftn_charting", "participation"])
def test_share_alike_datasets_are_never_loaded(name):
    with pytest.raises(ValueError, match="not allowed"):
        data._download(name, 2025)
    with pytest.raises(ValueError, match="not allowed"):
        data._load_cached(name, 2025, refresh=False, offline=True)


def test_player_stats_normalise_historical_team_field():
    raw = pl.DataFrame({"season": [2015], "week": [1], "season_type": ["REG"],
                        "recent_team": ["STL"], "player_id": ["QB"]})
    assert data.normalise_stats(raw, "player_stats")["team"].to_list() == ["LA"]
