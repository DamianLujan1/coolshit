"""Selection isolation, identical-game market comparisons, and input leakage gates."""

import datetime as dt
import json
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from gridiron import evaluate as E, models


def predictions(seasons, stage1=(0.4, 0.6), stage2=(0.2, 0.8)):
    rows = []
    for season in seasons:
        for i, label in enumerate((0, 1)):
            rows.append({"game_id": f"{season}_{i}", "season": season, "week": i + 1,
                         "home_win": label, "result": 7 if label else -7,
                         "spread_line": 3.0, "home_moneyline": -150.0, "away_moneyline": 130.0})
    base = pl.DataFrame(rows)
    return {stage: base.with_columns(pl.Series("p_home", list(probs) * len(seasons)))
            for stage, probs in (("stage1", stage1), ("stage2", stage2))}


def test_selection_is_frozen_before_audit_and_never_uses_audit_winner():
    selection = predictions([2022])
    audit = predictions([2023, 2024, 2025], stage1=(0.1, 0.9), stage2=(0.9, 0.1))
    result = E.evaluate_frozen_backtest(selection, audit, 0.55)
    assert result["stage_choice"] == "stage2"
    assert result["selection"]["stage_choice"] == "stage2"
    assert result["combined"]["stage1"]["brier"] < result["combined"]["stage2"]["brier"]
    opposite_audit = predictions([2023, 2024, 2025], stage1=(0.9, 0.1), stage2=(0.1, 0.9))
    assert E.evaluate_frozen_backtest(selection, opposite_audit, 0.55)["stage_choice"] == "stage2"
    assert result["audit_used_for_selection"] is False
    assert result["selection"]["selection_seasons"] == [2022]


def test_selection_requires_both_metrics_on_same_games_and_strict_improvement():
    base = {"n": 10, "brier": 0.2, "log_loss": 0.6}
    for stage2 in ({**base}, {**base, "brier": 0.1}, {**base, "log_loss": 0.5},
                   {"n": 9, "brier": 0.1, "log_loss": 0.5}):
        assert E.decide_stage({"stage1": base, "stage2": stage2}) == "stage1"
    assert E.decide_stage({"stage1": base, "stage2": {"n": 10, "brier": 0.1, "log_loss": 0.5}}) == "stage2"


@pytest.mark.parametrize("selection_seasons,audit_seasons", [([2023], [2023, 2024, 2025]), ([2022], [2022, 2023, 2024]), ([2022], [2023, 2024])])
def test_selection_and_audit_years_cannot_overlap_or_be_incomplete(selection_seasons, audit_seasons):
    with pytest.raises(ValueError):
        E.evaluate_frozen_backtest(predictions(selection_seasons), predictions(audit_seasons), 0.55)


def test_missing_odds_do_not_drop_model_games_and_matched_comparisons_match():
    pred = predictions([2023, 2024, 2025])
    pred = {stage: df.with_columns(
        pl.when(pl.col("game_id") == "2023_0").then(None).otherwise(pl.col("home_moneyline")).alias("home_moneyline"),
        pl.when(pl.col("game_id") == "2024_0").then(None).otherwise(pl.col("spread_line")).alias("spread_line"))
        for stage, df in pred.items()}
    out = E.evaluate_backtest(pred, 0.55)
    assert out["combined"]["stage1"]["n"] == out["combined"]["stage2"]["n"] == 6
    assert out["combined"]["p_base_market"]["n"] == out["combined"]["p_base_spread"]["n"] == 5
    assert {m["n"] for m in out["market_matched"]["combined"].values()} == {5}
    assert {m["n"] for m in out["spread_matched"]["combined"].values()} == {5}
    assert out["coverage"]["missing_market_games"] == 1
    assert out["coverage"]["missing_spread_games"] == 1
    assert len(out["season_calibration"]["2023"]["stage1"]) == 10
    assert sum(r["n"] for r in out["season_calibration"]["2023"]["stage1"]) == 2
    assert sum(r["n"] for r in out["calibration"]["p_base_market"]) == 5


def test_missing_market_rows_cannot_change_selection():
    pred = predictions([2022], stage1=(0.01, 0.6), stage2=(0.99, 0.9))
    missing = {stage: df.with_columns(
        pl.when(pl.col("home_win") == 0).then(None).otherwise(pl.col("home_moneyline")).alias("home_moneyline"))
        for stage, df in pred.items()}
    original = E.evaluate_backtest(pred, 0.55)
    changed = E.evaluate_backtest(missing, 0.55)
    assert E.decide_stage(original["combined"]) == E.decide_stage(changed["combined"]) == "stage1"


def test_mismatched_candidate_games_or_labels_are_rejected():
    pred = predictions([2022])
    pred["stage2"] = pred["stage2"].head(1)
    with pytest.raises(ValueError, match="identical"):
        E.evaluate_backtest(pred, 0.55)
    pred = predictions([2022])
    pred["stage2"] = pred["stage2"].with_columns((1 - pl.col("home_win")).alias("home_win"))
    with pytest.raises(ValueError, match="identical"):
        E.evaluate_backtest(pred, 0.55)


def test_ties_unfinished_games_and_probability_endpoints():
    pred = predictions([2022])
    pred = {stage: df.with_columns(
        pl.when(pl.col("home_win") == 0).then(None).otherwise(pl.col("home_win")).alias("home_win"),
        pl.when(pl.col("result") < 0).then(0).otherwise(pl.col("result")).alias("result"))
        for stage, df in pred.items()}
    result = E.evaluate_backtest(pred, 0.4)
    assert result["coverage"]["excluded_ties_or_unfinished"] == 1
    assert result["combined"]["p_base_home"]["accuracy"] == 1.0  # always home even if training home rate < 0.5
    assert E.metrics([0, 1], [0, 1])["brier"] == 0.0
    assert sum(r["n"] for r in E.calibration_table([0, 1], [0, 1])) == 2
    assert E.metrics([], []) == {"n": 0, "accuracy": None, "brier": None, "log_loss": None}
    assert E.expected_calibration_error([], []) is None
    with pytest.raises(ValueError, match="exclude ties"):
        E.metrics([0.5], [0.5])
    with pytest.raises(ValueError, match="finite"):
        E.metrics([float("nan")], [1])


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_benchmark_mutations_cannot_change_fitted_predictions(stage):
    rng = np.random.default_rng(24)
    row_count = 80
    columns = {feature: rng.normal(size=row_count) for feature in models.feature_names(stage)}
    columns.update({"game_id": [f"g{i:03d}" for i in range(row_count)], "home_win": [i % 2 for i in range(row_count)],
                    "spread_line": np.full(row_count, 3.5), "home_moneyline": np.full(row_count, -150),
                    "away_moneyline": np.full(row_count, 130), "total_line": np.full(row_count, 44)})
    frame = pl.DataFrame(columns)
    changed = frame.with_columns(pl.lit(-99).alias("spread_line"), pl.lit(99999).alias("home_moneyline"),
                                 pl.lit(-99999).alias("away_moneyline"), pl.lit(999).alias("total_line"))
    first, second = models.fit(frame, stage), models.fit(changed, stage)
    assert not {"spread_line", "home_moneyline", "away_moneyline", "total_line"}.intersection(first.features)
    np.testing.assert_array_equal(first.predict_proba(frame), second.predict_proba(changed))


def test_benchmark_feature_list_changes_fail_closed(monkeypatch):
    monkeypatch.setitem(models.STAGES, "stage1", [*models.STAGES["stage1"], "home_moneyline"])
    with pytest.raises(ValueError, match="cannot be model features"):
        models.fit(pl.DataFrame(), "stage1")


def test_walk_forward_excludes_prior_week_results_unavailable_at_cutoff(monkeypatch):
    utc = dt.timezone.utc
    cutoff = dt.datetime(2023, 9, 12, 16, tzinfo=utc)
    before = cutoff - dt.timedelta(days=1)
    after = cutoff + dt.timedelta(hours=1)
    frame = pl.DataFrame({"game_id": ["available", "delayed", "target"], "season": [2023] * 3,
                          "week": [1, 1, 2], "kickoff_utc": [before, before, after],
                          "history_available_at": [before, after, after],
                          "prediction_time": [before - dt.timedelta(days=1), before - dt.timedelta(days=1), cutoff],
                          "home_win": [1, 0, 1]})
    seen = []
    def fake_fit(train, stage):
        seen.append(train["game_id"].to_list())
        return SimpleNamespace(predict_proba=lambda test: np.full(test.height, 0.5))
    monkeypatch.setattr(models, "fit", fake_fit)
    result = models.walk_forward(frame, "stage1", [2023], min_train_games=1)
    assert seen == [["available"]]
    assert result["game_id"].to_list() == ["target"]


def test_report_has_every_season_calibration_and_strict_json(tmp_path):
    result = E.evaluate_frozen_backtest(predictions([2022]), predictions([2023, 2024, 2025]), 0.55)
    path = tmp_path / "report.json"
    E.save_json(result, path)
    parsed = json.loads(path.read_text())
    report = E.render_backtest_md(parsed)
    for season in (2023, 2024, 2025):
        for label in E.COLUMNS.values():
            assert f"Audit {season}: {label}" in report
    assert "QB inputs" in report
    assert "not an untouched or prospective validation" in report
    assert "Audit results do not change the selected stage" in report


def test_weekly_stage_choice_rejects_legacy_or_inconsistent_selection(tmp_path, monkeypatch):
    from gridiron import cli, config
    path = tmp_path / "backtest.json"
    monkeypatch.setattr(config, "BACKTEST_JSON", path)
    path.write_text(json.dumps({"stage_choice": "stage2"}))
    assert cli._stage_choice() == "stage1"
    result = E.evaluate_frozen_backtest(predictions([2022]), predictions([2023, 2024, 2025]), 0.55)
    path.write_text(json.dumps(result))
    assert cli._stage_choice() == "stage2"
    result["stage_choice"] = "stage1"
    path.write_text(json.dumps(result))
    assert cli._stage_choice() == "stage1"


def test_backtest_command_runs_selection_first_and_writes_review_artifacts_only(tmp_path, monkeypatch, capsys):
    from gridiron import cli, config, features
    frame = predictions([2015, 2022, 2023, 2024, 2025])["stage1"].drop("p_home")
    monkeypatch.setattr(cli, "_load", lambda **kwargs: (frame, None))
    monkeypatch.setattr(cli, "_features", lambda sched, pbp: frame)
    monkeypatch.setattr(features, "modeling_frame", lambda feats: feats)
    calls = []
    def fake_walk_forward(rows, stage, seasons):
        calls.append((stage, seasons))
        return rows.filter(pl.col("season").is_in(seasons)).with_columns(
            pl.when(pl.col("home_win") == 1).then(0.8 if stage == "stage2" else 0.6)
            .otherwise(0.2 if stage == "stage2" else 0.4).alias("p_home"), pl.lit(stage).alias("stage"))
    monkeypatch.setattr(models, "walk_forward", fake_walk_forward)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "BACKTEST_JSON", tmp_path / "reports/backtest.json")
    monkeypatch.setattr(config, "BACKTEST_MD", tmp_path / "reports/backtest.md")
    (tmp_path / "pyproject.toml").write_text("test input")
    cli.cmd_backtest(SimpleNamespace(now=None, refresh=False, stage1_only=False))
    assert calls == [("stage1", [2022]), ("stage2", [2022]), ("stage1", [2023, 2024, 2025]), ("stage2", [2023, 2024, 2025])]
    result = json.loads(config.BACKTEST_JSON.read_text())
    assert result["stage_choice"] == "stage2"
    assert result["home_rate_training_seasons"] == [2015]
    assert result["sample_accounting"]["2023"] == {"completed": 2, "ties": 0, "decisive": 2, "modeled": 2, "other_exclusions": 0}
    assert len(result["reproducibility"]["raw_predictions"]) == 2
    assert pl.read_csv(tmp_path / "reports/audit_2023_2025_predictions.csv").height == 6
    assert not (tmp_path / "data/picks").exists()
