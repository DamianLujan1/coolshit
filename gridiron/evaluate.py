"""Frozen model selection, independent audit, and benchmark-only market metrics."""

from __future__ import annotations

import json
import math

import numpy as np
import polars as pl
from scipy.stats import norm

from . import config

SELECTION_SEASONS = [2022]
AUDIT_SEASONS = [2023, 2024, 2025]
PROTOCOL_VERSION = "causal-qb-2022-selection-v1"


def implied_prob(moneyline: float) -> float:
    ml = float(moneyline)
    if not math.isfinite(ml) or ml == 0:
        raise ValueError("moneyline must be finite and nonzero")
    return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)


def market_prob_home(home_ml, away_ml) -> float | None:
    """Proportionally remove vig from two valid moneylines; missing odds stay missing."""
    if home_ml is None or away_ml is None:
        return None
    try:
        h, a = implied_prob(home_ml), implied_prob(away_ml)
    except (TypeError, ValueError):
        return None
    return h / (h + a)


def spread_prob_home(spread_line) -> float | None:
    """Closing spread proxy; nflverse positive spread means the home team is favored."""
    if spread_line is None or not math.isfinite(float(spread_line)):
        return None
    return float(norm.cdf(spread_line / config.SPREAD_SIGMA))


def add_baselines(df: pl.DataFrame, home_rate: float) -> pl.DataFrame:
    if not math.isfinite(home_rate) or not 0 <= home_rate <= 1:
        raise ValueError("home_rate must be estimated from decisive pre-selection games")
    spread = [spread_prob_home(s) for s in df["spread_line"].to_list()]
    market = [market_prob_home(h, a) for h, a in zip(df["home_moneyline"].to_list(), df["away_moneyline"].to_list())]
    return df.with_columns(
        pl.lit(home_rate).alias("p_base_home"),
        pl.Series("p_base_spread", spread, dtype=pl.Float64),
        pl.Series("p_base_market", market, dtype=pl.Float64),
    )


def _arrays(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if p.ndim != 1 or y.ndim != 1 or len(p) != len(y):
        raise ValueError("probabilities and labels must be equal-length vectors")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must be finite and between zero and one")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("binary metrics exclude ties and unfinished games")
    return p, y.astype(int)


def metrics(p: np.ndarray, y: np.ndarray, *, always_home: bool = False) -> dict:
    p, y = _arrays(p, y)
    if not len(y):
        return {"n": 0, "accuracy": None, "brier": None, "log_loss": None}
    picks = np.ones(len(y), dtype=int) if always_home else (p >= 0.5).astype(int)
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "accuracy": float((picks == y).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "log_loss": float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean()),
    }


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    p, y = _arrays(p, y)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        n = int(mask.sum())
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": n,
                     "mean_pred": float(p[mask].mean()) if n else None,
                     "observed": float(y[mask].mean()) if n else None})
    return rows


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float | None:
    rows = calibration_table(p, y, bins)
    n = sum(r["n"] for r in rows)
    return (float(sum(r["n"] / n * abs(r["mean_pred"] - r["observed"]) for r in rows if r["n"]))
            if n else None)


COLUMNS = {
    "stage1": "Stage 1 (logistic)", "stage2": "Stage 2 (LightGBM)",
    "p_base_home": "Always home / historical home rate",
    "p_base_spread": "Closing-spread favorite / normal probability proxy",
    "p_base_market": "Market (vig-free moneylines)",
}
BASELINES = ["p_base_home", "p_base_spread", "p_base_market"]


def _summary(base: pl.DataFrame, cols: list[str]) -> dict:
    out = {"n_games": base.height, "seasons": {}, "combined": {}, "calibration": {}, "season_calibration": {}}
    groups = {str(s): base.filter(pl.col("season") == s) for s in sorted(base["season"].unique().to_list())}
    groups["combined"] = base
    for name, group in groups.items():
        block, calibration = {}, {}
        for col in cols:
            valid = group.filter(pl.col(col).is_not_null() & pl.col(col).is_finite())
            p, y = valid[col].to_numpy(), valid["home_win"].to_numpy()
            block[col] = metrics(p, y, always_home=col == "p_base_home")
            block[col]["ece"] = expected_calibration_error(p, y)
            calibration[col] = calibration_table(p, y)
        if name == "combined":
            out["combined"], out["calibration"] = block, calibration
        else:
            out["seasons"][name], out["season_calibration"][name] = block, calibration
    return out


def evaluate_backtest(preds: dict[str, pl.DataFrame], home_rate: float) -> dict:
    """Describe supplied predictions, without choosing a model from audit results.

    Models are evaluated on the same decisive games, even without closing odds.
    Each baseline has its own coverage. Matched blocks compare identical games.
    """
    if "stage1" not in preds:
        raise ValueError("stage1 predictions are required")
    required = ["game_id", "season", "week", "home_win", "spread_line", "home_moneyline", "away_moneyline"]
    decisive = {}
    for stage, df in preds.items():
        if stage not in ("stage1", "stage2"):
            raise ValueError(f"unknown candidate {stage}")
        if df["game_id"].n_unique() != df.height:
            raise ValueError(f"duplicate game predictions for {stage}")
        df = df.filter(pl.col("home_win").is_in([0, 1]))
        if "result" in df.columns:
            df = df.filter(pl.col("result").is_not_null() & (pl.col("result") != 0))
        _arrays(df["p_home"].to_numpy(), df["home_win"].to_numpy())
        decisive[stage] = df
    first = decisive["stage1"]
    reference = first.select("game_id", "season", "week", "home_win").sort("game_id")
    base = add_baselines(first.select(required), home_rate)
    for stage, df in decisive.items():
        if not reference.equals(df.select("game_id", "season", "week", "home_win").sort("game_id")):
            raise ValueError("candidate models must cover identical decisive games and labels")
        base = base.join(df.select("game_id", pl.col("p_home").alias(stage)), on="game_id", how="left")
    cols = list(preds) + BASELINES
    out = _summary(base, cols)
    out["home_rate_train"] = home_rate
    out["coverage"] = {
        "input_rows": preds["stage1"].height,
        "excluded_ties_or_unfinished": preds["stage1"].height - base.height,
        "decisive_model_games": base.height,
        "market_games": base["p_base_market"].count(),
        "spread_games": base["p_base_spread"].count(),
        "missing_market_games": base["p_base_market"].null_count(),
        "missing_spread_games": base["p_base_spread"].null_count(),
    }
    out["market_matched"] = _summary(base.drop_nulls("p_base_market"), list(preds) + ["p_base_home", "p_base_market"])
    out["spread_matched"] = _summary(base.drop_nulls("p_base_spread"), list(preds) + ["p_base_home", "p_base_spread"])
    return out


def decide_stage(selection: dict) -> str:
    """Strictly better Brier AND log loss on identical 2022 selection games."""
    if "stage2" not in selection:
        return "stage1"
    s1, s2 = selection["stage1"], selection["stage2"]
    if not s1["n"] or s1["n"] != s2["n"]:
        return "stage1"
    values = [s[m] for s in (s1, s2) for m in ("log_loss", "brier")]
    if not all(v is not None and math.isfinite(v) for v in values):
        return "stage1"
    return "stage2" if s2["log_loss"] < s1["log_loss"] and s2["brier"] < s1["brier"] else "stage1"


def evaluate_frozen_backtest(selection_preds: dict[str, pl.DataFrame], audit_preds: dict[str, pl.DataFrame],
                             home_rate: float) -> dict:
    """Freeze stage using only 2022, then report the distinct 2023–2025 audit."""
    if set(selection_preds) != set(audit_preds):
        raise ValueError("selection and audit must evaluate the same candidates")
    for df in selection_preds.values():
        if set(df["season"].unique().to_list()) != set(SELECTION_SEASONS):
            raise ValueError("model selection requires only 2022 predictions")
    for df in audit_preds.values():
        if set(df["season"].unique().to_list()) != set(AUDIT_SEASONS):
            raise ValueError("audit requires all of 2023–2025, disjoint from selection")
    selection = evaluate_backtest(selection_preds, home_rate)
    choice = decide_stage(selection["combined"])
    result = evaluate_backtest(audit_preds, home_rate)
    result.update({
        "protocol_version": PROTOCOL_VERSION,
        "stage_choice": choice,
        "selection": {**selection, "stage_choice": choice, "selection_seasons": SELECTION_SEASONS,
                      "rule": "stage2 must strictly improve both Brier and log loss over stage1 on all shared 2022 decisive games"},
        "audit_seasons": AUDIT_SEASONS,
        "audit_used_for_selection": False,
        "ties_policy": "Exclude ties and unfinished games from binary training and scoring; published tied picks are pushes. Probabilities are conditional on a decisive result. A 0.5 probability picks home.",
        "baseline_policy": "Always-home accuracy always picks home; its probability is the fixed pre-2022 decisive home-win rate. Closing-spread favorite uses sign (pick'em chooses home) and a fixed normal margin sigma of 13.5 for probability scores. No-vig moneylines use proportional normalization. Odds are benchmarks only, never features or selection inputs.",
        "historical_limitations": [
            "Historical source files were rebuilt from the latest nflverse release; original Tuesday data vintages and correction arrival times are unavailable.",
            "Historical prediction cutoff is Tuesday 12:00 America/New_York before the week's first game. Completed history requires the conservative kickoff-plus-12-hours availability proxy; this is not a verified publication timestamp.",
            "QB inputs use only timestamped pre-cutoff observations when supplied, otherwise a strictly lagged completed-game QB prior. Current schedule actual starters cannot establish historical Tuesday availability.",
            "The candidates and 2022 selection rule are held fixed for this rebuilt audit. This retrospective repair is not an untouched or prospective validation; 2023–2025 results were examined in the earlier flawed build.",
            "Closing market lines can contain information unavailable on Tuesday. They are strong later-information benchmarks, not contemporaneous tradable prices.",
        ],
    })
    return result


def _fmt(x, nd=3):
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.{nd}f}"


def render_backtest_md(result: dict, extra: dict | None = None) -> str:
    lines = ["# Backtest: 2022 selection and frozen 2023–2025 audit", "",
             "The fixed logistic and boosting candidates are compared using 2022 only. Boosting is selected only if both its Brier score and log loss improve. The selected stage is then frozen before the 2023–2025 walk-forward audit. Each weekly fit uses only earlier-week games available by that week's prediction cutoff.", "",
             result.get("ties_policy", ""), "", result.get("baseline_policy", ""), ""]

    def table(block, title):
        lines.extend([f"## {title}", "", "| Model | Games | Accuracy | Brier | Log loss | ECE |", "|---|---:|---:|---:|---:|---:|"])
        for c, m in block.items():
            lines.append(f"| {COLUMNS.get(c, c)} | {m['n']} | {_fmt(m['accuracy'])} | {_fmt(m['brier'], 4)} | {_fmt(m['log_loss'], 4)} | {_fmt(m.get('ece'))} |")
        lines.append("")

    if "selection" in result:
        table(result["selection"]["combined"], "2022 model selection (not part of audit)")
    for season, block in result["seasons"].items():
        table(block, f"Audit season {season}")
    table(result["combined"], "Audit combined (each baseline's available coverage)")
    for key, label in (("market_matched", "market moneylines"), ("spread_matched", "closing-spread favorite")):
        if key in result:
            table(result[key]["combined"], f"Audit comparison on identical games: {label}")
    if "coverage" in result:
        cov = result["coverage"]
        lines.extend([f"Audit coverage: {cov['decisive_model_games']} decisive model games; {cov['market_games']} with valid two-sided moneylines; {cov['spread_games']} with valid spreads. Model scoring retains games without odds.", ""])
    if "sample_accounting" in result:
        lines.extend(["## Sample accounting", "", "| Season | Completed | Ties excluded | Decisive | Modeled | Other exclusions |", "|---|---:|---:|---:|---:|---:|"])
        for season, a in result["sample_accounting"].items():
            lines.append(f"| {season} | {a['completed']} | {a['ties']} | {a['decisive']} | {a['modeled']} | {a['other_exclusions']} |")
        lines.append("")
    lines.extend(["## Verdict", "", *verdict_lines(result), "", "## Historical reconstruction limits", ""])
    lines.extend(f"- {limitation}" for limitation in result.get("historical_limitations", []))
    lines.append("")
    if extra:
        table(extra, "2026 to date (descriptive only; not used for selection)")
    # Show every candidate and baseline, every season plus combined, with all 10 bins.
    scopes = {"Selection 2022": result.get("selection", {}).get("calibration", {})}
    scopes.update({f"Audit {s}": c for s, c in result["season_calibration"].items()})
    scopes["Audit combined"] = result["calibration"]
    lines.extend(["## Calibration (10 equal-width probability bins)", ""])
    for scope, tables in scopes.items():
        for c, rows in tables.items():
            lines.extend([f"### {scope}: {COLUMNS.get(c, c)}", "", "| Predicted home-win probability | Games | Mean predicted | Observed home-win rate |", "|---|---:|---:|---:|"])
            for row in rows:
                lines.append(f"| {row['bin']} | {row['n']} | {_fmt(row['mean_pred'])} | {_fmt(row['observed'])} |")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def verdict_lines(result: dict) -> list[str]:
    choice = result.get("stage_choice", "stage1")
    selection = result.get("selection", {})
    out = [f"- Selected **{COLUMNS[choice]}** using 2022 only. " +
           ("Boosting improved both selection Brier and log loss." if choice == "stage2" else
            "Boosting did not improve both selection metrics, or was not evaluated; logistic remains selected.")]
    c = result.get("market_matched", result)["combined"]
    model, market = c[choice], c["p_base_market"]
    if not market["n"]:
        out.append("- No matched valid moneylines are available; market superiority cannot be assessed.")
    else:
        wins = {"accuracy": model["accuracy"] > market["accuracy"], "brier": model["brier"] < market["brier"], "log_loss": model["log_loss"] < market["log_loss"]}
        conclusion = ("The model beats the closing market on all three reported metrics." if all(wins.values()) else
                      "The market beats or ties the model on accuracy, Brier and log loss." if not any(wins.values()) else
                      "The model and market have mixed results; the model does not clearly beat the closing line.")
        out.append(f"- **{conclusion}** On {market['n']} identical audit games: model/market accuracy {_fmt(model['accuracy'])}/{_fmt(market['accuracy'])}, Brier {_fmt(model['brier'], 4)}/{_fmt(market['brier'], 4)}, log loss {_fmt(model['log_loss'], 4)}/{_fmt(market['log_loss'], 4)}.")
        out.append(f"- Matched calibration ECE: model {_fmt(model['ece'])}, market {_fmt(market['ece'])}. Ten-bin calibration is descriptive, sensitive to sample size, and is not a model-selection criterion.")
    spread = result.get("spread_matched", result)["combined"]
    out.append(f"- On {spread['p_base_spread']['n']} spread-covered games, model accuracy is {_fmt(spread[choice]['accuracy'])} versus favorite {_fmt(spread['p_base_spread']['accuracy'])}; always-home accuracy across all modeled audit games is {_fmt(result['combined']['p_base_home']['accuracy'])}.")
    out.append("- Audit results do not change the selected stage. Both models are disclosed even when the selected model loses to the market.")
    return out


def save_json(obj: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
