"""Metrics, calibration, baselines and the honest backtest report."""

from __future__ import annotations

import json
import math

import numpy as np
import polars as pl
from scipy.stats import norm

from . import config


# --------------------------------------------------------------------------- #
# Baselines (all expressed as P(home wins))
# --------------------------------------------------------------------------- #

def implied_prob(moneyline: float) -> float:
    ml = float(moneyline)
    return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)


def market_prob_home(home_ml, away_ml) -> float | None:
    """Vig-free home win probability from the two moneylines (proportional normalisation)."""
    if home_ml is None or away_ml is None:
        return None
    h, a = implied_prob(home_ml), implied_prob(away_ml)
    return h / (h + a)


def spread_prob_home(spread_line) -> float | None:
    """Closing spread -> P(home wins) via a normal margin model. Positive spread = home favored."""
    if spread_line is None:
        return None
    return float(norm.cdf(spread_line / config.SPREAD_SIGMA))


def add_baselines(df: pl.DataFrame, home_rate: float) -> pl.DataFrame:
    """Columns p_base_home, p_base_spread, p_base_market."""
    spread = [spread_prob_home(s) for s in df["spread_line"].to_list()]
    market = [market_prob_home(h, a) for h, a in zip(df["home_moneyline"].to_list(), df["away_moneyline"].to_list())]
    return df.with_columns(
        pl.lit(home_rate).alias("p_base_home"),
        pl.Series("p_base_spread", spread, dtype=pl.Float64),
        pl.Series("p_base_market", market, dtype=pl.Float64),
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, int)
    picks = (p >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "accuracy": float((picks == y).mean()) if len(y) else float("nan"),
        "brier": float(((p - y) ** 2).mean()) if len(y) else float("nan"),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()) if len(y) else float("nan"),
    }


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    p = np.asarray(p, float)
    y = np.asarray(y, int)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        n = int(mask.sum())
        rows.append({
            "bin": f"{lo:.1f}-{hi:.1f}",
            "n": n,
            "mean_pred": float(p[mask].mean()) if n else None,
            "observed": float(y[mask].mean()) if n else None,
        })
    return rows


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    rows = calibration_table(p, y, bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] / n * abs(r["mean_pred"] - r["observed"]) for r in rows if r["n"]))


# --------------------------------------------------------------------------- #
# Backtest report
# --------------------------------------------------------------------------- #

COLUMNS = {
    "stage1": "Stage 1 (logistic)",
    "stage2": "Stage 2 (LightGBM)",
    "p_base_home": "Always home",
    "p_base_spread": "Closing spread",
    "p_base_market": "Market (vig-free)",
}


def evaluate_backtest(preds: dict[str, pl.DataFrame], home_rate: float) -> dict:
    """preds: stage -> walk-forward frame. Returns a JSON-serialisable summary."""
    base = next(iter(preds.values())).select(
        "game_id", "season", "week", "home_win", "spread_line", "home_moneyline", "away_moneyline"
    )
    base = add_baselines(base, home_rate)
    for stage, df in preds.items():
        base = base.join(df.select("game_id", pl.col("p_home").alias(stage)), on="game_id", how="left")
    cols = [s for s in preds] + ["p_base_home", "p_base_spread", "p_base_market"]
    base = base.drop_nulls(cols)

    out = {"seasons": {}, "combined": {}, "calibration": {}, "n_games": base.height, "home_rate_train": home_rate}
    groups = {str(s): base.filter(pl.col("season") == s) for s in sorted(base["season"].unique().to_list())}
    groups["combined"] = base
    for name, g in groups.items():
        y = g["home_win"].to_numpy()
        block = {}
        for c in cols:
            block[c] = metrics(g[c].to_numpy(), y)
            block[c]["ece"] = expected_calibration_error(g[c].to_numpy(), y)
        if name == "combined":
            out["combined"] = block
            out["calibration"] = {c: calibration_table(g[c].to_numpy(), y) for c in cols}
        else:
            out["seasons"][name] = block
    out["stage_choice"] = decide_stage(out["combined"])
    return out


def decide_stage(combined: dict) -> str:
    """Stage 2 ships only if it beats stage 1 on both log loss and Brier."""
    if "stage2" not in combined:
        return "stage1"
    s1, s2 = combined["stage1"], combined["stage2"]
    return "stage2" if (s2["log_loss"] < s1["log_loss"] and s2["brier"] < s1["brier"]) else "stage1"


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def render_backtest_md(result: dict, extra: dict | None = None) -> str:
    cols = list(result["combined"].keys())
    lines = ["# Backtest: walk-forward, seasons " + ", ".join(result["seasons"].keys()), ""]
    lines.append("Every test week is predicted by a model trained only on earlier seasons and earlier weeks of the same season. "
                 "Baselines use the closing line stored in the nflverse schedules file, which is never a model input.")
    lines.append("")

    def table(block: dict, title: str):
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Model | Games | Accuracy | Brier | Log loss | ECE |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for c in cols:
            m = block[c]
            lines.append(f"| {COLUMNS.get(c, c)} | {m['n']} | {_fmt(m['accuracy'])} | {_fmt(m['brier'], 4)} | {_fmt(m['log_loss'], 4)} | {_fmt(m['ece'])} |")
        lines.append("")

    for season, block in result["seasons"].items():
        table(block, f"Season {season}")
    table(result["combined"], "Combined")

    lines.append("## Calibration (combined, 10 bins)")
    lines.append("")
    for c in ["stage1", "stage2", "p_base_market"]:
        if c not in result["calibration"]:
            continue
        lines.append(f"### {COLUMNS[c]}")
        lines.append("")
        lines.append("| Predicted home win prob | Games | Mean predicted | Observed home win rate |")
        lines.append("|---|---:|---:|---:|")
        for r in result["calibration"][c]:
            lines.append(f"| {r['bin']} | {r['n']} | {_fmt(r['mean_pred'])} | {_fmt(r['observed'])} |")
        lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.extend(verdict_lines(result))
    if extra:
        lines.append("")
        lines.append("## Current season to date (not part of the holdout)")
        lines.append("")
        lines.append("| Model | Games | Accuracy | Brier | Log loss |")
        lines.append("|---|---:|---:|---:|---:|")
        for c, m in extra.items():
            lines.append(f"| {COLUMNS.get(c, c)} | {m['n']} | {_fmt(m['accuracy'])} | {_fmt(m['brier'], 4)} | {_fmt(m['log_loss'], 4)} |")
    return "\n".join(lines) + "\n"


def verdict_lines(result: dict) -> list[str]:
    c = result["combined"]
    choice = result["stage_choice"]
    model = c[choice]
    market = c["p_base_market"]
    spread = c["p_base_spread"]
    out = [f"- Shipping **{COLUMNS[choice]}**. " + (
        "Stage 2 beat stage 1 on both log loss and Brier." if choice == "stage2"
        else "Stage 2 did not beat stage 1 on both log loss and Brier, so the simpler model ships.")]
    beats_market_ll = model["log_loss"] < market["log_loss"]
    beats_market_acc = model["accuracy"] > market["accuracy"]
    if beats_market_ll and beats_market_acc:
        out.append("- The model beats the vig-free market on log loss and accuracy over the holdout. Treat that with suspicion until it holds on live picks.")
    elif beats_market_ll or beats_market_acc:
        out.append(f"- Mixed result against the market: log loss {_fmt(model['log_loss'], 4)} vs {_fmt(market['log_loss'], 4)}, "
                   f"accuracy {_fmt(model['accuracy'])} vs {_fmt(market['accuracy'])}. The model does not clearly beat the closing line.")
    else:
        out.append(f"- **The model does not beat the closing line.** Market log loss {_fmt(market['log_loss'], 4)} vs model {_fmt(model['log_loss'], 4)}; "
                   f"market accuracy {_fmt(market['accuracy'])} vs model {_fmt(model['accuracy'])}. That is the expected outcome for a public-data model, "
                   "and the public record will show it either way.")
    out.append(f"- Against the closing-spread favorite: model accuracy {_fmt(model['accuracy'])} vs {_fmt(spread['accuracy'])}; "
               f"against always-home: {_fmt(c['p_base_home']['accuracy'])}.")
    out.append(f"- Calibration: expected calibration error {_fmt(model['ece'])} for the model vs {_fmt(market['ece'])} for the market. "
               "Lower is better; this is the number that matters most because picks are published as probabilities.")
    return out


def save_json(obj: dict, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
