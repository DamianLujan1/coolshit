"""Stage one (logistic regression) and stage two (LightGBM), plus walk-forward evaluation.

Both models predict P(home team wins), conditional on a decisive result. Fixed
specifications are selected on 2022 only; 2023–2025 is a separate walk-forward audit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .features import STAGE1_FEATURES, STAGE2_FEATURES

log = logging.getLogger(__name__)

STAGES = {"stage1": STAGE1_FEATURES, "stage2": STAGE2_FEATURES}

# Explicitly reject benchmark/label leakage even if a future edit changes STAGES.
_FORBIDDEN_FEATURE_PARTS = ("moneyline", "spread", "odds", "market", "favorite", "total_line", "score", "result", "home_win")


def feature_names(stage: str) -> list[str]:
    if stage not in STAGES:
        raise ValueError(f"unknown model stage: {stage}")
    names = list(STAGES[stage])
    forbidden = [name for name in names if any(part in name.lower() for part in _FORBIDDEN_FEATURE_PARTS)]
    if forbidden:
        raise ValueError(f"benchmarks and outcomes cannot be model features: {forbidden}")
    return names


def model_specification() -> dict:
    """The fixed candidates, recorded with each selection/audit report."""
    return {
        "stage1": {"algorithm": "StandardScaler + LogisticRegression", "features": feature_names("stage1"),
                   "C": 0.5, "max_iter": 5000, "tol": 1e-8, "random_state": config.SEED},
        "stage2": {"algorithm": "LightGBM", "features": feature_names("stage2"),
                   "n_estimators": 400, "learning_rate": 0.02, "num_leaves": 7, "min_child_samples": 40,
                   "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 5.0,
                   "random_state": config.SEED, "deterministic": True, "n_jobs": 1},
    }


@dataclass
class FittedModel:
    stage: str
    features: list[str]
    model: object
    n_train: int

    def predict_proba(self, df: pl.DataFrame) -> np.ndarray:
        x = df.select(self.features).to_numpy().astype(float)
        if self.stage == "stage1":
            return self.model.predict_proba(x)[:, 1]
        return self.model.predict(x)


def fit(train: pl.DataFrame, stage: str = "stage1") -> FittedModel:
    feats = feature_names(stage)
    train = train.sort("game_id")  # row order must not change the fit
    if train.height == 0 or not train["home_win"].is_in([0, 1]).all() or train["home_win"].null_count():
        raise ValueError("training requires nonempty decisive-game binary labels; exclude ties and unfinished games")
    x = train.select(feats).to_numpy().astype(float)
    y = train["home_win"].to_numpy().astype(int)
    if stage == "stage1":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=5000, tol=1e-8, random_state=config.SEED),
        )
        model.fit(x, y)
    elif stage == "stage2":
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.02, num_leaves=7, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
            random_state=config.SEED, deterministic=True, n_jobs=1, verbose=-1,
        )
        model.fit(x, y)
        booster = model.booster_
        model = _LgbWrapper(booster)
    else:
        raise ValueError(stage)
    return FittedModel(stage=stage, features=feats, model=model, n_train=len(y))


class _LgbWrapper:
    """Thin wrapper so both stages expose the same call."""

    def __init__(self, booster):
        self.booster = booster

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.booster.predict(x)


def walk_forward(frame: pl.DataFrame, stage: str, test_seasons: list[int],
                 min_train_games: int = 500) -> pl.DataFrame:
    """Predict each (season, week) in test_seasons using only earlier games.

    Training set for week W of season S = decisive games from prior seasons/weeks
    whose reconstructed completion time precedes that week's prediction cutoff.
    Features are already built at that cutoff. Returns rows with `p_home`.
    """
    has_cutoff = "prediction_time" in frame.columns
    has_availability = "history_available_at" in frame.columns
    if has_cutoff != has_availability:
        raise ValueError("walk-forward needs both prediction_time and history_available_at")
    frame = frame.sort(["season", "week", "kickoff_utc"])
    weeks = (frame.filter(pl.col("season").is_in(test_seasons))
             .select("season", "week").unique().sort(["season", "week"]))
    preds = []
    for season, week in weeks.iter_rows():
        train = frame.filter((pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week)))
        test = frame.filter((pl.col("season") == season) & (pl.col("week") == week))
        if has_cutoff:
            cutoff = test["prediction_time"].min()
            if cutoff is None:
                raise ValueError(f"missing prediction cutoff for {season} week {week}")
            train = train.filter(pl.col("history_available_at").is_not_null() & (pl.col("history_available_at") <= cutoff))
        if train.height < min_train_games:
            continue
        m = fit(train, stage)
        latest = train["history_available_at"].max() if has_availability else None
        preds.append(test.with_columns(
            pl.Series("p_home", m.predict_proba(test)), pl.lit(stage).alias("stage"),
            pl.lit(train.height).alias("n_train_games"),
            pl.lit(latest).cast(pl.Datetime("us", "UTC")).alias("train_latest_available_at"),
        ))
    return pl.concat(preds) if preds else frame.head(0).with_columns(
        pl.lit(0.0).alias("p_home"), pl.lit(stage).alias("stage"), pl.lit(0).alias("n_train_games"),
        pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("train_latest_available_at"),
    )
