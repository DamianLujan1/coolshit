"""Stage one (logistic regression) and stage two (LightGBM), plus walk-forward evaluation.

Both models predict P(home team wins). Stage two ships only if it beats stage one on the
holdout; `evaluate.decide_stage` makes that call and the weekly command obeys it.
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
    feats = STAGES[stage]
    train = train.sort("game_id")  # row order must not change the fit
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


def training_set(frame: pl.DataFrame, season: int, week: int,
                 train_first_season: int | None = None) -> pl.DataFrame:
    """Rows a model predicting (season, week) may train on.

    Every played game from `train_first_season` (default config.TRAIN_FIRST_SEASON) up to the
    previous season, plus games of `season` with an earlier week. Nothing from the target week
    or later is ever included.
    """
    start = config.TRAIN_FIRST_SEASON if train_first_season is None else train_first_season
    return frame.filter(
        (pl.col("season") >= start)
        & ((pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week)))
    )


def walk_forward(frame: pl.DataFrame, stage: str, test_seasons: list[int],
                 min_train_games: int = 500, train_first_season: int | None = None) -> pl.DataFrame:
    """Predict each (season, week) in test_seasons using only earlier games.

    The training set for each test week is `training_set(frame, season, week)`.
    Returns the test rows with a `p_home` column.
    """
    frame = frame.sort(["season", "week", "kickoff_utc"])
    weeks = (frame.filter(pl.col("season").is_in(test_seasons))
             .select("season", "week").unique().sort(["season", "week"]))
    preds = []
    for season, week in weeks.iter_rows():
        train = training_set(frame, season, week, train_first_season)
        test = frame.filter((pl.col("season") == season) & (pl.col("week") == week))
        if train.height < min_train_games:
            continue
        m = fit(train, stage)
        preds.append(test.with_columns(pl.Series("p_home", m.predict_proba(test)), pl.lit(stage).alias("stage")))
    return pl.concat(preds) if preds else frame.head(0).with_columns(pl.lit(0.0).alias("p_home"), pl.lit(stage).alias("stage"))
