# Changelog

All notable changes to Gridiron Model. The pick log in `data/picks/picks.csv` is append-only and is not
covered here; its git history is the record.

## 0.2.0 - 2026-10-08

- Training window: the model now trains only on games from 2020 onward (`config.TRAIN_FIRST_SEASON`), at the
  founder's request. Earlier seasons still feed features (team priors, quarterback history). One function,
  `models.training_set`, now defines the training rows for both the weekly picks and the walk-forward backtest.
- Backtest effect, 2023 to 2025 walk-forward, stage 1, 854 games:

  | Training start | Accuracy | Brier | Log loss | ECE |
  |---|---:|---:|---:|---:|
  | 2015 (0.1.0) | 0.644 | 0.2194 | 0.6280 | 0.029 |
  | 2020 (0.2.0) | 0.646 | 0.2224 | 0.6351 | 0.031 |

  Accuracy is flat (two more correct picks out of 854); Brier, log loss and calibration are slightly worse.
  Later start years degrade log loss monotonically. 2020 was the empty-stadium season (home teams won 49.8%).
  Stage 1 still ships.
- Week 5 picks were logged under 0.1.0 and are not changed. Picks from week 6 onward carry model_version 0.2.0.

## 0.1.0 - 2026-10-07

First end-to-end version.

- Data: nflreadpy loaders for schedules and play-by-play (2014 for priors, 2015 to 2026 for modeling) with a
  local parquet cache, franchise-code normalisation (STL/SD/OAK to LA/LAC/LV), kickoff timestamps and a
  validation step that fails on missing seasons, unmapped teams, unscored past games or missing play-by-play.
  No FTN charting data is loaded.
- Features: per-team EPA per play (pass/rush, offense/defense), success rate and early-down EPA as
  exponentially weighted averages of prior games with a prior from the previous season; home field, rest
  differential, bye flags, division flag, neutral-site and playoff flags; starting quarterback EPA per dropback
  with a replacement-level prior and a last-starter fallback.
- Models: stage one logistic regression; stage two LightGBM, shipped only if it beats stage one on log loss
  and Brier in the walk-forward holdout.
- Evaluation: walk-forward over 2023, 2024 and 2025 with accuracy, Brier, log loss, ECE and 10-bin calibration;
  baselines: always home, closing-spread favorite, vig-free moneyline.
- Weekly command: fetch, validate, grade, train, log picks (append-only, refused after kickoff), update the
  record, write the recap, regenerate the site, commit.
- Site: this week's picks, running record, methodology, nflverse attribution and the entertainment-only footer.
