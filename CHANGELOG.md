# Changelog

All notable changes to Gridiron Model. The pick log in `data/picks/picks.csv` is append-only and is not
covered here; its git history is the record.

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
