# Changelog

All notable changes to Gridiron Model. The pick log in `data/picks/picks.csv` is append-only and is not
covered here; its git history is the record.

## Weekly delivery and provenance clarification - 2026-10-15

- Grade the 15 original Week 5 picks (10–4–1) and append 14 Week 6 picks issued at the actual pre-kickoff clock, with causal QB evidence. Preserve all original pick bytes.
- Exclude unversioned target-schedule QB names from future provenance serialization. Preserve the issued evidence and add separate append-only annotations; causal numeric inputs and issued probabilities are unchanged.
- Retain the frozen historical audit: the reconstructed models did not beat the market.

## Publication verification correction - 2026-10-09

- Remove `vercel curl` from publication. Verify public assets as raw bytes over unauthenticated HTTPS, without proxies, cookies, auth headers, redirects or bypass tokens. Protected previews are explicitly recorded unverified and stop CLI promotion.
- Require an existing explicit token, pin reviewed CLI 63.1.0, isolate global configuration, disable refresh/login/update paths, reject anonymous credential state, and allowlist publication commands. Sanitize command failures.
- Add publication-side-effect regression tests and expose the full backtest as a public static asset. Model formulas, audit results and all issued picks remain unchanged.

## 0.1.1 - 2026-10-09 (local repair; unpublished)

- Remove target-game PBP and unversioned schedule starter IDs from QB resolution. Use captured pre-prediction evidence or the lagged completed-game dominant passer/replacement prior; preserve source, observation and prediction timestamps for new picks.
- Apply common historical Tuesday cutoffs and conservative completed-game availability to team/QB history and weekly training. Regression tests mutate passer identities, counts, target/future plays and schedule starter values.
- Select logistic versus LightGBM using 2022 only, then freeze the choice for the 2023–2025 audit. Preserve both models' raw predictions, per-season/combined scoring and ten-bin calibration, and compare market/favorite baselines on identical games. Report ties and rebuilt-data limitations.
- Add nflreadpy weekly team/player statistics, strict offline cache handling, schema/coverage checks and cache provenance. Exclude FTN and participation data explicitly.
- Add transitive dependency lock and pin the build backend; correct the Python minimum to 3.12 for the pinned NumPy version.
- Reject synthetic-clock canonical mutations; rehearse weekly outputs only in a fresh isolated directory. Preserve all original Week 5 picks and their legacy CSV schema. New QB provenance uses a separate append-only JSONL file.
- Require explicit publication with clean-checkout/project/branch checks, commit, push, staged deployment verification and promotion. No publication was executed during this repair.
- Correct missing-odds grading and recap comparisons, preserve ties as pushes, and remove unsupported public-publication/calibration claims from generated content.

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
