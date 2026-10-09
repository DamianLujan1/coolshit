# Gridiron Model

NFL game win probabilities with an append-only pick history, a public record, and honest comparisons with the market. **For entertainment only; not betting advice.** City names and abbreviations only; no logos, fantasy, other leagues, or paid tier.

Version 0.1.1 repairs a historical quarterback identity leak. All 15 Week 5 picks issued with version 0.1.0 remain byte-for-byte unchanged. They are not rerun or relabeled as repaired predictions. The new backtest replaces the flawed research results, not the original picks.

## Local setup and verification

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required. Direct dependencies and the build backend are pinned in `pyproject.toml`; `uv.lock` locks transitive dependencies. The repair was verified on macOS with Python 3.14.6.

```bash
uv sync --locked --extra dev
.venv/bin/python -m pytest
.venv/bin/python -m gridiron fetch
.venv/bin/python -m gridiron validate
.venv/bin/python -m gridiron backtest --offline
```

`fetch` downloads schedules, play-by-play, weekly team statistics and weekly player statistics through [nflreadpy](https://github.com/nflverse/nflreadpy). Seasons 2015–2026 are modeled; 2014 supplies priors. All processing uses Polars. Team/player statistics are cached and validated for coverage; model performance features are derived from play-by-play. Betting data is never a feature.

Caches live under `data/cache/` and are gitignored. Completed seasons are reused; schedules and current-season datasets refresh in weekly online runs. An unavailable download can use a cached copy with a warning. `--offline` never downloads and fails on a cache miss. `validate` is offline. Validation checks schemas, seasons, teams, duplicate keys, scores, stale unfinished games, and coverage. New cache metadata records actual retrieval time, source loader, version and checksum; a legacy cache does not invent a retrieval timestamp.

Only PBP, schedules, team stats and player stats are allowed. **FTN charting and participation datasets are excluded.** Data by [nflverse](https://github.com/nflverse), with visible [CC BY 4.0 attribution](https://creativecommons.org/licenses/by/4.0/) in the site footer.

## Weekly local run and rehearsal

```bash
make weekly
# or: .venv/bin/python -m gridiron weekly --offline
```

This fetches/validates, retrains the selected model, appends genuinely new pre-kickoff picks, grades existing picks, updates the record, writes a recap and builds the static site. It is local by default. `--commit` explicitly adds a local artifact commit. `--season N --week W` selects a week; already-issued game IDs are skipped. New picks use the real issuance time and are checked against the real clock before append. No prior pick row or CSV header is rewritten.

A full rehearsal uses a new output directory:

```bash
make dry-run OUTPUT_DIR=/tmp/gridiron-rehearsal NOW=2026-10-07T21:00:00+00:00
# Equivalent:
.venv/bin/python -m gridiron --now 2026-10-07T21:00:00+00:00 weekly \
  --offline --dry-run --output-dir /tmp/gridiron-rehearsal
```

Use only one of those commands for a given directory. A rehearsal copies the original record, generates isolated picks/grades/recap/site, labels the directory with `DRY_RUN.json`, and cannot commit, push or deploy. It is a simulation against rebuilt data, not evidence that a forecast existed historically. Global `--now` is rejected for canonical pick/history/report mutations. `picks --dry-run` prints predictions without logging them.

Newly appended picks retain a separate append-only quarterback provenance record in `data/picks/qb_inputs.jsonl`. It contains the resolved identity, source and source game, evidence/observation timestamps, prediction cutoff and EPA estimate. Legacy picks have no invented provenance.

## Explicit one-command publication

After review, approved Vercel setup and an upstream branch are in place:

```bash
make publish PUBLISH_BRANCH=main
```

This is the explicit online weekly workflow through commit, push, deployment verification and promotion. It requires a clean reviewed checkout, an empty index, the named checked-out upstream branch, the `DamianLujan1/coolshit` remote, and a local link to the existing `gridiron-model` Vercel project. It refuses to create a new project implicitly. The CLI path requires an explicitly supplied existing `VERCEL_TOKEN` and reviewed CLI version 63.1.0. It uses an isolated empty global configuration, disables login/refresh/update flows, and refuses anonymous credential files. It pushes the artifact commit first, stages a production-target deployment without moving the public domain, then checks exact pick CSV and methodology bytes using unauthenticated HTTPS. It never calls `vercel curl`, creates bypass access, or changes protection. A protected preview is recorded as `protected_unverified` in `reports/publication-verification.json` and stops promotion. Only a publicly verified stage can be promoted by this CLI path; production is checked without authentication too. Failed checks stop later steps; a pushed commit can remain even if deployment fails.

The initial repair was committed locally without publication. See [deployment diagnosis](reports/deployment-diagnosis.md): the two Git deployment attempts failed `git_info_fail`; the old live site has no recorded Git SHA; the installed isolated CLI has an expired saved credential, so the CLI publication path cannot run without an explicitly supplied valid existing credential. The separately authorized connector source-files route does not use that CLI credential. Do not change access permissions or persistent credentials without approval.

## Causal features and QB evidence

The model predicts home win probability conditional on a decisive result. Inputs are:

- Offensive/defensive pass and rush EPA per play, success rate and early-down EPA. Prior games receive exponentially decaying weights (0.85 per game). A four-game prior blends last season's team and league means equally.
- Home advantage through the logistic intercept, neutral/playoff/division flags, rest difference and bye flags.
- QB EPA per dropback with a 150-dropback replacement prior at last season's league rate minus 0.05; history decays by 0.90 per prior QB game.

Historical features use Tuesday **12:00 America/New_York before the week's first kickoff**, shared across the week. Prior-game information requires a final score and a conservative availability proxy of kickoff plus 12 hours. A postponed earlier-week game cannot enter before that cutoff. Live future-game features use the actual prediction time. Both history and previous-season priors obey that cutoff.

QB resolution accepts separate timestamped evidence only when both its source availability and capture time precede the prediction. Otherwise it uses the dominant passer from the team's most recent eligible completed game, or replacement level with no history. The dominant passer is a lagged proxy and may be an injury replacement; it is not a claim about an announced starter. The latest schedule's actual starter is **never** accepted as proof of historical Tuesday availability. Target-game PBP cannot choose that game's QB.

The feature API accepts an optional `qb_inputs` Polars frame with `game_id`, `team`, `qb_id`, `available_at`, `observed_at`, and `source`. Timestamp columns must be timezone-aware. Current weekly runs use the lagged fallback unless separately captured evidence is supplied through that API; no historical news archive is fabricated.

## Selection, audit and limitations

Stage 1 is fixed logistic regression. Stage 2 is fixed LightGBM. **Only 2022 selects the shipping stage:** boosting must strictly improve both Brier and log loss on the same decisive games. The choice is frozen before the 2023–2025 walk-forward audit. Each weekly training set contains only earlier weeks available at the prediction cutoff. Both models are reported regardless of selection. No hyperparameters are tuned on the audit.

[Backtest report](reports/backtest.md) and [JSON](reports/backtest.json) include each audit season and the combined accuracy, Brier, log loss, ECE and ten-bin calibration for both models and all baselines. Raw selection/audit prediction CSVs preserve inputs and provenance. Baselines are always home (fixed pre-2022 home-win rate for probability scoring), closing-spread favorite (normal margin model, sigma 13.5), and proportionally normalized no-vig moneylines. Missing odds do not drop model games; market comparisons use identical covered games.

Tied games are excluded from binary training and historical scoring, reported explicitly, and recorded as pushes when grading published picks. A 0.5 probability picks home. Missing live benchmark odds remain ungraded for that benchmark, rather than silently becoming home picks.

These results are a **rebuilt retrospective audit**, not an untouched prospective test. The prior flawed build already examined 2023–2025. Current nflverse files can include later statistical corrections and revised schedules; original Tuesday file vintages and true publication times are unavailable. Closing lines also contain information later than Tuesday. The lagged QB proxy can miss pregame injuries or starter changes. These limitations remain after removing target-game leakage.

## Files

- `gridiron/data.py`, `features.py`: loading, validation, causal inputs and provenance.
- `models.py`, `evaluate.py`: fixed candidates, weekly training and independent selection/audit reports.
- `cli.py`, `picks.py`: clock guards, append-only logging, grading and explicit publication.
- `site.py`, `templates/`, `recap.py`: fan-facing site, methodology, record and recap.
- `tests/`: offline synthetic causal, workflow, evaluation, data and grading regressions.
- `data/picks/`: preserved picks and record; `reports/`: reproducible audit and repair evidence.

On this Mac `/usr/bin/git` was blocked by the unaccepted Xcode license. The installed Command Line Tools Git worked without changing system settings: `PATH=/Library/Developer/CommandLineTools/usr/bin:$PATH` for Git-dependent local checks.
