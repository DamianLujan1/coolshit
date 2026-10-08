# Gridiron Model

NFL game win probabilities, published weekly with a public, timestamped record of every hit and miss.

One command on Tuesday night fetches the latest data, grades last week, retrains, logs this week's picks,
updates the running record against three baselines, writes a recap and regenerates the site.

**Picks are analysis for entertainment and are not betting advice.**

## Clone to picks in five minutes

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or plain `pip`).

```bash
git clone <this repo> gridiron && cd gridiron
uv venv .venv && uv pip install -p .venv/bin/python -e ".[dev]"   # or: python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                                        # 14 tests, about a second
.venv/bin/python -m gridiron fetch                                # downloads ~140 MB of nflverse parquet into data/cache (once)
.venv/bin/python -m gridiron backtest                             # ~25 s; writes reports/backtest.md and picks which stage ships
.venv/bin/python -m gridiron weekly --no-commit                   # grades, trains, logs picks, writes the recap and site/
```

Open `site/index.html` for the picks, `recaps/<season>_week_<ww>.md` for the recap.

## The Tuesday-night command

```bash
make weekly            # = .venv/bin/python -m gridiron weekly
```

It runs, in order:

1. **fetch**: re-downloads the schedules file and the current season's play-by-play; completed seasons stay cached.
   If the download fails and a cache exists, it uses the cache and warns.
2. **validate**: fails loudly if a season is missing, a team abbreviation is unmapped, a game that already
   kicked off has no score, or a scored game has no play-by-play.
3. **grade**: joins every logged pick to final scores, grades the model and the three baselines, writes
   `data/picks/graded.csv` and `data/picks/record.json`.
4. **train and pick**: trains the shipping stage on every completed game, predicts the upcoming week and appends
   the picks to `data/picks/picks.csv`. Already-picked games are skipped. A pick for a game that has
   kicked off is refused.
5. **recap**: writes `recaps/<season>_week_<ww>.md`, a plain-English script for a 60-second video or a text post.
6. **site**: regenerates `site/` (picks, record, methodology, attribution).
7. **commit**: `git add` + `git commit` of picks, recaps, reports and site. Push it yourself, or add `--deploy`
   to also run `vercel deploy --prod site`.

Flags: `--no-commit`, `--offline` (cache only), `--deploy`, `--season N --week W` to target a specific week,
and the global `--now <ISO datetime>` to run as if it were another time (used for dry runs).

## What is in the repo

```
gridiron/
  config.py     constants: seasons, team codes and city names, EWMA settings, seed
  data.py       nflreadpy loaders, parquet cache, team normalisation, kickoff times, validation
  features.py   per-game team stats from play-by-play, rolling EWMA features with priors, QB features
  models.py     stage 1 logistic regression, stage 2 LightGBM, walk-forward evaluation
  evaluate.py   accuracy / Brier / log loss / calibration, the three baselines, the backtest report
  picks.py      pick generation, append-only pick log, grading, running record
  recap.py      markdown recap
  site.py       static site generation (Jinja2 templates in gridiron/templates/)
  cli.py        `gridiron` command line
tests/          feature pipeline, leakage proof, grading and pick-log tests (synthetic data, offline)
data/picks/     picks.csv (append-only, committed), graded.csv, record.json
reports/        backtest.md / backtest.json (walk-forward results; decides which stage ships)
recaps/         one markdown recap per week
site/           generated static site, deployed to Vercel
```

Notebooks are welcome under `notebooks/` for exploration, but nothing in them is the source of truth.

## Data

Everything comes from [nflverse](https://github.com/nflverse) through the
[nflreadpy](https://github.com/nflverse/nflreadpy) package: the schedules/games dataset (scores, rest days,
closing spread, total and moneylines) and play-by-play with expected points added. Features are built for
2015 to 2026, with 2014 fetched only to seed the 2015 priors. The model itself trains only on games from 2020
onward (`TRAIN_FIRST_SEASON` in `gridiron/config.py`); earlier seasons feed team priors and quarterback history.

- nflverse data is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution is
  in the footer of every page of the site and here: **data by nflverse**.
- FTN charting data is share-alike licensed and is never loaded.
- The closing spread and moneylines are the benchmark. They are never model inputs.
- Team names are city names and standard abbreviations only. No logos, wordmarks or team nicknames appear
  anywhere in the product; the NFL is not shy about enforcing its marks.

Data files live in `data/cache/` and are gitignored. Nothing larger than a few megabytes is committed.

## The model

**Features** (all computed from games that kicked off strictly before the game in question):

- Offensive and defensive EPA per play, split pass/rush; success rate; early-down (1st and 2nd) EPA per play.
  Each is an exponentially weighted average of the team's prior games this season (decay 0.85 per game) blended
  with a prior worth four games, where the prior is last season's team mean pulled halfway to last season's
  league mean. Week 1 is pure prior; by mid-season the current year dominates.
- Home field (the model is oriented home vs. away, so the intercept is home-field advantage), neutral-site flag,
  rest-day differential, bye-week flags, division-game flag, playoff flag.
- Starting quarterback EPA per dropback: a dropback-weighted EWMA of the starter's prior games across seasons,
  with a 150-dropback prior at replacement level (last season's league rate minus 0.05). The starter is the one
  listed by nflverse (actual for played games, projected for the coming week), falling back to the team's
  most recent starter. A new starter with no history is treated as replacement level.

**Stage 1** is a logistic regression on home-minus-away differences plus the flags, trained on every completed
game from 2020 through the week before the one being predicted.
**Stage 2** is LightGBM on the same plus each side's raw values. Stage 2 ships only if it beats stage 1 on both
log loss and Brier in the walk-forward holdout; `gridiron backtest` makes that call and writes it to
`reports/backtest.json`, which `gridiron weekly` obeys.

**Evaluation** is strictly walk-forward: each test week of 2023, 2024 and 2025 is predicted by a model trained
on earlier seasons and earlier weeks of that season only. Baselines: always pick home, the closing-spread
favorite (probability from a normal margin model with sigma 13.5), and the vig-free moneyline. See
`reports/backtest.md`. Short version as of the first build: the logistic model is about 64% accurate with
good calibration, and it does not beat the closing line (about 68%). Calibration is what the picks are judged on.

**Leakage** is treated as a critical bug. `tests/test_leakage.py` scrambles every play and score from a game
onward and asserts that the game's features do not change, and checks that walk-forward training sets never
contain the test week.

## Engineering notes

- Python 3.11+, Polars, pinned dependencies in `pyproject.toml`, fixed seed (`config.SEED`), deterministic
  ordering everywhere a sort could tie.
- The pick log is append-only by construction: `append_picks` opens the file in append mode, skips games
  already present, refuses games that have kicked off, and verifies the existing bytes are unchanged.
  Git history is the audit trail.
- `CHANGELOG.md` tracks model and pipeline changes.

## Deploying the site

Live at **https://gridiron-model.vercel.app** (Vercel project `gridiron-model`, deployed from `site/`).

`site/` is plain HTML. `vercel.json` at the repo root tells Vercel to serve it with no build step, so connecting
the GitHub repo to a Vercel project deploys on every push. Alternatively `make deploy` runs
`vercel deploy --prod --yes site` with the Vercel CLI.

## Scope

The game model and its public record come first. Fantasy projections, other leagues and any paid tier are out
of scope until the model has graded at least two weeks of real picks.
