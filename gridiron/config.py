"""Project-wide constants. Everything tunable lives here so the weekly run is reproducible."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
PICKS_DIR = DATA_DIR / "picks"
PICKS_FILE = PICKS_DIR / "picks.csv"
GRADED_FILE = PICKS_DIR / "graded.csv"
RECORD_FILE = PICKS_DIR / "record.json"
REPORTS_DIR = ROOT / "reports"
BACKTEST_JSON = REPORTS_DIR / "backtest.json"
BACKTEST_MD = REPORTS_DIR / "backtest.md"
RECAPS_DIR = ROOT / "recaps"
SITE_DIR = ROOT / "site"
TEMPLATES_DIR = ROOT / "gridiron" / "templates"

SEED = 20261015  # fixed everywhere: numpy, sklearn, lightgbm

# Seasons used for training and picks. One extra season before FIRST_SEASON is
# fetched only to seed the priors for FIRST_SEASON, so week 1 of the first
# season is not computed from nothing.
FIRST_SEASON = 2015
CURRENT_SEASON = 2026
PRIOR_SEED_SEASON = FIRST_SEASON - 1
SEASONS = list(range(FIRST_SEASON, CURRENT_SEASON + 1))
# The model is fit only on games from this season onward. Earlier seasons still feed the
# features (team priors, quarterback history) but are not training rows. Set to FIRST_SEASON
# to train on everything. 2020 was chosen by the founder; see CHANGELOG 0.2.0 for the backtest
# comparison (2020 was also the empty-stadium season, with home teams winning under half).
TRAIN_FIRST_SEASON = 2020
EVAL_SEASONS = [2023, 2024, 2025]

# Franchise relocations. Play-by-play already uses the new codes for every
# season; the schedules file uses the historical code. We normalise to the
# current code so a franchise is one continuous team.
TEAM_RENAMES = {"STL": "LA", "SD": "LAC", "OAK": "LV"}
TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
    "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO",
    "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]
# City names only: no nicknames, logos or wordmarks anywhere in the product.
TEAM_CITY = {
    "ARI": "Arizona", "ATL": "Atlanta", "BAL": "Baltimore", "BUF": "Buffalo",
    "CAR": "Carolina", "CHI": "Chicago", "CIN": "Cincinnati", "CLE": "Cleveland",
    "DAL": "Dallas", "DEN": "Denver", "DET": "Detroit", "GB": "Green Bay",
    "HOU": "Houston", "IND": "Indianapolis", "JAX": "Jacksonville", "KC": "Kansas City",
    "LA": "Los Angeles", "LAC": "Los Angeles", "LV": "Las Vegas", "MIA": "Miami",
    "MIN": "Minnesota", "NE": "New England", "NO": "New Orleans", "NYG": "New York",
    "NYJ": "New York", "PHI": "Philadelphia", "PIT": "Pittsburgh", "SEA": "Seattle",
    "SF": "San Francisco", "TB": "Tampa Bay", "TEN": "Tennessee", "WAS": "Washington",
}

# Rolling-feature hyperparameters (see features.py for the formula).
EWMA_DECAY = 0.85          # weight multiplier per game of age
PRIOR_WEIGHT_GAMES = 4.0   # how many game-equivalents the preseason prior is worth
PRIOR_SHRINK = 0.5         # share of last season's team mean kept (rest -> league mean)
QB_DECAY = 0.90            # per-game decay for the quarterback series
QB_PRIOR_DROPBACKS = 150.0 # prior weight for a quarterback, in dropbacks
QB_REPLACEMENT_PENALTY = 0.05  # league mean minus this = replacement-level QB EPA/dropback
BYE_REST_DAYS = 13         # rest >= this means the team is coming off a bye

# Spread -> probability conversion for the "favorite by closing spread" baseline.
# Standard deviation of NFL margin of victory around the spread, rounded.
SPREAD_SIGMA = 13.5

# Plain-English confidence labels keyed on the pick's probability.
CONFIDENCE_LABELS = [
    (0.75, "strong"),
    (0.65, "confident"),
    (0.55, "lean"),
    (0.0, "coin flip"),
]

KICKOFF_TZ = "America/New_York"


def team_name(abbr: str) -> str:
    """City name, with the abbreviation added only where two teams share a city."""
    city = TEAM_CITY.get(abbr, abbr)
    shared = sum(1 for c in TEAM_CITY.values() if c == city) > 1
    return f"{city} ({abbr})" if shared else city
