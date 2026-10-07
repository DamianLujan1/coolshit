"""Synthetic fixtures. A tiny league with deterministic play-by-play so tests run offline in milliseconds."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

TEAMS = ["AAA", "BBB", "CCC", "DDD"]
SKILL = {"AAA": 0.25, "BBB": 0.05, "CCC": -0.05, "DDD": -0.25}


def make_schedule(seasons=(2014, 2015), weeks=4, scored_through=(2015, 3)) -> pl.DataFrame:
    """Round-robin-ish schedule: each week two games among four teams, kickoff Sunday 13:00 ET."""
    rows = []
    for season in seasons:
        for week in range(1, weeks + 1):
            # rotate pairings so every team hosts and travels
            a, b, c, d = np.roll(TEAMS, week).tolist()
            day = dt.date(season, 9, 7) + dt.timedelta(days=7 * (week - 1))
            for home, away in ((a, b), (c, d)):
                played = (season, week) <= scored_through
                # margin follows skill plus a small home edge, so both home and away wins occur
                margin = round(20 * (SKILL[home] - SKILL[away])) + 2 + week
                hs, as_ = (20 + max(margin, 0), 20 + max(-margin, 0)) if played else (None, None)
                rows.append({
                    "game_id": f"{season}_{week:02d}_{away}_{home}", "season": season, "game_type": "REG",
                    "week": week, "gameday": day.isoformat(), "gametime": "13:00", "weekday": "Sunday",
                    "home_team": home, "away_team": away, "home_score": hs, "away_score": as_,
                    "result": (hs - as_) if played else None, "location": "Home",
                    "home_rest": 7, "away_rest": 7 if week != 3 else 14, "div_game": 1 if week % 2 else 0,
                    "spread_line": 3.0, "home_moneyline": -150, "away_moneyline": 130,
                    "home_qb_id": f"QB_{home}", "away_qb_id": f"QB_{away}",
                })
    return pl.DataFrame(rows)


def make_pbp(schedule: pl.DataFrame, plays_per_team: int = 30, seed: int = 0) -> pl.DataFrame:
    """Deterministic plays for every scored game. AAA's offense is good; DDD's is bad."""
    rng = np.random.default_rng(seed)
    rows = []
    skill = SKILL
    for g in schedule.filter(pl.col("result").is_not_null()).iter_rows(named=True):
        for team, opp in ((g["home_team"], g["away_team"]), (g["away_team"], g["home_team"])):
            for i in range(plays_per_team):
                is_pass = i % 3 != 2
                epa = skill[team] + rng.normal(0, 0.5)
                rows.append({
                    "game_id": g["game_id"], "season": g["season"], "week": g["week"], "season_type": "REG",
                    "posteam": team, "defteam": opp, "home_team": g["home_team"], "away_team": g["away_team"],
                    "play_type": "pass" if is_pass else "run", "pass": int(is_pass), "rush": int(not is_pass),
                    "qb_dropback": int(is_pass), "qb_scramble": 0, "down": float(1 + i % 4),
                    "epa": epa, "qb_epa": epa if is_pass else None, "success": int(epa > 0),
                    "passer_player_id": f"QB_{team}" if is_pass else None, "rusher_player_id": None if is_pass else f"RB_{team}",
                })
    return pl.DataFrame(rows)


@pytest.fixture
def schedule():
    from gridiron.data import normalise_schedules
    return normalise_schedules(make_schedule())


@pytest.fixture
def pbp(schedule):
    return make_pbp(schedule)
