"""Static site generation: picks, record, methodology, attribution. Output goes to site/."""

from __future__ import annotations

import datetime as dt
import json
import shutil

import polars as pl
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import __version__, config

DISCLAIMER = "Picks are analysis for entertainment only and are not betting advice."


def _env() -> Environment:
    env = Environment(loader=FileSystemLoader(config.TEMPLATES_DIR), autoescape=select_autoescape(["html"]))
    env.globals.update(city=lambda a: config.TEAM_CITY.get(a, a), disclaimer=DISCLAIMER, version=__version__)
    return env


def _pct(p) -> str:
    return "" if p is None else f"{p * 100:.0f}%"


def build_site(season: int | None, week: int | None, picks_this_week: pl.DataFrame, graded: pl.DataFrame,
               record: dict, backtest: dict | None, now: dt.datetime | None = None) -> None:
    now = now or dt.datetime.now(dt.timezone.utc)
    env = _env()
    config.SITE_DIR.mkdir(parents=True, exist_ok=True)

    picks = []
    for r in picks_this_week.sort("kickoff_utc").iter_rows(named=True):
        kickoff = dt.datetime.fromisoformat(r["kickoff_utc"])
        picks.append({
            **r,
            "kickoff_local": _format_kickoff(kickoff),
            "pick_pct": _pct(r["pick_prob"]), "p_home_pct": _pct(float(r["p_home"])),
        })

    recent = []
    if graded.height and "graded" in graded.columns:
        g = graded.filter(pl.col("graded")).sort(["season", "week", "kickoff_utc"], descending=[True, True, False])
        for r in g.head(64).iter_rows(named=True):
            recent.append({**r, "pick_pct": _pct(float(r["pick_prob"])),
                           "score": f"{r['away_team']} {int(r['away_score'])} at {r['home_team']} {int(r['home_score'])}"
                           if r.get("home_score") is not None else ""})

    ctx = {
        "season": season, "week": week, "picks": picks, "recent": recent, "record": record,
        "backtest": backtest, "generated": now.strftime("%Y-%m-%d %H:%M UTC"),
        "labels": record.get("labels", {}),
    }
    for page in ("index", "record", "methodology"):
        html = env.get_template(f"{page}.html").render(page=page, **ctx)
        (config.SITE_DIR / f"{page}.html").write_text(html)
    (config.SITE_DIR / "record.json").write_text(json.dumps(record, indent=2))
    if config.PICKS_FILE.exists():
        shutil.copy(config.PICKS_FILE, config.SITE_DIR / "picks.csv")


def _format_kickoff(kickoff: dt.datetime) -> str:
    import zoneinfo
    local = kickoff.astimezone(zoneinfo.ZoneInfo(config.KICKOFF_TZ))
    return local.strftime("%a %b %-d, %-I:%M %p ET")
