"""Weekly markdown recap: a 60-second script and a text post, written like a person talking."""

from __future__ import annotations

import polars as pl

from . import config


def city(abbr: str) -> str:
    return config.team_name(abbr)


def _pct(t: dict) -> str:
    return "no games yet" if t.get("pct") is None else f"{t['wins']}-{t['losses']}" + (f"-{t['pushes']}" if t.get("pushes") else "")


def write_recap(season: int, week: int, this_week: pl.DataFrame, last_week: pl.DataFrame | None, record: dict) -> str:
    lines = [f"# Week {week} picks, {season} season", ""]

    # 1. Grade last week honestly.
    if last_week is not None and last_week.height:
        lw_week = int(last_week["week"][0])
        hits = int(last_week["model_hit"].sum() or 0)
        n = last_week.filter(pl.col("model_hit").is_not_null()).height
        pushes = last_week.filter(pl.col("model_hit").is_null()).height
        lines.append(f"In Week {lw_week}, the model went {hits}-{n - hits} on {n} decisive graded games."
                     + (f" There were {pushes} tied game(s), counted as pushes and excluded from win percentages." if pushes else ""))
        common = last_week.filter(pl.col("model_hit").is_not_null() & pl.col("market_hit").is_not_null())
        if common.height:
            model_common = int(common["model_hit"].sum())
            market_common = int(common["market_hit"].sum())
            missing = n - common.height
            lines.append(
                f"On the {common.height} decisive games with closing moneylines, the model went "
                f"{model_common}-{common.height - model_common} and the moneyline favorite went "
                f"{market_common}-{common.height - market_common}, so "
                + ("we beat the market on those games." if model_common > market_common
                   else "the market did better on those games." if model_common < market_common
                   else "we tied the market on those games.")
                + (f" Closing moneylines were unavailable for {missing} decisive game(s)." if missing else "")
            )
        else:
            lines.append("No decisive graded games have closing moneylines available, so there is no market comparison yet.")
        wrong = last_week.filter(pl.col("model_hit") == 0).sort("pick_prob", descending=True)
        if wrong.height:
            w = wrong.row(0, named=True)
            lines.append(f"Worst miss: I had {city(w['pick'])} at {int(round(w['pick_prob'] * 100))} percent and they lost. That one is on me.")
        right = last_week.filter((pl.col("model_hit") == 1) & (pl.col("market_hit") == 0))
        if right.height:
            r = right.row(0, named=True)
            lines.append(f"Best call: {city(r['pick'])} at {int(round(r['pick_prob'] * 100))} percent when the market had the other side.")
        lines.append("")
    else:
        lines.append("No graded picks yet. This is week one of the public record, so everything from here counts.")
        lines.append("")

    # 2. The running record.
    o = record.get("overall", {})
    if record.get("n_graded"):
        lines.append(f"Running record: model {_pct(o['model'])}, always-home {_pct(o['home'])}, "
                     f"closing-spread favorite {_pct(o['spread'])}, moneyline favorite {_pct(o['market'])}.")
        lines.append("")

    # 3. This week's picks.
    if this_week.height:
        lines.append(f"This week, {this_week.height} games. Here is where I land.")
        lines.append("")
        for r in this_week.sort("pick_prob", descending=True).iter_rows(named=True):
            opp = r["away_team"] if r["pick"] == r["home_team"] else r["home_team"]
            where = "at home" if r["pick"] == r["home_team"] else "on the road"
            lines.append(f"- {city(r['pick'])} over {city(opp)} {where}, {int(round(r['pick_prob'] * 100))} percent, {r['confidence']}.")
        lines.append("")
        strong = this_week.filter(pl.col("confidence").is_in(["strong", "confident"]))
        flips = this_week.filter(pl.col("confidence") == "coin flip")
        lines.append(f"{strong.height} of those I actually like. {flips.height} are coin flips and I am only picking a side because the record demands it.")
        lines.append("")
    else:
        lines.append("No picks this week.")
        lines.append("")

    lines.append("The append-only pick log preserves each original prediction and its recorded timestamp. "
                 "This is analysis for entertainment, not betting advice.")
    return "\n".join(lines) + "\n"
