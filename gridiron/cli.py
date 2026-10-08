"""Command line. `gridiron weekly` is the one command the founder runs on Tuesday night."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys

import numpy as np
import polars as pl

from . import config, data, evaluate, features, models, picks as P, recap, site

log = logging.getLogger("gridiron")


def _now(args) -> dt.datetime:
    if getattr(args, "now", None):
        return dt.datetime.fromisoformat(args.now).astimezone(dt.timezone.utc)
    return dt.datetime.now(dt.timezone.utc)


def _load(refresh: bool, now: dt.datetime):
    sched = data.load_schedules(refresh=refresh)
    pbp = data.load_pbp(refresh_current=refresh)
    info = data.validate(sched, pbp, now=now)
    log.info("data ok: seasons %s..%s, %d games, %d plays, upcoming %s",
             info["seasons"][0], info["seasons"][-1], info["games"], info["plays"], info["upcoming"])
    return sched, pbp


def _features(sched: pl.DataFrame, pbp: pl.DataFrame) -> pl.DataFrame:
    feats = features.build_game_features(sched, pbp)
    return feats.filter(pl.col("season") >= config.FIRST_SEASON)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_fetch(args):
    now = _now(args)
    _load(refresh=True, now=now)
    print("fetched and validated")


def cmd_validate(args):
    now = _now(args)
    sched, pbp = _load(refresh=False, now=now)
    print(json.dumps(data.validate(sched, pbp, now=now), default=str, indent=2))


def cmd_backtest(args):
    now = _now(args)
    np.random.seed(config.SEED)
    sched, pbp = _load(refresh=args.refresh, now=now)
    frame = features.modeling_frame(_features(sched, pbp))
    preds, extra = {}, {}
    stages = ["stage1", "stage2"] if not args.stage1_only else ["stage1"]
    for stage in stages:
        wf = models.walk_forward(frame, stage, config.EVAL_SEASONS + [config.CURRENT_SEASON])
        preds[stage] = wf.filter(pl.col("season").is_in(config.EVAL_SEASONS))
        cur = wf.filter(pl.col("season") == config.CURRENT_SEASON)
        if cur.height:
            extra[stage] = evaluate.metrics(cur["p_home"].to_numpy(), cur["home_win"].to_numpy())
        log.info("walk-forward %s done (%d holdout games)", stage, preds[stage].height)
    home_rate = float(frame.filter(pl.col("season") < config.EVAL_SEASONS[0])["home_win"].mean())
    result = evaluate.evaluate_backtest(preds, home_rate)
    if extra:
        cur_ids = preds and wf.filter(pl.col("season") == config.CURRENT_SEASON)
        base = evaluate.add_baselines(cur_ids, home_rate).drop_nulls(["p_base_market"])
        y = base["home_win"].to_numpy()
        for c in ("p_base_home", "p_base_spread", "p_base_market"):
            extra[c] = evaluate.metrics(base[c].to_numpy(), y)
    result["generated_at"] = now.isoformat(timespec="seconds")
    result["current_season_to_date"] = extra
    evaluate.save_json(result, config.BACKTEST_JSON)
    config.BACKTEST_MD.write_text(evaluate.render_backtest_md(result, extra))
    print(evaluate.render_backtest_md(result, extra))
    print(f"wrote {config.BACKTEST_JSON} and {config.BACKTEST_MD}")


def _stage_choice() -> str:
    if config.BACKTEST_JSON.exists():
        return json.loads(config.BACKTEST_JSON.read_text()).get("stage_choice", "stage1")
    log.warning("no backtest.json found; defaulting to stage1. Run `gridiron backtest`.")
    return "stage1"


def _train_and_pick(feats: pl.DataFrame, season: int, week: int, now: dt.datetime, stage: str | None = None):
    stage = stage or _stage_choice()
    frame = features.modeling_frame(feats)
    train = models.training_set(frame, season, week)
    model = models.fit(train, stage)
    target = feats.filter((pl.col("season") == season) & (pl.col("week") == week)
                          & pl.col("result").is_null() & (pl.col("kickoff_utc") > now))
    target = target.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in model.features]))
    if target.height == 0:
        return stage, model, P.empty_picks()
    p = model.predict_proba(target)
    return stage, model, P.make_picks(target, p, season, week, stage, model.n_train, now=now)


def cmd_picks(args):
    now = _now(args)
    sched, pbp = _load(refresh=args.refresh, now=now)
    season, week = _target_week(sched, args, now)
    feats = _features(sched, pbp)
    stage, model, new = _train_and_pick(feats, season, week, now, args.stage)
    added = P.append_picks(new, now=now) if not args.dry_run else 0
    print(new.select(["game_id", "pick", "pick_prob", "confidence", "p_home"]))
    print(f"{stage}: {new.height} picks generated for {season} week {week}; {added} new rows appended to {config.PICKS_FILE}")


def _target_week(sched, args, now):
    if getattr(args, "season", None) and getattr(args, "week", None):
        return args.season, args.week
    up = data.upcoming_week(sched, now)
    if up is None:
        sys.exit("no upcoming games: season over")
    return up


def cmd_grade(args):
    now = _now(args)
    sched, _ = data.load_schedules(refresh=args.refresh), None
    graded, rec = _grade(sched)
    print(json.dumps(rec["overall"], indent=2))
    print(f"{rec['n_graded']} graded, {rec['n_pending']} pending; wrote {config.GRADED_FILE} and {config.RECORD_FILE}")


def _grade(sched: pl.DataFrame):
    picks = P.load_picks()
    graded = P.grade(picks, sched)
    rec = P.running_record(graded)
    config.PICKS_DIR.mkdir(parents=True, exist_ok=True)
    if graded.height:
        graded.write_csv(config.GRADED_FILE)
    P.save_record(rec)
    return graded, rec


def cmd_site(args):
    now = _now(args)
    sched = data.load_schedules(refresh=False)
    graded, rec = _grade(sched)
    picks = P.load_picks()
    season, week = _latest_picked_week(picks)
    this_week = picks.filter((pl.col("season") == season) & (pl.col("week") == week)) if season else P.empty_picks()
    backtest = json.loads(config.BACKTEST_JSON.read_text()) if config.BACKTEST_JSON.exists() else None
    site.build_site(season, week, this_week, graded, rec, backtest, now=now)
    print(f"site written to {config.SITE_DIR}")


def _latest_picked_week(picks: pl.DataFrame):
    if picks.height == 0:
        return None, None
    row = picks.select(pl.col("season").cast(pl.Int64), pl.col("week").cast(pl.Int64)).sort(["season", "week"]).row(-1)
    return int(row[0]), int(row[1])


def cmd_recap(args):
    now = _now(args)
    sched = data.load_schedules(refresh=False)
    graded, rec = _grade(sched)
    picks = P.load_picks()
    season, week = _latest_picked_week(picks)
    if season is None:
        sys.exit("no picks to recap")
    path = _write_recap(graded, picks, rec, season, week)
    print(path.read_text())


def _write_recap(graded, picks, rec, season, week):
    this_week = picks.filter((pl.col("season").cast(pl.Int64) == season) & (pl.col("week").cast(pl.Int64) == week))
    last = graded.filter(pl.col("graded") & (pl.col("season").cast(pl.Int64) == season)) if graded.height and "graded" in graded.columns else None
    last_week = None
    if last is not None and last.height:
        lw = int(last["week"].cast(pl.Int64).max())
        last_week = last.filter(pl.col("week").cast(pl.Int64) == lw)
    text = recap.write_recap(season, week, this_week, last_week, rec)
    config.RECAPS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RECAPS_DIR / f"{season}_week_{week:02d}.md"
    path.write_text(text)
    return path


def cmd_weekly(args):
    """Tuesday night: fetch, validate, grade, train, pick, record, recap, site, commit."""
    now = _now(args)
    np.random.seed(config.SEED)
    sched, pbp = _load(refresh=not args.offline, now=now)
    season, week = _target_week(sched, args, now)
    log.info("target: %s week %s", season, week)

    feats = _features(sched, pbp)
    stage, model, new = _train_and_pick(feats, season, week, now)
    added = P.append_picks(new, now=now)
    log.info("%s trained on %d games; %d picks, %d newly logged", stage, model.n_train, new.height, added)

    graded, rec = _grade(sched)
    picks = P.load_picks()
    recap_path = _write_recap(graded, picks, rec, season, week)
    backtest = json.loads(config.BACKTEST_JSON.read_text()) if config.BACKTEST_JSON.exists() else None
    this_week = picks.filter((pl.col("season").cast(pl.Int64) == season) & (pl.col("week").cast(pl.Int64) == week))
    site.build_site(season, week, this_week, graded, rec, backtest, now=now)

    print(recap_path.read_text())
    print(f"record: model {rec['overall']['model']}")
    if not args.no_commit:
        _git_commit(f"Week {week} picks, {season}: {new.height} games, {added} new; record updated")
    if args.deploy:
        _deploy()


def _git_commit(message: str) -> None:
    paths = [str(p) for p in (config.PICKS_DIR, config.RECAPS_DIR, config.SITE_DIR, config.REPORTS_DIR)]
    subprocess.run(["git", "add", *paths], check=True, cwd=config.ROOT)
    status = subprocess.run(["git", "status", "--porcelain", *paths], capture_output=True, text=True, cwd=config.ROOT).stdout
    if not status.strip():
        log.info("nothing new to commit")
        return
    subprocess.run(["git", "commit", "-q", "-m", message], check=True, cwd=config.ROOT)
    log.info("committed: %s", message)


def _deploy() -> None:
    cmd = ["vercel", "deploy", "--prod", "--yes", str(config.SITE_DIR)]
    log.info("running %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=config.ROOT)


# --------------------------------------------------------------------------- #

def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="gridiron", description="NFL game probabilities with a public record")
    ap.add_argument("--now", help="override the clock (ISO datetime), for tests and dry runs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch", help="download the latest data and validate it").set_defaults(fn=cmd_fetch)
    sub.add_parser("validate", help="validate cached data without downloading").set_defaults(fn=cmd_validate)

    b = sub.add_parser("backtest", help="walk-forward evaluation on 2023-2025; decides which stage ships")
    b.add_argument("--refresh", action="store_true")
    b.add_argument("--stage1-only", action="store_true")
    b.set_defaults(fn=cmd_backtest)

    pk = sub.add_parser("picks", help="train on everything played and log picks for the upcoming week")
    pk.add_argument("--refresh", action="store_true")
    pk.add_argument("--season", type=int)
    pk.add_argument("--week", type=int)
    pk.add_argument("--stage", choices=["stage1", "stage2"])
    pk.add_argument("--dry-run", action="store_true", help="print picks without logging them")
    pk.set_defaults(fn=cmd_picks)

    g = sub.add_parser("grade", help="grade logged picks against final scores and update the record")
    g.add_argument("--refresh", action="store_true")
    g.set_defaults(fn=cmd_grade)

    sub.add_parser("recap", help="write the markdown recap for the latest picked week").set_defaults(fn=cmd_recap)
    sub.add_parser("site", help="regenerate the static site").set_defaults(fn=cmd_site)

    w = sub.add_parser("weekly", help="the Tuesday-night command: everything, then commit")
    w.add_argument("--offline", action="store_true", help="use cached data only")
    w.add_argument("--no-commit", action="store_true")
    w.add_argument("--deploy", action="store_true", help="also run `vercel deploy --prod` on site/")
    w.add_argument("--season", type=int)
    w.add_argument("--week", type=int)
    w.set_defaults(fn=cmd_weekly)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
