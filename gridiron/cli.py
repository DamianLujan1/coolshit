"""Command line. `gridiron weekly` is the one command the founder runs on Tuesday night."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager

import numpy as np
import polars as pl

from . import config, data, evaluate, features, models, picks as P, recap, site

log = logging.getLogger("gridiron")
VERCEL_PROJECT_ID = "prj_ewgxRzaTU7Kga2OzkRco28m7TFnp"
VERCEL_ORG_ID = "team_RNPL71TWtCTxnPNal4E8dGvM"
VERCEL_SCOPE = "damianlujan1s-projects"
_MAC_GIT = Path("/Library/Developer/CommandLineTools/usr/bin/git")
GIT = os.environ.get("GRIDIRON_GIT", str(_MAC_GIT) if sys.platform == "darwin" and _MAC_GIT.exists() else "git")


def _now(args) -> dt.datetime:
    if getattr(args, "now", None):
        value = dt.datetime.fromisoformat(args.now)
        if value.tzinfo is None:
            raise ValueError("--now must include a UTC offset")
        return value.astimezone(dt.timezone.utc)
    return dt.datetime.now(dt.timezone.utc)


def _load(refresh: bool, now: dt.datetime, *, offline: bool = False, with_validation: bool = False):
    sched = data.load_schedules(refresh=refresh, offline=offline)
    pbp = data.load_pbp(refresh_current=refresh, offline=offline)
    team_stats = data.load_team_stats(refresh_current=refresh, offline=offline)
    player_stats = data.load_player_stats(refresh_current=refresh, offline=offline)
    info = data.validate(sched, pbp, now=now, team_stats=team_stats, player_stats=player_stats)
    log.info("data ok: seasons %s..%s, %d games, %d plays, upcoming %s",
             info["seasons"][0], info["seasons"][-1], info["games"], info["plays"], info["upcoming"])
    return (sched, pbp, info) if with_validation else (sched, pbp)


def _features(sched: pl.DataFrame, pbp: pl.DataFrame, *, prediction_time=None) -> pl.DataFrame:
    feats = features.build_game_features(sched, pbp, prediction_time=prediction_time)
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
    sched, pbp, info = _load(refresh=False, now=now, offline=True, with_validation=True)
    print(json.dumps(info, default=str, indent=2))


def cmd_backtest(args):
    _guard_execution(args, command="backtest")
    import hashlib
    import importlib.metadata

    now = _now(args)
    np.random.seed(config.SEED)
    sched, pbp = _load(refresh=args.refresh, now=now, offline=not args.refresh)
    feats = _features(sched, pbp)
    frame = features.modeling_frame(feats)
    stages = ["stage1", "stage2"] if not args.stage1_only else ["stage1"]
    home_rate = float(frame.filter(pl.col("season") < evaluate.SELECTION_SEASONS[0])["home_win"].mean())
    selection_preds = {}
    for stage in stages:
        selection_preds[stage] = models.walk_forward(frame, stage, evaluate.SELECTION_SEASONS)
        log.info("2022 selection %s complete (%d games)", stage, selection_preds[stage].height)
    selection = evaluate.evaluate_backtest(selection_preds, home_rate)
    frozen_choice = evaluate.decide_stage(selection["combined"])
    log.info("frozen selection before audit: %s (2022 Brier AND log-loss gate)", frozen_choice)
    audit_preds = {}
    for stage in stages:
        audit_preds[stage] = models.walk_forward(frame, stage, evaluate.AUDIT_SEASONS)
        log.info("2023–2025 audit %s complete (%d games)", stage, audit_preds[stage].height)
    result = evaluate.evaluate_frozen_backtest(selection_preds, audit_preds, home_rate)
    assert result["stage_choice"] == frozen_choice
    result["model_specification"] = models.model_specification()
    result["feature_parameters"] = {key: getattr(config, key) for key in (
        "EWMA_DECAY", "PRIOR_WEIGHT_GAMES", "PRIOR_SHRINK", "QB_DECAY", "QB_PRIOR_DROPBACKS",
        "QB_REPLACEMENT_PENALTY", "BYE_REST_DAYS", "SPREAD_SIGMA", "FIRST_SEASON", "PRIOR_SEED_SEASON", "SEED")}
    result["generated_at"] = now.isoformat(timespec="seconds")
    result["home_rate_training_seasons"] = sorted(frame.filter(pl.col("season") < 2022)["season"].unique().to_list())
    result["sample_accounting"] = {}
    for season in evaluate.SELECTION_SEASONS + evaluate.AUDIT_SEASONS:
        completed = feats.filter((pl.col("season") == season) & pl.col("result").is_not_null())
        ties = completed.filter(pl.col("result") == 0).height
        predicted = selection_preds["stage1"] if season in evaluate.SELECTION_SEASONS else audit_preds["stage1"]
        modeled = predicted.filter(pl.col("season") == season).height
        result["sample_accounting"][str(season)] = {
            "completed": completed.height, "ties": ties, "decisive": completed.height - ties,
            "modeled": modeled, "other_exclusions": completed.height - ties - modeled,
        }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    prediction_paths = []
    for phase, predictions in (("selection_2022", selection_preds), ("audit_2023_2025", audit_preds)):
        raw = predictions["stage1"].rename({"p_home": "p_stage1"}).drop("stage")
        for stage in stages[1:]:
            raw = raw.join(predictions[stage].select("game_id", pl.col("p_home").alias("p_" + stage)), on="game_id")
        raw = evaluate.add_baselines(raw, home_rate).with_columns(pl.lit(phase).alias("evaluation_phase"))
        path = config.REPORTS_DIR / f"{phase}_predictions.csv"
        raw.sort(["season", "week", "game_id"]).write_csv(path)
        prediction_paths.append(path)
    source_paths = sorted((config.ROOT / "gridiron").glob("*.py")) + [config.ROOT / "pyproject.toml"]
    cache_paths = sorted(config.CACHE_DIR.rglob("*.parquet"))
    def digest(path):
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        return {"path": str(path.relative_to(config.ROOT)), "bytes": path.stat().st_size, "sha256": h.hexdigest()}
    result["reproducibility"] = {
        "sources": [digest(path) for path in source_paths],
        "cached_data": [digest(path) for path in cache_paths],
        "raw_predictions": [digest(path) for path in prediction_paths],
        "packages": {name: importlib.metadata.version(name) for name in (
            "nflreadpy", "polars", "numpy", "scipy", "scikit-learn", "lightgbm", "pyarrow")},
    }
    evaluate.save_json(result, config.BACKTEST_JSON)
    report = evaluate.render_backtest_md(result)
    config.BACKTEST_MD.write_text(report)
    print(report)
    print(f"wrote {config.BACKTEST_JSON}, {config.BACKTEST_MD} and raw selection/audit predictions")


def _stage_choice() -> str:
    if config.BACKTEST_JSON.exists():
        report = json.loads(config.BACKTEST_JSON.read_text())
        selection = report.get("selection", {})
        # Legacy holdout-selected reports cannot authorize boosting.
        if (report.get("protocol_version") == evaluate.PROTOCOL_VERSION
                and selection.get("selection_seasons") == evaluate.SELECTION_SEASONS
                and report.get("audit_used_for_selection") is False):
            choice = evaluate.decide_stage(selection.get("combined", {})) if "stage1" in selection.get("combined", {}) else "stage1"
            if choice == report.get("stage_choice") == selection.get("stage_choice"):
                return choice
        log.warning("backtest lacks valid frozen 2022 selection provenance; defaulting to stage1")
    else:
        log.warning("no backtest.json found; defaulting to stage1. Run `gridiron backtest`.")
    return "stage1"


def _train_and_pick(feats: pl.DataFrame, season: int, week: int, now: dt.datetime, stage: str | None = None,
                    *, simulated: bool = False):
    if stage == "stage2" and not simulated and _stage_choice() != "stage2":
        raise ValueError("stage2 cannot issue real picks unless the frozen 2022 selection gate chose it")
    stage = stage or _stage_choice()
    frame = features.modeling_frame(feats)
    train = frame.filter((pl.col("season") < season) | ((pl.col("season") == season) & (pl.col("week") < week)))
    if "history_available_at" in train.columns:
        train = train.filter(pl.col("history_available_at") <= now)
    model = models.fit(train, stage)
    issued_at = now if simulated else dt.datetime.now(dt.timezone.utc)
    target = feats.filter((pl.col("season") == season) & (pl.col("week") == week)
                          & pl.col("result").is_null() & (pl.col("kickoff_utc") > issued_at))
    target = target.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in model.features]))
    if target.height == 0:
        return stage, model, P.empty_picks()
    p = model.predict_proba(target)
    return stage, model, P.make_picks(target, p, season, week, stage, model.n_train, now=issued_at)


def cmd_picks(args):
    _guard_execution(args, command="picks")
    now = _now(args)
    sched, pbp = _load(refresh=args.refresh, now=now)
    season, week = _target_week(sched, args, now)
    feats = _features(sched, pbp, prediction_time=now)
    stage, model, new = _train_and_pick(feats, season, week, now, args.stage, simulated=args.dry_run)
    added = P.append_picks(new) if not args.dry_run else 0
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
    _guard_execution(args, command="grade")
    now = _now(args)
    sched = data.load_schedules(refresh=args.refresh, offline=not args.refresh)
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
    _guard_execution(args, command="site")
    now = _now(args)
    sched = data.load_schedules(refresh=False, offline=True)
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
    _guard_execution(args, command="recap")
    now = _now(args)
    sched = data.load_schedules(refresh=False, offline=True)
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


def _guard_execution(args, *, command: str | None = None) -> None:
    """Reject synthetic clocks before any canonical output or publication changes."""
    command = command or args.cmd
    dry_run = getattr(args, "dry_run", False)
    if getattr(args, "now", None):
        safe = command in {"fetch", "validate"} or (command == "picks" and dry_run)
        safe |= command == "weekly" and dry_run and bool(getattr(args, "output_dir", None))
        if not safe:
            raise ValueError("--now cannot write real histories or reports; use picks --dry-run or weekly --dry-run --output-dir NEW_DIRECTORY")
    if command == "weekly":
        publishing = getattr(args, "publish", False)
        if dry_run != bool(getattr(args, "output_dir", None)):
            raise ValueError("weekly --dry-run requires --output-dir, and --output-dir requires --dry-run")
        if dry_run and (publishing or getattr(args, "commit", False)):
            raise ValueError("a dry run cannot commit, push, or deploy")
        if publishing and not getattr(args, "publish_branch", None):
            raise ValueError("--publish requires an explicit --publish-branch")
        if publishing and getattr(args, "no_commit", False):
            raise ValueError("--publish cannot be combined with --no-commit")


@contextmanager
def _isolated_outputs(directory: str, now: dt.datetime):
    """Copy proof files into a fresh directory; redirect every generated artifact."""
    output = Path(directory).resolve()
    protected = [config.PICKS_DIR.resolve(), config.RECAPS_DIR.resolve(), config.SITE_DIR.resolve(), config.REPORTS_DIR.resolve()]
    if any(output == p or output in p.parents or p in output.parents for p in protected):
        raise ValueError("dry-run output must be separate from canonical picks, recaps, reports, and site directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("dry-run output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    names = ["PICKS_DIR", "PICKS_FILE", "GRADED_FILE", "RECORD_FILE", "RECAPS_DIR", "SITE_DIR", "REPORTS_DIR", "BACKTEST_JSON", "BACKTEST_MD"]
    original = {name: getattr(config, name) for name in names}
    try:
        config.PICKS_DIR = output / "data" / "picks"
        config.PICKS_FILE = config.PICKS_DIR / "picks.csv"
        config.GRADED_FILE = config.PICKS_DIR / "graded.csv"
        config.RECORD_FILE = config.PICKS_DIR / "record.json"
        config.RECAPS_DIR = output / "recaps"
        config.SITE_DIR = output / "site"
        config.REPORTS_DIR = output / "reports"
        config.BACKTEST_JSON = config.REPORTS_DIR / "backtest.json"
        config.BACKTEST_MD = config.REPORTS_DIR / "backtest.md"
        if original["PICKS_DIR"].exists():
            shutil.copytree(original["PICKS_DIR"], config.PICKS_DIR)
        if original["REPORTS_DIR"].exists():
            shutil.copytree(original["REPORTS_DIR"], config.REPORTS_DIR)
        (output / "DRY_RUN.json").write_text(json.dumps({"dry_run": True, "simulation_time": now.isoformat(), "publishable": False}, indent=2) + "\n")
        yield
    finally:
        for name, value in original.items():
            setattr(config, name, value)


def cmd_weekly(args):
    """Fetch through recap/site; explicitly requested publication commits, pushes, deploys."""
    _guard_execution(args, command="weekly")
    now = _now(args)
    if getattr(args, "dry_run", False):
        with _isolated_outputs(args.output_dir, now):
            _weekly_outputs(args, now)
        return
    if getattr(args, "publish", False):
        _publication_preflight(args)
    elif getattr(args, "commit", False):
        _assert_clean_index()
    _weekly_outputs(args, now)


def _weekly_outputs(args, now):
    np.random.seed(config.SEED)
    sched, pbp = _load(refresh=not args.offline, now=now, offline=args.offline)
    season, week = _target_week(sched, args, now)
    log.info("target: %s week %s", season, week)

    feats = _features(sched, pbp, prediction_time=now)
    stage, model, new = _train_and_pick(feats, season, week, now, simulated=args.dry_run)
    added = P.append_picks(new, now=now) if args.dry_run else P.append_picks(new)
    log.info("%s trained on %d games; %d picks, %d newly logged", stage, model.n_train, new.height, added)

    graded, rec = _grade(sched)
    picks = P.load_picks()
    recap_path = _write_recap(graded, picks, rec, season, week)
    backtest = json.loads(config.BACKTEST_JSON.read_text()) if config.BACKTEST_JSON.exists() else None
    this_week = picks.filter((pl.col("season").cast(pl.Int64) == season) & (pl.col("week").cast(pl.Int64) == week))
    site.build_site(season, week, this_week, graded, rec, backtest, now=now)

    print(recap_path.read_text())
    print(f"record: model {rec['overall']['model']}")
    if args.commit or args.publish:
        _git_commit(f"Week {week} picks, {season}: {new.height} games, {added} new; record updated")
    if args.publish:
        _publish(args)


def _git(*arguments: str, check: bool = True):
    return subprocess.run([GIT, *arguments], check=check, capture_output=True, text=True, cwd=config.ROOT)


def _assert_clean_index() -> None:
    if _git("diff", "--cached", "--name-only").stdout.strip():
        raise RuntimeError("refusing to commit with a nonempty Git index; commit or unstage existing work first")


def _publication_preflight(args) -> None:
    _assert_clean_index()
    if _git("status", "--porcelain").stdout.strip():
        raise RuntimeError("publication requires a clean reviewed checkout before generating weekly artifacts")
    branch = _git("branch", "--show-current").stdout.strip()
    if branch != args.publish_branch:
        raise RuntimeError(f"publication branch mismatch: checked out {branch!r}, requested {args.publish_branch!r}")
    upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}").stdout.strip()
    if upstream != f"{args.remote}/{branch}":
        raise RuntimeError("publication upstream does not match the requested remote/branch")
    remote = _git("remote", "get-url", args.remote).stdout.strip().removesuffix(".git")
    if remote not in {"https://github.com/DamianLujan1/coolshit", "git@github.com:DamianLujan1/coolshit"}:
        raise RuntimeError("publication remote must be DamianLujan1/coolshit")
    # These IDs identify the existing project. Never let deploy auto-create/link a project.
    link = config.ROOT / ".vercel" / "project.json"
    if not link.exists():
        raise RuntimeError("existing Vercel project is not linked locally; complete approved setup before publication")
    context = json.loads(link.read_text())
    if context.get("projectId") != VERCEL_PROJECT_ID or context.get("orgId") != VERCEL_ORG_ID:
        raise RuntimeError("Vercel link does not target the verified gridiron-model project")
    inspected = _vercel("project", "inspect")
    if VERCEL_PROJECT_ID not in inspected.stdout + inspected.stderr:
        raise RuntimeError("Vercel project inspection did not confirm the expected project ID")


def _vercel(*arguments: str):
    # Exact per-process IDs override ambient CLI targeting without changing any
    # user's credentials or persistent project/access settings.
    env = {**os.environ, "VERCEL_PROJECT_ID": VERCEL_PROJECT_ID, "VERCEL_ORG_ID": VERCEL_ORG_ID}
    return subprocess.run(["vercel", *arguments, "--scope", VERCEL_SCOPE, "--non-interactive", "--no-color"], check=True,
                          capture_output=True, text=True, cwd=config.ROOT, env=env)


def _git_commit(message: str) -> None:
    _assert_clean_index()
    paths = [str(p) for p in (config.PICKS_DIR, config.RECAPS_DIR, config.SITE_DIR, config.REPORTS_DIR)]
    _git("add", *paths)
    status = _git("diff", "--cached", "--name-only").stdout
    if not status.strip():
        log.info("nothing new to commit")
        return
    _git("commit", "-q", "-m", message)
    log.info("committed: %s", message)


def _publish(args) -> None:
    """Fail closed: push the recorded commit before deployment, verify before promote."""
    if _git("status", "--porcelain").stdout.strip():
        raise RuntimeError("checkout changed during publication; refusing to push or deploy")
    sha = _git("rev-parse", "HEAD").stdout.strip()
    _git("push", args.remote, f"HEAD:refs/heads/{args.publish_branch}")
    deployed = _vercel("deploy", "--prod", "--skip-domain", "--yes", "--meta", f"sourceCommit={sha}")
    url = deployed.stdout.strip()
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc.endswith(".vercel.app") or any(c.isspace() for c in url):
        raise RuntimeError("Vercel did not return a single deployment URL; inspect before promoting")
    inspected = _vercel("inspect", url, "--wait", "--timeout", "60s")
    state = inspected.stdout + "\n" + inspected.stderr
    if not re.search(r"^\s*(?:status|state)\s+(?:●\s*)?ready\s*$", state, re.IGNORECASE | re.MULTILINE):
        raise RuntimeError("deployment READY status was not confirmed; production alias not moved")
    # Check the uploaded proof and methodology through the authenticated CLI
    # before assigning production. No access/security settings are changed.
    for route, expected in (("/picks.csv", config.PICKS_FILE.read_text()),
                            ("/methodology", "nflverse")):
        response = _vercel("curl", route, "--deployment", url)
        valid = response.stdout == expected if route.endswith(".csv") else expected in response.stdout
        if not valid:
            raise RuntimeError(f"deployment verification failed for {route}; production alias not moved")
    _vercel("promote", url, "--yes")
    production = _vercel("curl", "/picks.csv", "--deployment", "https://gridiron-model.vercel.app")
    if production.stdout != config.PICKS_FILE.read_text():
        raise RuntimeError("promotion returned but production pick proof does not match; inspect the alias before claiming publication")
    print(f"published Git commit {sha}: {url}")


# --------------------------------------------------------------------------- #

def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(prog="gridiron", description="NFL game probabilities with a public record")
    ap.add_argument("--now", help="synthetic ISO clock; mutation commands require an isolated dry run")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch", help="download the latest data and validate it").set_defaults(fn=cmd_fetch)
    sub.add_parser("validate", help="validate cached data without downloading").set_defaults(fn=cmd_validate)

    b = sub.add_parser("backtest", help="2022 model selection, then frozen 2023-2025 walk-forward audit")
    backtest_data = b.add_mutually_exclusive_group()
    backtest_data.add_argument("--refresh", action="store_true")
    backtest_data.add_argument("--offline", action="store_true", help="use cached data only (default without --refresh)")
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

    w = sub.add_parser("weekly", help="fetch, train, log, grade, record, recap, and build site")
    w.add_argument("--offline", action="store_true", help="use cached data only")
    commits = w.add_mutually_exclusive_group()
    commits.add_argument("--commit", action="store_true", help="also commit generated artifacts locally")
    commits.add_argument("--no-commit", action="store_true", help="keep generated artifacts uncommitted (default)")
    w.add_argument("--publish", action="store_true", help="explicitly commit, push, verify deployment, and promote production")
    w.add_argument("--publish-branch", help="required with --publish; must match the clean checked-out branch/upstream")
    w.add_argument("--remote", default="origin", help="Git remote for explicit publication (default: origin)")
    w.add_argument("--dry-run", action="store_true", help="run the entire pipeline in an isolated output directory")
    w.add_argument("--output-dir", help="new or empty isolated directory required by --dry-run")
    w.add_argument("--season", type=int)
    w.add_argument("--week", type=int)
    w.set_defaults(fn=cmd_weekly)

    args = ap.parse_args(argv)
    try:
        _guard_execution(args)
        _now(args)
    except ValueError as exc:
        ap.error(str(exc))
    args.fn(args)


if __name__ == "__main__":
    main()
