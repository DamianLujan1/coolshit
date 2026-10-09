"""Workflow safety: all filesystem writes use temporary fixture directories."""
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

import polars as pl
import pytest

from gridiron import cli, config, picks as P


@pytest.mark.parametrize("command", [
    ["picks"], ["grade"], ["site"], ["recap"], ["backtest"], ["weekly"],
    ["weekly", "--no-commit"], ["weekly", "--publish", "--publish-branch", "main"],
])
def test_synthetic_clock_rejected_before_any_command(monkeypatch, command):
    def unexpected(*a, **kw):
        pytest.fail("synthetic command reached a data load")
    monkeypatch.setattr(cli, "_load", unexpected)
    monkeypatch.setattr(cli.data, "load_schedules", unexpected)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--now", "2026-10-06T12:00:00+00:00", *command])
    assert exc.value.code == 2


@pytest.mark.parametrize("arguments", [
    ["weekly", "--dry-run"], ["weekly", "--output-dir", "/tmp/not-used"],
    ["weekly", "--dry-run", "--output-dir", "/tmp/not-used", "--commit"],
    ["weekly", "--dry-run", "--output-dir", "/tmp/not-used", "--publish", "--publish-branch", "main"],
    ["weekly", "--publish"],
    ["--now", "2026-10-06T12:00:00", "picks", "--dry-run"],
])
def test_unsafe_or_ambiguous_modes_rejected(arguments):
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments)
    assert exc.value.code == 2


@pytest.fixture
def output_tree(tmp_path, monkeypatch):
    root = tmp_path / "real"
    locations = {
        "ROOT": root, "PICKS_DIR": root / "data/picks", "PICKS_FILE": root / "data/picks/picks.csv",
        "GRADED_FILE": root / "data/picks/graded.csv", "RECORD_FILE": root / "data/picks/record.json",
        "RECAPS_DIR": root / "recaps", "SITE_DIR": root / "site", "REPORTS_DIR": root / "reports",
        "BACKTEST_JSON": root / "reports/backtest.json", "BACKTEST_MD": root / "reports/backtest.md",
    }
    for name, location in locations.items():
        monkeypatch.setattr(config, name, location)
    for name in ("PICKS_DIR", "RECAPS_DIR", "SITE_DIR", "REPORTS_DIR"):
        locations[name].mkdir(parents=True)
    P.empty_picks().write_csv(config.PICKS_FILE)
    config.RECORD_FILE.write_text('{"original": true}\n')
    (config.SITE_DIR / "picks.csv").write_bytes(config.PICKS_FILE.read_bytes())
    (config.RECAPS_DIR / "original.md").write_text("original recap\n")
    return root


def test_complete_synthetic_weekly_workflow_is_isolated(output_tree, tmp_path, monkeypatch, schedule, pbp):
    before = {p: p.read_bytes() for p in output_tree.rglob("*") if p.is_file()}
    target_dir = tmp_path / "simulation"
    events = []

    def load(refresh, now, *, offline):
        assert offline and not refresh
        events.append("fetch_validate")
        return schedule, pbp

    def train(feats, season, week, now, *, simulated):
        assert simulated
        events.append("train_pick")
        target = feats.filter((pl.col("season") == season) & (pl.col("week") == week))
        return "stage1", SimpleNamespace(n_train=10), P.make_picks(target, [.6] * target.height, season, week, "stage1", 10, now=now)

    monkeypatch.setattr(cli, "_load", load)
    monkeypatch.setattr(cli, "_features", lambda sched, pbp, **kwargs: sched.with_columns(
        pl.lit("lagged_completed_game").alias("home_qb_source"),
        pl.lit("lagged_completed_game").alias("away_qb_source")))
    monkeypatch.setattr(cli, "_train_and_pick", train)
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: pytest.fail("dry run invoked a subprocess"))
    cli.main(["--now", "2015-09-26T12:00:00+00:00", "weekly", "--offline", "--dry-run",
              "--output-dir", str(target_dir), "--season", "2015", "--week", "4"])
    assert events == ["fetch_validate", "train_pick"]
    assert {p: p.read_bytes() for p in output_tree.rglob("*") if p.is_file()} == before
    assert config.ROOT == output_tree
    assert config.PICKS_FILE == output_tree / "data/picks/picks.csv"
    assert P.load_picks(target_dir / "data/picks/picks.csv").height == 2
    for artifact in ("DRY_RUN.json", "data/picks/qb_inputs.jsonl", "data/picks/graded.csv", "data/picks/record.json",
                     "recaps/2015_week_04.md", "site/index.html", "site/record.html", "site/methodology.html", "site/picks.csv"):
        assert (target_dir / artifact).exists(), artifact
    assert json.loads((target_dir / "DRY_RUN.json").read_text())["publishable"] is False
    with pytest.raises(ValueError, match="new or empty"):
        with cli._isolated_outputs(str(target_dir), dt.datetime.now(dt.timezone.utc)):
            pass


def test_dry_run_cannot_target_canonical_path_or_ancestor(output_tree):
    for directory in (config.ROOT, config.PICKS_DIR, config.PICKS_DIR / "subdir", config.SITE_DIR):
        with pytest.raises(ValueError, match="separate"):
            with cli._isolated_outputs(str(directory), dt.datetime.now(dt.timezone.utc)):
                pass


def test_nonempty_index_refuses_commit_before_staging(monkeypatch):
    commands = []
    def run(cmd, **kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "unrelated.txt\n", "")
    monkeypatch.setattr(cli.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="nonempty Git index"):
        cli._git_commit("weekly output")
    assert commands == [[cli.GIT, "diff", "--cached", "--name-only"]]


def test_live_boosting_override_cannot_bypass_selection(monkeypatch):
    monkeypatch.setattr(cli, "_stage_choice", lambda: "stage1")
    with pytest.raises(ValueError, match="2022 selection"):
        cli._train_and_pick(pl.DataFrame(), 2026, 6, dt.datetime.now(dt.timezone.utc), stage="stage2")


@pytest.fixture
def publication_stub(output_tree, monkeypatch):
    link = output_tree / ".vercel/project.json"
    link.parent.mkdir()
    link.write_text(json.dumps({"projectId": cli.VERCEL_PROJECT_ID, "orgId": cli.VERCEL_ORG_ID}))
    calls = []
    state = {"staged": False, "fail": None, "wrong_proof": False}
    def run(cmd, **kwargs):
        calls.append(cmd)
        command = cmd[1:]
        if state["fail"] and command[0] == state["fail"]:
            raise subprocess.CalledProcessError(1, cmd)
        text = ""
        if cmd[0] == cli.GIT:
            if command == ["diff", "--cached", "--name-only"]:
                text = "site/index.html\n" if state["staged"] else ""
            elif command == ["branch", "--show-current"]:
                text = "main\n"
            elif command == ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"]:
                text = "origin/main\n"
            elif command == ["remote", "get-url", "origin"]:
                text = "https://github.com/DamianLujan1/coolshit.git\n"
            elif command == ["rev-parse", "HEAD"]:
                text = "a" * 40 + "\n"
            elif command[0] == "add":
                state["staged"] = True
            elif command[0] == "commit":
                state["staged"] = False
        else:
            assert kwargs["env"]["VERCEL_PROJECT_ID"] == cli.VERCEL_PROJECT_ID
            assert kwargs["env"]["VERCEL_ORG_ID"] == cli.VERCEL_ORG_ID
            if command[:2] == ["project", "inspect"]:
                text = cli.VERCEL_PROJECT_ID
            elif command[0] == "deploy":
                assert "--skip-domain" in command
                assert "sourceCommit=" + "a" * 40 in command
                text = "https://verified-deployment.vercel.app\n"
            elif command[0] == "inspect":
                text = "status Ready\n"
            elif command[:2] == ["curl", "/picks.csv"]:
                text = "wrong" if state["wrong_proof"] else config.PICKS_FILE.read_text()
            elif command[:2] == ["curl", "/methodology"]:
                text = "nflverse attribution"
        return subprocess.CompletedProcess(cmd, 0, text, "")
    monkeypatch.setattr(cli.subprocess, "run", run)
    return calls, state


def test_publication_subprocess_order_and_target(publication_stub):
    calls, state = publication_stub
    args = SimpleNamespace(publish_branch="main", remote="origin")
    cli._publication_preflight(args)
    cli._git_commit("weekly artifacts")
    cli._publish(args)
    operations = [cmd[1] for cmd in calls]
    assert operations.index("commit") < operations.index("push") < operations.index("deploy") < operations.index("curl") < operations.index("promote")
    assert [cli.GIT, "push", "origin", "HEAD:refs/heads/main"] in calls


@pytest.mark.parametrize("failure", ["push", "deploy", "inspect", "curl"])
def test_publication_stops_after_failed_step(publication_stub, failure):
    calls, state = publication_stub
    state["fail"] = failure
    with pytest.raises(subprocess.CalledProcessError):
        cli._publish(SimpleNamespace(publish_branch="main", remote="origin"))
    operations = [cmd[1] for cmd in calls]
    assert "promote" not in operations
    if failure == "push":
        assert "deploy" not in operations


def test_wrong_uploaded_proof_never_promotes(publication_stub):
    calls, state = publication_stub
    state["wrong_proof"] = True
    with pytest.raises(RuntimeError, match="verification failed"):
        cli._publish(SimpleNamespace(publish_branch="main", remote="origin"))
    assert "promote" not in [cmd[1] for cmd in calls]


def test_unlinked_project_blocks_before_mutation(publication_stub):
    calls, state = publication_stub
    (config.ROOT / ".vercel/project.json").unlink()
    with pytest.raises(RuntimeError, match="not linked"):
        cli._publication_preflight(SimpleNamespace(publish_branch="main", remote="origin"))
    assert not any(cmd[1] in {"add", "commit", "push", "deploy", "promote"} for cmd in calls)
