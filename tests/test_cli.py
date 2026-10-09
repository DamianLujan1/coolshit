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
    """No live HTTP or CLI: verify orchestration against exact fixture bytes."""
    link = output_tree / ".vercel/project.json"
    link.parent.mkdir()
    link.write_text(json.dumps({"projectId": cli.VERCEL_PROJECT_ID, "orgId": cli.VERCEL_ORG_ID}))
    (config.SITE_DIR / "methodology.html").write_bytes(b"<html>nflverse methodology fixture</html>\n")
    monkeypatch.setenv("VERCEL_TOKEN", "unit-test-token-not-real")
    calls, events, requests = [], [], []
    state = {"staged": False, "fail": None, "mismatch": None, "protected": False}

    def run(cmd, **kwargs):
        calls.append(cmd)
        command = cmd[1:]
        events.append(command[0])
        if state["fail"] == command[0]:
            raise subprocess.CalledProcessError(1, cmd, output="fixture failure", stderr="fixture failure")
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
            assert command[0] in {"--version", "project", "whoami", "deploy", "inspect", "promote"}
            assert kwargs["env"]["VERCEL_PROJECT_ID"] == cli.VERCEL_PROJECT_ID
            assert kwargs["env"]["VERCEL_ORG_ID"] == cli.VERCEL_ORG_ID
            if command[0] == "--version":
                text = "63.1.0\n"
            elif command[:2] == ["project", "inspect"]:
                text = cli.VERCEL_PROJECT_ID
            elif command[0] == "deploy":
                assert "--skip-domain" in command
                assert "sourceCommit=" + "a" * 40 in command
                assert command[command.index("--project") + 1] == cli.VERCEL_PROJECT_ID
                text = "https://verified-deployment.vercel.app\n"
            elif command[0] == "inspect":
                text = "status Ready\n"
        return subprocess.CompletedProcess(cmd, 0, text, "")

    def public_get(url):
        requests.append(url)
        production = url.startswith("https://gridiron-model.vercel.app/")
        events.append("public_production" if production else "public_preview")
        if state["protected"] and not production:
            raise cli.ProtectedPreviewError("fixture protected preview")
        if state["mismatch"] == "production" and production:
            return b"stale production"
        if url.endswith("/picks.csv"):
            return b"wrong proof" if state["mismatch"] == "picks" else config.PICKS_FILE.read_bytes()
        if url.endswith("/methodology"):
            return b"wrong methodology" if state["mismatch"] == "methodology" else (config.SITE_DIR / "methodology.html").read_bytes()
        pytest.fail("unexpected public verification route")

    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(cli, "_public_get", public_get)
    return calls, state, events, requests


def test_publication_order_target_and_public_verification(publication_stub):
    calls, state, events, requests = publication_stub
    args = SimpleNamespace(publish_branch="main", remote="origin")
    cli._publication_preflight(args)
    cli._git_commit("weekly artifacts")
    cli._publish(args)
    assert events.index("commit") < events.index("push") < events.index("deploy") < events.index("public_preview") < events.index("promote") < events.index("public_production")
    assert [cli.GIT, "push", "origin", "HEAD:refs/heads/main"] in calls
    assert len(requests) == 4
    report = json.loads(cli._publication_verification_path().read_text())
    assert report["status"] == "public_production_verified"
    assert report["preview_http_verified"] and report["production_http_verified"]
    assert report["credential_or_protection_changes"] is False


@pytest.mark.parametrize("failure", ["push", "deploy", "inspect"])
def test_publication_stops_after_failed_step(publication_stub, failure):
    calls, state, events, requests = publication_stub
    state["fail"] = failure
    with pytest.raises((subprocess.CalledProcessError, RuntimeError)):
        cli._publish(SimpleNamespace(publish_branch="main", remote="origin"))
    assert "promote" not in events
    if failure == "push":
        assert "deploy" not in events


@pytest.mark.parametrize("mismatch", ["picks", "methodology"])
def test_mismatched_public_content_never_promotes(publication_stub, mismatch):
    calls, state, events, requests = publication_stub
    state["mismatch"] = mismatch
    with pytest.raises(RuntimeError):
        cli._publish(SimpleNamespace(publish_branch="main", remote="origin"))
    assert "promote" not in events
    assert json.loads(cli._publication_verification_path().read_text())["status"] == "preview_verification_failed"


def test_protected_preview_records_unverified_without_auth_fallback(publication_stub):
    calls, state, events, requests = publication_stub
    state["protected"] = True
    with pytest.raises(RuntimeError):
        cli._publish(SimpleNamespace(publish_branch="main", remote="origin"))
    assert "promote" not in events and "public_production" not in events
    assert not any(cmd[1] in {"curl", "security", "login", "logout", "token", "tokens", "api"} for cmd in calls)
    report = json.loads(cli._publication_verification_path().read_text())
    assert report["status"] == "protected_unverified"
    assert not report["preview_http_verified"] and not report["production_http_verified"]
    assert report["credential_or_protection_changes"] is False


def test_stale_production_never_claims_verified_publication(publication_stub, capsys):
    calls, state, events, requests = publication_stub
    state["mismatch"] = "production"
    with pytest.raises(RuntimeError):
        cli._publish(SimpleNamespace(publish_branch="main", remote="origin"))
    assert "promote" in events
    assert "published Git commit" not in capsys.readouterr().out
    assert json.loads(cli._publication_verification_path().read_text())["status"] == "production_unverified"


def test_public_verification_is_byte_exact_including_line_endings(publication_stub, monkeypatch):
    expected = config.PICKS_FILE.read_bytes()
    assert b"\n" in expected
    monkeypatch.setattr(cli, "_public_get", lambda url: expected.replace(b"\n", b"\r\n"))
    with pytest.raises(RuntimeError, match="verification failed"):
        cli._verify_public_deployment("https://verified-deployment.vercel.app")


def test_unlinked_project_blocks_before_mutation(publication_stub):
    calls, state, events, requests = publication_stub
    (config.ROOT / ".vercel/project.json").unlink()
    with pytest.raises(RuntimeError, match="not linked"):
        cli._publication_preflight(SimpleNamespace(publish_branch="main", remote="origin"))
    assert not any(cmd[1] in {"add", "commit", "push", "deploy", "promote"} for cmd in calls)


@pytest.mark.parametrize("arguments", [
    (), ("curl", "/picks.csv"), ("security", "disable"), ("login",), ("logout",),
    ("token", "create"), ("tokens", "create"), ("refresh",), ("api", "/v9/projects"),
    ("project", "update"), ("teams", "switch", "other"), ("link",),
    ("deploy", "--debug"), ("deploy", "--token", "override-token"),
    ("deploy", "--global-config", "/tmp/stored-auth"), ("deploy", "--scope", "other-scope"),
])
def test_vercel_rejects_unsafe_commands_before_subprocess(monkeypatch, arguments):
    monkeypatch.setenv("VERCEL_TOKEN", "unit-test-token-not-real")
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: pytest.fail("unsafe CLI command ran"))
    with pytest.raises(RuntimeError):
        cli._vercel(*arguments)


def test_missing_token_never_uses_stored_auth_or_runs_cli(monkeypatch):
    monkeypatch.delenv("VERCEL_TOKEN", raising=False)
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **kw: pytest.fail("missing-token CLI ran"))
    with pytest.raises(RuntimeError):
        cli._vercel("project", "inspect")


def test_cli_auth_is_isolated_without_debug_or_proxy_environment(monkeypatch):
    token = "unit-test-token-not-real"
    monkeypatch.setenv("VERCEL_TOKEN", token)
    dangerous = ("VERCEL_DEBUG", "DEBUG", "NODE_DEBUG", "NODE_OPTIONS", "VERCEL_AUTOMATION_BYPASS_SECRET", "HTTPS_PROXY", "ALL_PROXY")
    for key in dangerous:
        monkeypatch.setenv(key, "unsafe-ambient-fixture")
    paths = []
    def run(cmd, **kwargs):
        assert "--non-interactive" in cmd and "--debug" not in cmd
        assert token not in " ".join(cmd)
        folder = Path(cmd[cmd.index("--global-config") + 1])
        paths.append(folder)
        assert folder.is_dir() and not list(folder.rglob("auth.json"))
        env = kwargs["env"]
        assert env["VERCEL_TOKEN"] == token
        assert env["CI"] == env["VERCEL_TELEMETRY_DISABLED"] == env["NO_UPDATE_NOTIFIER"] == "1"
        assert all(env.get(key) != "unsafe-ambient-fixture" for key in dangerous)
        assert kwargs.get("stdin") == subprocess.DEVNULL
        text = "63.1.0\n" if cmd[1] == "--version" else "fixture-user\n"
        return subprocess.CompletedProcess(cmd, 0, text, "")
    monkeypatch.setattr(cli.subprocess, "run", run)
    cli._vercel("whoami", "--format", "json")
    assert paths and all(not path.exists() for path in paths)


def test_unreviewed_cli_version_cannot_run_authenticated_commands(monkeypatch):
    monkeypatch.setenv("VERCEL_TOKEN", "unit-test-token-not-real")
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "99.0.0\n", "")
    monkeypatch.setattr(cli.subprocess, "run", run)
    with pytest.raises(RuntimeError):
        cli._vercel("project", "inspect")
    assert len(calls) == 1 and calls[0][1] == "--version"


def test_cli_failure_and_traceback_never_expose_credentials(monkeypatch, capsys):
    import traceback
    token = "unit-test-token-not-real"
    monkeypatch.setenv("VERCEL_TOKEN", token)
    def run(cmd, **kwargs):
        if cmd[1] == "--version":
            return subprocess.CompletedProcess(cmd, 0, "63.1.0\n", "")
        raise subprocess.CalledProcessError(1, cmd, output=f"token={token}", stderr=f"Authorization: Bearer {token}")
    monkeypatch.setattr(cli.subprocess, "run", run)
    with pytest.raises(RuntimeError) as exc:
        cli._vercel("whoami", "--format", "json")
    assert token not in "".join(traceback.format_exception(exc.value))
    assert token not in str(exc.value)
    captured = capsys.readouterr()
    assert token not in captured.out + captured.err


def test_successful_cli_output_is_redacted_before_return(monkeypatch):
    token = "unit-test-token-not-real"
    monkeypatch.setenv("VERCEL_TOKEN", token)
    def run(cmd, **kwargs):
        if cmd[1] == "--version":
            return subprocess.CompletedProcess(cmd, 0, "63.1.0\n", "")
        return subprocess.CompletedProcess(cmd, 0, f"fixture-user {token}", f"fixture-warning {token}")
    monkeypatch.setattr(cli.subprocess, "run", run)
    result = cli._vercel("whoami", "--format", "json")
    assert token not in result.stdout + result.stderr


def test_public_get_has_no_proxy_auth_cookie_or_bypass_context(monkeypatch):
    import urllib.request
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("VERCEL_TOKEN", "unit-test-token-not-real")
    monkeypatch.setenv("VERCEL_AUTOMATION_BYPASS_SECRET", "fixture-bypass-secret")
    seen = {}
    def build_opener(*handlers):
        seen["handlers"] = handlers
        class Response:
            def read(self, *args):
                return b"public bytes"
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
        class Opener:
            def open(self, request, *args, **kwargs):
                seen["request"] = request
                return Response()
        return Opener()
    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    assert cli._public_get("https://verified-deployment.vercel.app/picks.csv") == b"public bytes"
    handlers = seen["handlers"]
    assert any(isinstance(h, urllib.request.ProxyHandler) and h.proxies == {} for h in handlers)
    assert not any(isinstance(h, (urllib.request.HTTPBasicAuthHandler, urllib.request.HTTPDigestAuthHandler, urllib.request.HTTPCookieProcessor)) for h in handlers)
    redirect = next(h for h in handlers if isinstance(h, urllib.request.HTTPRedirectHandler))
    assert redirect.redirect_request(seen["request"], None, 302, "redirect", {}, "https://vercel.com/login") is None
    request = seen["request"]
    assert request.full_url == "https://verified-deployment.vercel.app/picks.csv"
    headers = {name.lower(): value for name, value in request.header_items()}
    assert not set(headers) & {"authorization", "proxy-authorization", "cookie", "x-vercel-protection-bypass"}
    assert "unit-test-token-not-real" not in str(headers) and "fixture-bypass-secret" not in str(headers)


@pytest.mark.parametrize("code", [401, 403, 301, 302, 303, 307, 308])
def test_protection_or_auth_redirect_is_unverified(monkeypatch, code):
    import urllib.error
    import urllib.request
    def build_opener(*handlers):
        class Opener:
            def open(self, request, *args, **kwargs):
                raise urllib.error.HTTPError(request.full_url, code, "protected fixture", {"Location": "https://vercel.com/login"}, None)
        return Opener()
    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    with pytest.raises(cli.ProtectedPreviewError):
        cli._public_get("https://verified-deployment.vercel.app/picks.csv")


@pytest.mark.parametrize("url", [
    "http://verified-deployment.vercel.app/picks.csv",
    "https://user:secret@verified-deployment.vercel.app/picks.csv",
    "https://verified-deployment.vercel.app/picks.csv?x-vercel-protection-bypass=secret",
])
def test_public_get_rejects_authenticated_or_non_https_urls_without_network(monkeypatch, url):
    import urllib.request
    monkeypatch.setattr(urllib.request, "build_opener", lambda *a, **kw: pytest.fail("unsafe URL attempted network"))
    with pytest.raises((RuntimeError, ValueError)):
        cli._public_get(url)
