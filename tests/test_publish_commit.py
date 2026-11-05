"""Offline tests for scripts/publish_commit.py and the nightly publication steps.

Every test uses temporary repositories: a bare "origin", a shallow clone of it
(as actions/checkout makes) where the nightly publishes, and a second clone
that pushes concurrent changes. Push and fetch failures are injected by
overriding ``Git.push`` / ``Git.fetch``; everything else is real git. Nothing
touches the network or this repository's data_files/.
"""
from __future__ import annotations

import inspect
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from test_output_gate import _bash, _condition, _posix
from test_team_features import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import publish_commit as pc  # noqa: E402

SCRIPT = ROOT / "scripts" / "publish_commit.py"
WORKFLOW = ROOT / ".github" / "workflows" / "nightly-update.yml"
NIGHTLY, BEST_BETS = "Commit and push changes", "Commit best_bets_today.json"
EXPORT = "Export best bets for Sports Picks Grid"
PRED, SCHED = "data_files/nfl_games_historical_with_predictions.csv", "data_files/nfl_games_historical.csv"
MANIFEST, BETS = "data_files/pipeline_run_manifest.json", "data_files/best_bets_today.json"
TRACKER, CAPTURE = "data_files/spread_tracker_log.csv", "data_files/ontario_spreads/captures/new.json"
TOKEN = "ghs_not_a_real_token_1234567890"

SEED = {SCHED: "game,home\n1,KC\n", PRED: "game,prob\n1,0.40\n", MANIFEST: '{"run": 1}\n',
        BETS: '{"bets": "old"}\n', TRACKER: "week,line\n1,-3\n", "data_files/model_metrics.json": "{}\n"}

FAULTS = {
    "500": (1, "remote: Internal Server Error\nTo https://github.com/o/r\n"
               " ! [remote rejected] main -> main (Internal Server Error)\n"
               "error: failed to push some refs to 'https://github.com/o/r'\n"),
    "502": (128, "fatal: unable to access 'https://github.com/o/r/': The requested URL returned error: 502\n"),
    "hangup": (128, "error: RPC failed; HTTP 500 curl 22\nfatal: the remote end hung up unexpectedly\n"),
    "timeout": (124, "git push timed out after 180s"),
    "auth403": (128, "remote: Permission to o/r.git denied to github-actions[bot].\n"
                     f"fatal: unable to access 'https://x-access-token:{TOKEN}@github.com/o/r/': "
                     "The requested URL returned error: 403\n"),
    "authfail": (128, f"fatal: Authentication failed for 'https://{TOKEN}@github.com/o/r/'\n"),
    "protected": (1, "remote: error: GH006: Protected branch update failed for refs/heads/main.\n"
                     " ! [remote rejected] main -> main (protected branch hook declined)\n"),
    "notfound": (128, "remote: Repository not found.\nfatal: repository 'https://github.com/o/r/' not found\n"),
}
TRANSIENT, PERMANENT = ("500", "502", "hangup", "timeout"), ("auth403", "authfail", "protected", "notfound")


# ----------------------------------------------------------------- repos --

def git(cwd, *args, check=True) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    if check and r.returncode:
        raise AssertionError(f"git {' '.join(args)} failed in {cwd}: {r.stderr}")
    return r.stdout.strip()


def write(repo: Path, files: dict) -> None:
    for rel, text in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode())


def show(repo: Path, rev: str, rel: str) -> str | None:
    r = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=repo, capture_output=True, text=True, encoding="utf-8")
    return r.stdout if r.returncode == 0 else None


@pytest.fixture(autouse=True)
def isolated_git(tmp_path, monkeypatch):
    """No user/system git config (credential helpers, autocrlf, signing) and no
    Actions files from the environment."""
    cfg = tmp_path / "gitconfig"
    cfg.write_text("[user]\n\tname = Nightly Bot\n\temail = bot@example.invalid\n[commit]\n\tgpgsign = false\n"
                   "[core]\n\tautocrlf = false\n[init]\n\tdefaultBranch = main\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for var in ("GITHUB_ACTIONS", "GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY", "PUBLISH_ATTEMPTS", "PUBLISH_RETRY_DELAY"):
        monkeypatch.delenv(var, raising=False)


class Repos:
    def __init__(self, tmp: Path):
        self.origin, self.work, self.other = tmp / "origin.git", tmp / "work", tmp / "other"
        git(tmp, "init", "--quiet", "--bare", "-b", "main", str(self.origin))
        seed = tmp / "seed"
        git(tmp, "init", "--quiet", "-b", "main", str(seed))
        write(seed, SEED)
        git(seed, "add", "-A")
        git(seed, "commit", "--quiet", "-m", "seed")
        git(seed, "push", "--quiet", str(self.origin), "main")
        self.base = self.tip()
        # The nightly's checkout is shallow, like actions/checkout's default.
        git(tmp, "clone", "--quiet", "--depth", "1", self.origin.as_uri(), str(self.work))
        git(tmp, "clone", "--quiet", str(self.origin), str(self.other))

    def tip(self) -> str:
        return git(self.origin, "rev-parse", "main")

    def count(self) -> int:
        return int(git(self.origin, "rev-list", "--count", "main"))

    def concurrent(self, files: dict, msg="concurrent update") -> str:
        """Another writer pushes to origin/main (e.g. the Ontario capture)."""
        git(self.other, "pull", "--quiet", "--ff-only")
        write(self.other, files)
        git(self.other, "add", "-A")
        git(self.other, "commit", "--quiet", "-m", msg)
        git(self.other, "push", "--quiet", "origin", "main")
        return self.tip()

    def stage(self, files: dict) -> None:
        write(self.work, files)
        git(self.work, "add", "-A", "data_files")


@pytest.fixture
def repos(tmp_path):
    return Repos(tmp_path)


TONIGHT = {PRED: "game,prob\n1,0.55\n", MANIFEST: '{"run": 2}\n'}


# --------------------------------------------------------- fault-injecting git --

class ScriptedGit(pc.Git):
    """Real git, except each push (and fetch) follows the next planned action:
    "real", a FAULTS key (fail without contacting the remote), "accepted" (the
    server takes the push, the client reports a failure), "lie" (report success
    without pushing), or a callable run first (e.g. a concurrent writer)."""

    def __init__(self, cwd, pushes=(), fetches=()):
        super().__init__(cwd)
        self.pushes, self.fetches, self.push_calls = list(pushes), list(fetches), []

    def push(self, remote, refspec):
        self.push_calls.append(refspec)
        action = self.pushes.pop(0) if self.pushes else "real"
        if callable(action):
            action, _ = "real", action()
        if action in FAULTS:
            code, err = FAULTS[action]
            return subprocess.CompletedProcess(["git", "push"], code, "", err)
        if action == "lie":
            return subprocess.CompletedProcess(["git", "push"], 0, "", "")
        r = super().push(remote, refspec)
        if action == "accepted":
            assert r.returncode == 0, r.stderr
            return subprocess.CompletedProcess(r.args, 128, r.stdout, FAULTS["hangup"][1])
        return r

    def fetch(self, remote, branch):
        action = self.fetches.pop(0) if self.fetches else "real"
        if action != "real":
            code, err = FAULTS[action]
            return subprocess.CompletedProcess(["git", "fetch"], code, "", err.replace("push", "fetch"))
        return super().fetch(remote, branch)


def run(repos, pushes=(), fetches=(), *, guards=(), attempts=4, message="Automated nightly update"):
    g, logs, sleeps = ScriptedGit(repos.work, pushes, fetches), [], []
    try:
        result, sha = pc.publish(g, message=message, guards=guards, attempts=attempts, delay=1.0,
                                 sleep=sleeps.append, log=logs.append)
        err = None
    except pc.PublishError as e:
        result, sha, err = None, None, e
    return dict(result=result, sha=sha, err=err, logs="\n".join(logs), sleeps=sleeps, pushes=g.push_calls)


# ------------------------------------------------------------------ helper --

class TestPublish:
    def test_immediate_success(self, repos):
        repos.stage(TONIGHT)
        r = run(repos)
        assert r["result"] == pc.PUBLISHED and r["err"] is None
        assert r["sha"] == repos.tip() == git(repos.work, "rev-parse", "HEAD")
        assert repos.count() == 2 and len(r["pushes"]) == 1 and r["sleeps"] == []
        assert show(repos.origin, "main", PRED) == TONIGHT[PRED]
        assert "Attempt 1/4" in r["logs"] and "Published:" in r["logs"]

    @pytest.mark.parametrize("fault", TRANSIENT)
    def test_transient_failure_then_success(self, repos, fault):
        repos.stage(TONIGHT)
        r = run(repos, pushes=[fault, "real"])
        assert r["result"] == pc.PUBLISHED and r["sha"] == repos.tip()
        assert len(r["pushes"]) == 2 and r["sleeps"] == [1.0]
        assert repos.count() == 2  # one commit, published once
        assert "Attempt 1/4" in r["logs"] and "Attempt 2/4" in r["logs"] and "push failed" in r["logs"]

    def test_exhausted_retries_fail_and_keep_the_local_commit(self, repos):
        repos.stage(TONIGHT)
        r = run(repos, pushes=["500"] * 3, attempts=3)
        assert r["result"] is None and r["err"].code == pc.EXIT_EXHAUSTED
        head = git(repos.work, "rev-parse", "HEAD")
        assert f"{head} is not on origin/main (checked by fetching it); not published" in str(r["err"])
        assert r["err"].status == pc.NOT_PUBLISHED and r["err"].commit == head and "unconfirmed" not in str(r["err"])
        assert r["sleeps"] == [1.0, 2.0] and len(r["pushes"]) == 3
        assert repos.tip() == repos.base and repos.count() == 1  # remote untouched
        head = git(repos.work, "rev-parse", "HEAD")
        assert head != repos.base and show(repos.work, head, PRED) == TONIGHT[PRED]  # nothing lost locally

    def test_main_advanced_without_conflict_is_replayed_on_top(self, repos):
        repos.stage(TONIGHT)
        theirs = repos.concurrent({CAPTURE: '{"capture": 1}\n', TRACKER: "week,line\n1,-3\n2,-7\n"})
        r = run(repos)
        assert r["result"] == pc.PUBLISHED and r["sha"] == repos.tip()
        local = git(repos.work, "rev-parse", "HEAD")
        assert r["sha"] != local  # rebuilt, not force-pushed
        assert git(repos.origin, "rev-parse", "main^") == theirs and repos.count() == 3
        # Our files exactly as validated; theirs exactly as they pushed them.
        for rel in TONIGHT:
            assert show(repos.origin, "main", rel) == TONIGHT[rel]
        assert show(repos.origin, "main", CAPTURE) == '{"capture": 1}\n'
        assert show(repos.origin, "main", TRACKER) == "week,line\n1,-3\n2,-7\n"
        assert show(repos.origin, "main", SCHED) == SEED[SCHED]
        assert git(repos.origin, "log", "-1", "--format=%s|%an", "main") == "Automated nightly update|Nightly Bot"
        assert "rebuilt as" in r["logs"] and "remote changes kept" in r["logs"]
        assert all(not p.startswith("+") for p in r["pushes"])

    def test_main_advancing_again_during_retries(self, repos):
        repos.stage(TONIGHT)
        repos.concurrent({CAPTURE: "1\n"}, "capture 1")
        second = lambda: repos.concurrent({"data_files/injury_reports/captures/x.json": "2\n"}, "capture 2")
        r = run(repos, pushes=["real", second])
        assert r["result"] == pc.PUBLISHED and r["sha"] == repos.tip() and repos.count() == 4
        assert show(repos.origin, "main", CAPTURE) == "1\n"
        assert show(repos.origin, "main", "data_files/injury_reports/captures/x.json") == "2\n"
        assert show(repos.origin, "main", PRED) == TONIGHT[PRED]

    def test_conflicting_change_fails_without_losing_either_side(self, repos):
        repos.stage(TONIGHT)
        theirs = repos.concurrent({PRED: "game,prob\n1,0.99\n"}, "another run's predictions")
        r = run(repos)
        assert r["err"].code == pc.EXIT_CONFLICT and PRED in str(r["err"])
        assert "neither side was changed" in str(r["err"])
        assert repos.tip() == theirs and show(repos.origin, "main", PRED) == "game,prob\n1,0.99\n"
        head = git(repos.work, "rev-parse", "HEAD")
        assert show(repos.work, head, PRED) == TONIGHT[PRED] and show(repos.work, head, MANIFEST) == TONIGHT[MANIFEST]
        assert len(r["pushes"]) == 1 and r["sleeps"] == []

    @pytest.mark.parametrize("guarded", [True, False])
    def test_remote_change_to_a_guarded_input_blocks(self, repos, guarded):
        """The schedule is unchanged tonight, so it is not in our commit; the gate
        validated the predictions against it, so a remote change must block."""
        repos.stage(TONIGHT)
        theirs = repos.concurrent({SCHED: "game,home\n1,BUF\n"})
        r = run(repos, guards=[SCHED] if guarded else [])
        if guarded:
            assert r["err"].code == pc.EXIT_CONFLICT and SCHED in str(r["err"]) and repos.tip() == theirs
        else:
            assert r["result"] == pc.PUBLISHED and show(repos.origin, "main", SCHED) == "game,home\n1,BUF\n"

    def test_our_deletion_survives_the_rebuild(self, repos):
        repos.stage(TONIGHT)
        git(repos.work, "rm", "--quiet", TRACKER)
        repos.concurrent({CAPTURE: "1\n"})
        r = run(repos)
        assert r["result"] == pc.PUBLISHED and r["sha"] == repos.tip()
        assert show(repos.origin, "main", TRACKER) is None and show(repos.origin, "main", CAPTURE) == "1\n"

    def test_a_faulty_rebuild_is_caught_before_pushing(self, repos):
        """Backstop: if the rebuilt tree did not carry exactly our files (here the
        index update is lost), nothing is pushed."""
        class LosesUpdate(ScriptedGit):
            def out(self, *args, **kw):
                return "" if args[:1] == ("update-index",) else super().out(*args, **kw)
        repos.stage(TONIGHT)
        theirs = repos.concurrent({CAPTURE: "1\n"})
        g = LosesUpdate(repos.work)
        with pytest.raises(pc.PublishError) as e:
            pc.publish(g, message="m", delay=0, sleep=lambda s: None, log=lambda s: None)
        assert e.value.code == pc.EXIT_INTEGRATION and "content checks" in str(e.value)
        assert len(g.push_calls) == 1 and repos.tip() == theirs

    def test_remote_deleting_our_file_conflicts(self, repos):
        repos.stage(TONIGHT)
        git(repos.other, "rm", "--quiet", MANIFEST)
        git(repos.other, "commit", "--quiet", "-m", "remove manifest")
        git(repos.other, "push", "--quiet", "origin", "main")
        r = run(repos)
        assert r["err"].code == pc.EXIT_CONFLICT and MANIFEST in str(r["err"])

    @pytest.mark.parametrize("fault", PERMANENT)
    def test_permanent_errors_stop_at_once(self, repos, fault):
        repos.stage(TONIGHT)
        r = run(repos, pushes=[fault])
        assert r["err"].code == pc.EXIT_PERMANENT and "a retry cannot fix" in str(r["err"])
        assert len(r["pushes"]) == 1 and r["sleeps"] == [] and repos.tip() == repos.base
        assert TOKEN not in str(r["err"]) and TOKEN not in r["logs"]

    def test_real_protected_branch_rejection_is_permanent(self, repos):
        hook = repos.origin / "hooks" / "pre-receive"
        hook.write_bytes(b"#!/bin/sh\necho 'error: GH006: Protected branch update failed for refs/heads/main.' >&2\nexit 1\n")
        hook.chmod(0o755)
        repos.stage(TONIGHT)
        r = run(repos)
        assert r["err"].code == pc.EXIT_PERMANENT and "GH006" in str(r["err"])
        assert len(r["pushes"]) == 1 and repos.tip() == repos.base

    def test_permanent_fetch_error_after_a_rejected_push_is_not_published(self, repos):
        """A non-fast-forward rejection never lands, so the outcome is known."""
        repos.stage(TONIGHT)
        theirs = repos.concurrent({CAPTURE: "1\n"})
        r = run(repos, fetches=["authfail"])
        assert r["err"].code == pc.EXIT_PERMANENT and "fetch failed" in str(r["err"])
        assert "was not published" in str(r["err"]) and r["err"].status == pc.NOT_PUBLISHED
        assert repos.tip() == theirs

    # ------------------------------------------- unconfirmed publication --

    @staticmethod
    def _assert_unconfirmed(r, repos):
        err, head = r["err"], git(repos.work, "rev-parse", "HEAD")
        assert err.code == pc.EXIT_UNCONFIRMED and err.status == pc.UNCONFIRMED and err.commit == head
        text = str(err)
        assert text.splitlines()[0] == "Publication unconfirmed; the commit may already be on origin/main."
        assert f"commit: {head}" in text and "remote: origin" in text and "branch: main" in text
        assert "git ls-remote origin refs/heads/main" in text
        # Never claims non-publication (or publication).
        assert "not published" not in text.lower() and "not on origin" not in text.lower()
        assert "Published:" not in r["logs"]
        return head

    def test_accepted_push_then_confirmation_fetches_exhausted(self, repos):
        repos.stage(TONIGHT)
        r = run(repos, pushes=["accepted", "real", "real"], fetches=["502"] * 3, attempts=3)
        head = self._assert_unconfirmed(r, repos)
        assert repos.tip() == head and repos.count() == 2  # really published, exactly once
        assert len(r["pushes"]) == 3 and set(r["pushes"]) == {f"{head}:refs/heads/main"}
        assert r["sleeps"] == [1.0, 2.0]

    def test_push_reported_success_then_confirmation_fetches_exhausted(self, repos):
        repos.stage(TONIGHT)
        r = run(repos, fetches=["hangup"] * 2, attempts=2)
        head = self._assert_unconfirmed(r, repos)
        assert repos.tip() == head and repos.count() == 2 and len(r["pushes"]) == 2

    def test_unlanded_push_with_failed_confirmation_is_still_unconfirmed(self, repos):
        """Reported success but nothing landed: without a read-back the helper
        cannot know, so it says "may", not "is"."""
        repos.stage(TONIGHT)
        r = run(repos, pushes=["lie", "lie"], fetches=["502"] * 2, attempts=2)
        self._assert_unconfirmed(r, repos)
        assert repos.tip() == repos.base

    def test_permanent_confirmation_fetch_error_after_accepted_push(self, repos):
        repos.stage(TONIGHT)
        r = run(repos, pushes=["accepted"], fetches=["authfail"])
        head = self._assert_unconfirmed(r, repos)
        assert "a retry cannot fix" in str(r["err"]) and len(r["pushes"]) == 1 and r["sleeps"] == []
        assert repos.tip() == head and repos.count() == 2
        assert TOKEN not in str(r["err"]) and TOKEN not in r["logs"]

    def test_permanent_push_error_after_an_unchecked_push(self, repos):
        """The first push landed unseen; the retry is refused (e.g. an expired
        token). The commit is on the remote, so this must not say "not published"."""
        repos.stage(TONIGHT)
        r = run(repos, pushes=["accepted", "auth403"], fetches=["502"])
        head = self._assert_unconfirmed(r, repos)
        assert repos.tip() == head and repos.count() == 2

    def test_successful_read_back_settles_an_earlier_unchecked_push(self, repos):
        """Attempt 1's outcome is unknown; attempt 2's fetch shows the commit
        absent, so exhaustion is a confirmed non-publication."""
        repos.stage(TONIGHT)
        r = run(repos, pushes=["500", "500"], fetches=["502", "real"], attempts=2)
        assert r["err"].code == pc.EXIT_EXHAUSTED and r["err"].status == pc.NOT_PUBLISHED
        assert repos.tip() == repos.base

    def test_nothing_to_commit_is_a_no_op(self, repos):
        write(repos.work, {"data_files/unstaged.csv": "x\n"})  # present but not staged
        r = run(repos)
        assert r["result"] == pc.NOTHING and r["sha"] == "" and r["err"] is None
        assert r["pushes"] == [] and git(repos.work, "rev-parse", "HEAD") == repos.base == repos.tip()
        assert "Nothing to commit" in r["logs"] and "Published" not in r["logs"]

    def test_unreadable_index_is_a_commit_error(self, tmp_path):
        (tmp_path / "plain").mkdir()
        with pytest.raises(pc.PublishError) as e:
            pc.publish(pc.Git(tmp_path / "plain"), message="m", log=lambda s: None)
        assert e.value.code == pc.EXIT_COMMIT and "staged changes" in str(e.value)

    def test_commit_error_publishes_nothing(self, repos):
        hook = repos.work / ".git" / "hooks" / "pre-commit"
        hook.write_bytes(b"#!/bin/sh\necho 'pre-commit: refused' >&2\nexit 1\n")
        hook.chmod(0o755)
        repos.stage(TONIGHT)
        r = run(repos)
        assert r["err"].code == pc.EXIT_COMMIT and "git commit failed" in str(r["err"])
        assert "pre-commit: refused" in str(r["err"])
        assert r["pushes"] == [] and repos.tip() == repos.base

    def test_accepted_push_reported_as_failure_is_not_repeated(self, repos):
        repos.stage(TONIGHT)
        r = run(repos, pushes=["accepted"])
        assert r["result"] == pc.PUBLISHED and r["sha"] == repos.tip()
        assert len(r["pushes"]) == 1 and r["sleeps"] == [] and repos.count() == 2
        assert "accepted it despite the reported error" in r["logs"]

    def test_accepted_push_with_unconfirmable_remote_is_pushed_again_idempotently(self, repos):
        """The push landed, but reading the branch back failed too: the retry
        pushes the same commit again (a no-op on the server), then confirms."""
        repos.stage(TONIGHT)
        r = run(repos, pushes=["accepted", "real"], fetches=["502", "real"])
        assert r["result"] == pc.PUBLISHED and r["sha"] == repos.tip()
        assert len(r["pushes"]) == 2 and r["pushes"][0] == r["pushes"][1] and repos.count() == 2

    def test_success_is_not_claimed_until_confirmed(self, repos):
        repos.stage(TONIGHT)
        r = run(repos, pushes=["lie", "real"])
        assert r["result"] == pc.PUBLISHED and len(r["pushes"]) == 2
        first = r["logs"].split("Attempt 2/4")[0]
        assert "push reported success" in first and "is not on it" in first and "Published" not in first

    def test_redact(self):
        text = f"https://x-access-token:{TOKEN}@github.com/o/r and {TOKEN} and github_pat_abc123"
        out = pc.redact(text)
        assert TOKEN not in out and "github_pat_abc123" not in out and "https://***@github.com/o/r" in out

    def test_push_never_forces(self):
        src = inspect.getsource(pc.Git.push)
        assert '"push", "--porcelain", remote, refspec' in src and "force" not in src
        assert "--force" not in SCRIPT.read_text(encoding="utf-8")


# --------------------------------------------------------------------- CLI --

FAULTY_MAIN = """\
import json, os, subprocess, sys
sys.path.insert(0, {scripts!r})
import publish_commit as pc
argv = sys.argv[1:]
label = argv[argv.index("--label") + 1] if "--label" in argv else "Publication"
plan = json.loads(os.environ.get("FAKE_PUSH_FAULTS", "{{}}")).get(label, [])
fetch_plan = json.loads(os.environ.get("FAKE_FETCH_FAULTS", "{{}}")).get(label, [])
faults = {faults!r}
real_push, real_fetch = pc.Git.push, pc.Git.fetch
def push(self, remote, refspec):
    action = plan.pop(0) if plan else "real"
    if action == "accepted":  # the server takes it; the client reports a failure
        r = real_push(self, remote, refspec)
        assert r.returncode == 0, r.stderr
        return subprocess.CompletedProcess(r.args, 128, r.stdout, faults["hangup"][1])
    if action != "real":
        code, err = faults[action]
        return subprocess.CompletedProcess(["git", "push"], code, "", err)
    return real_push(self, remote, refspec)
def fetch(self, remote, branch):
    if fetch_plan:
        code, err = faults[fetch_plan.pop(0)]
        return subprocess.CompletedProcess(["git", "fetch"], code, "", err)
    return real_fetch(self, remote, branch)
pc.Git.push, pc.Git.fetch = push, fetch
sys.exit(pc.main(argv))
"""


@pytest.fixture
def faulty_main(tmp_path):
    p = tmp_path / "faulty_main.py"
    p.write_text(FAULTY_MAIN.format(scripts=str(ROOT / "scripts"), faults=FAULTS), encoding="utf-8")
    return p


def cli(repos, tmp_path, faulty_main, faults=None, *args, fetch_faults=None):
    out, summary = tmp_path / "gh_output", tmp_path / "summary.md"
    env = {**os.environ, "GITHUB_ACTIONS": "true", "GITHUB_OUTPUT": str(out), "GITHUB_STEP_SUMMARY": str(summary),
           "PUBLISH_RETRY_DELAY": "0", "FAKE_PUSH_FAULTS": json.dumps(faults or {}),
           "FAKE_FETCH_FAULTS": json.dumps(fetch_faults or {})}
    r = subprocess.run([sys.executable, str(faulty_main), "--label", "Nightly data", "-m", "nightly", *args],
                       cwd=repos.work, env=env, capture_output=True, text=True, encoding="utf-8", timeout=300)
    read = lambda p: p.read_text(encoding="utf-8") if p.exists() else ""
    return r, read(out), read(summary)


class TestCli:
    def test_published_outputs(self, repos, tmp_path, faulty_main):
        repos.stage(TONIGHT)
        r, out, summary = cli(repos, tmp_path, faulty_main)
        assert r.returncode == 0, r.stderr
        assert out == f"result=published\ncommit={repos.tip()}\n"
        assert f"Nightly data: published `{repos.tip()}` to origin/main" in summary

    def test_nothing_to_commit_outputs(self, repos, tmp_path, faulty_main):
        r, out, summary = cli(repos, tmp_path, faulty_main)
        assert r.returncode == 0 and out == "result=nothing-to-commit\ncommit=\n"
        assert "Nightly data: nothing to commit" in summary

    def test_exhausted_retries_exit_nonzero_with_annotation(self, repos, tmp_path, faulty_main):
        repos.stage(TONIGHT)
        r, out, summary = cli(repos, tmp_path, faulty_main, {"Nightly data": ["500"] * 3}, "--attempts", "3")
        head = git(repos.work, "rev-parse", "HEAD")
        assert r.returncode == pc.EXIT_EXHAUSTED and out == f"result=not-published\ncommit={head}\n"
        assert "::error title=Nightly data was not published::" in r.stdout and head in r.stdout
        assert "Attempt 3/3" in r.stdout and "Nightly data: NOT published" in summary
        assert "unconfirmed" not in (r.stdout + r.stderr + summary).lower()
        assert repos.tip() == repos.base

    def test_permanent_error_exit_code_and_no_secret(self, repos, tmp_path, faulty_main):
        repos.stage(TONIGHT)
        r, out, summary = cli(repos, tmp_path, faulty_main, {"Nightly data": ["auth403"]})
        assert r.returncode == pc.EXIT_PERMANENT and out.startswith("result=not-published\ncommit=")
        assert TOKEN not in r.stdout + r.stderr + summary

    def test_unconfirmed_outputs_annotation_and_summary(self, repos, tmp_path, faulty_main):
        repos.stage(TONIGHT)
        r, out, summary = cli(repos, tmp_path, faulty_main, {"Nightly data": ["accepted"]}, "--attempts", "2",
                              fetch_faults={"Nightly data": ["502", "502"]})
        head = git(repos.work, "rev-parse", "HEAD")
        assert r.returncode == pc.EXIT_UNCONFIRMED
        assert repos.tip() == head and repos.count() == 2
        assert out == f"result=unconfirmed\ncommit={head}\n"
        assert ("::error title=Nightly data publication unconfirmed::Publication unconfirmed; the commit may "
                f"already be on origin/main. Commit {head} (origin/main).") in r.stdout
        assert "Nightly data: publication UNCONFIRMED" in summary and f"commit: {head}" in summary
        assert "remote: origin" in summary and "branch: main" in summary
        everything = r.stdout + r.stderr + summary
        assert "was not published" not in everything and "NOT published" not in everything

    def test_bad_arguments(self, repos, tmp_path, faulty_main):
        r, _, _ = cli(repos, tmp_path, faulty_main, None, "--attempts", "0")
        assert r.returncode == pc.EXIT_USAGE


# ------------------------------------------------------- workflow (bash) --

@pytest.fixture(scope="module")
def workflow():
    steps = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["update-predictions"]["steps"]
    return steps, [s["name"] for s in steps]


class TestWorkflowSteps:
    def test_publication_steps_are_required_and_use_the_helper(self, workflow):
        steps, names = workflow
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "git push" not in text and "--force" not in text
        for name in (NIGHTLY, BEST_BETS):
            step = steps[names.index(name)]
            assert "continue-on-error" not in step, name
            assert "python scripts/publish_commit.py" in step["run"], name
        nightly, bets = steps[names.index(NIGHTLY)], steps[names.index(BEST_BETS)]
        assert nightly["id"] == "publish_nightly" and bets["id"] == "publish_best_bets"
        assert "steps.publish_nightly.outcome == 'success'" in bets["if"]
        for guard in (SCHED, PRED, MANIFEST, "data_files/model_metrics.json"):
            assert f"--guard {guard}" in nightly["run"], guard
        assert f"--guard {PRED}" in bets["run"]
        # Still after the gate, in the same order.
        assert names.index("Validate pipeline outputs before publishing") < names.index(NIGHTLY) \
            < names.index(EXPORT) < names.index(BEST_BETS)


def _job(workflow, repos, tmp_path, faults=None, fetch_faults=None):
    """Run the job from the artifact upload onward as Actions would, after a
    passing gate: both publication steps and the export run for real in the
    shallow clone (stand-in `python` runs publish_commit.py with injected push
    faults, and the export writes a new best_bets_today.json)."""
    bash = _bash()
    if not bash:
        pytest.skip("bash not available")
    steps, names = workflow
    bin_, calls = tmp_path / "bin", tmp_path / "calls.log"
    bin_.mkdir()
    helper = tmp_path / "faulty_main.py"
    helper.write_text(FAULTY_MAIN.format(scripts=str(ROOT / "scripts"), faults=FAULTS), encoding="utf-8")
    (bin_ / "python").write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = "scripts/publish_commit.py" ]; then shift; exec "{_posix(Path(sys.executable))}" '
        f'"{_posix(helper)}" "$@"; fi\n'
        f'if [ "$1" = "scripts/export_best_bets.py" ]; then printf \'{{"bets": "tonight"}}\\n\' > {BETS}; exit 0; fi\n'
        f'echo "python $*" >> "{_posix(calls)}"\n', encoding="utf-8", newline="\n")
    (bin_ / "python").chmod(0o755)
    out, summary = tmp_path / "gh_output", tmp_path / "summary.md"
    env = {**os.environ, "GITHUB_OUTPUT": str(out), "GITHUB_STEP_SUMMARY": str(summary), "GITHUB_ACTIONS": "true",
           "PUBLISH_RETRY_DELAY": "0", "FAKE_PUSH_FAULTS": json.dumps(faults or {}),
           "FAKE_FETCH_FAULTS": json.dumps(fetch_faults or {})}
    executed = {NIGHTLY, EXPORT, BEST_BETS}
    outcomes = {"output_gate": "success", "pregame_capture": "success"}
    job_failed, ran, logs = False, [], ""
    for step in steps[names.index("Upload updated predictions"):]:
        if not _condition(step.get("if", ""), outcomes, job_failed):
            if "id" in step:
                outcomes[step["id"]] = "skipped"
            continue
        ran.append(step["name"])
        code = 0
        if step["name"] in executed:
            assert "${{" not in step["run"]
            script = tmp_path / "step.sh"
            script.write_bytes((f'export PATH="{_posix(bin_)}:$PATH"\n' + step["run"].replace("\r\n", "\n")).encode())
            r = subprocess.run([bash, "--noprofile", "--norc", "-eo", "pipefail", str(script)], cwd=repos.work,
                               env=env, capture_output=True, text=True, encoding="utf-8", timeout=300)
            code, logs = r.returncode, logs + f"--- {step['name']} (exit {r.returncode})\n" + r.stdout + r.stderr
        if "id" in step:
            outcomes[step["id"]] = "failure" if code else "success"
        if code and not step.get("continue-on-error"):
            job_failed = True
    read = lambda p: p.read_text(encoding="utf-8") if p.exists() else ""
    return dict(ran=ran, failed=job_failed, outcomes=outcomes, logs=logs, output=read(out), summary=read(summary))


def _messages(repos, n):
    return git(repos.origin, "log", f"-{n}", "--format=%s", "main").splitlines()


class TestWorkflowSimulation:
    def test_both_paths_publish_in_order(self, workflow, repos, tmp_path):
        write(repos.work, {**TONIGHT, "data_files/pregame_snapshots/run2.json": "{}\n"})
        j = _job(workflow, repos, tmp_path)
        assert not j["failed"], j["logs"]
        assert {NIGHTLY, EXPORT, BEST_BETS} <= set(j["ran"])
        assert _messages(repos, 2) == ["chore: update best_bets_today [skip ci]",
                                       "Automated nightly update: predictions and ESPN data [skip ci]"]
        assert repos.count() == 3
        assert show(repos.origin, "main", PRED) == TONIGHT[PRED]
        assert show(repos.origin, "main", "data_files/pregame_snapshots/run2.json") == "{}\n"
        assert show(repos.origin, "main", BETS) == '{"bets": "tonight"}\n'
        assert j["output"].count("result=published") == 2

    @pytest.mark.parametrize("faults", [["500"] * 4, ["auth403"]], ids=["exhausted", "permanent"])
    def test_failed_first_publication_fails_the_job_and_skips_the_rest(self, workflow, repos, tmp_path, faults):
        write(repos.work, TONIGHT)
        j = _job(workflow, repos, tmp_path, {"Nightly data": faults})
        assert j["failed"] and j["outcomes"]["publish_nightly"] == "failure"
        assert EXPORT not in j["ran"] and BEST_BETS not in j["ran"] and "Summary" not in j["ran"]
        assert repos.tip() == repos.base and repos.count() == 1
        assert "::error title=Nightly data was not published::" in j["logs"]
        assert "Nightly data: NOT published" in j["summary"] and "result=published" not in j["output"]

    def test_unconfirmed_first_publication_fails_the_job(self, workflow, repos, tmp_path):
        write(repos.work, TONIGHT)
        j = _job(workflow, repos, tmp_path, {"Nightly data": ["accepted"]}, {"Nightly data": ["502"] * 4})
        assert j["failed"] and j["outcomes"]["publish_nightly"] == "failure"
        assert EXPORT not in j["ran"] and BEST_BETS not in j["ran"]
        head = git(repos.work, "rev-parse", "HEAD")
        assert repos.tip() == head and repos.count() == 2  # it did land, once
        assert "::error title=Nightly data publication unconfirmed::" in j["logs"]
        assert "was not published" not in j["logs"] and "NOT published" not in j["summary"]
        assert "result=unconfirmed" in j["output"] and f"commit={head}" in j["output"]

    def test_nothing_to_commit_still_publishes_best_bets(self, workflow, repos, tmp_path):
        j = _job(workflow, repos, tmp_path)
        assert not j["failed"], j["logs"]
        assert "Nothing to commit" in j["logs"] and BEST_BETS in j["ran"]
        assert _messages(repos, 1) == ["chore: update best_bets_today [skip ci]"] and repos.count() == 2
        assert "result=nothing-to-commit" in j["output"] and "result=published" in j["output"]

    def test_failed_best_bets_publication_fails_the_job(self, workflow, repos, tmp_path):
        write(repos.work, TONIGHT)
        j = _job(workflow, repos, tmp_path, {"Best bets": ["500"] * 4})
        assert j["failed"] and j["outcomes"]["publish_best_bets"] == "failure"
        assert _messages(repos, 1) == ["Automated nightly update: predictions and ESPN data [skip ci]"]
        assert show(repos.origin, "main", BETS) == SEED[BETS]
        assert "::error title=Best bets was not published::" in j["logs"]

    def test_transient_failures_on_both_paths_recover(self, workflow, repos, tmp_path):
        write(repos.work, TONIGHT)
        j = _job(workflow, repos, tmp_path, {"Nightly data": ["500", "hangup"], "Best bets": ["502"]})
        assert not j["failed"], j["logs"]
        assert repos.count() == 3 and j["logs"].count("Attempt 3/4") == 1 and j["logs"].count("Attempt 2/4") == 2

    def test_main_advanced_during_the_run(self, workflow, repos, tmp_path):
        """A capture workflow pushed while the nightly ran: both publications are
        replayed on top of it, and its file is kept."""
        write(repos.work, TONIGHT)
        theirs = repos.concurrent({CAPTURE: '{"capture": 1}\n'})
        j = _job(workflow, repos, tmp_path)
        assert not j["failed"], j["logs"]
        assert repos.count() == 4 and git(repos.origin, "rev-parse", "main~2") == theirs
        assert show(repos.origin, "main", CAPTURE) == '{"capture": 1}\n'
        assert show(repos.origin, "main", PRED) == TONIGHT[PRED]
        assert show(repos.origin, "main", BETS) == '{"bets": "tonight"}\n'

    def test_conflicting_concurrent_change_fails_visibly(self, workflow, repos, tmp_path):
        write(repos.work, TONIGHT)
        theirs = repos.concurrent({MANIFEST: '{"run": "other"}\n'})
        j = _job(workflow, repos, tmp_path)
        assert j["failed"] and BEST_BETS not in j["ran"]
        assert repos.tip() == theirs and show(repos.origin, "main", MANIFEST) == '{"run": "other"}\n'
        assert MANIFEST in j["logs"] and "::error title=Nightly data was not published::" in j["logs"]
        head = git(repos.work, "rev-parse", "HEAD")
        assert show(repos.work, head, MANIFEST) == TONIGHT[MANIFEST]
