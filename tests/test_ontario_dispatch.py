"""Windows Task Scheduler dispatcher for the Ontario Spread Capture workflow
(scripts/ontario_dispatch.py).

GitHub is mocked (FakeGitHub serves real capture documents produced by Phase
1's capture() against the test store's mocked Odds API); clocks are fixed or
advanced explicitly; nothing is dispatched, registered or fetched for real.
"""
import io
import json
import os
import shutil
import socket
import subprocess
import time as _time
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import pytest

import ontario_spreads as on
from test_ontario_spread_report import Store, sun_events, wed_events
from test_ontario_spreads import SUN_SLOT, WED_SLOT, _event, _quotes
from test_team_features import ROOT

import importlib.util
_spec = importlib.util.spec_from_file_location("ontario_dispatch",
                                               ROOT / "scripts" / "ontario_dispatch.py")
od = importlib.util.module_from_spec(_spec)
import sys  # noqa: E402
sys.modules["ontario_dispatch"] = od
_spec.loader.exec_module(od)

UTC = timezone.utc
TOKEN = "github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ"
WED = "2026-10-07_wednesday_noon"
WED_1205 = WED_SLOT + timedelta(minutes=5)          # 12:05 EDT
WED_CLOSE = WED_SLOT + timedelta(hours=3)          # 15:00 EDT
MAIN_SHA = "c0ffee" + "0" * 34


# ------------------------------------------------------------------ fakes --

class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += timedelta(seconds=seconds)


class FakeGitHub:
    """Stands in for od.GitHub. Serves capture files from a directory and a
    mutable workflow-run list; a dispatch can optionally start a run that
    completes (and captures) as the clock advances."""

    def __init__(self, capture_dir, clock, runs=None, on_dispatch=None, fail=None, blobs=None,
                 advance=None):
        self.capture_dir, self.clock = capture_dir, clock
        self.run_list = list(runs or [])
        self.dispatched, self.on_dispatch, self.fail = [], on_dispatch, fail or {}
        self.calls, self.refs = [], []
        self.sha, self.blobs = MAIN_SHA, blobs or {}
        self.advance = advance or {}          # request name -> time it takes (slow GitHub)

    def _maybe_fail(self, what):
        self.calls.append(what)
        if what in self.advance:
            self.clock.t += self.advance[what]
        exc = self.fail.get(what)
        if exc:
            raise exc

    def main_sha(self):
        self._maybe_fail("main_sha")
        return self.sha

    def capture_names(self, ref="main"):
        self._maybe_fail("capture_names")
        self.refs.append(ref)
        return sorted(p.name for p in self.capture_dir.glob("*.json")) \
            if self.capture_dir.exists() else []

    def capture_doc(self, name, ref="main"):
        self._maybe_fail("capture_doc")
        self.refs.append(ref)
        return json.loads((self.capture_dir / name).read_text(encoding="utf-8"))

    def blob_sha(self, path, ref):
        # By default main's validator files match this checkout.
        return self.blobs.get(path) or od.local_blob_sha(od.ROOT / path)

    def runs(self, **params):
        self._maybe_fail("runs")
        if self.on_dispatch:
            self.on_dispatch(self, "poll")
        out = [dict(r) for r in self.run_list]
        if params.get("event"):
            out = [r for r in out if r["event"] == params["event"]]
        return out

    def dispatch(self, slot_name):
        # Records every request sent; a configured failure is raised after the
        # on_dispatch hook, so "GitHub accepted it but the answer was lost"
        # (hook creates a run, then DispatchUncertain) can be simulated.
        self.calls.append("dispatch")
        self.dispatched.append((slot_name, self.clock()))
        if self.on_dispatch:
            self.on_dispatch(self, "dispatch")
        if self.fail.get("dispatch"):
            raise self.fail["dispatch"]
        return 204


def run_rec(id_, created, status="completed", conclusion="success", event="workflow_dispatch"):
    return {"id": id_, "event": event, "status": status, "conclusion": conclusion,
            "created_at": on.iso(created), "html_url": f"https://example.invalid/runs/{id_}"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Temp capture store + state dir + fixed clock + logger; network blocked."""
    def blocked(*a, **k):
        raise AssertionError("real network access attempted")
    monkeypatch.setattr(od.urllib.request, "urlopen", blocked)
    store = Store(tmp_path, monkeypatch)
    state = tmp_path / "state"
    log = od.setup_logging(state, verbose=False)

    class Env:
        pass
    e = Env()
    e.store, e.state, e.log, e.tmp = store, state, log, tmp_path

    def ctx(now, **kw):
        e.clock = Clock(now)
        fake = kw.pop("gh", None) or FakeGitHub(store.capture_dir, e.clock, **kw)
        e.gh = fake
        return od.Context(gh=fake, log=log, state_dir=state, now=e.clock, sleep=e.clock.sleep,
                          wait_for_run=timedelta(minutes=12), poll=timedelta(seconds=30))
    e.ctx = ctx

    def logged():
        for h in log.handlers:
            h.flush()
        text = (state / "dispatch.log").read_text(encoding="utf-8")
        return text, [json.loads(line.split(" ", 3)[3]) for line in text.splitlines()
                      if line.split(" ", 3)[3].startswith("{")]
    e.logged = logged
    return e


def outcomes(env):
    return [r["outcome"] for r in env.logged()[1]]


# --------------------------------------------------------------- timezone --

class TestTimezone:
    def test_this_computer_follows_toronto_when_eastern(self):
        # Uses the real OS rules; on a machine set to Eastern with DST this is empty.
        problems = od.check_local_timezone(WED_1205)
        if _time.localtime(WED_1205.timestamp()).tm_gmtoff != -14400:
            pytest.skip("computer is not on Eastern time")
        assert problems == []

    def test_fixed_offset_computer_is_flagged(self):
        class TM:
            tm_gmtoff = -18000                               # EST all year (no DST)
        problems = od.check_local_timezone(WED_1205, localtime=lambda ts: TM)
        assert problems and "computer UTC-5" in problems[0] and "Toronto UTC-4" in problems[0]

    @pytest.mark.parametrize("now,expected", [
        (datetime(2026, 11, 1, 14, 5, tzinfo=UTC), "2026-11-01_sunday_morning"),   # 09:05 EST
        (datetime(2026, 11, 1, 13, 5, tzinfo=UTC), None),                         # 08:05 EST
        (datetime(2026, 11, 4, 17, 5, tzinfo=UTC), "2026-11-04_wednesday_noon"),   # 12:05 EST
        (datetime(2026, 11, 4, 16, 5, tzinfo=UTC), None),                         # 11:05 EST
        (datetime(2027, 3, 14, 13, 5, tzinfo=UTC), "2027-03-14_sunday_morning"),   # 09:05 EDT
        (datetime(2026, 10, 11, 13, 5, tzinfo=UTC), "2026-10-11_sunday_morning"),  # 09:05 EDT
    ])
    def test_dispatch_windows_follow_toronto_dst(self, env, now, expected):
        ctx = env.ctx(now)
        ctx.wait_for_run = timedelta(0)
        od.run(ctx, "dispatch")
        if expected is None:
            # Outside the window: never dispatched (earlier missed slots may be logged).
            assert env.gh.dispatched == []
        else:
            assert [s for s, _ in env.gh.dispatched] == [expected.split("_", 1)[1]]
            assert od.on.resolve_slot(now)["slot_id"] == expected


# --------------------------------------------------------- dispatch rules --

class TestDispatchDecisions:
    def test_scheduled_dispatch_names_the_slot(self, env):
        ctx = env.ctx(WED_1205)
        ctx.wait_for_run = timedelta(0)
        od.run(ctx, "dispatch")
        assert env.gh.dispatched == [("wednesday_noon", WED_1205)]
        assert "dispatch_accepted" in outcomes(env)

    def test_existing_usable_capture_prevents_duplicate(self, env):
        env.store.capture(WED_SLOT, wed_events())                    # usable scheduled capture
        ctx = env.ctx(WED_1205)
        assert od.run(ctx, "dispatch") == od.EXIT_OK
        assert env.gh.dispatched == [] and outcomes(env)[-1] == "capture_persisted"

    def test_unusable_capture_does_not_count(self, env):
        env.store.capture(WED_SLOT, [])                              # empty, kept as evidence
        ctx = env.ctx(WED_SLOT + timedelta(minutes=30))
        ctx.wait_for_run = timedelta(0)
        od.run(ctx, "dispatch")
        assert len(env.gh.dispatched) == 1

    def test_ad_hoc_capture_does_not_count(self, env):
        env.store.capture(WED_SLOT + timedelta(minutes=2), wed_events(), slot="ad_hoc")
        ctx = env.ctx(WED_1205)
        ctx.wait_for_run = timedelta(0)
        od.run(ctx, "dispatch")
        assert len(env.gh.dispatched) == 1

    @pytest.mark.parametrize("status", ["queued", "in_progress", "waiting", "requested",
                                        "pending"])
    def test_queued_or_running_job_blocks_dispatch(self, env, status):
        ctx = env.ctx(WED_SLOT + timedelta(minutes=30),
                      runs=[run_rec(7, WED_SLOT + timedelta(minutes=6), status=status,
                                    conclusion=None)])
        assert od.run(ctx, "dispatch") == od.EXIT_OK
        assert env.gh.dispatched == [] and outcomes(env)[-1] == "run_active_not_dispatching"

    def test_failed_run_allows_one_retry_then_limit(self, env):
        failed = run_rec(7, WED_1205, conclusion="failure")
        ctx = env.ctx(WED_SLOT + timedelta(minutes=30), runs=[failed])
        ctx.wait_for_run = timedelta(0)
        od.run(ctx, "dispatch")
        assert len(env.gh.dispatched) == 1                          # the 12:30 retry
        env.logged()
        ctx2 = env.ctx(WED_SLOT + timedelta(minutes=50),
                       runs=[failed, run_rec(8, WED_SLOT + timedelta(minutes=30),
                                             conclusion="failure")])
        assert od.run(ctx2, "dispatch") == od.EXIT_ATTENTION
        assert env.gh.dispatched == [] and outcomes(env)[-1] == "dispatch_limit_reached"

    def test_in_flight_dispatch_not_repeated(self, env):
        ctx = env.ctx(WED_1205)
        ctx.wait_for_run = timedelta(0)
        od.run(ctx, "dispatch")                                      # run not yet visible
        ctx2 = env.ctx(WED_1205 + timedelta(minutes=1))
        assert od.run(ctx2, "dispatch") == od.EXIT_OK
        assert env.gh.dispatched == [] and outcomes(env)[-1] == "dispatch_in_flight_not_dispatching"

    def test_no_dispatch_in_last_minutes_of_window(self, env):
        ctx = env.ctx(WED_CLOSE - timedelta(minutes=10))
        assert od.run(ctx, "dispatch") == od.EXIT_ATTENTION
        assert env.gh.dispatched == [] and outcomes(env)[-1] == "too_close_to_window_close"

    def test_invalid_capture_on_main_stops_dispatch(self, env):
        env.store.capture(WED_SLOT, wed_events())
        [path] = env.store.capture_dir.glob("*.json")
        path.write_text(path.read_text().replace('"usable": true', '"usable": false'))
        ctx = env.ctx(WED_SLOT + timedelta(minutes=30))
        assert od.run(ctx, "dispatch") == od.EXIT_ATTENTION
        assert env.gh.dispatched == [] and outcomes(env)[-1] == "capture_invalid"

    # -- the clock is read again after the GitHub checks --

    CHECKS_START = WED_CLOSE - timedelta(minutes=20)             # 14:40, cutoff 14:45

    def test_cutoff_crossed_during_github_checks_stops_dispatch(self, env):
        ctx = env.ctx(self.CHECKS_START, advance={"runs": timedelta(minutes=6)})
        assert od.run(ctx, "dispatch") == od.EXIT_ATTENTION
        assert env.gh.dispatched == [] and _ledger(env) == []
        rec = env.logged()[1][-1]
        assert rec["outcome"] == "too_close_to_window_close"
        assert rec["checks_started_at"] == on.iso(self.CHECKS_START)
        assert rec["checked_at"] == on.iso(self.CHECKS_START + timedelta(minutes=6))

    def test_window_closed_during_github_checks_stops_dispatch(self, env):
        ctx = env.ctx(self.CHECKS_START,
                      advance={"capture_names": timedelta(minutes=12),
                               "runs": timedelta(minutes=13)})        # e.g. retries, a suspend
        assert od.run(ctx, "dispatch") == od.EXIT_ATTENTION
        assert env.gh.dispatched == [] and _ledger(env) == []
        assert outcomes(env)[-1] == "window_closed_during_checks"

    def test_dry_run_rechecks_the_clock_too(self, env):
        ctx = env.ctx(self.CHECKS_START, advance={"main_sha": timedelta(minutes=6)})
        ctx.dry_run = True
        od.run(ctx, "dispatch")
        assert outcomes(env)[-1] == "too_close_to_window_close"

    def test_slow_checks_inside_the_limit_dispatch_at_the_new_time(self, env):
        ctx = env.ctx(self.CHECKS_START, advance={"runs": timedelta(minutes=4)})
        ctx.wait_for_run = timedelta(0)
        od.run(ctx, "dispatch")
        sent_at = self.CHECKS_START + timedelta(minutes=4)
        assert env.gh.dispatched == [("wednesday_noon", sent_at)]
        assert _ledger(env)[0]["at"] == on.iso(sent_at)

    def test_cutoff_is_inclusive_at_the_exact_boundary(self, env):
        ctx = env.ctx(self.CHECKS_START, advance={"runs": timedelta(minutes=5)})  # 14:45:00
        ctx.wait_for_run = timedelta(0)
        od.run(ctx, "dispatch")
        assert len(env.gh.dispatched) == 1

    def test_dry_run_never_dispatches(self, env):
        ctx = env.ctx(WED_1205)
        ctx.dry_run = True
        assert od.run(ctx, "dispatch") == od.EXIT_OK
        assert env.gh.dispatched == [] and outcomes(env)[-1] == "dry_run_would_dispatch"
        assert od.load_ledger(env.state)["dispatches"] == []


# ------------------------------------------------------- reboot recovery --

class TestRecovery:
    def test_reboot_inside_window_recovers(self, env):
        ctx = env.ctx(WED_SLOT + timedelta(hours=1, minutes=30))     # 13:30 EDT
        ctx.wait_for_run = timedelta(0)
        od.run(ctx, "recover")
        assert env.gh.dispatched == [("wednesday_noon", ctx.now())]
        assert json.loads(env.logged()[0].splitlines()[-2].split(" ", 3)[3])["reason"] == \
            "recovery"

    def test_reboot_outside_window_logs_missed_once_without_dispatch(self, env):
        ctx = env.ctx(WED_CLOSE + timedelta(minutes=30))             # 15:30 EDT
        assert od.run(ctx, "recover") == od.EXIT_ATTENTION
        assert env.gh.dispatched == []
        [missed] = [r for r in env.logged()[1] if r["outcome"] == "missed_opportunity"]
        assert missed["slot_id"] == WED and "no ad-hoc capture" in missed["note"]
        ctx2 = env.ctx(WED_CLOSE + timedelta(hours=2))
        assert od.run(ctx2, "recover") == od.EXIT_OK                 # not logged again
        assert outcomes(env).count("missed_opportunity") == 1 and env.gh.dispatched == []

    def test_outside_window_with_capture_is_not_missed(self, env):
        env.store.capture(WED_SLOT, wed_events())
        ctx = env.ctx(WED_CLOSE + timedelta(minutes=30))
        assert od.run(ctx, "recover") == od.EXIT_OK
        assert "missed_opportunity" not in outcomes(env)

    def test_before_tracking_start_nothing_missed(self, env):
        ctx = env.ctx(datetime(2026, 10, 6, 12, tzinfo=UTC))
        assert od.run(ctx, "recover") == od.EXIT_OK
        assert outcomes(env)[-1] == "outside_window" and env.gh.calls == []


# ------------------------------------------- outcomes after a dispatch --

def _simulate(env, conclusion="success", capture=True):
    """A dispatch starts a queued run; the next polls move it to running and
    then completed, optionally committing a real usable capture."""
    state = {"polls": 0}

    def hook(gh, what):
        if what == "dispatch":
            gh.run_list.append(run_rec(99, gh.clock(), status="queued", conclusion=None))
            return
        state["polls"] += 1
        run = gh.run_list[-1] if gh.run_list else None
        if run is None:
            return
        if state["polls"] == 2:
            run["status"] = "in_progress"
        if state["polls"] == 3:
            if capture:
                env.store.capture(gh.clock(), wed_events())
            run.update(status="completed", conclusion=conclusion)
    return hook


class TestOutcomes:
    def test_accepted_queued_then_persisted(self, env):
        ctx = env.ctx(WED_1205, on_dispatch=None)
        env.gh.on_dispatch = _simulate(env)
        assert od.run(ctx, "dispatch") == od.EXIT_OK
        seq = outcomes(env)
        assert seq.index("dispatch_accepted") < seq.index("run_queued") < \
            seq.index("capture_persisted")

    def test_failed_run_is_not_success(self, env):
        ctx = env.ctx(WED_1205)
        env.gh.on_dispatch = _simulate(env, conclusion="failure", capture=False)
        assert od.run(ctx, "dispatch") == od.EXIT_ATTENTION
        assert outcomes(env)[-1] == "run_failed"

    def test_green_run_without_capture_is_not_success(self, env):
        ctx = env.ctx(WED_1205)
        env.gh.on_dispatch = _simulate(env, conclusion="success", capture=False)
        assert od.run(ctx, "dispatch") == od.EXIT_ATTENTION
        assert outcomes(env)[-1] == "run_succeeded_without_capture"

    def test_bounded_wait(self, env):
        ctx = env.ctx(WED_1205)                                      # run never appears
        assert od.run(ctx, "dispatch") == od.EXIT_ATTENTION
        assert outcomes(env)[-1] == "run_not_found"
        assert ctx.now() - WED_1205 <= timedelta(minutes=12, seconds=30)

    @pytest.mark.parametrize("runs,expected,code", [
        ([], "status_no_run", od.EXIT_ATTENTION),
        ([run_rec(5, WED_1205, conclusion="failure")], "status_run_failed", od.EXIT_ATTENTION),
        ([run_rec(5, WED_1205, status="in_progress", conclusion=None)],
         "status_run_in_progress", od.EXIT_OK),
    ])
    def test_status_check(self, env, runs, expected, code):
        ctx = env.ctx(WED_SLOT + timedelta(hours=1), runs=runs)       # 13:00 EDT
        assert od.run(ctx, "status") == code
        assert env.gh.dispatched == [] and outcomes(env)[-1] == expected

    def test_status_check_with_capture(self, env):
        env.store.capture(WED_SLOT + timedelta(minutes=5), wed_events())
        ctx = env.ctx(WED_SLOT + timedelta(hours=1))
        assert od.run(ctx, "status") == od.EXIT_OK and outcomes(env)[-1] == "capture_persisted"


# ---------------------------------------- auth, network and redaction --

class TestAuthNetworkRedaction:
    def _main(self, env, monkeypatch, *, token=TOKEN, fake=None, now=WED_1205, args=()):
        def loader(source):
            if token is None:
                raise od.AuthUnavailable("no Windows Credential Manager entry")
            od.REDACT.add(token)
            return token
        monkeypatch.setattr(od, "load_token", loader)
        clock = Clock(now)
        monkeypatch.setattr(od, "utc_now", clock)
        fake = fake or FakeGitHub(env.store.capture_dir, clock)
        monkeypatch.setattr(od, "GitHub", lambda tok: fake)
        monkeypatch.setattr(od, "Context", lambda **kw: od.__dict__["_RealContext"](
            **{**kw, "now": clock, "sleep": clock.sleep, "wait_for_run": timedelta(0)}))
        code = od.main(["run", "--mode", "dispatch", "--state-dir", str(env.state), *args])
        return code, fake

    @pytest.fixture(autouse=True)
    def _real_context(self, monkeypatch):
        monkeypatch.setitem(od.__dict__, "_RealContext", od.Context)

    def test_unavailable_authentication(self, env, monkeypatch):
        code, fake = self._main(env, monkeypatch, token=None)
        assert code == od.EXIT_AUTH and fake.dispatched == []
        assert "auth_unavailable" in outcomes(env)

    def test_network_failure(self, env, monkeypatch):
        fake = FakeGitHub(env.store.capture_dir, Clock(WED_1205),
                          fail={"runs": od.NetworkError("GitHub unreachable (URLError)")})
        code, _ = self._main(env, monkeypatch, fake=fake)
        assert code == od.EXIT_NETWORK and fake.dispatched == []
        assert "network_error" in outcomes(env)

    def test_rejected_token_mid_run(self, env, monkeypatch):
        fake = FakeGitHub(env.store.capture_dir, Clock(WED_1205),
                          fail={"capture_names": od.AuthUnavailable("HTTP 401")})
        code, _ = self._main(env, monkeypatch, fake=fake)
        assert code == od.EXIT_AUTH and fake.dispatched == []

    def test_token_never_logged(self, env, monkeypatch):
        leak = f"boom Authorization: Bearer {TOKEN} and ghp_{'a' * 36} and {TOKEN}"
        fake = FakeGitHub(env.store.capture_dir, Clock(WED_1205),
                          fail={"runs": od.NetworkError(leak)})
        self._main(env, monkeypatch, fake=fake)
        text, _ = env.logged()
        assert TOKEN not in text and "ghp_" + "a" * 36 not in text
        assert "[REDACTED]" in text

    def test_dry_run_flag(self, env, monkeypatch):
        code, fake = self._main(env, monkeypatch, args=("--dry-run",))
        assert code == od.EXIT_OK and fake.dispatched == []
        assert outcomes(env)[-1] == "dry_run_would_dispatch"

    def test_second_instance_exits_quietly(self, env, monkeypatch):
        env.state.mkdir(parents=True, exist_ok=True)
        with on.SlotLock(env.state, "dispatcher", "other", what="dispatcher run"):
            code, fake = self._main(env, monkeypatch)
        assert code == od.EXIT_OK and fake.dispatched == []
        assert outcomes(env)[-1] == "another_instance_running"


class TestHttpLayer:
    def _resp(self, status, body=b"{}"):
        class R(io.BytesIO):
            def __init__(self):
                super().__init__(body)
                self.status = status

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return R()

    def test_headers_and_dispatch_body(self):
        seen = []

        def opener(req, timeout):
            seen.append(req)
            return self._resp(204, b"")
        gh = od.GitHub(TOKEN, opener=opener, sleep=lambda s: None)
        assert gh.dispatch("sunday_morning") == 204
        req = seen[0]
        assert req.get_method() == "POST" and req.full_url.endswith(
            "/repos/William-Bill1/nfl-predictions/actions/workflows/ontario-spread-capture.yml"
            "/dispatches")
        assert json.loads(req.data) == {"ref": "main", "inputs": {"slot": "sunday_morning"}}
        assert req.get_header("Authorization") == f"Bearer {TOKEN}"

    def test_401_is_auth_error_without_token(self):
        def opener(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 401, "Bad credentials", {},
                                         io.BytesIO(TOKEN.encode()))
        gh = od.GitHub(TOKEN, opener=opener, sleep=lambda s: None)
        with pytest.raises(od.AuthUnavailable) as exc:
            gh.runs()
        assert TOKEN not in str(exc.value) and "HTTP 401" in str(exc.value)

    def test_network_errors_are_bounded(self):
        attempts, sleeps = [], []

        def opener(req, timeout):
            attempts.append(timeout)
            raise urllib.error.URLError("Temporary failure in name resolution")
        gh = od.GitHub(TOKEN, opener=opener, sleep=sleeps.append)
        with pytest.raises(od.NetworkError):
            gh.runs()
        assert len(attempts) == od.HTTP_ATTEMPTS and all(t == od.HTTP_TIMEOUT for t in attempts)
        assert len(sleeps) == od.HTTP_ATTEMPTS - 1

    def test_no_captures_directory_is_empty(self):
        def opener(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, io.BytesIO(b""))
        assert od.GitHub(TOKEN, opener=opener, sleep=lambda s: None).capture_names() == []


class TestRedactor:
    def test_patterns_and_registered_secrets(self):
        r = od.Redactor()
        r.add("s3cret-value-123")
        text = r(f"token ghp_{'x' * 36} pat github_pat_{'y' * 40} Authorization: token abc "
                 "s3cret-value-123")
        assert "ghp_" not in text and "github_pat_" not in text and "abc" not in text
        assert "s3cret-value-123" not in text and text.count("[REDACTED]") >= 4


# ------------------------------------------------------------ task XML --

NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}


def _xml(task):
    text = od.task_xml(task, r"C:\repo\venv\Scripts\python.exe", r"C:\repo", r"PC\user")
    return text, ET.fromstring(text.split("\n", 1)[1])


class TestTaskXml:
    @pytest.mark.parametrize("task", sorted(od.TASKS))
    def test_common_settings(self, task):
        text, root = _xml(task)
        s = root.find("t:Settings", NS)
        get = lambda tag: s.find(f"t:{tag}", NS).text  # noqa: E731
        assert get("MultipleInstancesPolicy") == "IgnoreNew"
        assert get("DisallowStartIfOnBatteries") == "false"
        assert get("StopIfGoingOnBatteries") == "false"
        assert get("StartWhenAvailable") == "true"
        assert get("RunOnlyIfNetworkAvailable") == "true"
        assert get("WakeToRun") == ("false" if task == "recovery" else "true")
        p = root.find("t:Principals/t:Principal", NS)
        assert p.find("t:LogonType", NS).text == "Password"
        assert p.find("t:RunLevel", NS).text == "LeastPrivilege"
        args = root.find("t:Actions/t:Exec/t:Arguments", NS).text
        assert f"--mode {od.TASKS[task]['mode']}" in args
        # No credential material anywhere in the task definition.
        for secret_shape in ("github_pat_", "ghp_", "gho_", "bearer", "password>",
                             "authorization"):
            assert secret_shape not in text.lower(), secret_shape

    def test_trigger_times(self):
        def times(task):
            _, root = _xml(task)
            out = []
            for t in root.findall("t:Triggers/t:CalendarTrigger", NS):
                day = t.find("t:ScheduleByWeek/t:DaysOfWeek", NS)[0].tag.split("}")[1]
                out.append((day, t.find("t:StartBoundary", NS).text[-8:-3]))
            return sorted(out)
        assert times("dispatch") == [("Sunday", "09:05"), ("Sunday", "09:30"),
                                     ("Wednesday", "12:05"), ("Wednesday", "12:30")]
        assert times("status") == [("Sunday", "10:00"), ("Wednesday", "13:00")]
        _, root = _xml("recovery")
        assert root.find("t:Triggers/t:BootTrigger/t:Delay", NS).text == "PT3M"
        events = {t.find("t:Subscription", NS).text: t.find("t:Delay", NS).text
                  for t in root.findall("t:Triggers/t:EventTrigger", NS)}
        assert len(events) == 3
        [resume] = [d for s, d in events.items() if "Power-Troubleshooter" in s and "EventID=1]" in s]
        [standby] = [d for s, d in events.items() if "Kernel-Power" in s and "EventID=507]" in s]
        [network] = [d for s, d in events.items()
                     if "NetworkProfile/Operational" in s and "EventID=10000]" in s]
        assert (resume, standby, network) == ("PT2M", "PT2M", "PT1M")

    def test_triggers_are_local_time_without_offset(self):
        _, root = _xml("dispatch")
        for b in root.iter("{%s}StartBoundary" % NS["t"]):
            assert len(b.text) == 19 and not b.text.endswith("Z")       # no UTC/offset

    def test_retries_and_status_fall_inside_their_windows(self):
        for name, slot_name in (("wednesday_noon", "wednesday_noon"),
                                ("sunday_morning", "sunday_morning")):
            rule = on.SLOTS_BY_NAME[slot_name]
            times = od.TASKS["dispatch"]["times"][name] + od.TASKS["status"]["times"][name]
            for hhmm in times:
                h, m = map(int, hhmm.split(":"))
                offset = timedelta(hours=h - rule.local_time.hour, minutes=m)
                assert timedelta(0) < offset < rule.window - od.DISPATCH_CUTOFF


def test_dispatcher_never_touches_the_odds_api():
    src = (ROOT / "scripts" / "ontario_dispatch.py").read_text(encoding="utf-8")
    assert "the-odds-api" not in src.lower()
    assert "ontario_spreads.capture(" not in src and "on.capture(" not in src
    # The key's name appears only where it is removed from the environment.
    uses = [line.strip() for line in src.splitlines() if '"ODDS_API_KEY"' in line]
    assert uses == ['_INHERITED_ODDS_KEY = os.environ.pop("ODDS_API_KEY", None)',
                    'os.environ["ODDS_API_KEY"] = ""']


def test_odds_api_key_kept_out_of_the_dispatcher_process():
    # A key in the environment is dropped and redacted, and the repository's
    # .env (loaded by python-dotenv when ontario_spreads is imported) is not
    # copied in either. Only booleans are printed - never a key.
    fake = "fake-odds-key-0123456789abcdef"
    code = (
        "import importlib.util, os, sys\n"
        f"spec = importlib.util.spec_from_file_location('od', r'{ROOT / 'scripts' / 'ontario_dispatch.py'}')\n"
        "od = importlib.util.module_from_spec(spec); sys.modules['od'] = od\n"
        "spec.loader.exec_module(od)\n"
        "import player_props.market_odds as mo\n"
        "print(os.environ.get('ODDS_API_KEY') == '', mo.ODDS_API_KEY == '', "
        f"od.REDACT('k={fake}') == 'k=[REDACTED]')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=ROOT, timeout=120,
                       env={**os.environ, "ODDS_API_KEY": fake})
    assert fake not in r.stdout + r.stderr
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split() == ["True", "True", "True"]


def test_state_dir_avoids_redirected_appdata():
    # Microsoft Store Python redirects %LOCALAPPDATA% writes into its package
    # folder; logs must live where the inspect script and the user look.
    assert "appdata" not in str(od.STATE_DIR).lower()
    assert od.STATE_DIR.parts[-2:] == (".nfl-predictions", "ontario-dispatch")


# ------------------------------------- uncertain and refused dispatches --

def _ledger(env):
    return od.load_ledger(env.state)["dispatches"]


class TestUncertainDispatch:
    def test_run_after_uncertain_request_is_only_a_candidate(self, env):
        # A run appears after the request whose answer never arrived. It may be
        # this request's run or another dispatch's: reported as a candidate,
        # never as proof the request was accepted; the request isn't re-sent.
        ctx = env.ctx(WED_1205, fail={"dispatch": od.DispatchUncertain("timed out")})
        env.gh.on_dispatch = _simulate(env)
        assert od.run(ctx, "dispatch") == od.EXIT_OK
        assert len(env.gh.dispatched) == 1                           # never re-sent
        seq = outcomes(env)
        assert seq.index("dispatch_uncertain") < seq.index("dispatch_candidate_run") < \
            seq.index("capture_persisted")
        assert "dispatch_accepted" not in seq and "dispatch_reconciled" not in seq
        [cand] = [r for r in env.logged()[1] if r["outcome"] == "dispatch_candidate_run"]
        assert cand["run_identity"].startswith("candidate") and "may be" in cand["note"]
        text = env.logged()[0].lower()
        assert "so github accepted" not in text and "proves" not in text
        assert [d["result"] for d in _ledger(env)] == ["uncertain"]

    def test_followed_runs_are_labelled_candidates(self, env):
        ctx = env.ctx(WED_1205)
        env.gh.on_dispatch = _simulate(env, conclusion="failure", capture=False)
        od.run(ctx, "dispatch")
        recs = {r["outcome"]: r for r in env.logged()[1]}
        assert recs["run_queued"]["run_identity"].startswith("candidate")
        assert recs["run_failed"]["run_identity"].startswith("candidate")

    def test_uncertain_without_a_run_is_reported_not_resent(self, env):
        ctx = env.ctx(WED_1205, fail={"dispatch": od.DispatchUncertain("timed out")})
        assert od.run(ctx, "dispatch") == od.EXIT_ATTENTION
        assert len(env.gh.dispatched) == 1
        assert outcomes(env)[-2:] == ["dispatch_uncertain", "dispatch_uncertain_no_run"]
        assert ctx.now() - WED_1205 <= od.IN_FLIGHT + timedelta(seconds=30)   # bounded

    def test_uncertain_request_blocks_a_resume_trigger_while_in_flight(self, env):
        ctx = env.ctx(WED_1205, fail={"dispatch": od.DispatchUncertain("timed out")})
        ctx.wait_for_run = timedelta(0)
        od.run(ctx, "dispatch")
        ctx2 = env.ctx(WED_1205 + timedelta(minutes=1))              # e.g. resume from sleep
        assert od.run(ctx2, "recover") == od.EXIT_OK
        assert env.gh.dispatched == []
        assert outcomes(env)[-1] == "dispatch_in_flight_not_dispatching"

    def test_uncertain_requests_without_runs_do_not_use_the_limit_but_are_capped(self, env):
        sent = []
        for minutes in (5, 30, 50, 70):                              # four triggers, no run ever
            ctx = env.ctx(WED_SLOT + timedelta(minutes=minutes),
                          fail={"dispatch": od.DispatchUncertain("timed out")})
            ctx.wait_for_run = timedelta(0)
            od.run(ctx, "dispatch")
            sent.append(len(env.gh.dispatched))
        # GitHub shows no runs, so the limit of 2 runs is never reached; the
        # local ceiling stops the requests after MAX_REQUESTS.
        assert sent == [1, 1, 1, 0] and od.MAX_REQUESTS == 3
        assert outcomes(env)[-1] == "dispatch_request_ceiling_reached"
        assert "dispatch_limit_reached" not in outcomes(env)

    def test_interrupted_request_counts_as_in_flight(self, env):
        # The process was killed between recording the request and the answer.
        od.save_ledger(env.state, {"missed_logged": [], "dispatches": [
            {"slot_id": WED, "at": on.iso(WED_1205), "reason": "scheduled", "result": "sending"}]})
        ctx = env.ctx(WED_1205 + timedelta(minutes=2))
        assert od.run(ctx, "recover") == od.EXIT_OK and env.gh.dispatched == []
        assert outcomes(env)[-1] == "dispatch_in_flight_not_dispatching"

    def test_refused_request_does_not_block_or_count(self, env):
        ctx = env.ctx(WED_1205, fail={"dispatch": od.NetworkError("GitHub HTTP 422", status=422)})
        assert od.run(ctx, "dispatch") == od.EXIT_ATTENTION
        assert outcomes(env)[-1] == "dispatch_rejected"
        assert [d["result"] for d in _ledger(env)] == ["rejected_http_422"]
        ctx2 = env.ctx(WED_1205 + timedelta(minutes=1))
        ctx2.wait_for_run = timedelta(0)
        od.run(ctx2, "recover")
        assert len(env.gh.dispatched) == 1                           # not "in flight"

    def test_unreachable_post_is_a_network_error_and_does_not_block(self, env):
        ctx = env.ctx(WED_1205, fail={"dispatch": od.NetworkError("GitHub unreachable")})
        with pytest.raises(od.NetworkError):
            od.run(ctx, "dispatch")
        assert [d["result"] for d in _ledger(env)] == ["not_accepted"]
        ctx2 = env.ctx(WED_1205 + timedelta(minutes=1))
        ctx2.wait_for_run = timedelta(0)
        od.run(ctx2, "recover")
        assert len(env.gh.dispatched) == 1


class TestLimitCounting:
    def test_cron_runs_do_not_use_the_dispatch_limit(self, env):
        cron = [run_rec(i, WED_SLOT + timedelta(minutes=i), event="schedule") for i in (1, 2)]
        ctx = env.ctx(WED_SLOT + timedelta(minutes=30), runs=cron)
        ctx.wait_for_run = timedelta(0)
        od.run(ctx, "dispatch")
        assert len(env.gh.dispatched) == 1

    def test_dispatches_before_the_window_do_not_count(self, env):
        early = [run_rec(i, WED_SLOT - timedelta(hours=i)) for i in (1, 2)]
        ctx = env.ctx(WED_1205, runs=early)
        ctx.wait_for_run = timedelta(0)
        od.run(ctx, "dispatch")
        assert len(env.gh.dispatched) == 1

    @pytest.mark.parametrize("failure", [
        {"runs": od.NetworkError("GitHub unreachable")},
        {"main_sha": od.NetworkError("GitHub unreachable")},
        {"capture_names": od.AuthUnavailable("HTTP 401")},
    ])
    def test_failures_before_dispatching_leave_no_record(self, env, failure):
        ctx = env.ctx(WED_1205, fail=failure)
        with pytest.raises((od.NetworkError, od.AuthUnavailable)):
            od.run(ctx, "dispatch")
        assert env.gh.dispatched == [] and _ledger(env) == []
        ctx2 = env.ctx(WED_1205 + timedelta(minutes=1))              # next trigger proceeds
        ctx2.wait_for_run = timedelta(0)
        od.run(ctx2, "recover")
        assert len(env.gh.dispatched) == 1


# ----------------------------------------------------------- local state --

class TestLedger:
    @pytest.mark.parametrize("content", [
        "{not json", "[]", '{"dispatches": [{"at": "yesterday"}], "missed_logged": []}',
        '{"dispatches": {}, "missed_logged": []}'])
    def test_corrupt_ledger_is_quarantined_and_nothing_dispatched(self, env, content):
        (env.state / "ledger.json").write_text(content, encoding="utf-8")
        ctx = env.ctx(WED_1205)
        assert od.run(ctx, "dispatch") == od.EXIT_ATTENTION
        assert env.gh.dispatched == [] and outcomes(env)[-1] == "state_corrupt"
        [moved] = env.state.glob("ledger.corrupt-*.json")
        assert moved.read_text(encoding="utf-8") == content          # kept for inspection
        assert not (env.state / "ledger.json").exists()
        ctx2 = env.ctx(WED_1205 + timedelta(minutes=1))              # next run starts afresh
        ctx2.wait_for_run = timedelta(0)
        od.run(ctx2, "dispatch")
        assert len(env.gh.dispatched) == 1

    def test_corrupt_ledger_outside_window_is_visible(self, env):
        (env.state / "ledger.json").write_text("{", encoding="utf-8")
        ctx = env.ctx(WED_CLOSE + timedelta(minutes=30))
        assert od.run(ctx, "recover") == od.EXIT_ATTENTION
        assert outcomes(env)[-1] == "state_corrupt" and env.gh.dispatched == []

    def test_unreadable_ledger_is_left_in_place(self, env, monkeypatch):
        (env.state / "ledger.json").write_text("{}", encoding="utf-8")
        real = od.Path.read_text

        def read_text(self, *a, **k):
            if self.name == "ledger.json":
                raise PermissionError("locked")
            return real(self, *a, **k)
        monkeypatch.setattr(od.Path, "read_text", read_text)
        with pytest.raises(od.StateCorrupt, match="left in place"):
            od.load_ledger(env.state)
        monkeypatch.undo()
        assert (env.state / "ledger.json").read_text(encoding="utf-8") == "{}"

    def test_save_is_atomic_and_flushed(self, env, monkeypatch):
        od.save_ledger(env.state, {"dispatches": [], "missed_logged": ["old"]})
        synced = []
        real_fsync = od.os.fsync
        monkeypatch.setattr(od.os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd)))

        def broken_replace(src, dst):
            raise OSError("disk full")
        monkeypatch.setattr(od.os, "replace", broken_replace)
        with pytest.raises(OSError):
            od.save_ledger(env.state, {"dispatches": [], "missed_logged": ["new"]})
        monkeypatch.undo()
        assert synced                                                # flushed before rename
        assert od.load_ledger(env.state)["missed_logged"] == ["old"]  # old ledger intact
        assert not list(env.state.glob("*.tmp"))                     # no debris


# ------------------------------------------------- one main revision --

class TestMainRevision:
    def test_captures_read_at_one_pinned_commit(self, env):
        env.store.capture(WED_SLOT, wed_events())
        ctx = env.ctx(WED_SLOT + timedelta(minutes=30))
        assert od.run(ctx, "dispatch") == od.EXIT_OK
        assert env.gh.refs and set(env.gh.refs) == {MAIN_SHA}
        assert env.logged()[1][-1]["main_sha"] == MAIN_SHA

    def test_outcome_rereads_main_after_the_run(self, env):
        ctx = env.ctx(WED_1205)
        sim = _simulate(env)

        def hook(gh, what):
            sim(gh, what)
            if gh.run_list and gh.run_list[-1]["status"] == "completed":
                gh.sha = "d" * 40                                    # the capture's commit
        env.gh.on_dispatch = hook
        assert od.run(ctx, "dispatch") == od.EXIT_OK
        assert env.gh.refs[-1] == "d" * 40 and outcomes(env)[-1] == "capture_persisted"

    def test_real_client_uses_the_sha_for_listing_and_documents(self, env):
        env.store.capture(WED_SLOT, wed_events())
        [path] = env.store.capture_dir.glob("*.json")
        urls = []

        def opener(req, timeout):
            urls.append(req.full_url)
            if req.full_url.endswith("/git/ref/heads/main"):
                body = json.dumps({"object": {"sha": MAIN_SHA}}).encode()
            elif "/contents/" in req.full_url and path.name in req.full_url:
                body = path.read_bytes()
            elif "/contents/" in req.full_url:
                body = json.dumps([{"name": path.name, "type": "file"}]).encode()
            else:
                body = json.dumps({"workflow_runs": []}).encode()
            return TestHttpLayer()._resp(200, body)
        gh = od.GitHub(TOKEN, opener=opener, sleep=lambda s: None)
        st = od.slot_state(gh, on.slot_from_id(WED))
        assert st.captured and st.captured["file"] == path.name and st.ref == MAIN_SHA
        contents = [u for u in urls if "/contents/" in u]
        assert len(contents) == 2 and all(u.endswith(f"?ref={MAIN_SHA}") for u in contents)

    def test_malformed_sha_is_refused(self):
        def opener(req, timeout):
            return TestHttpLayer()._resp(200, b'{"object": {"sha": "main"}}')
        with pytest.raises(od.NetworkError):
            od.GitHub(TOKEN, opener=opener, sleep=lambda s: None).main_sha()

    def test_validator_differing_from_main_is_warned(self, env):
        ctx = env.ctx(WED_1205, blobs={"ontario_spreads.py": "0" * 40})
        ctx.wait_for_run = timedelta(0)
        od.run(ctx, "dispatch")
        [warn] = [r for r in env.logged()[1] if r["outcome"] == "validator_differs_from_main"]
        assert warn["files"] == ["ontario_spreads.py"] and warn["main_sha"] == MAIN_SHA

    def test_validator_matching_main_is_silent(self, env):
        ctx = env.ctx(WED_1205)
        ctx.wait_for_run = timedelta(0)
        od.run(ctx, "dispatch")
        assert "validator_differs_from_main" not in outcomes(env)

    def test_local_blob_sha_matches_git(self):
        if not shutil.which("git"):
            pytest.skip("git not available")
        for rel in od.VALIDATOR_FILES:
            r = subprocess.run(["git", "hash-object", rel], cwd=ROOT, capture_output=True,
                               text=True)
            assert r.returncode == 0 and r.stdout.strip() == od.local_blob_sha(ROOT / rel)


# ---------------------------------------------------- POST semantics --

def _http_error(code):
    return lambda req: urllib.error.HTTPError(req.full_url, code, "x", {}, io.BytesIO(b""))


class TestDispatchPost:
    def _gh(self, exc=None):
        attempts, sleeps = [], []

        def opener(req, timeout):
            attempts.append(req.get_method())
            if exc is not None:
                raise exc(req) if callable(exc) and not isinstance(exc, BaseException) else exc
            return TestHttpLayer()._resp(204, b"")
        return od.GitHub(TOKEN, opener=opener, sleep=sleeps.append), attempts, sleeps

    @pytest.mark.parametrize("exc", [
        TimeoutError("timed out"),
        urllib.error.URLError(TimeoutError("timed out")),
        ConnectionResetError("reset by peer"),
        _http_error(502),
    ], ids=["timeout", "urlerror-timeout", "reset", "http-502"])
    def test_unknown_outcome_is_uncertain_and_sent_once(self, exc):
        gh, attempts, sleeps = self._gh(exc)
        with pytest.raises(od.DispatchUncertain) as info:
            gh.dispatch("wednesday_noon")
        assert attempts == ["POST"] and sleeps == []
        assert TOKEN not in str(info.value)

    @pytest.mark.parametrize("exc", [
        urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed")),
        urllib.error.URLError(ConnectionRefusedError("refused")),
        _http_error(429),
    ], ids=["dns", "refused", "http-429"])
    def test_request_that_never_reached_github_is_a_network_error(self, exc):
        gh, attempts, _ = self._gh(exc)
        with pytest.raises(od.NetworkError) as info:
            gh.dispatch("wednesday_noon")
        assert attempts == ["POST"] and info.value.status is None

    def test_refusal_carries_its_status(self):
        gh, attempts, _ = self._gh(_http_error(422))
        with pytest.raises(od.NetworkError) as info:
            gh.dispatch("wednesday_noon")
        assert info.value.status == 422 and attempts == ["POST"]

    def test_reads_are_still_retried(self):
        gh, attempts, sleeps = self._gh(_http_error(502))
        with pytest.raises(od.NetworkError):
            gh.runs()
        assert attempts == ["GET"] * od.HTTP_ATTEMPTS and len(sleeps) == od.HTTP_ATTEMPTS - 1


# --------------------------------------------------------------- runtime --

class TestRuntime:
    def test_reports_the_running_interpreter(self):
        info = od.runtime_info()
        assert info["executable"] == sys.executable
        assert info["version"].startswith("%d.%d." % sys.version_info[:2])
        assert set(info["dependencies"]) == set(od.DEPENDENCIES)

    def test_store_python_detected_through_a_venv(self, monkeypatch):
        # A venv's own python.exe is an ordinary path; only its base reveals
        # that it was made from the Microsoft Store Python.
        monkeypatch.setattr(sys, "executable", r"C:\repo\venv-dispatch\Scripts\python.exe")
        monkeypatch.setattr(sys, "_base_executable",
                            r"C:\Users\u\AppData\Local\Microsoft\WindowsApps\PythonSoftware"
                            r"Foundation.Python.3.13_qbz5n2kfra8p0\python.exe", raising=False)
        monkeypatch.setattr(sys, "base_prefix", r"C:\Users\u\AppData\Local\Programs\Python\X")
        info = od.runtime_info()
        assert info["store_python"] and not info["ok"]
        assert any("Store" in p for p in info["problems"])

    def test_python_org_base_is_accepted(self, monkeypatch):
        monkeypatch.setattr(sys, "executable", r"C:\repo\venv-dispatch\Scripts\python.exe")
        monkeypatch.setattr(sys, "_base_executable",
                            r"C:\Users\u\AppData\Local\Programs\Python\Python312\python.exe",
                            raising=False)
        monkeypatch.setattr(sys, "base_prefix",
                            r"C:\Users\u\AppData\Local\Programs\Python\Python312")
        assert not od.runtime_info()["store_python"]

    @pytest.mark.parametrize("version,ok", [((3, 11, 9), False), ((3, 12, 0), True),
                                            ((3, 14, 7), True), ((3, 15, 0), False)])
    def test_supported_versions(self, monkeypatch, version, ok):
        monkeypatch.setattr(sys, "version_info", version)
        assert od.runtime_info()["supported_version"] is ok

    def test_missing_dependency_is_a_problem(self, monkeypatch):
        monkeypatch.setattr(od, "DEPENDENCIES", od.DEPENDENCIES + ("no_such_module_xyz",))
        info = od.runtime_info()
        assert info["dependencies"]["no_such_module_xyz"] is None and not info["ok"]

    def test_runtime_command_exit_code(self, monkeypatch, capsys):
        monkeypatch.setattr(od, "runtime_info", lambda: {"ok": False, "problems": ["x"]})
        assert od.main(["runtime"]) == od.EXIT_CONFIG
        assert json.loads(capsys.readouterr().out)["problems"] == ["x"]


# ------------------------------------------------- leftover dispatcher lock --

HOST = "THIS-PC"


def _leave_lock(state, started, pid=999_999, host=HOST, raw=None, token="t0", drop=()):
    state.mkdir(parents=True, exist_ok=True)
    record = {"what": "dispatcher run", "id": "dispatch-1", "pid": pid, "host": host,
              "started_at": on.iso(started), "token": token}
    for key in drop:
        record.pop(key)
    path = state / ".dispatcher.lock"
    path.write_text(raw if raw is not None else json.dumps(record), encoding="utf-8")
    return path


class TestLeftoverLock:
    NOW = WED_1205
    LONG_UP = WED_1205 - timedelta(days=2)                       # booted two days ago

    def _acquire(self, env, boot=LONG_UP, alive=lambda p: True, release=True):
        lock, code = od.acquire_lock(env.state, "dispatch", env.log, now=self.NOW, boot=boot,
                                     alive=alive, host=HOST)
        if lock is not None and release:
            lock.__exit__(None, None, None)
        return lock, code

    def _kept(self, env):
        return [r for r in env.logged()[1] if r["outcome"] == "another_instance_running"][-1]

    # -- taken over: trustworthy evidence the holder can't still run --

    def test_lock_from_before_the_last_boot_is_taken_over(self, env):
        # e.g. power lost mid-run: no process from before the boot exists.
        _leave_lock(env.state, self.NOW - timedelta(minutes=10))
        lock, _ = self._acquire(env, boot=self.NOW - timedelta(minutes=3))
        assert lock is not None and outcomes(env)[-1] == "stale_lock_removed"

    def test_lock_of_a_process_that_no_longer_exists_is_taken_over(self, env):
        # Task Scheduler hard-terminated the run at its time limit.
        _leave_lock(env.state, self.NOW - timedelta(minutes=21))
        lock, _ = self._acquire(env, alive=lambda p: False)
        assert lock is not None
        assert "no longer exists" in env.logged()[1][-1]["reason"]

    # -- kept: no trustworthy evidence --

    def test_live_process_lock_is_kept_and_reported_when_old(self, env):
        path = _leave_lock(env.state, self.NOW - timedelta(minutes=45))
        lock, code = self._acquire(env, alive=lambda p: True)
        assert lock is None and code == od.EXIT_ATTENTION and path.exists()
        rec = self._kept(env)
        assert "Get-Process -Id 999999" in rec["note"] and "process IDs are reused" in rec["note"]

    def test_recent_live_lock_is_respected_quietly(self, env):
        path = _leave_lock(env.state, self.NOW - timedelta(minutes=5))
        lock, code = self._acquire(env, alive=lambda p: True)
        assert lock is None and code == od.EXIT_OK and path.exists()

    def test_unknown_liveness_is_kept(self, env):
        path = _leave_lock(env.state, self.NOW - timedelta(minutes=45))
        lock, code = self._acquire(env, alive=lambda p: None)
        assert lock is None and code == od.EXIT_ATTENTION and path.exists()
        assert "can't be checked" in self._kept(env)["note"]

    @pytest.mark.parametrize("raw,drop", [
        ("", ()),                                        # killed mid-write
        ("{not json", ()),
        ('"just a string"', ()),
        (None, ("pid",)), (None, ("host",)), (None, ("token",)),
        (None, ("started_at",)), (None, ("what",)),
        (json.dumps({"what": "capture", "id": "x", "pid": 4, "host": HOST,
                     "started_at": "2026-10-01T00:00:00Z", "token": "t"}), ()),  # not ours
    ])
    def test_unknown_owner_is_never_removed_even_if_old_and_before_boot(self, env, raw, drop):
        path = _leave_lock(env.state, self.NOW - timedelta(days=3), raw=raw, drop=drop)
        old = (self.NOW - timedelta(days=3)).timestamp()
        os.utime(path, (old, old))
        lock, code = self._acquire(env, boot=self.NOW - timedelta(minutes=5),
                                   alive=lambda p: False)
        assert lock is None and code == od.EXIT_ATTENTION and path.exists()
        rec = self._kept(env)
        assert rec["owner_known"] is False and "never removed automatically" in rec["note"]
        assert "Investigate" in rec["note"] and str(path) in rec["note"]

    def test_unknown_owner_is_never_judged_stale(self):
        # First layer (the removal step re-reads the owner as a second layer).
        for boot in (None, self.NOW):
            assert od.stale_lock_reason(None, self.NOW, boot, lambda p: False, HOST) is None

    def test_unknown_owner_while_recent_is_quiet(self, env):
        _leave_lock(env.state, self.NOW, raw="")          # another run mid-write, just now
        os.utime(env.state / ".dispatcher.lock", (self.NOW.timestamp(),) * 2)
        lock, code = self._acquire(env, alive=lambda p: False)
        assert lock is None and code == od.EXIT_OK

    def test_lock_from_another_host_is_kept(self, env):
        # Same state folder seen from another computer (e.g. synced): its PID
        # and boot time say nothing about that computer.
        path = _leave_lock(env.state, self.NOW - timedelta(days=1), host="OTHER-PC")
        lock, code = self._acquire(env, boot=self.NOW - timedelta(minutes=5),
                                   alive=lambda p: False)
        assert lock is None and code == od.EXIT_ATTENTION and path.exists()
        assert "host OTHER-PC, not this one" in self._kept(env)["note"]

    # -- concurrent recovery --

    def test_lock_replaced_before_removal_is_kept(self, env, monkeypatch):
        # Run A judged the old lock stale; before A removes it, run B has
        # already replaced it with its own live lock. A must not remove B's.
        _leave_lock(env.state, self.NOW - timedelta(minutes=1), pid=os.getpid(), token="live")
        stale = {"what": "dispatcher run", "id": "old", "pid": 999_999, "host": HOST,
                 "started_at": on.iso(self.NOW - timedelta(hours=3)), "token": "t0"}
        real, calls = od.read_lock, []

        def read_lock(path):
            calls.append(path)
            return (dict(stale), None) if len(calls) == 1 else real(path)
        monkeypatch.setattr(od, "read_lock", read_lock)
        lock, code = self._acquire(env, alive=lambda p: False)
        assert lock is None
        assert json.loads((env.state / ".dispatcher.lock").read_text())["token"] == "live"
        assert "stale_lock_removed" not in outcomes(env)

    def test_recovery_in_progress_elsewhere_blocks_removal(self, env):
        path = _leave_lock(env.state, self.NOW - timedelta(minutes=10))
        with on.SlotLock(env.state, "dispatcher-recovery", "other", what="dispatcher lock recovery"):
            lock, code = self._acquire(env, boot=self.NOW - timedelta(minutes=3))
        assert lock is None and code == od.EXIT_OK and path.exists()
        assert json.loads(path.read_text())["token"] == "t0"

    def test_concurrent_recoveries_yield_exactly_one_holder(self, env):
        import threading
        for round_ in range(15):
            for p in env.state.glob(".dispatcher*.lock"):
                p.unlink()
            _leave_lock(env.state, self.NOW - timedelta(minutes=10), token=f"old{round_}")
            barrier, results = threading.Barrier(6), []

            def contender():
                barrier.wait()
                results.append(od.acquire_lock(env.state, "dispatch", env.log, now=self.NOW,
                                               boot=self.NOW - timedelta(minutes=3),
                                               alive=lambda p: True, host=HOST)[0])
            threads = [threading.Thread(target=contender) for _ in range(6)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            held = [lk for lk in results if lk is not None]
            assert len(held) == 1, f"round {round_}: {len(held)} holders"
            on_disk = json.loads((env.state / ".dispatcher.lock").read_text())["token"]
            assert on_disk == held[0].token                  # the winner's lock survived
            held[0].__exit__(None, None, None)
            assert not (env.state / ".dispatcher-recovery.lock").exists()

    def test_pid_probe(self):
        assert od._pid_alive(os.getpid()) is True
        assert od._pid_alive(None) is None


@pytest.mark.parametrize("policy,sids,expected", [
    # Administrators holds the right (the Windows default).
    (["SeBatchLogonRight = *S-1-5-32-544,*S-1-5-32-551,*S-1-5-32-559"],
     ["S-1-5-21-1-2-3-1001", "S-1-5-32-544", "S-1-5-3"], "Granted"),
    # Policy removed it from Administrators: membership alone is not enough.
    (["SeBatchLogonRight = *S-1-5-32-551"], ["S-1-5-21-1-2-3-1001", "S-1-5-32-544"], "NotGranted"),
    # A deny entry wins over a grant.
    (["SeBatchLogonRight = *S-1-5-32-544", "SeDenyBatchLogonRight = *S-1-5-21-1-2-3-1001"],
     ["S-1-5-21-1-2-3-1001", "S-1-5-32-544"], "Denied"),
    # No entry at all.
    (["SeServiceLogonRight = *S-1-5-80-0"], ["S-1-5-21-1-2-3-1001"], "NotGranted"),
])
def test_batch_logon_right_decision(policy, sids, expected):
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if not ps:
        pytest.skip("PowerShell not available")
    common = ROOT / "scripts" / "windows" / "OntarioDispatch.Common.ps1"
    quote = lambda items: "@(" + ",".join(f"'{i}'" for i in items) + ")"  # noqa: E731
    cmd = (f". '{common}'; (Test-BatchLogonRightFromPolicy -PolicyLines {quote(policy)} "
           f"-Sids {quote(sids)}).Status")
    r = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-Command", cmd],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == expected


def test_auth_ok_logged_without_the_token(env, monkeypatch):
    monkeypatch.setitem(od.__dict__, "_RealContext", od.Context)
    TestAuthNetworkRedaction()._main(env, monkeypatch)
    text, records = env.logged()
    [ok] = [r for r in records if r["outcome"] == "auth_ok"]
    assert ok["source"] == "credman" and "session_id" in ok and TOKEN not in text
