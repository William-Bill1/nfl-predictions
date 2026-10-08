"""Windows Task Scheduler dispatcher for the Ontario Spread Capture workflow.

GitHub's cron for .github/workflows/ontario-spread-capture.yml has run hours
late in this fork, after the slot windows had closed. This script is run by
Windows Task Scheduler on this computer near each slot and asks GitHub to run
the EXISTING workflow on main for that slot (workflow_dispatch, input
`slot=wednesday_noon` or `slot=sunday_morning`). The workflow itself still
does the capture, its budget check, validation and persistence. GitHub cron
stays configured as a backup. See docs/ONTARIO_DISPATCH_WINDOWS.md.

Rules (all in America/Toronto, using ontario_spreads' own slot windows):
* dispatch only inside a slot window (Wed 12:00-15:00, Sun 09:00-11:00), and
  not in its last DISPATCH_CUTOFF (the run must start inside the window);
* never when main already has a validated usable capture for the slot, when
  a workflow run is queued or running, or after MAX_DISPATCHES per slot;
* outside a window, a missed slot is logged (once), never captured ad hoc;
* "dispatch accepted" (HTTP 204), "run queued", "run failed" and "capture
  persisted" (a validated usable capture for the slot on main) are reported
  as different outcomes. Only the last one is success.
* the dispatch POST is never retried automatically: a timeout or 5xx is
  "dispatch uncertain", reconciled from GitHub's run list. Dispatch is at most
  MAX_REQUESTS requests per slot from this computer, not exactly once; a
  duplicate run finds the slot captured and makes no API call.

This script never calls The Odds API and keeps ODDS_API_KEY out of its
environment. It reads a GitHub token from Windows Credential Manager (or,
opt-in, Git Credential Manager); the token is only ever placed in an
Authorization header and is redacted from every log line.

Commands:
  python scripts/ontario_dispatch.py run --mode dispatch|status|recover [--dry-run]
  python scripts/ontario_dispatch.py check
  python scripts/ontario_dispatch.py runtime
  python scripts/ontario_dispatch.py task-xml --task dispatch|status|recovery ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import logging.handlers
import os
import platform
import re
import secrets
import subprocess
import sys
import time as _time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The dispatcher never needs ODDS_API_KEY, but ontario_spreads (and modules it
# imports) call python-dotenv's load_dotenv() on import, which would copy the
# key from the repository's .env into this process. dotenv never overrides a
# variable that is already set, so an empty placeholder keeps it out; a key
# inherited from the environment is dropped and registered for redaction.
_INHERITED_ODDS_KEY = os.environ.pop("ODDS_API_KEY", None)
os.environ["ODDS_API_KEY"] = ""
try:
    import ontario_spreads as on  # noqa: E402
except ImportError as _exc:      # a missing dependency; `runtime` reports which
    on, _IMPORT_ERROR = None, _exc
else:
    _IMPORT_ERROR = None

OWNER, REPO, REF = "William-Bill1", "nfl-predictions", "main"
WORKFLOW = "ontario-spread-capture.yml"
CAPTURE_PATH = "data_files/ontario_spreads/captures"
API = "https://api.github.com"
CRED_TARGET = "nfl-predictions/ontario-dispatch"      # Windows Credential Manager target
# Not under %LOCALAPPDATA%: Microsoft Store (MSIX) Python silently redirects
# AppData writes into its package folder, which would hide the logs.
STATE_DIR = Path.home() / ".nfl-predictions" / "ontario-dispatch"

MAX_DISPATCHES = 2                         # per slot, counted from GitHub's run list
MAX_REQUESTS = 3                           # per slot, dispatch POSTs sent (local ledger)
DISPATCH_CUTOFF = timedelta(minutes=15)    # no dispatch this close to the window's end
CAPTURE_SLACK = timedelta(minutes=10)      # capture files named up to this after close
IN_FLIGHT = timedelta(minutes=3)           # local dispatch not yet visible in the API
STALE_LOCK = timedelta(minutes=30)         # longer than any task's time limit (PT20M)
SUPPORTED_PYTHON = ((3, 12), (3, 14))      # python.org CPython, inclusive
DEPENDENCIES = ("pandas", "numpy", "requests", "dotenv", "tzdata")
# The capture validator runs from this checkout; these are the files it uses.
VALIDATOR_FILES = ("ontario_spreads.py", "pregame_snapshots.py")
MISSED_LOOKBACK = timedelta(days=7)
ACTIVE_STATUSES = {"queued", "in_progress", "waiting", "requested", "pending"}
# A run followed after a dispatch is matched by creation time only.
CANDIDATE = "candidate: earliest workflow_dispatch run created since the request; not proven to be its run"
# Ledger results of a dispatch request that may have created a run.
MAYBE_RAN = {"sending", "accepted", "uncertain"}
HTTP_TIMEOUT = 20
HTTP_ATTEMPTS = 3

EXIT_OK, EXIT_ATTENTION, EXIT_AUTH, EXIT_NETWORK, EXIT_CONFIG = 0, 1, 2, 3, 4

# Task Scheduler (local time; the installer refuses unless the computer's
# timezone is Eastern with DST, verified by check_local_timezone()).
TASKS = {
    "dispatch": {"name": "Ontario Spread Dispatch", "mode": "dispatch",
                 "times": {"wednesday_noon": ["12:05", "12:30"],
                           "sunday_morning": ["09:05", "09:30"]},
                 "limit": "PT20M", "wake": True},
    "status": {"name": "Ontario Spread Dispatch Status", "mode": "status",
               "times": {"wednesday_noon": ["13:00"], "sunday_morning": ["10:00"]},
               "limit": "PT10M", "wake": True},
    "recovery": {"name": "Ontario Spread Dispatch Recovery", "mode": "recover",
                 "times": {}, "limit": "PT20M", "wake": False},
}
TASK_FOLDER = "\\NFL Predictions\\"


class AuthUnavailable(RuntimeError):
    """No usable GitHub token (missing, unreadable, rejected or expired)."""


class NetworkError(RuntimeError):
    """GitHub couldn't be reached or answered with an error."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class DispatchUncertain(RuntimeError):
    """The dispatch POST got no definite answer (timeout, dropped connection,
    5xx): GitHub may or may not have accepted it. Reconciled from the run
    list; never re-sent in the same run."""


class StateCorrupt(RuntimeError):
    """The local ledger couldn't be read; it was moved aside, not trusted."""


# --------------------------------------------------------------------------
# Secret redaction
# --------------------------------------------------------------------------

_TOKEN_PATTERNS = [
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer|token)?\s*\S+"),
    re.compile(r"(?i)(\"?password\"?\s*[:=]\s*)\S+"),
]


class Redactor:
    """Removes known secrets and token-shaped strings from any text."""

    def __init__(self):
        self.secrets: set[str] = set()

    def add(self, secret: str | None) -> None:
        if secret and len(secret) >= 8:
            self.secrets.add(secret)

    def __call__(self, text) -> str:
        text = str(text)
        for s in self.secrets:
            text = text.replace(s, "[REDACTED]")
        for pat in _TOKEN_PATTERNS:
            text = pat.sub(lambda m: (m.group(1) if m.groups() else "") + "[REDACTED]", text)
        return text


REDACT = Redactor()
REDACT.add(_INHERITED_ODDS_KEY)


class _RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = REDACT(record.getMessage())
        record.args = None
        return True


def setup_logging(state_dir: Path, verbose: bool = True) -> logging.Logger:
    log = logging.getLogger("ontario_dispatch")
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False
    state_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(state_dir / "dispatch.log", maxBytes=1_000_000,
                                              backupCount=5, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    handlers = [fh]
    if verbose:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        handlers.append(sh)
    for h in handlers:
        h.addFilter(_RedactingFilter())
        log.addHandler(h)
    return log


def event(log: logging.Logger, outcome: str, level=logging.INFO, **fields) -> None:
    """One structured, redacted log line per outcome."""
    log.log(level, json.dumps({"outcome": outcome, **fields}, default=str, sort_keys=True))


# --------------------------------------------------------------------------
# Authentication (never logged, never on a command line)
# --------------------------------------------------------------------------

def token_from_credential_manager(target: str = CRED_TARGET) -> str:
    """Generic credential from Windows Credential Manager (CredReadW)."""
    if os.name != "nt":
        raise AuthUnavailable("Windows Credential Manager is only available on Windows")
    import ctypes
    from ctypes import wintypes

    class CREDENTIAL(ctypes.Structure):
        _fields_ = [("Flags", wintypes.DWORD), ("Type", wintypes.DWORD),
                    ("TargetName", wintypes.LPWSTR), ("Comment", wintypes.LPWSTR),
                    ("LastWritten", wintypes.FILETIME), ("CredentialBlobSize", wintypes.DWORD),
                    ("CredentialBlob", ctypes.POINTER(ctypes.c_char)),
                    ("Persist", wintypes.DWORD), ("AttributeCount", wintypes.DWORD),
                    ("Attributes", ctypes.c_void_p), ("TargetAlias", wintypes.LPWSTR),
                    ("UserName", wintypes.LPWSTR)]

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    pcred = ctypes.POINTER(CREDENTIAL)()
    if not advapi.CredReadW(target, 1, 0, ctypes.byref(pcred)):        # CRED_TYPE_GENERIC
        raise AuthUnavailable(f"no Windows Credential Manager entry {target!r} "
                              f"(error {ctypes.get_last_error()}); run "
                              "scripts/windows/Set-OntarioDispatchToken.ps1")
    try:
        blob = ctypes.string_at(pcred.contents.CredentialBlob, pcred.contents.CredentialBlobSize)
    finally:
        advapi.CredFree(pcred)
    token = blob.decode("utf-16-le").strip()
    if not token:
        raise AuthUnavailable(f"Windows Credential Manager entry {target!r} is empty")
    return token


def token_from_git_credential_manager() -> str:
    """Opt-in fallback: the token Git Credential Manager holds for github.com.
    Broader scope than a fine-grained token; see the docs."""
    try:
        r = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                           capture_output=True, text=True, timeout=30,
                           env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"})
    except (OSError, subprocess.SubprocessError) as exc:
        raise AuthUnavailable(f"git credential fill failed: {type(exc).__name__}") from None
    fields = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
    token = fields.get("password", "")
    if r.returncode != 0 or not token:
        raise AuthUnavailable("Git Credential Manager returned no GitHub token")
    return token


def load_token(source: str) -> str:
    token = (token_from_credential_manager() if source == "credman" else
             token_from_git_credential_manager())
    REDACT.add(token)
    return token


# --------------------------------------------------------------------------
# GitHub (bounded, read mostly; one POST to dispatch)
# --------------------------------------------------------------------------

def _not_sent(exc: BaseException) -> bool:
    """True only for transport errors that prove the request never reached
    GitHub (name resolution failed, or the connection was refused)."""
    import socket
    reason = getattr(exc, "reason", exc)
    return isinstance(reason, (socket.gaierror, ConnectionRefusedError))


class GitHub:
    """Minimal GitHub REST client. Every request has a timeout. Reads are
    retried a bounded number of times; the dispatch POST is sent at most once
    per call, because a timed-out or failed POST may still have been accepted.
    Errors carry no token (the redactor also scrubs them)."""

    def __init__(self, token: str, owner: str = OWNER, repo: str = REPO, opener=None,
                 sleep=_time.sleep):
        self._token = token
        self.base = f"{API}/repos/{owner}/{repo}"
        self._open = opener or urllib.request.urlopen
        self._sleep = sleep

    def _request(self, method: str, path: str, body: dict | None = None,
                 accept: str = "application/vnd.github+json", retry: bool = True):
        url = path if path.startswith("http") else self.base + path
        data = json.dumps(body).encode() if body is not None else None
        attempts = HTTP_ATTEMPTS if retry else 1
        last = None
        for attempt in range(attempts):
            req = urllib.request.Request(url, data=data, method=method, headers={
                "Authorization": f"Bearer {self._token}", "Accept": accept,
                "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "nfl-predictions-dispatcher",
                **({"Content-Type": "application/json"} if data else {})})
            try:
                with self._open(req, timeout=HTTP_TIMEOUT) as resp:
                    return resp.status, resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403):
                    raise AuthUnavailable(f"GitHub rejected the token (HTTP {exc.code}) for "
                                          f"{method} {path}: expired, revoked or missing "
                                          "permission") from None
                if exc.code < 500 and exc.code != 429:
                    raise NetworkError(f"GitHub HTTP {exc.code} for {method} {path}",
                                       status=exc.code) from None
                if not retry and exc.code >= 500:
                    raise DispatchUncertain(f"GitHub answered HTTP {exc.code} to {method} "
                                            f"{path}; it may still have been processed") from None
                last = f"HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"
                if not retry and not _not_sent(exc):
                    raise DispatchUncertain(f"no answer to {method} {path} ({REDACT(last)}); "
                                            "it may still have been accepted") from None
            if attempt + 1 < attempts:
                self._sleep(2 ** attempt * 2)
        raise NetworkError(f"GitHub unreachable for {method} {path} after {attempts} "
                           f"attempt{'s' if attempts > 1 else ''} ({REDACT(last)})")

    def get_json(self, path: str):
        _, raw = self._request("GET", path)
        return json.loads(raw)

    def main_sha(self) -> str:
        """The commit main points at now. Captures and the validator comparison
        are all read at this one revision."""
        sha = self.get_json(f"/git/ref/heads/{REF}")["object"]["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise NetworkError(f"unexpected commit id for {REF}")
        return sha

    def capture_names(self, ref: str = REF) -> list[str]:
        try:
            items = self.get_json(f"/contents/{CAPTURE_PATH}?ref={ref}")
        except NetworkError as exc:
            if exc.status == 404:
                return []                                      # no captures yet
            raise
        return [i["name"] for i in items if i.get("type") == "file" and i["name"].endswith(".json")]

    def capture_doc(self, name: str, ref: str = REF) -> dict:
        _, raw = self._request("GET", f"/contents/{CAPTURE_PATH}/{name}?ref={ref}",
                               accept="application/vnd.github.raw+json")
        return json.loads(raw)

    def blob_sha(self, path: str, ref: str) -> str | None:
        try:
            return self.get_json(f"/contents/{path}?ref={ref}")["sha"]
        except NetworkError as exc:
            if exc.status == 404:
                return None
            raise

    def runs(self, **params) -> list[dict]:
        q = "&".join(f"{k}={v}" for k, v in {"branch": REF, "per_page": 50, **params}.items())
        return self.get_json(f"/actions/workflows/{WORKFLOW}/runs?{q}")["workflow_runs"]

    def dispatch(self, slot_name: str) -> int:
        """One POST, never retried here. Raises DispatchUncertain when the
        outcome is unknown, NetworkError when it certainly wasn't accepted."""
        status, _ = self._request("POST", f"/actions/workflows/{WORKFLOW}/dispatches",
                                  body={"ref": REF, "inputs": {"slot": slot_name}}, retry=False)
        return status


# --------------------------------------------------------------------------
# Time and slots (ontario_spreads' own rules)
# --------------------------------------------------------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def check_local_timezone(now: datetime, localtime=_time.localtime) -> list[str]:
    """Problems if the computer's timezone doesn't follow America/Toronto,
    checked at now and either side of both DST changes of the season. Task
    triggers are in local time, so a mismatch makes them fire at the wrong
    Toronto time (the dispatch decision itself always uses Toronto)."""
    y = now.astimezone(on.TORONTO).year
    probes = [now]
    for year in (y, y + 1):
        for month, day in ((3, 8), (11, 1)):               # search each DST change
            for d in range(day, day + 7):
                for h in (5, 6, 7, 8):
                    probes.append(datetime(year, month, d, h, 30, tzinfo=timezone.utc))
    problems = []
    for t in probes:
        os_off = localtime(t.timestamp()).tm_gmtoff
        tor_off = int(t.astimezone(on.TORONTO).utcoffset().total_seconds())
        if os_off != tor_off:
            problems.append(f"{on.iso(t)}: computer UTC{os_off / 3600:+g}, "
                            f"Toronto UTC{tor_off / 3600:+g}")
    return problems


def slot_window(slot: dict) -> tuple[datetime, datetime]:
    s = on.slot_from_id(slot["slot_id"])
    return on.parse_utc(s["opens_utc"]), on.parse_utc(s["closes_utc"])


def recent_closed_slots(now: datetime) -> list[dict]:
    """Scheduled slots whose window closed within MISSED_LOOKBACK (and on or
    after the tracking start), newest first."""
    out = []
    for rule in on.SLOTS:
        intended = on.slot_intended(rule, now)
        for _ in range(2):
            day = intended.astimezone(on.TORONTO).date()
            slot = on.slot_for(rule.name, day)
            closes = on.parse_utc(slot["closes_utc"])
            if closes < now and now - closes <= MISSED_LOOKBACK and day >= on.TRACKING_START:
                out.append(slot)
            intended -= timedelta(days=7)
    return sorted(out, key=lambda s: s["closes_utc"], reverse=True)


# --------------------------------------------------------------------------
# GitHub state for one slot
# --------------------------------------------------------------------------

@dataclass
class SlotState:
    captured: dict | None = None          # validated usable capture on main
    unusable: list = field(default_factory=list)
    invalid: list = field(default_factory=list)
    active_runs: list = field(default_factory=list)
    window_runs: list = field(default_factory=list)   # runs created inside the window
    ref: str | None = None                # the main commit the captures were read at


def _name_time(name: str) -> datetime | None:
    m = re.match(r"^(\d{8}T\d{6}Z)-[0-9a-f]{12}\.json$", name)
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc) if m \
        else None


def slot_state(gh: GitHub, slot: dict) -> SlotState:
    """The slot as GitHub sees it now. The capture listing and every capture
    document are read at one main commit, resolved first, so a push in
    between can't mix two revisions."""
    opens, closes = slot_window(slot)
    st = SlotState(ref=gh.main_sha())
    for name in sorted(gh.capture_names(st.ref)):
        t = _name_time(name)
        if t is None or not (opens - timedelta(minutes=1) <= t <= closes + CAPTURE_SLACK):
            continue
        doc = gh.capture_doc(name, st.ref)
        try:
            on.validate_capture(doc, name)
        except (on.ValidationError, KeyError, TypeError, ValueError) as exc:
            st.invalid.append(f"{name}: {exc}")
            continue
        if doc["slot"]["slot_id"] != slot["slot_id"] or doc["slot"]["name"] != slot["name"]:
            continue                                             # e.g. an ad-hoc capture
        if doc["usable"] and st.captured is None:
            st.captured = {"file": name, "run_id": doc["run_id"],
                           "captured_at": doc["captured_at"], "status": doc["slot"]["status"]}
        elif not doc["usable"]:
            st.unusable.append(name)
    for r in gh.runs():
        created = on.parse_provider_time(r["created_at"])
        if r["status"] in ACTIVE_STATUSES:
            st.active_runs.append(r)
        if opens <= created <= closes:
            st.window_runs.append(r)
    return st


# --------------------------------------------------------------------------
# Local ledger (in-flight dispatches, missed slots already logged)
# --------------------------------------------------------------------------

def _ledger_ok(ledger) -> bool:
    try:
        return (isinstance(ledger, dict) and isinstance(ledger.get("missed_logged"), list)
                and isinstance(ledger.get("dispatches"), list)
                and all(isinstance(d, dict) and isinstance(d["slot_id"], str)
                        and on.parse_utc(d["at"]) for d in ledger["dispatches"]))
    except (KeyError, TypeError, ValueError):
        return False


def load_ledger(state_dir: Path) -> dict:
    """The ledger, or an empty one if there is none yet. A corrupt ledger is
    never silently replaced: it is moved aside to ledger.corrupt-<UTC>.json
    and StateCorrupt is raised, so this run dispatches nothing."""
    path = state_dir / "ledger.json"
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {"dispatches": [], "missed_logged": []}
    except OSError as exc:
        raise StateCorrupt(f"ledger {path} couldn't be read ({type(exc).__name__}); "
                           "left in place") from None
    try:
        ledger = json.loads(raw)
    except ValueError:
        ledger = None
    if not _ledger_ok(ledger):
        dest = state_dir / f"ledger.corrupt-{utc_now():%Y%m%dT%H%M%S%fZ}.json"
        os.replace(path, dest)
        raise StateCorrupt(f"ledger {path} was corrupt; moved to {dest.name}")
    return ledger


def save_ledger(state_dir: Path, ledger: dict) -> None:
    """Atomic: a unique temporary file, flushed to disk, then renamed over the
    ledger. A crash leaves either the old or the new ledger, never half."""
    tmp = state_dir / f"ledger.json.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(ledger, f, indent=1, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, state_dir / "ledger.json")
    finally:
        tmp.unlink(missing_ok=True)


def local_blob_sha(path: Path) -> str:
    """Git's blob id for a working-tree file, with CRLF normalised to LF as
    core.autocrlf stores it."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def check_validator(ctx: "Context", ref: str) -> list[str]:
    """Captures are validated with this checkout's code. Warn if the files it
    uses differ from the main commit the captures were read at."""
    differs = [rel for rel in VALIDATOR_FILES
               if ctx.gh.blob_sha(rel, ref) != local_blob_sha(ROOT / rel)]
    if differs:
        event(ctx.log, "validator_differs_from_main", logging.WARNING, files=differs,
              main_sha=ref, checkout=str(ROOT),
              note="captures are validated with this checkout's code; update it to main "
                   "(git pull) so the validation matches")
    return differs


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------

@dataclass
class Context:
    gh: GitHub | None
    log: logging.Logger
    state_dir: Path
    now: callable = utc_now
    sleep: callable = _time.sleep
    dry_run: bool = False
    wait_for_run: timedelta = timedelta(minutes=12)
    poll: timedelta = timedelta(seconds=30)


def _run_brief(r: dict) -> dict:
    return {"id": r["id"], "event": r["event"], "status": r["status"],
            "conclusion": r.get("conclusion"), "created_at": r["created_at"],
            "url": r.get("html_url")}


def handle_outside_window(ctx: Context, now: datetime) -> int:
    """Log each recently closed slot without a usable capture as missed -
    once - and never dispatch an ad-hoc capture."""
    ledger = load_ledger(ctx.state_dir)
    code = EXIT_OK
    pending = [s for s in recent_closed_slots(now) if s["slot_id"] not in ledger["missed_logged"]]
    if not pending:
        event(ctx.log, "outside_window", now=on.iso(now))
        return EXIT_OK
    for slot in pending:
        st = slot_state(ctx.gh, slot)
        if st.captured:
            ledger["missed_logged"].append(slot["slot_id"])
            continue
        check_validator(ctx, st.ref)
        event(ctx.log, "missed_opportunity", logging.WARNING, slot_id=slot["slot_id"],
              main_sha=st.ref,
              window_closed=slot["closes_utc"], unusable_captures=st.unusable,
              runs_in_window=[_run_brief(r) for r in st.window_runs],
              note="window closed without a usable capture; no ad-hoc capture is dispatched")
        ledger["missed_logged"].append(slot["slot_id"])
        code = EXIT_ATTENTION
    if not ctx.dry_run:
        save_ledger(ctx.state_dir, ledger)
    return code


def classify(ctx: Context, slot: dict, run_id: int | None) -> tuple[str, dict]:
    """Outcome for a slot, from GitHub: capture_persisted is the only success."""
    st = slot_state(ctx.gh, slot)
    if st.captured:
        return "capture_persisted", {"capture": st.captured}
    if st.invalid:
        return "capture_invalid", {"invalid": st.invalid}
    run = next((r for r in st.window_runs if r["id"] == run_id), None) if run_id else None
    if run is None and st.window_runs:
        run = max(st.window_runs, key=lambda r: r["created_at"])
    if run is None:
        return "no_run", {}
    brief = _run_brief(run)
    if run["status"] in ACTIVE_STATUSES:
        return "run_queued" if run["status"] != "in_progress" else "run_in_progress", \
            {"run": brief}
    if run.get("conclusion") != "success":
        return "run_failed", {"run": brief, "unusable_captures": st.unusable}
    return "run_succeeded_without_capture", {"run": brief, "unusable_captures": st.unusable,
                                             "note": "e.g. not_due, empty or Ontario-less"}


def handle_in_window(ctx: Context, slot: dict, now: datetime, mode: str) -> int:
    opens, closes = slot_window(slot)
    st = slot_state(ctx.gh, slot)
    base = {"slot_id": slot["slot_id"], "now": on.iso(now), "window_closes": on.iso(closes),
            "main_sha": st.ref}
    check_validator(ctx, st.ref)
    if st.invalid:
        event(ctx.log, "capture_invalid", logging.ERROR, invalid=st.invalid, **base,
              note="a capture on main fails validation; not dispatching - investigate")
        return EXIT_ATTENTION
    if st.captured:
        event(ctx.log, "capture_persisted", capture=st.captured, **base)
        return EXIT_OK
    if mode == "status":
        outcome, info = classify(ctx, slot, None)
        level = logging.INFO if outcome in ("run_queued", "run_in_progress") else logging.WARNING
        event(ctx.log, "status_" + outcome, level, **info, **base)
        return EXIT_OK if outcome in ("run_queued", "run_in_progress") else EXIT_ATTENTION
    if st.active_runs:
        event(ctx.log, "run_active_not_dispatching",
              runs=[_run_brief(r) for r in st.active_runs], **base)
        return EXIT_OK
    ledger = load_ledger(ctx.state_dir)
    mine = [d for d in ledger["dispatches"] if d["slot_id"] == slot["slot_id"]]
    # Requests that may have created a run. Definite rejections (nothing
    # created) neither block a retry nor count toward the request ceiling.
    maybe_ran = [d for d in mine if d.get("result", "accepted") in MAYBE_RAN
                 or d.get("result", "").startswith("http_")]
    recent = [d for d in maybe_ran if now - on.parse_utc(d["at"]) < IN_FLIGHT]
    if recent:
        event(ctx.log, "dispatch_in_flight_not_dispatching", recent=recent, **base)
        return EXIT_OK
    dispatched = [r for r in st.window_runs if r["event"] == "workflow_dispatch"]
    if len(dispatched) >= MAX_DISPATCHES:
        event(ctx.log, "dispatch_limit_reached", logging.WARNING, max=MAX_DISPATCHES,
              runs=[_run_brief(r) for r in dispatched], **base)
        return EXIT_ATTENTION
    if len(maybe_ran) >= MAX_REQUESTS:
        event(ctx.log, "dispatch_request_ceiling_reached", logging.WARNING, max=MAX_REQUESTS,
              requests=maybe_ran, runs=[_run_brief(r) for r in dispatched],
              note="this computer already sent this many dispatch requests for the slot "
                   "(some got no definite answer); not sending more", **base)
        return EXIT_ATTENTION
    if now > closes - DISPATCH_CUTOFF:
        event(ctx.log, "too_close_to_window_close", logging.WARNING,
              cutoff_minutes=DISPATCH_CUTOFF.total_seconds() / 60, **base)
        return EXIT_ATTENTION
    reason = "recovery" if mode == "recover" else ("retry" if dispatched or st.window_runs
                                                   else "scheduled")
    # The GitHub reads above can take minutes (retries, slow answers, a
    # suspend). Decide again on the clock as it is now, immediately before the
    # request is recorded and sent: the same slot must still be due, and the
    # cutoff not yet reached.
    at = ctx.now()
    timing = {"checks_started_at": on.iso(now), "checked_at": on.iso(at)}
    current = on.resolve_slot(at, "auto")
    if current is None or current["slot_id"] != slot["slot_id"]:
        event(ctx.log, "window_closed_during_checks", logging.WARNING, **timing,
              note="the slot window closed while GitHub was being checked; not dispatching",
              **base)
        return EXIT_ATTENTION
    if at > closes - DISPATCH_CUTOFF:
        event(ctx.log, "too_close_to_window_close", logging.WARNING, **timing,
              cutoff_minutes=DISPATCH_CUTOFF.total_seconds() / 60,
              note="the cutoff passed while GitHub was being checked", **base)
        return EXIT_ATTENTION
    if ctx.dry_run:
        event(ctx.log, "dry_run_would_dispatch", reason=reason, slot=slot["name"], **base)
        return EXIT_OK
    # Recorded before the POST, so a crash or kill mid-request still counts as
    # a request that may have created a run.
    entry ={"slot_id": slot["slot_id"], "at": on.iso(at), "reason": reason, "result": "sending"}
    ledger["dispatches"].append(entry)
    save_ledger(ctx.state_dir, ledger)
    try:
        status = ctx.gh.dispatch(slot["name"])
    except DispatchUncertain as exc:
        entry["result"] = "uncertain"
        save_ledger(ctx.state_dir, ledger)
        event(ctx.log, "dispatch_uncertain", logging.WARNING, error=str(exc), reason=reason,
              note="not re-sent; reconciling from GitHub's run list", **base)
        return follow_run(ctx, slot, at, base, uncertain=True)
    except NetworkError as exc:
        entry["result"] = f"rejected_http_{exc.status}" if exc.status else "not_accepted"
        save_ledger(ctx.state_dir, ledger)
        if exc.status is None:
            raise                                     # GitHub unreachable: network_error
        event(ctx.log, "dispatch_rejected", logging.ERROR, http_status=exc.status,
              error=str(exc), note="GitHub refused the request; no run was created", **base)
        return EXIT_ATTENTION
    except AuthUnavailable:
        entry["result"] = "rejected_auth"
        save_ledger(ctx.state_dir, ledger)
        raise
    entry["result"] = "accepted" if status == 204 else f"http_{status}"
    save_ledger(ctx.state_dir, ledger)
    if status != 204:
        event(ctx.log, "dispatch_unexpected_response", logging.ERROR, http_status=status, **base)
        return EXIT_ATTENTION
    event(ctx.log, "dispatch_accepted", http_status=status, reason=reason, slot=slot["name"],
          note="accepted is not captured; following the run", **base)
    return follow_run(ctx, slot, at, base)


def follow_run(ctx: Context, slot: dict, dispatched_at: datetime, base: dict,
               uncertain: bool = False) -> int:
    """Bounded wait: find a run for the request, then wait for its outcome.

    GitHub's dispatch API returns no run ID, so the run followed is a
    *candidate*: the earliest workflow_dispatch run created since the request.
    Nothing establishes that this request created it (a manual dispatch, or a
    late run of an earlier uncertain request, looks the same), and outcomes
    say so. After an uncertain request, no candidate within IN_FLIGHT means
    the request most likely wasn't accepted; it is never re-sent here."""
    deadline = dispatched_at + ctx.wait_for_run
    find_by = dispatched_at + min(IN_FLIGHT, ctx.wait_for_run) if uncertain else deadline
    run_id, seen_queued = None, False
    while True:
        runs = [r for r in ctx.gh.runs(event="workflow_dispatch")
                if on.parse_provider_time(r["created_at"]) >= dispatched_at - timedelta(seconds=30)]
        if runs and run_id is None:
            run_id = min(runs, key=lambda r: r["created_at"])["id"]
            if uncertain:
                event(ctx.log, "dispatch_candidate_run", logging.WARNING, run=_run_brief(
                    next(r for r in runs if r["id"] == run_id)), run_identity=CANDIDATE,
                      note="a workflow_dispatch run was created after the uncertain request. "
                           "It may be this request's run, or another dispatch's; GitHub "
                           "doesn't say. Following it as a candidate", **base)
        run = next((r for r in runs if r["id"] == run_id), None)
        if run is None and uncertain and ctx.now() >= find_by:
            event(ctx.log, "dispatch_uncertain_no_run", logging.WARNING,
                  waited_minutes=(find_by - dispatched_at).total_seconds() / 60,
                  note="no candidate run appeared, so GitHub most likely did not accept the "
                       "request, though a later run can't be ruled out. It is not re-sent now; "
                       "a later trigger inside the window may dispatch again (only runs GitHub "
                       "lists count toward the limit)", **base)
            return EXIT_ATTENTION
        if run and run["status"] in ACTIVE_STATUSES and not seen_queued:
            event(ctx.log, "run_queued", run=_run_brief(run), run_identity=CANDIDATE, **base)
            seen_queued = True
        if run and run["status"] == "completed":
            outcome, info = classify(ctx, slot, run_id)
            ok = outcome == "capture_persisted"
            if "run" in info:
                info["run_identity"] = CANDIDATE
            event(ctx.log, outcome, logging.INFO if ok else logging.ERROR, **info, **base)
            return EXIT_OK if ok else EXIT_ATTENTION
        if ctx.now() >= deadline:
            outcome = "run_not_found" if run is None else "run_still_running"
            event(ctx.log, outcome, logging.WARNING,
                  run=_run_brief(run) if run else None,
                  run_identity=CANDIDATE if run else None,
                  waited_minutes=ctx.wait_for_run.total_seconds() / 60,
                  note="the status task reports the final outcome", **base)
            return EXIT_ATTENTION if run is None else EXIT_OK
        ctx.sleep(ctx.poll.total_seconds())


def run(ctx: Context, mode: str) -> int:
    now = ctx.now()
    problems = check_local_timezone(now)
    if problems:
        event(ctx.log, "timezone_mismatch", logging.ERROR, problems=problems[:3],
              note="task triggers fire at the wrong Toronto time; decisions still use Toronto")
    slot = on.resolve_slot(now, "auto")
    try:
        if slot is None:
            # Outside every window (the status task too: it then reports the
            # slot that just closed). Never dispatches.
            return handle_outside_window(ctx, now)
        return handle_in_window(ctx, slot, now, mode)
    except StateCorrupt as exc:
        event(ctx.log, "state_corrupt", logging.ERROR, error=str(exc),
              note="nothing dispatched in this run; the next run starts a new ledger (the "
                   "limit of 2 is counted from GitHub's run list, not the ledger)")
        return EXIT_ATTENTION


# --------------------------------------------------------------------------
# Runtime (which interpreter actually runs, and with what)
# --------------------------------------------------------------------------

_STORE_RE = re.compile(r"\\WindowsApps\\|PythonSoftwareFoundation\.Python\.", re.I)


def runtime_info() -> dict:
    """The interpreter behind sys.executable - for a venv, the base install
    it was created from - its version, and the dispatcher's dependencies."""
    import importlib.metadata
    import importlib.util
    base_exe = getattr(sys, "_base_executable", sys.executable)
    seen = [sys.executable, base_exe, os.path.realpath(base_exe), sys.base_prefix,
            os.path.realpath(sys.base_prefix)]
    deps = {}
    for name in DEPENDENCIES:
        dist = {"dotenv": "python-dotenv"}.get(name, name)
        try:
            deps[name] = importlib.metadata.version(dist) if importlib.util.find_spec(name) \
                else None
        except importlib.metadata.PackageNotFoundError:
            deps[name] = None
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo("America/Toronto")
        toronto = True
    except Exception:                                           # noqa: BLE001
        toronto = False
    version = sys.version_info[:3]
    supported = SUPPORTED_PYTHON[0] <= version[:2] <= SUPPORTED_PYTHON[1]
    cpython = sys.implementation.name == "cpython"
    store = any(_STORE_RE.search(p or "") for p in seen)
    problems = []
    if not cpython:
        problems.append(f"{sys.implementation.name} is not CPython")
    if not supported:
        problems.append("Python %d.%d is not supported (3.12-3.14)" % version[:2])
    if store:
        problems.append("Microsoft Store Python (directly or as the venv's base)")
    problems += [f"missing dependency {n}" for n, v in deps.items() if v is None]
    if not toronto:
        problems.append("zoneinfo has no America/Toronto (install tzdata)")
    if _IMPORT_ERROR is not None:
        problems.append(f"ontario_spreads import failed: {_IMPORT_ERROR}")
    return {"executable": sys.executable, "base_executable": base_exe,
            "base_executable_resolved": os.path.realpath(base_exe),
            "base_prefix": sys.base_prefix, "venv": sys.prefix != sys.base_prefix,
            "version": ".".join(map(str, version)), "implementation": sys.implementation.name,
            "bits": 64 if sys.maxsize > 2 ** 32 else 32, "store_python": store,
            "supported_version": supported, "dependencies": deps, "toronto_tz": toronto,
            "problems": problems, "ok": not problems}


# --------------------------------------------------------------------------
# The dispatcher's own single-instance lock
# --------------------------------------------------------------------------

def _boot_time() -> datetime | None:
    if os.name != "nt":
        return None
    import ctypes
    tick = ctypes.windll.kernel32.GetTickCount64
    tick.restype = ctypes.c_uint64
    return utc_now() - timedelta(milliseconds=tick())


def _pid_alive(pid) -> bool | None:
    """True/False if it can be told whether process `pid` runs; None if not."""
    if not isinstance(pid, int) or pid <= 0:
        return None
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except OSError:
            return None
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = k32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False if ctypes.get_last_error() == 87 else None   # 87: no such process
    try:
        code = ctypes.c_ulong()
        if not k32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return None
        return code.value == 259                                  # STILL_ACTIVE
    finally:
        k32.CloseHandle(handle)


def read_lock(path: Path) -> tuple[dict | None, datetime | None]:
    """(owner, file time) for a lock file. `owner` is the recorded holder
    only when the record is complete and well-formed - purpose, run id,
    process ID, host, start time and token; otherwise None (unknown owner:
    unreadable, empty, half-written or foreign)."""
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    except OSError:
        mtime = None
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
        ok = (isinstance(info, dict) and info.get("what") == "dispatcher run"
              and isinstance(info.get("id"), str) and info["id"]
              and isinstance(info.get("pid"), int) and info["pid"] > 0
              and isinstance(info.get("host"), str) and info["host"]
              and isinstance(info.get("token"), str) and info["token"]
              and on.parse_utc(info["started_at"]) is not None)
    except (OSError, ValueError, TypeError, KeyError):
        ok = False
    return (info if ok else None), mtime


def stale_lock_reason(owner: dict | None, now: datetime, boot: datetime | None, alive,
                      host: str) -> str | None:
    """Why a dispatcher lock left behind can be taken over, or None.

    A task killed at its time limit, or a power loss, never releases its lock.
    It is taken over only on trustworthy evidence that the recorded holder
    can't still be running: the record is complete, it was made on this
    computer, and either it predates this computer's last start or the
    recorded process no longer exists. A lock with no trustworthy owner, from
    another host, or whose process may still run is never removed."""
    if owner is None or owner["host"] != host:
        return None
    started = on.parse_utc(owner["started_at"])
    if boot is not None and started < boot - timedelta(minutes=1):
        return "recorded on this computer before it last started"
    if alive(owner["pid"]) is False:
        return f"recorded process {owner['pid']} on this computer no longer exists"
    return None


def _kept_lock_note(path: Path, owner: dict | None, host: str, alive) -> str:
    """What to check before deleting a lock that is kept."""
    running = ("Get-CimInstance Win32_Process | Where-Object CommandLine -like "
               "'*ontario_dispatch.py*'")
    if owner is None:
        return (f"Lock file {path} has no trustworthy owner record (unreadable, empty, "
                "incomplete or not a dispatcher lock), so it is never removed automatically. "
                f"Investigate: check that no dispatcher is running ({running}) and that no "
                "task in \\NFL Predictions\\ is Running (Get-ScheduledTask -TaskPath "
                f"'\\NFL Predictions\\'); then delete {path}.")
    if owner["host"] != host:
        return (f"Lock file {path} was taken on host {owner['host']}, not this one "
                f"({host}); its process can't be checked from here. Confirm that process "
                f"{owner['pid']} on {owner['host']} is no longer running, then delete {path}.")
    state = {True: "is still running", None: "can't be checked"}.get(alive(owner["pid"]),
                                                                     "is gone")
    return (f"Lock file {path} is held by process {owner['pid']} on this computer, which "
            f"{state} (run {owner['id']}, started {owner['started_at']}). If that process isn't "
            f"the dispatcher (Get-Process -Id {owner['pid']}; process IDs are reused) and no "
            f"dispatcher is running ({running}), delete {path}.")


def acquire_lock(state_dir: Path, mode: str, log: logging.Logger, now=None, boot=None,
                 alive=_pid_alive, host=None):
    """The single-instance lock shared by all three tasks, or (None, exit code).

    A lock left behind is removed only under a second, short-lived recovery
    lock, and only if it still carries the token that was judged stale, so two
    runs recovering at once can't remove each other's fresh lock."""
    host = host or platform.node()
    lock = on.SlotLock(state_dir, "dispatcher", f"{mode}-{os.getpid()}", what="dispatcher run")
    for attempt in (1, 2):
        try:
            return lock.__enter__(), None
        except on.CaptureError:
            pass
        now_ = now or utc_now()
        owner, mtime = read_lock(lock.path)
        reason = stale_lock_reason(owner, now_, boot if boot is not None else _boot_time(),
                                   alive, host) if attempt == 1 else None
        if reason is None:
            started = on.parse_utc(owner["started_at"]) if owner else mtime
            old = started is None or now_ - started > STALE_LOCK
            event(log, "another_instance_running", logging.WARNING if old else logging.INFO,
                  holder=owner and {k: owner[k] for k in ("id", "pid", "host", "started_at")},
                  owner_known=owner is not None, lock=str(lock.path),
                  note=_kept_lock_note(lock.path, owner, host, alive) if old else
                  "another dispatcher run holds the lock; it normally finishes within minutes")
            return None, EXIT_ATTENTION if old else EXIT_OK
        try:
            with on.SlotLock(state_dir, "dispatcher-recovery", f"{mode}-{os.getpid()}",
                             what="dispatcher lock recovery"):
                again, _ = read_lock(lock.path)
                if again is None or again["token"] != owner["token"]:
                    continue                      # replaced meanwhile: never removed
                lock.path.unlink()
        except on.CaptureError:
            event(log, "another_instance_running", holder=None, lock=str(lock.path),
                  note="another run is recovering the dispatcher lock")
            return None, EXIT_OK
        except OSError as exc:
            event(log, "another_instance_running", logging.WARNING, lock=str(lock.path),
                  note=f"a lock left behind couldn't be removed ({type(exc).__name__}); "
                       + _kept_lock_note(lock.path, owner, host, alive))
            return None, EXIT_ATTENTION
        event(log, "stale_lock_removed", logging.WARNING, reason=reason,
              holder={k: owner[k] for k in ("id", "pid", "host", "started_at")},
              lock=str(lock.path))
    return None, EXIT_OK                                 # pragma: no cover - loop returns


def _session_id() -> int | None:
    """Windows session of this process (0 = no interactive desktop, as for a
    task that runs while the user is signed out)."""
    if os.name != "nt":
        return None
    import ctypes
    sid = ctypes.c_ulong()
    ok = ctypes.windll.kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(sid))
    return sid.value if ok else None


# --------------------------------------------------------------------------
# Task Scheduler XML (registered by scripts/windows/Install-OntarioDispatch.ps1)
# --------------------------------------------------------------------------

_DAY = {"wednesday_noon": "Wednesday", "sunday_morning": "Sunday"}


def task_xml(task: str, python: str, repo_root: str, user_id: str,
             first_date: str = "2026-10-11") -> str:
    """Task Scheduler 1.4 XML for one task. Calendar triggers are local time:
    correct only when the computer's timezone follows America/Toronto."""
    spec = TASKS[task]
    triggers = []
    for slot_name, times in spec["times"].items():
        for hhmm in times:
            triggers.append(
                "    <CalendarTrigger>\n"
                f"      <StartBoundary>{first_date}T{hhmm}:00</StartBoundary>\n"
                "      <Enabled>true</Enabled>\n"
                "      <ScheduleByWeek>\n"
                f"        <DaysOfWeek><{_DAY[slot_name]} /></DaysOfWeek>\n"
                "        <WeeksInterval>1</WeeksInterval>\n"
                "      </ScheduleByWeek>\n"
                "    </CalendarTrigger>")
    if task == "recovery":
        triggers.append("    <BootTrigger>\n      <Enabled>true</Enabled>\n"
                        "      <Delay>PT3M</Delay>\n    </BootTrigger>")
        triggers.append(
            "    <EventTrigger>\n      <Enabled>true</Enabled>\n      <Delay>PT2M</Delay>\n"
            "      <Subscription>&lt;QueryList&gt;&lt;Query Id=\"0\" Path=\"System\"&gt;"
            "&lt;Select Path=\"System\"&gt;*[System[Provider[@Name='Microsoft-Windows-Power-"
            "Troubleshooter'] and EventID=1]]&lt;/Select&gt;&lt;/Query&gt;&lt;/QueryList&gt;"
            "</Subscription>\n    </EventTrigger>")
        # Waking from Modern Standby (S0 low-power idle) doesn't log the
        # Power-Troubleshooter event above; Kernel-Power 507 ("exiting Modern
        # Standby") is logged instead.
        triggers.append(
            "    <EventTrigger>\n      <Enabled>true</Enabled>\n      <Delay>PT2M</Delay>\n"
            "      <Subscription>&lt;QueryList&gt;&lt;Query Id=\"0\" Path=\"System\"&gt;"
            "&lt;Select Path=\"System\"&gt;*[System[Provider[@Name='Microsoft-Windows-Kernel-"
            "Power'] and EventID=507]]&lt;/Select&gt;&lt;/Query&gt;&lt;/QueryList&gt;"
            "</Subscription>\n    </EventTrigger>")
        # A network connection (re)established. A trigger that fired while
        # offline didn't start (network condition), so this is its retry.
        log_ = "Microsoft-Windows-NetworkProfile/Operational"
        triggers.append(
            "    <EventTrigger>\n      <Enabled>true</Enabled>\n      <Delay>PT1M</Delay>\n"
            f"      <Subscription>&lt;QueryList&gt;&lt;Query Id=\"0\" Path=\"{log_}\"&gt;"
            f"&lt;Select Path=\"{log_}\"&gt;*[System[Provider[@Name='Microsoft-Windows-"
            "NetworkProfile'] and EventID=10000]]&lt;/Select&gt;&lt;/Query&gt;&lt;/QueryList&gt;"
            "</Subscription>\n    </EventTrigger>")
    args = f'"{repo_root}\\scripts\\ontario_dispatch.py" run --mode {spec["mode"]}'
    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Author>{escape(user_id)}</Author>
    <Description>Dispatches the Ontario Spread Capture GitHub workflow ({escape(spec["mode"])}). See docs/ONTARIO_DISPATCH_WINDOWS.md.</Description>
  </RegistrationInfo>
  <Triggers>
{chr(10).join(triggers)}
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>{escape(user_id)}</UserId>
      <LogonType>Password</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <DisallowStartOnRemoteAppSession>false</DisallowStartOnRemoteAppSession>
    <UseUnifiedSchedulingEngine>true</UseUnifiedSchedulingEngine>
    <WakeToRun>{str(spec["wake"]).lower()}</WakeToRun>
    <ExecutionTimeLimit>{spec["limit"]}</ExecutionTimeLimit>
    <Priority>7</Priority>
    <RestartOnFailure>
      <Interval>PT5M</Interval>
      <Count>2</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(python)}</Command>
      <Arguments>{escape(args)}</Arguments>
      <WorkingDirectory>{escape(repo_root)}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one bounded dispatcher run")
    r.add_argument("--mode", choices=["dispatch", "status", "recover"], required=True)
    r.add_argument("--dry-run", action="store_true",
                   help="read GitHub state and log decisions; never dispatch")
    r.add_argument("--auth", choices=["credman", "git"], default="credman")
    r.add_argument("--state-dir", type=Path, default=STATE_DIR)
    c = sub.add_parser("check", help="timezone, auth presence and state directory (read-only)")
    c.add_argument("--auth", choices=["credman", "git"], default="credman")
    c.add_argument("--state-dir", type=Path, default=STATE_DIR)
    x = sub.add_parser("task-xml", help="print Task Scheduler XML for one task")
    x.add_argument("--task", choices=sorted(TASKS), required=True)
    x.add_argument("--python", required=True)
    x.add_argument("--repo-root", required=True)
    x.add_argument("--user", required=True)
    sub.add_parser("runtime", help="report the interpreter and dependencies as JSON")
    args = ap.parse_args(argv)

    if args.cmd == "runtime":
        info = runtime_info()
        print(json.dumps(info, indent=1, sort_keys=True))
        return EXIT_OK if info["ok"] else EXIT_CONFIG
    if on is None:
        print(f"cannot run: {_IMPORT_ERROR} (run: python scripts/ontario_dispatch.py runtime)")
        return EXIT_CONFIG
    if args.cmd == "task-xml":
        sys.stdout.write(task_xml(args.task, args.python, args.repo_root, args.user))
        return EXIT_OK
    if args.cmd == "check":
        now = utc_now()
        problems = check_local_timezone(now)
        print(f"timezone: {'OK (follows America/Toronto incl. DST)' if not problems else 'MISMATCH'}")
        for p in problems[:5]:
            print(f"  {p}")
        try:
            load_token(args.auth)
            print(f"auth ({args.auth}): token present (not shown)")
            auth_ok = True
        except AuthUnavailable as exc:
            print(f"auth ({args.auth}): UNAVAILABLE - {REDACT(exc)}")
            auth_ok = False
        print(f"state dir: {args.state_dir} "
              f"({'exists' if args.state_dir.exists() else 'created on first run'})")
        slot = on.resolve_slot(now, "auto")
        print(f"now {on.iso(now)}: {'inside ' + slot['slot_id'] if slot else 'outside any slot window'}")
        return EXIT_CONFIG if problems else (EXIT_OK if auth_ok else EXIT_AUTH)

    log = setup_logging(args.state_dir)
    lock, code = acquire_lock(args.state_dir, args.mode, log)
    if lock is None:
        return code
    try:
        try:
            gh = GitHub(load_token(args.auth))
        except AuthUnavailable as exc:
            event(log, "auth_unavailable", logging.ERROR, error=str(exc), source=args.auth,
                  session_id=_session_id(), user=os.environ.get("USERNAME"),
                  note="nothing dispatched; GitHub cron remains the backup")
            return EXIT_AUTH
        # Evidence for the unattended test: which token source worked, in which
        # Windows session and interpreter. Says nothing about the token itself.
        event(log, "auth_ok", source=args.auth, mode=args.mode, session_id=_session_id(),
              user=os.environ.get("USERNAME"), python=sys.executable)
        ctx = Context(gh=gh, log=log, state_dir=args.state_dir, dry_run=args.dry_run)
        try:
            return run(ctx, args.mode)
        except AuthUnavailable as exc:
            event(log, "auth_rejected", logging.ERROR, error=str(exc))
            return EXIT_AUTH
        except NetworkError as exc:
            event(log, "network_error", logging.ERROR, error=str(exc),
                  note="nothing more dispatched in this run; a later trigger may retry inside "
                       "the window")
            return EXIT_NETWORK
    finally:
        lock.__exit__(None, None, None)


if __name__ == "__main__":
    raise SystemExit(main())
