# Windows dispatcher for Ontario spread captures

GitHub's cron for `.github/workflows/ontario-spread-capture.yml` has run hours
late in this fork, so scheduled runs arrived after the slot windows had
closed. On 2026-10-07 two runs arrived 4.7–5 h late and returned `not_due`.

This computer can therefore ask GitHub to run the **existing** workflow on
time, using Windows Task Scheduler and `scripts/ontario_dispatch.py`. The
workflow still does everything else:

- the Odds API request;
- the credit check;
- validation;
- the commit;
- the artifact upload.

**GitHub cron stays configured as a backup.** If both fire, the later run
finds the slot already captured and makes no API call (see
[Duplicates](#duplicates-and-what-is-not-guaranteed)).

The dispatcher only dispatches the existing workflow on `main`, with
`slot=wednesday_noon` or `slot=sunday_morning`. It never calls The Odds API.
It keeps `ODDS_API_KEY` out of its own process, even though the repository's
`.env` holds it.

Nothing is installed by default. The steps below register the tasks.

## Schedule (America/Toronto)

| Task (folder `\NFL Predictions\`) | Triggers | Does |
|---|---|---|
| Ontario Spread Dispatch | weekly: Wed 12:05 and 12:30; Sun 09:05 and 09:30 | dispatch, or retry, if needed |
| Ontario Spread Dispatch Status | weekly: Wed 13:00; Sun 10:00 | reports the outcome; never dispatches |
| Ontario Spread Dispatch Recovery | at startup (+3 min); on resume from sleep or hibernation (Power-Troubleshooter event 1, +2 min); on leaving Modern Standby (Kernel-Power event 507, +2 min); when a network connects (NetworkProfile event 10000, +1 min) | dispatches only inside a window; otherwise logs a missed slot |

**Slot windows.** These are the windows of `ontario_spreads.py`:

- **Wednesday 12:00–15:00**;
- **Sunday 09:00–11:00**;
- both ends inclusive.

Every task runs the same decision. Only the dispatch and recovery tasks may
dispatch, and only inside a window. The status task never dispatches.

**Why four recovery triggers.** On this laptop (Modern Standby), waking up
logs Kernel-Power 507, not Power-Troubleshooter 1. In the week to 2026-10-08
there were 39 Kernel-Power 507 events, and the last Power-Troubleshooter 1
was on 2026-10-04. The network trigger retries a scheduled start that was
skipped because the computer was offline.

## When it dispatches

**Before every dispatch**, the dispatcher reads GitHub and dispatches only if
all of these hold:

1. **No capture yet:** main has no validated, usable capture for the slot
   (one with an Ontario quote). Empty, US-only and ad-hoc captures don't
   count.
2. **Nothing running:** no run of the workflow is queued or running.
3. **Run limit:** GitHub shows fewer than **2** `workflow_dispatch` runs
   created inside this window.
4. **Not just sent:** this computer sent no request for the slot in the last
   3 minutes that may have created a run GitHub doesn't show yet.
5. **Request ceiling:** this computer has sent fewer than **3** requests for
   the slot that may have created a run.
6. **Enough time:** it's at least **15 minutes before the window closes**, so
   the run starts inside the window.

**The time is checked twice.** The checks above read GitHub, and that can take
minutes: retries, slow answers, or the computer suspending mid-run. So,
immediately before the request is recorded and sent, the dispatcher reads the
clock again and requires both of these:

- **Same slot still due:** the slot it checked is still the one due.
- **Before the cutoff:** at least 15 minutes remain in the window.

If the window closed meanwhile, it logs `window_closed_during_checks`. If only
the cutoff passed, it logs `too_close_to_window_close`, with
`checks_started_at` and `checked_at`. Either way nothing is sent or recorded.
Dry runs apply the same recheck.

**Outside a window** it never dispatches. In particular, it never starts an
ad-hoc capture to make up for a missed slot. A slot that closed without a
usable capture is logged once as `missed_opportunity`.

**One main revision per decision.** The dispatcher first reads the commit
`main` points at. It then lists the captures and reads every capture document
at that commit, so a push in between can't mix two revisions. When the run
finishes, it reads `main` again.

Captures are validated with this checkout's `ontario_spreads.py` and
`pregame_snapshots.py`. If either differs from `main`, it logs
`validator_differs_from_main`; then `git pull` the checkout.

### What counts toward the limits

| Event | Run limit (2, from GitHub) | Request ceiling (3, this computer) | Blocks the next 3 min |
|---|---|---|---|
| request accepted (HTTP 204), run created | yes, once GitHub lists the run | yes | yes |
| request with no definite answer (timeout, dropped connection, HTTP 5xx) | only if GitHub created a run | yes | yes |
| process killed mid-request | only if GitHub created a run | yes | yes |
| request refused (HTTP 4xx such as 422, 401/403) | no | no | no |
| request that never reached GitHub (DNS failure, connection refused, HTTP 429) | no | no | no |
| failure before any request (GitHub unreachable, token missing or rejected, timezone or runtime problem, corrupt state, another instance running) | no | no | no |
| GitHub cron runs, and dispatches before the window opened | no | no | no |

## How a dispatch request is handled

The dispatch is a single `POST`, sent **at most once per run and never
retried automatically**. Before the POST, the request is written to the
ledger as `sending`, so a crash still counts it.

| Answer | Ledger | What happens |
|---|---|---|
| HTTP 204 | `accepted` | `dispatch_accepted`, then it follows the run |
| timeout, dropped connection, HTTP 5xx | `uncertain` | `dispatch_uncertain`. It is **not re-sent**. The dispatcher looks for a `workflow_dispatch` run created since the request. If one appears within 3 minutes, it logs `dispatch_candidate_run` and follows it as a *candidate*; otherwise `dispatch_uncertain_no_run` (exit 1) |
| HTTP 4xx | `rejected_http_<code>` | `dispatch_rejected` (exit 1) |
| DNS failure, connection refused, HTTP 429 | `not_accepted` | `network_error` (exit 3) |

A later trigger inside the window decides again from GitHub's run list.

**Runs are matched by time, not identity.** GitHub's dispatch API returns no
run ID, so the run followed after any request is the earliest
`workflow_dispatch` run created since it. That run may have been started by:

- this request;
- an earlier uncertain request whose run appeared late;
- someone dispatching the workflow by hand.

Every run-related outcome after a dispatch carries
`"run_identity": "candidate: …"`. A candidate run after an uncertain request
does **not** show that the request was accepted. Only `capture_persisted`
(the capture on `main`) is evidence of the result, whichever run produced it.

### Duplicates, and what is not guaranteed

Dispatch is **not exactly-once**. Two or more runs for one slot are possible:

- **Late run:** a request with no definite answer was accepted, but its run
  appears after the 3-minute check. A later trigger then sends again.
- **Mistaken candidate:** a candidate run was actually someone else's. The
  uncertain request may still produce its own run.
- **Cron:** GitHub cron and the dispatcher both start a run.
- **By hand:** someone dispatches the workflow by hand.

**What prevents a second paid capture of a slot already on `main`:**

1. **One at a time:** the workflow's `concurrency` group runs one capture at
   a time. GitHub keeps at most one run pending, and cancels an older pending
   run when a newer one arrives.
2. **Latest `main` first:** each run on `main` first fast-forwards to the
   latest `main` ("Use the latest main"). It then sees a capture that an
   earlier run committed, and stops with `already_captured` without an API
   call.
3. **Fails closed:** if that refresh can't fetch or fast-forward `main`
   (GitHub unreachable, or `main` rewritten), the step fails and the capture
   step is skipped. No Odds API request is made from a possibly stale commit.
   Upload and the persistence report still run, and the report shows
   "latest main: failure".
4. **Recheck under the lock:** `capture()` checks the slot again while
   holding the slot lock, before the credit check.

**When a second paid capture can still happen:**

- **Push failed:** the earlier run captured, but its push to `main` failed.
  The capture then isn't on `main`, so a later run captures the slot again.
  This is a real retry, not a duplicate on `main`.
- **Commit outside this workflow:** a capture is committed to `main` after a
  run's refresh but before its capture step, by something outside the
  workflow's concurrency group. One example is a `python ontario_spreads.py
  capture` run elsewhere and pushed by hand.
- **Not on `main`:** runs on a branch other than `main` skip the refresh and
  never commit.

Each costs one more Odds API credit, and a second capture fills no extra
slot.

**Bounds per slot from this computer:**

- at most 3 requests that may have created a run;
- at most 2 runs that GitHub listed when the decision was made.

## Outcomes (log `outcome` field)

**Three different things are reported, and only the last is success:**

1. **dispatch acceptance** (`dispatch_accepted`; after an uncertain request,
   at most a `dispatch_candidate_run`);
2. **run completion** (`run_failed`, `run_succeeded_without_capture`);
3. **capture persistence** (`capture_persisted`).

| Outcome | Meaning |
|---|---|
| `auth_ok` | the token was read. Logs its source (`credman` or `git`), the Windows session ID (0 = no desktop, as when signed out), the user and the interpreter, never the token |
| `dispatch_accepted` | GitHub accepted the request (HTTP 204). **Not** a capture |
| `dispatch_uncertain` | the request got no definite answer; not re-sent |
| `dispatch_candidate_run` | a `workflow_dispatch` run was created after an uncertain request. It may or may not be that request's run; it is followed as a candidate |
| `dispatch_uncertain_no_run` | no candidate run appeared within 3 minutes; most likely not accepted, though a later run can't be ruled out |
| `dispatch_rejected` | GitHub refused the request (HTTP 4xx); no run created |
| `run_queued` / `run_in_progress` | the candidate run exists and hasn't finished |
| `run_failed` | the run finished without success |
| `run_succeeded_without_capture` | the run was green, but no usable capture is on main (e.g. `not_due`, empty or Ontario-less) |
| `capture_persisted` | **success:** a validated, usable capture for the slot is committed on main |
| `run_not_found` / `run_still_running` | the bounded wait (12 min) ended; the status task reports the final outcome |
| `capture_invalid` | a capture on main fails validation; nothing is dispatched; investigate |
| `run_active_not_dispatching`, `dispatch_in_flight_not_dispatching`, `dispatch_limit_reached`, `dispatch_request_ceiling_reached`, `too_close_to_window_close`, `window_closed_during_checks` | why nothing was dispatched |
| `missed_opportunity` | a window closed without a usable capture (logged once per slot) |
| `validator_differs_from_main` | this checkout's validator files differ from `main`; `git pull` |
| `state_corrupt` | the ledger couldn't be read; see [Local state](#local-state) |
| `another_instance_running` | the dispatcher lock is held. If it is older than 30 minutes, or has no trustworthy owner, this is a warning (exit 1) with what to check before deleting the lock |
| `stale_lock_removed` | a lock whose recorded holder can't still be running was taken over |
| `auth_unavailable` / `auth_rejected` | no token, or GitHub rejected it (expired, revoked, missing permission) |
| `network_error` | GitHub unreachable after 3 bounded attempts (reads), or the dispatch request never reached it |
| `timezone_mismatch` | the computer's timezone doesn't follow America/Toronto |

**Exit codes** (shown as "Last Run Result" in Task Scheduler):

| Code | Meaning |
|---|---|
| 0 | OK, or nothing to do |
| 1 | needs attention: not captured, failed, missed, uncertain, limit reached or corrupt state |
| 2 | authentication |
| 3 | network |
| 4 | configuration: timezone, Python runtime or a missing dependency |

## Local state

Everything is in `%USERPROFILE%\.nfl-predictions\ontario-dispatch\`. The
dispatcher creates it on the first run. `check` and `runtime` don't create
it.

**`dispatch.log`:**

- JSON lines, rotated at 1 MB × 5.
- Every line passes a redaction filter. The token, `ghp_…` / `github_pat_…`
  shapes, `Authorization` headers and any `ODDS_API_KEY` inherited from the
  environment are replaced with `[REDACTED]`.

**`ledger.json`** records the dispatch requests and the missed slots already
logged:

- **Writes are atomic:** a unique temporary file is flushed to disk, then
  renamed over the ledger.
- **A corrupt ledger is never silently replaced.** It is moved to
  `ledger.corrupt-<UTC time>.json` for inspection, `state_corrupt` is logged,
  and that run dispatches nothing (exit 1).
- **The next run starts a new ledger.** The run limit comes from GitHub, so it
  still holds, but the request ceiling and the in-flight record for that slot
  are lost.
- **An unreadable ledger** (e.g. locked) is left in place, and the run
  dispatches nothing.

**`.dispatcher.lock`** lets one dispatcher run at a time across all three
tasks:

- **Left behind** when Task Scheduler kills a run at its time limit, or when
  power is lost.
- **Taken over automatically** (`stale_lock_removed`) only on trustworthy
  evidence that the recorded holder can't still be running. All three of
  these must hold:
  - the lock holds a complete dispatcher owner record: purpose, run ID,
    process ID, host, start time and token;
  - its host is this computer;
  - either it was recorded before this computer last started, or the recorded
    process no longer exists.
- **Never removed automatically:**
  - **Unknown owner:** the record is unreadable, empty, half-written,
    incomplete or not a dispatcher lock. Age, file time and boot time don't
    change this.
  - **Another host:** the lock was taken on another host.
  - **Possibly running:** the recorded process may still run, or can't be
    checked. Process IDs are reused.
- **When kept:** it is reported as `another_instance_running`. If it is older
  than 30 minutes, or has no trustworthy owner, the report is a warning (exit
  1) with the checks to make before deleting it by hand:
  - the recorded process (`Get-Process -Id …`);
  - any running dispatcher (`Get-CimInstance Win32_Process | Where-Object
    CommandLine -like '*ontario_dispatch.py*'`);
  - the tasks' state.
- **Two runs recovering at once:** removal happens only while holding a
  second, short-lived `.dispatcher-recovery.lock`, and only if the lock still
  carries the token that was judged stale. One run can't remove another
  run's fresh lock, and exactly one ends up holding it.
- **If the recovery lock itself is left behind:** this needs a crash within
  milliseconds. It then blocks automatic recovery, never a normal start, and
  is reported the same way.

**Why not `%LOCALAPPDATA%`.** The Microsoft Store Python redirects AppData
writes into its package folder, where nobody would find them.

## Setup

### 1. Python from python.org (required)

**Supported:** CPython **3.12, 3.13 or 3.14**, 64-bit, from python.org. For
all three versions, pip resolves binary wheels for every dependency
(checked with `pip install --dry-run` for win_amd64).

**Not the Store Python.** This repository's `venv` is built on the Microsoft
Store Python. That Python starts through an app-execution alias, which isn't
reliable from a task that runs while you're signed out. The installer refuses
it, and a venv made from it, unless you pass `-AllowStorePython`.

**Dependencies:** `requirements-dispatch.txt` lists only what the dispatcher
imports: pandas, numpy, requests, python-dotenv and tzdata. Windows has no
IANA timezone database, so tzdata is required. Versions are constrained by
`requirements.txt`.

```powershell
# python.org installer, "for current user" is fine; e.g. 3.13:
& "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m venv venv-dispatch
.\venv-dispatch\Scripts\python.exe -m pip install -r requirements-dispatch.txt
.\venv-dispatch\Scripts\python.exe scripts\ontario_dispatch.py runtime
```

**`runtime`** asks the interpreter itself and prints JSON (exit 0 = usable).
For a venv, it reports the base install the venv was made from:
`sys._base_executable` and `sys.base_prefix`, not only the venv's own
`python.exe`. It reports:

- the version;
- whether it is the Store Python;
- each dependency's version;
- whether `America/Toronto` loads.

### 2. GitHub token

Create a **fine-grained personal access token** limited to repository
`William-Bill1/nfl-predictions`, with:

- **Actions: Read and write.** This dispatches the workflow and reads its
  runs.
- **Contents: Read-only.** This reads `main`'s commit and its captures.
- **Metadata: Read-only** (added automatically).

Give it an expiry (for example 90 days) and note the date. When it expires,
runs log `auth_rejected` and exit with code 2.

Store it, which prompts with hidden input:

```powershell
.\scripts\windows\Set-OntarioDispatchToken.ps1
```

The token goes into **Windows Credential Manager** as the generic credential
`nfl-predictions/ontario-dispatch`, protected for your Windows account. It is
never written to a file, a task argument, an environment variable or a log.
Remove it with `-Remove`.

The existing Git Credential Manager login (`--auth git`) also works, but its
OAuth token has much broader scope. Use it only for a one-off dry run.

### 3. Check, plan, install

```powershell
.\venv-dispatch\Scripts\python.exe scripts\ontario_dispatch.py check
.\scripts\windows\Install-OntarioDispatch.ps1 -Python "$PWD\venv-dispatch\Scripts\python.exe" -PlanOnly
# from an elevated PowerShell (Run as administrator), elevated as yourself:
.\scripts\windows\Install-OntarioDispatch.ps1 -Python "$PWD\venv-dispatch\Scripts\python.exe"
```

**What the installer checks before registering anything:**

- **Timezone:** Windows is on **Eastern Time with automatic DST**. Task times
  are local, so this is required. The dispatcher also compares the OS rules
  with America/Toronto either side of each DST change.
- **Python:** `runtime` must pass. That means a supported CPython with every
  dependency, not the Store Python directly or as a venv's base.
- **Token:** it is present.
- **"Log on as a batch job":** read from the effective local security policy
  (`secedit /export`, elevated only), which also covers policy applied by a
  domain:
  - **Granted:** your account, or one of its groups, holds
    `SeBatchLogonRight`, and none of them is denied it.
  - **Refused:** your account or one of its groups is denied it
    (`SeDenyBatchLogonRight`, which wins), or nobody grants it.
  - **Not assumed:** Administrators membership isn't taken to grant it.
    `-PlanOnly` without elevation reports **Unverified**.

**What it then does:**

- **Elevated session:** it needs one to read the user rights and to register
  tasks with a stored password. The tasks themselves run as you with **least
  privilege**.
- **Your Windows password:** it asks for it, and Task Scheduler stores it.
  This is the "run whether user is logged on or not" logon type (Password),
  intended to let the tasks run after a reboot while you're signed out, and to
  read your Credential Manager token. For a Microsoft account, enter the
  account password, not the PIN. If you change your Windows password, run the
  installer again.
- **Idempotent:** running it again replaces the tasks.

### 4. Verify, including one unattended run (required)

```powershell
.\scripts\windows\Get-OntarioDispatchStatus.ps1
.\venv-dispatch\Scripts\python.exe scripts\ontario_dispatch.py run --mode dispatch --dry-run
```

`Get-OntarioDispatchStatus.ps1` is read-only. It shows:

- the tasks, their settings, and their last and next runs;
- the timezone and the wake-timer settings;
- whether the token is present;
- the batch-logon right (needs elevation);
- the runtime of the interpreter the tasks run;
- the latest `auth_ok` lines and log lines.

A dry run reads GitHub and logs what it *would* do. It never dispatches.

**That Credential Manager can be read from the task is not yet verified.**
The token is meant to come from a task running with a stored password, while
you're signed out. Until both tests below pass, treat the dispatcher as not
working. Both are safe **outside a slot window**: recovery then dispatches
nothing.

1. **Task logon.** Outside a window, in Task Scheduler, right-click **Ontario
   Spread Dispatch Recovery → Run**. Then check that:
   - "Last Run Result" is `0x0` (or `0x1` if it logged a missed slot);
   - the log has a new `auth_ok` line with `"source": "credman"`. A
     `"session_id"` of 0 is expected for a task that runs without a desktop.

   `auth_unavailable` (exit 2) means the task's logon can't read the
   credential.
2. **Signed out, after a restart.** Outside a window, restart and **don't
   sign in** for 5 minutes. Then sign in and check for an `auth_ok` line
   timestamped about 3 minutes after the restart.

## Task settings

| Setting | Value | Effect |
|---|---|---|
| Run as | you, **whether logged on or not** (stored password, logon type Password), least privilege | runs without a desktop, also while signed out |
| Wake the computer to run | yes for dispatch and status; no for recovery | see the power limits below |
| Start when available | yes | a weekly start missed while off or asleep runs as soon as possible. The dispatcher still dispatches only inside the window, at least 15 minutes before it closes |
| Start only if network available | yes | a start while offline is skipped, not queued. The network trigger is the retry |
| Battery | start on battery: yes; stop when switching to battery: no | |
| Multiple instances | ignore new (`IgnoreNew`) | a trigger while the *same* task runs is dropped. The dispatcher lock covers the other tasks |
| Stop if it runs longer than | 20 min (dispatch, recovery), 10 min (status) | the run's own wait is bounded to 12 min |
| Restart on failure | 2 × every 5 min | Task Scheduler applies this when the task fails to run. The dispatcher's own exit codes (1–4) are results, and Task Scheduler isn't expected to restart on them (not verified) |

## Recovery: reboot, sleep, network, overlaps

**Reboot or update restart:**

- A run in progress is killed. If its request had been sent, the ledger holds
  `sending` or a later result.
- The next start takes over the lock left behind, because it predates the
  boot.
- The startup trigger runs recovery 3 minutes after boot. It dispatches only
  if the window is open with at least 15 minutes left, and the usual checks
  pass.
- A dispatch time missed while the computer was off also runs once through
  "start when available", with the same checks.
- **Signed out:** a startup while you're signed out relies on the stored
  password (see the unattended test above).

**Sleep, hibernation, Modern Standby:**

- **Leaving a sleep state** fires recovery about 2 minutes later.
- **While asleep:** a weekly start set to wake the computer may or may not
  wake it (see Limitations). If it doesn't, "start when available" runs it
  after wake-up.
- **Asleep during a run:** the run is suspended, and its bounded wait
  continues on wake-up. A dispatch request suspended mid-flight may come back
  as `dispatch_uncertain`. It is never re-sent; a later run is followed only
  as a candidate.
- **Waking up past the cutoff:** a run that wakes up after the cutoff, or
  after the window closed, sends nothing, because the clock is checked again
  just before sending.

**Network restoration:**

- **Offline at a weekly time:** the task isn't started (network condition).
  When a connection comes up, the network trigger runs recovery a minute
  later.
- **Unreachable GitHub during a run:**
  - reads are retried 3 times, then `network_error` (exit 3);
  - the dispatch request is never retried in the same run;
  - nothing in these cases uses the run limit or the request ceiling, and
    nothing blocks the next trigger.

**Overlapping triggers:**

- **Same task:** while a task runs, another trigger of that task is dropped
  (`IgnoreNew`).
- **Different tasks:** these can overlap, e.g. a resume at 12:04 followed by
  the 12:05 dispatch. The second one finds the lock, logs
  `another_instance_running` and exits 0.
- **Nothing is lost.** The run holding the lock makes the same decision, and
  the 12:30 / 09:30 retry and the status task follow.

**Retries are bounded:**

- at most 3 requests that may have created a run, per slot, from this
  computer;
- at most 2 runs within the window;
- none in the last 15 minutes of the window.

## Limitations (read before relying on it)

**Powered off, or "shut down":**

- Nothing runs, and Fast Startup makes "shut down" a hibernation.
- Windows can sometimes wake from hibernation for a wake timer, but not from
  a full power-off.
- A missed start runs at the next boot, through the startup trigger and
  "start when available". It only dispatches if the window is still open.

**Asleep:**

- This laptop has **wake timers disabled on battery** in the active power
  plan (enabled on AC).
- It uses **Modern Standby** (S0 low-power idle), where scheduled wakes are
  best-effort.
- On battery, or with the lid closed, assume it won't wake.
- Keep it plugged in on Wednesdays and Sundays, or enable wake timers on
  battery yourself. This setup doesn't change power settings.

**Updates and reboots:** this setup doesn't change Windows Update settings.
Active hours, or a pause around Wednesday noon and Sunday morning, reduce the
risk.

**Token expiry:** the fine-grained token expires. Runs then log
`auth_rejected` (exit code 2) and nothing is dispatched; GitHub cron remains
the only trigger.

**Accepted is not captured.** Only `capture_persisted` is success. If a run
fails or finishes without a capture, the 12:30 / 09:30 retry dispatches once
more, within the limits.

**Timezone:** triggers are local times. If Windows' timezone changes, the
dispatcher logs `timezone_mismatch`. Its decisions still use Toronto time,
but the triggers fire at the wrong moments until you fix the timezone.

**Checkout:** the dispatcher runs from this repository's working copy. Keep
it on an up-to-date `main`; a different validator is reported.

## Not yet verified

These can only be confirmed on this computer after installing, or on GitHub:

- **Credential Manager from the task:** that the token can be read from a
  task with logon type Password, while signed out. See the unattended test
  above.
- **Batch logon right:** whether "Log on as a batch job" is granted here.
  Without elevation, the status script reports "Unverified"; the elevated
  installer checks it.
- **Store vs python.org Python:** that the Store Python fails from such a
  task is a documented risk, not tested here. That a python.org interpreter
  works is also untested until the unattended test passes.
- **Wake-ups:** whether this Modern Standby laptop actually wakes for the
  weekly triggers.
- **Restart on failure:** that it isn't applied to non-zero exit codes.
- **Network trigger:** that network event 10000 fires on every reconnect.
- **Workflow refresh:** the "Use the latest main" step has been run by the
  tests, with real `git` and `bash`, against local repositories:
  - a fast-forward after another run's push;
  - deepening a shallow checkout;
  - failing closed when `origin` is unreachable or `main` was rewritten.

  It hasn't run on GitHub yet. Two things there are unverified: that
  `actions/checkout`'s credentials allow the fetch, and that GitHub skips the
  capture step after a failure, as its documented `success()` default says.
- **Run identity:** GitHub's dispatch API gives no run ID, so which run
  belongs to which request can't be established. Runs are reported as
  candidates.

## Uninstall

```powershell
.\scripts\windows\Uninstall-OntarioDispatch.ps1                       # tasks only
.\scripts\windows\Uninstall-OntarioDispatch.ps1 -RemoveToken -RemoveLogs
```

Both forms are safe to run repeatedly and support `-WhatIf`. GitHub's cron is
unaffected.
