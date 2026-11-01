"""
Player Props verified publication (S2), offline: publish a latest feed and a
frozen snapshot only after verifying them against a valid S1 release package.
Nothing here is wired into predict.py, the nightly workflow or production
paths: destinations must be isolated directories outside the repository.

Two sources of outputs, and what each publication proves:

* ``packaged=True`` publishes the package's own verified snapshot as both the
  latest feed and the snapshot. That is allowed only when the package's
  rehearsal summary records that its latest feed and snapshot had exactly
  those bytes. The rehearsal evidence then names these bytes, but they are
  verified afresh anyway.
* Supplied files (``latest`` and ``snapshot``) are new outputs. The package's
  rehearsal evidence says nothing about them, even when they were produced
  from the package's inputs, and the sidecar records that it doesn't apply.
  Only this publication's own verification supports them.

Every publication runs the same checks:

1. ``release_manifest.verify_package`` must pass. The package is then re-read,
   and its manifest must be the bytes that verified.
2. The prediction schedule (``nfl_schedule_<season>.csv``, which S1 doesn't
   package) must have the hash the package's rehearsal summary recorded.
3. The package's inference stats, runtime reference, schedules, training stats
   and bundle are written to a private directory. The bundle is restored with
   ``verify_bundle_dir``, and the Stage 2 gate (``check_inputs``) is re-run for
   the package's inference target.
4. Both output files are read once into memory, and every check uses those
   bytes. The rehearsal's own verifier (``rehearsal._stage_verify_outputs``)
   runs on private copies of them:
   * independent expected-row completeness;
   * per-row identity, eligibility, tier, model, raw score, clamp and label,
     probabilities and skipped inputs;
   * bundle ID, season and week;
   * snapshot == latest feed;
   * metric parity.

   The copies' hashes must still equal the captured ones. Each row's
   ``game_date`` must also be its uniquely matched schedule game's calendar
   day at UTC midnight (``YYYY-MM-DD 00:00:00+00:00``, the form predict.py
   writes), compared as recorded: no Toronto shift, no inferred kickoff time.
5. Publication writes those same captured bytes, never the source paths again.

Reused library code prints progress (some with emoji). That output is
captured while verifying, so it can't fail on a console that can't encode it
or mix with the CLI's JSON; its last lines are kept on PublicationError.

The sidecar ``player_props_predictions_week<W>_<S>.provenance.json`` links the
output hashes to the release-manifest hash, bundle, target, runtime and
candidate reference, policy and verification result. It is deterministic (no
publication time), so an identical repeat produces identical bytes.

Writing:
* Guards run before any write, on fully resolved paths, links and junctions
  included. The destination may not overlap the repository, a production path,
  a protected path, the package or any input file.
* An existing snapshot or sidecar with different content is a conflict, and
  nothing is written. Identical existing files are kept untouched.
* The files to write are staged in ``<dest>/.publish-<id>`` and installed as
  a set by ``install_set``, with an explicit commit boundary: the last move
  into place.

Failure handling, kept distinct:
* **Before the commit boundary (catchable errors):** staging, backups and the
  moves. A failure restores the complete previous set (files that didn't
  exist are removed): PublicationError, state ``rolled_back``. If a restore
  fails: state ``rollback_failed``, with the backups kept and listed in
  ``recovery_files``.
* **After the commit boundary:** every file holds its new bytes, so the
  publication has committed. Deleting backups, the staging directory or the
  lock is cleanup. A failure there is reported in ``cleanup_warnings``, with
  the paths in ``retained_files`` (state ``committed_with_cleanup_warnings``),
  and never as a rollback.
* **Crashes:** NOT atomic. A crash or power loss between the three renames can
  leave a mixed set and ``.bak`` files, and nothing is fsynced.
* **Concurrent writers:** an advisory lock file ``<dest>/.prop-publication.lock``
  (ontario_spreads.SlotLock) makes a second publication to the same
  destination refuse while one holds it. Writers that don't take the lock,
  such as predict.py, are not excluded.

verify_published has two explicit modes. ``current`` (default) verifies the
complete current set: sidecar, snapshot and latest feed. ``historical``
verifies only the frozen snapshot and its sidecar, and says the latest feed
was not verified.

CLI (offline; JSON output, ASCII-only).
* publish exits 0 when committed, 3 when committed with cleanup warnings, 1 when
  refused or rolled back, and 2 when it could not run.
* verify-published exits 0 when verified and 1 otherwise.

    python player_props/publication.py publish --package PKG --dest DIR --prediction-schedule CSV \\
        (--packaged | --latest CSV --snapshot CSV)
    python player_props/publication.py verify-published --dest DIR --season S --week W [--package PKG] \\
        [--historical]

Policy: docs/architecture.md ("Verified publication (S2)").
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import secrets
import shutil
import stat
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from player_props import rehearsal as rh  # noqa: E402
from player_props import release_manifest as rm  # noqa: E402
from player_props.aggregators import SCHEDULE_FILE, STATS_FILES  # noqa: E402
from player_props.provenance import MANIFEST_FILE as STATS_MANIFEST, TORONTO  # noqa: E402

LATEST_NAME = 'player_props_predictions.csv'
SIDECAR_KIND = 'player_props_publication'
SIDECAR_VERSION = 1
LOCK_SLOT = 'prop-publication'
MAX_OUTPUT_BYTES = 256 * 1024 * 1024
# Every column the verifier reads; a file without one is refused before checking.
REQUIRED_COLUMNS = list(dict.fromkeys(
    rh.IDENTITY_FIELDS + ['player_name', 'display_name', 'position', 'is_home', 'season', 'week', 'game_date',
                          'model_bundle', 'eligibility_group'] + rh.EXPECTED_FIELDS + list(rh.SKIPPED_INPUT_FIELDS)
    + rh.EMPTY_MARKET_FIELDS))


class PublicationError(RuntimeError):
    """The publication did not commit. ``state`` says what is on disk:
    'not_published' (nothing was written or replaced), 'rolled_back' (the
    previous set was restored) or 'rollback_failed' (``recovery_files`` names
    the backups kept). ``retained_files`` lists other files left behind;
    ``library_output`` holds the last lines reused library code printed."""

    def __init__(self, problems, recovery_files: Optional[List[str]] = None, state: str = 'not_published',
                 retained_files: Optional[List[str]] = None):
        self.problems = [str(p) for p in problems]
        self.recovery_files = list(recovery_files or [])
        self.state = state
        self.retained_files = list(retained_files or [])
        self.library_output: List[str] = []
        super().__init__("; ".join(self.problems))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot_name(season: int, week: int) -> str:
    return f'player_props_predictions_week{week}_{season}.csv'


def sidecar_name(season: int, week: int) -> str:
    return f'player_props_predictions_week{week}_{season}.provenance.json'


# ---------------------------------------------------------------------------
# Paths and inputs
# ---------------------------------------------------------------------------

def check_destination(dest: Path, package: Path, inputs: Dict[str, Optional[Path]],
                      protected: Optional[List[Path]] = None) -> Path:
    """The resolved destination, or PublicationError if it overlaps the
    repository, a production path, a protected path, the package or an input
    (equal, ancestor or descendant, after resolving links). Nothing is read or
    created."""
    real = rh._real(dest)
    for p in [ROOT] + list(rh.PRODUCTION_PATHS) + list(protected or []):
        if rh._overlaps(real, rh._real(p)):
            raise PublicationError([f"destination {dest} (resolves to {real}) overlaps protected location {p}; "
                                    f"publication destinations must be isolated from production"])
    if rh._overlaps(real, rh._real(package)):
        raise PublicationError([f"destination {dest} (resolves to {real}) overlaps the release package {package}"])
    for what, p in inputs.items():
        if p is not None and rh._overlaps(real, rh._real(p)):
            raise PublicationError([f"destination {dest} (resolves to {real}) overlaps input {what} ({p}); "
                                    f"publishing would overwrite or write into an input"])
    return real


def _capture(path: Path, what: str) -> bytes:
    """The bytes of a regular file, read once."""
    try:
        st = os.stat(path)
    except OSError as exc:
        raise PublicationError([f"{what} {path} can't be read ({type(exc).__name__}: {exc})"]) from exc
    if not stat.S_ISREG(st.st_mode):
        raise PublicationError([f"{what} {path} is not a regular file"])
    if st.st_size > MAX_OUTPUT_BYTES:
        raise PublicationError([f"{what} {path} is larger than {MAX_OUTPUT_BYTES} bytes"])
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        raise PublicationError([f"{what} {path} can't be read ({type(exc).__name__}: {exc})"]) from exc


def parse_outputs(data: bytes, label: str) -> pd.DataFrame:
    """A predictions file as the verifier reads it, or PublicationError with
    what is wrong (not CSV, empty, missing columns)."""
    if not data.strip():
        raise PublicationError([f"{label} is empty"])
    try:
        frame = rh.read_predictions(io.BytesIO(data))
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise PublicationError([f"{label} is not readable as a predictions CSV ({type(exc).__name__}: "
                                f"{str(exc)[:300]})"]) from exc
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise PublicationError([f"{label} lacks required column(s) {missing}"])
    if frame.empty:
        raise PublicationError([f"{label} has a header but no predictions"])
    return frame


# ---------------------------------------------------------------------------
# The package, prepared once
# ---------------------------------------------------------------------------

@dataclass
class Context:
    """A verified package's assets, written to a private directory, with its
    bundle restored and its inference inputs re-checked."""
    workdir: Path
    manifest: dict
    manifest_sha256: str
    contents: Dict[str, bytes]
    summary: dict
    season: int
    week: int
    schedule_sha256: str
    runtime_data: Path
    train_stats: Path
    bundle: dict
    bundle_dir: Path
    check: object


def prepare(package: Path, prediction_schedule: bytes, workdir: Path) -> Context:
    """Verify the package (S1), re-read it, link the prediction schedule to its
    evidence, and set up the private inputs the verifier reads."""
    from player_props.bundle_store import read_archive
    from player_props.features import PROP_LINES
    from player_props.model_bundle import BUNDLES_DIR, BundleError, verify_bundle_dir
    from player_props.provenance import check_inputs
    try:
        report = rm.verify_package(package)
    except rm.ReleaseError as exc:
        raise PublicationError(["the release package does not verify: " + p for p in exc.problems]) from exc
    if not report['ok']:
        raise PublicationError(["the release package does not verify: " + p for p in report['problems']])
    try:
        manifest, manifest_bytes, contents = rm.read_package(package)
    except rm.ReleaseError as exc:
        raise PublicationError(["the release package changed after it verified: " + p for p in exc.problems]) from exc
    if _sha256(manifest_bytes) != report['manifest_sha256']:
        raise PublicationError(["the release package's manifest changed after it verified"])
    # Every asset matched this (verified) manifest's hashes when re-read, so
    # these are the bytes that verified.
    target = manifest['derived']['inference_target']
    season, week = target['season'], target['week']
    summary = json.loads(contents['rehearsal_summary'].decode('utf-8'))
    sched_key = f"runtime/data/nfl_schedule_{season}.csv"
    want = (summary.get('artifacts') or {}).get(sched_key)
    got = _sha256(prediction_schedule)
    if want != got:
        raise PublicationError([f"the prediction schedule ({got[:12]}) is not the one the package's verified "
                                f"rehearsal used for {season} week {week} ({str(want)[:12]}, recorded as "
                                f"{sched_key})"])

    data, train = workdir / 'runtime' / 'data', workdir / 'train' / 'stats'
    for d in (data, train):
        d.mkdir(parents=True)
    for k, name in STATS_FILES.items():
        (data / name).write_bytes(contents[f'inference_stats_{k}'])
        (train / name).write_bytes(contents[f'training_stats_{k}'])
    (data / STATS_MANIFEST).write_bytes(contents['inference_stats_manifest'])
    (train / STATS_MANIFEST).write_bytes(contents['training_stats_manifest'])
    (data / SCHEDULE_FILE).write_bytes(contents['inference_schedule'])
    (train / SCHEDULE_FILE).write_bytes(contents['training_schedule'])
    (data / 'players.csv').write_bytes(contents['reference_runtime'])
    (data / f'nfl_schedule_{season}.csv').write_bytes(prediction_schedule)

    archive = workdir / 'bundle.tar'
    archive.write_bytes(contents['bundle_archive'])
    try:
        index, files = read_archive(archive)
        bundle_dir = workdir / 'models' / BUNDLES_DIR / index['bundle_id']
        bundle_dir.mkdir(parents=True)
        for name, blob in files.items():
            (bundle_dir / Path(name).name).write_bytes(blob)
        bundle = verify_bundle_dir(bundle_dir, PROP_LINES, index['manifest_sha256'], index['bundle_id'])
    except BundleError as exc:
        raise PublicationError(["the package's bundle can't be restored: " + p for p in exc.problems]) from exc

    finished = rm._timestamp(summary.get('finished_at'))
    check = check_inputs(season, week, data, today=finished.astimezone(TORONTO).date() if finished else None)
    if not check.ok:
        raise PublicationError([f"the package's inference inputs fail the Stage 2 gate for {season} week {week}: "
                                + p for p in check.provenance_problems + check.freshness_problems])
    return Context(workdir, manifest, _sha256(manifest_bytes), contents, summary, season, week, got,
                   data, train, bundle, bundle_dir, check)


# ---------------------------------------------------------------------------
# Verification of captured output bytes
# ---------------------------------------------------------------------------

# predict.load_schedule reads the schedule's calendar date as UTC midnight and
# writes it as-is: '2026-09-21 00:00:00+00:00'. That is the only accepted
# form; dates are compared as recorded, never shifted to Toronto time, and no
# kickoff time is inferred.
_UTC_MIDNIGHT = re.compile(r'(\d{4}-\d{2}-\d{2}) 00:00:00\+00:00')


def _examples(items, n: int = 3) -> str:
    more = f" (+{len(items) - n} more)" if len(items) > n else ""
    return "; ".join(items[:n]) + more


def game_date_problems(frame: pd.DataFrame, label: str, games: pd.DataFrame, season: int, week: int) -> List[str]:
    """Each row's game_date against its uniquely matched schedule game (by
    season, week, and home and away team from team, opponent and is_home)."""
    week_games = games[(games['season'] == season) & (games['week'] == week)]
    missing, malformed, unmatched, wrong = [], [], [], []
    for i, r in enumerate(frame.to_dict('records')):
        tag = f"row {i + 2} ({r.get('player_id')} {r.get('team')} {r.get('prop_type')})"
        value = r.get('game_date')
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            missing.append(tag)
            continue
        m = _UTC_MIDNIGHT.fullmatch(str(value))
        try:
            day = date.fromisoformat(m.group(1)) if m else None
        except ValueError:
            day = None
        if day is None:
            malformed.append(f"{tag}: {str(value)[:40]!r}")
            continue
        side = r.get('is_home')
        if not isinstance(side, (bool, np.bool_)):
            unmatched.append(f"{tag}: is_home is {side!r}, not true or false")
            continue
        home, away = (r['team'], r['opponent']) if side else (r['opponent'], r['team'])
        game = week_games[(week_games['home_team'] == home) & (week_games['away_team'] == away)]
        if len(game) != 1:
            unmatched.append(f"{tag}: {len(game)} schedule games for {away}@{home}")
            continue
        scheduled = str(game['gameday'].iloc[0])
        if day.isoformat() != scheduled:
            wrong.append(f"{tag}: {day.isoformat()} vs the schedule's {scheduled} ({game['game_id'].iloc[0]})")
    problems = []
    if missing:
        problems.append(f"{label}: {len(missing)} row(s) without a game_date: {_examples(missing)}")
    if malformed:
        problems.append(f"{label}: {len(malformed)} row(s) whose game_date isn't a calendar date at UTC midnight "
                        f"(YYYY-MM-DD 00:00:00+00:00): {_examples(malformed)}")
    if unmatched:
        problems.append(f"{label}: {len(unmatched)} row(s) without exactly one schedule game to date them by: "
                        f"{_examples(unmatched)}")
    if wrong:
        problems.append(f"{label}: {len(wrong)} row(s) whose game_date is not their game's day: {_examples(wrong)}")
    return problems


def verify_outputs(ctx: Context, latest: bytes, snapshot: bytes) -> dict:
    """Verify captured latest-feed and snapshot bytes against the package.
    Returns {'ok', 'problems', 'detail'}. Malformed files raise PublicationError."""
    from player_props.provenance import load_games
    frames = {'latest feed': parse_outputs(latest, "the latest feed"),
              'frozen snapshot': parse_outputs(snapshot, "the frozen snapshot")}
    games = load_games(ctx.runtime_data)          # the package's inference schedule (Stage 2-checked)
    date_problems = [p for label, f in frames.items()
                     for p in game_date_problems(f, label, games, ctx.season, ctx.week)]
    out = ctx.workdir / f"outputs-{secrets.token_hex(4)}"
    out.mkdir()
    paths = {'latest': out / LATEST_NAME, 'snapshot': out / snapshot_name(ctx.season, ctx.week)}
    paths['latest'].write_bytes(latest)
    paths['snapshot'].write_bytes(snapshot)
    run = rh.Run(config=rh.RehearsalConfig(workspace=ctx.workdir, target_season=ctx.season, target_week=ctx.week),
                 workspace=ctx.workdir, run_id='publication', dir=ctx.workdir,
                 paths={'runtime_data': ctx.runtime_data, 'train_stats': ctx.train_stats},
                 info={'bundle': ctx.bundle, 'bundle_dir': ctx.bundle_dir, 'check': ctx.check,
                       'latest': paths['latest'], 'snapshot': paths['snapshot']})
    problems: List[str] = list(date_problems)
    detail: dict = {}
    try:
        detail = rh._stage_verify_outputs(run)
    except rh.RehearsalError as exc:
        problems.append(str(exc))
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"the outputs could not be checked ({type(exc).__name__}: {str(exc)[:300]})")
        # Completeness runs first; keep what it found before the row checks raised.
        for label in ('latest feed', 'frozen snapshot'):
            counts = (run.info.get('completeness') or {}).get(label) or {}
            found = {k: counts.get(k) for k in ('missing', 'unexpected', 'duplicates', 'wrong_side') if counts.get(k)}
            if found:
                problems.append(f"{label} completeness: " + ", ".join(f"{k} {v}" for k, v in found.items()))
    if not problems:
        captured = {'latest': _sha256(latest), 'snapshot': _sha256(snapshot)}
        if run.info.get('verified') != captured:
            problems.append("the verifier's copies of the outputs no longer match the captured bytes")
        derived = ctx.manifest['derived']
        elig = run.info.get('eligibility') or {}
        if elig.get('players_reference_sha256') != derived['identities']['reference_runtime']['sha256']:
            problems.append("the outputs were checked against a players reference other than the package's "
                            "runtime reference")
        if elig.get('policy') != derived['eligibility']['policy']:
            problems.append(f"eligibility policy {elig.get('policy')!r} is not the package's "
                            f"{derived['eligibility']['policy']!r}")
    completeness = run.info.get('completeness')
    return {'ok': not problems, 'problems': problems, 'detail': detail, 'completeness': completeness,
            'eligibility': run.info.get('eligibility')}


# ---------------------------------------------------------------------------
# Sidecar
# ---------------------------------------------------------------------------

def build_sidecar(ctx: Context, latest: bytes, snapshot: bytes, result: dict) -> bytes:
    """Deterministic: a function of the package and the verified bytes only (no time, no
    source path), so the same verified bytes always give the same sidecar."""
    d = ctx.manifest['derived']
    verified = (ctx.summary.get('outputs') or {}).get('verify_outputs') or {}
    names_bytes = (_sha256(snapshot) == _sha256(ctx.contents['verified_predictions'])
                   and verified.get('snapshot_sha256') == _sha256(snapshot)
                   and verified.get('latest_sha256') == _sha256(latest))
    comp = result['completeness'] or {}
    doc = {
        'kind': SIDECAR_KIND, 'sidecar_version': SIDECAR_VERSION,
        'release': {'release_id': ctx.manifest['release_id'], 'manifest_sha256': ctx.manifest_sha256},
        'bundle': {'bundle_id': d['bundle']['bundle_id'], 'manifest_sha256': d['bundle']['manifest_sha256'],
                   'feature_contract': d['bundle']['feature_contract']},
        'inference_target': {'season': ctx.season, 'week': ctx.week},
        'inputs': {'inference_source_sha256': d['identities']['inference_source']['sha256'],
                   'inference_stats_manifest_sha256': d['identities']['inference_stats']['manifest_sha256'],
                   'prediction_schedule': {'file': f'nfl_schedule_{ctx.season}.csv',
                                           'sha256': ctx.schedule_sha256}},
        'reference': {'runtime_sha256': d['identities']['reference_runtime']['sha256'],
                      'candidate_sha256': d['identities']['reference_candidate']['sha256'],
                      'derivation': ctx.manifest['reference']['derivation']},
        'eligibility': {k: result['eligibility'][k] for k in ('policy', 'restated_policy', 'field',
                                                              'players_reference_sha256')},
        'outputs': {'latest': {'file': LATEST_NAME, 'sha256': _sha256(latest), 'bytes': len(latest)},
                    'snapshot': {'file': snapshot_name(ctx.season, ctx.week), 'sha256': _sha256(snapshot),
                                 'bytes': len(snapshot)}},
        'verification': {
            'result': 'passed', 'verifier': 'rehearsal._stage_verify_outputs on captured bytes',
            'rows': result['detail'].get('predictions_verified'),
            'expected_rows': comp.get('expected'), 'completeness': comp.get('status'),
            'models_metric_parity': result['detail'].get('models_metric_parity'),
            'adjustments': result['detail'].get('adjustments'),
        },
        'evidence': {
            'package_rehearsal_names_these_bytes': names_bytes,
            'statement': (
                "these exact bytes are the outputs the package's rehearsal verified; they were also verified "
                "again here" if names_bytes else
                "the package's rehearsal evidence does not cover these bytes; only this publication's "
                "verification supports them"),
        },
        'limitations': [
            "offline consistency verification only: not a deployment authorization",
            "hashes detect changed files; they do not make the published set immutable",
            "no accuracy, calibration or betting edge is measured or claimed",
        ],
    }
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode('utf-8')


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _library_output():
    """Capture what reused library code prints (progress lines, emoji) while
    verifying, so it can't fail on a console that can't encode it (Windows
    cp1252) or mix with the CLI's JSON. Yields the buffer."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        yield buf


def _tail(buf: io.StringIO, lines: int = 20) -> List[str]:
    return [line for line in buf.getvalue().splitlines() if line.strip()][-lines:]


def publish(package: Path | str, dest: Path | str, prediction_schedule: Path | str,
            latest: Optional[Path | str] = None, snapshot: Optional[Path | str] = None,
            packaged: bool = False, protected: Optional[List[Path]] = None,
            after_verify: Optional[Callable[[], None]] = None) -> dict:
    """Verify and publish one target's latest feed, frozen snapshot and
    provenance sidecar into ``dest``. Returns the publication report (state
    'committed' or 'committed_with_cleanup_warnings'); raises PublicationError
    when nothing was published or the previous set was restored (or could not
    be). ``after_verify`` is a test hook called between verification and
    installation."""
    package, dest, schedule_path = Path(package), Path(dest), Path(prediction_schedule)
    if packaged == (latest is not None or snapshot is not None) or (not packaged and (latest is None or snapshot is None)):
        raise PublicationError(["give either packaged=True, or both latest and snapshot files"])
    inputs = {'prediction schedule': schedule_path,
              'latest feed': None if latest is None else Path(latest),
              'frozen snapshot': None if snapshot is None else Path(snapshot)}
    dest_real = check_destination(dest, package, inputs, protected)       # before anything is read or written

    schedule_bytes = _capture(schedule_path, "prediction schedule")
    latest_bytes = None if latest is None else _capture(Path(latest), "latest feed")
    snapshot_bytes = None if snapshot is None else _capture(Path(snapshot), "frozen snapshot")

    with tempfile.TemporaryDirectory(prefix='prop-publication-') as tmp, _library_output() as library:
        try:
            ctx = prepare(package, schedule_bytes, Path(tmp))
            if packaged:
                packaged_bytes = ctx.contents['verified_predictions']
                v = (ctx.summary.get('outputs') or {}).get('verify_outputs') or {}
                if not (v.get('latest_sha256') == v.get('snapshot_sha256') == _sha256(packaged_bytes)):
                    raise PublicationError(["the package holds only the verified snapshot; its rehearsal's latest "
                                            "feed had other bytes, so the packaged snapshot can't stand in for it"])
                latest_bytes = snapshot_bytes = packaged_bytes
            elif Path(snapshot).name != snapshot_name(ctx.season, ctx.week):
                raise PublicationError([f"the snapshot file is named {Path(snapshot).name}, but the package's "
                                        f"target is {ctx.season} week {ctx.week} "
                                        f"({snapshot_name(ctx.season, ctx.week)})"])
            result = verify_outputs(ctx, latest_bytes, snapshot_bytes)
            if not result['ok']:
                raise PublicationError(["outputs not published: " + p for p in result['problems']])
        except PublicationError as exc:
            exc.library_output = _tail(library)
            raise
        sidecar_bytes = build_sidecar(ctx, latest_bytes, snapshot_bytes, result)
        season, week, release_id = ctx.season, ctx.week, ctx.manifest['release_id']

    if after_verify is not None:
        after_verify()
    report = _install(dest, dest_real, season, week, release_id, latest_bytes, snapshot_bytes, sidecar_bytes,
                      package, inputs, protected)
    report['outputs_source'] = 'package_snapshot' if packaged else 'supplied_files'
    report['package_rehearsal_names_these_bytes'] = json.loads(sidecar_bytes)['evidence'][
        'package_rehearsal_names_these_bytes']
    return report


def install_set(files: Dict[Path, bytes], staging: Path) -> dict:
    """Install ``files`` (destination -> bytes) as a set.

    Before the commit boundary (the last destination moved into place):
    staging, backing up existing destinations and the moves themselves. An
    error there restores the previous set (files that didn't exist are
    removed) and raises PublicationError with state 'rolled_back'; if a
    restore fails, state 'rollback_failed', and ``recovery_files`` names the
    backups kept. Nothing is reported as restored that wasn't.

    After the commit boundary every destination holds its new bytes. Deleting
    the backups and the staging directory is cleanup: a failure there never
    raises or claims a rollback; it is returned in ``cleanup_warnings`` with
    the paths left in ``retained_files``. Not crash-atomic (no fsync)."""
    token = secrets.token_hex(4)
    backups: Dict[Path, Path] = {}
    staged: Dict[Path, Path] = {}
    moved: List[Path] = []
    leftovers: List[str] = []

    def remove(path: Path) -> Optional[str]:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
            return None
        except OSError as exc:
            return f"{path} could not be removed ({type(exc).__name__}: {exc})"

    def abandon(cause: BaseException, before_moves: bool) -> PublicationError:
        unrestored = []
        for dest in reversed(moved):
            try:
                if dest in backups:
                    os.replace(backups[dest], dest)
                    del backups[dest]                  # consumed by the restore
                else:
                    dest.unlink()
            except OSError:
                unrestored.append(dest)
        for dest, bak in list(backups.items()):        # backups of destinations never replaced
            if dest not in unrestored:
                why = remove(bak)
                if why:
                    leftovers.append(why)
        why = remove(staging)
        if why:
            leftovers.append(why)
        if unrestored:
            recovery = [str(backups[d]) for d in unrestored if d in backups]
            detail = [f"{d}: previous content kept at {backups[d]}" if d in backups
                      else f"{d}: did not exist before, and the new file could not be removed" for d in unrestored]
            return PublicationError(
                [f"publication failed ({type(cause).__name__}: {cause}) and the rollback also failed: "
                 + "; ".join(detail)] + [f"recovery file kept: {r}" for r in recovery]
                + [f"also left behind: {x}" for x in leftovers],
                recovery_files=recovery, state='rollback_failed', retained_files=leftovers)
        what = ("nothing was replaced" if before_moves
                else "the previous latest feed, snapshot and sidecar were restored")
        return PublicationError([f"publication failed ({type(cause).__name__}: {cause}); {what}"]
                                + [f"left behind (the previous set is intact): {x}" for x in leftovers],
                                state='rolled_back', retained_files=leftovers)

    try:
        staging.mkdir()
        for dest, data in files.items():
            tmp = staging / dest.name
            tmp.write_bytes(data)
            staged[dest] = tmp
        for dest in files:
            if dest.exists():
                bak = dest.with_name(f".{dest.name}.{token}.bak")
                backups[dest] = bak                    # registered first, so a failed copy is cleaned up
                shutil.copy2(dest, bak)
    except Exception as exc:  # noqa: BLE001 - nothing has been replaced yet
        raise abandon(exc, before_moves=True) from exc
    try:
        for dest, tmp in staged.items():
            os.replace(tmp, dest)
            moved.append(dest)
    except Exception as exc:  # noqa: BLE001 - restore what was replaced
        raise abandon(exc, before_moves=False) from exc
    # ---- commit boundary: every destination now holds its new bytes ----
    warnings, retained = [], []
    for dest, bak in backups.items():
        why = remove(bak)
        if why:
            warnings.append(f"backup of the previous {dest.name} was not deleted: {why}")
            retained.append(str(bak))
    why = remove(staging)
    if why:
        warnings.append(f"staging directory was not deleted: {why}")
        retained.append(str(staging))
    return {'committed': True, 'cleanup_warnings': warnings, 'retained_files': retained}


def _release_lock(lock) -> Optional[tuple]:
    """Release ``lock`` (SlotLock) and confirm it is gone. None when released;
    otherwise (diagnostic, retained path or None).

    SlotLock.__exit__ deletes the file only if it still holds this lock's
    token, retries a sharing violation for a bounded time, and then returns
    normally whether or not the file was deleted. So the result is checked
    here, read-only. Nothing is deleted or retried here, and a lock is never
    removed by age. The file is described by what it says, which is not
    evidence that any process is still running."""
    try:
        lock.__exit__(None, None, None)
    except Exception as exc:  # noqa: BLE001 - reported, never masks the outcome
        error = f" ({type(exc).__name__}: {exc})"
    else:
        error = ""
    path = lock.path
    if not os.path.lexists(path):
        return None if not error else (f"releasing the lock raised{error}, but {path} no longer exists", None)
    try:
        info = json.loads(path.read_text(encoding='utf-8'))
        if not (isinstance(info, dict) and isinstance(info.get('token'), str)):
            raise ValueError("no token recorded")
        token = info['token']
    except (OSError, ValueError) as exc:
        return (f"the lock file {path} was not deleted{error} and can't be read or parsed "
                f"({type(exc).__name__}: {exc}), so whose it is can't be determined; it was left in place and "
                f"blocks the next publication to this destination. Check that no publication to this "
                f"destination is in progress before deleting it.", str(path))
    if token == lock.token:
        return (f"this publication's lock file {path} was not deleted{error} (it still holds this "
                f"publication's token); it blocks the next publication to this destination until removed. "
                f"This publication has finished with it, so it can be deleted once this process has exited.",
                str(path))
    return (f"the lock file {path} holds another token, not this publication's{error}; it was left "
            f"untouched. Either it replaced this publication's lock while that was held, or another "
            f"publication took the lock after this one released it; this can't tell which.", None)


def _install(dest: Path, dest_real: Path, season: int, week: int, release_id: str, latest: bytes,
             snapshot: bytes, sidecar: bytes, package: Path, inputs: dict, protected) -> dict:
    from ontario_spreads import CaptureError, SlotLock, new_id
    dest_real.mkdir(parents=True, exist_ok=True)
    if check_destination(dest, package, inputs, protected) != dest_real:      # re-checked after creation
        raise PublicationError([f"destination {dest} now resolves elsewhere; nothing was published"])
    files = {'latest': (dest_real / LATEST_NAME, latest),
             'snapshot': (dest_real / snapshot_name(season, week), snapshot),
             'sidecar': (dest_real / sidecar_name(season, week), sidecar)}
    lock = SlotLock(dest_real, LOCK_SLOT, new_id(datetime.now(timezone.utc)), what='prop publication')
    try:
        lock.__enter__()
    except CaptureError as exc:
        raise PublicationError([str(exc)]) from exc
    status = {}
    outcome = {'committed': True, 'cleanup_warnings': [], 'retained_files': []}
    failure = None
    try:
        conflicts = []
        for key in ('snapshot', 'sidecar'):
            path, data = files[key]
            if path.exists():
                if not path.is_file():
                    conflicts.append(f"{path.name} exists and is not a regular file")
                elif _sha256(path.read_bytes()) != _sha256(data):
                    conflicts.append(f"{path.name} is already published with different content; it was not "
                                     f"replaced and nothing was published")
                else:
                    status[key] = 'identical_existing'
        if conflicts:
            raise PublicationError(conflicts)
        if files['latest'][0].is_file() and _sha256(files['latest'][0].read_bytes()) == _sha256(latest):
            status['latest'] = 'identical_existing'
        todo = {}
        for key, (path, data) in files.items():
            if status.get(key) != 'identical_existing':
                status[key] = 'replaced' if path.exists() else 'created'
                todo[path] = data
        if todo:
            outcome = install_set(todo, dest_real / f".publish-{secrets.token_hex(6)}")
    except PublicationError as exc:
        failure = exc
    finally:
        released = _release_lock(lock)
        if released is not None:
            note, retained = released
            if failure is not None:                    # the original error and state stand
                failure.problems.append(note)
                if retained:
                    failure.retained_files.append(retained)
            else:
                outcome['cleanup_warnings'].append(note)
                if retained:
                    outcome['retained_files'].append(retained)
    if failure is not None:
        raise failure
    return {'published': True,
            'state': 'committed_with_cleanup_warnings' if outcome['cleanup_warnings'] else 'committed',
            'cleanup_warnings': outcome['cleanup_warnings'], 'retained_files': outcome['retained_files'],
            'release_id': release_id, 'target': {'season': season, 'week': week}, 'destination': str(dest_real),
            'files': {k: {'path': str(p), 'sha256': _sha256(d), 'status': status[k]}
                      for k, (p, d) in files.items()}}


# ---------------------------------------------------------------------------
# Read-only check of a published set
# ---------------------------------------------------------------------------

def _file_entry_problems(entry, name: str, where: str) -> List[str]:
    if not isinstance(entry, dict):
        return [f"the sidecar's {where} must be an object"]
    problems = []
    if entry.get('file') != name:
        problems.append(f"the sidecar's {where}.file must be {name}")
    if not rm._is_hash(entry.get('sha256')):
        problems.append(f"the sidecar's {where}.sha256 must be a 64-character hex SHA-256")
    if not rm._is_int(entry.get('bytes')) or entry['bytes'] < 0:
        problems.append(f"the sidecar's {where}.bytes must be a non-negative integer")
    return problems


def sidecar_problems(side, season: int, week: int, current: bool) -> List[str]:
    """Every problem with the sidecar fields a verification mode relies on."""
    if not isinstance(side, dict):
        return ["the sidecar must be a JSON object"]
    if side.get('kind') != SIDECAR_KIND or side.get('sidecar_version') != SIDECAR_VERSION:
        return [f"the sidecar is not a version-{SIDECAR_VERSION} {SIDECAR_KIND} sidecar"]
    problems = []
    if side.get('inference_target') != {'season': season, 'week': week}:
        problems.append(f"the sidecar's inference_target is {side.get('inference_target')!r}, not "
                        f"{{'season': {season}, 'week': {week}}}")
    release = side.get('release')
    if not (isinstance(release, dict) and isinstance(release.get('release_id'), str)
            and rm._is_hash(release.get('manifest_sha256'))):
        problems.append("the sidecar's release must name a release_id and manifest_sha256")
    if rm._get(side, 'verification', 'result') != 'passed':
        problems.append("the sidecar does not record a passed verification")
    outputs = side.get('outputs')
    if not isinstance(outputs, dict):
        return problems + ["the sidecar's outputs must be an object"]
    problems += _file_entry_problems(outputs.get('snapshot'), snapshot_name(season, week), 'outputs.snapshot')
    if current:
        problems += _file_entry_problems(outputs.get('latest'), LATEST_NAME, 'outputs.latest')
    return problems


def _matches(path: Path, entry: dict, what: str) -> List[str]:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return [f"{what} {path.name} is missing"]
    except OSError as exc:
        return [f"{what} {path.name} can't be read ({type(exc).__name__}: {exc})"]
    if len(data) != entry['bytes'] or _sha256(data) != entry['sha256']:
        return [f"{what} {path.name} doesn't match its sidecar (size {len(data)} vs {entry['bytes']}, "
                f"sha256 {_sha256(data)[:12]} vs {entry['sha256'][:12]})"]
    return []


def _latest_accounted_for(dest: Path, season: int, week: int) -> Optional[str]:
    """Evidence for a latest feed that isn't this target's: another sidecar
    here records its exact hash and size, and that sidecar's own snapshot
    verifies. None when nothing accounts for it."""
    try:
        data = (dest / LATEST_NAME).read_bytes()
    except OSError:
        return None
    for path in sorted(dest.glob('player_props_predictions_week*_*.provenance.json')):
        if path.name == sidecar_name(season, week):
            continue
        try:
            other = json.loads(path.read_text(encoding='utf-8'))
            t = other['inference_target']
            if (not sidecar_problems(other, t['season'], t['week'], current=True)
                    and not _matches(dest / LATEST_NAME, other['outputs']['latest'], 'latest')
                    and _sha256(data) == other['outputs']['latest']['sha256']
                    and not _matches(dest / snapshot_name(t['season'], t['week']), other['outputs']['snapshot'],
                                     'snapshot')):
                return f"{t['season']} week {t['week']} ({path.name})"
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def verify_published(dest: Path | str, season: int, week: int, package: Optional[Path | str] = None,
                     mode: str = 'current') -> dict:
    """Read-only check of a published target, in one of two explicit modes.

    * ``'current'`` verifies the complete current set: the sidecar, the frozen
      snapshot AND the latest feed must exist, be readable and match the
      sidecar. A latest feed that doesn't match fails. If another sidecar here
      records the latest feed's exact bytes (and its own snapshot verifies),
      that is reported as evidence of a later publication, but this target is
      still not the current set; otherwise no explanation is offered.
    * ``'historical'`` verifies only the frozen snapshot and its sidecar, and
      reports the current latest feed under ``not_verified``.

    With ``package``, the sidecar must name that package's manifest hash."""
    if mode not in ('current', 'historical'):
        raise ValueError("mode must be 'current' or 'historical'")
    dest = Path(dest)
    current = mode == 'current'
    report = {'ok': False, 'mode': mode, 'target': {'season': season, 'week': week}, 'problems': [], 'notes': [],
              'verified': [], 'not_verified': []}
    problems, notes = report['problems'], report['notes']
    if not current:
        report['not_verified'].append("the current latest feed (historical mode verifies only the frozen "
                                      "snapshot and its sidecar)")
    side_path = dest / sidecar_name(season, week)
    try:
        side = json.loads(side_path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        problems.append(f"the sidecar {side_path.name} is missing")
        return report
    except (OSError, UnicodeDecodeError) as exc:
        problems.append(f"the sidecar {side_path.name} can't be read ({type(exc).__name__}: {exc})")
        return report
    except json.JSONDecodeError as exc:
        problems.append(f"the sidecar {side_path.name} is not valid JSON: line {exc.lineno}, column {exc.colno}: "
                        f"{exc.msg}")
        return report
    found = sidecar_problems(side, season, week, current)
    if found:
        problems += found
        return report
    report['release'] = side['release']
    problems += _matches(dest / snapshot_name(season, week), side['outputs']['snapshot'], "the frozen snapshot")
    other = _latest_accounted_for(dest, season, week)
    if current:
        latest_problems = _matches(dest / LATEST_NAME, side['outputs']['latest'], "the latest feed")
        if latest_problems:
            problems += latest_problems
            notes.append(f"the latest feed matches the published set for {other}, so {season} week {week} is "
                         f"not the current set" if other else
                         "nothing here accounts for the latest feed's current state")
    elif other:
        notes.append(f"for information: the current latest feed matches the published set for {other}")
    if package is not None:
        try:
            _, mbytes, _ = rm.read_package(package)
        except rm.ReleaseError as exc:
            problems += [f"package: {p}" for p in exc.problems]
        else:
            if _sha256(mbytes) != side['release']['manifest_sha256']:
                problems.append("the sidecar names another release package")
    if not problems:
        report['verified'] = (["the sidecar's fields", "the frozen snapshot matches the sidecar"]
                              + (["the latest feed matches the sidecar: this is the current set"] if current else [])
                              + (["the sidecar names the given package"] if package is not None else []))
    report['ok'] = not problems
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verified, offline publication of Player Props outputs")
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('publish')
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--dest', type=Path, required=True, help="isolated directory outside the repository")
    p.add_argument('--prediction-schedule', type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--packaged', action='store_true', help="publish the package's own verified snapshot")
    g.add_argument('--latest', type=Path)
    p.add_argument('--snapshot', type=Path)
    v = sub.add_parser('verify-published')
    v.add_argument('--dest', type=Path, required=True)
    v.add_argument('--season', type=int, required=True)
    v.add_argument('--week', type=int, required=True)
    v.add_argument('--package', type=Path)
    v.add_argument('--historical', action='store_true',
                   help="verify only the frozen snapshot and its sidecar, not the current latest feed")
    args = ap.parse_args(argv)
    # JSON is printed ASCII-only (json's default), so any console encoding can show it.
    try:
        if args.cmd == 'publish':
            if not args.packaged and args.snapshot is None:
                ap.error("--latest needs --snapshot")
            report = publish(args.package, args.dest, args.prediction_schedule, latest=args.latest,
                             snapshot=args.snapshot, packaged=args.packaged)
        else:
            report = verify_published(args.dest, args.season, args.week, args.package,
                                      mode='historical' if args.historical else 'current')
            print(json.dumps(report, indent=2, default=str))
            return 0 if report['ok'] else 1
    except PublicationError as exc:
        print(json.dumps({'published': False, 'state': exc.state, 'problems': exc.problems,
                          'recovery_files': exc.recovery_files, 'retained_files': exc.retained_files,
                          'library_output_tail': exc.library_output}, indent=2))
        return 1
    except Exception as exc:  # noqa: BLE001 - could not run: nothing is claimed
        print(json.dumps({'published': False, 'error': f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    print(json.dumps(report, indent=2))
    # 3: published (committed), but cleanup left files behind (see cleanup_warnings).
    return 3 if report.get('cleanup_warnings') else 0


if __name__ == '__main__':
    sys.exit(main())
