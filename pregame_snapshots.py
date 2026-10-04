"""Immutable pregame spread-prediction snapshots.

An auditable record of what the spread model said BEFORE kickoff. The nightly
pipeline rewrites `nfl_games_historical_with_predictions.csv` for every game
(including played ones), so that CSV can't show what was predicted at the
time. This module freezes it:

* `write_run_manifest` - called by nfl-gather-data.py at the end of a
  successful run. Gives the run a unique identity (run_id) ONCE and records
  its provenance (code revision, config and feature-set identifiers, cutoffs)
  plus SHA-256 hashes of the exact schedule bytes it parsed and the
  predictions CSV it wrote, in `pipeline_run_manifest.json`. Kept OUT of the
  deterministic artifacts so the pipeline's determinism check is unaffected.
* `capture` - after the pipeline: reads the schedule and predictions bytes
  once, checks them against the manifest, then writes ONE immutable JSON file
  per run (named by the manifest's run_id) holding every eligible upcoming
  game - including games with no bet signal, no line, or a pick'em line.
* `select_captures` - read-only: the earliest or latest eligible capture per
  game across all snapshot files.

Times: nflverse `gameday` + `gametime` are US Eastern wall-clock times
(America/New_York - London games are listed at 09:30), converted to UTC with
the IANA zone. A wall-clock time in the autumn repeat hour (ambiguous) or the
spring gap (nonexistent) is skipped with a reason rather than guessed.

Integrity: every snapshot is validated (schema version, required fields and
types, UTC timestamps, game identifiers, statuses, probabilities and
signal/line consistency) before it is written, before an existing file is
accepted on retry, and before selection. `payload_sha256` covers the whole
canonical document except itself - capture time, kickoff times and
provenance included. It is an UNKEYED checksum: it detects accidental or
partial changes, but anyone who edits a file deliberately can recompute it,
so it does not authenticate a file. Git history of the committed snapshots is
the tamper-evidence.

Immutability: a snapshot file is named `<run_id>.json` and created with an
exclusive link (temp file, then os.link to the final name), so an existing file
is NEVER replaced. Re-capturing the same run - same manifest - keeps the
original file (even after kickoffs have passed); the same run ID with ANY
different provenance is refused. Concurrent writers for the same run: exactly
one link succeeds; the others validate the winner and no-op or raise.

Snapshots contain no final scores, results or settlement fields. Completed
games are never captured, so regenerated probabilities can't be backfilled as
"pregame". The betting log (`betting_recommendations_log.csv`) is separate.

CLI:
    python pregame_snapshots.py capture
    python pregame_snapshots.py select --which earliest [--status predicted] [--output picks.csv]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import re
import secrets
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
KIND = "pregame_spread_snapshot"
DATA_DIR = Path("data_files")
SNAPSHOT_DIR = DATA_DIR / "pregame_snapshots"
MANIFEST_NAME = "pipeline_run_manifest.json"
PREDICTIONS_NAME = "nfl_games_historical_with_predictions.csv"
SCHEDULE_NAME = "nfl_games_historical.csv"
SOURCE_TZ = ZoneInfo("America/New_York")      # nflverse gameday/gametime are US Eastern
SPREAD_CONVENTION = ("nflverse spread_line: points the HOME team is favored by "
                     "(positive = home favored, negative = away favored, 0 = pick'em)")
PROBABILITY_MEANING = ("prob_underdog_covers = model P(underdog covers spread_line), pushes "
                       "excluded; null when there is no valid line. A model estimate, not a "
                       "demonstrated betting edge.")
CHECKSUM_FIELD = "payload_sha256"
SEASON_BOUNDS, WEEK_BOUNDS = (1920, 2100), (1, 22)

# prediction_status values
PREDICTED, NO_LINE, PICKEM, NO_PROBABILITY = "predicted", "no_line", "pickem", "no_probability"
STATUSES = (PREDICTED, NO_LINE, PICKEM, NO_PROBABILITY)
LINE_STATUS_FOR = {PREDICTED: "valid", NO_PROBABILITY: "valid", NO_LINE: "missing", PICKEM: "pickem"}
# skip reasons for upcoming (not completed) games
MISSING_KICKOFF, INVALID_KICKOFF = "missing_kickoff", "invalid_kickoff"
AMBIGUOUS_KICKOFF, NONEXISTENT_KICKOFF = "ambiguous_kickoff", "nonexistent_kickoff"
NOT_BEFORE_KICKOFF = "capture_not_before_kickoff"
SKIP_REASONS = (MISSING_KICKOFF, INVALID_KICKOFF, AMBIGUOUS_KICKOFF, NONEXISTENT_KICKOFF, NOT_BEFORE_KICKOFF)

GAME_FIELDS = ("season", "week", "game_id", "home_team", "away_team", "kickoff_utc", "kickoff_source",
               "spread_line", "line_status", "underdog_team", "prob_underdog_covers", "bet_signal",
               "prediction_status")
MANIFEST_FIELDS = ("schema_version", "run_id", "generated_at", "code_revision", "code_dirty", "code_sha256",
                   "config_id", "config", "feature_set_id", "features", "training_cutoff", "data_cutoff",
                   "spread_threshold", "ci_run", "platform", "artifact", "schedule")
SNAPSHOT_FIELDS = ("schema_version", "kind", "run_id", "captured_at", "run", "kickoff_timezone",
                   "spread_convention", "probability", "counts", "games", "skipped", CHECKSUM_FIELD)

_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX16 = re.compile(r"^[0-9a-f]{16}$")
_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{12}$")


class ProvenanceError(RuntimeError):
    """The manifest doesn't describe the prediction artifact (or is missing/invalid)."""


class SnapshotConflictError(RuntimeError):
    """A snapshot file for this run ID already exists and doesn't match this run."""


class SnapshotValidationError(ValueError):
    """A snapshot document breaks the schema or fails its integrity checksum."""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_iso(text: str) -> datetime:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"timestamp {text!r} has no timezone")
    return dt.astimezone(timezone.utc)


def _parse_utc(text) -> datetime:
    """Strict: only our canonical 'YYYY-MM-DDTHH:MM:SSZ' form."""
    if not isinstance(text, str) or not _UTC_RE.match(text):
        raise ValueError(f"{text!r} is not a UTC timestamp of the form YYYY-MM-DDTHH:MM:SSZ")
    return _parse_iso(text)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    return sha256_bytes(Path(path).read_bytes())


def read_bytes_once(path: Path | str) -> tuple[bytes, str]:
    """(content, sha256 of exactly that content). Parse the returned bytes - never
    re-open the file - so the hash always describes what was parsed."""
    data = Path(path).read_bytes()
    return data, sha256_bytes(data)


def parse_tsv(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), sep="\t")


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _short_hash(obj) -> str:
    return hashlib.sha256(_canonical(obj).encode()).hexdigest()[:16]


def payload_checksum(doc: dict) -> str:
    """SHA-256 of the canonical document without its checksum field (unkeyed)."""
    return hashlib.sha256(_canonical({k: v for k, v in doc.items() if k != CHECKSUM_FIELD}).encode()).hexdigest()


def _jsonable(value):
    """Primitive, JSON-safe (no NaN) representation of a config value."""
    if isinstance(value, (np.generic,)):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return f"<{type(value).__name__}>"


# --------------------------------------------------------------------------
# Kickoff times
# --------------------------------------------------------------------------

def kickoff_utc(gameday, gametime, tz: ZoneInfo = SOURCE_TZ) -> tuple[datetime | None, str | None]:
    """Scheduled kickoff in UTC from nflverse local date/time, or (None, reason)."""
    if gameday is None or gametime is None or pd.isna(gameday) or pd.isna(gametime) \
            or not str(gameday).strip() or not str(gametime).strip():
        return None, MISSING_KICKOFF
    try:
        naive = datetime.strptime(f"{str(gameday).strip()[:10]} {str(gametime).strip()}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None, INVALID_KICKOFF
    first, second = naive.replace(tzinfo=tz, fold=0), naive.replace(tzinfo=tz, fold=1)
    if first.utcoffset() != second.utcoffset():
        # Repeated autumn hour -> ambiguous; skipped spring hour -> nonexistent
        # (the wall-clock time doesn't survive a round trip through UTC).
        roundtrip = first.astimezone(timezone.utc).astimezone(tz).replace(tzinfo=None)
        return None, (AMBIGUOUS_KICKOFF if roundtrip == naive else NONEXISTENT_KICKOFF)
    return first.astimezone(timezone.utc), None


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def _fail(where: str, msg: str):
    raise SnapshotValidationError(f"{where}: {msg}")


def _check_fields(obj, required, where):
    if not isinstance(obj, dict):
        _fail(where, f"expected an object, got {type(obj).__name__}")
    missing = [f for f in required if f not in obj]
    if missing:
        _fail(where, f"missing field(s) {missing}")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _check_cutoff(c, where):
    _check_fields(c, ("season", "week", "gameday", "games"), where)
    if not (_is_int(c["season"]) and _is_int(c["week"]) and _is_int(c["games"]) and c["games"] >= 1
            and isinstance(c["gameday"], str) and c["gameday"]):
        _fail(where, f"malformed cutoff {c!r}")


def validate_manifest(m: dict, where: str = "manifest") -> None:
    """Structure and internal consistency of a run manifest (raises SnapshotValidationError)."""
    _check_fields(m, MANIFEST_FIELDS, where)
    if m["schema_version"] != SCHEMA_VERSION:
        _fail(where, f"unsupported schema_version {m['schema_version']!r}")
    if not isinstance(m["run_id"], str) or not _RUN_ID_RE.match(m["run_id"]):
        _fail(where, f"malformed run_id {m['run_id']!r}")
    try:
        generated = _parse_utc(m["generated_at"])
    except ValueError as exc:
        _fail(where, f"generated_at: {exc}")
    if m["run_id"][:16] != f"{generated:%Y%m%dT%H%M%SZ}":
        _fail(where, "run_id timestamp doesn't match generated_at")
    if not isinstance(m["code_revision"], str) or not m["code_revision"]:
        _fail(where, "code_revision must be a non-empty string")
    if m["code_dirty"] not in (True, False, None):
        _fail(where, "code_dirty must be true, false or null")
    for key, pat in (("code_sha256", _HEX64), ("config_id", _HEX16), ("feature_set_id", _HEX16)):
        if not isinstance(m[key], str) or not pat.match(m[key]):
            _fail(where, f"malformed {key} {m[key]!r}")
    if not isinstance(m["config"], dict) or _short_hash(m["config"]) != m["config_id"]:
        _fail(where, "config_id doesn't match config")
    feats = m["features"]
    if not (isinstance(feats, list) and feats and all(isinstance(f, str) and f for f in feats)
            and feats == sorted(feats)):
        _fail(where, "features must be a non-empty sorted list of names")
    if _short_hash(feats) != m["feature_set_id"]:
        _fail(where, "feature_set_id doesn't match features")
    _check_cutoff(m["training_cutoff"], f"{where}.training_cutoff")
    _check_cutoff(m["data_cutoff"], f"{where}.data_cutoff")
    thr = m["spread_threshold"]
    if not (isinstance(thr, float) and 0.0 < thr < 1.0):
        _fail(where, f"spread_threshold must be a float in (0, 1), got {thr!r}")
    for key in ("ci_run", "platform"):
        if not isinstance(m[key], dict):
            _fail(where, f"{key} must be an object")
    for key, name in (("artifact", PREDICTIONS_NAME), ("schedule", SCHEDULE_NAME)):
        _check_fields(m[key], ("path", "sha256"), f"{where}.{key}")
        if m[key]["path"] != name or not _HEX64.match(str(m[key]["sha256"])):
            _fail(f"{where}.{key}", f"malformed {m[key]!r}")


def _validate_game(g, captured_at: datetime, where: str) -> None:
    if not isinstance(g, dict) or set(g) != set(GAME_FIELDS):
        _fail(where, f"fields must be exactly {sorted(GAME_FIELDS)}")
    if not (_is_int(g["season"]) and SEASON_BOUNDS[0] <= g["season"] <= SEASON_BOUNDS[1]):
        _fail(where, f"bad season {g['season']!r}")
    if not (_is_int(g["week"]) and WEEK_BOUNDS[0] <= g["week"] <= WEEK_BOUNDS[1]):
        _fail(where, f"bad week {g['week']!r}")
    for team in ("home_team", "away_team"):
        if not isinstance(g[team], str) or not re.fullmatch(r"[A-Za-z0-9]+", g[team]):
            _fail(where, f"bad {team} {g[team]!r}")
    expected_id = f"{g['season']}_{g['week']:02d}_{g['away_team']}_{g['home_team']}"
    if g["game_id"] != expected_id:
        _fail(where, f"game_id {g['game_id']!r} != {expected_id!r} (season_week_away_home)")
    try:
        ko = _parse_utc(g["kickoff_utc"])
    except ValueError as exc:
        _fail(where, f"kickoff_utc: {exc}")
    if not ko > captured_at:
        _fail(where, "kickoff_utc is not strictly after captured_at")
    if not isinstance(g["kickoff_source"], str) or not g["kickoff_source"]:
        _fail(where, "kickoff_source must be a non-empty string")
    status, line, prob = g["prediction_status"], g["spread_line"], g["prob_underdog_covers"]
    if status not in STATUSES:
        _fail(where, f"unknown prediction_status {status!r}")
    if g["line_status"] != LINE_STATUS_FOR[status]:
        _fail(where, f"line_status {g['line_status']!r} inconsistent with prediction_status {status!r}")
    if g["line_status"] == "missing":
        if line is not None:
            _fail(where, "spread_line must be null when the line is missing")
    elif not (isinstance(line, (int, float)) and not isinstance(line, bool) and math.isfinite(line)):
        _fail(where, f"bad spread_line {line!r}")
    elif (g["line_status"] == "pickem") != (line == 0):
        _fail(where, f"spread_line {line!r} inconsistent with line_status {g['line_status']!r}")
    expected_dog = None if g["line_status"] != "valid" else (g["away_team"] if line > 0 else g["home_team"])
    if g["underdog_team"] != expected_dog:
        _fail(where, f"underdog_team {g['underdog_team']!r} != {expected_dog!r} for spread_line {line!r}")
    if status == PREDICTED:
        if not (isinstance(prob, float) and 0.0 <= prob <= 1.0):
            _fail(where, f"predicted game needs a probability in [0, 1], got {prob!r}")
    elif prob is not None:
        _fail(where, f"{status} game must have a null probability")
    if not isinstance(g["bet_signal"], bool):
        _fail(where, "bet_signal must be a boolean")
    if g["bet_signal"] and status != PREDICTED:
        _fail(where, f"bet_signal is true for a {status} game")


def validate_snapshot(doc, where: str = "snapshot") -> None:
    """Full schema + integrity check of a snapshot document (raises SnapshotValidationError)."""
    _check_fields(doc, SNAPSHOT_FIELDS, where)
    if doc["schema_version"] != SCHEMA_VERSION or doc["kind"] != KIND:
        _fail(where, f"unsupported schema {doc['schema_version']!r}/{doc['kind']!r} "
                     f"(expected {SCHEMA_VERSION}/{KIND})")
    if not isinstance(doc[CHECKSUM_FIELD], str) or doc[CHECKSUM_FIELD] != payload_checksum(doc):
        _fail(where, f"{CHECKSUM_FIELD} doesn't match the content - changed after capture")
    validate_manifest(doc["run"], f"{where}.run")
    if doc["run_id"] != doc["run"]["run_id"]:
        _fail(where, "run_id differs from run.run_id")
    try:
        captured = _parse_utc(doc["captured_at"])
    except ValueError as exc:
        _fail(where, f"captured_at: {exc}")
    if captured < _parse_utc(doc["run"]["generated_at"]):
        _fail(where, "captured_at precedes the run's generated_at")
    if doc["kickoff_timezone"] != SOURCE_TZ.key:
        _fail(where, f"kickoff_timezone must be {SOURCE_TZ.key}")
    for key in ("spread_convention", "probability"):
        if not isinstance(doc[key], str) or not doc[key]:
            _fail(where, f"{key} must be a non-empty string")
    games, skipped, counts = doc["games"], doc["skipped"], doc["counts"]
    if not isinstance(games, list) or not isinstance(skipped, list) or not isinstance(counts, dict):
        _fail(where, "games/skipped must be lists and counts an object")
    seen = set()
    for i, g in enumerate(games):
        _validate_game(g, captured, f"{where}.games[{i}]")
        if g["game_id"] in seen:
            _fail(where, f"duplicate game_id {g['game_id']!r}")
        seen.add(g["game_id"])
    for i, s in enumerate(skipped):
        _check_fields(s, ("game_id", "reason", "gameday", "gametime"), f"{where}.skipped[{i}]")
        if s["reason"] not in SKIP_REASONS or not isinstance(s["game_id"], str) or s["game_id"] in seen:
            _fail(f"{where}.skipped[{i}]", f"bad skipped entry {s!r}")
    expected = {"captured": len(games), "skipped": len(skipped)}
    for st in STATUSES:
        n = sum(g["prediction_status"] == st for g in games)
        if n:
            expected[st] = n
    if not (_is_int(counts.get("completed_excluded")) and counts["completed_excluded"] >= 0):
        _fail(where, "counts.completed_excluded must be a non-negative integer")
    if {k: v for k, v in counts.items() if k != "completed_excluded"} != expected:
        _fail(where, f"counts {counts!r} don't match the games/skipped lists")


# --------------------------------------------------------------------------
# Pipeline run manifest (written by nfl-gather-data.py)
# --------------------------------------------------------------------------

def code_revision(repo_dir: Path | str) -> tuple[str, bool | None]:
    """(git revision, dirty?) of the code that is running. GITHUB_SHA wins in CI."""
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"], False
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True,
                             text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_dir,
                               capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        return rev or "unknown", bool(dirty)
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


def new_run_id(generated_at: datetime) -> str:
    """Unique generating-run identity: UTC timestamp + 48 random bits."""
    return f"{generated_at.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(6)}"


def remove_stale_manifest(data_dir: Path | str) -> None:
    """Called when a pipeline run starts, so a failed run never leaves an old manifest
    sitting next to new (or half-written) artifacts."""
    p = Path(data_dir) / MANIFEST_NAME
    if p.exists():
        p.unlink()


def write_run_manifest(data_dir: Path | str, *, schedule_sha256: str, code_files: list[Path],
                       repo_dir: Path | str, config: dict, features: list[str],
                       training_cutoff: dict, data_cutoff: dict, spread_threshold: float,
                       now: datetime | None = None) -> dict:
    """Write pipeline_run_manifest.json for the run that just produced the predictions CSV.

    The run_id is assigned here, once, and never regenerated: capture retries
    reuse this manifest. `schedule_sha256` must be the hash of the exact bytes
    the run parsed (see read_bytes_once).
    """
    data_dir = Path(data_dir)
    generated_at = (now or utc_now()).replace(microsecond=0)
    _, artifact_sha = read_bytes_once(data_dir / PREDICTIONS_NAME)
    revision, dirty = code_revision(repo_dir)
    code_sha = hashlib.sha256()
    for f in sorted(Path(p) for p in code_files):
        code_sha.update(f.name.encode() + b"\0" + f.read_bytes())
    config = _jsonable(config)
    feats = sorted(features)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": new_run_id(generated_at),
        "generated_at": _iso(generated_at),
        "code_revision": revision,
        "code_dirty": dirty,
        "code_sha256": code_sha.hexdigest(),
        "config_id": _short_hash(config),
        "config": config,
        "feature_set_id": _short_hash(feats),
        "features": feats,
        "training_cutoff": _jsonable(training_cutoff),
        "data_cutoff": _jsonable(data_cutoff),
        "spread_threshold": float(spread_threshold),
        "ci_run": {k: os.environ.get(k) for k in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_WORKFLOW")
                   if os.environ.get(k)},
        "platform": {"python": platform.python_version()},
        "artifact": {"path": PREDICTIONS_NAME, "sha256": artifact_sha},
        "schedule": {"path": SCHEDULE_NAME, "sha256": schedule_sha256},
    }
    validate_manifest(manifest)
    tmp = data_dir / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, data_dir / MANIFEST_NAME)
    return manifest


def load_verified_inputs(data_dir: Path | str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """(manifest, schedule, predictions), with both frames parsed from the SAME bytes
    whose hashes were checked against the manifest - no re-read between check and use."""
    data_dir = Path(data_dir)
    path = data_dir / MANIFEST_NAME
    if not path.exists():
        raise ProvenanceError(f"{path} not found - the pipeline did not finish successfully "
                              "(or predates provenance); refusing to capture unlabelled predictions")
    try:
        manifest = json.loads(path.read_text())
        validate_manifest(manifest, str(path))
    except (ValueError, SnapshotValidationError) as exc:
        raise ProvenanceError(f"invalid manifest: {exc}") from None
    frames = {}
    for key, name in (("artifact", PREDICTIONS_NAME), ("schedule", SCHEDULE_NAME)):
        data, actual = read_bytes_once(data_dir / name)
        if actual != manifest[key]["sha256"]:
            raise ProvenanceError(
                f"{name} does not match the pipeline run that wrote {MANIFEST_NAME} "
                f"(run {manifest['run_id']}): sha256 {actual[:12]}... != recorded "
                f"{manifest[key]['sha256'][:12]}... - it was modified or regenerated afterwards")
        frames[key] = parse_tsv(data)
    return manifest, frames["schedule"], frames["artifact"]


def load_verified_manifest(data_dir: Path | str) -> dict:
    return load_verified_inputs(data_dir)[0]


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------

def build_snapshot(manifest: dict, schedule: pd.DataFrame, predictions: pd.DataFrame,
                   captured_at: datetime) -> dict:
    """The (validated, checksummed) snapshot document for one run. Pure: no I/O."""
    if captured_at.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    captured_at = captured_at.astimezone(timezone.utc).replace(microsecond=0)
    if captured_at < _parse_utc(manifest["generated_at"]):
        raise ProvenanceError("capture time precedes the pipeline run that generated the predictions")
    if list(schedule["game_id"]) != list(predictions["game_id"]):
        raise ProvenanceError("schedule and predictions rows don't line up by game_id")

    games, skipped = [], []
    counts = {"completed_excluded": 0}
    for (_, s), (_, p) in zip(schedule.iterrows(), predictions.iterrows()):
        if pd.notna(s.get("home_score")) and pd.notna(s.get("away_score")):
            counts["completed_excluded"] += 1                  # never captured (no backfill)
            continue
        ko, reason = kickoff_utc(s.get("gameday"), s.get("gametime"))
        if ko is not None and ko <= captured_at:
            reason = NOT_BEFORE_KICKOFF
        if reason:
            skipped.append({"game_id": str(s["game_id"]), "reason": reason,
                            "gameday": None if pd.isna(s.get("gameday")) else str(s.get("gameday")),
                            "gametime": None if pd.isna(s.get("gametime")) else str(s.get("gametime"))})
            continue

        line = s.get("spread_line")
        line = None if pd.isna(line) else float(line)
        line_status = "missing" if line is None else "pickem" if line == 0 else "valid"
        prob = p.get("prob_underdogCovered")
        prob = None if pd.isna(prob) else float(prob)
        if line_status == "missing":
            status, prob = NO_LINE, None
        elif line_status == "pickem":
            status, prob = PICKEM, None
        elif prob is None:
            status = NO_PROBABILITY
        else:
            status = PREDICTED
        signal = status == PREDICTED and int(p.get("pred_spreadCovered_optimal", 0) or 0) == 1
        underdog = None if line_status != "valid" else (s["away_team"] if line > 0 else s["home_team"])
        games.append({
            "season": int(s["season"]), "week": int(s["week"]), "game_id": str(s["game_id"]),
            "home_team": str(s["home_team"]), "away_team": str(s["away_team"]),
            "kickoff_utc": _iso(ko),
            "kickoff_source": f"{str(s['gameday'])[:10]} {s['gametime']} {SOURCE_TZ.key}",
            "spread_line": line, "line_status": line_status,
            "underdog_team": None if underdog is None else str(underdog),
            "prob_underdog_covers": prob, "bet_signal": bool(signal), "prediction_status": status,
        })
    games.sort(key=lambda g: (g["kickoff_utc"], g["game_id"]))
    skipped.sort(key=lambda g: g["game_id"])
    for g in games:
        counts[g["prediction_status"]] = counts.get(g["prediction_status"], 0) + 1
    counts["captured"] = len(games)
    counts["skipped"] = len(skipped)
    doc = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "run_id": manifest["run_id"],
        "captured_at": _iso(captured_at),
        "run": manifest,
        "kickoff_timezone": SOURCE_TZ.key,
        "spread_convention": SPREAD_CONVENTION,
        "probability": PROBABILITY_MEANING,
        "counts": counts,
        "games": games,
        "skipped": skipped,
    }
    doc[CHECKSUM_FIELD] = payload_checksum(doc)
    validate_snapshot(doc, "new snapshot")                 # never write something we'd reject
    return doc


def _exclusive_write(path: Path, text: str) -> bool:
    """Create `path` with `text` atomically, never replacing an existing file.

    Returns True if created, False if `path` already existed. The content is
    written to a temp file in the same directory first, then hard-linked to
    the final name - link fails if the name exists, so there is no window in
    which a partial file is visible or an existing file is replaced.
    """
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
            return True
        except FileExistsError:
            return False
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _accept_existing(path: Path, manifest: dict) -> dict:
    """An existing snapshot for this run_id is accepted only if it is valid AND was
    produced from exactly this manifest (every provenance field)."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        validate_snapshot(doc, str(path))
    except (OSError, ValueError) as exc:
        raise SnapshotConflictError(f"{path} already exists but is not a valid snapshot ({exc}); "
                                    "refusing to overwrite it") from None
    if doc["run"] != manifest:
        diff = sorted(k for k in set(doc["run"]) | set(manifest) if doc["run"].get(k) != manifest.get(k))
        raise SnapshotConflictError(
            f"{path} already exists for run {manifest['run_id']} with different provenance "
            f"({', '.join(diff)}); refusing to overwrite it")
    return doc


def capture(data_dir: Path | str = DATA_DIR, snapshot_dir: Path | str | None = None,
            now: datetime | None = None) -> tuple[Path, bool, dict]:
    """Capture the current run's pregame predictions. Returns (path, created, snapshot).

    created=False means this run was already captured (idempotent retry); the
    existing file is left exactly as it was - even if kickoffs have since passed.
    """
    data_dir = Path(data_dir)
    snapshot_dir = Path(snapshot_dir) if snapshot_dir else data_dir / "pregame_snapshots"
    manifest, schedule, predictions = load_verified_inputs(data_dir)
    path = snapshot_dir / f"{manifest['run_id']}.json"
    if path.exists():
        return path, False, _accept_existing(path, manifest)

    snap = build_snapshot(manifest, schedule, predictions, now or utc_now())
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(snap, indent=1, sort_keys=True, allow_nan=False) + "\n"
    if _exclusive_write(path, text):
        return path, True, snap
    return path, False, _accept_existing(path, manifest)     # lost a race to a concurrent writer


# --------------------------------------------------------------------------
# Read-only selection
# --------------------------------------------------------------------------

def load_snapshots(snapshot_dir: Path | str = SNAPSHOT_DIR) -> pd.DataFrame:
    """Every captured game row from every snapshot file, with run provenance columns.

    Read-only. Raises SnapshotValidationError naming the file for anything that
    isn't a valid schema-1 snapshot (including checksum mismatches).
    """
    rows = []
    for path in sorted(Path(snapshot_dir).glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise SnapshotValidationError(f"{path}: not valid JSON ({exc})") from None
        validate_snapshot(doc, str(path))
        run = doc["run"]
        for g in doc["games"]:
            rows.append({**g, "run_id": doc["run_id"], "captured_at": doc["captured_at"],
                         "code_revision": run["code_revision"], "config_id": run["config_id"],
                         "feature_set_id": run["feature_set_id"],
                         "artifact_sha256": run["artifact"]["sha256"], "snapshot_file": path.name})
    return pd.DataFrame(rows)


def select_captures(snapshot_dir: Path | str = SNAPSHOT_DIR, which: str = "earliest",
                    statuses: list[str] | None = None, before: datetime | None = None) -> pd.DataFrame:
    """One capture per game: the earliest or latest ELIGIBLE capture.

    Every file is validated first. Eligible: captured strictly before the game's
    scheduled kickoff (re-checked here), strictly before `before` if given, and
    with a prediction_status in `statuses` if given. Order is (captured_at,
    run_id); "earliest" takes the first and "latest" the last, so ties on
    captured_at are broken by run_id - deterministic regardless of file order.
    Read-only.
    """
    if which not in ("earliest", "latest"):
        raise ValueError("which must be 'earliest' or 'latest'")
    df = load_snapshots(snapshot_dir)
    if df.empty:
        return df
    cap = df["captured_at"].map(_parse_utc)
    ko = df["kickoff_utc"].map(_parse_utc)
    keep = cap < ko
    if before is not None:
        keep &= cap < before.astimezone(timezone.utc)
    if statuses:
        keep &= df["prediction_status"].isin(statuses)
    df = df[keep].assign(_cap=cap[keep])
    df = df.sort_values(["game_id", "_cap", "run_id"], kind="mergesort")
    pick = df.groupby("game_id", sort=True).head(1) if which == "earliest" \
        else df.groupby("game_id", sort=True).tail(1)
    return pick.drop(columns="_cap").sort_values(["kickoff_utc", "game_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture", help="capture the current pipeline run (after nfl-gather-data.py)")
    c.add_argument("--data-dir", default=str(DATA_DIR))
    c.add_argument("--snapshot-dir", default=None)
    s = sub.add_parser("select", help="earliest/latest eligible capture per game (read-only)")
    s.add_argument("--snapshot-dir", default=str(SNAPSHOT_DIR))
    s.add_argument("--which", choices=("earliest", "latest"), default="earliest")
    s.add_argument("--status", action="append", choices=STATUSES, help="keep only this status (repeatable)")
    s.add_argument("--before", help="only captures strictly before this ISO-8601 time (with offset)")
    s.add_argument("--output", help="write CSV here (default: print)")
    args = ap.parse_args(argv)

    if args.cmd == "capture":
        path, created, snap = capture(args.data_dir, args.snapshot_dir)
        verb = "wrote" if created else "already captured (unchanged)"
        print(f"[pregame_snapshots] {verb} {path} - run {snap['run_id']}, captured_at {snap['captured_at']}, "
              f"counts {snap['counts']}")
        return 0

    before = _parse_iso(args.before) if args.before else None
    picks = select_captures(args.snapshot_dir, args.which, args.status, before)
    if args.output:
        out = Path(args.output).resolve()
        if Path(args.snapshot_dir).resolve() in out.parents:
            raise SystemExit("refusing to write selection output inside the snapshot directory")
        picks.to_csv(out, index=False)
        print(f"[pregame_snapshots] {len(picks)} games -> {out}")
    else:
        print(picks.to_string(index=False) if len(picks) else "(no eligible captures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
