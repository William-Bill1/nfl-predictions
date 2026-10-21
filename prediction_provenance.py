"""When the displayed predictions were generated, from the pipeline's own record.

The home page used to show the predictions CSV's file modification time, which
a git checkout, copy or fast-forward resets, so it showed when the file
landed on disk, not when the pipeline produced it. This reads
``pipeline_run_manifest.json`` (written by the pipeline run that produced the
predictions; see pregame_snapshots.write_run_manifest) and uses its
``generated_at`` ONLY after checking that:

* the manifest exists, parses and passes pregame_snapshots.validate_manifest;
* its ``artifact`` entry names the predictions file; and
* the sha256 of the predictions file on disk equals the one the manifest
  recorded, so the manifest describes exactly the file being displayed.

Otherwise the time is unavailable, with a short reason. File modification
time, checkout time and the current time are never used as a substitute.
Display only: nothing is written.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pregame_snapshots as ps

TORONTO = ZoneInfo("America/Toronto")
UNAVAILABLE = "Generation time unavailable"


def verified_generation_time(data_dir: Path | str) -> tuple[datetime | None, str | None, str]:
    """(generated_at as aware UTC, run_id, "") when the manifest verifiably
    describes the predictions file; otherwise (None, None, reason)."""
    data_dir = Path(data_dir)
    manifest_path = data_dir / ps.MANIFEST_NAME
    predictions_path = data_dir / ps.PREDICTIONS_NAME
    if not manifest_path.is_file():
        return None, None, "no pipeline run manifest was found"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ps.validate_manifest(manifest, str(manifest_path))
    except (OSError, ValueError, KeyError, TypeError, ps.SnapshotValidationError):
        return None, None, "the pipeline run manifest is missing fields or malformed"
    # validate_manifest also requires artifact.path to name the predictions file.
    try:
        _, actual = ps.read_bytes_once(predictions_path)
    except OSError:
        return None, None, "the predictions file can't be read"
    if actual != manifest["artifact"]["sha256"]:
        return None, None, ("the predictions file differs from the one the pipeline run "
                            "recorded; it was changed or regenerated afterwards")
    return ps._parse_utc(manifest["generated_at"]), manifest["run_id"], ""


def toronto_text(at: datetime) -> str:
    """'2026-10-21 04:12 EDT' - explicit America/Toronto, with EDT/EST."""
    return at.astimezone(TORONTO).strftime("%Y-%m-%d %H:%M %Z")


def generation_caption(data_dir: Path | str) -> str:
    at, run_id, reason = verified_generation_time(data_dir)
    if at is None:
        return f"Predictions last generated: {UNAVAILABLE} ({reason})."
    return f"Predictions last generated: {toronto_text(at)} (pipeline run {run_id})"
