"""Capture a new research snapshot and report source changes; never promote audited facts."""
from __future__ import annotations
import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.request
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import football_repository as repository

REPORTS = ROOT / "reports" / "football_repository_pilot"
SOURCE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
MAX_BYTES = 8 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def parse(data, season):
    text = data.decode("utf-8-sig")
    header = text.splitlines()[0] if text else ""
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t" if "\t" in header else ",")
    fields = reader.fieldnames or []
    if len(fields) != len(set(fields)) or not set(repository.REQUIRED_COLUMNS).issubset(fields):
        raise ValueError("Missing required columns or duplicate column names")
    rows = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed source row: inconsistent column count")
        if row["season"] == str(season):
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows for season {season}")
    ids = [row["game_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate game IDs in selected season")
    return fields, rows


def write_tsv(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def refresh(*, source=None, previous_source, output_root, season, evaluation_date, download=False):
    if (source is not None) == download:
        raise ValueError("Choose exactly one local source or --download")
    if not 1900 <= season <= 2200:
        raise ValueError("Invalid season")
    output_root = Path(output_root).resolve()
    if not output_root.is_relative_to(REPORTS.resolve()) or output_root == REPORTS.resolve():
        raise ValueError("Output must be below reports/football_repository_pilot")
    previous_source = Path(previous_source).resolve()
    old_bytes = previous_source.read_bytes()
    old_fields, old_rows = parse(old_bytes, season)
    captured = datetime.now(timezone.utc)
    if download:
        with urllib.request.urlopen(SOURCE_URL, timeout=45) as response:
            data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError("Schedule exceeds download size limit")
        origin = SOURCE_URL
    else:
        origin = str(Path(source).resolve())
        data = Path(source).read_bytes()
    fields, rows = parse(data, season)
    output_root.mkdir(parents=True, exist_ok=True)
    run = output_root / (captured.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    run.mkdir()  # Never reuse or replace an existing attempt.
    with (run / "source.raw").open("xb") as handle:
        handle.write(data)
    write_tsv(run / "previous.tsv", old_fields, old_rows)
    write_tsv(run / "games.tsv", fields, rows)
    conn = repository.connect(run / "repository.sqlite3")
    try:
        old_id, _ = repository.import_source(conn, run / "previous.tsv")
        old_assessment, _ = repository.assess_completion(conn, old_id, evaluation_date)
        import_id, _ = repository.import_source(conn, run / "games.tsv")
        assessment_id, _ = repository.assess_completion(conn, import_id, evaluation_date)
        def completed(aid):
            return {r[0] for r in conn.execute("SELECT game_id FROM game_completions WHERE assessment_id=? AND completion_status='final'", (aid,))}
        old_final, final = completed(old_assessment), completed(assessment_id)
        missing = []
        for row in conn.execute("SELECT o.game_id,o.roof_category,o.surface_category FROM game_observations o JOIN game_completions c ON o.game_id=c.game_id WHERE o.import_id=? AND c.assessment_id=? AND c.completion_status='final'", (import_id, assessment_id)):
            unknown = [name for name, value in zip(("roof", "surface"), row[1:]) if value == repository.UNKNOWN]
            if unknown:
                missing.append({"game_id": row[0], "fields": unknown})
        pending = [{"game_id": row[0], "reason": row[1]} for row in conn.execute("SELECT game_id,completion_reason FROM game_completions WHERE assessment_id=? AND completion_status!='final'", (assessment_id,))]
    finally:
        conn.close()
    old = {r["game_id"]: r for r in old_rows}
    current = {r["game_id"]: r for r in rows}
    changed = []
    for gid in sorted(old.keys() & current.keys()):
        differences = {field: {"before": old[gid].get(field), "after": current[gid].get(field)}
                       for field in sorted(set(old_fields) | set(fields)) if old[gid].get(field) != current[gid].get(field)}
        if differences:
            changed.append({"game_id": gid, "fields": differences})
    report = {
        "status": "captured_not_audited", "captured_at_utc": captured.isoformat(),
        "acquisition": "public_schedule_download" if download else "local_file_copy",
        "transformation_version": repository.TRANSFORMATION_VERSION,
        "completion_rule_version": repository.COMPLETION_RULE_VERSION,
        "evaluation_toronto_date": evaluation_date.isoformat(), "season": season,
        "source": origin, "source_sha256": digest(data), "previous_source": str(previous_source),
        "previous_source_sha256": digest(old_bytes), "season_tsv_sha256": digest((run / "games.tsv").read_bytes()),
        "import_id": import_id, "assessment_id": assessment_id,
        "row_count": len(rows), "completed_count": len(final), "pending_or_unverifiable": pending,
        "added_games": sorted(current.keys() - old.keys()), "removed_games": sorted(old.keys() - current.keys()),
        "changed_games": changed, "newly_completed": sorted(final - old_final),
        "no_longer_completed": sorted(old_final - final), "completed_missing_conditions": missing,
        "limits": "Raw source observations only. No gamebook verification or inherited supplements. Completion uses the same evaluation date on both sources. Raw string changes can be formatting changes. No injury/weather/referee coverage claim; population is relative to the provider schedule.",
    }
    with (run / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return run, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=Path)
    group.add_argument("--download", action="store_true", help="Download only the public nflverse schedule")
    parser.add_argument("--previous-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=REPORTS / "refreshes")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--date", type=date.fromisoformat, required=True, help="Toronto evaluation date")
    args = parser.parse_args(argv)
    try:
        run, report = refresh(source=args.source, previous_source=args.previous_source,
                              output_root=args.output_root, season=args.season,
                              evaluation_date=args.date, download=args.download)
    except (OSError, ValueError, csv.Error, repository.sqlite3.DatabaseError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 2
    print(json.dumps({"output": str(run), "status": report["status"], "completed": report["completed_count"],
                      "added": len(report["added_games"]), "changed": len(report["changed_games"]),
                      "missing_conditions": len(report["completed_missing_conditions"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
