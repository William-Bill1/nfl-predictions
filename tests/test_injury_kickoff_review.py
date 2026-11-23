"""Reviewed 2026 Week 11 kickoff corrections end to end: injury collection,
capture validation, import provenance, the legacy-database upgrade and the
stored-context panel's identity from the predictions table.

Synthetic captures in temporary folders; the predictions-table fixture copies the
real stale Week 11 rows (PIT-PHI 16:25, JAX-NYG 13:00 Eastern). No network.
"""
import copy
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import injury_history as ih  # noqa: E402
import injury_reports as ir  # noqa: E402
import kickoff_corrections as kc  # noqa: E402
import ontario_spreads as on  # noqa: E402
import pregame_snapshots as ps  # noqa: E402
import stored_context as sc  # noqa: E402
from test_injury_reports import Resp, Source, _payload, run  # noqa: E402
from test_stored_context import config, weather_capture, weather_import  # noqa: E402
from test_weather_history import FakeResp, body  # noqa: E402

UTC = timezone.utc
PIT, JAX, ARI, LV = "2026_11_PIT_PHI", "2026_11_JAX_NYG", "2026_11_ARI_KC", "2026_11_LV_DEN"
TEAMS = ["PIT", "PHI", "JAX", "NYG", "ARI", "KC", "LV", "DEN"]
WED = datetime(2026, 11, 18, 17, 5, tzinfo=UTC)          # Wednesday noon slot (EST)
WED_LATER = datetime(2026, 11, 18, 17, 30, tzinfo=UTC)   # retained-style stale capture, received later
WED_PANEL = datetime(2026, 11, 18, 18, 0, tzinfo=UTC)
Z = "%Y-%m-%dT%H:%M:%SZ"

# Real stale rows from nfl_games_historical_with_predictions.csv (2026 Week 11),
# with illustrative probability columns that must never change.
PRED_ROWS = [
    {"game_id": PIT, "season": 2026, "week": 11, "gameday": "2026-11-22", "gametime": "16:25",
     "away_team": "PIT", "home_team": "PHI", "stadium_id": "PHI00", "stadium": "Lincoln Financial Field",
     "prob_underdogCovered": 0.57, "spread_line": 2.5},
    {"game_id": JAX, "season": 2026, "week": 11, "gameday": "2026-11-22", "gametime": "13:00",
     "away_team": "JAX", "home_team": "NYG", "stadium_id": "NYC01", "stadium": "MetLife Stadium",
     "prob_underdogCovered": 0.52, "spread_line": -1.5},
    {"game_id": ARI, "season": 2026, "week": 11, "gameday": "2026-11-22", "gametime": "13:00",
     "away_team": "ARI", "home_team": "KC", "stadium_id": "KAN00", "stadium": "GEHA Field at Arrowhead Stadium",
     "prob_underdogCovered": 0.61, "spread_line": 7.5},
    {"game_id": LV, "season": 2026, "week": 11, "gameday": "2026-11-22", "gametime": "16:25",
     "away_team": "LV", "home_team": "DEN", "stadium_id": "DEN00", "stadium": "Empower Field at Mile High",
     "prob_underdogCovered": 0.49, "spread_line": 3.0},
]
SOURCE_KO = {PIT: "2026-11-22T21:25:00Z", JAX: "2026-11-22T18:00:00Z"}
EFFECTIVE_KO = {PIT: "2026-11-22T18:00:00Z", JAX: "2026-11-22T21:25:00Z"}
SCHED_COLUMNS = ["game_id", "season", "week", "gameday", "gametime", "away_team", "home_team"]


def pred_frame(corrected=False, changes=None):
    df = pd.DataFrame(copy.deepcopy(PRED_ROWS))
    if corrected:
        df.loc[df.game_id == PIT, "gametime"] = "13:00"
        df.loc[df.game_id == JAX, "gametime"] = "16:25"
    for (gid, col), value in (changes or {}).items():
        df.loc[df.game_id == gid, col] = value
    return df


def schedule_file(tmp, frame, name="schedule.tsv"):
    path = tmp / name
    frame[SCHED_COLUMNS].assign(away_score=None, home_score=None).to_csv(path, sep="\t", index=False)
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corrected_capture(tmp, frame, at=WED, slot="auto", folder="caps"):
    """Through injury_reports.capture (the corrected collection path)."""
    res = run(schedule_file(tmp, frame), tmp / folder, Source(Resp(_payload(TEAMS))), at, slot=slot)
    assert res.path is not None, res
    return res


def reseal(doc, mutate):
    d = copy.deepcopy(doc)
    d.pop(ir.CHECKSUM_FIELD)
    mutate(d)
    return ir._seal(d)


def stale_capture(tmp, at=WED_LATER, folder="caps"):
    """Retained-style capture: made from the stale source with no kickoff review,
    exactly the shape of every capture written before this change."""
    raw, digest = on.load_schedule(schedule_file(tmp, pred_frame(), "stale_source.tsv"))
    meta = {"http_status": 200, "http_last_modified": None, "http_date": None, "bytes": 1, "error": None}
    doc = ir.build_capture(capture_id=on.new_id(at), resolved=on.resolve_slot(at, on.AD_HOC),
                           request_started_at=at, received_at=at + timedelta(seconds=2),
                           payload=_payload(TEAMS), meta=meta, schedule=raw, schedule_sha256=digest,
                           kickoff_review=kc.review_block([]), code=ps.code_revision(ROOT))
    doc = reseal(doc, lambda d: d.__setitem__("schedule", {"path": d["schedule"]["path"],
                                                           "sha256": d["schedule"]["sha256"]}))
    ir.validate_capture(doc)
    assert ir.schedule_kind(doc) == "legacy"
    path, _ = ir.write_capture(doc, tmp / folder)
    return path, doc


def import_into(db, *paths):
    conn = ih.connect_for_import(db)
    try:
        return ih.import_captures(conn, list(paths))
    finally:
        conn.close()


def game_rows(db):
    conn = sqlite3.connect(db)
    try:
        return {(c, g): (k, s, b) for c, g, k, s, b in conn.execute(
            "SELECT c.capture_id, g.game_id, g.kickoff_utc, g.source_kickoff_utc, g.kickoff_basis "
            "FROM injury_capture_games g JOIN injury_captures c USING (capture_import_id)")}
    finally:
        conn.close()


def contexts(frame, db, now=WED_PANEL, ids=(PIT, JAX, ARI, LV)):
    res = sc.build_all(frame, list(ids), environ=config(Path(db).parent, injury_db=db), now=now)
    assert res["state"] == "ok"
    return res["contexts"]


# ---------------------------------------------------------------- end to end (both games)


def test_corrected_capture_selected_and_stale_wednesday_capture_excluded(tmp_path):
    pred = pred_frame()
    before = pred.copy(deep=True)
    new = corrected_capture(tmp_path, pred)
    stale_path, stale = stale_capture(tmp_path)
    db = tmp_path / "injury.sqlite3"
    import_into(db, new.path, stale_path)

    rows = game_rows(db)
    new_id, stale_id = new.doc["capture_id"], stale["capture_id"]
    for gid in (PIT, JAX):
        assert rows[(new_id, gid)] == (EFFECTIVE_KO[gid], SOURCE_KO[gid], "reviewed_correction")
        assert rows[(stale_id, gid)] == (SOURCE_KO[gid], None, "capture_without_kickoff_review")
    assert rows[(new_id, ARI)] == ("2026-11-22T18:00:00Z", "2026-11-22T18:00:00Z", "source")

    ctx = contexts(pred, db)
    for gid in (PIT, JAX):
        game, inj = ctx[gid]["game"], ctx[gid]["injuries"]
        assert game["kickoff_utc"].strftime(Z) == EFFECTIVE_KO[gid]
        assert game["source_kickoff_utc"].strftime(Z) == SOURCE_KO[gid]
        assert game["kickoff_review"]["status"] == "applied"
        assert inj["state"] == "ok" and inj["latest"]["capture_id"] == new_id
        assert any(stale_id in n and f"kicking off {SOURCE_KO[gid]}" in n for n in inj["notices"])
        # Before this change the identity used the stale table time, so the
        # stale capture matched it and was shown as the latest:
        old_identity = {**game, "kickoff_utc": game["source_kickoff_utc"], "kickoff_review": None}
        old = sc.injury_context(db, old_identity, WED_PANEL)
        assert old["latest"]["capture_id"] == stale_id
    for gid in (ARI, LV):                       # unrelated games: unchanged behaviour
        game, inj = ctx[gid]["game"], ctx[gid]["injuries"]
        assert game["kickoff_review"] is None and game["kickoff_utc"] == game["source_kickoff_utc"]
        assert inj["latest"]["capture_id"] == stale_id and not inj["notices"]
    assert ctx[LV]["game"]["kickoff_utc"].strftime(Z) == "2026-11-22T21:25:00Z"
    pd.testing.assert_frame_equal(pred, before)


@pytest.mark.parametrize("pred_corrected", [False, True])
@pytest.mark.parametrize("source_corrected", [False, True])
def test_already_corrected_inputs(tmp_path, pred_corrected, source_corrected):
    src = pred_frame(corrected=source_corrected)
    new = corrected_capture(tmp_path, src)
    status = "source_already_corrected" if source_corrected else "applied"
    assert [r["status"] for r in new.doc["schedule"]["kickoff_review"]["games"]] == [status, status]
    db = tmp_path / "injury.sqlite3"
    import_into(db, new.path)
    basis = "source_already_corrected" if source_corrected else "reviewed_correction"
    rows = game_rows(db)
    for gid in (PIT, JAX):
        source_ko = EFFECTIVE_KO[gid] if source_corrected else SOURCE_KO[gid]
        assert rows[(new.doc["capture_id"], gid)] == (EFFECTIVE_KO[gid], source_ko, basis)
    pred = pred_frame(corrected=pred_corrected)
    ctx = contexts(pred, db, ids=(PIT, JAX))
    for gid in (PIT, JAX):
        assert ctx[gid]["game"]["kickoff_review"]["status"] == ("source_already_corrected" if pred_corrected
                                                                else "applied")
        assert ctx[gid]["injuries"]["latest"]["capture_id"] == new.doc["capture_id"]


def test_sunday_scope_and_cutoffs_use_the_corrected_kickoffs(tmp_path):
    pred = pred_frame()
    sunday = corrected_capture(tmp_path, pred, at=datetime(2026, 11, 22, 14, 5, tzinfo=UTC))
    late = corrected_capture(tmp_path, pred, at=datetime(2026, 11, 22, 18, 5, tzinfo=UTC), slot=on.AD_HOC)
    states = {g["game_id"]: g["state"] for g in late.doc["scope"]["games"]}
    assert states[PIT] != "pregame" and states[JAX] == "pregame"
    db = tmp_path / "injury.sqlite3"
    import_into(db, sunday.path, late.path)
    ctx = contexts(pred, db, now=datetime(2026, 11, 22, 18, 30, tzinfo=UTC), ids=(PIT, JAX))
    assert ctx[PIT]["cutoff"].strftime(Z) == EFFECTIVE_KO[PIT] and ctx[PIT]["cutoff_basis"] == "kickoff"
    assert ctx[PIT]["injuries"]["latest"]["capture_id"] == sunday.doc["capture_id"]
    assert ctx[PIT]["injuries"]["excluded_received_after_cutoff"] == 1
    assert ctx[JAX]["injuries"]["latest"]["capture_id"] == late.doc["capture_id"]


# ---------------------------------------------------------------- unexpected table changes fail closed


@pytest.mark.parametrize("col,value", [("gametime", "20:20"), ("gameday", "2026-11-23"), ("home_team", "DAL"),
                                       ("away_team", "CLE"), ("season", 2025), ("week", 12), ("season", None)])
def test_unexpected_predictions_table_change_fails_closed(tmp_path, col, value):
    new = corrected_capture(tmp_path, pred_frame())
    db = tmp_path / "injury.sqlite3"
    import_into(db, new.path)
    ctx = contexts(pred_frame(changes={(PIT, col): value}), db)
    assert ctx[PIT]["state"] == "identity_error" and "review required" in ctx[PIT]["message"]
    assert ctx[JAX]["injuries"]["latest"]["capture_id"] == new.doc["capture_id"]
    assert ctx[ARI]["state"] == "ok"


def test_missing_identity_columns_fail_closed_only_for_reviewed_games(tmp_path):
    new = corrected_capture(tmp_path, pred_frame())
    db = tmp_path / "injury.sqlite3"
    import_into(db, new.path)
    ctx = contexts(pred_frame().drop(columns=["season", "week"]), db)
    assert ctx[PIT]["state"] == ctx[JAX]["state"] == "identity_error"
    assert ctx[ARI]["injuries"]["latest"]["capture_id"] == new.doc["capture_id"]


def test_duplicate_predictions_rows_still_refused(tmp_path):
    pred = pred_frame()
    ctx = contexts(pd.concat([pred, pred[pred.game_id == PIT]]), tmp_path / "absent.sqlite3", ids=(PIT,))
    assert ctx[PIT]["state"] == "identity_error"


# ---------------------------------------------------------------- provenance validation


def _rec(d, gid):
    return next(r for r in d["schedule"]["kickoff_review"]["games"] if r["game_id"] == gid)


def _scope(d, gid):
    return next(g for g in d["scope"]["games"] if g["game_id"] == gid)


TAMPERING = {
    "review_removed_with_corrected_times": (lambda d: d.__setitem__(
        "schedule", {"path": d["schedule"]["path"], "sha256": d["schedule"]["sha256"]}), "undocumented correction"),
    "review_not_an_object": (lambda d: d["schedule"].__setitem__("kickoff_review", "anything"), "fields"),
    "duplicate_record": (lambda d: d["schedule"]["kickoff_review"]["games"].append(
        copy.deepcopy(_rec(d, PIT))), "twice"),
    "unreviewed_game_record": (lambda d: d["schedule"]["kickoff_review"]["games"].append(
        {**_rec(d, PIT), "game_id": LV}), "not a reviewed game"),
    "record_authority": (lambda d: _rec(d, PIT).__setitem__("authority", "https://example.com"), "differs"),
    "block_authority": (lambda d: d["schedule"]["kickoff_review"].__setitem__("authorities", ["x"]), "authority"),
    "status_flipped": (lambda d: _rec(d, PIT).__setitem__("status", "source_already_corrected"), "differs"),
    "effective_time": (lambda d: _rec(d, JAX).__setitem__("effective_kickoff_utc", "2026-11-22T21:30:00Z"),
                       "differs"),
    "unknown_source_time": (lambda d: _rec(d, PIT).__setitem__("source_gametime", "20:20"), "isn't reviewed"),
    "extra_field": (lambda d: _rec(d, PIT).__setitem__("note", "x"), "fields"),
    "record_missing": (lambda d: d["schedule"]["kickoff_review"].__setitem__(
        "games", [_rec(d, PIT)]), "without its kickoff_review record"),
    "scope_kickoff_stale": (lambda d: _scope(d, PIT).__setitem__("kickoff_utc", SOURCE_KO[PIT]),
                            "isn't its reviewed effective kickoff"),
    "scope_identity": (lambda d: _scope(d, JAX).__setitem__("week", 12), "scope identity"),
    "sha256_basis": (lambda d: d["schedule"].__setitem__("sha256_basis", "corrected bytes"), "sha256_basis"),
    "sha256_shape": (lambda d: d["schedule"].__setitem__("sha256", "not-a-hash"), "schedule sha256"),
    "review_version_bool": (lambda d: d["schedule"]["kickoff_review"].__setitem__("review_version", True),
                            "review_version"),
    "extra_schedule_field": (lambda d: d["schedule"].__setitem__("kickoff_corrections", []), "schedule fields"),
}


@pytest.mark.parametrize("name", list(TAMPERING))
def test_provenance_tampering_refused_by_validation_and_import(tmp_path, name):
    mutate, msg = TAMPERING[name]
    new = corrected_capture(tmp_path, pred_frame())
    bad = reseal(new.doc, mutate)
    with pytest.raises(ir.ValidationError, match=msg):
        ir.validate_capture(bad)
    folder = tmp_path / "tampered"
    folder.mkdir()
    path = folder / f"{bad['capture_id']}.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    db = tmp_path / "injury.sqlite3"
    with pytest.raises(ih.ImportRefused):
        import_into(db, path)
    assert ih.schema_state(db) == "current" and game_rows(db) == {}


def test_unsealed_provenance_edit_is_a_checksum_mismatch(tmp_path):
    new = corrected_capture(tmp_path, pred_frame())
    doc = copy.deepcopy(new.doc)
    _rec(doc, PIT)["source_gametime"] = "13:00"
    with pytest.raises(ir.ValidationError, match="checksum"):
        ir.validate_capture(doc)


# ---------------------------------------------------------------- duplicate and conflicting imports


def _conflicting(doc):
    """Same capture, valid on its own, but claiming the source already had the
    reviewed PIT-PHI time."""
    def change(d):
        rec = _rec(d, PIT)
        rec.update(kc.expected_record(PIT, "13:00"))
    return reseal(doc, change)


def test_duplicate_and_conflicting_imports(tmp_path):
    new = corrected_capture(tmp_path, pred_frame())
    db = tmp_path / "injury.sqlite3"
    assert import_into(db, new.path)[0]["created"]
    rows = game_rows(db)
    again = import_into(db, new.path)
    assert again[0]["created"] is False and game_rows(db) == rows

    other = _conflicting(new.doc)
    ir.validate_capture(other)
    folder = tmp_path / "other"
    folder.mkdir()
    other_path = folder / new.path.name
    other_path.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(ih.ImportRefused, match="different content"):
        import_into(db, other_path)
    assert game_rows(db) == rows and rows[(new.doc["capture_id"], PIT)][2] == "reviewed_correction"

    fresh = tmp_path / "fresh.sqlite3"
    with pytest.raises(ih.ImportRefused, match="selected twice"):
        import_into(fresh, new.path, other_path)
    assert game_rows(fresh) == {}


def test_import_stores_raw_hash_basis_and_review_evidence(tmp_path):
    src = pred_frame()
    new = corrected_capture(tmp_path, src)
    db = tmp_path / "injury.sqlite3"
    import_into(db, new.path)
    conn = sqlite3.connect(db)
    try:
        prov, digest, basis, review = conn.execute(
            "SELECT schedule_provenance, schedule_sha256, schedule_sha256_basis, schedule_kickoff_review_json "
            "FROM injury_captures").fetchone()
    finally:
        conn.close()
    assert prov == "reviewed" and digest == sha(tmp_path / "schedule.tsv")
    assert basis == ir.SCHEDULE_SHA256_BASIS and "not reflected in this hash" in basis
    assert json.loads(review) == new.doc["schedule"]["kickoff_review"]


# ---------------------------------------------------------------- legacy captures and databases


def test_retained_captures_still_validate_and_import_unchanged(tmp_path):
    """Retained captures (legacy, and reviewed ones written since the Week 11 change)
    validate and import as captured; reviewed ones keep their own authorities."""
    caps = sorted((ROOT / "data_files" / "injury_reports" / "captures").glob("*.json"))
    assert caps, "retained captures expected in the repository"
    before = {p: sha(p) for p in caps}
    db = tmp_path / "injury.sqlite3"
    import_into(db, *caps)
    docs = {json.loads(p.read_text(encoding="utf-8"))["capture_id"]: json.loads(p.read_text(encoding="utf-8"))
            for p in caps}
    kinds = {cid: ir.schedule_kind(d) for cid, d in docs.items()}
    assert "legacy" in kinds.values()
    for (cid, gid), (kickoff, source, basis) in game_rows(db).items():
        assert kickoff == next(g["kickoff_utc"] for g in docs[cid]["scope"]["games"] if g["game_id"] == gid)
        if kinds[cid] == "legacy":
            assert basis == "capture_without_kickoff_review" and source is None
            continue
        recs = {r["game_id"]: r for r in docs[cid]["schedule"]["kickoff_review"]["games"]}
        if gid in recs:
            assert basis == ("reviewed_correction" if recs[gid]["status"] == "applied"
                             else "source_already_corrected")
            assert source == recs[gid]["source_kickoff_utc"]
        else:
            assert basis == "source" and source == kickoff
    for d in docs.values():
        if kinds[d["capture_id"]] == "legacy":
            continue
        block = d["schedule"]["kickoff_review"]
        if block["review_version"] == 1:
            assert block["authority"] == kc.WEEK11_AUTHORITY
        for r in block["games"]:
            assert r["authority"] == kc.REVIEWED[r["game_id"]].authority
    assert {p: sha(p) for p in caps} == before


def legacy_copy(current_db, legacy_db):
    """A first-schema database holding the same original columns (as an
    injury-history database built before kickoff provenance would)."""
    conn = sqlite3.connect(legacy_db)
    try:
        conn.executescript(ih.SCHEMA_V1)
        conn.execute("ATTACH DATABASE ? AS cur", (str(current_db),))
        for table in ih._TABLE_KEYS:
            cols = ", ".join(r[1] for r in conn.execute(f"PRAGMA main.table_info({table})"))
            conn.execute(f"INSERT INTO main.{table} ({cols}) SELECT {cols} FROM cur.{table}")
        conn.commit()
        conn.execute("DETACH DATABASE cur")
    finally:
        conn.close()
    assert ih.schema_state(legacy_db) == "legacy"
    return legacy_db


@pytest.fixture
def legacy_db(tmp_path):
    pred = pred_frame()
    new = corrected_capture(tmp_path, pred)
    stale_path, stale = stale_capture(tmp_path)
    current = tmp_path / "built.sqlite3"
    import_into(current, new.path, stale_path)
    db = legacy_copy(current, tmp_path / "legacy.sqlite3")
    return {"db": db, "new": new, "stale": stale, "stale_path": stale_path, "pred": pred}


def test_legacy_database_is_read_but_never_imported_into(legacy_db, tmp_path, capsys):
    db, new = legacy_db["db"], legacy_db["new"]
    digest = sha(db)
    with pytest.raises(ih.fr.SchemaRefused, match="upgrade"):
        ih.connect_for_import(db)
    assert ih.main(["import", "--db", str(db), str(new.path)]) == 2
    assert ih.main(["list", "--db", str(db)]) == 0
    ctx = contexts(legacy_db["pred"], db)
    for gid in (PIT, JAX):                        # no stored provenance: nothing for reviewed games
        inj = ctx[gid]["injuries"]
        assert inj["state"] == "none"
        assert any(new.doc["capture_id"] in n and "no recorded kickoff review" in n for n in inj["notices"])
        assert any(legacy_db["stale"]["capture_id"] in n and "kicking off" in n for n in inj["notices"])
    assert ctx[ARI]["injuries"]["latest"]["capture_id"] == legacy_db["stale"]["capture_id"]
    assert sha(db) == digest


def test_upgrade_copies_every_row_and_never_reinterprets(legacy_db, tmp_path, capsys):
    src, new = legacy_db["db"], legacy_db["new"]
    digest = sha(src)
    v1 = sqlite3.connect(src)
    try:
        tables = {t: [r[1] for r in v1.execute(f"PRAGMA table_info({t})")] for t in ih._TABLE_KEYS}
        before = {t: ih._table_rows(v1, t, cols) for t, cols in tables.items()}
    finally:
        v1.close()
    dst = tmp_path / "upgraded.sqlite3"
    capsys.readouterr()
    assert ih.main(["upgrade", "--from", str(src), "--to", str(dst)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["rows"] == {t: len(r) for t, r in before.items()} and report["rows"]["injury_observations"] > 0
    assert sha(src) == digest and ih.schema_state(dst) == "current"
    up = sqlite3.connect(dst)
    try:
        assert {t: ih._table_rows(up, t, cols) for t, cols in tables.items()} == before
        assert {r[0] for r in up.execute("SELECT kickoff_basis FROM injury_capture_games")} == {ih.PRE_PROVENANCE}
        assert up.execute("SELECT DISTINCT schedule_provenance, schedule_sha256_basis, "
                          "schedule_kickoff_review_json FROM injury_captures").fetchall() == [
            (ih.PRE_PROVENANCE, None, None)]
        assert up.execute("SELECT COUNT(*) FROM injury_capture_games WHERE source_kickoff_utc IS NOT NULL"
                          ).fetchone()[0] == 0
    finally:
        up.close()
    # Re-importing the same capture is a no-op: carried-over rows keep their basis.
    assert import_into(dst, new.path)[0]["created"] is False
    ctx = contexts(legacy_db["pred"], dst)
    assert ctx[PIT]["injuries"]["state"] == "none"
    assert any(ih.PRE_PROVENANCE in n for n in ctx[PIT]["injuries"]["notices"])
    # Rebuild path: a new database from the retained capture files restores provenance.
    rebuilt = tmp_path / "rebuilt.sqlite3"
    import_into(rebuilt, new.path, legacy_db["stale_path"])
    assert contexts(legacy_db["pred"], rebuilt)[PIT]["injuries"]["latest"]["capture_id"] == new.doc["capture_id"]


def test_upgrade_refusals_leave_files_untouched(legacy_db, tmp_path, monkeypatch):
    src = legacy_db["db"]
    digest = sha(src)
    existing = tmp_path / "existing.sqlite3"
    existing.write_bytes(b"keep me")
    with pytest.raises(ih.fr.SchemaRefused, match="already exists"):
        ih.upgrade_database(src, existing)
    assert existing.read_bytes() == b"keep me"
    current = tmp_path / "built.sqlite3"
    for not_legacy in (current, tmp_path / "missing.sqlite3"):
        with pytest.raises(ih.fr.SchemaRefused, match="only a legacy"):
            ih.upgrade_database(not_legacy, tmp_path / "out1.sqlite3")
    other = tmp_path / "other.sqlite3"
    c = sqlite3.connect(other)
    c.execute("CREATE TABLE t (x)")
    c.close()
    with pytest.raises(ih.fr.SchemaRefused):
        ih.upgrade_database(other, tmp_path / "out2.sqlite3")
    monkeypatch.setattr(ih, "PROVENANCE_DDL", "ALTER TABLE no_such_table ADD COLUMN x TEXT;")
    with pytest.raises(sqlite3.OperationalError):
        ih.upgrade_database(src, tmp_path / "out3.sqlite3")
    assert not any((tmp_path / f"out{i}.sqlite3").exists() for i in (1, 2, 3))
    assert sha(src) == digest


# ---------------------------------------------------------------- weather keeps its own checks


def test_weather_keeps_its_own_kickoff_check_under_corrected_identity(tmp_path):
    """A forecast captured for the stale kickoff is refused by weather's own
    kickoff check; the same capture matches only under the old stale identity."""
    row = (PIT, "2026", "11", "2026-11-22", "16:25", "PIT", "", "PHI", "", "Home", "outdoors",
           "PHI00", "Lincoln Financial Field")
    w = weather_capture(tmp_path, datetime(2026, 11, 20, 12, tzinfo=UTC),
                        FakeResp(200, body(start=datetime(2026, 11, 22, 18, tzinfo=UTC))), game_id=PIT, rows=(row,))
    assert w["usable"], w
    db = tmp_path / "w.sqlite3"
    weather_import(db, w)
    now = datetime(2026, 11, 21, 12, tzinfo=UTC)
    res = sc.build_all(pred_frame(), [PIT], environ=config(tmp_path, weather_db=db), now=now)
    ctx = res["contexts"][PIT]
    wx = ctx["weather"]
    assert wx["state"] == "error" and "doesn't match this game" in wx["message"] and wx["hours"] == []
    stale_identity = {**ctx["game"], "kickoff_utc": ctx["game"]["source_kickoff_utc"], "kickoff_review": None}
    assert sc.weather_context(db, stale_identity, now, now)["state"] == "ok"


# ---------------------------------------------------------------- collection-side helper


def test_helper_leaves_other_games_and_seasons_alone(tmp_path):
    frame = pred_frame()
    extra = pd.DataFrame([
        {"game_id": "2025_11_PIT_PHI", "season": 2025, "week": 11, "gameday": "2025-11-23", "gametime": "16:25",
         "away_team": "PIT", "home_team": "PHI"},
        {"game_id": "2026_11_pit_phi", "season": 2026, "week": 11, "gameday": "2026-11-22", "gametime": "16:25",
         "away_team": "PIT", "home_team": "PHI"}])
    frame = pd.concat([frame, extra], ignore_index=True)
    before = frame.copy(deep=True)
    out, records = kc.apply_to_frame(frame)
    pd.testing.assert_frame_equal(frame, before)
    changed = out.gametime != frame.gametime
    assert sorted(out.game_id[changed]) == [JAX, PIT] and {r["game_id"] for r in records} == {PIT, JAX}
    assert kc.apply_to_frame(frame[~frame.game_id.isin([PIT, JAX])])[1] == []


# ---------------------------------------------------------------- panel rendering


def test_panel_renders_reviewed_kickoff_and_corrected_capture(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from test_stored_context import all_text, panel_app
    pred = pred_frame()
    new = corrected_capture(tmp_path, pred)
    stale_path, stale = stale_capture(tmp_path)
    db = tmp_path / "injury.sqlite3"
    import_into(db, new.path, stale_path)
    csv = tmp_path / "games.csv"
    pred.to_csv(csv, index=False)
    monkeypatch.setenv(sc.CONFIG_ENV, config(tmp_path, injury_db=db)[sc.CONFIG_ENV])
    at = AppTest.from_function(panel_app, args=(str(csv), WED_PANEL.isoformat()), default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    text = all_text(at)
    assert "reviewed Week 11 schedule change" in text and "04:25 PM EST" in text
    assert "Kickoff Sun Nov 22, 2026 01:00 PM EST" in text
    assert new.doc["capture_id"] in text and kc.WEEK11_AUTHORITY in text
