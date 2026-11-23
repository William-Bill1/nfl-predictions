"""Reviewed 2026 Week 12 kickoff corrections (November 29, 2026), alongside the
Week 11 ones: the shared rule, collection and validation, pregame boundaries,
historical review blocks, the panel identity and the weekly market context.
Synthetic schedules in temporary folders; no network.
"""
import copy
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import injury_reports as ir  # noqa: E402
import kickoff_corrections as kc  # noqa: E402
import stored_context as sc  # noqa: E402
from scripts import football_market_context as market  # noqa: E402
from test_injury_reports import Resp, Source, _payload, run  # noqa: E402

UTC = timezone.utc
TEN, NE, LV = "2026_12_TEN_JAX", "2026_12_NE_LAC", "2026_12_LV_CLE"
PIT, JAX = "2026_11_PIT_PHI", "2026_11_JAX_NYG"
OTHER = "2026_12_BUF_PIT"              # unreviewed Week 12 game, unchanged throughout
# game_id: (away, home, pre-change time, reviewed time, reviewed kickoff UTC)
WEEK12 = {TEN: ("TEN", "JAX", "16:05", "13:00", "2026-11-29T18:00:00Z"),
          NE: ("NE", "LAC", "20:20", "16:05", "2026-11-29T21:05:00Z"),
          LV: ("LV", "CLE", "13:00", "20:20", "2026-11-30T01:20:00Z")}
TEAMS = ["TEN", "JAX", "NE", "LAC", "LV", "CLE", "BUF", "PIT"]
WED = datetime(2026, 11, 25, 17, 5, tzinfo=UTC)          # Wednesday noon slot (EST)


def rows(corrected=False):
    out = [{"game_id": gid, "season": 2026, "week": 12, "gameday": "2026-11-29",
            "gametime": new if corrected else old, "away_team": away, "home_team": home,
            "away_score": None, "home_score": None}
           for gid, (away, home, old, new, _) in WEEK12.items()]
    out.append({"game_id": OTHER, "season": 2026, "week": 12, "gameday": "2026-11-29", "gametime": "13:00",
                "away_team": "BUF", "home_team": "PIT", "away_score": None, "home_score": None})
    return out


def schedule(tmp_path, corrected=False, changes=None, name="schedule.tsv"):
    df = pd.DataFrame(rows(corrected))
    for (gid, column), value in (changes or {}).items():
        df.loc[df.game_id == gid, column] = value
    path = tmp_path / name
    df.to_csv(path, sep="\t", index=False)
    return path


# ---------------------------------------------------------------- the shared rule

@pytest.mark.parametrize("gid", list(WEEK12))
def test_each_week12_correction_resolves_with_its_own_authority(gid):
    away, home, old, new, effective = WEEK12[gid]
    rec = kc.resolve(gid, season=2026, week=12, away_team=away, home_team=home,
                     gameday="2026-11-29", gametime=old)
    assert rec["status"] == kc.APPLIED and rec["effective_gametime"] == new
    assert rec["effective_kickoff_utc"] == effective
    assert rec["authority"] == kc.WEEK12_AUTHORITY != kc.WEEK11_AUTHORITY
    assert "Week%2012%20Schedule%20Change.pdf" in rec["authority"]
    already = kc.resolve(gid, season=2026, week=12, away_team=away, home_team=home,
                         gameday="2026-11-29", gametime=new)
    assert already["status"] == kc.SOURCE_ALREADY_CORRECTED and already["effective_kickoff_utc"] == effective


def test_week11_records_keep_week11_authority():
    for gid in (PIT, JAX):
        r = kc.REVIEWED[gid]
        assert kc.expected_record(gid, r.old_gametime)["authority"] == kc.WEEK11_AUTHORITY


def test_apply_to_frame_corrects_stale_and_keeps_corrected_input():
    stale = pd.DataFrame(rows())
    before = stale.copy(deep=True)
    out, records = kc.apply_to_frame(stale)
    pd.testing.assert_frame_equal(stale, before)
    assert dict(zip(out.game_id, out.gametime)) == {TEN: "13:00", NE: "16:05", LV: "20:20", OTHER: "13:00"}
    assert {r["game_id"]: r["status"] for r in records} == {g: kc.APPLIED for g in WEEK12}
    corrected = pd.DataFrame(rows(corrected=True))
    out2, records2 = kc.apply_to_frame(corrected)
    pd.testing.assert_frame_equal(out2, corrected)
    assert {r["status"] for r in records2} == {kc.SOURCE_ALREADY_CORRECTED}


@pytest.mark.parametrize("gid", list(WEEK12))
@pytest.mark.parametrize("column,value", [("gametime", "18:30"), ("gameday", "2026-11-30"),
                                          ("week", 13), ("season", 2025), ("away_team", "MIA"),
                                          ("home_team", "MIA"), ("gametime", None)])
def test_unexpected_identity_or_time_needs_review(gid, column, value):
    df = pd.DataFrame(rows())
    df.loc[df.game_id == gid, column] = value
    with pytest.raises(kc.KickoffReviewError, match="review required"):
        kc.apply_to_frame(df)


def test_other_reviewed_time_is_not_accepted_for_a_different_game():
    # TEN-JAX's old 16:05 is NE-LAC's reviewed time; each game only accepts its own two times.
    df = pd.DataFrame(rows())
    df.loc[df.game_id == LV, "gametime"] = "16:05"
    with pytest.raises(kc.KickoffReviewError, match="review required"):
        kc.apply_to_frame(df)


# ---------------------------------------------------------------- review blocks: new and historical

def mixed_records():
    return [kc.expected_record(PIT, "16:25"), kc.expected_record(JAX, "16:25"),
            *(kc.expected_record(g, WEEK12[g][2]) for g in WEEK12)]


def test_new_block_lists_each_records_authority():
    block = kc.review_block(mixed_records())
    assert block["review_version"] == 2 and set(block) == {"review_version", "authorities", "games"}
    assert block["authorities"] == sorted([kc.WEEK11_AUTHORITY, kc.WEEK12_AUTHORITY])
    assert set(kc.check_review_block(block)) == {PIT, JAX, *WEEK12}
    only12 = kc.review_block([kc.expected_record(TEN, "16:05")])
    assert only12["authorities"] == [kc.WEEK12_AUTHORITY]
    assert kc.check_review_block(kc.review_block([])) == {}


def v1_block(records):
    return {"review_version": 1, "authority": kc.WEEK11_AUTHORITY, "games": records}


def test_historical_version1_block_still_validates():
    records = [kc.expected_record(PIT, "16:25"), kc.expected_record(JAX, "13:00")]
    assert set(kc.check_review_block(v1_block(records))) == {PIT, JAX}


@pytest.mark.parametrize("mutate,msg", [
    (lambda b: b["games"].append(kc.expected_record(TEN, "16:05")), "version 1"),
    (lambda b: b.__setitem__("authority", kc.WEEK12_AUTHORITY), "authority differs"),
    (lambda b: b.__setitem__("authorities", [kc.WEEK11_AUTHORITY]), "fields"),
    (lambda b: b.__setitem__("review_version", 3), "review_version"),
])
def test_version1_block_changes_refused(mutate, msg):
    block = v1_block([kc.expected_record(PIT, "16:25")])
    mutate(block)
    with pytest.raises(kc.KickoffReviewError, match=msg):
        kc.check_review_block(block)


@pytest.mark.parametrize("mutate,msg", [
    (lambda b: b.__setitem__("authorities", [kc.WEEK12_AUTHORITY]), "authority list"),
    (lambda b: b.__setitem__("authorities", sorted([kc.WEEK11_AUTHORITY, kc.WEEK12_AUTHORITY, "x"])),
     "authority list"),
    (lambda b: b["games"][2].__setitem__("authority", kc.WEEK11_AUTHORITY), "differs"),
    (lambda b: b.__setitem__("authority", kc.WEEK11_AUTHORITY), "fields"),
])
def test_version2_block_changes_refused(mutate, msg):
    block = kc.review_block(mixed_records())
    mutate(block)
    with pytest.raises(kc.KickoffReviewError, match=msg):
        kc.check_review_block(block)


def test_retained_reviewed_captures_validate_unchanged():
    caps = sorted((ROOT / "data_files" / "injury_reports" / "captures").glob("*.json"))
    reviewed = []
    for p in caps:
        before = p.read_bytes()
        doc = json.loads(before.decode("utf-8"))
        ir.validate_capture(doc, str(p))
        if ir.schedule_kind(doc) == "reviewed":
            reviewed.append(doc)
        assert p.read_bytes() == before
    v1 = [d for d in reviewed if d["schedule"]["kickoff_review"]["review_version"] == 1]
    assert v1, "the retained Week 11 reviewed capture is expected"
    for d in v1:
        assert d["schedule"]["kickoff_review"]["authority"] == kc.WEEK11_AUTHORITY


# ---------------------------------------------------------------- collection, validation, pregame boundaries

def test_capture_corrects_stale_week12_and_records_authority(tmp_path):
    path = schedule(tmp_path)
    before = path.read_bytes()
    result = run(path, tmp_path / "caps", Source(Resp(_payload(TEAMS))), WED)
    doc = result.doc
    assert doc["scope"]["week"] == 12
    games = {g["game_id"]: g for g in doc["scope"]["games"]}
    assert {g: games[g]["kickoff_utc"] for g in WEEK12} == {g: v[4] for g, v in WEEK12.items()}
    assert games[OTHER]["kickoff_utc"] == "2026-11-29T18:00:00Z"
    block = doc["schedule"]["kickoff_review"]
    assert block["review_version"] == 2 and block["authorities"] == [kc.WEEK12_AUTHORITY]
    assert {r["game_id"]: (r["status"], r["authority"]) for r in block["games"]} == {
        g: (kc.APPLIED, kc.WEEK12_AUTHORITY) for g in WEEK12}
    ir.validate_capture(doc)
    assert path.read_bytes() == before


def test_capture_from_corrected_source(tmp_path):
    result = run(schedule(tmp_path, corrected=True), tmp_path / "caps", Source(Resp(_payload(TEAMS))), WED)
    block = result.doc["schedule"]["kickoff_review"]
    assert {r["status"] for r in block["games"]} == {kc.SOURCE_ALREADY_CORRECTED}
    ir.validate_capture(result.doc)


def test_unexpected_week12_schedule_refused_before_request(tmp_path):
    path = schedule(tmp_path, changes={(NE, "gametime"): "19:00"})
    src = Source(Resp(_payload(TEAMS)))
    with pytest.raises(ir.ValidationError, match="review required"):
        run(path, tmp_path / "caps", src, WED)
    assert not src.calls and not (tmp_path / "caps").exists()


def test_week12_capture_with_stale_scope_kickoff_refused(tmp_path):
    doc = copy.deepcopy(run(schedule(tmp_path), tmp_path / "caps", Source(Resp(_payload(TEAMS))), WED).doc)
    next(g for g in doc["scope"]["games"] if g["game_id"] == LV)["kickoff_utc"] = "2026-11-29T18:00:00Z"
    doc[ir.CHECKSUM_FIELD] = ir.ps.payload_checksum(doc)
    with pytest.raises(ir.ValidationError, match="reviewed effective kickoff"):
        ir.validate_capture(doc)


@pytest.mark.parametrize("gid", list(WEEK12))
@pytest.mark.parametrize("corrected", [False, True])
def test_pregame_boundary_follows_reviewed_kickoff(tmp_path, gid, corrected):
    frame, _, _ = ir.load_injury_schedule(schedule(tmp_path, corrected=corrected))
    kickoff = datetime.fromisoformat(WEEK12[gid][4].replace("Z", "+00:00"))

    def state(at):
        _, games = ir.scope_games(frame, at)
        return {g["game_id"]: g["state"] for g in games}.get(gid)
    assert state(kickoff - timedelta(seconds=1)) == "pregame"
    assert state(kickoff) != "pregame"


def test_moved_late_game_is_still_pregame_after_its_old_time(tmp_path):
    # LV-CLE was 13:00 (18:00Z); at 19:00Z it is still pregame under the reviewed 20:20 kickoff.
    frame, _, _ = ir.load_injury_schedule(schedule(tmp_path))
    _, games = ir.scope_games(frame, datetime(2026, 11, 29, 19, 0, tzinfo=UTC))
    states = {g["game_id"]: g["state"] for g in games}
    assert states[LV] == "pregame" and states[TEN] != "pregame" and states[OTHER] != "pregame"


def test_version1_week12_capture_stays_valid_but_never_matches_as_reviewed(tmp_path):
    """A Week 12-scope capture written by the pre-change code (version 1 block, which
    could not know these games) stays valid and imports as "source"; the panel then
    excludes it for the reviewed games instead of treating it as a reviewed capture."""
    from test_injury_kickoff_review import game_rows, import_into, reseal
    from test_stored_context import config
    path = schedule(tmp_path, corrected=True)
    new = run(path, tmp_path / "caps", Source(Resp(_payload(TEAMS))), WED)
    v1 = reseal(new.doc, lambda d: d["schedule"].__setitem__(
        "kickoff_review", {"review_version": 1, "authority": kc.WEEK11_AUTHORITY, "games": []}))
    v1 = reseal(v1, lambda d: d.__setitem__("capture_id", d["capture_id"][:-12] + "0" * 12))
    ir.validate_capture(v1)
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    old_path = old_dir / f"{v1['capture_id']}.json"
    old_path.write_text(json.dumps(v1), encoding="utf-8")
    db = tmp_path / "injury.sqlite3"
    import_into(db, old_path)
    stored = game_rows(db)
    assert {g: stored[(v1["capture_id"], g)][2] for g in (*WEEK12, OTHER)} == {g: "source" for g in (*WEEK12, OTHER)}
    table = pd.DataFrame(rows(corrected=True))
    now = datetime(2026, 11, 25, 18, 0, tzinfo=UTC)
    ctx = sc.build_all(table, [TEN, OTHER], environ=config(tmp_path, injury_db=db), now=now)["contexts"]
    assert ctx[TEN]["injuries"].get("latest") is None
    assert any("kickoff review" in n for n in ctx[TEN]["injuries"]["notices"])
    assert ctx[OTHER]["injuries"]["latest"]["capture_id"] == v1["capture_id"]
    import_into(db, new.path)                                   # a version 2 capture does match
    ctx = sc.build_all(table, [TEN], environ=config(tmp_path, injury_db=db), now=now)["contexts"]
    assert ctx[TEN]["injuries"]["latest"]["capture_id"] == new.doc["capture_id"]


def test_version1_capture_still_needs_week11_records(tmp_path):
    """The version 1 allowance covers only games it could not know: a Week 11 game
    in scope without its record is still refused."""
    rows11 = [{"game_id": PIT, "season": 2026, "week": 11, "gameday": "2026-11-22", "gametime": "16:25",
               "away_team": "PIT", "home_team": "PHI", "away_score": None, "home_score": None}]
    path = tmp_path / "s11.tsv"
    pd.DataFrame(rows11).to_csv(path, sep="\t", index=False)
    doc = run(path, tmp_path / "caps", Source(Resp(_payload(["PIT", "PHI"]))),
              datetime(2026, 11, 18, 17, 5, tzinfo=UTC)).doc
    from test_injury_kickoff_review import reseal
    bad = reseal(doc, lambda d: d["schedule"].__setitem__(
        "kickoff_review", {"review_version": 1, "authority": kc.WEEK11_AUTHORITY, "games": []}))
    with pytest.raises(ir.ValidationError, match="without its kickoff_review record"):
        ir.validate_capture(bad)


# ---------------------------------------------------------------- panel identity and market context

@pytest.mark.parametrize("corrected", [False, True])
def test_panel_identity_uses_reviewed_kickoff(corrected):
    table = pd.DataFrame(rows(corrected))
    before = table.copy(deep=True)
    for gid, (_, _, old, new, effective) in WEEK12.items():
        ident = sc.game_identity(table, gid)
        assert ident["kickoff_utc"].strftime("%Y-%m-%dT%H:%M:%SZ") == effective
        assert ident["kickoff_review"]["authority"] == kc.WEEK12_AUTHORITY
        assert ident["kickoff_review"]["status"] == (kc.SOURCE_ALREADY_CORRECTED if corrected else kc.APPLIED)
    assert sc.game_identity(table, OTHER)["kickoff_review"] is None
    pd.testing.assert_frame_equal(table, before)


def test_panel_identity_refuses_unexpected_week12_row():
    table = pd.DataFrame(rows())
    table.loc[table.game_id == TEN, "gametime"] = "17:00"
    with pytest.raises(sc.GameIdentityError, match="review required"):
        sc.game_identity(table, TEN)


def loaded_market(corrected, captured_at="2026-11-25T12:00:00Z"):
    return {"schedule": pd.DataFrame(rows(corrected)), "model": {"captured_at": captured_at, "games": []},
            "groups": {}, "days": [], "history": {"as_of": "2026-11-29T19:00:00Z"}, "hashes": {}}


@pytest.mark.parametrize("gid", list(WEEK12))
def test_market_context_refuses_stale_week12_schedule(gid):
    away, home = WEEK12[gid][:2]
    with pytest.raises(ValueError, match="pre-change kickoff"):
        market.for_game(loaded_market(False), gid, away, home, 2026, 12, gameday="2026-11-29")
    assert market.for_game(loaded_market(True), gid, away, home, 2026, 12, gameday="2026-11-29")["sides"]


def test_market_model_timing_uses_reviewed_week12_kickoff():
    late = loaded_market(True, captured_at="2026-11-29T19:00:00Z")   # after TEN-JAX, before LV-CLE
    with pytest.raises(ValueError, match="not pregame"):
        market.for_game(late, TEN, "TEN", "JAX", 2026, 12, gameday="2026-11-29")
    assert market.for_game(late, LV, "LV", "CLE", 2026, 12, gameday="2026-11-29")["sides"]
