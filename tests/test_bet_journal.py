"""Bet journal: actual single-game spread wagers placed at Ontario sportsbooks
(bet_journal.py and pages/8_Bet_Journal.py).

Temporary journal and schedule only; every socket connection and requests
call raises; the clock is fixed.
"""

import hashlib
import json
import socket
import threading
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
import requests

import bet_journal as bj
import betting_log
import ontario_spreads as on

UTC = timezone.utc
ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "pages" / "8_Bet_Journal.py"

# Wednesday Oct 7 2026, 13:00 Toronto (EDT).
NOW = datetime(2026, 10, 7, 17, 0, tzinfo=UTC)
KC_BUF = "2026_04_KC_BUF"      # Sun Oct 4 16:25 EDT = 20:25 UTC; final KC 24, BUF 27
CHI_GB = "2026_05_CHI_GB"      # Sun Oct 11 13:00 EDT = 17:00 UTC; not played yet
HEADER = ("season\tweek\tgame_id\tgameday\tgametime\taway_team\thome_team\taway_score"
          "\thome_score\tresult\ttotal\tovertime\n")


def schedule_text(kc_buf=(24, 27), **fields):
    """The fixture schedule. KC @ BUF's result/total/overtime follow its
    scores (as in nflverse) unless overridden through `fields`."""
    a, h = kc_buf
    done = isinstance(a, int) and isinstance(h, int)
    f = {"result": h - a if done else None, "total": h + a if done else None,
         "overtime": 0 if done else None, **fields}
    cell = lambda v: "" if v is None else str(v)  # noqa: E731
    return (HEADER
            + f"2026\t4\t{KC_BUF}\t2026-10-04\t16:25\tKC\tBUF\t{cell(a)}\t{cell(h)}\t"
            + f"{cell(f['result'])}\t{cell(f['total'])}\t{cell(f['overtime'])}\n"
            + f"2026\t5\t{CHI_GB}\t2026-10-11\t13:00\tCHI\tGB\t\t\t\t\t\n"
            + "2026\t9\t2026_09_DAL_NYG\t2026-11-08\t13:00\tDAL\tNYG\t\t\t\t\t\n")


@pytest.fixture
def jr(tmp_path, monkeypatch):
    """Temporary journal + schedule, fixed clock, network blocked."""
    def blocked(*a, **k):
        raise AssertionError("network call attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(requests, "get", blocked)
    monkeypatch.setattr(requests, "post", blocked)
    journal, sched = tmp_path / "bet_journal", tmp_path / "nfl_games_historical.csv"
    sched.write_text(schedule_text(), encoding="utf-8")
    monkeypatch.setattr(bj, "JOURNAL_DIR", journal)
    monkeypatch.setattr(bj, "SCHEDULE_PATH", sched)
    monkeypatch.setattr(bj, "now_utc", lambda: NOW)

    class J:
        dir, schedule = journal, sched

        def files(self, kind=None):
            if not journal.exists():
                return []
            out = sorted(p for p in journal.iterdir() if not p.name.startswith("."))
            return [p for p in out if kind is None or p.name.startswith(bj.PREFIX[kind])]

        def set_scores(self, scores, **fields):
            sched.write_text(schedule_text(scores, **fields), encoding="utf-8")

    return J()


def terms(jr, **kw):
    a = dict(sportsbook_key="betmgm_ca_on", sportsbook_name=None, game_id=KC_BUF, team="BUF",
             handicap=-2.5, odds=-110, stake="50.00",
             placed_at=datetime(2026, 10, 4, 16, 0, tzinfo=UTC), reference=None, note=None,
             now=NOW, schedule_path=jr.schedule)
    a.update(kw)
    return bj.build_terms(**a)


def record(jr, confirmed=(), at=NOW, **kw):
    t, sha = terms(jr, **kw)
    doc = bj.prepare_wager(t, sha, confirmed_existing=list(confirmed), now=at)
    bj.save_wager(doc, jr.dir)
    return doc


def state(jr):
    return bj.current_state(bj.load_records(jr.dir))


def tip(jr, wager_id):
    """The wager's current effective terms record (or the ID itself if the
    wager isn't in the journal)."""
    s = state(jr).get(wager_id) if jr.dir.exists() else None
    return s.terms_record_id if s else wager_id


def prep_amend(jr, wager_id, terms_, sha, reason, *, now=NOW, confirmed=()):
    return bj.prepare_amendment(wager_id, tip(jr, wager_id), terms_, sha, reason,
                                confirmed_existing=list(confirmed), now=now)


def prep_void(jr, wager_id, reason, *, now=NOW):
    return bj.prepare_void(wager_id, tip(jr, wager_id), reason, now=now)


# ------------------------------------------------------------- settlement --

class TestSettlement:
    # Final: KC 24 @ BUF 27 -> BUF won by 3.
    @pytest.mark.parametrize("team,h,result", [
        ("BUF", -2.5, "win"), ("BUF", -3, "push"), ("BUF", -3.5, "loss"), ("BUF", 0, "win"),
        ("BUF", +1.5, "win"),
        ("KC", +2.5, "loss"), ("KC", +3, "push"), ("KC", +3.5, "win"), ("KC", 0, "loss"),
        ("KC", -1.5, "loss"),
    ])
    def test_both_sides_and_spread_signs(self, jr, team, h, result):
        t, _ = terms(jr, team=team, handicap=h)
        assert t["team_side"] == ("home" if team == "BUF" else "away")
        assert bj.settle(t, 27, 24)[0] == result

    @pytest.mark.parametrize("odds,stake,win", [
        (-110, "50.00", "45.45"), (-110, "110.00", "100.00"), (+150, "20.00", "30.00"),
        (-105, "33.33", "31.74"), (+105, "10.01", "10.51"), (+105, "0.10", "0.11"),  # half-up
        (-100, "7.00", "7.00"), (+100, "7.00", "7.00"), (-250, "25.00", "10.00"),
        (+340, "12.50", "42.50"),
    ])
    def test_win_loss_push_profit_scaled_to_stake(self, jr, odds, stake, win):
        base = dict(odds=odds, stake=stake)
        assert bj.settle(terms(jr, handicap=-2.5, **base)[0], 27, 24) == ("win", Decimal(win))
        assert bj.settle(terms(jr, handicap=-3.5, **base)[0], 27, 24) == \
            ("loss", -Decimal(stake))
        assert bj.settle(terms(jr, handicap=-3, **base)[0], 27, 24) == ("push", Decimal("0.00"))

    def test_uses_reviewed_betting_log_rule(self, jr, monkeypatch):
        calls = []
        real = betting_log.spread_result
        monkeypatch.setattr(betting_log, "spread_result",
                            lambda *a: calls.append(a) or real(*a))
        bj.settle(terms(jr)[0], 27, 24)
        assert calls == [("BUF", "BUF", "KC", -2.5, 27, 24)]


# ------------------------------------------------------------- validation --

class TestTermsValidation:
    def test_valid_terms(self, jr):
        t, sha = terms(jr, stake="$25", reference=" 123-ABC ", note="  ")
        assert t["stake_cad"] == "25.00" and t["reference"] == "123-ABC" and t["note"] is None
        assert t["placed_at"] == "2026-10-04T16:00:00Z" and t["jurisdiction"] == "CA-ON"
        assert t["kickoff_utc"] == "2026-10-04T20:25:00Z"
        assert sha == hashlib.sha256(jr.schedule.read_bytes()).hexdigest()

    @pytest.mark.parametrize("kw,match", [
        ({"stake": "0"}, "more than"), ({"stake": "-5"}, "more than"),
        ({"stake": "10.005"}, "2 decimals"), ({"stake": "abc"}, "not an amount"),
        ({"stake": "NaN"}, "2 decimals"), ({"stake": "100000.01"}, "at most"),
        ({"odds": -99}, "American"), ({"odds": 0}, "American"), ({"odds": None}, "American"),
        ({"handicap": 2.25}, "half-point"), ({"handicap": 61}, "half-point"),
        ({"team": "DAL"}, "not playing"), ({"game_id": "nope"}, "not found"),
        ({"sportsbook_key": "fanduel"}, "unknown sportsbook"),
        ({"sportsbook_key": "other_on", "sportsbook_name": " "}, "name the"),
    ])
    def test_rejections(self, jr, kw, match):
        with pytest.raises(bj.JournalError, match=match):
            terms(jr, **kw)

    def test_other_sportsbook_needs_and_keeps_name(self, jr):
        t, _ = terms(jr, sportsbook_key="other_on", sportsbook_name=" NorthStar Bets ")
        assert t["sportsbook_name"] == "NorthStar Bets"
        t, _ = terms(jr, sportsbook_key="betmgm_ca_on", sportsbook_name="ignored")
        assert t["sportsbook_name"] == "BetMGM (CA - ON)"

    def test_late_entry_of_pregame_wager_allowed(self, jr):
        doc = record(jr)                        # placed Oct 4, recorded Oct 7 after the game
        assert doc["terms"]["placed_at"] < doc["terms"]["kickoff_utc"] < doc["recorded_at"]

    def test_future_placement_rejected_now_allowed(self, jr):
        with pytest.raises(bj.JournalError, match="future"):
            terms(jr, game_id=CHI_GB, team="CHI", placed_at=NOW + timedelta(minutes=1))
        t, _ = terms(jr, game_id=CHI_GB, team="CHI", placed_at=NOW + timedelta(seconds=40))
        assert t["placed_at"] == "2026-10-07T17:00:00Z"          # same minute as now

    def test_kickoff_boundaries(self, jr):
        kickoff = datetime(2026, 10, 4, 20, 25, tzinfo=UTC)
        for at in (kickoff, kickoff + timedelta(minutes=1), kickoff + timedelta(seconds=30)):
            with pytest.raises(bj.JournalError, match="not before kickoff"):
                terms(jr, placed_at=at)
        t, _ = terms(jr, placed_at=kickoff - timedelta(seconds=1))   # truncated to 20:24
        assert t["placed_at"] == "2026-10-04T20:24:00Z"

    def test_naive_placement_rejected(self, jr):
        with pytest.raises(bj.JournalError, match="timezone"):
            terms(jr, placed_at=datetime(2026, 10, 4, 12, 0))

    def test_toronto_dst(self):
        assert bj.toronto_to_utc(date(2026, 10, 4), time(12, 0)) == \
            datetime(2026, 10, 4, 16, 0, tzinfo=UTC)                         # EDT
        assert bj.toronto_to_utc(date(2026, 11, 8), time(12, 0)) == \
            datetime(2026, 11, 8, 17, 0, tzinfo=UTC)                         # EST
        with pytest.raises(bj.JournalError, match="happens twice"):
            bj.toronto_to_utc(date(2026, 11, 1), time(1, 30))
        with pytest.raises(bj.JournalError, match="doesn't exist"):
            bj.toronto_to_utc(date(2026, 3, 8), time(2, 30))


# ------------------------------------------------- records and duplicates --

class TestRecords:
    def test_prepare_writes_nothing(self, jr):
        t, sha = terms(jr)
        bj.prepare_wager(t, sha, confirmed_existing=[], now=NOW)
        assert jr.files() == [] and not jr.dir.exists()

    def test_record_schema_and_checksum(self, jr):
        doc = record(jr)
        [path] = jr.files()
        stored = json.loads(path.read_text())
        assert path.stem == doc["record_id"] and stored == doc
        assert set(stored) == {"schema_version", "kind", "record_id", "recorded_at",
                               "code_revision", "payload_sha256", "terms",
                               "confirmed_existing", "schedule"}
        assert stored["kind"] == bj.KIND_WAGER and stored["schema_version"] == 1
        assert stored["payload_sha256"] == on.ps.payload_checksum(stored)
        assert not list(jr.dir.glob(".*"))                       # lock and temp files gone

    def test_duplicate_refused_across_reruns(self, jr):
        record(jr)
        with pytest.raises(bj.DuplicateWager, match="identical wager"):
            record(jr, at=NOW + timedelta(seconds=5))             # rerun / resubmission
        assert len(jr.files()) == 1

    def test_same_doc_saved_twice_refused(self, jr):
        t, sha = terms(jr)
        doc = bj.prepare_wager(t, sha, confirmed_existing=[], now=NOW)
        bj.save_wager(doc, jr.dir)
        with pytest.raises(bj.DuplicateWager):
            bj.save_wager(doc, jr.dir)
        assert len(jr.files()) == 1

    def test_note_does_not_make_a_new_wager(self, jr):
        record(jr)
        with pytest.raises(bj.DuplicateWager):
            record(jr, note="typed again")

    def test_intentional_identical_second_wager(self, jr):
        w1 = record(jr)
        w2 = record(jr, confirmed=[w1["record_id"]])
        assert w2["confirmed_existing"] == [w1["record_id"]]
        # A third needs confirming against both; confirming only one is refused.
        with pytest.raises(bj.DuplicateWager, match="changed since preview"):
            record(jr, confirmed=[w1["record_id"]])
        with pytest.raises(bj.DuplicateWager):
            record(jr, confirmed=[w1["record_id"], "W-20261007T170000Z-000000000000"])
        record(jr, confirmed=sorted([w1["record_id"], w2["record_id"]]))
        assert bj.totals(state(jr))["wagers"] == 3

    def test_confirmed_second_wager_refused_if_first_voided_meanwhile(self, jr):
        w1 = record(jr)
        bj.save_void(prep_void(jr, w1["record_id"], "entered by mistake", now=NOW), jr.dir)
        with pytest.raises(bj.DuplicateWager, match="changed since preview.*now: none"):
            record(jr, confirmed=[w1["record_id"]])

    def test_different_terms_are_not_duplicates(self, jr):
        record(jr)
        record(jr, stake="50.01")
        record(jr, placed_at=datetime(2026, 10, 4, 16, 1, tzinfo=UTC))
        record(jr, sportsbook_key="proline_ca_on")
        assert len(jr.files()) == 4

    def test_references(self, jr):
        record(jr, reference="T-1")
        # Identical terms but a different ticket: a different wager.
        record(jr, reference="T-2")
        # A reused ticket at the same book is refused even with other terms.
        with pytest.raises(bj.DuplicateWager, match="reference"):
            record(jr, reference="T-1", stake="10.00")
        # Same ticket text at another sportsbook is fine.
        record(jr, reference="T-1", sportsbook_key="proline_ca_on")
        assert len(jr.files()) == 3

    def test_concurrent_identical_saves_write_once(self, jr):
        t, sha = terms(jr)
        docs = [bj.prepare_wager(t, sha, confirmed_existing=[], now=NOW + timedelta(seconds=i))
                for i in range(8)]
        outcomes = []

        def save(d):
            try:
                bj.save_wager(d, jr.dir)
                outcomes.append("saved")
            except bj.JournalBusy:
                outcomes.append("busy")
            except bj.DuplicateWager:
                outcomes.append("duplicate")

        threads = [threading.Thread(target=save, args=(d,)) for d in docs]
        [th.start() for th in threads]
        [th.join() for th in threads]
        assert outcomes.count("saved") == 1 and len(jr.files()) == 1
        assert not list(jr.dir.glob(".*"))

    def test_busy_lock_is_reported_and_kept(self, jr):
        jr.dir.mkdir(parents=True)
        lock = jr.dir / ".journal_write.lock"
        lock.write_text('{"what": "bet-journal write", "id": "x", "pid": 42, "host": "h"}')
        with pytest.raises(bj.JournalBusy, match="only after confirming that process 42"):
            record(jr)
        assert lock.exists() and jr.files() == []


# ------------------------------------------------ amendments, voids, totals --

class TestCorrections:
    def test_amendment_is_linked_and_append_only(self, jr):
        w = record(jr)
        wager_file = jr.dir / f"{w['record_id']}.json"
        original = wager_file.read_bytes()
        t, sha = terms(jr, stake="75.00")
        a = prep_amend(jr, w["record_id"], t, sha, "stake typo", now=NOW)
        bj.save_amendment(a, jr.dir)
        assert wager_file.read_bytes() == original                       # untouched
        s = state(jr)[w["record_id"]]
        assert s.terms["stake_cad"] == "75.00" and s.terms_record_id == a["record_id"]
        assert bj.totals(state(jr))["wagers"] == 1

    @pytest.mark.parametrize("make", ["amend", "void"])
    def test_reason_required(self, jr, make):
        w = record(jr)
        with pytest.raises(bj.JournalError, match="reason"):
            if make == "amend":
                prep_amend(jr, w["record_id"], *terms(jr, stake="1.00"), "  ", now=NOW)
            else:
                prep_void(jr, w["record_id"], "", now=NOW)

    def test_amendment_rules(self, jr):
        w = record(jr)
        same = prep_amend(jr, w["record_id"], *terms(jr), "nothing", now=NOW)
        with pytest.raises(bj.JournalError, match="same as the current"):
            bj.save_amendment(same, jr.dir)
        other_game = prep_amend(jr,
            w["record_id"], *terms(jr, game_id=CHI_GB, team="CHI",
                                   placed_at=datetime(2026, 10, 7, 16, 0, tzinfo=UTC)),
            "wrong game", now=NOW)
        with pytest.raises(bj.JournalError, match="can't change the game"):
            bj.save_amendment(other_game, jr.dir)
        missing = prep_amend(jr, "W-20261007T170000Z-000000000000",
                                       *terms(jr, stake="1.00"), "x", now=NOW)
        with pytest.raises(bj.JournalError, match="not found"):
            bj.save_amendment(missing, jr.dir)

    def test_void_excluded_from_totals_but_kept(self, jr):
        w1, w2 = record(jr), record(jr, stake="20.00")
        bj.save_void(prep_void(jr, w1["record_id"], "never placed", now=NOW), jr.dir)
        st_ = state(jr)
        tot = bj.totals(st_)
        assert tot["wagers"] == 1 and tot["voided"] == 1
        assert tot["stake_pending_cad"] == Decimal("20.00")
        assert [r["Wager ID"] for r in bj.current_rows(st_)] == [w2["record_id"]]
        assert len(bj.current_rows(st_, include_void=True)) == 2
        assert len(bj.history_rows(bj.load_records(jr.dir))) == 3
        with pytest.raises(bj.JournalError, match="already void"):
            bj.save_void(prep_void(jr, w1["record_id"], "again", now=NOW), jr.dir)
        with pytest.raises(bj.JournalError, match="is void"):
            bj.save_amendment(prep_amend(jr, w1["record_id"], *terms(jr, stake="1.00"),
                                                   "x", now=NOW), jr.dir)

    def test_totals_and_roi(self, jr):
        # BUF won 27-24. Win -2.5 at -110 for $110 (+100), loss -3.5 for $50 (-50),
        # push -3 for $40 (0), win KC +3.5 at +150 for $20 (+30), pending CHI $60,
        # voided $1000.
        record(jr, stake="110.00")
        record(jr, handicap=-3.5, stake="50.00")
        record(jr, handicap=-3, stake="40.00")
        record(jr, team="KC", handicap=3.5, odds=150, stake="20.00")
        record(jr, game_id=CHI_GB, team="GB", handicap=-6.5,
               placed_at=datetime(2026, 10, 7, 12, 0, tzinfo=UTC), stake="60.00")
        v = record(jr, handicap=-1.5, stake="1000.00")
        bj.save_void(prep_void(jr, v["record_id"], "test", now=NOW), jr.dir)
        bj.grade(jr.dir, jr.schedule, now=NOW)
        tot = bj.totals(state(jr))
        assert (tot["win"], tot["loss"], tot["push"], tot["pending"], tot["voided"]) == \
            (2, 1, 1, 1, 1)
        assert tot["net_profit_cad"] == Decimal("80.00")
        assert tot["stake_graded_cad"] == Decimal("220.00")
        assert tot["stake_pending_cad"] == Decimal("60.00")
        assert tot["roi_pct"] == Decimal("36.36")                     # 80 / 220

    def test_roi_undefined_with_nothing_graded(self, jr):
        record(jr, game_id=CHI_GB, team="CHI", handicap=3,
               placed_at=datetime(2026, 10, 7, 12, 0, tzinfo=UTC))
        assert bj.totals(state(jr))["roi_pct"] is None


# ---------------------------------------------------------------- grading --

class TestGrading:
    def test_pending_without_scores(self, jr):
        jr.set_scores((None, None))
        w = record(jr)
        assert bj.grade(jr.dir, jr.schedule, now=NOW) == []
        assert state(jr)[w["record_id"]].status == "pending" and jr.files(bj.KIND_GRADE) == []

    @pytest.mark.parametrize("scores", [(24, None), ("x", 27), (-1, 27)])
    def test_uncertain_scores_stay_pending(self, jr, scores):
        jr.set_scores(scores)
        record(jr)
        assert bj.grade(jr.dir, jr.schedule, now=NOW) == []

    def test_scores_before_kickoff_are_uncertain(self, jr):
        record(jr)
        assert bj.grade(jr.dir, jr.schedule, now=datetime(2026, 10, 4, 20, 0, tzinfo=UTC)) == []

    def test_duplicate_schedule_rows_are_uncertain(self, jr):
        record(jr)
        text = schedule_text()
        jr.schedule.write_text(text + text.splitlines(True)[1], encoding="utf-8")
        assert bj.grade(jr.dir, jr.schedule, now=NOW) == []

    def test_grade_provenance_and_repeat_grading(self, jr):
        w = record(jr, odds=-120, stake="30.00")
        [g] = bj.grade(jr.dir, jr.schedule, now=NOW)
        assert (g["result"], g["profit_cad"], g["reason"], g["supersedes"]) == \
            ("win", "25.00", "initial grade", None)
        assert g["wager_id"] == g["terms_record_id"] == w["record_id"]
        assert g["schedule"]["sha256"] == hashlib.sha256(jr.schedule.read_bytes()).hexdigest()
        assert g["evidence"] == {"gameday": "2026-10-04", "home_score": 27, "away_score": 24,
                                 "result": 3, "total": 51, "overtime": 0}
        assert bj.grade(jr.dir, jr.schedule, now=NOW + timedelta(hours=1)) == []   # idempotent
        assert len(jr.files(bj.KIND_GRADE)) == 1

    def test_score_correction_is_audited(self, jr):
        w = record(jr)                                      # BUF -2.5: win at 27-24
        [g1] = bj.grade(jr.dir, jr.schedule, now=NOW)
        jr.set_scores((24, 26))                             # corrected: BUF by 2 -> loss
        [g2] = bj.grade(jr.dir, jr.schedule, now=NOW + timedelta(hours=1))
        assert g2["supersedes"] == g1["record_id"] and g2["result"] == "loss"
        assert g2["reason"].startswith("score correction: 24-27 -> 24-26")
        s = state(jr)[w["record_id"]]
        assert s.status == "loss" and [g["record_id"] for g in s.settlements] == \
            [g1["record_id"], g2["record_id"]]
        assert bj.totals(state(jr))["net_profit_cad"] == Decimal("-50.00")
        # A correction that keeps the result still records the new score.
        jr.set_scores((20, 22))
        [g3] = bj.grade(jr.dir, jr.schedule, now=NOW + timedelta(hours=2))
        assert g3["result"] == "loss" and g3["supersedes"] == g2["record_id"]

    def test_amended_terms_are_regraded(self, jr):
        w = record(jr)
        bj.grade(jr.dir, jr.schedule, now=NOW)
        bj.save_amendment(prep_amend(jr, w["record_id"], *terms(jr, handicap=-3.5),
                                               "wrong line", now=NOW), jr.dir)
        assert state(jr)[w["record_id"]].status == "pending"         # old grade no longer applies
        assert bj.totals(state(jr))["graded"] == 0
        assert bj.totals(state(jr))["net_profit_cad"] == Decimal("0.00")
        [g] = bj.grade(jr.dir, jr.schedule, now=NOW)
        assert g["reason"] == "terms amended since the last grade" and g["result"] == "loss"

    def test_voided_wagers_not_graded(self, jr):
        w = record(jr)
        bj.save_void(prep_void(jr, w["record_id"], "x", now=NOW), jr.dir)
        assert bj.grade(jr.dir, jr.schedule, now=NOW) == []


# -------------------------------------------------------------- integrity --

class TestIntegrity:
    def test_edited_record_is_an_integrity_error(self, jr):
        record(jr)
        [path] = jr.files()
        path.write_text(path.read_text().replace('"50.00"', '"5.00"'))
        with pytest.raises(bj.IntegrityError, match="checksum mismatch"):
            bj.load_records(jr.dir)

    @pytest.mark.parametrize("damage", ["truncate", "stray", "rename", "dir"])
    def test_damage_is_never_silently_skipped(self, jr, damage):
        record(jr)
        [path] = jr.files()
        if damage == "truncate":
            path.write_bytes(path.read_bytes()[:40])
        elif damage == "stray":
            (jr.dir / "notes.txt").write_text("x")
        elif damage == "rename":
            path.rename(path.with_suffix(".bak"))
        else:
            (jr.dir / "W-20261007T170000Z-aaaaaaaaaaaa.json").mkdir()
        with pytest.raises(bj.IntegrityError):
            bj.load_records(jr.dir)

    def test_file_name_must_match_record_id(self, jr):
        record(jr)
        [path] = jr.files()
        path.rename(jr.dir / "W-20261007T170000Z-bbbbbbbbbbbb.json")
        with pytest.raises(bj.IntegrityError, match="doesn't match"):
            bj.load_records(jr.dir)

    def test_dangling_reference(self, jr):
        doc = prep_void(jr, "W-20261007T170000Z-cccccccccccc", "x", now=NOW)
        jr.dir.mkdir()
        (jr.dir / f"{doc['record_id']}.json").write_text(json.dumps(doc))
        with pytest.raises(bj.IntegrityError, match="voids W-20261007T170000Z-cccccccccccc "
                                                    "doesn't exist"):
            bj.current_state(bj.load_records(jr.dir))


# ------------------------------------------------------- source separation --

PROTECTED = ["data_files/betting_recommendations_log.csv",
             "data_files/nfl_games_historical.csv",
             "data_files/nfl_games_historical_with_predictions.csv"]


def digests():
    out = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
           for p in PROTECTED if (ROOT / p).exists()}
    for d in ("data_files/ontario_spreads", "data_files/pregame_snapshots"):
        for p in sorted((ROOT / d).rglob("*")) if (ROOT / d).exists() else []:
            if p.is_file():
                out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


class TestSourceSeparation:
    def test_only_the_journal_is_written(self, jr, tmp_path):
        before_repo = digests()
        sched_before = jr.schedule.read_bytes()
        w = record(jr)
        bj.grade(jr.dir, jr.schedule, now=NOW)
        bj.save_amendment(prep_amend(jr, w["record_id"], *terms(jr, stake="9.00"),
                                               "x", now=NOW), jr.dir)
        written = {p for p in tmp_path.rglob("*") if p.is_file()}
        assert written == set(jr.files()) | {jr.schedule}
        assert jr.schedule.read_bytes() == sched_before
        assert digests() == before_repo
        assert bj.JOURNAL_DIR != on.MANUAL_DIR and bj.JOURNAL_DIR != on.CAPTURE_DIR

    def test_no_network_or_recommendation_code(self):
        src = (ROOT / "bet_journal.py").read_text() + PAGE.read_text()
        for forbidden in ("requests", "ODDS_API", "fetch_odds", "log_recommendations",
                          "betting_recommendations_log.csv\"", "LOG_PATH", "to_csv"):
            assert forbidden not in src, forbidden


# ------------------------------------------------------------------ page --

def label(jr, game_id):
    sched = on.load_schedule(jr.schedule)[0]
    return next(g["label"] for g in bj.selectable_games(sched, NOW) if g["game_id"] == game_id)


@pytest.fixture
def render(jr, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import streamlit as st

    def run(now=NOW, at=None):
        monkeypatch.setattr(bj, "now_utc", lambda: now)
        if at is None:
            st.cache_data.clear()
            at = AppTest.from_file(str(PAGE), default_timeout=60)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return at
    return run


def fill(at, jr, *, game=KC_BUF, book="betmgm_ca_on", team="BUF", spread=-2.5, odds=-110,
         stake="50.00", day=date(2026, 10, 4), at_time=time(12, 0), ref="", note="",
         second=False, preview=True):
    box = at.selectbox(key="bj_game")
    box.select_index(box.options.index(label(jr, game)))
    at.run()
    at.selectbox(key="bj_book").set_value(book)
    at.selectbox(key="bj_team").set_value(team)
    at.number_input(key="bj_spread").set_value(spread)
    at.number_input(key="bj_odds").set_value(odds)
    at.text_input(key="bj_stake").set_value(stake)
    at.date_input(key="bj_date").set_value(day)
    at.time_input(key="bj_time").set_value(at_time)
    at.text_input(key="bj_ref").set_value(ref)
    at.text_input(key="bj_note").set_value(note)
    at.checkbox(key="bj_second").set_value(second)
    if preview:
        at.button(key="bj_preview").click()
        at.run()
    return at


def text(at, element):
    return " ".join(str(e.value) for e in getattr(at, element))


class TestPage:
    def test_empty_journal_renders(self, jr, render):
        at = render()
        assert "No wagers recorded yet" in text(at, "info")
        assert at.button(key="bj_save").disabled and jr.files() == []
        assert not jr.dir.exists()

    def test_preview_shows_terms_and_writes_nothing(self, jr, render):
        at = fill(render(), jr, ref="TK-9")
        md = text(at, "markdown") + text(at, "caption")
        assert "Buffalo Bills −2.5 at −110" in md and "$50.00 CAD" in md
        assert "2026-10-04 12:00 EDT" in md and "2026-10-04T16:00:00Z UTC" in md
        assert "User-confirmed placed wager" in md and "does not place a bet" in md
        assert jr.files() == [] and not at.button(key="bj_save").disabled

    def test_save_writes_exactly_the_preview_once(self, jr, render):
        at = fill(render(), jr, team="KC", spread=3.5, odds=150, stake="20")
        at.button(key="bj_save").click()
        at.run()
        [path] = jr.files()
        t = json.loads(path.read_text())["terms"]
        assert (t["team"], t["handicap"], t["odds"], t["stake_cad"], t["placed_at"]) == \
            ("KC", 3.5, 150, "20.00", "2026-10-04T16:00:00Z")
        assert "Saved wager" in text(at, "success")
        at.button(key="bj_save").click()               # stale click / rerun
        at.run()
        assert len(jr.files()) == 1

    @pytest.mark.parametrize("field,value", [
        ("bj_stake", "51.00"), ("bj_spread", -3.5), ("bj_odds", -115), ("bj_note", "x"),
        ("bj_ref", "T1"), ("bj_team", "KC"), ("bj_book", "proline_ca_on"),
        ("bj_time", time(12, 1)), ("bj_second", True),
    ])
    def test_change_after_preview_requires_new_preview(self, jr, render, field, value):
        at = fill(render(), jr)
        widget = next(w for w in (at.text_input, at.number_input, at.selectbox, at.time_input,
                                  at.checkbox) if any(x.key == field for x in w))
        widget(key=field).set_value(value)
        at.button(key="bj_save").click()
        at.run()
        assert "form changed after Preview" in text(at, "error") and jr.files() == []

    def test_late_entry_future_and_kickoff(self, jr, render):
        at = fill(render(), jr, at_time=time(16, 25))            # = kickoff
        assert "not before kickoff" in text(at, "error")
        at = fill(at, jr, game=CHI_GB, team="CHI", spread=3, day=date(2026, 10, 7),
                  at_time=time(13, 1))                           # 1 minute in the future
        assert "future" in text(at, "error")
        at = fill(at, jr, at_time=time(16, 24))                  # late entry, just before kickoff
        at.button(key="bj_save").click()
        at.run()
        assert len(jr.files()) == 1

    def test_dst_times_rejected(self, jr, render):
        now = datetime(2026, 11, 2, 12, 0, tzinfo=UTC)
        at = render(now)
        box = at.selectbox(key="bj_game")
        box.select_index(next(i for i, o in enumerate(box.options) if "NYG" in o or
                              "New York Giants" in o))
        at.run()
        for day, t, msg in ((date(2026, 11, 1), time(1, 30), "happens twice"),):
            at.number_input(key="bj_spread").set_value(3)
            at.number_input(key="bj_odds").set_value(-110)
            at.text_input(key="bj_stake").set_value("10")
            at.date_input(key="bj_date").set_value(day)
            at.time_input(key="bj_time").set_value(t)
            at.button(key="bj_preview").click()
            at.run()
            assert msg in text(at, "error") and jr.files() == []

    def test_duplicate_and_intentional_second_wager(self, jr, render):
        at = fill(render(), jr)
        at.button(key="bj_save").click()
        at.run()
        at = fill(at, jr)
        assert "already recorded" in text(at, "error") and at.button(key="bj_save").disabled
        at = fill(at, jr, second=True)
        assert "separate second wager" in text(at, "warning")
        at.button(key="bj_save").click()
        at.run()
        wagers = [json.loads(p.read_text()) for p in jr.files()]
        assert len(wagers) == 2 and wagers[1]["confirmed_existing"] == [wagers[0]["record_id"]] \
            or wagers[0]["confirmed_existing"] == [wagers[1]["record_id"]]

    def test_concurrent_save_between_preview_and_save(self, jr, render):
        at = fill(render(), jr)
        record(jr)                                     # another session saves the same wager
        at.button(key="bj_save").click()
        at.run()
        assert "identical wager" in text(at, "error") and len(jr.files()) == 1

    def test_rendering_never_grades(self, jr, render):
        record(jr)
        at = render()
        render(at=at)
        assert jr.files(bj.KIND_GRADE) == []
        at.button(key="bj_grade").click()
        at.run()
        assert len(jr.files(bj.KIND_GRADE)) == 1 and "Wrote 1 grade" in text(at, "success")
        assert "$45.45" in " ".join(str(m.value) for m in at.metric)

    def test_amend_and_void_through_page(self, jr, render):
        w = record(jr)
        at = render()
        at.radio(key="bjf_action").set_value("amend")
        at.run()
        assert at.text_input(key="bjf_stake").value == "50.00"      # prefilled
        at.text_input(key="bjf_stake").set_value("55.00")
        at.button(key="bjf_preview").click()
        at.run()
        assert "reason is required" in text(at, "error") and len(jr.files()) == 1
        at.text_input(key="bjf_reason").set_value("stake typo")
        at.button(key="bjf_preview").click()
        at.run()
        assert "$55.00 CAD" in text(at, "markdown") and len(jr.files()) == 1
        at.button(key="bjf_save").click()
        at.run()
        assert len(jr.files(bj.KIND_AMENDMENT)) == 1
        assert state(jr)[w["record_id"]].terms["stake_cad"] == "55.00"
        at.radio(key="bjf_action").set_value("void")
        at.run()
        at.text_input(key="bjf_reason").set_value("duplicate entry")
        at.button(key="bjf_preview").click()
        at.run()
        at.button(key="bjf_save").click()
        at.run()
        assert state(jr)[w["record_id"]].void and len(jr.files()) == 3
        assert "No current wagers to correct" in text(at, "info")

    def test_cache_refresh_and_integrity_error(self, jr, render):
        at = render()
        record(jr)
        at = render(at=at)                             # same session, cache keyed on content
        assert len(at.dataframe[0].value) == 1
        [path] = jr.files()
        path.write_text(path.read_text().replace('"50.00"', '"5.00"'))
        at = render(at=at)
        assert "Integrity error" in text(at, "error") and not at.metric

    def test_replayed_preview_is_saved_at_most_once(self, jr, render):
        # A preview replayed after its save (e.g. a double click that resubmits
        # the same pending state) is refused by its one-time token before the
        # journal's locked duplicate check is even reached.
        at = fill(render(), jr)
        pending = dict(at.session_state["bj_pending"])
        at.button(key="bj_save").click()
        at.run()
        at.session_state["bj_pending"] = pending
        at.run()
        assert not at.button(key="bj_save").disabled
        at.button(key="bj_save").click()
        at.run()
        assert "already saved" in text(at, "info") and len(jr.files()) == 1


# ------------------------------------------------- completion eligibility --

def evidence_for(jr, now=NOW):
    return bj.completion_evidence(on.load_schedule(jr.schedule)[0], KC_BUF, now)


class TestCompletionEvidence:
    def test_completed_game(self, jr):
        ev, why = evidence_for(jr)
        assert why == "" and ev == {"gameday": "2026-10-04", "home_score": 27, "away_score": 24,
                                    "result": 3, "total": 51, "overtime": 0}

    def test_kickoff_passed_is_not_enough(self, jr):
        # The review's case: live partial scores an hour after kickoff.
        record(jr)
        live = datetime(2026, 10, 4, 21, 25, tzinfo=UTC)
        jr.set_scores((3, 0), result=None, total=None, overtime=None)
        assert evidence_for(jr, live)[0] is None
        assert bj.grade(jr.dir, jr.schedule, now=live) == []
        # Even fully consistent fields don't count on the game day itself.
        jr.set_scores((3, 0))
        assert evidence_for(jr, live) == (None, "game day is not before today")
        assert bj.grade(jr.dir, jr.schedule, now=live) == []
        assert jr.files(bj.KIND_GRADE) == []

    def test_game_day_must_be_before_today_in_toronto(self, jr):
        # 2026-10-05 03:30 UTC is still Oct 4 in Toronto -> not yet.
        assert evidence_for(jr, datetime(2026, 10, 5, 3, 30, tzinfo=UTC))[0] is None
        assert evidence_for(jr, datetime(2026, 10, 5, 4, 0, tzinfo=UTC))[0] is not None

    @pytest.mark.parametrize("scores,fields,why", [
        ((0, 0), {}, "0-0 placeholder"),
        ((None, None), {}, "no final score"),
        ((24, None), {}, "no final score"),
        ((24, 27), {"result": None}, "no final score"),
        ((24, 27), {"total": None}, "no final score"),
        ((24, 27), {"overtime": None}, "no final score"),
        ((24, 27), {"result": 5}, "inconsistent"),
        ((24, 27), {"result": -3}, "inconsistent"),           # away-minus-home sign
        ((24, 27), {"total": 50}, "inconsistent"),
        ((24, 27), {"overtime": 2}, "inconsistent"),
        (("24.5", 27), {"result": "2.5", "total": "51.5", "overtime": 0}, "(away_score, "
         "result, total"),
        (("24", "inf"), {"result": 3, "total": 51, "overtime": 0}, "(home_score missing"),
        (("nan", 27), {"result": 3, "total": 51, "overtime": 0}, "(away_score missing"),
        (("x", 27), {"result": 3, "total": 51, "overtime": 0}, "(away_score missing"),
        ((-3, 0), {}, "negative score"),
    ])
    def test_placeholders_and_invalid_fields_stay_pending(self, jr, scores, fields, why):
        record(jr)
        jr.set_scores(scores, **fields)
        ev, reason = evidence_for(jr)
        assert ev is None and why in reason
        assert bj.grade(jr.dir, jr.schedule, now=NOW) == []
        assert state(jr)[next(iter(state(jr)))].status == "pending"

    def test_duplicated_and_missing_rows_stay_pending(self, jr):
        record(jr)
        text = schedule_text()
        jr.schedule.write_text(text + text.splitlines(True)[1], encoding="utf-8")
        assert evidence_for(jr) == (None, "game listed more than once")
        jr.schedule.write_text(HEADER, encoding="utf-8")
        assert evidence_for(jr) == (None, "game not found")
        assert bj.grade(jr.dir, jr.schedule, now=NOW) == []


# ----------------------------------------------------------- invalidation --

class TestInvalidation:
    def test_removed_scores_invalidate_and_restore_regrades(self, jr):
        w = record(jr)                                           # BUF -2.5 wins 27-24
        [g1] = bj.grade(jr.dir, jr.schedule, now=NOW)
        jr.set_scores((None, None))
        [x] = bj.grade(jr.dir, jr.schedule, now=NOW + timedelta(hours=1))
        assert x["kind"] == bj.KIND_INVALIDATION and x["supersedes"] == g1["record_id"]
        assert x["terms_record_id"] == w["record_id"] and "no final score" in x["reason"]
        assert x["schedule"]["sha256"] == hashlib.sha256(jr.schedule.read_bytes()).hexdigest()
        s = state(jr)[w["record_id"]]
        assert s.status == "unverified" and s.grade is None and g1["record_id"] in s.status_note
        tot = bj.totals(state(jr))
        assert tot["net_profit_cad"] == 0 and tot["roi_pct"] is None
        assert (tot["graded"], tot["unverified"], tot["stake_pending_cad"]) == \
            (0, 1, Decimal("50.00"))
        assert [r["kind"] for r in bj.load_records(jr.dir)].count(bj.KIND_GRADE) == 1  # kept
        # Repeating writes nothing.
        assert bj.grade(jr.dir, jr.schedule, now=NOW + timedelta(hours=2)) == []
        # Restored evidence -> a new grade after the invalidation.
        jr.set_scores((24, 27))
        [g2] = bj.grade(jr.dir, jr.schedule, now=NOW + timedelta(hours=3))
        assert g2["supersedes"] == x["record_id"] and g2["reason"] == \
            "evidence restored after invalidation"
        assert state(jr)[w["record_id"]].status == "win"
        assert bj.totals(state(jr))["net_profit_cad"] == Decimal("45.45")

    @pytest.mark.parametrize("scores,fields", [
        ((24, 27), {"overtime": None}), ((24, 27), {"total": 52}), ((0, 0), {}),
    ])
    def test_evidence_turning_invalid_invalidates(self, jr, scores, fields):
        w = record(jr)
        bj.grade(jr.dir, jr.schedule, now=NOW)
        jr.set_scores(scores, **fields)
        [x] = bj.grade(jr.dir, jr.schedule, now=NOW)
        assert x["kind"] == bj.KIND_INVALIDATION
        assert state(jr)[w["record_id"]].status == "unverified"

    def test_duplicate_row_after_grading_invalidates(self, jr):
        record(jr)
        bj.grade(jr.dir, jr.schedule, now=NOW)
        text = schedule_text()
        jr.schedule.write_text(text + text.splitlines(True)[1], encoding="utf-8")
        [x] = bj.grade(jr.dir, jr.schedule, now=NOW)
        assert "listed more than once" in x["reason"]

    def test_amended_wager_without_evidence_writes_nothing(self, jr):
        w = record(jr)
        bj.grade(jr.dir, jr.schedule, now=NOW)
        bj.save_amendment(prep_amend(jr, w["record_id"], *terms(jr, stake="60.00"), "typo"),
                          jr.dir)
        jr.set_scores((None, None))
        assert bj.grade(jr.dir, jr.schedule, now=NOW) == []      # old-terms grade isn't counted
        assert state(jr)[w["record_id"]].status == "pending"

    def test_page_flags_unverified(self, jr, render):
        record(jr)
        bj.grade(jr.dir, jr.schedule, now=NOW)
        jr.set_scores((None, None))
        bj.grade(jr.dir, jr.schedule, now=NOW)
        at = render()
        assert "1 unverified settlement" in text(at, "warning")
        row = at.dataframe[0].value.iloc[0]
        assert row["Status"] == "unverified" and "invalidated" in row["Status note"]
        assert "$0.00" in " ".join(str(m.value) for m in at.metric)


# -------------------------------------------------- explicit history links --

def forge(jr, doc, **changes):
    """Write a record directly (bypassing the save checks), resealed so only
    the intended defect remains."""
    import copy
    d = copy.deepcopy({k: v for k, v in doc.items() if k != "_file"})
    for k, v in changes.items():
        if "." in k:
            outer, inner = k.split(".")
            d[outer][inner] = v
        else:
            d[k] = v
    d.pop(bj.CHECKSUM_FIELD, None)
    d[bj.CHECKSUM_FIELD] = on.ps.payload_checksum(d)
    jr.dir.mkdir(exist_ok=True)
    (jr.dir / f"{d['record_id']}.json").write_text(json.dumps(d), encoding="utf-8")
    return d


def rid(kind, n, at="20261007T170000Z"):
    return f"{bj.PREFIX[kind]}-{at}-{n:012x}"


def broken(jr, match):
    with pytest.raises(bj.IntegrityError, match=match):
        bj.current_state(bj.load_records(jr.dir))


class TestExplicitHistory:
    def test_same_second_amendments_follow_links_not_ids(self, jr, monkeypatch):
        # The review's case: the later amendment gets the smaller random ID.
        w = record(jr)
        ids = iter([rid(bj.KIND_AMENDMENT, 0xfff, "20261007T180000Z"),
                    rid(bj.KIND_AMENDMENT, 0x001, "20261007T180000Z")])
        monkeypatch.setattr(bj, "_new_id", lambda kind, at: next(ids))
        later = NOW + timedelta(hours=1)
        for stake in ("60.00", "70.00"):
            bj.save_amendment(prep_amend(jr, w["record_id"], *terms(jr, stake=stake), stake,
                                         now=later), jr.dir)
        s = state(jr)[w["record_id"]]
        assert s.terms["stake_cad"] == "70.00"
        assert [a["terms"]["stake_cad"] for a in s.amendments] == ["60.00", "70.00"]

    def test_stale_previous_refused_under_lock(self, jr):
        w = record(jr)
        a1 = prep_amend(jr, w["record_id"], *terms(jr, stake="60.00"), "first")
        a2 = prep_amend(jr, w["record_id"], *terms(jr, stake="70.00"), "second")  # same previous
        bj.save_amendment(a1, jr.dir)
        with pytest.raises(bj.JournalError, match="changed since preview"):
            bj.save_amendment(a2, jr.dir)
        v = bj.prepare_void(w["record_id"], w["record_id"], "stale", now=NOW)
        with pytest.raises(bj.JournalError, match="changed since preview"):
            bj.save_void(v, jr.dir)
        assert len(jr.files()) == 2

    def test_fork(self, jr):
        w = record(jr)
        a1 = prep_amend(jr, w["record_id"], *terms(jr, stake="60.00"), "first")
        bj.save_amendment(a1, jr.dir)
        forge(jr, a1, record_id=rid(bj.KIND_AMENDMENT, 2), **{"terms.stake_cad": "70.00"})
        broken(jr, "history forks after " + w["record_id"])

    def test_cycle(self, jr):
        w = record(jr)
        a = prep_amend(jr, w["record_id"], *terms(jr, stake="60.00"), "x")
        forge(jr, a, record_id=rid(bj.KIND_AMENDMENT, 1), previous=rid(bj.KIND_AMENDMENT, 2))
        forge(jr, a, record_id=rid(bj.KIND_AMENDMENT, 2), previous=rid(bj.KIND_AMENDMENT, 1))
        broken(jr, "not on the chain")

    def test_missing_previous(self, jr):
        w = record(jr)
        a = prep_amend(jr, w["record_id"], *terms(jr, stake="60.00"), "x")
        forge(jr, a, previous=rid(bj.KIND_AMENDMENT, 9))
        broken(jr, "previous .* doesn't exist")

    def test_previous_in_another_wagers_chain(self, jr):
        w1, w2 = record(jr), record(jr, stake="20.00")
        a = prep_amend(jr, w2["record_id"], *terms(jr, stake="21.00"), "x")
        forge(jr, a, previous=w1["record_id"])
        broken(jr, "previous belongs to another wager")

    def test_duplicate_voids(self, jr):
        # The review's case: two voids for one wager.
        w = record(jr)
        v = prep_void(jr, w["record_id"], "a")
        forge(jr, v)
        forge(jr, v, record_id=rid(bj.KIND_VOID, 7), reason="b")
        broken(jr, "voided more than once")

    def test_record_after_void(self, jr):
        w = record(jr)
        v = prep_void(jr, w["record_id"], "a")
        bj.save_void(v, jr.dir)
        late = bj.prepare_amendment(w["record_id"], w["record_id"], *terms(jr, stake="61.00"),
                                    "x", now=NOW)
        with pytest.raises(bj.JournalError, match="is void"):
            bj.save_amendment(late, jr.dir)
        forge(jr, late, previous=v["record_id"])               # linked after the void
        broken(jr, "previous must name the wager or an amendment")

    def test_settlement_recorded_after_void_is_rejected(self, jr):
        w = record(jr)
        [g] = bj.grade(jr.dir, jr.schedule, now=NOW)
        v = prep_void(jr, w["record_id"], "mistake", now=NOW + timedelta(seconds=1))
        bj.save_void(v, jr.dir)
        forge(jr, g, record_id=rid(bj.KIND_GRADE, 8, "20261007T170002Z"),
              recorded_at="2026-10-07T17:00:02Z", supersedes=g["record_id"])
        broken(jr, "settlement is not recorded strictly before void")

    def test_settlement_tied_with_void_is_unorderable(self, jr):
        w = record(jr)
        [g] = bj.grade(jr.dir, jr.schedule, now=NOW)
        v = prep_void(jr, w["record_id"], "mistake", now=NOW + timedelta(seconds=1))
        bj.save_void(v, jr.dir)
        forge(jr, g, record_id=rid(bj.KIND_GRADE, 10, "20261007T170001Z"),
              recorded_at="2026-10-07T17:00:01Z", supersedes=g["record_id"])
        broken(jr, "settlement is not recorded strictly before void")

    def test_old_terms_cannot_be_graded_after_amendment(self, jr):
        w = record(jr)
        [g] = bj.grade(jr.dir, jr.schedule, now=NOW)
        amendment = prep_amend(jr, w["record_id"], *terms(jr, handicap=-3.5), "line correction",
                               now=NOW + timedelta(seconds=1))
        bj.save_amendment(amendment, jr.dir)
        forge(jr, g, record_id=rid(bj.KIND_GRADE, 9, "20261007T170002Z"),
              recorded_at="2026-10-07T17:00:02Z", supersedes=g["record_id"])
        broken(jr, "settlement recorded after its terms were superseded")

    def test_timestamp_order(self, jr):
        w = record(jr)
        early = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
        a = prep_amend(jr, w["record_id"], *terms(jr, stake="60.00"), "x", now=early)
        forge(jr, a)
        broken(jr, "recorded before its")

    def test_record_id_must_match_recorded_at(self, jr):
        w = record(jr)
        forge(jr, w, record_id=rid(bj.KIND_WAGER, 5, "20261007T170001Z"))
        broken(jr, "record_id doesn't match recorded_at")

    def test_existing_ambiguous_history_fails_visibly(self, jr, render):
        w = record(jr)
        v = prep_void(jr, w["record_id"], "a")
        forge(jr, v)
        forge(jr, v, record_id=rid(bj.KIND_VOID, 7))
        before = sorted(p.name for p in jr.files())
        at = render()
        assert "Integrity error" in text(at, "error") and "voided more than once" in \
            text(at, "error") and not at.metric
        for action in (lambda: record(jr, stake="1.00"),
                       lambda: bj.grade(jr.dir, jr.schedule, now=NOW)):
            with pytest.raises(bj.IntegrityError):
                action()
        assert sorted(p.name for p in jr.files()) == before          # nothing repaired


class TestRecordValidation:
    @pytest.fixture
    def graded(self, jr):
        w = record(jr)
        [g] = bj.grade(jr.dir, jr.schedule, now=NOW)
        (jr.dir / f"{g['record_id']}.json").unlink()
        return w, g

    # The review's malformed grade: each defect on its own.
    @pytest.mark.parametrize("changes,match", [
        ({"evidence.home_score": "abc"}, "home_score must be an integer"),
        ({"evidence.away_score": None}, "away_score must be an integer"),
        ({"evidence.home_score": 27.0}, "home_score must be an integer"),
        ({"evidence.home_score": True}, "home_score must be an integer"),
        ({"evidence.result": 4}, "inconsistent"),
        ({"evidence.overtime": 2}, "overtime"),
        ({"evidence.home_score": 0, "evidence.away_score": 0, "evidence.result": 0,
          "evidence.total": 0}, "placeholder"),
        ({"evidence.gameday": "Oct 4"}, "gameday"),
        ({"schedule": None}, "schedule provenance"),
        ({"schedule.sha256": "abc"}, "schedule provenance"),
        ({"terms_record_id": rid(bj.KIND_WAGER, 3)}, "terms_record_id .* doesn't exist"),
        ({"terms_record_id": "W-nonexistent"}, "terms_record_id"),
        ({"supersedes": "bogus"}, "supersedes"),
        ({"supersedes": rid(bj.KIND_GRADE, 4)}, "supersedes .* doesn't exist"),
        ({"wager_id": rid(bj.KIND_WAGER, 3)}, "wager_id .* doesn't exist"),
        ({"terms_sha256": "0" * 64}, "terms_sha256 doesn't match"),
        ({"profit_cad": "45.46"}, "don't follow"),
        ({"result": "loss"}, "don't follow"),
        ({"reason": ""}, "reason"),
    ])
    def test_malformed_grade(self, jr, graded, changes, match):
        forge(jr, graded[1], **changes)
        broken(jr, match)

    def test_non_finite_json_is_rejected(self, jr, graded):
        d = forge(jr, graded[1])
        path = jr.dir / f"{d['record_id']}.json"
        path.write_text(path.read_text().replace('"home_score": 27', '"home_score": NaN'))
        broken(jr, "non-finite number NaN")

    def test_grade_of_another_wagers_terms(self, jr, graded):
        w2 = record(jr, stake="20.00")
        forge(jr, graded[1], terms_record_id=w2["record_id"])
        broken(jr, "isn't in this wager's terms chain")

    def test_two_first_grades(self, jr, graded):
        forge(jr, graded[1])
        forge(jr, graded[1], record_id=rid(bj.KIND_GRADE, 8))
        broken(jr, "exactly one first grade")

    def test_invalidation_must_supersede_a_grade(self, jr):
        record(jr)
        [g] = bj.grade(jr.dir, jr.schedule, now=NOW)
        jr.set_scores((None, None))
        [x] = bj.grade(jr.dir, jr.schedule, now=NOW)
        forge(jr, x, record_id=rid(bj.KIND_INVALIDATION, 1), supersedes=x["record_id"])
        broken(jr, "supersedes")

    @pytest.mark.parametrize("changes,match", [
        ({"terms.season": "2026"}, "season"), ({"terms.season": 2025}, "season"),
        ({"terms.week": 5}, "week"), ({"terms.week": True}, "week"),
        ({"terms.home_team": "KC", "terms.away_team": "BUF"}, "teams"),
        ({"terms.handicap": 2}, "handicap"), ({"terms.odds": -110.0}, "odds"),
        ({"terms.placed_at": "2026-10-04 16:00"}, "UTC timestamp"),
        ({"confirmed_existing": [rid(bj.KIND_WAGER, 1)]}, "_c .* doesn't exist"),
        ({"confirmed_existing": ["nope"]}, "confirmed_existing"),
        ({"confirmed_existing": [rid(bj.KIND_WAGER, 2), rid(bj.KIND_WAGER, 1)]},
         "confirmed_existing"),
    ])
    def test_malformed_wager(self, jr, changes, match):
        forge(jr, record(jr), **changes)
        broken(jr, match)

    def test_confirmed_wager_must_predate(self, jr):
        w1 = record(jr)
        w2 = record(jr, stake="20.00", at=NOW + timedelta(hours=1))
        forge(jr, w1, confirmed_existing=[w2["record_id"]])
        broken(jr, "recorded before")

    def test_confirmed_wager_must_be_an_actual_duplicate(self, jr):
        w1 = record(jr, stake="20.00")
        w2 = record(jr, stake="30.00")
        forge(jr, w2, confirmed_existing=[w1["record_id"]])
        broken(jr, "confirmed_existing doesn't match")

    def test_confirmed_self(self, jr):
        w = record(jr)
        forge(jr, w, confirmed_existing=[w["record_id"]])
        broken(jr, "confirms itself")


# -------------------------------------------- amendment duplicate checks --

class TestAmendmentDuplicates:
    def test_amending_into_another_wager_needs_confirmation(self, jr):
        w1, w2 = record(jr), record(jr, stake="20.00")
        t, sha = terms(jr)                                   # = w1's terms
        with pytest.raises(bj.DuplicateWager, match="identical wager"):
            bj.save_amendment(prep_amend(jr, w2["record_id"], t, sha, "typo"), jr.dir)
        a = prep_amend(jr, w2["record_id"], t, sha, "typo", confirmed=[w1["record_id"]])
        bj.save_amendment(a, jr.dir)
        assert state(jr)[w2["record_id"]].terms == state(jr)[w1["record_id"]].terms

    def test_confirmation_rechecked_under_lock(self, jr):
        w1, w2 = record(jr), record(jr, stake="20.00")
        a = prep_amend(jr, w2["record_id"], *terms(jr), "typo", confirmed=[w1["record_id"]])
        bj.save_void(prep_void(jr, w1["record_id"], "gone"), jr.dir)
        with pytest.raises(bj.DuplicateWager, match="changed since preview.*now: none"):
            bj.save_amendment(a, jr.dir)

    def test_reused_reference_still_rejected(self, jr):
        record(jr, reference="T-1")
        w2 = record(jr, stake="20.00", reference="T-2")
        with pytest.raises(bj.DuplicateWager, match="reference"):
            bj.save_amendment(prep_amend(jr, w2["record_id"],
                                         *terms(jr, stake="20.00", reference="T-1"), "x"), jr.dir)

    def test_page_requires_confirmation(self, jr, render):
        w1, w2 = record(jr), record(jr, stake="20.00")
        at = render()
        at.selectbox(key="bjf_wager").set_value(w2["record_id"])
        at.radio(key="bjf_action").set_value("amend")
        at.run()
        at.text_input(key="bjf_stake").set_value("50.00")
        at.text_input(key="bjf_reason").set_value("stake typo")
        at.button(key="bjf_preview").click()
        at.run()
        assert "identical to another current wager" in text(at, "error")
        assert at.button(key="bjf_save").disabled and len(jr.files()) == 2
        at.checkbox(key="bjf_second").set_value(True)
        at.button(key="bjf_preview").click()
        at.run()
        assert w1["record_id"] in text(at, "warning")
        at.button(key="bjf_save").click()
        at.run()
        [a] = [json.loads(p.read_text()) for p in jr.files(bj.KIND_AMENDMENT)]
        assert a["confirmed_existing"] == [w1["record_id"]] and a["previous"] == w2["record_id"]

    def test_page_refuses_amendment_after_concurrent_change(self, jr, render):
        w = record(jr)
        at = render()
        at.radio(key="bjf_action").set_value("amend")
        at.run()
        at.text_input(key="bjf_stake").set_value("55.00")
        at.text_input(key="bjf_reason").set_value("typo")
        at.button(key="bjf_preview").click()
        at.run()
        bj.save_amendment(prep_amend(jr, w["record_id"], *terms(jr, stake="51.00"), "other tab"),
                          jr.dir)
        at.button(key="bjf_save").click()
        at.run()
        assert "changed since preview" in text(at, "error")
        assert state(jr)[w["record_id"]].terms["stake_cad"] == "51.00"


# ------------------------------------------- cross-process concurrency --

CHILD = r'''
import json, os, socket, sys, time
from pathlib import Path

def blocked(*a, **k):
    raise AssertionError("network call attempted")

socket.socket.connect = blocked
sys.path.insert(0, sys.argv[1])
import bet_journal as bj

mode, jdir, doc_path, ready, go, retries = sys.argv[2:8]
jdir = Path(jdir)
Path(ready).write_text(str(os.getpid()))
deadline = time.monotonic() + 60
while not Path(go).exists():
    if time.monotonic() > deadline:
        print(json.dumps({"outcome": "timeout"})); sys.exit(3)
    time.sleep(0.005)
if mode == "stress":                     # take and release the lock repeatedly
    ok = busy = 0
    end = time.monotonic() + float(retries)
    while time.monotonic() < end:
        try:
            with bj._lock(jdir, "stress"):
                ok += 1
        except bj.JournalBusy:
            busy += 1
    print(json.dumps({"outcome": "stress", "ok": ok, "busy": busy})); sys.exit(0)
if mode == "hold":                       # hold the journal lock until released
    with bj._lock(jdir, "holder"):
        Path(doc_path).write_text("held")
        while not Path(go + ".release").exists():
            if time.monotonic() > deadline:
                sys.exit(3)
            time.sleep(0.01)
    print(json.dumps({"outcome": "released", "pid": os.getpid()})); sys.exit(0)
doc = json.loads(Path(doc_path).read_text())
save = {"wager": bj.save_wager, "amend": bj.save_amendment, "void": bj.save_void}[mode]
out = "busy"
for _ in range(int(retries) + 1):
    try:
        save(doc, jdir); out = "saved"; break
    except bj.JournalBusy as exc:
        out = "busy: " + str(exc); time.sleep(0.003)
    except bj.DuplicateWager:
        out = "duplicate"; break
    except bj.JournalError as exc:
        out = "refused: " + str(exc); break
print(json.dumps({"outcome": out, "pid": os.getpid()}))
'''


@pytest.fixture
def procs(jr, tmp_path):
    """Run independent Python processes against the shared temp journal, with
    a start barrier, bounded waits and guaranteed cleanup."""
    import subprocess
    import sys
    import time as time_
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    go = tmp_path / "go"
    started = []

    def spawn(jobs, retries=2000):
        """jobs: list of (mode, doc-or-None). Returns per-child results."""
        children = []
        for i, (mode, doc) in enumerate(jobs):
            n = len(started)
            doc_path, ready = tmp_path / f"doc{n}.json", tmp_path / f"ready{n}"
            if doc is not None:
                doc_path.write_text(json.dumps(doc), encoding="utf-8")
            p = subprocess.Popen([sys.executable, str(script), str(ROOT), mode, str(jr.dir),
                                  str(doc_path), str(ready), str(go), str(retries)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            started.append(p)
            children.append((p, ready, doc_path))
        deadline = time_.monotonic() + 120
        while not all(r.exists() for _, r, _ in children):
            assert time_.monotonic() < deadline, "children didn't start"
            assert all(p.poll() is None for p, _, _ in children), \
                [p.communicate()[1] for p, _, _ in children if p.poll() is not None]
            time_.sleep(0.02)
        return children

    def finish(children, timeout=120):
        out = []
        for p, _, _ in children:
            stdout, stderr = p.communicate(timeout=timeout)
            assert p.returncode == 0, stderr
            out.append(json.loads(stdout.strip().splitlines()[-1]))
        return out

    class P:
        pass

    P.spawn, P.finish, P.go = staticmethod(spawn), staticmethod(finish), go
    yield P
    for p in started:
        if p.poll() is None:
            p.kill()
            p.communicate(timeout=30)


class TestCrossProcess:
    def test_identical_saves_from_separate_processes_write_once(self, jr, procs):
        t, sha = terms(jr)
        docs = [bj.prepare_wager(t, sha, confirmed_existing=[], now=NOW + timedelta(seconds=i))
                for i in range(6)]
        children = procs.spawn([("wager", d) for d in docs])
        procs.go.write_text("go")
        outcomes = [r["outcome"] for r in procs.finish(children)]
        assert outcomes.count("saved") == 1 and outcomes.count("duplicate") == 5, outcomes
        [path] = jr.files()
        assert json.loads(path.read_text())["record_id"] in {d["record_id"] for d in docs}
        assert not list(jr.dir.glob(".*"))                       # lock and temp files gone
        state(jr)                                                # loads cleanly

    def test_conflicting_amendments_and_void_one_wins(self, jr, procs):
        w = record(jr)
        wager_bytes = (jr.dir / f"{w['record_id']}.json").read_bytes()
        jobs = [("amend", prep_amend(jr, w["record_id"], *terms(jr, stake=f"{60 + i}.00"),
                                     f"fix {i}", now=NOW + timedelta(seconds=i)))
                for i in range(4)]
        jobs.append(("void", prep_void(jr, w["record_id"], "mistake",
                                       now=NOW + timedelta(seconds=9))))
        children = procs.spawn(jobs)
        procs.go.write_text("go")
        outcomes = [r["outcome"] for r in procs.finish(children)]
        assert outcomes.count("saved") == 1, outcomes
        assert all(o == "saved" or o.startswith("refused:") for o in outcomes), outcomes
        assert (jr.dir / f"{w['record_id']}.json").read_bytes() == wager_bytes
        assert len(jr.files()) == 2                              # the wager + one winner
        s = state(jr)[w["record_id"]]
        assert len(s.amendments) + (s.void is not None) == 1
        assert not list(jr.dir.glob(".*"))

    def test_lock_held_by_another_process(self, jr, procs, tmp_path):
        t, sha = terms(jr)
        doc = bj.prepare_wager(t, sha, confirmed_existing=[], now=NOW)
        [holder] = procs.spawn([("hold", None)])
        procs.go.write_text("go")
        held = holder[2]
        import time as time_
        deadline = time_.monotonic() + 60
        while not held.exists():
            assert time_.monotonic() < deadline
            time_.sleep(0.01)
        lock = jr.dir / ".journal_write.lock"
        owner = json.loads(lock.read_text())
        # The child's own PID (on Windows a venv python.exe is a launcher, so
        # Popen.pid can differ from the interpreter's).
        holder_pid = int(holder[1].read_text())
        assert owner["pid"] == holder_pid and owner["what"] == "bet-journal write"
        # A saver in another process gives up without touching the lock.
        [r] = procs.finish(procs.spawn([("wager", doc)], retries=0)[0:1])
        assert r["outcome"].startswith("busy:") and f"process {holder_pid}" in r["outcome"]
        assert "only after confirming that process" in r["outcome"]
        assert json.loads(lock.read_text())["token"] == owner["token"] and jr.files() == []
        (tmp_path / "go.release").write_text("x")
        [done] = procs.finish([holder])
        assert done["outcome"] == "released" and not lock.exists()

    def test_lock_never_leaks_under_contention(self, jr, procs):
        # Regression: on Windows a holder's unlink could fail while another
        # process read the lock (leaving it behind for good), and a create
        # during a pending delete raised a raw PermissionError.
        children = procs.spawn([("stress", None)] * 5, retries=2)       # 2 s each
        procs.go.write_text("go")
        results = procs.finish(children)
        assert all(r["ok"] >= 3 for r in results), results               # everyone progressed
        assert not list(jr.dir.glob(".*"))

    def test_foreign_lock_is_preserved(self, jr, procs):
        jr.dir.mkdir(parents=True)
        lock = jr.dir / ".journal_write.lock"
        lock.write_text('{"what": "bet-journal write", "id": "x", "pid": 4242, "host": "h"}')
        t, sha = terms(jr)
        docs = [bj.prepare_wager(t, sha, confirmed_existing=[], now=NOW + timedelta(seconds=i))
                for i in range(3)]
        children = procs.spawn([("wager", d) for d in docs], retries=3)
        procs.go.write_text("go")
        for r in procs.finish(children):
            assert r["outcome"].startswith("busy:")
            assert "only after confirming that process 4242 on host h" in r["outcome"]
        assert lock.exists() and json.loads(lock.read_text())["pid"] == 4242
        assert jr.files() == []

    def test_unrelated_permission_error_is_not_reported_as_contention(self, jr, monkeypatch):
        jr.dir.mkdir(parents=True)
        lock_path = jr.dir / ".journal_write.lock"
        lock_path.write_text('{"unrelated": true}', encoding="utf-8")
        real_open = on.os.open

        def denied(path, *args, **kwargs):
            if Path(path) == lock_path:
                raise PermissionError("unrelated access denied")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(on.os, "open", denied)
        with pytest.raises(PermissionError, match="unrelated access denied"):
            with on.SlotLock(jr.dir, "journal_write", "test", what="bet-journal write"):
                pass

    def test_release_keeps_a_lock_replaced_by_another_owner(self, jr):
        lock = on.SlotLock(jr.dir, "journal_write", "first", what="bet-journal write")
        with lock:
            replacement = {"what": "bet-journal write", "id": "second", "pid": 99,
                           "host": "other", "started_at": bj.iso(NOW), "token": "other-token"}
            lock.path.write_text(json.dumps(replacement), encoding="utf-8")
        assert json.loads(lock.path.read_text(encoding="utf-8"))["token"] == "other-token"


class TestMarkdownAmounts:
    def test_dollar_amounts_are_escaped(self, jr, render):
        # Regression (found in a real browser): Streamlit markdown reads a pair
        # of "$" as a LaTeX math span, garbling text like "Before: $50.00 ...
        # After: $55.00". Every markdown "$" must be escaped.
        import re
        record(jr)
        at = render()
        at.radio(key="bjf_action").set_value("amend")
        at.run()
        at.text_input(key="bjf_stake").set_value("55.00")
        at.text_input(key="bjf_reason").set_value("typo")
        at.button(key="bjf_preview").click()
        at.run()
        values = [str(e.value) for kind in ("markdown", "caption", "error", "warning", "info",
                                            "success") for e in getattr(at, kind)]
        assert any("55.00 CAD" in v for v in values) and any("$0." in v for v in values)
        unescaped = [v for v in values if re.search(r"(?<!\\)\$", v)]
        assert unescaped == []
        # A result message with two amounts (the stake-limit error).
        at = fill(at, jr, stake="200000.00")
        assert "at most" in text(at, "error")
        assert not re.search(r"(?<!\\)\$", text(at, "error")), text(at, "error")


class TestStableSelectLabels:
    # Regression (found in a real browser): the browser sends a select box's
    # *label* back, and Streamlit maps it to an option only if it matches a
    # current label. A label containing the stake changed when the wager was
    # amended, so the next action got the stale label instead of the wager ID.
    def test_wager_label_survives_amendment(self, jr, render):
        w = record(jr)
        at = render()
        before = list(at.selectbox(key="bjf_wager").options)
        bj.save_amendment(prep_amend(jr, w["record_id"], *terms(jr, stake="60.00", odds=-105,
                                                                  handicap=-3.0), "fix"), jr.dir)
        at = render(at=at)
        assert list(at.selectbox(key="bjf_wager").options) == before
        assert "$60.00 CAD" in text(at, "caption")                  # current terms shown


class TestHistoryRows:
    def test_same_second_records_read_in_link_order(self, jr, monkeypatch):
        # IDs chosen so that ID order would be wrong (amendment/grade before wager).
        ids = iter([rid(bj.KIND_WAGER, 0xfff), rid(bj.KIND_AMENDMENT, 0x002),
                    rid(bj.KIND_AMENDMENT, 0x001), rid(bj.KIND_GRADE, 0x001)])
        monkeypatch.setattr(bj, "_new_id", lambda kind, at: next(ids))
        w = record(jr)
        for stake in ("60.00", "70.00"):
            bj.save_amendment(prep_amend(jr, w["record_id"], *terms(jr, stake=stake), stake),
                              jr.dir)
        bj.grade(jr.dir, jr.schedule, now=NOW)
        records = bj.load_records(jr.dir)
        rows = bj.history_rows(records, bj.current_state(records))
        assert [(r["Record"], r["Reason"]) for r in rows] == [
            ("wager", ""), ("amendment", "60.00"), ("amendment", "70.00"),
            ("grade", "initial grade")]
        assert rows[2]["Link"] == f"after {rid(bj.KIND_AMENDMENT, 0x002)}"


class TestValidateCli:
    def test_ok_and_integrity_error(self, jr, capsys):
        w = record(jr)
        bj.grade(jr.dir, jr.schedule, now=NOW)
        assert bj.main(["validate", "--dir", str(jr.dir)]) == 0
        assert capsys.readouterr().out.startswith("OK: 2 records, 1 wagers")
        backup = jr.dir.parent / "backup"
        import shutil
        shutil.copytree(jr.dir, backup)
        assert bj.main(["validate", "--dir", str(backup)]) == 0
        (backup / f"{w['record_id']}.json").unlink()            # partial copy
        assert bj.main(["validate", "--dir", str(backup)]) == 1
        assert "INTEGRITY ERROR" in capsys.readouterr().out
        assert bj.main(["validate", "--dir", str(jr.dir)]) == 0   # read-only: original intact
