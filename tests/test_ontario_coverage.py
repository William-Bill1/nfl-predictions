"""Expected-slot coverage (ontario_spreads.expected_coverage), shared by the
`coverage` CLI and the Ontario Line Timing page.

Temporary capture/manual/schedule directories and fixed clocks only; every
network call raises while the page renders. Artifacts are produced by Phase
1's own capture()/manual_quote() against a mocked API.
"""

import hashlib
import shutil
from datetime import date, datetime, timedelta, timezone

import pytest

import ontario_line_timing as olt
import ontario_spreads as on
from test_ontario_line_timing import (  # noqa: F401 (fixtures)
    coverage, frame_with, render, store, table, uncompared)
from test_ontario_spread_report import sun_events, wed_events
from test_ontario_spreads import SUN_SLOT, WED_SLOT, _event, _quotes
from test_team_features import ROOT

UTC = timezone.utc
WED = "2026-10-07_wednesday_noon"
SUN = "2026-10-11_sunday_morning"
WED_CLOSE = WED_SLOT + timedelta(hours=3)        # Wed 15:00 Toronto
SUN_CLOSE = SUN_SLOT + timedelta(hours=2)        # Sun 11:00 Toronto
# The real situation on 2026-10-07: after the Wednesday window, before Sunday.
WED_EVENING = datetime(2026, 10, 7, 20, 6, tzinfo=UTC)
AD_HOC_AT = datetime(2026, 10, 7, 19, 6, 56, tzinfo=UTC)


def states(rows):
    return [(r["slot_id"], r["state"]) for r in rows]


def cov(store, now):
    return on.slot_coverage(now, store.capture_dir)


def us_only_events(updated="2026-10-07T15:55:00Z"):
    return [_event(g, _quotes(updated, keys=("fanduel",)))
            for g in ("2026_05_TB_DAL", "2026_05_CHI_GB")]


def cli_rows(monkeypatch, capsys, store, now):
    """Run `python ontario_spreads.py coverage` against the temp store."""
    monkeypatch.setattr(on, "CAPTURE_DIR", store.capture_dir)
    monkeypatch.setattr(on.ps, "utc_now", lambda: now)
    code = on.main(["coverage"])
    out = capsys.readouterr().out.splitlines()
    rows = []
    for line in out[1:]:
        slot_id, state, run, label = line[:32].strip(), line[33:49].strip(), \
            line[50:80].strip(), line[81:]
        rows.append((slot_id, state, None if run == "-" else run, label))
    return code, out[0], rows


# --------------------------------------------------------- slot states --

class TestStates:
    def test_no_files(self, store):
        assert states(cov(store, WED_EVENING)) == [(WED, "missed"), (SUN, "pending")]

    def test_todays_ad_hoc_only_dataset(self, store):
        store.capture(AD_HOC_AT, wed_events(updated="2026-10-07T19:05:00Z"), slot="ad_hoc")
        rows = cov(store, WED_EVENING)
        assert states(rows) == [(WED, "missed"), (SUN, "pending")]
        assert all(r["attempts"] == 0 and r["run_id"] is None for r in rows)

    def test_manual_only_never_counts(self, store):
        store.manual("2026-10-07T16:30:00Z", WED_SLOT + timedelta(hours=1))   # in the window
        assert states(cov(store, WED_EVENING)) == [(WED, "missed"), (SUN, "pending")]

    @pytest.mark.parametrize("events,kind,evidence", [
        ([], "empty", "1 capture with no quotes (not counted)"),
        (us_only_events(), "us_only", "1 capture with only US reference quotes (not counted)"),
    ])
    def test_empty_and_us_only_captures_are_evidence_only(self, store, events, kind, evidence):
        store.capture(WED_SLOT, events)
        open_ = {r["slot_id"]: r for r in cov(store, WED_SLOT + timedelta(hours=1))}[WED]
        assert open_["state"] == "awaiting_capture" and open_["attempts"] == 1
        assert [u["kind"] for u in open_["unusable"]] == [kind] and evidence in open_["label"]
        closed = {r["slot_id"]: r for r in cov(store, WED_EVENING)}[WED]
        assert closed["state"] == "missed" and evidence in closed["label"]

    def test_usable_on_time_and_late(self, store):
        store.capture(WED_SLOT + timedelta(minutes=5), wed_events())
        store.capture(SUN_SLOT + timedelta(minutes=95), sun_events())
        rows = {r["slot_id"]: r for r in cov(store, SUN_CLOSE + timedelta(hours=1))}
        assert (rows[WED]["state"], rows[WED]["detail"]) == ("captured", "on_time")
        assert (rows[SUN]["state"], rows[SUN]["detail"]) == ("captured", "late")
        assert rows[WED]["label"] == "Captured Wed Oct 7, 12:05 EDT (on time)"
        assert rows[SUN]["label"] == "Captured Sun Oct 11, 10:35 EDT (late, 95 min after the slot)"

    def test_multiple_attempts_earliest_usable_fills(self, store, tmp_path):
        store.capture(WED_SLOT, [])                                   # empty first attempt
        later = store.capture(WED_SLOT + timedelta(minutes=50), wed_events())
        # A second usable capture of the same slot (e.g. a race) stored alongside.
        other = tmp_path / "elsewhere"
        dup = store.capture(WED_SLOT + timedelta(minutes=70), wed_events(), directory=other)
        shutil.copy(dup.path, store.capture_dir / dup.path.name)
        row = {r["slot_id"]: r for r in cov(store, WED_EVENING)}[WED]
        assert (row["state"], row["run_id"], row["attempts"]) == \
            ("captured", later.doc["run_id"], 3)
        assert [u["kind"] for u in row["unusable"]] == ["empty"]

    def test_ad_hoc_inside_the_window_does_not_fill(self, store):
        store.capture(WED_SLOT + timedelta(minutes=30), wed_events(), slot="ad_hoc")
        assert {r["slot_id"]: r["state"] for r in cov(store, WED_EVENING)}[WED] == "missed"


# ------------------------------------------------- windows and calendar --

class TestWindows:
    @pytest.mark.parametrize("name,day", [("wednesday_noon", date(2026, 10, 7)),
                                          ("sunday_morning", date(2026, 10, 11))])
    def test_exact_boundaries_match_capture_eligibility(self, name, day):
        slot = on.slot_for(name, day)
        opens, closes = on.parse_utc(slot["opens_utc"]), on.parse_utc(slot["closes_utc"])
        s = timedelta(seconds=1)
        for at, state in ((opens - s, "pending"), (opens, "awaiting_capture"),
                          (closes, "awaiting_capture"), (closes + s, "missed")):
            assert on.slot_state(slot, [], at)["state"] == state, at
            # Phase 1's capture rule agrees at the same instants.
            resolved = on.resolve_slot(at, name)
            assert (resolved is not None and resolved["slot_id"] == slot["slot_id"]) == \
                (state == "awaiting_capture"), at

    @pytest.mark.parametrize("name,day,opens,closes", [
        ("wednesday_noon", date(2026, 10, 28), "2026-10-28T16:00:00Z", "2026-10-28T19:00:00Z"),
        ("sunday_morning", date(2026, 11, 1), "2026-11-01T14:00:00Z", "2026-11-01T16:00:00Z"),
        ("wednesday_noon", date(2026, 11, 4), "2026-11-04T17:00:00Z", "2026-11-04T20:00:00Z"),
        ("sunday_morning", date(2027, 3, 7), "2027-03-07T14:00:00Z", "2027-03-07T16:00:00Z"),
        ("sunday_morning", date(2027, 3, 14), "2027-03-14T13:00:00Z", "2027-03-14T15:00:00Z"),
    ])
    def test_dst_aware_windows(self, name, day, opens, closes):
        # DST ends Sun 2026-11-01 02:00 and starts Sun 2027-03-14 02:00 (Toronto);
        # the local slot time stays 12:00 / 09:00, the UTC instants move.
        slot = on.slot_for(name, day)
        assert (slot["opens_utc"], slot["closes_utc"]) == (opens, closes)

    def test_slots_follow_the_calendar_across_dst(self):
        rows = on.tracked_slots(datetime(2026, 11, 5, tzinfo=UTC))
        ids = [r["slot_id"] for r in rows]
        assert ids[-4:] == ["2026-10-28_wednesday_noon", "2026-11-01_sunday_morning",
                            "2026-11-04_wednesday_noon", "2026-11-08_sunday_morning"]
        assert len(set(ids)) == len(ids)
        assert all(on.parse_utc(r["intended_utc"]).astimezone(on.TORONTO).strftime("%H:%M")
                   in ("12:00", "09:00") for r in rows)

    def test_no_slots_before_tracking_start(self, store):
        rows = cov(store, datetime(2026, 12, 1, tzinfo=UTC))
        assert rows[0]["slot_id"] == WED and on.TRACKING_START == date(2026, 10, 7)
        assert all(r["slot_id"] >= "2026-10-07" for r in rows)
        early = on.slot_for("sunday_morning", date(2026, 10, 4))
        assert on.slot_state(early, [], WED_EVENING)["state"] == "not_tracked"

    def test_tracking_start_is_not_inferred_from_captures(self, store):
        # A later first capture doesn't move the start.
        store.capture(SUN_SLOT, sun_events())
        assert states(cov(store, SUN_CLOSE + timedelta(minutes=1)))[:2] == [
            (WED, "missed"), (SUN, "captured")]

    @pytest.mark.parametrize("now,last", [
        (WED_EVENING, SUN),                                            # this Sunday pending
        (SUN_SLOT + timedelta(minutes=30), SUN),                        # Sunday window open
        (SUN_CLOSE, SUN),                                               # closing instant
        (SUN_CLOSE + timedelta(seconds=1), "2026-10-18_sunday_morning"),  # next Sunday
    ])
    def test_bounded_horizon(self, store, now, last):
        rows = cov(store, now)
        assert rows[-1]["slot_id"] == last and rows[0]["slot_id"] == WED


# ----------------------------------------------------------------- CLI --

class TestCli:
    def test_cli_todays_dataset(self, store, monkeypatch, capsys):
        store.capture(AD_HOC_AT, wed_events(updated="2026-10-07T19:05:00Z"), slot="ad_hoc")
        code, header, rows = cli_rows(monkeypatch, capsys, store, WED_EVENING)
        assert code == on.EXIT_OK and "since 2026-10-07" in header
        assert [(s, st) for s, st, _, _ in rows] == [(WED, "missed"), (SUN, "pending")]
        assert rows[0][3] == "Missed: no capture in the window ending Wed Oct 7, 15:00 EDT"

    def test_cli_corrupt_capture_fails_visibly(self, store, monkeypatch, capsys):
        store.capture_dir.mkdir(parents=True, exist_ok=True)
        (store.capture_dir / "20261007T160000Z-aaaaaaaaaaaa.json").write_text("{broken")
        monkeypatch.setattr(on, "CAPTURE_DIR", store.capture_dir)
        assert on.main(["coverage"]) != on.EXIT_OK
        assert "::error::" in capsys.readouterr().out

    def test_cli_days_limits_history_only(self, store, monkeypatch, capsys):
        now = datetime(2026, 11, 20, tzinfo=UTC)
        monkeypatch.setattr(on, "CAPTURE_DIR", store.capture_dir)
        monkeypatch.setattr(on.ps, "utc_now", lambda: now)
        on.main(["coverage", "--days", "7"])
        lines = capsys.readouterr().out.splitlines()[1:]
        assert lines and all(line[:10] >= "2026-11-13" for line in lines)


# ---------------------------------------------------------------- page --

class TestPage:
    def test_page_todays_dataset(self, store, render):
        store.capture(AD_HOC_AT, wed_events(updated="2026-10-07T19:05:00Z"), slot="ad_hoc")
        at = render(WED_EVENING)
        df = coverage(at)
        assert list(df["Slot"]) == [SUN, WED]                       # newest first
        assert list(df["Status"]) == ["Pending", "Missed"]
        assert "No Wednesday/Sunday comparison yet" in " ".join(i.value for i in at.info)
        assert table(at) is None and not at.metric                  # no invented comparison
        assert list(uncompared(at)["Slot"]) == ["2026-10-07_ad_hoc"]

    def test_page_with_no_files_shows_expected_slots(self, store, render):
        at = render(WED_EVENING)
        assert "No observations yet" in " ".join(i.value for i in at.info)
        assert list(coverage(at)["Status"]) == ["Pending", "Missed"]

    def test_page_and_cli_agree(self, store, render, monkeypatch, capsys):
        store.capture(WED_SLOT, [])
        store.capture(SUN_SLOT + timedelta(minutes=40), sun_events())
        now = SUN_CLOSE + timedelta(hours=1)
        _, _, cli = cli_rows(monkeypatch, capsys, store, now)
        page = coverage(render(now))
        assert [(s, label) for s, _, _, label in reversed(cli)] == \
            list(zip(page["Slot"], page["Detail"]))
        assert [run or "" for _, _, run, _ in reversed(cli)] == list(page["Capture"])
        assert [olt.COVERAGE_STATE_LABELS[st] for _, st, _, _ in reversed(cli)] == \
            list(page["Status"])

    def test_state_changes_during_session_without_file_changes(self, store, render):
        store.capture(WED_SLOT, [])                                   # evidence only
        at = render(WED_SLOT - timedelta(minutes=1))
        status = lambda: dict(zip(coverage(at)["Slot"], coverage(at)["Status"]))  # noqa: E731
        assert status()[WED] == "Pending"
        at = render(WED_SLOT + timedelta(hours=1), at)
        assert status()[WED] == "Awaiting capture"
        at = render(WED_CLOSE + timedelta(seconds=1), at)
        assert status()[WED] == "Missed"

    def test_corrupt_capture_is_an_integrity_error_not_coverage(self, store, render):
        store.capture(WED_SLOT, wed_events())
        (store.capture_dir / "20261007T200000Z-aaaaaaaaaaaa.json").write_text("{broken")
        at = render(WED_EVENING)
        assert any("Integrity error" in e.value for e in at.error)
        assert coverage(at) is None and not at.info

    def test_existing_comparison_unchanged_and_sources_separate(self, store, render):
        store.capture(WED_SLOT, wed_events())
        store.capture(SUN_SLOT, sun_events())
        store.manual("2026-10-11T13:30:00Z", SUN_SLOT + timedelta(hours=1))
        at = render(SUN_CLOSE + timedelta(hours=1))
        rows = table(at)
        assert (rows["Status"] == "Compared").any() and at.metric       # comparison as before
        assert set(rows["Source"]) == {"automated feed"}                # default Ontario view
        cov_rows = coverage(at)
        assert list(cov_rows.loc[cov_rows["Slot"].isin([WED, SUN]), "Status"]) == \
            ["Captured", "Captured"]
        assert "manual" not in " ".join(cov_rows["Detail"]).lower()


# ------------------------------------------------- production untouched --

def test_production_captures_unchanged(store, render, monkeypatch, capsys):
    real = ROOT / "data_files" / "ontario_spreads"

    def snapshot():
        return {p: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in real.rglob("*") if p.is_file()} if real.exists() else {}
    before = snapshot()
    store.capture(WED_SLOT, wed_events())
    cli_rows(monkeypatch, capsys, store, WED_EVENING)
    render(WED_EVENING)
    assert snapshot() == before


# ------------------------------------------------------- review checks --

class TestSlotIntegrity:
    """A slot is filled only by a validated capture whose slot block is
    exactly what Phase 1's resolve_slot produces for that calendar slot."""

    def _forge(self, store, **slot_changes):
        doc = store.capture(WED_SLOT + timedelta(minutes=5), wed_events()).doc
        path = store.capture_dir / f"{doc['run_id']}.json"
        doc = {**doc, "slot": {**doc["slot"], **slot_changes}}
        doc.pop(on.CHECKSUM_FIELD)
        on._seal(doc)                                   # checksum recomputed: only the slot lies
        path.write_text(on.json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
        return doc

    @pytest.mark.parametrize("changes,match", [
        ({"slot_id": "2026-10-14_wednesday_noon"}, "disagree"),           # another week's slot
        ({"slot_id": "2026-10-11_wednesday_noon"}, "calendar slot"),      # not a Wednesday
        ({"name": "sunday_morning"}, "disagree"),
        ({"intended_utc": "2026-10-07T17:00:00Z"}, "disagree"),
        ({"delay_minutes": 181.0, "status": "late"}, "outside the window"),
        ({"delay_minutes": -1.0}, "outside the window"),
        ({"status": "late"}, "status disagrees"),
        ({"delay_minutes": 60.0, "status": "late"}, "doesn't match the slot's delay"),
        ({"name": "ad_hoc", "status": "ad_hoc"}, "inconsistent ad-hoc"),
        ({"timezone": "UTC"}, "timezone"),
    ])
    def test_inconsistent_slot_block_is_rejected(self, store, changes, match):
        self._forge(store, **changes)
        with pytest.raises(on.ValidationError, match=match):
            on.load_captures(store.capture_dir)
        with pytest.raises(on.ValidationError):
            on.slot_coverage(WED_EVENING, store.capture_dir)

    def test_forged_slot_fails_visibly_in_cli_and_page(self, store, render, monkeypatch, capsys):
        self._forge(store, slot_id="2026-10-14_wednesday_noon")
        monkeypatch.setattr(on, "CAPTURE_DIR", store.capture_dir)
        assert on.main(["coverage"]) != on.EXIT_OK
        assert "::error::" in capsys.readouterr().out
        at = render(WED_EVENING)
        assert any("Integrity error" in e.value for e in at.error) and coverage(at) is None

    def test_validated_captures_still_pass(self, store):
        store.capture(WED_SLOT, wed_events())                               # on time
        store.capture(SUN_SLOT + timedelta(hours=2), sun_events())         # last instant, late
        store.capture(AD_HOC_AT, wed_events(updated="2026-10-07T19:05:00Z"), slot="ad_hoc")
        assert len(on.load_captures(store.capture_dir)) == 3


class TestDeterminism:
    def test_timestamp_tie_breaks_by_run_id_like_the_report(self, store, tmp_path):
        # Two usable captures of one slot with the same captured_at.
        a = store.capture(WED_SLOT + timedelta(minutes=10), wed_events())
        other = tmp_path / "elsewhere"
        b = store.capture(WED_SLOT + timedelta(minutes=10), wed_events(), directory=other)
        shutil.copy(b.path, store.capture_dir / b.path.name)
        assert a.doc["captured_at"] == b.doc["captured_at"] and a.doc["run_id"] != b.doc["run_id"]
        first = min(a.doc["run_id"], b.doc["run_id"])
        for _ in range(3):                                   # same answer every time
            row = {r["slot_id"]: r for r in cov(store, WED_EVENING)}[WED]
            assert row["run_id"] == first and row["attempts"] == 2
        report = olt.build(store.capture_dir, store.manual_dir)
        assert report["weeks"][0]["wednesday"]["run_id"] == first      # comparison agrees


class TestDaysFilter:
    def test_days_only_hides_whole_rows_never_evidence(self, store, monkeypatch, capsys):
        store.capture(WED_SLOT, [])                                    # evidence, early in window
        now = WED_EVENING
        monkeypatch.setattr(on, "CAPTURE_DIR", store.capture_dir)
        monkeypatch.setattr(on.ps, "utc_now", lambda: now)
        on.main(["coverage", "--days", "1"])
        shown = {line[:32].strip(): line for line in capsys.readouterr().out.splitlines()[1:]}
        assert "1 capture with no quotes (not counted)" in shown[WED]   # evidence intact
        on.main(["coverage", "--days", "0"])                           # closed > 0 days ago
        shown = {line[:32].strip() for line in capsys.readouterr().out.splitlines()[1:]}
        assert WED not in shown and SUN in shown                       # row hidden, not altered


class TestCalendarBoundaries:
    @pytest.mark.parametrize("now,last", [
        # Toronto midnight, not UTC midnight, decides the calendar day.
        (datetime(2026, 10, 11, 3, 59, 59, tzinfo=UTC), SUN),            # Sat 23:59:59 EDT
        (datetime(2026, 10, 11, 4, 0, 0, tzinfo=UTC), SUN),              # Sun 00:00 EDT
        # DST ends Sun 2026-11-01: the 11:00 EST close is 16:00 UTC.
        (datetime(2026, 11, 1, 15, 59, 59, tzinfo=UTC), "2026-11-01_sunday_morning"),
        (datetime(2026, 11, 1, 16, 0, 0, tzinfo=UTC), "2026-11-01_sunday_morning"),
        (datetime(2026, 11, 1, 16, 0, 1, tzinfo=UTC), "2026-11-08_sunday_morning"),
        # DST starts Sun 2027-03-14: the 11:00 EDT close is 15:00 UTC.
        (datetime(2027, 3, 14, 15, 0, 0, tzinfo=UTC), "2027-03-14_sunday_morning"),
        (datetime(2027, 3, 14, 15, 0, 1, tzinfo=UTC), "2027-03-21_sunday_morning"),
    ])
    def test_next_sunday_horizon(self, now, last):
        slots = on.tracked_slots(now)
        assert slots[-1]["slot_id"] == last and slots[0]["slot_id"] == WED
        assert sum(s["name"] == "sunday_morning" and on.parse_utc(s["closes_utc"]) >= now
                   for s in slots) == 1                                  # exactly one open Sunday

    @pytest.mark.parametrize("now", [
        datetime(2026, 10, 7, 3, 59, tzinfo=UTC),      # still Oct 6 in Toronto
        datetime(2026, 10, 7, 4, 30, tzinfo=UTC),      # Oct 7, 00:30 Toronto
        datetime(2026, 10, 1, tzinfo=UTC),             # a week before tracking starts
    ])
    def test_tracking_start_is_a_toronto_date(self, now):
        rows = on.expected_coverage([], now)
        assert [(r["slot_id"], r["state"]) for r in rows] == [(WED, "pending"), (SUN, "pending")]
