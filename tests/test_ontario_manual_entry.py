"""FanDuel Ontario manual-quote entry form on the Ontario Line Timing page, and
the Phase 1 helpers it reuses (prepare/save/Toronto conversion).

Temporary capture/manual/schedule directories only; every network call raises
while the page renders; the clock is fixed.
"""

import hashlib
import threading
from datetime import date, datetime, time, timedelta, timezone

import pytest

import ontario_line_timing as olt
import ontario_spreads as on
from test_ontario_line_timing import render, store, table  # noqa: F401 (fixtures)
from test_ontario_spread_report import sun_events, wed_events
from test_ontario_spreads import SUN_SLOT, WED_SLOT
from test_team_features import ROOT

UTC = timezone.utc
MAN = olt.rpt.GROUP_MANUAL
# Wednesday Oct 7 2026, 13:00 Toronto (EDT) - inside the Wednesday slot window.
WED_1300 = datetime(2026, 10, 7, 17, 0, tzinfo=UTC)
CHI_GB = "Week 5: Chicago Bears @ Green Bay Packers (Sun Oct 11, 13:00 EDT)"


@pytest.fixture
def entry(store, monkeypatch):
    monkeypatch.setattr(on, "SCHEDULE_PATH", store.schedule)
    return store


def manual_files(store):
    return sorted(store.manual_dir.glob("*.json")) if store.manual_dir.exists() else []


def fill(at, *, game=CHI_GB, team="CHI", spread=3.5, price=-110, opp=None,
         day=date(2026, 10, 7), at_time=time(12, 30), note=""):
    box = at.selectbox(key="fd_game")
    box.select_index(box.options.index(game))
    at.run()
    at.selectbox(key="fd_team").set_value(team)
    at.number_input(key="fd_spread").set_value(spread)
    at.number_input(key="fd_price").set_value(price)
    at.number_input(key="fd_opp_price").set_value(opp)
    at.date_input(key="fd_date").set_value(day)
    at.time_input(key="fd_time").set_value(at_time)
    at.text_input(key="fd_note").set_value(note)
    at.button(key="fd_preview").click()
    return at


def text(at, element):
    return " ".join(e.value for e in getattr(at, element))


# ------------------------------------------------------- Phase 1 helpers --

class TestTorontoConversion:
    def test_edt_and_est(self):
        assert on.toronto_local_to_utc(date(2026, 10, 7), time(12, 5)) == \
            datetime(2026, 10, 7, 16, 5, tzinfo=UTC)
        assert on.toronto_local_to_utc(date(2026, 11, 4), time(12, 5)) == \
            datetime(2026, 11, 4, 17, 5, tzinfo=UTC)

    def test_ambiguous_and_nonexistent_times_rejected(self):
        with pytest.raises(on.ValidationError, match="happens twice"):
            on.toronto_local_to_utc(date(2026, 11, 1), time(1, 30))      # DST ends
        with pytest.raises(on.ValidationError, match="doesn't exist"):
            on.toronto_local_to_utc(date(2026, 3, 8), time(2, 30))       # DST starts


class TestPrepareAndSave:
    def _args(self, **kw):
        a = dict(game_id="2026_05_CHI_GB", team="CHI", handicap=3.5, price=-110,
                 observed_at="2026-10-07T16:30:00Z")
        a.update(kw)
        return a

    def test_prepare_writes_nothing(self, entry):
        doc = on.prepare_manual_quote(**self._args(), now=WED_1300, schedule_path=entry.schedule)
        on.validate_manual(doc)
        assert doc["source"] == "manual" and doc["book_key"] == "fanduel_on_manual"
        assert manual_files(entry) == []

    @pytest.mark.parametrize("kw,match", [
        ({"price": -50}, "American"), ({"price": 0}, "American"),
        ({"handicap": 3.25}, "half-point"), ({"handicap": 61}, "half-point"),
        ({"observed_at": "2026-10-07T17:30:00Z"}, "future"),
        ({"observed_at": "2026-10-07T12:30:00"}, "timezone"),
        ({"team": "DAL"}, "not playing"),
        ({"opponent_price": 50}, "American"),
    ])
    def test_rejections(self, entry, kw, match):
        with pytest.raises(on.ValidationError, match=match):
            on.prepare_manual_quote(**self._args(**kw), now=WED_1300, schedule_path=entry.schedule)

    def test_observation_at_or_after_kickoff_rejected(self, entry):
        after = datetime(2026, 10, 11, 17, 30, tzinfo=UTC)          # kickoff 17:00 UTC
        for observed in ("2026-10-11T17:00:00Z", "2026-10-11T17:10:00Z"):
            with pytest.raises(on.ValidationError, match="not before kickoff"):
                on.prepare_manual_quote(**self._args(observed_at=observed), now=after,
                                        schedule_path=entry.schedule)

    def test_concurrent_identical_saves_write_once(self, entry):
        # Two submissions of the same observation, prepared separately (so with
        # different quote IDs), saved at the same time.
        docs = [on.prepare_manual_quote(**self._args(), now=WED_1300 + timedelta(seconds=i),
                                        schedule_path=entry.schedule) for i in range(6)]
        results, errors = [], []

        def save(d):
            try:
                results.append(on.save_manual_quote(d, entry.manual_dir)[1])
            except on.CaptureError:          # lock briefly held by another saver
                errors.append("busy")

        threads = [threading.Thread(target=save, args=(d,)) for d in docs]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert len(manual_files(entry)) == 1 and results.count(True) == 1
        assert not list(entry.manual_dir.glob(".*.lock"))

    def test_cli_path_unchanged(self, entry):
        path, created, doc = on.manual_quote(**self._args(), now=WED_1300,
                                             schedule_path=entry.schedule,
                                             manual_dir=entry.manual_dir, repo_dir=ROOT)
        again = on.manual_quote(**self._args(), now=WED_1300 + timedelta(minutes=5),
                                schedule_path=entry.schedule, manual_dir=entry.manual_dir,
                                repo_dir=ROOT)
        assert created and again[1] is False and again[0] == path


# ------------------------------------------------------------- page form --

class TestForm:
    def test_preview_writes_nothing_then_save_writes_once(self, entry, render):
        at = render(WED_1300)
        assert manual_files(entry) == []                 # nothing before submission
        fill(at, team="CHI", spread=3.5, price=-110)
        at = render(WED_1300, at)
        assert "Preview: Chicago Bears +3.5 at −110" in text(at, "markdown")
        assert "not a placed bet" in text(at, "caption")
        assert manual_files(entry) == []                 # preview only
        at.button(key="fd_save").click()
        at = render(WED_1300, at)
        files = manual_files(entry)
        assert len(files) == 1
        doc = on.json.loads(files[0].read_text(encoding="utf-8"))
        assert (doc["team"], doc["team_side"], doc["handicap"], doc["price"]) == ("CHI", "away", 3.5, -110)
        assert doc["opponent_handicap"] == -3.5 and doc["observed_at"] == "2026-10-07T16:30:00Z"
        assert (doc["source"], doc["book_key"], doc["jurisdiction"]) == ("manual", "fanduel_on_manual", "CA-ON")
        assert "Saved: Chicago Bears +3.5 at −110" in text(at, "success")

    def test_favourite_sign_and_opponent(self, entry, render):
        at = fill(render(WED_1300), team="GB", spread=-3.0, price=105, opp=-125)
        at = render(WED_1300, at)
        assert "Preview: Green Bay Packers −3 at +105" in text(at, "markdown")
        assert "Opponent: Chicago Bears +3 at −125" in text(at, "caption")
        at.button(key="fd_save").click()
        render(WED_1300, at)
        doc = on.json.loads(manual_files(entry)[0].read_text(encoding="utf-8"))
        assert (doc["team_side"], doc["handicap"], doc["opponent_handicap"], doc["opponent_price"]) == (
            "home", -3.0, 3.0, -125)

    def test_toronto_time_shown_and_converted(self, entry, render):
        at = fill(render(WED_1300), at_time=time(12, 5))
        at = render(WED_1300, at)
        assert "Observed Wed Oct 7, 12:05 EDT (2026-10-07T16:05:00Z UTC)" in text(at, "caption")

    @pytest.mark.parametrize("kw,match", [
        ({"price": -50}, "American odds"),
        ({"spread": 3.25}, "half-point"),
        ({"price": None}, "enter both"),
        ({"spread": None}, "enter both"),
        ({"at_time": time(13, 30)}, "future"),                  # now is 13:00 Toronto
    ])
    def test_invalid_input_rejected_without_writing(self, entry, render, kw, match):
        at = fill(render(WED_1300), **kw)
        at = render(WED_1300, at)
        assert any(match in e.value for e in at.error)
        assert at.button(key="fd_save").disabled                    # no preview -> Save disabled
        assert manual_files(entry) == []

    def test_rerun_after_save_does_not_write_again(self, entry, render):
        at = fill(render(WED_1300))
        at = render(WED_1300, at)
        at.button(key="fd_save").click()
        at = render(WED_1300, at)
        for _ in range(3):                                 # plain reruns
            at = render(WED_1300, at)
        assert len(manual_files(entry)) == 1

    def test_identical_resubmission_is_not_duplicated(self, entry, render):
        at = fill(render(WED_1300))
        at = render(WED_1300, at)
        at.button(key="fd_save").click()
        at = render(WED_1300, at)
        at = fill(at)                                      # same quote again
        at = render(WED_1300, at)
        assert "already recorded" in text(at, "info")
        assert at.button(key="fd_save").disabled
        assert len(manual_files(entry)) == 1

    def test_slot_labels(self, entry, render):
        at = fill(render(WED_1300), at_time=time(12, 30))
        at = render(WED_1300, at)
        assert "Inside this game's Wednesday slot window (2026-10-07_wednesday_noon)" in text(at, "info")
        thursday = datetime(2026, 10, 8, 16, 0, tzinfo=UTC)
        at = fill(render(thursday), day=date(2026, 10, 8), at_time=time(11, 0))
        at = render(thursday, at)
        assert "Outside the Wednesday 12:00-15:00 and Sunday 09:00-11:00" in text(at, "warning")

    def test_out_of_slot_observation_listed_but_not_compared(self, entry, render):
        thursday = datetime(2026, 10, 8, 16, 0, tzinfo=UTC)
        at = fill(render(thursday), day=date(2026, 10, 8), at_time=time(11, 0))
        at = render(thursday, at)
        at.button(key="fd_save").click()
        at = render(thursday, at)
        assert at.segmented_control[0].value == MAN
        recorded = at.dataframe[0].value
        assert recorded["Quote"].tolist() == ["Chicago Bears +3.5 at −110"]
        assert recorded["Observed (Toronto)"].tolist() == ["Thu Oct 8, 11:00 EDT"]
        assert recorded["Source"].tolist() == ["manual observation (FanDuel Ontario)"]
        assert recorded["Status"].tolist() == ["Outside the slot windows (not compared)"]
        assert "No comparison rows from this source" in text(at, "info")


# ------------------------------------------------------ after saving --

class TestAfterSave:
    def test_saved_slot_observation_appears_in_manual_view(self, entry, render):
        entry.capture(WED_SLOT, wed_events())
        at = fill(render(WED_1300))
        at = render(WED_1300, at)
        at.button(key="fd_save").click()
        at = render(WED_1300, at)                          # no cache clear between runs
        assert at.segmented_control[0].value == MAN
        assert (at.selectbox(key="olt_season").value, at.selectbox(key="olt_week").value) == (2026, 5)
        recorded = at.dataframe[0].value
        assert recorded["Status"].tolist() == ["Wednesday slot observation (in the comparison)"]
        rows = table(at)
        chi = rows[rows["Team"] == "CHI"].iloc[0]
        assert (chi["Wed spread"], chi["Wed price"], chi["Source"]) == ("+3.5", "-110", "manual entry")
        assert chi["Wed captured (Toronto)"] == "Wed Oct 7, 12:30 EDT"
        assert chi["Status"] == "Sunday comparison pending"
        # After the Sunday window closes with no manual Sunday quote:
        at = render(datetime(2026, 10, 11, 15, 1, tzinfo=UTC), at)
        chi = table(at)[table(at)["Team"] == "CHI"].iloc[0]
        assert chi["Status"] == "No manual Sunday observation in the slot window"

    def test_manual_never_in_ontario_or_us_views(self, entry, render):
        entry.capture(WED_SLOT, wed_events())
        entry.capture(SUN_SLOT, sun_events())
        at = fill(render(WED_1300))
        at = render(WED_1300, at)
        at.button(key="fd_save").click()
        at = render(WED_1300, at)
        for view in (olt.rpt.GROUP_ONTARIO, olt.rpt.GROUP_US_REFERENCE):
            at.segmented_control[0].set_value(view)
            at = render(WED_1300, at)
            assert "manual entry" not in set(table(at)["Source"])

    def test_only_the_manual_file_is_written(self, entry, render, tmp_path):
        entry.capture(WED_SLOT, wed_events())
        log = ROOT / "data_files" / "betting_recommendations_log.csv"
        log_hash = hashlib.sha256(log.read_bytes()).hexdigest() if log.exists() else None
        before = {p for p in tmp_path.rglob("*") if p.is_file()}
        at = fill(render(WED_1300))
        at = render(WED_1300, at)
        at.button(key="fd_save").click()
        render(WED_1300, at)
        new = {p for p in tmp_path.rglob("*") if p.is_file()} - before
        assert [p.parent.name for p in new] == ["manual"]
        assert (hashlib.sha256(log.read_bytes()).hexdigest() if log.exists() else None) == log_hash
        assert not (ROOT / "data_files" / "ontario_spreads").exists()

    def test_corrupt_manual_file_blocks_form_with_integrity_error(self, entry, render):
        entry.manual_dir.mkdir(parents=True)
        (entry.manual_dir / "20261007T160000Z-aaaaaaaaaaaa.json").write_text("{broken")
        at = render(WED_1300)
        assert any("Integrity error" in e.value for e in at.error)
        assert not [b for b in at.button if b.key == "fd_preview"]


@pytest.mark.parametrize("observed,state", [
    (datetime(2026, 10, 7, 16, 30, tzinfo=UTC), "intended_wednesday"),   # Wed 12:30 EDT
    (datetime(2026, 10, 11, 13, 30, tzinfo=UTC), "intended_sunday"),     # Sun 09:30 EDT
    (datetime(2026, 9, 30, 16, 30, tzinfo=UTC), "other_slot"),           # a week early
    (datetime(2026, 10, 8, 16, 30, tzinfo=UTC), "outside"),              # Thursday
    (datetime(2026, 10, 7, 19, 1, tzinfo=UTC), "outside"),               # Wed 15:01, window closed
])
def test_manual_slot_status(observed, state):
    assert olt.manual_slot_status(observed, "2026-10-11T17:00:00Z")["state"] == state



# ------------------------------------------------------- review fixes --

class TestPreviewSaveConsistency:
    def test_values_changed_after_preview_are_not_saved(self, entry, render):
        at = fill(render(WED_1300), spread=3.5)
        at = render(WED_1300, at)
        assert "Preview: Chicago Bears +3.5" in text(at, "markdown")
        at.number_input(key="fd_spread").set_value(7.5)       # edited, not re-previewed
        at.button(key="fd_save").click()
        at = render(WED_1300, at)
        assert "the form changed after Preview" in text(at, "error")
        assert manual_files(entry) == []
        assert at.button(key="fd_save").disabled                # preview cleared

    @pytest.mark.parametrize("key,value", [("fd_team", "GB"), ("fd_price", -115),
                                           ("fd_time", time(12, 45)), ("fd_note", "changed")])
    def test_any_changed_field_requires_a_new_preview(self, entry, render, key, value):
        at = fill(render(WED_1300))
        at = render(WED_1300, at)
        setter = {"fd_team": at.selectbox, "fd_price": at.number_input,
                  "fd_time": at.time_input, "fd_note": at.text_input}[key]
        setter(key=key).set_value(value)
        at.button(key="fd_save").click()
        at = render(WED_1300, at)
        assert "the form changed after Preview" in text(at, "error") and manual_files(entry) == []

    def test_game_kicking_off_between_preview_and_save_is_refused(self, entry, render):
        before = datetime(2026, 10, 9, 0, 10, tzinfo=UTC)       # TB@DAL kicks off 00:15Z
        at = fill(render(before), game="Week 5: Tampa Bay Buccaneers @ Dallas Cowboys "
                                        "(Thu Oct 8, 20:15 EDT)",
                  team="DAL", spread=-3.5, day=date(2026, 10, 8), at_time=time(20, 5))
        at = render(before, at)
        assert "Preview: Dallas Cowboys" in text(at, "markdown")
        at.button(key="fd_save").click()
        at = render(datetime(2026, 10, 9, 0, 20, tzinfo=UTC), at)   # saved after kickoff
        assert "kicked off at Thu Oct 8, 20:15 EDT" in text(at, "error")
        assert manual_files(entry) == []

    def test_save_revalidates_with_current_clock(self, entry, render, monkeypatch):
        at = fill(render(WED_1300))
        at = render(WED_1300, at)
        # The schedule changes before Save: the game now kicked off before the
        # observation, so the observation is no longer pregame.
        import pandas as pd
        df = pd.read_csv(entry.schedule, sep="\t")
        df.loc[df["game_id"] == "2026_05_CHI_GB", ["gameday", "gametime"]] = ["2026-10-07", "12:00"]
        df.to_csv(entry.schedule, sep="\t", index=False)
        at.button(key="fd_save").click()
        at = render(WED_1300, at)
        assert "not before kickoff" in text(at, "error") and manual_files(entry) == []

    def test_saving_clears_manual_view_filters(self, entry, render):
        entry.capture(WED_SLOT, wed_events())
        at = fill(render(WED_1300), team="GB", spread=-3.5)
        at = render(WED_1300, at)
        at.button(key="fd_save").click()
        at = render(WED_1300, at)
        at.selectbox(key=f"olt_team_{MAN}").set_value("GB")    # a manual-view filter
        at = render(WED_1300, at)
        at = fill(at, team="CHI", spread=3.5, at_time=time(12, 40))
        at = render(WED_1300, at)
        at.button(key="fd_save").click()
        at = render(WED_1300, at)
        assert at.selectbox(key=f"olt_team_{MAN}").value == "All teams"
        assert sorted(set(table(at)["Team"])) == ["CHI", "GB"]


class TestQuoteIdentity:
    def test_identity_fields(self):
        assert on._MANUAL_IDENTITY == ("game_id", "team", "observed_at", "handicap", "price",
                                       "opponent_price")

    def test_changed_note_does_not_defeat_duplicate_protection(self, entry):
        args = dict(game_id="2026_05_CHI_GB", team="CHI", handicap=3.5, price=-110,
                    observed_at="2026-10-07T16:30:00Z")
        first = on.prepare_manual_quote(**args, note="first", now=WED_1300,
                                        schedule_path=entry.schedule)
        on.save_manual_quote(first, entry.manual_dir)
        again = on.prepare_manual_quote(**args, note="different note", entered_by="someone else",
                                        now=WED_1300 + timedelta(minutes=3),
                                        schedule_path=entry.schedule)
        path, created, doc = on.save_manual_quote(again, entry.manual_dir)
        assert created is False and doc["note"] == "first" and len(manual_files(entry)) == 1

    def test_new_preview_of_identical_quote_is_flagged(self, entry, render):
        at = fill(render(WED_1300), note="first")
        at = render(WED_1300, at)
        at.button(key="fd_save").click()
        at = render(WED_1300, at)
        at = fill(at, note="second")                          # new preview, new token
        at = render(WED_1300, at)
        assert "already recorded" in text(at, "info") and at.button(key="fd_save").disabled
        assert len(manual_files(entry)) == 1


class TestLocks:
    def _lock(self, entry, age_minutes, legacy=False):
        entry.manual_dir.mkdir(parents=True, exist_ok=True)
        path = entry.manual_dir / ".manual_entry.lock"
        started = WED_1300 - timedelta(minutes=age_minutes)
        if legacy:
            path.write_text("someone-else")
            import os
            os.utime(path, (started.timestamp(), started.timestamp()))
        else:
            path.write_text(on.json.dumps({"what": "manual-quote save", "id": "x", "pid": 4242,
                                           "host": "h", "started_at": on.iso(started),
                                           "token": "t"}))
        return path

    def _doc(self, entry):
        return on.prepare_manual_quote(game_id="2026_05_CHI_GB", team="CHI", handicap=3.5,
                                       price=-110, observed_at="2026-10-07T16:30:00Z",
                                       now=WED_1300, schedule_path=entry.schedule)

    @pytest.mark.parametrize("legacy", [False, True])
    def test_stale_lock_reported_with_recovery_and_not_deleted(self, entry, monkeypatch, legacy):
        monkeypatch.setattr(on.ps, "utc_now", lambda: WED_1300)
        lock = self._lock(entry, age_minutes=25, legacy=legacy)
        with pytest.raises(on.CaptureError) as exc:
            on.save_manual_quote(self._doc(entry), entry.manual_dir)
        msg = str(exc.value)
        assert "looks stale" in msg and str(lock) in msg and "delete" in msg
        assert lock.exists() and manual_files(entry) == []

    def test_active_lock_reported_as_in_progress_and_kept(self, entry, monkeypatch):
        monkeypatch.setattr(on.ps, "utc_now", lambda: WED_1300)
        lock = self._lock(entry, age_minutes=0)
        with pytest.raises(on.CaptureError, match="in progress - try again"):
            on.save_manual_quote(self._doc(entry), entry.manual_dir)
        assert lock.exists() and on.json.loads(lock.read_text())["pid"] == 4242

    def test_lock_removed_after_caught_failure(self, entry, monkeypatch):
        def boom(*a, **k):
            raise OSError("disk full")
        monkeypatch.setattr(on, "write_artifact", boom)
        with pytest.raises(OSError, match="disk full"):
            on.save_manual_quote(self._doc(entry), entry.manual_dir)
        assert not (entry.manual_dir / ".manual_entry.lock").exists()

    def test_exit_never_deletes_someone_elses_lock(self, entry):
        lock = on.SlotLock(entry.manual_dir, "manual_entry", "me", what="manual-quote save")
        with lock:
            lock.path.write_text(on.json.dumps({"token": "someone-else"}))   # taken over
        assert lock.path.exists()

    def test_page_shows_lock_message_and_writes_nothing(self, entry, render, monkeypatch):
        monkeypatch.setattr(on.ps, "utc_now", lambda: WED_1300)
        at = fill(render(WED_1300))
        at = render(WED_1300, at)
        lock = self._lock(entry, age_minutes=25)
        at.button(key="fd_save").click()
        at = render(WED_1300, at)
        assert "looks stale" in text(at, "error") and str(lock) in text(at, "error")
        assert manual_files(entry) == [] and lock.exists()


def test_other_week_observation_visible_but_never_substitutes(entry, render):
    # Saved on Wed Sep 30 inside that Wednesday's slot window, for the Oct 11
    # game (whose intended Wednesday is Oct 7), then a Sunday Oct 11 quote.
    sep30 = datetime(2026, 9, 30, 17, 0, tzinfo=UTC)
    at = fill(render(sep30), day=date(2026, 9, 30), at_time=time(12, 30))
    at = render(sep30, at)
    assert "isn't this game's Wednesday or Sunday slot" in text(at, "warning")
    at.button(key="fd_save").click()
    render(sep30, at)
    on.manual_quote(game_id="2026_05_CHI_GB", team="CHI", handicap=3.0, price=-110,
                    observed_at="2026-10-11T13:20:00Z",
                    now=datetime(2026, 10, 11, 14, 0, tzinfo=UTC), schedule_path=entry.schedule,
                    manual_dir=entry.manual_dir, repo_dir=ROOT)
    at = render(datetime(2026, 10, 12, 18, 0, tzinfo=UTC))
    at.segmented_control[0].set_value(MAN)
    at = render(datetime(2026, 10, 12, 18, 0, tzinfo=UTC), at)
    recorded = at.dataframe[0].value.set_index("Observed (Toronto)")["Status"]
    assert recorded["Wed Sep 30, 12:30 EDT"] == "Another week's slot (not compared)"
    assert recorded["Sun Oct 11, 09:20 EDT"] == "Sunday slot observation (in the comparison)"
    chi = table(at)[table(at)["Team"] == "CHI"].iloc[0]
    assert chi["Wed spread"] == "" and chi["Status"] == "No manual Wednesday observation in the slot window"
