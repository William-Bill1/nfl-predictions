"""Phase 3: "Ontario Line Timing" page (pages/7_Ontario_Line_Timing.py) and its
display logic (ontario_line_timing.py).

Artifacts are produced by Phase 1's own capture()/manual_quote() against a
mocked API in temporary directories; while the page renders, every network
call raises. The page is rendered with Streamlit's test runtime (AppTest).
"""

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone

import pytest
import requests

import ontario_line_timing as olt
import ontario_spreads as on
from test_ontario_spread_report import Store, ev, sun_events, wed_events, SUN_UPD
from test_ontario_spreads import SUN_SLOT, WED_SLOT, _event
from test_team_features import ROOT

UTC = timezone.utc
PAGE = ROOT / "pages" / "7_Ontario_Line_Timing.py"
THURSDAY = datetime(2026, 10, 8, 18, 0, tzinfo=UTC)       # after Wed slot, before Sunday's
MONDAY = datetime(2026, 10, 12, 18, 0, tzinfo=UTC)        # after the Sunday window closed


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = Store(tmp_path, monkeypatch)
    # The page reads these at run time.
    monkeypatch.setattr(on, "CAPTURE_DIR", s.capture_dir)
    monkeypatch.setattr(on, "MANUAL_DIR", s.manual_dir)
    return s


@pytest.fixture
def render(store, monkeypatch):
    """Render the page with the network blocked and a fixed clock."""
    from streamlit.testing.v1 import AppTest
    import streamlit as st

    def blocked(*a, **k):
        raise AssertionError("network call attempted while rendering the page")

    def run(now, at=None):
        monkeypatch.setattr(requests, "get", blocked)
        monkeypatch.setattr(on.requests, "get", blocked)
        monkeypatch.setattr(olt, "now_utc", lambda: now)
        if at is None:
            st.cache_data.clear()
            at = AppTest.from_file(str(PAGE), default_timeout=60)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return at
    return run


def table(at):
    """The comparison table (always the page's last table; the manual view
    shows its recorded-observations table first)."""
    return at.dataframe[-1].value if at.dataframe else None


def gb_row(df, book="BetMGM (CA - ON)"):
    return df[(df["Team"] == "GB") & (df["Sportsbook"] == book)].iloc[0]


# ------------------------------------------------------------- formatting --

class TestFormatting:
    @pytest.mark.parametrize("h,label", [(-3.5, "-3.5"), (3.5, "+3.5"), (-3.0, "-3"),
                                         (7.0, "+7"), (0.0, "PK"), (None, "")])
    def test_signed_handicaps(self, h, label):
        assert olt.signed_handicap(h) == label

    @pytest.mark.parametrize("p,label", [(-110, "-110"), (105, "+105"), (100, "+100"), (None, "")])
    def test_signed_prices(self, p, label):
        assert olt.signed_price(p) == label

    def test_toronto_times_follow_dst(self):
        assert olt.toronto("2026-10-07T16:00:00Z") == "Wed Oct 7, 12:00 EDT"
        assert olt.toronto("2026-11-04T17:00:00Z") == "Wed Nov 4, 12:00 EST"
        assert olt.toronto("2026-10-11T13:00:00Z") == "Sun Oct 11, 09:00 EDT"

    def test_default_week_is_latest_captured_not_future(self):
        report = {"weeks": [
            {"season": 2026, "week": 5, "wednesday": {"x": 1}, "sunday": None},
            {"season": 2026, "week": 6, "wednesday": None, "sunday": None},
            {"season": 2026, "week": 18, "wednesday": None, "sunday": None}], "rows": []}
        assert olt.default_week(report) == (2026, 5)


# ------------------------------------------------------------- page states --

class TestPage:
    def test_no_observations(self, render):
        at = render(THURSDAY)
        info = " ".join(i.value for i in at.info)
        assert "No observations yet" in info
        assert "Wednesday at 12:00" in info and "Sunday at 09:00" in info
        assert "America/Toronto" in info
        assert not at.dataframe            # never sample odds

    def test_wednesday_only_sunday_pending(self, store, render):
        store.capture(WED_SLOT, wed_events())
        at = render(THURSDAY)
        df = table(at)
        r = gb_row(df)
        assert (r["Wed spread"], r["Wed price"], r["Sun spread"]) == ("-3.5", "-110", "")
        assert r["Status"] == "Sunday comparison pending"
        assert r["Wed captured (Toronto)"] == "Wed Oct 7, 12:00 EDT"
        captions = " ".join(c.value for c in at.caption)
        assert "Pending: slot Sun Oct 11, 09:00 EDT" in captions

    def test_wednesday_only_sunday_missed(self, store, render):
        store.capture(WED_SLOT, wed_events())
        df = table(render(MONDAY))
        assert gb_row(df)["Status"] == "Sunday slot missed"

    def test_complete_comparison(self, store, render):
        store.capture(WED_SLOT, wed_events())
        store.capture(SUN_SLOT, sun_events())
        at = render(MONDAY)
        df = table(at)
        r = gb_row(df)
        assert (r["Wed spread"], r["Wed price"], r["Sun spread"], r["Sun price"]) == (
            "-3.5", "-110", "-2.5", "-115")
        assert r["Spread change"] == "+1" and r["Key 3"] == "Crossed"
        assert r["Outcome"] == "Trade-off (not ranked)" and r["Status"] == "Compared"
        assert r["Wed break-even"] == pytest.approx(52.381, abs=0.01)
        assert r["Sun break-even"] == pytest.approx(53.488, abs=0.01)
        assert r["Sun captured (Toronto)"] == "Sun Oct 11, 09:00 EDT"
        assert r["Sun provider update (Toronto)"] == "Sun Oct 11, 08:55 EDT"
        assert r["Wed quote"] == "fresh, 5 min old at capture"
        la = df[(df["Team"] == "LA")].iloc[0]
        assert (la["Wed spread"], la["Sun spread"], la["Outcome"]) == ("-1.5", "+1.5", "Sunday dominates")
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Compared sides"] == "12" and metrics["Trade-offs (not ranked)"] == "2"
        assert metrics["Sunday dominates"] == "4" and metrics["Wednesday dominates"] == "4"
        assert metrics["Not compared"] == "2"   # TB@DAL (Thursday game), both sides

    def test_thursday_game_and_stale_quote(self, store, render):
        store.capture(WED_SLOT, wed_events())
        stale = sun_events()
        stale[1] = ev("2026_05_CHI_GB", "2026-10-11T10:00:00Z", betmgm_ca_on=(-2.5, -105, 2.5, -115))
        store.capture(SUN_SLOT, stale)
        df = table(render(MONDAY))
        dal = df[df["Team"] == "DAL"].iloc[0]
        assert dal["Status"] == "Unmatched" and dal["Reasons"] == "Kicked off before the Sunday slot"
        r = gb_row(df)
        assert r["Status"] == "Not compared (quote quality)" and r["Outcome"] == ""
        assert r["Reasons"] == "Sunday quote stale" and r["Sun quote"] == "stale, 180 min old at capture"

    def test_late_slot_delay_shown(self, store, render):
        store.capture(WED_SLOT + timedelta(minutes=95), wed_events("2026-10-07T17:30:00Z"))
        at = render(THURSDAY)
        captions = " ".join(c.value for c in at.caption)
        assert "late, 95 min after the slot" in captions
        assert gb_row(table(at))["Wed slot"] == "late"

    def test_corrupt_artifact_is_an_integrity_error(self, store, render):
        r = store.capture(WED_SLOT, wed_events())
        doc = json.loads(r.path.read_text(encoding="utf-8"))
        doc["games"][0]["quotes"][0]["home_price"] = -105          # edited, not re-sealed
        r.path.write_text(json.dumps(doc), encoding="utf-8")
        at = render(THURSDAY)
        assert any("Integrity error" in e.value and "checksum" in e.value for e in at.error)
        assert not at.dataframe

    def test_unparseable_artifact_is_an_integrity_error(self, store, render):
        store.capture_dir.mkdir(parents=True)
        (store.capture_dir / "20261007T160000Z-aaaaaaaaaaaa.json").write_text("{not json")
        at = render(THURSDAY)
        assert any("Integrity error" in e.value for e in at.error) and not at.dataframe


# ---------------------------------------------------------- source views --

class TestSources:
    def test_default_is_ontario_automated_only(self, store, render):
        store.capture(WED_SLOT, wed_events())
        store.capture(SUN_SLOT, sun_events())
        at = render(MONDAY)
        df = table(at)
        assert set(df["Source"]) == {"automated feed"}
        assert "FanDuel" not in " ".join(df["Sportsbook"])
        assert at.segmented_control[0].value == olt.rpt.GROUP_ONTARIO

    def test_us_reference_view_is_labelled_not_ontario(self, store, render):
        store.capture(WED_SLOT, wed_events())
        store.capture(SUN_SLOT, sun_events())
        at = render(MONDAY)
        at.segmented_control[0].set_value(olt.rpt.GROUP_US_REFERENCE)
        at = render(MONDAY, at)
        assert any("US reference quotes; not verified as available in Ontario" in w.value
                   for w in at.warning)
        df = table(at)
        assert set(df["Sportsbook"]) == {"FanDuel"} and set(df["Source"]) == {"US reference feed"}

    def test_manual_view_separate(self, store, render):
        store.capture(WED_SLOT, wed_events())
        store.capture(SUN_SLOT, sun_events())
        store.manual("2026-10-07T16:30:00Z", WED_SLOT + timedelta(hours=1), opponent_price=-110)
        store.manual("2026-10-11T13:20:00Z", SUN_SLOT + timedelta(hours=1), handicap=-3.0,
                     price=-120, opponent_price=100)
        at = render(MONDAY)
        assert "manual" not in set(table(at)["Source"])
        at.segmented_control[0].set_value(olt.rpt.GROUP_MANUAL)
        at = render(MONDAY, at)
        df = table(at)
        assert set(df["Source"]) == {"manual entry"}
        assert set(df["Wed quote"]) == {"manual observation"}
        assert any("Manual entries" in i.value for i in at.info)

    def test_filters(self, store, render):
        store.capture(WED_SLOT, wed_events())
        store.capture(SUN_SLOT, sun_events())
        at = render(MONDAY)
        at.multiselect[0].set_value(["Betano (CA - ON)"])
        at = render(MONDAY, at)
        assert set(table(at)["Sportsbook"]) == {"Betano (CA - ON)"}
        at.multiselect[0].set_value([])
        at.selectbox(key=f"olt_game_{olt.rpt.GROUP_ONTARIO}").set_value("CHI @ GB")
        at = render(MONDAY, at)
        assert set(table(at)["Matchup"]) == {"CHI @ GB"}
        at.selectbox(key=f"olt_team_{olt.rpt.GROUP_ONTARIO}").set_value("CHI")
        at = render(MONDAY, at)
        assert set(table(at)["Team"]) == {"CHI"}

    def test_default_week_is_captured_week(self, store, render):
        store.capture(WED_SLOT, wed_events())
        at = render(THURSDAY)
        assert at.selectbox(key="olt_week").value == 5
        assert at.selectbox(key="olt_season").value == 2026


# ---------------------------------------------------------- caching --

class TestCache:
    def test_fingerprint_changes_on_add_and_edit(self, store):
        empty = olt.source_fingerprint(store.capture_dir, store.manual_dir)
        r = store.capture(WED_SLOT, wed_events())
        added = olt.source_fingerprint(store.capture_dir, store.manual_dir)
        assert added != empty
        r.path.write_bytes(r.path.read_bytes() + b" ")
        assert olt.source_fingerprint(store.capture_dir, store.manual_dir) != added

    def test_new_capture_shown_without_stale_cache(self, store, render):
        store.capture(WED_SLOT, wed_events())
        at = render(MONDAY)
        assert gb_row(table(at))["Status"] == "Sunday slot missed"
        store.capture(SUN_SLOT, sun_events())      # a refresh lands while the app runs
        at = render(MONDAY, at)                     # same session, cache not cleared
        assert gb_row(table(at))["Status"] == "Compared"

    def test_page_is_read_only(self, store, render):
        store.capture(WED_SLOT, wed_events())
        store.capture(SUN_SLOT, sun_events())
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in store.capture_dir.iterdir()}
        render(MONDAY)
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in store.capture_dir.iterdir()}
        assert after == before


# ---------------------------------------------------------- wording, nav --

def test_page_wording(store, render):
    store.capture(WED_SLOT, wed_events())
    at = render(THURSDAY)
    text = " ".join(m.value for m in at.markdown)
    assert "payoff dominance" in text and "integer final" in text
    assert "not ranked" in text and "not a closing line" in text
    assert "does **not** establish a best betting time" in text


def test_page_registered_in_navigation():
    src = (ROOT / "predictions.py").read_text(encoding="utf-8")
    assert re.search(r'st\.Page\("pages/7_Ontario_Line_Timing\.py",\s*title="Ontario Line Timing"', src)
    assert PAGE.exists()



# ------------------------------------------------------- review fixes --

ONT, MAN, US = olt.rpt.GROUP_ONTARIO, olt.rpt.GROUP_MANUAL, olt.rpt.GROUP_US_REFERENCE
# Sunday Oct 11: slot 09:00, window closes 11:00 Toronto (15:00 UTC).
SUN_BEFORE_CLOSE = datetime(2026, 10, 11, 14, 30, tzinfo=UTC)
SUN_AFTER_CLOSE = datetime(2026, 10, 11, 15, 1, tzinfo=UTC)


class TestReviewFixes:
    def test_pending_turns_missed_in_same_session_from_current_time(self, store, render):
        store.capture(WED_SLOT, wed_events())
        at = render(SUN_BEFORE_CLOSE)
        assert gb_row(table(at))["Status"] == "Sunday comparison pending"
        assert "Pending: slot Sun Oct 11, 09:00 EDT, window closes Sun Oct 11, 11:00 EDT" in \
            " ".join(c.value for c in at.caption)
        at = render(SUN_AFTER_CLOSE, at)          # same session, same files, cached report
        assert gb_row(table(at))["Status"] == "Sunday slot missed"
        assert "Missed: no capture in the window ending Sun Oct 11, 11:00 EDT" in \
            " ".join(c.value for c in at.caption)

    def test_cache_invalidated_by_deleted_artifact(self, store, render):
        store.capture(WED_SLOT, wed_events())
        sun = store.capture(SUN_SLOT, sun_events())
        at = render(MONDAY)
        assert gb_row(table(at))["Status"] == "Compared"
        sun.path.unlink()
        at = render(MONDAY, at)
        assert gb_row(table(at))["Status"] == "Sunday slot missed"

    def test_cache_invalidated_by_changed_artifact(self, store, render):
        store.capture(WED_SLOT, wed_events())
        sun = store.capture(SUN_SLOT, sun_events())
        at = render(MONDAY)
        assert table(at) is not None
        sun.path.write_text("{corrupted", encoding="utf-8")
        at = render(MONDAY, at)
        assert any("Integrity error" in e.value for e in at.error) and not at.dataframe

    def test_empty_filter_result_shows_message(self, store, render):
        store.capture(WED_SLOT, wed_events())
        store.capture(SUN_SLOT, sun_events())
        at = render(MONDAY)
        at.multiselect[0].set_value(["PROLINE (CA - ON)"])        # only quotes BUF @ LA
        at.selectbox(key=f"olt_game_{ONT}").set_value("CHI @ GB")
        at = render(MONDAY, at)
        assert any("No rows match these filters" in i.value for i in at.info)
        assert not at.dataframe

    def test_switching_views_keeps_filters_separate(self, store, render):
        store.capture(WED_SLOT, wed_events())
        store.capture(SUN_SLOT, sun_events())
        store.manual("2026-10-07T16:30:00Z", WED_SLOT + timedelta(hours=1), opponent_price=-110)
        store.manual("2026-10-11T13:20:00Z", SUN_SLOT + timedelta(hours=1), handicap=-3.0,
                     price=-120, opponent_price=100)
        at = render(MONDAY)
        at.multiselect[0].set_value(["Betano (CA - ON)"])
        at.selectbox(key=f"olt_game_{ONT}").set_value("DEN @ LAC")
        at.selectbox(key=f"olt_team_{ONT}").set_value("LAC")
        at = render(MONDAY, at)
        assert table(at)[["Sportsbook", "Team"]].values.tolist() == [["Betano (CA - ON)", "LAC"]]
        for view, book in ((US, "FanDuel"), (MAN, "FanDuel Ontario (manual entry)")):
            at.segmented_control[0].set_value(view)
            at = render(MONDAY, at)
            df = table(at)
            assert set(df["Sportsbook"]) == {book}             # no Ontario filter carried over
            assert set(df["Team"]) == {"CHI", "GB"}
        at.segmented_control[0].set_value(ONT)
        at = render(MONDAY, at)            # back to Ontario: filters start cleared, no carry-over
        df = table(at)
        assert len(df) == 14 and not df["Sportsbook"].str.contains("FanDuel").any()

    def test_us_and_manual_never_in_ontario_view(self, store, render):
        store.capture(WED_SLOT, wed_events())
        store.capture(SUN_SLOT, sun_events())
        store.manual("2026-10-07T16:30:00Z", WED_SLOT + timedelta(hours=1), opponent_price=-110)
        df = table(render(MONDAY))
        assert not df["Sportsbook"].str.contains("FanDuel").any()
        assert set(df["Source"]) == {"automated feed"}

    def test_wednesday_only_rows_remain_visible_after_window(self, store, render):
        store.capture(WED_SLOT, wed_events())
        df = table(render(MONDAY))
        assert len(df) == 14 and (df["Wed spread"] != "").sum() == 14
        assert set(df["Sun spread"]) == {""}

    def test_no_writes_anywhere(self, store, render, tmp_path):
        store.capture(WED_SLOT, wed_events())
        before = sorted((p.relative_to(tmp_path).as_posix(), p.stat().st_size)
                        for p in tmp_path.rglob("*") if p.is_file())
        at = render(MONDAY)
        at.segmented_control[0].set_value(US)
        render(MONDAY, at)
        after = sorted((p.relative_to(tmp_path).as_posix(), p.stat().st_size)
                       for p in tmp_path.rglob("*") if p.is_file())
        assert after == before
        assert not (ROOT / "reports").exists()


def test_valid_filter_does_not_carry_into_us_view(store, render):
    # A game/team that also exists in the US view must not filter it silently.
    store.capture(WED_SLOT, wed_events())
    store.capture(SUN_SLOT, sun_events())
    at = render(MONDAY)
    at.selectbox(key=f"olt_game_{ONT}").set_value("CHI @ GB")
    at.selectbox(key=f"olt_team_{ONT}").set_value("GB")
    at = render(MONDAY, at)
    assert set(table(at)["Team"]) == {"GB"}
    at.segmented_control[0].set_value(US)
    at = render(MONDAY, at)
    assert set(table(at)["Team"]) == {"CHI", "GB"}


def test_display_rows_keeps_columns_when_nothing_matches(store):
    store.capture(WED_SLOT, wed_events())
    report = olt.build(store.capture_dir, store.manual_dir)
    df = olt.display_rows(report, ONT, 2026, 5, MONDAY, books=["no_such_book"])
    assert df.empty and list(df.columns) == list(olt.DISPLAY_COLUMNS)
    assert olt.summarize(df)["sides"] == 0
