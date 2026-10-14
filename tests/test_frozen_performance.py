"""Frozen Performance page (pages/9_Frozen_Performance.py) and its display
logic (frozen_performance.py).

Fixtures reuse the frozen-report tests' World: synthetic pregame snapshots
that pass the real validation, temporary schedules, and Ontario captures
written by Phase 1's capture code against a fake provider. The page is
rendered with Streamlit's AppTest, with the network blocked and a fixed date.
"""
import hashlib
import json
import socket
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

import frozen_performance as fp
import pregame_snapshots as ps
from test_frozen_spread_report import S1, S2, World, _two_games_market
from test_ontario_spread_report import ev
from test_ontario_spreads import WED_SLOT
from test_team_features import ROOT

PAGE = ROOT / "pages" / "9_Frozen_Performance.py"
TODAY = date(2026, 10, 14)
UTC = timezone.utc


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture
def w(tmp_path, monkeypatch):
    world = World(tmp_path, monkeypatch)
    monkeypatch.setattr(fp, "SNAPSHOT_DIR", world.snapshots)
    monkeypatch.setattr(fp, "SCHEDULE_PATH", world.schedule)
    monkeypatch.setattr(fp, "CAPTURE_DIR", world.captures)
    monkeypatch.setattr(fp, "today_toronto", lambda: TODAY)
    return world


@pytest.fixture
def render(w, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import streamlit as st

    def run(at=None, today=None):
        if today is not None:
            monkeypatch.setattr(fp, "today_toronto", lambda: today)
        w.write_schedule()
        if at is None:
            st.cache_data.clear()
            at = AppTest.from_file(str(PAGE), default_timeout=60)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        return at
    return run


def table(at, column="Spread result"):
    return next((d.value for d in at.dataframe if column in d.value.columns), None)


def metrics(at) -> list[tuple[str, str]]:
    return [(m.label, m.value) for m in at.metric]


def metric(at, label, nth=0):
    return [v for k, v in metrics(at) if k == label][nth]


def row(at, game):
    t = table(at)
    return t[t["Game"] == game].iloc[0]


def report(w, **kw):
    """The report from the same inputs and settings, for reconciliation."""
    w.write_schedule()
    kw.setdefault("as_of", TODAY)
    return fp.fr.build_report(w.snapshots, w.schedule, w.captures, **kw)


def week5_complete(w):
    """CHI@GB, DEN@LAC, BUF@LA all final (every week-5 game in the schedule)."""
    w.game("2026_05_CHI_GB", 20, 24)        # GB by 4
    w.game("2026_05_DEN_LAC", 23, 20)       # LAC by 3
    w.game("2026_05_BUF_LA", 30, 20)        # LA by 10


# ------------------------------------------------------------- empty states --

class TestEmptyStates:
    def test_no_snapshot_history_shows_completed_games_as_lacking_snapshots(self, w, render):
        week5_complete(w)
        at = render()
        assert any("No snapshot history exists" in i.value for i in at.info)
        assert any("No frozen predictions are selected" in i.value for i in at.info)
        cov = table(at, "Coverage").set_index("Coverage")["Games"]
        assert cov["Completed games in scope"] == 3 and cov["- no snapshot history at all"] == 3
        assert metric(at, "Evaluated (win/loss)") == "0" and metric(at, "Model Brier") == "n/a"
        assert table(at) is None

    def test_no_completed_games_at_all(self, w, render):
        w.game("2026_05_CHI_GB")
        at = render()
        assert any("no completed regular-season games" in i.value for i in at.info)


# ---------------------------------------------------- weeks and provisional --

class TestWeeks:
    def test_default_is_the_latest_fully_completed_week(self, w, render):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, True)])
        week5_complete(w)
        w.game("2026_06_DAL_NYG", 17, 20, day="2026-10-13", at="20:15")
        w.game("2026_06_SF_ARI", day="2026-10-18", at="16:05")         # not played
        at = render()
        season, week = at.selectbox
        assert (season.value, week.value) == (2026, 5)
        assert "Week 6 (provisional)" in week.options and "Week 5" in week.options
        assert not at.warning                                          # week 5 is final

    def test_partly_completed_week_is_labelled_provisional(self, w, render):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, True)])
        week5_complete(w)
        w.game("2026_06_DAL_NYG", 17, 20, day="2026-10-13", at="20:15")
        w.game("2026_06_SF_ARI", day="2026-10-18", at="16:05")
        at = render()
        at.selectbox[1].set_value(6)
        at = render(at)
        assert any("Provisional" in x.value for x in at.warning)

    def test_all_weeks_option_removes_the_week_filter(self, w, render):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, True)])
        week5_complete(w)
        w.game("2026_04_ATL_NO", 24, 17, day="2026-10-05", at="20:15")
        at = render()
        assert at.selectbox[1].options[0] == "All weeks"
        at.selectbox[1].set_value(0)
        at = render(at)
        r = report(w, season=2026, week=None)
        cov = table(at, "Coverage").set_index("Coverage")["Games"]
        assert cov["Completed games in scope"] == \
            r["coverage"]["completed_without_snapshot"]["completed_games_in_scope"] == 4

    def test_pending_game_makes_the_view_provisional(self, w, render):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, True), ("2026_05_BUF_LA", 3.0, 0.5, False)])
        w.game("2026_05_CHI_GB", 20, 24)
        w.game("2026_05_BUF_LA")                                       # Monday, no score yet
        at = render()
        assert any("Provisional" in x.value for x in at.warning)
        assert row(at, "BUF @ LA")["Status"].startswith("Pending")

    def test_unplayed_games_make_the_view_provisional_even_when_excluded(self, w, render):
        # A future game whose frozen snapshot had no line is "excluded", not
        # pending - but the view still isn't final.
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, True),
                        ("2026_06_DAL_NYG", None, None, False, datetime(2026, 10, 16, 0, 15, tzinfo=UTC))])
        week5_complete(w)
        w.game("2026_06_DAL_NYG", day="2026-10-15", at="20:15")
        at = render()
        assert not at.warning                                          # week 5: final
        at.selectbox[1].set_value(0)                                   # all weeks
        at = render(at)
        assert any("Provisional" in x.value for x in at.warning)
        assert row(at, "DAL @ NYG")["Status"] == "Excluded - no line in the frozen snapshot"

    def test_clock_dependent_evaluation_is_outside_the_file_cache(self, w, render):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, True)])
        w.game("2026_05_CHI_GB", 20, 24)
        at = render(today=date(2026, 10, 11))                           # game day = today
        assert row(at, "CHI @ GB")["Status"].startswith("Pending")
        at = render(at, today=TODAY)                                    # same files, later date
        assert row(at, "CHI @ GB")["Spread result"] == "win"


# ------------------------------------------------------------------ results --

class TestResults:
    def test_wins_losses_pushes_at_the_frozen_signed_handicap(self, w, render):
        # Frozen: CHI +3.5 (today's schedule says 7), LAC +2.5 (home dog), BUF +3.
        w.snapshot(S2, [("2026_05_CHI_GB", 3.5, 0.6, True), ("2026_05_DEN_LAC", -2.5, 0.55, False),
                        ("2026_05_BUF_LA", 3.0, 0.52, False)])
        w.game("2026_05_CHI_GB", 25, 20, line=7.0)         # GB by 5: CHI +3.5 loses
        w.game("2026_05_DEN_LAC", 21, 23)                  # LAC loses by 2: +2.5 wins
        w.game("2026_05_BUF_LA", 23, 20)                   # LA by 3: BUF +3 pushes
        at = render()
        chi, lac, buf = row(at, "CHI @ GB"), row(at, "DEN @ LAC"), row(at, "BUF @ LA")
        assert (chi["Team"], chi["Handicap"], chi["Spread result"]) == ("CHI", "+3.5", "loss")
        assert (lac["Team"], lac["Handicap"], lac["Spread result"]) == ("LAC", "+2.5", "win")
        assert (buf["Team"], buf["Handicap"], buf["Spread result"]) == ("BUF", "+3", "push")
        assert chi["Frozen P(cover)"] == "60.0%" and chi["Frozen signal"] == "Yes"
        assert chi["Final score"] == "CHI 20 - 25 GB"
        assert chi["Frozen capture (Toronto)"] == "Wed Oct 7, 05:00 EDT"
        assert chi["Kickoff (Toronto)"] == "Sun Oct 11, 13:00 EDT"
        assert buf["Status"].startswith("Push")

    def test_no_line_and_timing_unverified_are_excluded_with_reasons(self, w, render):
        w.snapshot(S2, [("2026_05_PHI_JAX", None, None, False),
                        ("2026_05_CHI_GB", 3.0, 0.6, True)])
        w.game("2026_05_PHI_JAX", 30, 10)
        w.game("2026_05_CHI_GB", 20, 24, day="2026-10-07", at="04:00")   # moved before capture
        at = render()
        phi, chi = row(at, "PHI @ JAX"), row(at, "CHI @ GB")
        assert phi["Status"] == "Excluded - no line in the frozen snapshot" and phi["Team"] == ""
        assert chi["Status"] == "Excluded - kickoff timing unverified"
        assert "not before the current scheduled kickoff" in chi["Reason"]
        assert chi["Spread result"] == ""
        assert metric(at, "Excluded") == "2"

    def test_latest_snapshot_without_a_prediction_is_shown_as_is(self, w, render):
        # An earlier snapshot had a probability and a signal; the latest has
        # none. The page shows the latest - no fallback to the earlier signal.
        w.snapshot(S1, [("2026_05_CHI_GB", 3.0, 0.70, True)])
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, None, False)])
        w.game("2026_05_CHI_GB", 20, 24)
        at = render()
        r = row(at, "CHI @ GB")
        assert r["Status"] == "Excluded - no frozen probability"
        assert (r["Frozen P(cover)"], r["Frozen signal"]) == ("", "No")
        assert metric(at, "Evaluated (win/loss)", 1) == "0"            # no signal subset
        assert metric(at, "Bets settled", 1) == "0"

    def test_earliest_selection_uses_that_snapshot_only(self, w, render):
        w.snapshot(S1, [("2026_05_CHI_GB", 3.0, 0.40, False)])
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.62, True)])
        w.game("2026_05_CHI_GB", 20, 24)
        at = render()
        assert row(at, "CHI @ GB")["Frozen P(cover)"] == "62.0%"
        at.radio[0].set_value("earliest")
        at = render(at)
        r = row(at, "CHI @ GB")
        assert (r["Frozen P(cover)"], r["Frozen signal"]) == ("40.0%", "No")


# ---------------------------------------------------------------- summaries --

class TestSummaries:
    def test_all_and_signal_summaries_are_separate_and_match_the_report(self, w, render):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.8, True), ("2026_05_DEN_LAC", 3.0, 0.6, False),
                        ("2026_05_BUF_LA", 3.0, 0.3, False)])
        week5_complete(w)                      # CHI covers; DEN +3 pushes (LAC by 3); BUF loses
        at = render()
        r = report(w, season=2026, week=5)
        pm = r["probability_metrics"]
        assert metric(at, "Evaluated (win/loss)", 0) == "2"           # all
        assert metric(at, "Evaluated (win/loss)", 1) == "1"           # signal subset
        assert metric(at, "Pushes", 0) == "1" and metric(at, "Pushes", 1) == "0"
        assert metric(at, "Model Brier", 0) == f"{pm['all_valid_probabilities']['overall']['brier']:.3f}"
        assert metric(at, "Model Brier", 1) == f"{pm['signal_only']['overall']['brier']:.3f}"
        assert metric(at, "Model log loss", 0) == \
            f"{pm['all_valid_probabilities']['overall']['log_loss']:.3f}"

    def test_simulated_returns_match_the_report_and_are_labelled(self, w, render):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, True), ("2026_05_DEN_LAC", 3.5, 0.6, True)])
        w.game("2026_05_CHI_GB", 20, 24)
        w.game("2026_05_DEN_LAC", 20, 24)
        _two_games_market(w, chi=(-3.0, 105, 3.0, -125))
        at = render()
        sr = report(w, season=2026, week=5)["simulated_returns"]
        assert metric(at, "Net units", 0) == sr["archived_price"]["net_units"] == "80.00"
        assert metric(at, "Net units", 1) == sr["assumed_minus_110"]["net_units"] == "181.82"
        md = " ".join(m.value for m in at.markdown)
        assert "Simulated, not placed" in md and "Assumed -110 (scenario only)" in md
        assert "not necessarily a sportsbook offer" in md
        assert any("settled but no archived exact-line price: 1" in c.value for c in at.caption)
        assert any("Bet Journal" in i.value for i in at.info)


# ------------------------------------------------------------------- market --

class TestMarket:
    def test_matched_sample_is_shown_separately(self, w, render):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, True), ("2026_05_DEN_LAC", 3.5, 0.55, False)])
        w.game("2026_05_CHI_GB", 20, 24)
        w.game("2026_05_DEN_LAC", 20, 24)
        _two_games_market(w)                     # CHI +3 matches; DEN quoted +3, frozen +3.5
        at = render()
        mc = report(w, season=2026, week=5)["market_comparison"]
        assert mc["matched_games"] == 1
        assert any("Matched sample: **1** game(s), of 2 evaluated overall" in c.value
                   for c in at.caption)
        assert metric(at, "Ontario market Brier") == f"{mc['market']['brier']:.3f}"
        assert metric(at, "Model Brier (matched)") == f"{mc['model']['brier']:.3f}"
        assert metric(at, "Evaluated (win/loss)", 0) == "2"            # overall sample differs

    def test_no_exact_line_quote_means_unavailable(self, w, render):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.5, 0.6, True)])
        w.game("2026_05_CHI_GB", 20, 24)
        _two_games_market(w)                                           # only +3 quoted
        at = render()
        assert any("Market comparison unavailable" in i.value for i in at.info)
        assert "Ontario market Brier" not in dict(metrics(at))

    def test_us_and_manual_quotes_never_reach_the_market_metrics(self, w, render):
        # Only FanDuel US quotes the exact line in the capture, and a manual
        # FanDuel Ontario observation exists at it too: still unavailable.
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, True)])
        w.game("2026_05_CHI_GB", 20, 24)
        w.market(WED_SLOT, [ev("2026_05_CHI_GB", "2026-10-07T15:55:00Z",
                               fanduel=(-3.0, -160, 3.0, 130))])
        w.store.manual(ps._iso(WED_SLOT - timedelta(hours=1)), WED_SLOT, team="CHI",
                       handicap=3.0, price=150)
        assert list(w.store.manual_dir.glob("*.json"))                 # the manual quote exists
        at = render()
        assert any("Market comparison unavailable" in i.value for i in at.info)
        assert metric(at, "Bets settled", 0) == "0"                    # no archived price

    def test_mixed_capture_shows_ontario_only_values(self, w, render):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, True)])
        w.game("2026_05_CHI_GB", 20, 24)
        w.market(WED_SLOT, [ev("2026_05_CHI_GB", "2026-10-07T15:55:00Z",
                               betmgm_ca_on=(-3.0, -110, 3.0, -110),
                               fanduel=(-3.0, 130, 3.0, -160))])
        at = render()
        assert metric(at, "Ontario market Brier") == "0.250"           # p = 0.5 exactly
        assert metric(at, "Net units", 0) == "90.91"                   # -110, not US -160


# ------------------------------------------------------- integrity and cache --

class TestIntegrityAndCache:
    @pytest.mark.parametrize("what", ["snapshot", "capture", "schedule"])
    def test_corrupt_artifacts_stop_the_page(self, w, render, what):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, True)])
        w.game("2026_05_CHI_GB", 20, 24)
        w.write_schedule()
        if what == "snapshot":
            (w.snapshots / "20261008T000000Z-0000000000ff.json").write_text("{not json")
        elif what == "capture":
            w.captures.mkdir(parents=True, exist_ok=True)
            (w.captures / "20261007T160000Z-bad.json").write_text("[]")
        from streamlit.testing.v1 import AppTest
        import streamlit as st
        st.cache_data.clear()
        if what == "schedule":
            w.rows.clear()
            w.schedule.write_text("game_id\tseason\nx\t2026\n")
            at = AppTest.from_file(str(PAGE), default_timeout=60)
        else:
            at = AppTest.from_file(str(PAGE), default_timeout=60)
        at.run()
        assert not at.exception
        assert any("Integrity error" in e.value for e in at.error)
        assert table(at) is None and not at.metric                     # nothing partial shown

    @pytest.mark.parametrize("what", ["snapshot", "capture", "schedule"])
    def test_unreadable_artifacts_stop_the_page(self, w, render, monkeypatch, what):
        """A file that exists but can't be read (e.g. permission denied) is an
        integrity error - shown on the page, never a crash or partial view."""
        snap = w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, True)])
        w.game("2026_05_CHI_GB", 20, 24)
        _two_games_market(w)
        w.write_schedule()
        target = {"snapshot": snap, "capture": next(w.captures.glob("*.json")),
                  "schedule": w.schedule}[what].resolve()
        real_bytes, real_text = Path.read_bytes, Path.read_text

        def deny(real):
            def read(self, *a, **k):
                if self.resolve() == target:
                    raise PermissionError(13, "Permission denied", str(self))
                return real(self, *a, **k)
            return read
        monkeypatch.setattr(Path, "read_bytes", deny(real_bytes))
        monkeypatch.setattr(Path, "read_text", deny(real_text))
        with pytest.raises(fp.IntegrityError, match="can't be read"):
            fp.source_fingerprint(w.snapshots, w.schedule, w.captures)
        at = render()                                                   # no exception
        assert any("Integrity error" in e.value and "can't be read" in e.value for e in at.error)
        assert table(at) is None and not at.metric and not at.selectbox

    def test_schedule_without_game_type_is_rejected_before_controls(self, w, render):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, True)])
        w.game("2026_05_CHI_GB", 20, 24)
        for rows in w.rows.values():
            for r in rows:
                del r["game_type"]
        w.write_schedule()
        with pytest.raises(fp.IntegrityError, match="game_type"):
            fp.load(w.snapshots, w.schedule, w.captures, "latest")
        at = render()                                                   # no exception
        assert any("Integrity error" in e.value and "game_type" in e.value for e in at.error)
        assert not at.selectbox and table(at) is None and not at.metric

    def test_cache_follows_added_changed_and_deleted_files(self, w, render):
        week5_complete(w)
        at = render()                                                   # no snapshots yet
        assert table(at) is None
        path = w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, True)])     # added
        at = render(at)
        assert row(at, "CHI @ GB")["Frozen P(cover)"] == "60.0%"
        doc = json.loads(path.read_text())                              # changed (validly)
        doc["games"][0]["prob_underdog_covers"] = 0.7
        doc[ps.CHECKSUM_FIELD] = ps.payload_checksum(doc)
        path.write_text(json.dumps(doc))
        at = render(at)
        assert row(at, "CHI @ GB")["Frozen P(cover)"] == "70.0%"
        path.unlink()                                                   # deleted
        at = render(at)
        assert table(at) is None

    def test_fingerprint_changes_with_every_source_kind(self, w):
        w.game("2026_05_CHI_GB", 20, 24)
        w.write_schedule()
        f0 = fp.source_fingerprint(w.snapshots, w.schedule, w.captures)
        p = w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, True)])
        f1 = fp.source_fingerprint(w.snapshots, w.schedule, w.captures)
        _two_games_market(w)
        f2 = fp.source_fingerprint(w.snapshots, w.schedule, w.captures)
        w.schedule.write_text(w.schedule.read_text() + "\n")
        f3 = fp.source_fingerprint(w.snapshots, w.schedule, w.captures)
        p.unlink()
        f4 = fp.source_fingerprint(w.snapshots, w.schedule, w.captures)
        assert len({f0, f1, f2, f3, f4}) == 5

    def test_rendering_writes_nothing(self, w, render):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, True)])
        week5_complete(w)
        _two_games_market(w)
        w.write_schedule()

        def tree():
            return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in w.tmp.rglob("*") if p.is_file()}
        prod = sorted(p.name for p in (ROOT / "data_files").iterdir())
        before = tree()
        at = render()
        at.radio[0].set_value("earliest")
        render(at)
        assert tree() == before
        assert sorted(p.name for p in (ROOT / "data_files").iterdir()) == prod


# ------------------------------------------------------------- pure helpers --

def test_status_counts_split_signal_subset():
    games = [{"status": "evaluated", "bet_signal": True}, {"status": "evaluated", "bet_signal": False},
             {"status": "push", "bet_signal": True}, {"status": "pending", "bet_signal": True},
             {"status": "no_line", "bet_signal": False}]
    rep = {"games": games, "probability_metrics": {"all_valid_probabilities": {},
                                                   "signal_only": {}}}
    s = fp.summaries(rep)
    assert {k: s["all"][k] for k in ("selected", "evaluated", "push", "pending", "excluded")} == \
        {"selected": 5, "evaluated": 2, "push": 1, "pending": 1, "excluded": 1}
    assert {k: s["signal"][k] for k in ("selected", "evaluated", "push", "pending", "excluded")} == \
        {"selected": 3, "evaluated": 1, "push": 1, "pending": 1, "excluded": 0}
