"""'Was waiting worthwhile?' (scripts/ontario_waiting_report.py) and its section
of the Ontario Line Timing page.

Ontario captures are written by Phase 1's own capture()/manual_quote() against
a fake provider; pregame snapshots are synthetic but validated exactly like
real ones (the frozen report's World fixture); schedules are temporary.
Network access is blocked and nothing in data_files/ is read or written.
"""
import hashlib
import json
import socket
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
import requests

import ontario_line_timing as olt
import ontario_spreads as on
import pregame_snapshots as ps
from test_frozen_spread_report import World
from test_ontario_spread_report import ev, sun_events, wed_events
from test_ontario_spreads import SUN_SLOT, WED_SLOT
from test_team_features import ROOT

wr = olt.wr
UTC = timezone.utc
AS_OF = date(2026, 10, 14)
SNAP_WED = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)          # nightly, before Wed 12:00 Toronto
SNAP_SUN = datetime(2026, 10, 11, 9, 0, tzinfo=UTC)         # nightly, before Sun 09:00 Toronto
WED_DECISION = datetime(2026, 10, 7, 16, 0, tzinfo=UTC)     # Wed 12:00 EDT
SUN_DECISION = datetime(2026, 10, 11, 13, 0, tzinfo=UTC)    # Sun 09:00 EDT
PAGE = ROOT / "pages" / "7_Ontario_Line_Timing.py"
MONDAY = datetime(2026, 10, 12, 18, 0, tzinfo=UTC)


def ko(day="2026-10-11", at="13:00"):
    return ps.kickoff_utc(day, at)[0]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def build(w, **kw):
    w.write_schedule()
    kw.setdefault("as_of", AS_OF)
    kw.setdefault("season", 2026)
    kw.setdefault("week", 5)
    return wr.build_report(w.captures, w.store.manual_dir, w.snapshots, w.schedule, **kw)


def offer(rep, gid, book, team):
    return next(o for o in rep["offers"] if (o["game_id"], o["book_key"], o["team"]) ==
                (gid, book, team))


def decision(rep, gid):
    return next(d for d in rep["model_decisions"] if d["game_id"] == gid)


def scores(w, **finals):
    """Schedule rows for the capture fixture's week-5 games; finals: gid=(home, away)."""
    for gid in ("2026_05_TB_DAL", "2026_05_PHI_JAX", "2026_05_CHI_GB", "2026_05_DEN_LAC",
                "2026_05_BUF_LA"):
        h, a = finals.get(gid, (None, None))
        w.game(gid, h, a)


@pytest.fixture
def paired(w):
    """Both scheduled slots captured (the Phase 2 fixture week) and final scores
    for every game except BUF @ LA (Monday night, still pending)."""
    w.market(WED_SLOT, wed_events())
    w.market(SUN_SLOT, sun_events())
    scores(w, **{"2026_05_TB_DAL": (24, 20), "2026_05_PHI_JAX": (23, 20),
                 "2026_05_CHI_GB": (20, 13), "2026_05_DEN_LAC": (24, 20)})
    return w


# ------------------------------------------------------------------ pairing --

class TestPairing:
    def test_rows_are_exactly_the_phase2_eligible_ontario_pairs(self, paired):
        rep = build(paired)
        osr_rows = paired.store.report(include_manual=True, include_us_reference=True)["rows"]
        expected = {(r["game_id"], r["book_key"], r["comparison_team"], r["outcome"])
                    for r in osr_rows if r["group"] == "ontario_api" and r["status"] == "compared"}
        got = {(o["game_id"], o["book_key"], o["team"], o["classification"])
               for o in rep["offers"]}
        assert got == expected and len(got) == 12
        assert all(o["book_key"] != "fanduel" for o in rep["offers"])       # US reference
        # Thursday game: kicked off before the Sunday slot -> a visible gap
        assert {(m["game_id"], tuple(m["reasons"])) for m in rep["missing_pairs"]} == \
            {("2026_05_TB_DAL", ("no_sunday_observation_kicked_off",))}

    def test_missing_intended_wednesday_is_a_gap_not_a_comparison(self, w):
        w.market(SUN_SLOT, sun_events())
        scores(w)
        rep = build(w)
        assert rep["status"] == "no_eligible_pairs" and rep["offers"] == []
        assert rep["weeks"][0]["wednesday_capture"] is None
        assert rep["weeks"][0]["intended_wednesday"] == "2026-10-07_wednesday_noon"
        assert {r for m in rep["missing_pairs"] for r in m["reasons"]} == \
            {"no_wednesday_capture_for_week"}

    def test_older_wednesday_never_stands_in(self, w):
        older = datetime(2026, 9, 30, 16, 0, tzinfo=UTC)
        w.market(older, wed_events(updated="2026-09-30T15:55:00Z"))
        w.market(SUN_SLOT, sun_events())
        scores(w)
        rep = build(w)
        assert rep["offers"] == [] and rep["weeks"][0]["wednesday_capture"] is None
        assert {r for m in rep["missing_pairs"] for r in m["reasons"]} == \
            {"no_wednesday_capture_for_week"}

    def test_ad_hoc_wednesday_never_stands_in(self, w):
        w.store.capture(WED_SLOT + timedelta(hours=3), wed_events(updated="2026-10-07T18:55:00Z"),
                        slot="ad_hoc")
        w.market(SUN_SLOT, sun_events())
        scores(w)
        rep = build(w)
        assert rep["offers"] == [] and rep["weeks"][0]["wednesday_capture"] is None

    def test_manual_wednesday_never_pairs_with_an_automated_sunday(self, w):
        w.store.manual("2026-10-07T16:30:00Z", WED_SLOT + timedelta(hours=1), opponent_price=-110)
        w.market(SUN_SLOT, sun_events())
        scores(w)
        rep = build(w)
        assert rep["offers"] == []
        assert all(m["book_key"] != on.FANDUEL_ONTARIO_MANUAL.key for m in rep["missing_pairs"])

    def test_stale_quote_is_excluded_without_cross_book_substitution(self, w):
        stale = "2026-10-07T14:00:00Z"                     # 2 h before the Wednesday capture
        events = [e for e in wed_events() if e["id"] != "ev_2026_05_CHI_GB"]
        events.append(ev("2026_05_CHI_GB", WED_UPD_FRESH, betano_ca_on=(-3.5, -110, 3.5, -110)))
        stale_ev = ev("2026_05_CHI_GB", stale, betmgm_ca_on=(-3.5, -110, 3.5, -110))
        events[-1]["bookmakers"] += stale_ev["bookmakers"]
        w.market(WED_SLOT, events)
        w.market(SUN_SLOT, sun_events())
        scores(w)
        rep = build(w)
        gb = {(o["book_key"], o["team"]) for o in rep["offers"] if o["game_id"] == "2026_05_CHI_GB"}
        assert gb == {("betano_ca_on", "GB"), ("betano_ca_on", "CHI")}
        stale_rows = [m for m in rep["missing_pairs"] if m["book_key"] == "betmgm_ca_on"
                      and m["game_id"] == "2026_05_CHI_GB"]
        assert len(stale_rows) == 2 and all("wednesday_quote_stale" in m["reasons"]
                                            for m in stale_rows)

    def test_same_book_only_no_cross_book_pair(self, w):
        # DEN @ LAC: Betano on Wednesday only, BetMGM on Sunday only -> no pair at all.
        w.market(WED_SLOT, wed_events())
        w.market(SUN_SLOT, sun_events(**{"2026_05_DEN_LAC": {"betmgm_ca_on": (-6.5, -105, 6.5, -115)}}))
        scores(w)
        rep = build(w)
        assert not [o for o in rep["offers"] if o["game_id"] == "2026_05_DEN_LAC"]
        books = {m["book_key"] for m in rep["missing_pairs"] if m["game_id"] == "2026_05_DEN_LAC"}
        assert books == {"betano_ca_on", "betmgm_ca_on"}

    def test_moved_kickoff_before_a_quote_is_timing_unverified(self, paired):
        # The current schedule now puts CHI @ GB before the Sunday capture.
        paired.rows["2026_05_CHI_GB"][0].update(gameday="2026-10-11", gametime="08:00")
        rep = build(paired)
        assert not [o for o in rep["offers"] if o["game_id"] == "2026_05_CHI_GB"]
        assert {tuple(m["reasons"]) for m in rep["missing_pairs"]
                if m["game_id"] == "2026_05_CHI_GB"} == {(wr.TIMING_UNVERIFIED,)}


WED_UPD_FRESH = "2026-10-07T15:55:00Z"


# ---------------------------------------------------------- current kickoff --

CHI = "2026_05_CHI_GB"


def _no_time(w):
    w.rows[CHI][0]["gametime"] = None


def _no_date(w):
    w.rows[CHI][0]["gameday"] = None


def _not_in_schedule(w):
    del w.rows[CHI]


def _identical_duplicate(w):
    w.rows[CHI].append(dict(w.rows[CHI][0]))


def _conflicting_duplicate(w):
    w.rows[CHI].append({**w.rows[CHI][0], "gametime": "16:25"})


KICKOFF_CASES = [
    ("missing kickoff time", _no_time, wr.KICKOFF_UNAVAILABLE_PREFIX + ps.MISSING_KICKOFF),
    ("missing kickoff date", _no_date, wr.KICKOFF_UNAVAILABLE_PREFIX + ps.MISSING_KICKOFF),
    ("game not in schedule", _not_in_schedule, wr.KICKOFF_NOT_IN_SCHEDULE),
    ("identical duplicate rows", _identical_duplicate, wr.KICKOFF_DUPLICATED),
    ("conflicting duplicate rows", _conflicting_duplicate, wr.KICKOFF_DUPLICATED),
]


class TestCurrentKickoff:
    """Pairs need ONE usable current scheduled kickoff; otherwise fail closed."""

    @pytest.mark.parametrize("label,breaks,reason", KICKOFF_CASES, ids=[c[0] for c in KICKOFF_CASES])
    def test_unverifiable_kickoff_is_a_gap_not_a_pair(self, paired, label, breaks, reason):
        baseline = build(paired)
        breaks(paired)
        rep = build(paired)
        assert not [o for o in rep["offers"] if o["game_id"] == CHI]
        gaps = [m for m in rep["missing_pairs"] if m["game_id"] == CHI]
        assert len(gaps) == 4                                  # 2 books x 2 sides, all kept
        assert all(m["reasons"] == [reason] for m in gaps)
        assert {m["status"] for m in gaps} == {wr.KICKOFF_UNVERIFIED_STATUS}
        # The other games' offers are untouched ...
        assert [o for o in rep["offers"]] == [o for o in baseline["offers"] if o["game_id"] != CHI]
        # ... and CHI contributes to no rollup (settled CHI rows were 4 of the 10).
        d = rep["rollups"]["denominators"]
        assert (d["games_with_paired_captures"], d["game_book_pairs"], d["team_side_comparisons"],
                d["settled_comparisons"]) == (3, 4, 8, 6)
        oc = rep["rollups"]["offer_classifications"]
        assert sum(v for k, v in oc.items() if k != "denominator") == 8
        rp = rep["rollups"]["realized_profit_difference"]
        assert rp["sunday_higher"] + rp["wednesday_higher"] + rp["equal"] == 6
        assert (rp["games_represented"], rp["game_book_pairs_represented"]) == (2, 3)

    @pytest.mark.parametrize("label,breaks,reason", KICKOFF_CASES, ids=[c[0] for c in KICKOFF_CASES])
    def test_model_decision_needs_the_same_unique_kickoff(self, paired, label, breaks, reason):
        paired.snapshot(SNAP_WED, [(CHI, 3.0, 0.58, True)])
        paired.snapshot(SNAP_SUN, [(CHI, 3.0, 0.60, True)])
        breaks(paired)
        rep = build(paired)
        if label == "game not in schedule":                   # not a scheduled game: no row
            assert CHI not in {d["game_id"] for d in rep["model_decisions"]}
            return
        d = decision(rep, CHI)
        assert d["wednesday"]["status"] == d["sunday"]["status"] == wr.KICKOFF_UNKNOWN
        assert d["classification"] == wr.UNAVAILABLE and d["kickoff_utc"] is None

    def test_unique_valid_kickoff_keeps_existing_results(self, paired):
        rep = build(paired)
        assert len([o for o in rep["offers"] if o["game_id"] == CHI]) == 4
        o = offer(rep, CHI, "betmgm_ca_on", "GB")
        assert (o["wed_profit"], o["sun_profit"], o["profit_difference"]) == \
            ("90.91", "86.96", "-3.95")
        assert rep["rollups"]["denominators"]["team_side_comparisons"] == 12

    @pytest.mark.parametrize("day,at", [("2026-10-11", "09:00"),      # = Sunday quote (13:00Z)
                                        ("2026-10-11", "08:30"),      # before Sunday's quote
                                        ("2026-10-07", "12:00"),      # = Wednesday quote (16:00Z)
                                        ("2026-10-07", "11:00")])     # before Wednesday's quote
    def test_kickoff_at_or_before_either_quote_stays_excluded(self, paired, day, at):
        paired.rows[CHI][0].update(gameday=day, gametime=at)
        rep = build(paired)
        assert not [o for o in rep["offers"] if o["game_id"] == CHI]
        assert {tuple(m["reasons"]) for m in rep["missing_pairs"] if m["game_id"] == CHI} == \
            {(wr.TIMING_UNVERIFIED,)}
        assert rep["rollups"]["denominators"]["team_side_comparisons"] == 8

    @pytest.mark.parametrize("label,breaks,reason", KICKOFF_CASES, ids=[c[0] for c in KICKOFF_CASES])
    def test_page_shows_the_kickoff_gap(self, paired, page, label, breaks, reason):
        breaks(paired)
        at = page(MONDAY)
        offers = frame(at, "Offer comparison")
        assert "CHI @ GB" not in set(offers["Matchup"])
        gaps = frame(at, "Why no pair").set_index("Why no pair")
        assert gaps.loc["Current kickoff unavailable", "Team-sides"] == 4
        assert gaps.loc["Current kickoff unavailable", "Detail"] == olt.waiting_reason_text(reason)
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Team-side comparisons"] == "8"

    def test_page_shows_the_moved_kickoff_gap(self, paired, page):
        paired.rows[CHI][0].update(gameday="2026-10-11", gametime="09:00")
        at = page(MONDAY)
        gaps = frame(at, "Why no pair").set_index("Why no pair")
        assert gaps.loc["Quote not before current kickoff", "Team-sides"] == 4


# --------------------------------------------------------------- settlement --

class TestSettlement:
    def test_whole_number_pushes_and_asymmetric_prices(self, paired):
        rep = build(paired)
        # PHI @ JAX final 20-23 (JAX by 3). BetMGM: JAX -2.5 Wed, -3 Sun.
        o = offer(rep, "2026_05_PHI_JAX", "betmgm_ca_on", "JAX")
        assert (o["wed_result"], o["wed_profit"], o["sun_result"], o["sun_profit"],
                o["profit_difference"]) == ("win", "90.91", "push", "0.00", "-90.91")
        o = offer(rep, "2026_05_PHI_JAX", "betmgm_ca_on", "PHI")       # +2.5 then +3
        assert (o["wed_result"], o["sun_result"], o["profit_difference"]) == \
            ("loss", "push", "100.00")
        # DEN @ LAC final 20-24: DEN +6.5 at -110 Wed, -115 Sun (asymmetric)
        o = offer(rep, "2026_05_DEN_LAC", "betano_ca_on", "DEN")
        assert (o["wed_profit"], o["sun_profit"], o["profit_difference"]) == \
            ("90.91", "86.96", "-3.95") and o["classification"] == "wednesday_dominates"
        o = offer(rep, "2026_05_DEN_LAC", "betano_ca_on", "LAC")       # -6.5 at -110 then -105
        assert (o["wed_result"], o["sun_result"], o["profit_difference"]) == \
            ("loss", "loss", "0.00") and o["classification"] == "sunday_dominates"

    def test_seven_point_push_both_signs(self, w):
        w.market(WED_SLOT, wed_events(**{"2026_05_DEN_LAC": {"betano_ca_on": (-7.0, -110, 7.0, -110)}}))
        w.market(SUN_SLOT, sun_events(**{"2026_05_DEN_LAC": {"betano_ca_on": (-6.5, -105, 6.5, -115)}}))
        scores(w, **{"2026_05_DEN_LAC": (27, 20)})                     # LAC by exactly 7
        rep = build(w)
        lac = offer(rep, "2026_05_DEN_LAC", "betano_ca_on", "LAC")      # -7 then -6.5
        den = offer(rep, "2026_05_DEN_LAC", "betano_ca_on", "DEN")      # +7 then +6.5
        assert (lac["wed_handicap"], lac["wed_result"], lac["wed_profit"]) == (-7.0, "push", "0.00")
        assert (lac["sun_result"], lac["sun_profit"], lac["profit_difference"]) == \
            ("win", "95.24", "95.24")
        assert (den["wed_handicap"], den["wed_result"], den["sun_result"], den["sun_profit"],
                den["profit_difference"]) == (7.0, "push", "loss", "-100.00", "-100.00")
        assert (lac["classification"], den["classification"]) == \
            ("sunday_dominates", "wednesday_dominates")

    def test_plus_and_minus_three(self, w):
        w.market(WED_SLOT, wed_events(**{"2026_05_CHI_GB": {"betmgm_ca_on": (-3.0, -120, 3.0, 100)}}))
        w.market(SUN_SLOT, sun_events(**{"2026_05_CHI_GB": {"betmgm_ca_on": (-3.5, 100, 3.5, -120)}}))
        scores(w, **{"2026_05_CHI_GB": (16, 13)})                      # GB by exactly 3
        rep = build(w)
        gb = offer(rep, "2026_05_CHI_GB", "betmgm_ca_on", "GB")         # -3 -120 then -3.5 +100
        chi = offer(rep, "2026_05_CHI_GB", "betmgm_ca_on", "CHI")       # +3 +100 then +3.5 -120
        assert (gb["wed_result"], gb["sun_result"], gb["profit_difference"]) == \
            ("push", "loss", "-100.00")
        assert (chi["wed_result"], chi["sun_result"], chi["sun_profit"],
                chi["profit_difference"]) == ("push", "win", "83.33", "83.33")
        assert gb["classification"] == chi["classification"] == "trade_off"

    @pytest.mark.parametrize("over,status", [({}, wr.PENDING),
                                             ({"result": 99}, wr.UNRESOLVED)])
    def test_pending_and_unresolved_are_not_settled(self, w, over, status):
        w.market(WED_SLOT, wed_events())
        w.market(SUN_SLOT, sun_events())
        for gid in ("2026_05_TB_DAL", "2026_05_PHI_JAX", "2026_05_DEN_LAC", "2026_05_BUF_LA"):
            w.game(gid)
        w.game("2026_05_CHI_GB", *((20, 13) if over else (None, None)), **over)
        rep = build(w)
        o = offer(rep, "2026_05_CHI_GB", "betmgm_ca_on", "GB")
        assert o["result_status"] == status and o["result_reason"]
        assert (o["wed_result"], o["wed_profit"], o["profit_difference"]) == (None, None, None)
        assert rep["rollups"]["denominators"]["settled_comparisons"] == 0

    def test_final_is_judged_at_the_as_of_date(self, paired):
        rep = build(paired, as_of=date(2026, 10, 11))         # Sunday's games not yet before it
        assert offer(rep, "2026_05_CHI_GB", "betmgm_ca_on", "GB")["result_status"] == wr.PENDING


# ---------------------------------------------------------- model decisions --

def model_world(w):
    """Sunday capture (anchors the week) + snapshots covering every class."""
    w.market(SUN_SLOT, sun_events())
    scores(w)
    for gid in ("2026_05_NYG_WAS", "2026_05_LV_NE", "2026_05_MIN_NO", "2026_05_HOU_TEN"):
        w.game(gid)
    x = ko()
    w.snapshot(SNAP_WED, [
        ("2026_05_TB_DAL", 3.0, 0.50, False),
        ("2026_05_CHI_GB", 3.0, 0.58, True),
        ("2026_05_DEN_LAC", 3.5, 0.57, True),
        ("2026_05_BUF_LA", 3.0, 0.50, False),
        ("2026_05_PHI_JAX", 2.5, 0.56, True),
        ("2026_05_NYG_WAS", 3.5, 0.45, False, x),
        ("2026_05_LV_NE", 3.5, 0.60, True, x),
        ("2026_05_MIN_NO", 2.5, 0.55, True, x),
    ])
    # Saturday: LV still signalled ...
    w.snapshot(SNAP_SUN - timedelta(days=1), [("2026_05_LV_NE", 3.5, 0.61, True, x)])
    # ... but the latest snapshot before Sunday's decision has no line for it.
    w.snapshot(SNAP_SUN, [
        ("2026_05_CHI_GB", 3.0, 0.60, True),
        ("2026_05_DEN_LAC", 3.5, 0.52, False),
        ("2026_05_BUF_LA", 3.5, 0.58, True),
        ("2026_05_PHI_JAX", -1.5, 0.57, True),
        ("2026_05_NYG_WAS", 3.5, 0.44, False, x),
        ("2026_05_LV_NE", None, None, False, x),
        ("2026_05_MIN_NO", 2.5, None, False, x),
        ("2026_05_HOU_TEN", 7.0, 0.50, False, x),
    ])
    return w


class TestModelDecisions:
    def test_every_classification(self, w):
        rep = build(model_world(w))
        got = {d["game_id"]: d["classification"] for d in rep["model_decisions"]}
        assert got == {
            "2026_05_CHI_GB": wr.RETAINED, "2026_05_DEN_LAC": wr.WITHDRAWN,
            "2026_05_BUF_LA": wr.ADDED, "2026_05_PHI_JAX": wr.SIDE_CHANGED,
            "2026_05_NYG_WAS": wr.NEITHER, "2026_05_LV_NE": wr.UNAVAILABLE,
            "2026_05_MIN_NO": wr.UNAVAILABLE, "2026_05_HOU_TEN": wr.UNAVAILABLE,
            "2026_05_TB_DAL": wr.UNAVAILABLE}
        mc = rep["rollups"]["model_classifications"]
        assert (mc[wr.RETAINED], mc[wr.WITHDRAWN], mc[wr.ADDED], mc[wr.SIDE_CHANGED],
                mc[wr.NEITHER], mc[wr.UNAVAILABLE]) == (1, 1, 1, 1, 1, 4)

    def test_missing_inputs_are_unavailable_not_no_signal(self, w):
        rep = build(model_world(w))
        lv, mn = decision(rep, "2026_05_LV_NE"), decision(rep, "2026_05_MIN_NO")
        assert (lv["sunday"]["status"], lv["sunday"]["signal"]) == (ps.NO_LINE, None)
        assert (mn["sunday"]["status"], mn["sunday"]["signal"]) == (ps.NO_PROBABILITY, None)
        assert "not a no-bet decision" in lv["sunday"]["reason"]
        hou = decision(rep, "2026_05_HOU_TEN")
        assert hou["wednesday"]["status"] == wr.NO_SNAPSHOT and hou["sunday"]["status"] == wr.AVAILABLE
        tb = decision(rep, "2026_05_TB_DAL")
        assert tb["wednesday"]["status"] == wr.AVAILABLE and tb["sunday"]["status"] == wr.KICKED_OFF

    def test_latest_snapshot_is_used_without_signal_fallback(self, w):
        rep = build(model_world(w))
        lv = decision(rep, "2026_05_LV_NE")["sunday"]
        assert lv["captured_at"] == ps._iso(SNAP_SUN)           # not Saturday's signalled one
        assert lv["status"] == ps.NO_LINE

    def test_snapshot_age_and_provenance(self, w):
        rep = build(model_world(w))
        d = decision(rep, "2026_05_CHI_GB")
        assert d["wednesday"]["decision_time"] == ps._iso(WED_DECISION)
        assert d["sunday"]["decision_time"] == ps._iso(SUN_DECISION)
        assert (d["wednesday"]["age_minutes"], d["sunday"]["age_minutes"]) == (420, 240)
        assert d["wednesday"]["team"] == "CHI" and d["wednesday"]["handicap"] == 3.0

    def test_probability_comparability(self, w):
        rep = build(model_world(w))
        chi = decision(rep, "2026_05_CHI_GB")["probability"]
        assert chi["comparable"] and chi["change"] == pytest.approx(0.02)
        buf = decision(rep, "2026_05_BUF_LA")["probability"]          # +3 then +3.5
        assert not buf["comparable"] and buf["change"] is None and "handicap" in buf["note"]
        phi = decision(rep, "2026_05_PHI_JAX")["probability"]         # PHI then JAX
        assert not phi["comparable"] and phi["change"] is None and "team" in phi["note"]
        # Displayed the same way: no percentage-point change for different propositions.
        rows = olt.waiting_decision_rows(rep).set_index("Matchup")
        assert rows.loc["BUF @ LA", "Probability change"].startswith("Not comparable")
        assert rows.loc["CHI @ GB", "Probability change"].startswith("+2.0 pp")

    @pytest.mark.parametrize("offset,picked", [(timedelta(0), "boundary"),
                                               (timedelta(seconds=1), "earlier")])
    def test_decision_time_boundary(self, w, offset, picked):
        w.market(SUN_SLOT, sun_events())
        scores(w)
        w.snapshot(SNAP_WED, [("2026_05_CHI_GB", 3.0, 0.50, False)])
        w.snapshot(WED_DECISION + offset, [("2026_05_CHI_GB", 3.0, 0.70, True)])
        w.snapshot(SUN_DECISION + offset, [("2026_05_CHI_GB", 3.0, 0.40, False)])
        rep = build(w)
        d = decision(rep, "2026_05_CHI_GB")
        if picked == "boundary":            # captured exactly at the decision time: used
            assert (d["wednesday"]["probability"], d["sunday"]["probability"]) == (0.70, 0.40)
            assert d["wednesday"]["age_minutes"] == 0
        else:                               # one second later: not yet available
            assert (d["wednesday"]["probability"], d["sunday"]["probability"]) == (0.50, 0.70)

    def test_decision_time_must_be_before_kickoff(self, w):
        w.market(SUN_SLOT, sun_events())
        scores(w)
        w.game("2026_05_NYG_WAS", day="2026-10-11", at="09:00")      # = Sun 13:00 UTC
        w.snapshot(SNAP_SUN, [("2026_05_NYG_WAS", 3.0, 0.6, True, ko(at="09:00"))])
        rep = build(w)
        d = decision(rep, "2026_05_NYG_WAS")
        assert d["sunday"]["status"] == wr.KICKED_OFF and d["classification"] == wr.UNAVAILABLE

    def test_no_snapshot_directory_means_no_decisions(self, w):
        w.market(SUN_SLOT, sun_events())
        scores(w)
        w.snapshots.rmdir()
        rep = build(w)
        assert {d["wednesday"]["status"] for d in rep["model_decisions"]} == {wr.NO_SNAPSHOT}
        assert rep["rollups"]["denominators"]["games_with_model_decisions_at_both_times"] == 0


def at_clock(w, now):
    """The week-5 analysis at an exact clock (results AND decision timing)."""
    w.write_schedule()
    src = wr.load_sources(w.captures, w.store.manual_dir, w.snapshots, w.schedule)
    return wr.analyze(src, now=now, season=2026, week=5)


class TestDecisionTimeNotReached:
    """A decision time later than the clock has no decision yet: it is never
    filled with the latest snapshot that happens to exist."""

    def test_future_sunday_decision_is_pending(self, w):
        rep = at_clock(model_world(w), datetime(2026, 10, 9, 16, 0, tzinfo=UTC))   # Friday
        chi = decision(rep, "2026_05_CHI_GB")
        assert chi["wednesday"]["status"] == wr.AVAILABLE                 # Wed decision passed
        assert chi["sunday"]["status"] == wr.DECISION_PENDING
        assert (chi["sunday"]["captured_at"], chi["sunday"]["signal"], chi["sunday"]["team"]) == \
            (None, None, None)
        assert chi["classification"] == wr.PENDING
        assert not chi["probability"]["comparable"]
        mc = rep["rollups"]["model_classifications"]
        # 7 pending; HOU (no Wednesday snapshot) and TB (kicked off before Sunday's
        # decision time) are permanently unavailable, whatever the clock.
        assert (mc[wr.PENDING], mc[wr.UNAVAILABLE]) == (7, 2)
        assert sum(mc[c] for c in (wr.RETAINED, wr.WITHDRAWN, wr.ADDED, wr.SIDE_CHANGED,
                                   wr.NEITHER)) == 0
        assert rep["rollups"]["denominators"]["games_with_model_decisions_at_both_times"] == 0
        assert decision(rep, "2026_05_TB_DAL")["sunday"]["status"] == wr.KICKED_OFF
        assert decision(rep, "2026_05_HOU_TEN")["classification"] == wr.UNAVAILABLE

    def test_future_wednesday_decision_is_pending(self, w):
        rep = at_clock(model_world(w), WED_DECISION - timedelta(hours=1))
        chi = decision(rep, "2026_05_CHI_GB")
        assert chi["wednesday"]["status"] == chi["sunday"]["status"] == wr.DECISION_PENDING
        assert chi["wednesday"]["captured_at"] is None and chi["classification"] == wr.PENDING

    @pytest.mark.parametrize("offset,status", [(timedelta(0), "available"),
                                               (timedelta(seconds=-1), "pending")])
    def test_decision_time_reached_boundary(self, w, offset, status):
        rep = at_clock(model_world(w), SUN_DECISION + offset)
        chi = decision(rep, "2026_05_CHI_GB")
        if status == "available":           # the clock AT the decision time: decided
            assert chi["sunday"]["status"] == wr.AVAILABLE and chi["classification"] == wr.RETAINED
        else:
            assert chi["sunday"]["status"] == wr.DECISION_PENDING
            assert chi["classification"] == wr.PENDING

    def test_page_shows_pending_not_decided(self, w, page):
        model_world(w)
        at = page(datetime(2026, 10, 9, 18, 0, tzinfo=UTC))
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Games with model decisions at both times"] == "0 of 9"
        assert metrics[olt.DECISION_LABELS[wr.PENDING]] == "7"
        assert metrics["Signal retained"] == "0"
        dec = frame(at, "Sunday vs Wednesday").set_index("Matchup")
        assert dec.loc["CHI @ GB", "Sun decision"].startswith("Not reached yet")
        assert dec.loc["CHI @ GB", "Sunday vs Wednesday"] == olt.DECISION_LABELS[wr.PENDING]

    def test_one_second_after_the_decision_time(self, w):
        rep = at_clock(model_world(w), SUN_DECISION + timedelta(seconds=1))
        chi = decision(rep, "2026_05_CHI_GB")
        assert chi["sunday"]["status"] == wr.AVAILABLE and chi["classification"] == wr.RETAINED
        assert chi["sunday"]["captured_at"] == ps._iso(SNAP_SUN)

    def test_snapshot_exactly_at_the_cutoff_with_the_clock_at_the_decision_time(self, w):
        w.market(SUN_SLOT, sun_events())
        scores(w)
        w.snapshot(SNAP_WED, [("2026_05_CHI_GB", 3.0, 0.50, False)])
        w.snapshot(SUN_DECISION, [("2026_05_CHI_GB", 3.0, 0.70, True)])           # at the cutoff
        w.snapshot(SUN_DECISION + timedelta(seconds=1), [("2026_05_CHI_GB", 3.0, 0.90, True)])
        rep = at_clock(w, SUN_DECISION)
        sun = decision(rep, "2026_05_CHI_GB")["sunday"]
        assert (sun["status"], sun["probability"], sun["age_minutes"]) == (wr.AVAILABLE, 0.70, 0)
        assert decision(rep, "2026_05_CHI_GB")["classification"] == wr.ADDED

    def test_kickoff_at_the_decision_time_is_unavailable_before_it_arrives(self, w):
        w.market(SUN_SLOT, sun_events())
        scores(w)
        w.game("2026_05_NYG_WAS", day="2026-10-11", at="09:00")      # = Sun 13:00 UTC
        w.snapshot(SNAP_WED, [("2026_05_NYG_WAS", 3.0, 0.6, True, ko(at="09:00"))])
        rep = at_clock(w, datetime(2026, 10, 9, 16, 0, tzinfo=UTC))   # Friday: not reached
        d = decision(rep, "2026_05_NYG_WAS")
        assert d["sunday"]["status"] == wr.KICKED_OFF                  # never pending
        assert d["classification"] == wr.UNAVAILABLE

    def test_no_signal_fallback_once_the_decision_time_arrives(self, w):
        rep = at_clock(model_world(w), SUN_DECISION)
        lv = decision(rep, "2026_05_LV_NE")["sunday"]                  # Saturday's had a signal
        assert (lv["status"], lv["captured_at"]) == (ps.NO_LINE, ps._iso(SNAP_SUN))

    def test_same_session_crosses_the_decision_time_without_file_changes(self, w, page):
        model_world(w)
        at = page(SUN_DECISION - timedelta(seconds=1))
        dec = frame(at, "Sunday vs Wednesday").set_index("Matchup")
        assert dec.loc["CHI @ GB", "Sunday vs Wednesday"] == olt.DECISION_LABELS[wr.PENDING]
        at = page(SUN_DECISION, at)                                     # same session, rerun
        dec = frame(at, "Sunday vs Wednesday").set_index("Matchup")
        assert dec.loc["CHI @ GB", "Sunday vs Wednesday"] == "Signal retained"
        metrics = {m.label: m.value for m in at.metric}
        assert metrics[olt.DECISION_LABELS[wr.PENDING]] == "0"


SEL = {"captured_at": "2026-10-28T09:00:00Z", "run_id": "r", "snapshot_file": "f.json",
       "code_revision": "c", "prediction_status": ps.PREDICTED, "underdog_team": "CHI",
       "spread_line": 3.0, "prob_underdog_covers": 0.6, "bet_signal": True}


@pytest.mark.parametrize("local,expected", [
    # Week across the end of DST (Sun Nov 1, 2026 02:00): Wednesday is EDT, Sunday EST.
    (("wednesday", datetime(2026, 10, 28, 11, 59, 59)), wr.DECISION_PENDING),
    (("wednesday", datetime(2026, 10, 28, 12, 0, 0)), wr.AVAILABLE),
    (("sunday", datetime(2026, 11, 1, 8, 59, 59)), wr.DECISION_PENDING),
    (("sunday", datetime(2026, 11, 1, 9, 0, 0)), wr.AVAILABLE),
])
def test_decision_times_follow_toronto_dst(local, expected):
    wed_at, sun_at = wr.decision_times({"intended_wednesday": "2026-10-28_wednesday_noon",
                                        "intended_sunday": "2026-11-01_sunday_morning"})
    assert (wed_at, sun_at) == (datetime(2026, 10, 28, 16, 0, tzinfo=UTC),    # 12:00 EDT
                                datetime(2026, 11, 1, 14, 0, tzinfo=UTC))      # 09:00 EST
    day, clock = local
    now = clock.replace(tzinfo=on.TORONTO)            # a Toronto-aware clock
    at = wed_at if day == "wednesday" else sun_at
    d = wr.decision(SEL, at, datetime(2026, 11, 1, 18, 0, tzinfo=UTC), True, now=now)
    assert d["status"] == expected
    # 09:00 Toronto on Nov 1 is 14:00 UTC; 13:30 UTC (09:30 if EDT were wrongly kept) is
    # still before it.
    if day == "sunday":
        early = wr.decision(SEL, at, datetime(2026, 11, 1, 18, 0, tzinfo=UTC), True,
                            now=datetime(2026, 11, 1, 13, 30, tzinfo=UTC))
        assert early["status"] == wr.DECISION_PENDING


# ------------------------------------------------------------------ rollups --

class TestRollups:
    def test_separate_denominators(self, paired):
        paired.snapshot(SNAP_WED, [("2026_05_CHI_GB", 3.0, 0.58, True)])
        paired.snapshot(SNAP_SUN, [("2026_05_CHI_GB", 3.0, 0.6, True),
                                   ("2026_05_DEN_LAC", 3.5, 0.5, False)])
        rep = build(paired)
        d = rep["rollups"]["denominators"]
        assert d == {"games_with_paired_captures": 4, "game_book_pairs": 6,
                     "team_side_comparisons": 12, "games_in_model_table": 5,
                     "games_with_model_decisions_at_both_times": 1,
                     "settled_comparisons": 10}                # BUF @ LA pending (2 sides)
        oc = rep["rollups"]["offer_classifications"]
        assert sum(oc[k] for k in ("unchanged", "equivalent", "sunday_dominates",
                                   "wednesday_dominates", "trade_off")) == 12
        rp = rep["rollups"]["realized_profit_difference"]
        assert rp["sunday_higher"] + rp["wednesday_higher"] + rp["equal"] == 10
        assert (rp["games_represented"], rp["game_book_pairs_represented"]) == (3, 5)
        assert rep["rollups"]["missing_pairs"]["team_sides_without_eligible_pair"] == 2

    def test_profit_is_never_summed_across_rows(self, paired):
        rep = build(paired)
        rp = rep["rollups"]["realized_profit_difference"]
        assert set(rp) == {"denominator", "sunday_higher", "wednesday_higher", "equal",
                           "games_represented", "game_book_pairs_represented"}
        assert "not independent" in rp["denominator"]

    def test_injury_and_hindsight_statements(self, paired):
        rep = build(paired)
        assert rep["rules"]["injuries"] == (
            "Injury information was not archived at both decision times; this analysis "
            "cannot determine whether injuries caused a change.")
        assert "hindsight" in rep["rules"]["hindsight"]


# ---------------------------------------------------------------- integrity --

def tree(path: Path) -> dict:
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob("*")) if p.is_file()}


class TestIntegrity:
    def test_corrupt_snapshot_stops_the_analysis(self, paired):
        (paired.snapshots / "20261007T000000Z-000000000000.json").write_text("{", encoding="utf-8")
        with pytest.raises(wr.ReportError, match="snapshot"):
            build(paired)
        with pytest.raises(olt.IntegrityError):
            olt.waiting_inputs(paired.snapshots, paired.schedule)

    def test_tampered_snapshot_stops_the_analysis(self, paired):
        path = paired.snapshot(SNAP_WED, [("2026_05_CHI_GB", 3.0, 0.58, True)])
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["games"][0]["prob_underdog_covers"] = 0.99
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(wr.ReportError, match="payload_sha256"):
            build(paired)

    def test_corrupt_capture_stops_the_analysis(self, paired):
        path = next(paired.captures.glob("*.json"))
        path.write_text(path.read_text(encoding="utf-8")[:-40], encoding="utf-8")   # truncated
        with pytest.raises(wr.ReportError, match="Ontario capture"):
            build(paired)

    def test_schedule_without_result_columns_stops_the_analysis(self, paired):
        paired.write_schedule()
        import pandas as pd
        df = pd.read_csv(paired.schedule, sep="\t").drop(columns=["result"])
        df.to_csv(paired.schedule, sep="\t", index=False)
        with pytest.raises(olt.IntegrityError, match="result"):
            olt.waiting_inputs(paired.snapshots, paired.schedule)

    def test_unreadable_files_are_integrity_errors(self, paired, monkeypatch):
        paired.write_schedule()
        snap = paired.snapshot(SNAP_WED, [("2026_05_CHI_GB", 3.0, 0.58, True)])
        cap = next(paired.captures.glob("*.json"))
        real = Path.read_bytes

        def deny(self):
            if self.name in (snap.name, cap.name):
                raise PermissionError(13, "Permission denied", str(self))
            return real(self)
        monkeypatch.setattr(Path, "read_bytes", deny)
        with pytest.raises(olt.IntegrityError, match="can't be read"):
            olt.waiting_fingerprint(paired.snapshots, paired.schedule)
        with pytest.raises(olt.IntegrityError, match="can't be read"):
            olt.source_fingerprint(paired.captures, paired.store.manual_dir)

    def test_fingerprint_changes_on_add_change_delete(self, paired):
        paired.write_schedule()
        f0 = olt.waiting_fingerprint(paired.snapshots, paired.schedule)
        path = paired.snapshot(SNAP_WED, [("2026_05_CHI_GB", 3.0, 0.58, True)])
        f1 = olt.waiting_fingerprint(paired.snapshots, paired.schedule)
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        f2 = olt.waiting_fingerprint(paired.snapshots, paired.schedule)
        path.unlink()
        f3 = olt.waiting_fingerprint(paired.snapshots, paired.schedule)
        paired.schedule.write_bytes(paired.schedule.read_bytes() + b"\n")
        f4 = olt.waiting_fingerprint(paired.snapshots, paired.schedule)
        assert len({f0, f1, f2, f4}) == 4 and f3 == f0

    def test_analysis_writes_nothing(self, paired, tmp_path):
        paired.write_schedule()
        before = tree(tmp_path)
        build(paired)
        inputs = olt.waiting_inputs(paired.snapshots, paired.schedule)
        olt.waiting_analysis(paired.store.report(include_manual=True, include_us_reference=True),
                             inputs, 2026, 5, MONDAY)
        assert tree(tmp_path) == before

    def test_output_is_refused_inside_source_data(self, paired, tmp_path):
        rep = build(paired)
        for bad in (ROOT / "data_files" / "x", paired.snapshots / "out", paired.captures):
            with pytest.raises(wr.ReportError, match="refusing"):
                wr.write_report(rep, bad, (paired.captures, paired.snapshots))
        json_path, csv_path = wr.write_report(rep, tmp_path / "out",
                                              (paired.captures, paired.snapshots))
        assert json.loads(json_path.read_text(encoding="utf-8"))["kind"] == wr.REPORT_KIND
        assert csv_path.read_text(encoding="utf-8").count("\n") == 13       # header + 12 rows

    def test_report_is_deterministic_for_the_same_inputs(self, paired):
        assert build(paired) == build(paired)


# --------------------------------------------------------------------- page --

@pytest.fixture
def page(w, monkeypatch):
    """Render the Ontario Line Timing page on the World's fixtures."""
    from streamlit.testing.v1 import AppTest
    import streamlit as st

    monkeypatch.setattr(on, "CAPTURE_DIR", w.captures)
    monkeypatch.setattr(on, "MANUAL_DIR", w.store.manual_dir)
    monkeypatch.setattr(on, "SCHEDULE_PATH", w.schedule)
    monkeypatch.setattr(wr, "SNAPSHOT_DIR", w.snapshots)
    monkeypatch.setattr(wr, "SCHEDULE_PATH", w.schedule)

    def blocked(*a, **k):
        raise AssertionError("network call attempted while rendering the page")

    def run(now, at=None):
        w.write_schedule()
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


def frame(at, column):
    return next((d.value for d in at.dataframe if column in d.value.columns), None)


class TestPage:
    def test_section_shows_pairs_decisions_and_sample_sizes(self, paired, page):
        model_world_snapshots(paired)
        at = page(datetime(2026, 10, 14, 18, 0, tzinfo=UTC))
        assert "Was waiting worthwhile?" in [h.value for h in at.subheader]
        offers = frame(at, "Offer comparison")
        assert len(offers) == 12
        jax = offers[(offers["Matchup"] == "PHI @ JAX") & (offers["Team"] == "JAX")
                     & (offers["Sportsbook"] == "BetMGM (CA - ON)")].iloc[0]
        assert (jax["Wednesday spread"], jax["Sunday spread"], jax["Wednesday result"],
                jax["Sunday result"], jax["Sunday − Wednesday profit"]) == \
            ("-2.5", "-3", "win", "push", "-90.91")
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Team-side comparisons"] == "12"
        assert metrics["Game/sportsbook pairs"] == "6"
        assert metrics["Games with paired captures"] == "4"
        assert metrics["Settled comparisons"] == "10"
        assert metrics["Games with model decisions at both times"] == "1 of 5"
        dec = frame(at, "Sunday vs Wednesday")
        assert dec.set_index("Matchup").loc["CHI @ GB", "Sunday vs Wednesday"] == "Signal retained"
        assert any(wr.INJURY_NOTE == i.value for i in at.info)
        assert any("hindsight" in x.value for x in at.warning)

    def test_missing_wednesday_shows_the_gap(self, w, page):
        w.market(SUN_SLOT, sun_events())
        scores(w)
        at = page(MONDAY)
        assert frame(at, "Offer comparison") is None
        assert any("No eligible Wednesday/Sunday pair" in i.value
                   and "2026-10-07_wednesday_noon" in i.value for i in at.info)
        gaps = frame(at, "Why no pair")
        assert gaps["Why no pair"].tolist() == ["Wednesday slot missed"]
        assert gaps["Detail"].tolist() == ["No usable scheduled Wednesday capture for this week"]

    def test_manual_and_us_views_do_not_show_the_analysis(self, paired, page):
        at = page(MONDAY)
        at.segmented_control[0].set_value("fanduel_ontario_manual")
        at = page(MONDAY, at)
        assert frame(at, "Offer comparison") is None
        assert any("automated Ontario sportsbook feeds only" in i.value for i in at.info)

    def test_corrupt_snapshot_stops_only_the_analysis(self, paired, page):
        (paired.snapshots / "bad.json").write_text("{", encoding="utf-8")
        at = page(MONDAY)
        assert any("model snapshot" in e.value for e in at.error)
        assert frame(at, "Offer comparison") is None and frame(at, "Sunday vs Wednesday") is None
        assert frame(at, "Window closes (Toronto)") is not None      # rest of the page intact

    def test_clock_change_alone_updates_results(self, paired, page):
        at = page(datetime(2026, 10, 11, 18, 0, tzinfo=UTC))           # Sunday: no finals yet
        assert frame(at, "Offer comparison")["Result"].str.startswith("Pending").all()
        at = page(datetime(2026, 10, 14, 18, 0, tzinfo=UTC), at)       # same files, later clock
        results = frame(at, "Offer comparison")["Result"]
        assert (results == "Final").sum() == 10 and results.str.startswith("Pending").sum() == 2

    def test_new_snapshot_alone_updates_decisions(self, paired, page):
        at = page(MONDAY)
        assert frame(at, "Sunday vs Wednesday")["Sunday vs Wednesday"].eq("Unavailable").all()
        paired.snapshot(SNAP_WED, [("2026_05_CHI_GB", 3.0, 0.58, True)])
        paired.snapshot(SNAP_SUN, [("2026_05_CHI_GB", 3.0, 0.6, True)])
        at = page(MONDAY, at)                                           # same clock, rerun
        dec = frame(at, "Sunday vs Wednesday").set_index("Matchup")
        assert dec.loc["CHI @ GB", "Sunday vs Wednesday"] == "Signal retained"

    def test_rendering_writes_nothing(self, paired, page, tmp_path):
        paired.write_schedule()
        before = tree(tmp_path)
        page(MONDAY)
        assert tree(tmp_path) == before


def model_world_snapshots(w):
    w.snapshot(SNAP_WED, [("2026_05_CHI_GB", 3.0, 0.58, True)])
    w.snapshot(SNAP_SUN, [("2026_05_CHI_GB", 3.0, 0.60, True),
                          ("2026_05_DEN_LAC", 3.5, 0.5, False)])
