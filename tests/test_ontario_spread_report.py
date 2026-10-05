"""Phase 2 Wednesday-vs-Sunday Ontario spread report (scripts/ontario_spread_report.py).

Every fixture artifact is produced by Phase 1's own capture() and
manual_quote() against a mocked API, so it passes real Phase 1 validation.
No network, no real captures.
"""

import csv
import hashlib
import importlib.util
import io
import json
import shutil
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import ontario_spreads as on
from test_ontario_spreads import FakeAPI, KEY, SCHEDULE_ROWS, SUN_SLOT, WED_SLOT, _event
from test_team_features import ROOT

UTC = timezone.utc
spec = importlib.util.spec_from_file_location("ontario_spread_report",
                                              ROOT / "scripts" / "ontario_spread_report.py")
rpt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rpt)

WED_UPD = "2026-10-07T15:55:00Z"
SUN_UPD = "2026-10-11T12:55:00Z"


# ------------------------------------------------------------- fixtures --

class Store:
    """Temporary capture/manual/schedule store fed through Phase 1's API."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp, self.mp = tmp_path, monkeypatch
        self.capture_dir, self.manual_dir = tmp_path / "captures", tmp_path / "manual"
        self.schedule = tmp_path / "nfl_games_historical.csv"
        df = pd.DataFrame(SCHEDULE_ROWS, columns=["game_id", "season", "week", "gameday", "gametime",
                                                  "away_team", "home_team", "away_score", "home_score"])
        df["game_type"] = "REG"
        df.to_csv(self.schedule, sep="\t", index=False)

    def capture(self, now, events, slot="auto", directory=None):
        self.mp.setattr(on.requests, "get", FakeAPI(events))
        return on.capture(now=now, slot=slot, api_key=KEY,
                          capture_dir=directory or self.capture_dir, schedule_path=self.schedule,
                          snapshot_dir=self.tmp / "snapshots", reserve=20, repo_dir=ROOT)

    def manual(self, observed_at, now, **kw):
        args = dict(game_id="2026_05_CHI_GB", team="GB", handicap=-3.5, price=-110,
                    observed_at=observed_at, now=now, schedule_path=self.schedule,
                    manual_dir=self.manual_dir, repo_dir=ROOT)
        args.update(kw)
        return on.manual_quote(**args)

    def report(self, **kw):
        return rpt.build_report(self.capture_dir, self.manual_dir, **kw)


@pytest.fixture
def store(tmp_path, monkeypatch):
    return Store(tmp_path, monkeypatch)


def ev(game_id, updated, **books):
    """books: key=(home_point, home_price, away_point, away_price)."""
    return _event(game_id, {k: (*v, updated) for k, v in books.items()})


def wed_events(updated=WED_UPD, **overrides):
    base = {
        "2026_05_TB_DAL": {"betmgm_ca_on": (-3.5, -110, 3.5, -110)},
        "2026_05_PHI_JAX": {"betmgm_ca_on": (-2.5, -110, 2.5, -110),
                            "betano_ca_on": (-7.0, -110, 7.0, -110)},
        "2026_05_CHI_GB": {"betmgm_ca_on": (-3.5, -110, 3.5, -110),
                           "betano_ca_on": (-3.5, -110, 3.5, -110),
                           "fanduel": (-3.5, -110, 3.5, -110)},
        "2026_05_DEN_LAC": {"betano_ca_on": (-6.5, -110, 6.5, -110)},
        "2026_05_BUF_LA": {"proline_ca_on": (-1.5, 100, 1.5, -120)},
    }
    base.update(overrides)
    return [ev(g, updated, **b) for g, b in base.items()]


def sun_events(updated=SUN_UPD, **overrides):
    base = {
        "2026_05_PHI_JAX": {"betmgm_ca_on": (-3.0, -110, 3.0, -110),
                            "betano_ca_on": (-7.5, -110, 7.5, -110)},
        "2026_05_CHI_GB": {"betmgm_ca_on": (-2.5, -115, 2.5, -105),
                           "betano_ca_on": (-3.5, -110, 3.5, -110),
                           "fanduel": (-1.5, -110, 1.5, -110)},
        "2026_05_DEN_LAC": {"betano_ca_on": (-6.5, -105, 6.5, -115)},
        "2026_05_BUF_LA": {"proline_ca_on": (1.5, 100, -1.5, -120)},
    }
    base.update(overrides)
    return [ev(g, updated, **b) for g, b in base.items()]


def row(report, game_id, book, team, group=rpt.GROUP_ONTARIO):
    return next(r for r in report["rows"] if r["game_id"] == game_id and r["book_key"] == book
                and r["comparison_team"] == team and r["group"] == group)


@pytest.fixture
def week5(store):
    store.capture(WED_SLOT, wed_events())
    store.capture(SUN_SLOT, sun_events())
    return store


# -------------------------------------------------------- odds arithmetic --

class TestOddsArithmetic:
    @pytest.mark.parametrize("price,be", [(-110, 0.52381), (100, 0.5), (-100, 0.5),
                                          (150, 0.4), (-200, 0.666667), (120, 0.454545)])
    def test_break_even(self, price, be):
        assert rpt.break_even(price) == pytest.approx(be, abs=1e-6)

    def test_invalid_price_has_no_break_even(self):
        assert rpt.break_even(-50) is None and rpt.break_even(None) is None

    @pytest.mark.parametrize("w_h,w_p,s_h,s_p,out", [
        (-3.5, -110, -3.5, -110, "unchanged"),
        (-3.5, -110, -2.5, -105, "sunday_dominates"),     # more points AND better price
        (-3.5, -110, -3.5, -105, "sunday_dominates"),     # same points, better price
        (3.5, -110, 4.0, -110, "sunday_dominates"),       # underdog gets more
        (-3.5, -110, -4.5, -115, "wednesday_dominates"),
        (3.5, -105, 3.0, -110, "wednesday_dominates"),
        (-3.5, -110, -2.5, -120, "trade_off"),            # better line, worse price
        (3.5, -120, 3.0, 100, "trade_off"),               # worse line, better price
        (-1.5, 100, 1.5, 100, "sunday_dominates"),        # favourite became underdog
        (1.5, -120, -1.5, -120, "wednesday_dominates"),   # underdog became favourite
        (2.5, 110, 2.5, -110, "wednesday_dominates"),     # plus to minus price
        (2.5, -110, 2.5, 105, "sunday_dominates"),        # minus to plus price
    ])
    def test_classify_same_team(self, w_h, w_p, s_h, s_p, out):
        assert rpt.classify(w_h, w_p, s_h, s_p) == out

    @pytest.mark.parametrize("w,s,k,move", [
        (-2.5, -3.5, 3, "through"), (-3.5, -2.5, 3, "through"),     # both directions
        (2.5, 3.5, 3, "through"), (3.5, 2.5, 3, "through"),         # underdog side
        (-2.5, -3.0, 3, "onto"), (3.0, 3.5, 3, "off"),
        (-7.0, -7.5, 7, "off"), (6.5, 7.0, 7, "onto"),
        (-3.0, 3.0, 3, "on_both_sides"), (-3.0, -3.0, 3, "stays_on"),
        (-2.5, -2.5, 3, "unchanged"), (-6.5, -6.0, 3, "none"),
        (-2.5, -7.5, 3, "through"), (-2.5, -7.5, 7, "through"),
        (-1.5, 1.5, 3, "none"),
    ])
    def test_key_numbers(self, w, s, k, move):
        assert rpt.key_number_move(w, s, k) == move


# ------------------------------------------------------------- matching --

class TestComparison:
    def test_trade_off_with_key_number_crossing(self, week5):
        rep = week5.report()
        gb = row(rep, "2026_05_CHI_GB", "betmgm_ca_on", "GB")
        assert (gb["wed_handicap"], gb["wed_price"], gb["sun_handicap"], gb["sun_price"]) == (
            -3.5, -110, -2.5, -115)
        assert gb["spread_change"] == 1.0 and gb["break_even_change"] > 0
        assert gb["outcome"] == "trade_off" and gb["key_3"] == "through" and gb["key_7"] == "none"
        chi = row(rep, "2026_05_CHI_GB", "betmgm_ca_on", "CHI")
        assert chi["spread_change"] == -1.0 and chi["break_even_change"] < 0
        assert chi["outcome"] == "trade_off" and chi["key_3"] == "through"
        assert gb["matchup"] == "CHI @ GB" and gb["opponent"] == "CHI" and gb["side"] == "home"

    def test_dominance_by_price_alone(self, week5):
        rep = week5.report()
        assert row(rep, "2026_05_DEN_LAC", "betano_ca_on", "LAC")["outcome"] == "sunday_dominates"
        assert row(rep, "2026_05_DEN_LAC", "betano_ca_on", "DEN")["outcome"] == "wednesday_dominates"

    def test_favourite_flip_both_signs(self, week5):
        rep = week5.report()
        la = row(rep, "2026_05_BUF_LA", "proline_ca_on", "LA")
        buf = row(rep, "2026_05_BUF_LA", "proline_ca_on", "BUF")
        assert (la["wed_handicap"], la["sun_handicap"], la["spread_change"]) == (-1.5, 1.5, 3.0)
        assert (buf["wed_handicap"], buf["sun_handicap"], buf["spread_change"]) == (1.5, -1.5, -3.0)
        assert la["outcome"] == "sunday_dominates" and buf["outcome"] == "wednesday_dominates"

    def test_onto_and_off_key_numbers(self, week5):
        rep = week5.report()
        assert row(rep, "2026_05_PHI_JAX", "betmgm_ca_on", "JAX")["key_3"] == "onto"
        assert row(rep, "2026_05_PHI_JAX", "betmgm_ca_on", "PHI")["key_3"] == "onto"
        assert row(rep, "2026_05_PHI_JAX", "betano_ca_on", "JAX")["key_7"] == "off"

    def test_unchanged(self, week5):
        r = row(week5.report(), "2026_05_CHI_GB", "betano_ca_on", "GB")
        assert r["outcome"] == "unchanged" and r["spread_change"] == 0 and r["break_even_change"] == 0

    def test_provenance_preserved(self, week5):
        r = row(week5.report(), "2026_05_CHI_GB", "betmgm_ca_on", "GB")
        assert r["wed_slot_id"] == "2026-10-07_wednesday_noon" and r["wed_slot_status"] == "on_time"
        assert r["sun_slot_id"] == "2026-10-11_sunday_morning"
        assert r["wed_observed_at"] == "2026-10-07T16:00:00Z" and r["wed_provider_update"] == WED_UPD
        assert r["wed_provider_update_basis"] == "market_last_update"
        assert r["wed_run_id"] and r["wed_file"] == f"{r['wed_run_id']}.json"
        assert r["wed_quote_status"] == "quoted" and r["wed_age_minutes"] == 5.0
        assert r["jurisdiction"] == "CA-ON" and r["source"] == "the_odds_api"

    def test_never_matches_across_books(self, week5):
        rep = week5.report()
        for r in rep["rows"]:
            if r["status"] == "compared":
                assert r["wed_file"] != r["sun_file"]
        # betano CHI@GB compares betano with betano only (its own unchanged quotes).
        r = row(rep, "2026_05_CHI_GB", "betano_ca_on", "GB")
        assert (r["wed_handicap"], r["sun_handicap"]) == (-3.5, -3.5)

    def test_us_reference_excluded_by_default(self, week5):
        rep = week5.report()
        assert all(r["book_key"] != "fanduel" for r in rep["rows"])
        assert rep["groups"] == [rpt.GROUP_ONTARIO]

    def test_us_reference_is_a_separate_group(self, week5):
        rep = week5.report(include_us_reference=True)
        fd = row(rep, "2026_05_CHI_GB", "fanduel", "GB", group=rpt.GROUP_US_REFERENCE)
        assert fd["jurisdiction"] == "US" and fd["spread_change"] == 2.0
        ontario = rep["rollups"][rpt.GROUP_ONTARIO]["overall"]
        assert ontario == week5.report()["rollups"][rpt.GROUP_ONTARIO]["overall"]
        assert "fanduel" not in rep["rollups"][rpt.GROUP_ONTARIO]["per_book"]


# --------------------------------------------------- exclusions and gaps --

class TestExclusions:
    def test_thursday_game_unmatched(self, week5):
        r = row(week5.report(), "2026_05_TB_DAL", "betmgm_ca_on", "DAL")
        assert r["status"] == "unmatched" and r["outcome"] is None
        assert r["reasons"] == ["no_sunday_observation_kicked_off"]
        assert r["wed_handicap"] == -3.5 and r["sun_handicap"] is None

    def test_early_sunday_kickoff_unmatched_on_late_run(self, store):
        store.capture(WED_SLOT, wed_events())
        late = SUN_SLOT + timedelta(minutes=40)      # 09:40 ET: London game under way
        events = [e for e in sun_events() if e["id"] != "ev_2026_05_PHI_JAX"]
        store.capture(late, events)
        rep = store.report()
        r = row(rep, "2026_05_PHI_JAX", "betmgm_ca_on", "JAX")
        assert r["status"] == "unmatched" and r["reasons"] == ["no_sunday_observation_kicked_off"]
        gb = row(rep, "2026_05_CHI_GB", "betmgm_ca_on", "GB")
        assert gb["status"] == "compared" and gb["sun_slot_status"] == "late"

    def test_stale_quote_never_produces_an_outcome(self, store):
        store.capture(WED_SLOT, wed_events())
        stale = sun_events()
        stale[1] = ev("2026_05_CHI_GB", "2026-10-11T10:00:00Z", betmgm_ca_on=(-2.5, -105, 2.5, -115))
        store.capture(SUN_SLOT, stale)
        r = row(store.report(), "2026_05_CHI_GB", "betmgm_ca_on", "GB")
        assert r["sun_quote_status"] == "stale" and r["status"] == "quality_excluded"
        assert r["outcome"] is None and r["spread_change"] is None
        assert r["reasons"] == ["sunday_quote_stale"]

    def test_missing_book_on_sunday(self, store):
        store.capture(WED_SLOT, wed_events())
        store.capture(SUN_SLOT, sun_events(**{"2026_05_DEN_LAC": {"betmgm_ca_on": (-6.5, -110, 6.5, -110)}}))
        r = row(store.report(), "2026_05_DEN_LAC", "betano_ca_on", "LAC")
        assert r["status"] == "missing_quote" and r["reasons"] == ["sunday_quote_absent"]
        assert r["sun_handicap"] is None and r["outcome"] is None

    def test_missing_sunday_slot(self, store):
        store.capture(WED_SLOT, wed_events())
        rep = store.report()
        assert all(r["status"] == "unmatched" for r in rep["rows"])
        assert {tuple(r["reasons"]) for r in rep["rows"]} == {("no_sunday_capture_for_week",)}
        assert rep["status"] == "no_comparable_observations"

    def test_late_wednesday_capture_is_compared_and_labelled(self, store):
        store.capture(WED_SLOT + timedelta(minutes=95), wed_events("2026-10-07T17:30:00Z"))
        store.capture(SUN_SLOT, sun_events())
        r = row(store.report(), "2026_05_CHI_GB", "betmgm_ca_on", "GB")
        assert r["status"] == "compared" and r["wed_slot_status"] == "late"

    def test_ad_hoc_captures_not_compared(self, store):
        store.capture(WED_SLOT + timedelta(hours=20), wed_events(), slot="ad_hoc")
        store.capture(SUN_SLOT, sun_events())
        rep = store.report()
        assert [n["reason"] for n in rep["capture_notes"]] == ["ad_hoc_capture_not_compared"]
        assert all(r["wed_run_id"] is None for r in rep["rows"])


# ---------------------------------------------------- slot selection --

class TestSelection:
    def test_empty_capture_then_retry(self, store):
        store.capture(WED_SLOT, [])                                       # empty, kept
        store.capture(WED_SLOT + timedelta(minutes=40), wed_events())     # retry fills slot
        store.capture(SUN_SLOT, sun_events())
        rep = store.report()
        assert [n["reason"] for n in rep["capture_notes"]] == ["not_usable_no_ontario_quote"]
        r = row(rep, "2026_05_CHI_GB", "betmgm_ca_on", "GB")
        assert r["status"] == "compared" and r["wed_observed_at"] == "2026-10-07T16:40:00Z"

    def test_us_only_capture_never_represents_a_slot(self, store):
        us_only = [ev("2026_05_CHI_GB", WED_UPD, fanduel=(-3.5, -110, 3.5, -110))]
        store.capture(WED_SLOT, us_only)
        store.capture(SUN_SLOT, sun_events())
        rep = store.report(include_us_reference=True)
        assert rep["weeks"][0]["wednesday"] is None
        assert rep["capture_notes"][0]["reason"] == "not_usable_no_ontario_quote"

    def test_duplicate_usable_captures_earliest_wins(self, store, tmp_path):
        first = store.capture(WED_SLOT, wed_events())
        # A second usable capture of the same slot (e.g. a push that failed and a
        # retry) - written by Phase 1 in another directory, then both present.
        other = store.capture(WED_SLOT + timedelta(minutes=20),
                              wed_events(**{"2026_05_CHI_GB": {"betmgm_ca_on": (-9.5, -110, 9.5, -110)}}),
                              directory=tmp_path / "other")
        shutil.copy(other.path, store.capture_dir / other.path.name)
        store.capture(SUN_SLOT, sun_events())
        rep = store.report()
        r = row(rep, "2026_05_CHI_GB", "betmgm_ca_on", "GB")
        assert r["wed_run_id"] == first.doc["run_id"] and r["wed_handicap"] == -3.5
        assert rep["capture_notes"] == [{"file": other.path.name, "run_id": other.doc["run_id"],
                                         "slot_id": "2026-10-07_wednesday_noon",
                                         "reason": "superseded_by_earlier_usable_capture"}]

    def test_missing_quote_not_filled_from_other_capture(self, store, tmp_path):
        # The representative capture lacks DEN@LAC betano; a later duplicate has it.
        first = store.capture(WED_SLOT, wed_events(**{"2026_05_DEN_LAC": {}}))
        other = store.capture(WED_SLOT + timedelta(minutes=20), wed_events(),
                              directory=tmp_path / "other")
        shutil.copy(other.path, store.capture_dir / other.path.name)
        store.capture(SUN_SLOT, sun_events())
        r = row(store.report(), "2026_05_DEN_LAC", "betano_ca_on", "LAC")
        assert r["wed_run_id"] == first.doc["run_id"] and r["wed_quote_status"] == "absent"
        assert r["status"] == "missing_quote" and r["outcome"] is None

    def test_selection_independent_of_file_order(self, store, tmp_path):
        store.capture(WED_SLOT, wed_events())
        store.capture(SUN_SLOT, sun_events())
        a = store.report()
        moved = tmp_path / "moved"
        shutil.copytree(store.capture_dir, moved)
        assert rpt.build_report(moved, store.manual_dir) == a


# ------------------------------------------------------------- manual --

class TestManual:
    def test_manual_excluded_unless_requested(self, week5):
        week5.manual("2026-10-07T16:30:00Z", WED_SLOT + timedelta(hours=1))
        assert all(r["group"] != rpt.GROUP_MANUAL for r in week5.report()["rows"])

    def test_manual_compared_separately(self, week5):
        week5.manual("2026-10-07T16:30:00Z", WED_SLOT + timedelta(hours=1), opponent_price=-110)
        week5.manual("2026-10-11T13:20:00Z", SUN_SLOT + timedelta(hours=1), handicap=-3.0,
                     price=-120, opponent_price=100)
        rep = week5.report(include_manual=True)
        gb = row(rep, "2026_05_CHI_GB", "fanduel_on_manual", "GB", group=rpt.GROUP_MANUAL)
        assert gb["source"] == "manual" and gb["wed_slot_status"] == "manual"
        assert (gb["wed_handicap"], gb["sun_handicap"], gb["outcome"]) == (-3.5, -3.0, "trade_off")
        assert gb["key_3"] == "onto"
        chi = row(rep, "2026_05_CHI_GB", "fanduel_on_manual", "CHI", group=rpt.GROUP_MANUAL)
        assert chi["outcome"] == "trade_off"
        assert "fanduel_on_manual" not in rep["rollups"][rpt.GROUP_ONTARIO]["per_book"]
        assert rep["rollups"][rpt.GROUP_MANUAL]["overall"]["sides"]["compared"] == 2

    def test_manual_outside_window_not_assigned(self, week5):
        week5.manual("2026-10-08T14:00:00Z", datetime(2026, 10, 8, 15, tzinfo=UTC))   # Thursday
        rep = week5.report(include_manual=True)
        assert rep["manual_notes"][0]["reason"] == "manual_observation_outside_slot_windows"
        assert not [r for r in rep["rows"] if r["group"] == rpt.GROUP_MANUAL]

    def test_manual_only_one_slot_is_unmatched(self, week5):
        week5.manual("2026-10-07T16:30:00Z", WED_SLOT + timedelta(hours=1))
        r = row(week5.report(include_manual=True), "2026_05_CHI_GB", "fanduel_on_manual", "GB",
                group=rpt.GROUP_MANUAL)
        assert r["status"] == "unmatched"
        assert r["reasons"] == ["no_sunday_manual_observation_in_slot_window"]

    def test_manual_earliest_in_slot_wins(self, week5):
        week5.manual("2026-10-07T17:00:00Z", WED_SLOT + timedelta(hours=2), handicap=-4.5)
        week5.manual("2026-10-07T16:15:00Z", WED_SLOT + timedelta(hours=2), handicap=-3.5)
        week5.manual("2026-10-11T13:20:00Z", SUN_SLOT + timedelta(hours=1))
        rep = week5.report(include_manual=True)
        gb = row(rep, "2026_05_CHI_GB", "fanduel_on_manual", "GB", group=rpt.GROUP_MANUAL)
        assert gb["wed_observed_at"] == "2026-10-07T16:15:00Z" and gb["wed_handicap"] == -3.5
        assert [n["reason"] for n in rep["manual_notes"]] == [
            "superseded_by_earlier_manual_observation_in_slot"]


# ------------------------------------------------------------- rollups --

class TestRollups:
    def test_denominators(self, week5):
        o = week5.report()["rollups"][rpt.GROUP_ONTARIO]["overall"]
        # Pairs: TB_DAL/betmgm (unmatched), PHI_JAX betmgm+betano, CHI_GB betmgm+betano,
        # DEN_LAC betano, BUF_LA proline -> 7 pairs, 6 compared; sides = 2 per pair.
        assert o["game_book_pairs"] == {"total": 7, "compared": 6, "partially_compared": 0,
                                        "unmatched": 1, "missing_quote": 0, "quality_excluded": 0}
        assert o["sides"]["total"] == 14 and o["sides"]["compared"] == 12
        assert sum(o["sides"]["outcomes"].values()) == o["sides"]["compared"]
        # PHI@JAX x2 books and DEN@LAC, BUF@LA: one side each way; CHI@GB betmgm:
        # trade-off on both sides; CHI@GB betano: unchanged.
        assert o["sides"]["outcomes"] == {"unchanged": 2, "equivalent": 0, "sunday_dominates": 4,
                                          "wednesday_dominates": 4, "trade_off": 2}
        assert o["games"] == {"total": 5, "with_any_compared_pair": 4, "with_no_compared_pair": 1}
        # Requested Ontario books that quoted neither slot: 6 books x 5 games - 7 quoted pairs.
        assert o["game_book_pairs_not_quoted_in_either_slot"] == 23

    def test_per_book_has_no_best_book(self, week5):
        g = week5.report()["rollups"][rpt.GROUP_ONTARIO]
        quoted = {b for b, c in g["per_book"].items() if c["game_book_pairs"]["total"]}
        assert quoted == {"betmgm_ca_on", "betano_ca_on", "proline_ca_on"}
        assert g["per_book"]["pointsbetca"]["game_book_pairs_not_quoted_in_either_slot"] == 5
        assert "no overall best book" in g["note"]
        text = json.dumps(g).lower()
        assert "best_book" not in text and "winner" not in text

    def test_filters(self, week5):
        rep = week5.report(books=["betano_ca_on"], season=2026, week=5)
        assert {r["book_key"] for r in rep["rows"]} == {"betano_ca_on"}
        assert week5.report(week=6)["rows"] == []


# ------------------------------------------------ empty, files, read-only --

def _tree_hashes(*dirs):
    out = {}
    for d in dirs:
        for p in sorted(d.rglob("*")) if d.exists() else []:
            if p.is_file():
                out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


class TestFiles:
    def test_empty_history(self, tmp_path):
        rep = rpt.build_report(tmp_path / "none", tmp_path / "none_manual", include_manual=True)
        assert rep["status"] == "no_observations_yet" and rep["rows"] == [] and rep["weeks"] == []
        j, c = rpt.write_report(rep, tmp_path / "out")
        assert json.loads(j.read_text(encoding="utf-8"))["status"] == "no_observations_yet"
        assert c.read_text(encoding="utf-8") == ",".join(rpt.CSV_FIELDS) + "\n"

    def test_read_only_and_deterministic(self, week5, tmp_path):
        week5.manual("2026-10-07T16:30:00Z", WED_SLOT + timedelta(hours=1))
        sources = lambda: {**_tree_hashes(week5.capture_dir, week5.manual_dir),  # noqa: E731
                           "schedule": hashlib.sha256(week5.schedule.read_bytes()).hexdigest()}
        before = sources()
        out1, out2 = tmp_path / "r1", tmp_path / "r2"
        for out in (out1, out2):
            assert rpt.main(["--capture-dir", str(week5.capture_dir), "--manual-dir",
                             str(week5.manual_dir), "--include-manual", "--include-us-reference",
                             "--output-dir", str(out)]) == 0
        assert sources() == before and len(before) == 4     # 2 captures, 1 manual, schedule
        for name in (rpt.JSON_NAME, rpt.CSV_NAME):
            assert (out1 / name).read_bytes() == (out2 / name).read_bytes()
        rows = list(csv.DictReader(io.StringIO((out1 / rpt.CSV_NAME).read_text(encoding="utf-8"))))
        assert len(rows) == len(json.loads((out1 / rpt.JSON_NAME).read_text(encoding="utf-8"))["rows"])

    def test_refuses_to_write_into_source_data(self, tmp_path):
        rep = rpt.build_report(tmp_path / "none", tmp_path / "none")
        with pytest.raises(rpt.ReportError):
            rpt.write_report(rep, on.CAPTURE_DIR)
        with pytest.raises(rpt.ReportError):
            rpt.write_report(rep, on.DATA_DIR / "x")

    def test_invalid_capture_stops_the_report(self, week5, tmp_path):
        bad = next(week5.capture_dir.glob("*.json"))
        doc = json.loads(bad.read_text(encoding="utf-8"))
        doc["games"][0]["quotes"][0]["home_price"] = -105            # not re-sealed
        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / bad.name).write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(on.ValidationError, match="checksum"):
            rpt.build_report(broken, tmp_path / "none")
        assert rpt.main(["--capture-dir", str(broken), "--output-dir", str(tmp_path / "o")]) == 1

    def test_report_wording_makes_no_timing_or_roi_claim(self, week5):
        text = json.dumps(week5.report()).lower()
        assert "closing line" in text            # only in the explicit disclaimer
        assert "not a closing line" in text
        for phrase in ("best time", "increased roi", "profitable", "recommend"):
            assert phrase not in text


def test_default_output_dir_is_git_ignored():
    import subprocess
    out = subprocess.run(["git", "check-ignore", "-q", str(rpt.DEFAULT_OUTPUT_DIR / rpt.JSON_NAME)],
                         cwd=ROOT)
    assert out.returncode == 0



# --------------------------------------------- review fixes (Phase 2) --

class TestPayoffDominance:
    """Outcomes are decided by settling both bets for every integer margin."""

    @pytest.mark.parametrize("h,p,m,result", [
        (-3, -110, 3, 0), (-3, -110, 4, pytest.approx(100 / 110)), (-3, -110, 2, -1),
        (3, 150, -3, 0), (3, 150, -2, pytest.approx(1.5)), (-6.5, -200, 7, pytest.approx(0.5)),
    ])
    def test_settle_win_push_loss(self, h, p, m, result):
        assert float(rpt.settle(h, p, m)) == result

    @pytest.mark.parametrize("w_h,w_p,s_h,s_p,out", [
        # Whole-number lines: a push turns into a win or a loss.
        (-3, -110, -2.5, -110, "sunday_dominates"),       # margin 3: push -> win
        (-3, -110, -3.5, -110, "wednesday_dominates"),    # margin 3: push -> loss
        (-3, -110, -2.5, -120, "trade_off"),              # margin 3 better, wins pay less
        (-3, -110, -3.5, -105, "trade_off"),              # margin 3 worse, wins pay more
        (7, -110, 7, -105, "sunday_dominates"),
        # Odds-on prices: winning profit below the stake.
        (-7, -200, -6.5, -250, "trade_off"),              # push->win at 7, but 0.5 -> 0.4 profit
        (-7, -250, -6.5, -200, "sunday_dominates"),
        (-1.5, -300, -1.5, -400, "wednesday_dominates"),
        # Equal payoffs, different quote.
        (2.5, -100, 2.5, 100, "equivalent"),
        (-3, -110, -3, -110, "unchanged"),
    ])
    def test_classify(self, w_h, w_p, s_h, s_p, out):
        assert rpt.classify(w_h, w_p, s_h, s_p) == out

    def test_payoff_agrees_with_handicap_and_break_even_rule(self):
        # On a grid of half-point lines and prices, the payoff decision equals
        # "larger handicap / lower break-even" whenever quotes aren't equivalent.
        import itertools
        hs = [x / 2 for x in range(-15, 16, 3)]
        prices = [-300, -150, -115, -110, -100, 100, 105, 130, 250]
        for wh, wp, sh, sp in itertools.product(hs, prices, hs, prices):
            got = rpt.classify(wh, wp, sh, sp)
            dh, dbe = sh - wh, rpt.break_even(sp) - rpt.break_even(wp)
            if got in ("unchanged", "equivalent"):
                assert dh == 0 and dbe == 0
                continue
            better, worse = dh > 0 or dbe < 0, dh < 0 or dbe > 0
            simple = ("sunday_dominates" if better and not worse else
                      "wednesday_dominates" if worse and not better else "trade_off")
            assert got == simple, (wh, wp, sh, sp)


OPENER_ROWS = [
    # game_id, season, week, gameday, gametime, away, home, away_score, home_score
    ("2026_01_DAL_PHI", 2026, 1, "2026-09-10", "20:20", "DAL", "PHI", None, None),  # Thursday
    ("2026_01_CHI_GB", 2026, 1, "2026-09-13", "13:00", "CHI", "GB", None, None),
]


def _opener_event(game_id, updated, point):
    from test_ontario_spreads import _book
    row = next(r for r in OPENER_ROWS if r[0] == game_id)
    away, home = row[5], row[6]
    kickoff = on.ps.kickoff_utc(row[3], row[4])[0]
    return {"id": f"ev_{game_id}", "commence_time": on.iso(kickoff),
            "home_team": on.TEAM_FULL_NAME[home], "away_team": on.TEAM_FULL_NAME[away],
            "bookmakers": [_book("betmgm_ca_on", point, -110, -point, -110, updated, home, away)]}


class TestSameWeekPairing:
    @pytest.fixture
    def opener(self, store):
        df = pd.DataFrame(OPENER_ROWS, columns=["game_id", "season", "week", "gameday", "gametime",
                                                "away_team", "home_team", "away_score", "home_score"])
        df["game_type"] = "REG"
        df.to_csv(store.schedule, sep="\t", index=False)
        return store

    def test_earlier_wednesday_never_stands_in(self, opener):
        # Wednesday Sept 2 (eight days before the opener) contains week-1 games;
        # the intended Wednesday, Sept 9, is missing.
        early = datetime(2026, 9, 2, 16, 0, tzinfo=UTC)
        opener.capture(early, [_opener_event(g, "2026-09-02T15:55:00Z", -3.5)
                               for g in ("2026_01_DAL_PHI", "2026_01_CHI_GB")])
        sunday = datetime(2026, 9, 13, 13, 0, tzinfo=UTC)
        opener.capture(sunday, [_opener_event("2026_01_CHI_GB", "2026-09-13T12:55:00Z", -2.5)])
        rep = opener.report()
        assert rep["weeks"][0]["wednesday"] is None
        assert any(n["slot_id"] == "2026-09-02_wednesday_noon"
                   and n["reason"].startswith("wednesday_not_in_same_nfl_week")
                   for n in rep["capture_notes"])
        r = row(rep, "2026_01_CHI_GB", "betmgm_ca_on", "GB")
        assert r["status"] == "unmatched" and r["reasons"] == ["no_wednesday_capture_for_week"]
        assert r["wed_run_id"] is None and r["outcome"] is None

    def test_same_week_wednesday_is_used(self, opener):
        opener.capture(datetime(2026, 9, 2, 16, 0, tzinfo=UTC),
                       [_opener_event(g, "2026-09-02T15:55:00Z", -6.5)
                        for g in ("2026_01_DAL_PHI", "2026_01_CHI_GB")])
        opener.capture(datetime(2026, 9, 9, 16, 0, tzinfo=UTC),
                       [_opener_event(g, "2026-09-09T15:55:00Z", -3.5)
                        for g in ("2026_01_DAL_PHI", "2026_01_CHI_GB")])
        opener.capture(datetime(2026, 9, 13, 13, 0, tzinfo=UTC),
                       [_opener_event("2026_01_CHI_GB", "2026-09-13T12:55:00Z", -2.5)])
        r = row(opener.report(), "2026_01_CHI_GB", "betmgm_ca_on", "GB")
        assert r["wed_slot_id"] == "2026-09-09_wednesday_noon" and r["wed_handicap"] == -3.5
        assert r["status"] == "compared"


class TestOutputSafety:
    def _source(self, week5):
        return next(week5.capture_dir.glob("*.json"))

    def test_hard_link_at_output_name_is_replaced_not_written_through(self, week5, tmp_path):
        src = self._source(week5)
        before = src.read_bytes()
        out = tmp_path / "out"
        out.mkdir()
        import os
        os.link(src, out / rpt.JSON_NAME)          # same file under the report's name
        rep = week5.report()
        rpt.write_report(rep, out)
        assert src.read_bytes() == before
        assert json.loads((out / rpt.JSON_NAME).read_text(encoding="utf-8"))["kind"] == rpt.REPORT_KIND

    def test_symlink_at_output_name_is_replaced_not_followed(self, week5, tmp_path):
        import os
        src = self._source(week5)
        before = src.read_bytes()
        out = tmp_path / "out"
        out.mkdir()
        try:
            os.symlink(src, out / rpt.JSON_NAME)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not permitted here")
        rpt.write_report(week5.report(), out)
        assert src.read_bytes() == before and not (out / rpt.JSON_NAME).is_symlink()

    def test_symlinked_output_dir_into_captures_refused(self, week5, tmp_path):
        import os
        link = tmp_path / "link_to_captures"
        try:
            os.symlink(week5.capture_dir, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not permitted here")
        with pytest.raises(rpt.ReportError, match="overlaps source data"):
            rpt.write_report(week5.report(), link, protected=(week5.capture_dir,))

    @pytest.mark.parametrize("variant", ["same", "dotdot", "case", "child", "parent"])
    def test_custom_capture_dir_and_alternate_paths_refused(self, week5, variant):
        cap = week5.capture_dir
        out = {"same": cap, "dotdot": cap / ".." / cap.name,
               "case": cap.parent / cap.name.upper(), "child": cap / "reports",
               "parent": cap.parent}[variant]
        if variant == "case" and not out.exists():
            pytest.skip("case-sensitive filesystem")
        before = {p.name: p.read_bytes() for p in cap.glob("*.json")}
        with pytest.raises(rpt.ReportError):
            rpt.write_report(week5.report(), out, protected=(cap, week5.manual_dir))
        assert {p.name: p.read_bytes() for p in cap.glob("*.json")} == before

    def test_cli_protects_its_own_capture_dir(self, week5, capsys):
        assert rpt.main(["--capture-dir", str(week5.capture_dir),
                         "--output-dir", str(week5.capture_dir)]) == 1
        assert not (week5.capture_dir / rpt.JSON_NAME).exists()

    def test_repo_data_dir_variants_refused(self, tmp_path):
        rep = rpt.build_report(tmp_path / "none", tmp_path / "none")
        for out in (on.DATA_DIR / "ontario_spreads" / ".." / "x", on.STORE_DIR / "captures"):
            with pytest.raises(rpt.ReportError):
                rpt.write_report(rep, out)
