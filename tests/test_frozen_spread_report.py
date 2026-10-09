"""Frozen-pregame spread performance report (scripts/frozen_spread_report.py).

Synthetic pregame snapshots (validated exactly like real ones), temporary
schedules, Ontario market captures written through Phase 1's own capture code
against a fake provider, and blocked network access. Nothing in data_files/
is read or written.
"""
import importlib.util
import json
import math
import os
import socket
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import bet_journal as bj
import pregame_snapshots as ps
from test_ontario_spread_report import Store, ev
from test_ontario_spreads import WED_SLOT
from test_team_features import ROOT

spec = importlib.util.spec_from_file_location("frozen_spread_report",
                                              ROOT / "scripts" / "frozen_spread_report.py")
fr = importlib.util.module_from_spec(spec)
sys.modules["frozen_spread_report"] = fr
spec.loader.exec_module(fr)

UTC = timezone.utc
S1 = datetime(2026, 10, 6, 9, 0, tzinfo=UTC)      # nightly captures
S2 = datetime(2026, 10, 7, 9, 0, tzinfo=UTC)
AS_OF = date(2026, 10, 14)                         # every fixture game is in the past
DANGEROUS_SHA = "85e6046866697474fdc3e98f317d510f27dd4072"

# Same games and kickoffs as the Ontario capture fixtures (US Eastern).
KICKOFF = {"2026_05_TB_DAL": ("2026-10-08", "20:15"), "2026_05_CHI_GB": ("2026-10-11", "13:00"),
           "2026_05_DEN_LAC": ("2026-10-11", "16:05"), "2026_05_BUF_LA": ("2026-10-12", "20:15"),
           "2026_05_PHI_JAX": ("2026-10-11", "09:30")}


def _ko(game_id, day=None, at=None):
    d, t = KICKOFF[game_id]
    return ps.kickoff_utc(day or d, at or t)[0]


# ------------------------------------------------------------- fixtures --

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)


class World:
    """Temporary snapshots, schedule and market captures for one test."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp, self.mp = tmp_path, monkeypatch
        self.snapshots = tmp_path / "pregame_snapshots"
        self.snapshots.mkdir()
        self.schedule = tmp_path / "sched" / ps.SCHEDULE_NAME
        self.schedule.parent.mkdir()
        (tmp_path / "market").mkdir()
        self.store = Store(tmp_path / "market", monkeypatch)
        self.captures = self.store.capture_dir
        self._n = 0
        self.rows = {}

    # snapshots ------------------------------------------------------------
    def manifest(self, generated_at, revision):
        self._n += 1
        config, feats = {"model": "xgb"}, ["spread_line"]
        cut = {"season": 2026, "week": 4, "gameday": "2026-10-05", "games": 64}
        m = {"schema_version": 1, "run_id": f"{generated_at:%Y%m%dT%H%M%SZ}-{self._n:012x}",
             "generated_at": ps._iso(generated_at), "code_revision": revision,
             "code_dirty": False, "code_sha256": "a" * 64, "config_id": ps._short_hash(config),
             "config": config, "feature_set_id": ps._short_hash(feats), "features": feats,
             "training_cutoff": cut, "data_cutoff": cut, "spread_threshold": 0.5438,
             "ci_run": {}, "platform": {"python": "3.13"},
             "artifact": {"path": ps.PREDICTIONS_NAME, "sha256": "b" * 64},
             "schedule": {"path": ps.SCHEDULE_NAME, "sha256": "c" * 64}}
        ps.validate_manifest(m)
        return m

    def snapshot(self, captured_at, games, revision="rev-a"):
        """games: (game_id, line, prob, signal[, kickoff]) - status derived as
        pregame_snapshots.build_snapshot derives it."""
        out = []
        for spec_ in games:
            gid, line, prob, signal = spec_[:4]
            ko = spec_[4] if len(spec_) > 4 else _ko(gid)
            season, week, away, home = gid.split("_")
            status = ps.NO_LINE if line is None else ps.PICKEM if line == 0 else \
                ps.NO_PROBABILITY if prob is None else ps.PREDICTED
            out.append({"season": int(season), "week": int(week), "game_id": gid,
                        "home_team": home, "away_team": away, "kickoff_utc": ps._iso(ko),
                        "kickoff_source": "fixture", "spread_line": None if line is None else float(line),
                        "line_status": ps.LINE_STATUS_FOR[status],
                        "underdog_team": None if status in (ps.NO_LINE, ps.PICKEM)
                        else (away if line > 0 else home),
                        "prob_underdog_covers": prob if status == ps.PREDICTED else None,
                        "bet_signal": bool(signal) and status == ps.PREDICTED,
                        "prediction_status": status})
        counts = {"completed_excluded": 0, "captured": len(out), "skipped": 0}
        for g in out:
            counts[g["prediction_status"]] = counts.get(g["prediction_status"], 0) + 1
        m = self.manifest(captured_at - timedelta(minutes=5), revision)
        doc = {"schema_version": 1, "kind": ps.KIND, "run_id": m["run_id"],
               "captured_at": ps._iso(captured_at), "run": m, "kickoff_timezone": ps.SOURCE_TZ.key,
               "spread_convention": ps.SPREAD_CONVENTION, "probability": ps.PROBABILITY_MEANING,
               "counts": counts, "games": out, "skipped": []}
        doc[ps.CHECKSUM_FIELD] = ps.payload_checksum(doc)
        ps.validate_snapshot(doc)
        path = self.snapshots / f"{m['run_id']}.json"
        path.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
        return path

    # schedule -------------------------------------------------------------
    def game(self, gid, home_score=None, away_score=None, line=3.0, day=None, at=None, **over):
        season, week, away, home = gid.split("_")
        d, t = KICKOFF.get(gid, ("2026-10-11", "13:00"))
        row = {"game_id": gid, "season": int(season), "week": int(week), "game_type": "REG",
               "gameday": day or d, "gametime": at or t, "home_team": home, "away_team": away,
               "home_score": home_score, "away_score": away_score, "spread_line": line,
               "result": None if home_score is None else home_score - away_score,
               "total": None if home_score is None else home_score + away_score,
               "overtime": None if home_score is None else 0}
        row.update(over)
        self.rows.setdefault(gid, []).append(row)

    def write_schedule(self):
        rows = [r for gid in sorted(self.rows) for r in self.rows[gid]]
        pd.DataFrame(rows).to_csv(self.schedule, sep="\t", index=False)

    # market ---------------------------------------------------------------
    def market(self, now, events):
        return self.store.capture(now, events)

    def report(self, **kw):
        self.write_schedule()
        kw.setdefault("as_of", AS_OF)
        return fr.build_report(self.snapshots, self.schedule, self.captures, **kw)


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def game(report, gid):
    return next(g for g in report["games"] if g["game_id"] == gid)


# -------------------------------------------------------------- selection --

class TestSelection:
    def test_latest_is_default_and_earliest_is_explicit(self, w):
        w.snapshot(S1, [("2026_05_CHI_GB", 3.0, 0.40, False)], revision="rev-1")
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.62, True)], revision="rev-2")
        w.game("2026_05_CHI_GB", 20, 24)                     # GB by 4 -> CHI +3 loses
        latest = game(w.report(), "2026_05_CHI_GB")
        assert latest["captured_at"] == ps._iso(S2) and latest["code_revision"] == "rev-2"
        assert latest["prob_underdog_covers"] == 0.62 and latest["bet_signal"] is True
        earliest = game(w.report(which="earliest"), "2026_05_CHI_GB")
        assert earliest["captured_at"] == ps._iso(S1) and earliest["prob_underdog_covers"] == 0.40
        for g in (latest, earliest):                          # provenance recorded
            assert g["run_id"] and g["snapshot_file"] == f"{g['run_id']}.json"

    def test_latest_without_a_prediction_never_falls_back(self, w):
        # The earlier capture had a probability AND a signal; the latest has
        # no probability. The latest is still the observation, so no signal.
        w.snapshot(S1, [("2026_05_CHI_GB", 3.0, 0.70, True)])
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, None, False)])
        w.game("2026_05_CHI_GB", 17, 24)
        r = w.report()
        g = game(r, "2026_05_CHI_GB")
        assert g["status"] == ps.NO_PROBABILITY and g["captured_at"] == ps._iso(S2)
        assert g["bet_signal"] is False and g["prob_underdog_covers"] is None
        assert r["simulated_returns"]["assumed_minus_110"]["bets"] == 0
        assert r["probability_metrics"]["all_valid_probabilities"]["overall"]["n"] == 0

    @pytest.mark.parametrize("current_at,expected", [
        ("13:00", "unchanged"),     # kickoff as frozen: capture before it
        ("16:00", "moved_later"),   # flexed later: still pregame
        ("05:00", "capture_not_before_current_kickoff"),  # == S2 (09:00Z = 05:00 EDT): AT kickoff
        ("04:00", "capture_not_before_current_kickoff"),  # moved before the capture: AFTER kickoff
    ])
    def test_capture_must_precede_the_current_kickoff(self, w, current_at, expected):
        w.snapshot(S1, [("2026_05_CHI_GB", 3.0, 0.55, False)])
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.65, True)])
        w.game("2026_05_CHI_GB", 20, 24, day="2026-10-07" if current_at in ("05:00", "04:00")
               else None, at=current_at)
        g = game(w.report(), "2026_05_CHI_GB")
        assert g["kickoff_check"] == expected
        if expected.startswith("capture_not"):
            # Excluded - and the earlier S1 capture is NOT used instead.
            assert g["status"] == fr.TIMING_UNVERIFIED and g["captured_at"] == ps._iso(S2)
            assert g["underdog_result"] is None and g["market_status"] == fr.NOT_APPLICABLE
        else:
            assert g["status"] == fr.EVALUATED

    @pytest.mark.parametrize("offset", [timedelta(0), -timedelta(minutes=1)],
                             ids=["at_kickoff", "after_kickoff"])
    def test_capture_at_or_after_its_recorded_kickoff_fails_visibly(self, w, offset):
        # A valid snapshot can't hold such a game. One edited to it (and
        # resealed, as a forger could) is rejected by the shared validation.
        path = w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.55, False)])
        doc = json.loads(path.read_text())
        doc["games"][0]["kickoff_utc"] = ps._iso(S2 + offset)
        doc[ps.CHECKSUM_FIELD] = ps.payload_checksum(doc)
        path.write_text(json.dumps(doc))
        w.game("2026_05_CHI_GB", 20, 24)
        with pytest.raises(fr.ReportError, match="not strictly after captured_at"):
            w.report()


# --------------------------------------------------------------- outcomes --

class TestOutcomes:
    def test_frozen_handicap_not_todays_line(self, w):
        # Frozen: GB favoured by 3.5 -> CHI +3.5. Today's schedule says 7.
        # GB wins by 5: CHI +3.5 loses (it would have covered +7).
        w.snapshot(S2, [("2026_05_CHI_GB", 3.5, 0.6, True)])
        w.game("2026_05_CHI_GB", 25, 20, line=7.0)
        g = game(w.report(), "2026_05_CHI_GB")
        assert g["underdog_team"] == "CHI" and g["underdog_handicap"] == 3.5
        assert g["underdog_result"] == "loss" and g["underdog_covered"] == 0

    def test_home_underdog_handicap(self, w):
        # spread_line -2.5: the away team (CHI) is favoured, so home GB is the dog at +2.5.
        w.snapshot(S2, [("2026_05_CHI_GB", -2.5, 0.5, False)])
        w.game("2026_05_CHI_GB", 21, 23)                   # GB loses by 2 -> covers +2.5
        g = game(w.report(), "2026_05_CHI_GB")
        assert (g["underdog_team"], g["underdog_handicap"], g["underdog_result"]) == ("GB", 2.5, "win")

    @pytest.mark.parametrize("scores,over,status,result", [
        ((24, 17), {}, fr.EVALUATED, "loss"),        # home by 7, dog +3
        ((20, 24), {}, fr.EVALUATED, "win"),
        ((23, 20), {}, fr.PUSH, "push"),             # exactly 3
        ((None, None), {}, fr.PENDING, None),        # no score yet
        ((0, 0), {"result": 0, "total": 0}, fr.INVALID_OUTCOME, None),       # placeholder
        ((24, 17), {"result": 3}, fr.INVALID_OUTCOME, None),                 # inconsistent
    ])
    def test_settlement_and_unresolved(self, w, scores, over, status, result):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.55, False)])
        w.game("2026_05_CHI_GB", *scores, **over)
        g = game(w.report(), "2026_05_CHI_GB")
        assert (g["status"], g["underdog_result"]) == (status, result)

    def test_duplicated_schedule_row_is_unresolved(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.55, False)])
        w.game("2026_05_CHI_GB", 20, 24)
        w.game("2026_05_CHI_GB", 24, 20)
        g = game(w.report(), "2026_05_CHI_GB")
        assert g["status"] == fr.INVALID_OUTCOME and "more than once" in g["status_reason"]

    def test_schedule_teams_differing_from_the_snapshot_are_unresolved(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.55, False)])
        w.game("2026_05_CHI_GB", 20, 24, home_team="CHI", away_team="GB")   # listed reversed
        g = game(w.report(), "2026_05_CHI_GB")
        assert g["status"] == fr.INVALID_OUTCOME and "teams differ" in g["status_reason"]
        assert g["underdog_result"] is None

    def test_game_day_not_before_as_of_is_pending(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.55, False)])
        w.game("2026_05_CHI_GB", 20, 24)
        g = game(w.report(as_of=date(2026, 10, 11)), "2026_05_CHI_GB")
        assert g["status"] == fr.PENDING


# --------------------------------------------------------------- coverage --

class TestCoverage:
    def test_every_selected_game_is_accounted_for(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.55, False),          # evaluated
                        ("2026_05_DEN_LAC", 3.0, 0.45, True),          # push
                        ("2026_05_BUF_LA", 3.0, 0.52, False),          # pending
                        ("2026_05_PHI_JAX", None, None, False),        # no line
                        ("2026_05_TB_DAL", 0, None, False)])           # pick'em
        w.game("2026_05_CHI_GB", 20, 24)
        w.game("2026_05_DEN_LAC", 23, 20)
        w.game("2026_05_BUF_LA")
        w.game("2026_05_PHI_JAX", 30, 10)
        w.game("2026_05_TB_DAL", 24, 16)
        r = w.report()
        c = r["coverage"]
        assert c["selected_games"] == 5 == sum(c["by_status"].values())
        assert {k: v for k, v in c["by_status"].items() if v} == {
            fr.EVALUATED: 1, fr.PUSH: 1, fr.PENDING: 1, ps.NO_LINE: 1, ps.PICKEM: 1}
        assert r["probability_metrics"]["all_valid_probabilities"]["pushes_excluded"] == 1

    def test_completed_games_without_a_snapshot_are_listed_not_backfilled(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.55, False)])
        w.game("2026_05_CHI_GB", 20, 24)
        w.game("2026_05_DEN_LAC", 30, 3)                       # after first capture: a gap
        w.game("2026_04_ATL_NO", 24, 17, day="2026-10-05", at="20:15")   # before it
        r = w.report()
        cw = r["coverage"]["completed_without_snapshot"]
        assert cw["completed_games_in_scope"] == 3
        assert (cw["before_first_snapshot"], cw["after_first_snapshot"]) == (1, 1)
        assert cw["after_first_snapshot_games"] == ["2026_05_DEN_LAC"]
        assert [g["game_id"] for g in r["games"]] == ["2026_05_CHI_GB"]   # not backfilled
        assert "not the whole season" in r["coverage"]["sample_note"]


class TestScope:
    def test_no_snapshot_history_still_reports_completed_games(self, w):
        # Empty snapshot directory: completed games are reported as lacking
        # snapshots, not as zero coverage.
        w.game("2026_05_CHI_GB", 20, 24)
        w.game("2026_05_DEN_LAC", 27, 20)
        w.game("2026_05_BUF_LA")                                       # not completed
        r = w.report()
        cw = r["coverage"]["completed_without_snapshot"]
        assert r["coverage"]["selected_games"] == 0
        assert (cw["completed_games_in_scope"], cw["completed_without_snapshot"],
                cw[fr.NO_HISTORY], cw[fr.BEFORE_FIRST], cw[fr.AFTER_FIRST]) == (2, 2, 2, 0, 0)
        assert cw["first_snapshot_captured_at"] is None
        assert r["scope"]["snapshot_history"] == "none"
        assert "no snapshot history" in r["coverage"]["sample_note"]

    def test_only_ineligible_snapshots(self, w):
        # Snapshots exist, but only for a later game: completed games after the
        # first capture are coverage gaps; one before it is not.
        w.snapshot(S2, [("2026_05_BUF_LA", 3.0, 0.55, False)])
        w.game("2026_05_BUF_LA")
        w.game("2026_05_CHI_GB", 20, 24)                               # after S2: gap
        w.game("2026_04_ATL_NO", 24, 17, day="2026-10-05", at="20:15") # before S2
        cw = w.report()["coverage"]["completed_without_snapshot"]
        assert (cw[fr.BEFORE_FIRST], cw[fr.AFTER_FIRST], cw[fr.NO_HISTORY]) == (1, 1, 0)
        assert cw["after_first_snapshot_games"] == ["2026_05_CHI_GB"]
        assert cw["first_snapshot_captured_at"] == ps._iso(S2)

    def test_timing_unverified_game_is_selected_not_missing(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.55, False)])
        w.game("2026_05_CHI_GB", 20, 24, day="2026-10-07", at="04:00")  # moved before S2
        r = w.report()
        cw = r["coverage"]["completed_without_snapshot"]
        assert r["coverage"]["by_status"][fr.TIMING_UNVERIFIED] == 1
        assert (cw["completed_with_selected_snapshot"], cw["completed_without_snapshot"]) == (1, 0)

    def test_explicit_season_with_no_selected_games(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.55, False)])
        w.game("2026_05_CHI_GB", 20, 24)
        w.game("2025_05_KC_BUF", 24, 21, day="2025-10-05", at="13:00")
        r = w.report(season=2025)
        cw = r["coverage"]["completed_without_snapshot"]
        assert r["coverage"]["selected_games"] == 0 and r["scope"]["seasons"] == [2025]
        assert r["scope"]["rule"] == "explicit --season" and r["scope"]["selected_outside_scope"] == 1
        assert (cw["completed_games_in_scope"], cw[fr.BEFORE_FIRST]) == (1, 1)

    def test_week_filter_with_no_selected_games(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.55, False)])
        w.game("2026_05_CHI_GB", 20, 24)
        w.game("2026_06_DAL_NYG", 20, 24, day="2026-10-15", at="20:15")
        r = w.report(week=6, as_of=date(2026, 10, 20))                 # Oct 15 game completed
        cw = r["coverage"]["completed_without_snapshot"]
        assert r["coverage"]["selected_games"] == 0 and r["scope"]["selected_outside_scope"] == 1
        assert (cw["completed_games_in_scope"], cw[fr.AFTER_FIRST]) == (1, 1)
        assert cw["after_first_snapshot_games"] == ["2026_06_DAL_NYG"]

    def test_default_scope_is_the_latest_schedule_season(self, w):
        # Defined from the schedule, not from which predictions were selected:
        # a 2025 snapshot game is outside the default 2026 scope and counted.
        w.snapshot(S1, [("2025_05_KC_BUF", 3.0, 0.55, False,
                         datetime(2026, 10, 6, 20, 0, tzinfo=UTC))])
        w.game("2025_05_KC_BUF", 24, 21, day="2026-10-06", at="16:00")
        w.game("2026_05_CHI_GB", 20, 24)
        r = w.report()
        assert r["scope"]["seasons"] == [2026] and r["scope"]["rule"].startswith("default")
        assert r["scope"]["selected_outside_scope"] == 1 and r["coverage"]["selected_games"] == 0
        cw = r["coverage"]["completed_without_snapshot"]
        assert cw["completed_games_in_scope"] == 1 and cw["after_first_snapshot_games"] == [
            "2026_05_CHI_GB"]

    def test_as_of_is_documented_as_an_outcome_cutoff(self, w):
        w.game("2026_05_CHI_GB", 20, 24)
        r = w.report()
        assert "not a historical reconstruction" in r["as_of_semantics"]
        assert "later score corrections may be present" in r["as_of_semantics"]


# ---------------------------------------------------------------- metrics --

class TestMetrics:
    def test_hand_calculated_brier_log_loss_and_calibration(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.8, True),           # covers
                        ("2026_05_DEN_LAC", 3.0, 0.6, False),         # doesn't
                        ("2026_05_BUF_LA", 3.0, 0.3, False)],         # doesn't
                   revision="rev-x")
        w.game("2026_05_CHI_GB", 20, 24)
        w.game("2026_05_DEN_LAC", 27, 20)
        w.game("2026_05_BUF_LA", 30, 20)
        m = w.report()["probability_metrics"]["all_valid_probabilities"]
        o = m["overall"]
        assert o["n"] == 3 and o["covers"] == 1
        assert o["brier"] == pytest.approx((0.2 ** 2 + 0.6 ** 2 + 0.3 ** 2) / 3, abs=1e-6)
        assert o["log_loss"] == pytest.approx(-(math.log(0.8) + math.log(0.4) + math.log(0.7)) / 3,
                                              abs=1e-6)
        assert (o["brier_baseline_50"], o["log_loss_baseline_50"]) == (0.25, round(math.log(2), 6))
        bins = {b["bin"]: b for b in m["calibration"] if b["n"]}
        assert {k: (b["n"], b["mean_probability"], b["observed_cover_rate"])
                for k, b in bins.items()} == {"[0.3, 0.4)": (1, 0.3, 0.0),
                                              "[0.6, 0.7)": (1, 0.6, 0.0),
                                              "[0.8, 0.9)": (1, 0.8, 1.0)}
        assert len(m["calibration"]) == 10
        assert set(m["by_season_week"]) == {"2026-W05"} and set(m["by_code_revision"]) == {"rev-x"}
        signal = w.report()["probability_metrics"]["signal_only"]["overall"]
        assert signal["n"] == 1 and signal["brier"] == pytest.approx(0.04)

    def test_zero_and_one_probabilities_are_clipped_for_log_loss_only(self):
        m = fr.prob_metrics([(1.0, 0), (0.0, 0)])
        assert m["n_probability_0_or_1"] == 2 and m["n_clipped_for_log_loss"] == 2
        assert m["brier"] == pytest.approx(0.5)                       # unclipped
        eps = fr.LOG_LOSS_EPS                                         # documented clipping
        hi = 1 - eps                                                  # p=1.0 -> 1 - eps
        assert m["log_loss"] == pytest.approx((-math.log(1 - hi) - math.log(1 - eps)) / 2, rel=1e-6)
        assert math.isfinite(m["log_loss"])
        assert fr.calibration([(1.0, 1)])[-1]["n"] == 1               # 1.0 in the last bin

    def test_empty_metrics_are_json_safe(self):
        m = fr.prob_metrics([])
        assert m["n"] == 0 and m["brier"] is None
        json.dumps(m, allow_nan=False)


# ----------------------------------------------------------------- market --

def _two_games_market(w, chi=(-3.0, -110, 3.0, -110), den=(-3.0, -105, 3.0, -115), chi2=None):
    """Capture at WED_SLOT (16:00Z Oct 7); quotes updated at 15:55Z. CHI @ GB
    is quoted by BetMGM (`chi`) and Betano (`chi2`, default the same)."""
    upd = "2026-10-07T15:55:00Z"
    return w.market(WED_SLOT, [ev("2026_05_CHI_GB", upd, betmgm_ca_on=chi, betano_ca_on=chi2 or chi),
                               ev("2026_05_DEN_LAC", upd, betmgm_ca_on=den)])


class TestMarket:
    def test_exact_line_match_devigs_both_prices(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True),
                        ("2026_05_DEN_LAC", 3.0, 0.55, False)])
        w.game("2026_05_CHI_GB", 20, 24)
        w.game("2026_05_DEN_LAC", 27, 20)
        _two_games_market(w)
        r = w.report()
        chi, den = game(r, "2026_05_CHI_GB"), game(r, "2026_05_DEN_LAC")
        assert chi["market_status"] == fr.MATCHED and chi["market_books"] == 2
        assert chi["market_prob_underdog"] == pytest.approx(0.5)
        # -115 dog vs -105 fav: (115/215) / (115/215 + 105/205)
        exp = (115 / 215) / (115 / 215 + 105 / 205)
        assert den["market_prob_underdog"] == pytest.approx(exp, abs=1e-6)
        mc = r["market_comparison"]
        assert mc["status"] == "available" and mc["matched_games"] == 2
        assert mc["model"]["n"] == mc["market"]["n"] == mc["baseline_50"]["n"] == 2
        assert mc["baseline_50"]["brier"] == 0.25

    def test_different_handicap_is_a_gap_not_a_substitute(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.5, 0.60, True)])          # frozen +3.5
        w.game("2026_05_CHI_GB", 20, 24)
        _two_games_market(w)                                           # market +3
        r = w.report()
        g = game(r, "2026_05_CHI_GB")
        assert g["market_status"] == fr.NO_EXACT_QUOTE and g["market_prob_underdog"] is None
        assert "+3" in g["market_status_detail"] and "+3.5" in g["market_status_detail"]
        assert r["market_comparison"]["status"] == "unavailable"
        assert "spread tracker" in r["market_comparison"]["reason"]
        assert r["simulated_returns"]["archived_price"]["settled_without_archived_price"] == 1

    def test_capture_after_the_current_kickoff_is_not_used(self, w):
        # Game moved earlier, to before the market capture: the capture is postgame.
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True)])
        w.game("2026_05_CHI_GB", 20, 24, day="2026-10-07", at="11:00")   # 15:00Z < 16:00Z
        _two_games_market(w)
        g = game(w.report(), "2026_05_CHI_GB")
        assert g["market_status"] == fr.NO_CAPTURE

    def test_stale_quote_is_not_used(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True)])
        w.game("2026_05_CHI_GB", 20, 24)
        old = "2026-10-07T13:00:00Z"                                   # 3 h before capture
        w.market(WED_SLOT, [ev("2026_05_CHI_GB", old, betmgm_ca_on=(-3.0, -110, 3.0, -110))])
        # Stale quotes are filtered out with non-Ontario feeds, before matching.
        assert game(w.report(), "2026_05_CHI_GB")["market_status"] == fr.NO_ONTARIO_QUOTE

    def test_us_only_capture_is_not_a_market_baseline(self, w):
        # Only FanDuel US (the reference feed) quotes the exact line.
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True)])
        w.game("2026_05_CHI_GB", 20, 24)
        w.market(WED_SLOT, [ev("2026_05_CHI_GB", "2026-10-07T15:55:00Z",
                               fanduel=(-3.0, -110, 3.0, -110))])
        r = w.report()
        g = game(r, "2026_05_CHI_GB")
        assert g["market_status"] == fr.NO_ONTARIO_QUOTE
        assert g["market_prob_underdog"] is None and g["archived_price"] is None
        assert r["market_comparison"]["status"] == "unavailable"
        assert r["simulated_returns"]["archived_price"]["bets"] == 0

    def test_mixed_capture_uses_only_ontario_prices(self, w):
        # Ontario: even money. FanDuel US: a materially different price that
        # would move both the devigged probability and the lowest-paying price.
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True)])
        w.game("2026_05_CHI_GB", 20, 24)                              # CHI +3 wins
        w.market(WED_SLOT, [ev("2026_05_CHI_GB", "2026-10-07T15:55:00Z",
                               betmgm_ca_on=(-3.0, -110, 3.0, -110),
                               fanduel=(-3.0, 130, 3.0, -160))])
        r = w.report()
        g = game(r, "2026_05_CHI_GB")
        assert g["market_status"] == fr.MATCHED and g["market_books"] == 1
        assert g["market_prob_underdog"] == pytest.approx(0.5)          # not pulled by US -160
        assert (g["archived_price"], g["archived_price_book"]) == (-110, "betmgm_ca_on")
        assert "fanduel" not in g["market_status_detail"]
        assert r["simulated_returns"]["archived_price"]["net_units"] == "90.91"   # not at -160

    def test_us_line_is_not_collected_as_a_quoted_point(self, w):
        # Ontario quotes +2.5 only; the US feed has the exact +3. Still a gap,
        # and the detail lists only the Ontario point.
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True)])
        w.game("2026_05_CHI_GB", 20, 24)
        w.market(WED_SLOT, [ev("2026_05_CHI_GB", "2026-10-07T15:55:00Z",
                               betmgm_ca_on=(-2.5, -110, 2.5, -110),
                               fanduel=(-3.0, -110, 3.0, -110))])
        g = game(w.report(), "2026_05_CHI_GB")
        assert g["market_status"] == fr.NO_EXACT_QUOTE
        assert "[+2.5]" in g["market_status_detail"]

    @pytest.mark.parametrize("change,ok", [
        ({}, True),
        ({"book_key": "fanduel", "jurisdiction": "US", "role": "us_reference"}, False),
        ({"book_key": "fanduel"}, False),                 # registry says US, whatever is stored
        ({"jurisdiction": "US"}, False),                  # stored metadata disagrees
        ({"role": "us_reference"}, False),
        ({"role": "manual_ontario"}, False),
        ({"source": "manual"}, False),
        ({"book_key": "unknown_book"}, False),
    ])
    def test_ontario_feed_filter(self, change, ok):
        q = {"book_key": "betmgm_ca_on", "jurisdiction": "CA-ON", "role": "ontario",
             "source": "the_odds_api", **change}
        assert fr.is_ontario_feed(q) is ok

    def test_no_capture_at_all(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True)])
        w.game("2026_05_CHI_GB", 20, 24)
        r = w.report()
        assert game(r, "2026_05_CHI_GB")["market_status"] == fr.NO_CAPTURE
        assert r["market_comparison"]["status"] == "unavailable"


# -------------------------------------------------------- simulated returns --

class TestSimulatedReturns:
    def test_archived_and_assumed_prices_are_separate(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True),          # archived -110/-125
                        ("2026_05_DEN_LAC", 3.5, 0.58, True),         # no exact quote
                        ("2026_05_BUF_LA", 3.0, 0.40, False)])        # not a signal
        w.game("2026_05_CHI_GB", 20, 24)                              # CHI +3 wins
        w.game("2026_05_DEN_LAC", 20, 24)                             # DEN +3.5 wins
        w.game("2026_05_BUF_LA", 20, 24)
        _two_games_market(w, chi=(-3.0, 105, 3.0, -125))
        s = w.report()["simulated_returns"]
        a, b = s["archived_price"], s["assumed_minus_110"]
        # Archived: only CHI, at its lowest-paying matched price (-125 at both books).
        assert (a["bets"], a["wins"], a["net_units"], a["staked_units"]) == (1, 1, "80.00", "100.00")
        assert a["settled_without_archived_price"] == 1 and a["roi"] == pytest.approx(0.8)
        # Assumed -110: both signals, labelled as a scenario.
        assert (b["bets"], b["net_units"]) == (2, "181.82")
        assert "SCENARIO" in b["label"] and b["settled_without_archived_price"] is None

    def test_archived_price_is_the_lowest_paying_matched_quote(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True)])
        w.game("2026_05_CHI_GB", 20, 24)                              # CHI +3 wins
        _two_games_market(w, chi=(-3.0, 105, 3.0, -125), chi2=(-3.0, -110, 3.0, -110))
        r = w.report()
        g = game(r, "2026_05_CHI_GB")
        assert (g["archived_price"], g["archived_price_book"]) == (-125, "betmgm_ca_on")
        assert g["sim_profit_archived"] == "80.00"                    # not 90.91 at -110
        # The market probability still uses both books' devigged prices.
        exp = ((125 / 225) / (125 / 225 + 100 / 205) + 0.5) / 2
        assert g["market_prob_underdog"] == pytest.approx(exp, abs=1e-6)

    def test_push_and_loss_and_pending_signals(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True),
                        ("2026_05_DEN_LAC", 3.0, 0.58, True),
                        ("2026_05_BUF_LA", 3.0, 0.58, True)])
        w.game("2026_05_CHI_GB", 23, 20)        # push
        w.game("2026_05_DEN_LAC", 30, 20)       # loss
        w.game("2026_05_BUF_LA")                # pending
        b = w.report()["simulated_returns"]["assumed_minus_110"]
        assert (b["bets"], b["pushes"], b["losses"], b["net_units"]) == (2, 1, 1, "-100.00")
        assert b["staked_units"] == "200.00" and b["roi"] == pytest.approx(-0.5)
        assert b["signals_not_settled"] == 1

    def test_actual_wagers_are_never_read(self, w, monkeypatch):
        monkeypatch.setattr(bj, "load_records", lambda *a, **k: pytest.fail("journal read"))
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True)])
        w.game("2026_05_CHI_GB", 20, 24)
        w.report()


# --------------------------------------------------- inputs and outputs --

_READ_CHILD = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import importlib.util
spec = importlib.util.spec_from_file_location("fr", sys.argv[1] + "/scripts/frozen_spread_report.py")
fr = importlib.util.module_from_spec(spec); sys.modules["fr"] = fr; spec.loader.exec_module(fr)
df = fr.read_games_csv(sys.argv[2])
print(json.dumps({"rev": df["code_revision"].tolist(), "run": df["run_id"].tolist(),
                  "prob": str(df["prob_underdog_covers"].dtype)}))
"""


class TestInputsAndOutputs:
    def test_identifier_columns_round_trip_as_text(self, w, tmp_path):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True)], revision=DANGEROUS_SHA)
        w.game("2026_05_CHI_GB", 20, 24)
        out = tmp_path / "out"
        _, csv_path = fr.write_report(w.report(), out)
        r = subprocess.run([sys.executable, "-c", _READ_CHILD, str(ROOT), str(csv_path)],
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-500:]                    # no parser crash
        got = json.loads(r.stdout)
        assert got["rev"] == [DANGEROUS_SHA] and got["prob"] == "float64"

    def test_corrupt_snapshot_fails_visibly(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True)])
        (w.snapshots / "20261008T000000Z-0000000000ff.json").write_text("{not json")
        w.game("2026_05_CHI_GB", 20, 24)
        with pytest.raises(fr.ReportError, match="0000000000ff"):
            w.report()

    def test_tampered_snapshot_fails_visibly(self, w):
        path = w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True)])
        doc = json.loads(path.read_text())
        doc["games"][0]["prob_underdog_covers"] = 0.99                # checksum now wrong
        path.write_text(json.dumps(doc))
        w.game("2026_05_CHI_GB", 20, 24)
        with pytest.raises(fr.ReportError, match="payload_sha256"):
            w.report()

    def test_corrupt_market_capture_fails_visibly(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True)])
        w.game("2026_05_CHI_GB", 20, 24)
        w.captures.mkdir(parents=True, exist_ok=True)
        (w.captures / "20261007T160000Z-bad.json").write_text("[]")
        with pytest.raises(fr.ReportError, match="20261007T160000Z-bad"):
            w.report()

    def test_missing_or_incomplete_schedule_fails(self, w, tmp_path):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True)])
        with pytest.raises(fr.ReportError, match="can't be read"):
            fr.build_report(w.snapshots, tmp_path / "nope.csv", w.captures, as_of=AS_OF)
        bad = tmp_path / "bad.csv"
        pd.DataFrame([{"game_id": "x"}]).to_csv(bad, sep="\t", index=False)
        with pytest.raises(fr.ReportError, match="lacks column"):
            fr.build_report(w.snapshots, bad, w.captures, as_of=AS_OF)

    def test_missing_snapshot_directory_fails(self, w, tmp_path):
        w.game("2026_05_CHI_GB", 20, 24)
        w.write_schedule()
        with pytest.raises(fr.ReportError, match="not found"):
            fr.build_report(tmp_path / "none", w.schedule, w.captures, as_of=AS_OF)

    def test_report_is_deterministic_and_json_safe(self, w, tmp_path):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True), ("2026_05_PHI_JAX", None, None, False)])
        w.game("2026_05_CHI_GB", 20, 24)
        a, b = tmp_path / "a", tmp_path / "b"
        fr.write_report(w.report(), a)
        fr.write_report(w.report(), b)
        for name in (fr.JSON_NAME, fr.CSV_NAME):
            assert (a / name).read_bytes() == (b / name).read_bytes()
        json.loads((a / fr.JSON_NAME).read_text(), parse_constant=lambda c: pytest.fail(c))

    def test_sources_are_never_modified(self, w, tmp_path):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True)])
        w.game("2026_05_CHI_GB", 20, 24)
        _two_games_market(w)
        w.write_schedule()
        before = {p: p.read_bytes() for d in (w.snapshots, w.captures, w.schedule.parent)
                  for p in Path(d).rglob("*") if p.is_file()}
        fr.write_report(w.report(), tmp_path / "out")
        after = {p: p.read_bytes() for d in (w.snapshots, w.captures, w.schedule.parent)
                 for p in Path(d).rglob("*") if p.is_file()}
        assert before == after


class TestProtectedOutput:
    def _protected(self, w):
        return fr.protected_roots(w.snapshots, w.schedule, w.captures)

    @pytest.mark.parametrize("where", ["snapshots", "snap_child", "captures", "schedule_dir",
                                       "parent_of_sources", "data_files", "alias"])
    def test_refused(self, w, where):
        report = {"games": []}
        target = {"snapshots": w.snapshots, "snap_child": w.snapshots / "out",
                  "captures": w.captures, "schedule_dir": w.schedule.parent,
                  "parent_of_sources": w.tmp,
                  "data_files": ROOT / "data_files" / "frozen_out",
                  "alias": w.tmp / "sched" / ".." / "pregame_snapshots" / "x"}[where]
        if where == "alias" and os.name == "nt":
            target = Path(str(target).upper())                       # case-insensitive alias
        with pytest.raises(fr.ReportError, match="refusing"):
            fr.write_report(report, target, self._protected(w))
        assert not (ROOT / "data_files" / "frozen_out").exists()

    def test_symlink_into_protected_data_is_refused(self, w, tmp_path):
        link = tmp_path / "link_out"
        try:
            os.symlink(w.snapshots, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not permitted here")
        with pytest.raises(fr.ReportError, match="refusing"):
            fr.write_report({"games": []}, link, self._protected(w))

    def test_cli_writes_only_to_the_output_directory(self, w, tmp_path, capsys):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.60, True)])
        w.game("2026_05_CHI_GB", 20, 24)
        w.write_schedule()
        out = tmp_path / "reports" / "frozen"
        code = fr.main(["--snapshot-dir", str(w.snapshots), "--schedule", str(w.schedule),
                        "--capture-dir", str(w.captures), "--as-of", "2026-10-14",
                        "--output-dir", str(out)])
        assert code == 0 and sorted(p.name for p in out.iterdir()) == sorted(
            [fr.JSON_NAME, fr.CSV_NAME])
        assert fr.main(["--snapshot-dir", str(w.snapshots), "--schedule", str(w.schedule),
                        "--capture-dir", str(w.captures), "--output-dir", str(w.snapshots)]) == 1
        assert "refusing" in capsys.readouterr().out

    def test_default_output_directory_is_git_ignored(self):
        r = subprocess.run(["git", "check-ignore", "-q", "reports/frozen_spread/x.json"],
                           cwd=ROOT, capture_output=True)
        assert r.returncode == 0
