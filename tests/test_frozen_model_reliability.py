"""Frozen-model reliability report (scripts/frozen_model_reliability.py).

Pure-function checks on hand-computed synthetic games, plus end-to-end runs
on temporary snapshots, schedules and Ontario captures built with the frozen
spread report's own fixtures. Network access is blocked; nothing in
data_files/ is read or written.
"""
import hashlib
import importlib.util
import json
import math
import os
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from test_frozen_spread_report import (AS_OF, DANGEROUS_SHA, S1, S2, World, fr, ev, game)
from test_ontario_spreads import WED_SLOT
from test_team_features import ROOT

spec = importlib.util.spec_from_file_location("frozen_model_reliability",
                                              ROOT / "scripts" / "frozen_model_reliability.py")
rel = importlib.util.module_from_spec(spec)
sys.modules["frozen_model_reliability"] = rel
spec.loader.exec_module(rel)

UTC = timezone.utc
GEN = datetime(2026, 10, 21, 14, 0, tzinfo=UTC)
UPD = "2026-10-07T15:55:00Z"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)


@pytest.fixture(autouse=True)
def no_journal(monkeypatch):
    import bet_journal as bj

    def refuse(*a, **k):
        raise AssertionError("bet journal accessed")
    for name in ("load_records", "current_state", "save_wager", "grade"):
        monkeypatch.setattr(bj, name, refuse)


@pytest.fixture
def w(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def G(gid, p, y, week=5, season=2026, signal=False, market=None, status="evaluated", books=1):
    """A per-game row shaped like frozen_spread_report's."""
    return {"game_id": gid, "season": season, "week": week, "prob_underdog_covers": p,
            "underdog_covered": y, "status": status, "bet_signal": signal,
            "market_status": fr.MATCHED if market is not None else fr.NO_CAPTURE,
            "market_status_detail": "" if market is not None else "no archived capture",
            "market_prob_underdog": market, "market_books": books if market is not None else None}


def report(w, **kw):
    w.write_schedule()
    kw.setdefault("as_of", AS_OF)
    kw.setdefault("generated_at", GEN)
    return rel.build_report(w.snapshots, w.schedule, w.captures, **kw)


# ----------------------------------------------------------------- scores --

class TestScores:
    def test_perfect_reversed_constant(self):
        eps = rel.LOG_LOSS_EPS
        assert rel.brier(1.0, 1) == 0 and rel.brier(0.0, 0) == 0
        assert rel.brier(1.0, 0) == 1 and rel.brier(0.0, 1) == 1
        assert rel.log_loss(1.0, 1) == pytest.approx(-math.log(1 - eps))
        assert rel.log_loss(1.0, 0) == pytest.approx(-math.log(1 - (1 - eps)))  # clipped, finite
        assert math.isfinite(rel.log_loss(0.0, 1)) and rel.log_loss(0.0, 1) > 30
        assert rel.brier(0.5, 1) == rel.brier(0.5, 0) == 0.25
        assert rel.log_loss(0.5, 0) == pytest.approx(math.log(2))

    def test_constant_50_model_has_zero_difference(self):
        games = [G(f"g{i}", 0.5, i % 2, week=i % 6 + 1) for i in range(12)]
        c = rel.comparison(games, "prob_underdog_covers", 0.5, "x")
        assert c["brier"]["mean_difference"] == 0 and c["log_loss"]["mean_difference"] == 0
        assert c["brier"]["interval_95"] == [0.0, 0.0]

    def test_sign_negative_means_model_better(self):
        good = rel.paired_differences([G("a", 0.8, 1), G("b", 0.2, 0)], "prob_underdog_covers", 0.5)
        assert all(d["d_brier"] < 0 and d["d_log_loss"] < 0 for d in good)
        # hand: (0.8-1)^2 - 0.25 = 0.04 - 0.25
        assert good[0]["d_brier"] == pytest.approx(-0.21)
        assert good[0]["d_log_loss"] == pytest.approx(-math.log(0.8) - math.log(2))
        bad = rel.paired_differences([G("a", 0.2, 1)], "prob_underdog_covers", 0.5)
        assert bad[0]["d_brier"] == pytest.approx(0.64 - 0.25)

    def test_market_baseline_is_the_games_own_market_probability(self):
        d = rel.paired_differences([G("a", 0.6, 1, market=0.7)], "prob_underdog_covers",
                                   "market_prob_underdog")
        assert d[0]["d_brier"] == pytest.approx(0.16 - 0.09)


# ---------------------------------------------------------- calibration --

class TestCalibration:
    @pytest.mark.parametrize("p,i", [(0.0, 0), (0.39999, 0), (0.40, 1), (0.49999, 1), (0.5, 2),
                                     (0.6, 3), (0.69999, 3), (0.70, 4), (0.95, 4), (1.0, 4)])
    def test_bin_endpoints(self, p, i):
        assert rel.bin_index(p) == i

    @pytest.mark.parametrize("p", [-0.01, 1.01, float("nan"), None, True])
    def test_out_of_range_probability_fails(self, p):
        with pytest.raises(rel.ReportError):
            rel.bin_index(p)

    def test_labels_show_endpoint_handling(self):
        assert [rel.bin_label(i) for i in range(5)] == [
            "[0.00, 0.40)", "[0.40, 0.50)", "[0.50, 0.60)", "[0.60, 0.70)", "[0.70, 1.00]"]

    def test_empty_sparse_and_filled_bins(self):
        pairs = [(0.55, i % 2) for i in range(12)] + [(0.45, 1), (0.45, 0), (0.4, 1)]
        t = {r["bin"]: r for r in rel.calibration_table(pairs)}
        assert len(t) == 5                                            # empty bins kept
        empty = t["[0.00, 0.40)"]
        assert (empty["n"], empty["observed_cover_rate"], empty["wilson_low"]) == (0, None, None)
        assert empty["note"].startswith("empty")
        sparse = t["[0.40, 0.50)"]
        assert (sparse["n"], sparse["covers"]) == (3, 2)
        assert sparse["mean_predicted"] == pytest.approx((0.45 + 0.45 + 0.4) / 3, abs=1e-6)
        assert "descriptive only" in sparse["note"]
        full = t["[0.50, 0.60)"]
        assert (full["n"], full["observed_cover_rate"]) == (12, 0.5)
        assert "not a reliability claim" in full["note"]


def _wilson_by_inversion(k, n, z=rel.WILSON_Z):
    """Independent reference: invert the score test |phat - p| = z sqrt(p(1-p)/n)
    by bisection on each side of phat."""
    phat = k / n

    def f(p):
        return (phat - p) ** 2 - z * z * p * (1 - p) / n

    def solve(a, b):
        for _ in range(200):
            m = (a + b) / 2
            if (f(a) > 0) == (f(m) > 0):
                a = m
            else:
                b = m
        return (a + b) / 2
    lo = 0.0 if k == 0 else solve(0.0, phat)
    hi = 1.0 if k == n else solve(phat, 1.0)
    return lo, hi


class TestWilson:
    @pytest.mark.parametrize("k,n", [(0, 1), (1, 1), (6, 9), (0, 10), (10, 10), (17, 29),
                                     (3, 7), (50, 100)])
    def test_matches_score_test_inversion(self, k, n):
        lo, hi = rel.wilson(k, n)
        rlo, rhi = _wilson_by_inversion(k, n)
        assert lo == pytest.approx(rlo, abs=1e-9) and hi == pytest.approx(rhi, abs=1e-9)

    @pytest.mark.parametrize("k,n", [(6, 9), (0, 10), (10, 10), (17, 29), (3, 7)])
    def test_matches_scipy(self, k, n):
        stats = pytest.importorskip("scipy.stats")
        ci = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
        lo, hi = rel.wilson(k, n)
        assert lo == pytest.approx(ci.low, abs=1e-9) and hi == pytest.approx(ci.high, abs=1e-9)

    def test_published_value_and_empty(self):
        lo, hi = rel.wilson(6, 9)
        assert (round(lo, 4), round(hi, 4)) == (0.3542, 0.8794)
        assert rel.wilson(0, 0) is None


# ------------------------------------------------------------- bootstrap --

def _diffs(spec):
    """spec: {cluster: [d, ...]} -> paired-difference rows."""
    return [{"game_id": f"{c}-{i}", "cluster": c, "d_brier": d, "d_log_loss": d}
            for c, ds in spec.items() for i, d in enumerate(ds)]


class TestBootstrap:
    def test_insufficient_clusters_is_unavailable(self):
        out = rel.cluster_bootstrap(_diffs({"W1": [0.1] * 20, "W2": [0.2], "W3": [0.0],
                                            "W4": [-0.1]}), "d_brier")
        assert out["interval_95"] is None and out["n_clusters"] == 4
        assert out["interval_status"].startswith("unavailable: 4 week cluster(s)")
        assert out["mean_difference"] == pytest.approx((2.0 + 0.2 - 0.1) / 23, abs=1e-6)

    def test_exactly_the_minimum_clusters_is_available(self):
        spec = {f"W{i}": [0.02 * i - 0.05, 0.01] for i in range(1, 6)}
        out = rel.cluster_bootstrap(_diffs(spec), "d_brier")
        assert out["n_clusters"] == 5 and out["interval_status"] == "available (exploratory)"
        lo, hi = out["interval_95"]
        assert lo <= out["mean_difference"] <= hi

    def test_no_games(self):
        out = rel.cluster_bootstrap([], "d_brier")
        assert out["mean_difference"] is None and out["interval_status"] == "unavailable: no games"

    def test_deterministic_with_fixed_seed(self):
        d = _diffs({f"W{i}": [0.01 * i * i - 0.1] * i for i in range(1, 8)})
        a, b = rel.cluster_bootstrap(d, "d_brier"), rel.cluster_bootstrap(d, "d_brier")
        assert a["interval_95"] == b["interval_95"] and a["interval_status"].startswith("available")
        other = rel.cluster_bootstrap(d, "d_brier", seed=rel.BOOTSTRAP_SEED + 1)
        assert other["interval_95"] != a["interval_95"]
        assert (a["replicates"], a["seed"]) == (10_000, 20261005)

    def test_whole_clusters_are_resampled(self):
        # Cluster sizes 1, 2, 3, 4, 5, all differences in a cluster equal its
        # index. Every replicate is a multiset of whole clusters, so its value
        # is sum(i * size_i * c_i) / sum(size_i * c_i) for counts c_i summing to 5.
        spec = {f"W{i}": [float(i)] * i for i in range(1, 6)}
        stats = rel.bootstrap_distribution(_diffs(spec), "d_brier", replicates=2000, seed=1)
        from itertools import combinations_with_replacement
        possible = {round(sum(i * i for i in combo) / sum(combo), 9)
                    for combo in combinations_with_replacement(range(1, 6), 5)}
        assert {round(float(s), 9) for s in stats} <= possible
        # Game-level resampling would produce values outside this set.
        rng = np.random.default_rng(1)
        games = np.array([float(i) for i in range(1, 6) for _ in range(i)])
        game_level = {round(float(games[rng.integers(0, 15, 15)].mean()), 9) for _ in range(500)}
        assert not game_level <= possible

    def test_pairs_stay_together(self):
        # Model and baseline scores differ wildly between games, but each
        # game's paired difference is exactly 0.1: every replicate is 0.1.
        games = [G(f"g{i}", p, 1, week=i % 6) for i, p in enumerate([0.9, 0.6, 0.3] * 4)]
        diffs = rel.paired_differences(games, "prob_underdog_covers", "prob_underdog_covers")
        for d in diffs:
            d["d_brier"] += 0.1
        out = rel.cluster_bootstrap(diffs, "d_brier")
        assert out["interval_95"] == [0.1, 0.1]

    def test_clusters_are_season_and_week(self):
        diffs = rel.paired_differences([G("a", 0.6, 1, week=5, season=2026),
                                        G("b", 0.6, 1, week=5, season=2027)],
                                       "prob_underdog_covers", 0.5)
        assert {d["cluster"] for d in diffs} == {"2026-W05", "2027-W05"}


# ---------------------------------------------------------------- samples --

class TestSamples:
    GAMES = [G("a", 0.6, 1, market=0.55, signal=True, books=3), G("b", 0.6, 0, market=0.5),
             G("c", 0.3, 0), G("d", 0.7, 1, signal=True),
             G("p", 0.6, None, status=fr.PUSH), G("q", 0.6, None, status=fr.PENDING),
             G("n", None, None, status=fr.ps.NO_LINE),
             G("t", 0.6, None, status=fr.TIMING_UNVERIFIED)]

    def test_overall_vs_matched_samples_differ(self):
        o, m = rel.overall_sample(self.GAMES), rel.matched_sample(self.GAMES)
        assert o["n_games"] == 4 and m["n_games"] == 2
        # hand: overall Brier = (0.16 + 0.36 + 0.09 + 0.09) / 4
        assert o["model"]["brier"] == pytest.approx(0.175)
        # matched: model (0.16 + 0.36) / 2; market (0.2025 + 0.25) / 2
        assert m["model"]["brier"] == pytest.approx(0.26)
        assert m["market"]["brier"] == pytest.approx(0.22625)
        assert m["baseline_50"]["brier"] == 0.25
        assert m["evaluated_without_market"] == 2
        assert m["unavailable_market_by_reason"][fr.NO_CAPTURE]["n"] == 2

    def test_multiple_books_count_once(self):
        m = rel.matched_sample(self.GAMES)
        assert m["n_games"] == 2 and m["books_per_game"]["max"] == 3
        assert m["model"]["n"] == m["market"]["n"] == 2

    def test_pushes_pending_and_exclusions_counted_not_scored(self):
        c = rel.status_counts(self.GAMES)
        assert (c["evaluated"], c["pushes"], c["pending"], c["excluded_total"]) == (4, 1, 1, 2)
        assert c["excluded"][fr.ps.NO_LINE] == 1 and c["excluded"][fr.TIMING_UNVERIFIED] == 1

    def test_signal_subset_is_separate(self):
        v = rel.view(self.GAMES)
        assert v["signal_only"]["overall"]["n_games"] == 2
        assert v["signal_only"]["matched_market"]["n_games"] == 1
        assert v["all_predictions"]["overall"]["n_games"] == 4

    def test_cover_rate_and_mean_probability_are_probabilities(self):
        o = rel.overall_sample(self.GAMES)["model"]
        assert o["observed_cover_rate"] == 0.5 and o["mean_probability"] == pytest.approx(0.55)
        assert "accuracy" not in json.dumps(rel.overall_sample(self.GAMES))

    def test_by_week(self):
        games = [G("a", 0.6, 1, week=5), G("b", 0.4, 1, week=6, market=0.5),
                 G("c", 0.6, None, week=6, status=fr.PENDING)]
        wk = rel.by_week(games)
        assert list(wk) == ["2026-W05", "2026-W06"]
        assert wk["2026-W06"]["counts"]["pending"] == 1 and wk["2026-W06"]["matched_n"] == 1
        assert wk["2026-W05"]["overall_model"]["brier"] == pytest.approx(0.16)


# ---------------------------------------------------- end-to-end (fixtures) --

class TestEndToEnd:
    def _world(self, w):
        # CHI +3 at GB: GB wins by 4 -> CHI loses. DEN +3 at LAC: LAC by 1 -> DEN covers.
        # BUF +3 at LA: LA by 3 -> push. TB at DAL: pending. PHI_JAX: no line.
        w.snapshot(S1, [("2026_05_CHI_GB", 3.0, 0.30, False)])
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.62, True), ("2026_05_DEN_LAC", 3.0, 0.55, False),
                        ("2026_05_BUF_LA", 3.0, 0.50, False), ("2026_05_PHI_JAX", None, None, False)])
        w.game("2026_05_CHI_GB", 24, 20)
        w.game("2026_05_DEN_LAC", 21, 20)
        w.game("2026_05_BUF_LA", 23, 20)
        w.game("2026_05_PHI_JAX", 20, 17)
        w.market(WED_SLOT, [ev("2026_05_CHI_GB", UPD, betmgm_ca_on=(-3.0, -110, 3.0, -110),
                               betano_ca_on=(-3.0, -120, 3.0, 100),
                               fanduel=(-3.0, -200, 3.0, 170))])

    def test_hand_computed_report(self, w):
        self._world(w)
        r = report(w)
        a = r["latest"]["all_predictions"]
        assert a["counts"]["evaluated"] == 2 and a["counts"]["pushes"] == 1
        assert a["counts"]["excluded"][fr.ps.NO_LINE] == 1
        o = a["overall"]
        # latest snapshot (0.62), not the earlier 0.30: CHI lost -> 0.3844; DEN covered -> 0.2025
        assert o["model"]["brier"] == pytest.approx((0.62 ** 2 + 0.45 ** 2) / 2, abs=1e-6)
        m = a["matched_market"]
        assert m["n_games"] == 1 and m["books_per_game"]["max"] == 2     # US reference excluded
        # market: mean of devigged BetMGM (0.5) and Betano (100 vs -120)
        betano = (100 / 200) / (100 / 200 + 120 / 220)
        assert m["market"]["mean_probability"] == pytest.approx((0.5 + betano) / 2, abs=1e-6)
        # DEN @ LAC is in the capture but unquoted: shown with its reason, not dropped.
        [(reason, info)] = m["unavailable_market_by_reason"].items()
        assert reason == fr.NO_ONTARIO_QUOTE and info["games"][0]["game_id"] == "2026_05_DEN_LAC"
        assert m["evaluated_without_market"] == 1
        assert a["overall"]["model_minus_50"]["brier"]["interval_status"].startswith("unavailable")
        assert r["latest"]["signal_only"]["overall"]["n_games"] == 1

    def test_matches_the_frozen_spread_report(self, w):
        self._world(w)
        r = report(w)
        base = w.report()
        assert r["latest"]["all_predictions"]["overall"]["model"] == \
            base["probability_metrics"]["all_valid_probabilities"]["overall"]
        assert r["latest"]["signal_only"]["overall"]["model"] == \
            base["probability_metrics"]["signal_only"]["overall"]
        assert r["latest"]["all_predictions"]["matched_market"]["model"] == \
            base["market_comparison"]["model"]
        assert r["latest"]["all_predictions"]["matched_market"]["market"] == \
            base["market_comparison"]["market"]
        for k, v in base["probability_metrics"]["all_valid_probabilities"]["by_season_week"].items():
            assert r["latest"]["all_predictions"]["by_week"][k]["overall_model"] == v

    def test_snapshot_first_selection_without_fallback(self, w):
        w.snapshot(S1, [("2026_05_CHI_GB", 3.0, 0.70, True)])
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, None, False)])
        w.game("2026_05_CHI_GB", 24, 20)
        a = report(w)["latest"]["all_predictions"]
        assert a["overall"]["n_games"] == 0 and a["counts"]["excluded"][fr.ps.NO_PROBABILITY] == 1

    def test_timing_unverified_excluded(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, False)])
        w.game("2026_05_CHI_GB", 24, 20, day="2026-10-07", at="04:00")
        a = report(w)["latest"]["all_predictions"]
        assert a["counts"]["excluded"][fr.TIMING_UNVERIFIED] == 1 and a["overall"]["n_games"] == 0

    def test_pending_is_not_scored(self, w):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.6, False)])
        w.game("2026_05_CHI_GB")
        a = report(w)["latest"]["all_predictions"]
        assert a["counts"]["pending"] == 1 and a["overall"]["model"]["n"] == 0

    def test_earliest_sensitivity_is_separate_and_never_pooled(self, w):
        self._world(w)
        plain, both = report(w), report(w, earliest_sensitivity=True)
        assert plain["earliest_sensitivity"] is None
        assert both["latest"] == plain["latest"]                      # unchanged by the flag
        e = both["earliest_sensitivity"]
        assert e["label"].startswith("SENSITIVITY ONLY")
        chi = (0.30 - 0) ** 2                                          # earliest CHI prob 0.30
        assert e["all_predictions"]["overall"]["model"]["brier"] == pytest.approx(
            (chi + 0.45 ** 2) / 2, abs=1e-6)
        text = json.dumps(both).lower()
        assert "best" not in text and "preferred" not in text

    def test_provenance_and_settings(self, w):
        self._world(w)
        r = report(w)
        assert r["protocol_version"] == 1 and r["generated_at_utc"] == "2026-10-21T14:00:00Z"
        assert r["as_of_date"] == AS_OF.isoformat() and "not a historical" in r["as_of_semantics"]
        fp = r["sources"]["fingerprints"]
        assert len(fp["pregame_snapshots"]) == 2 and len(fp["ontario_captures"]) == 1
        name = next(iter(fp["pregame_snapshots"]))
        assert fp["pregame_snapshots"][name] == hashlib.sha256(
            (w.snapshots / name).read_bytes()).hexdigest()
        assert r["settings"]["bin_edges"] == [0.0, 0.4, 0.5, 0.6, 0.7, 1.0]
        assert r["sources"]["bet_journal"] == "never read"
        assert any("untouched evaluation period" in x for x in r["limitations"])
        assert r["tracking_start"] == "2026-10-05"

    def test_protocol_statements(self, w):
        # Fixed for SUBSEQUENT evaluation (not prospectively registered);
        # Wilson bars are binomial and ignore dependence; 5 weeks only permits
        # an exploratory interval.
        self._world(w)
        r = report(w)
        p = r["protocol"]
        assert "subsequent evaluation" in p["status"] and "exploratory" in p["status"]
        assert "not prospectively registered" in p["status"]
        assert "does not account for dependence" in p["calibration_bins"]["interval"]
        assert "distinct from the week-cluster bootstrap" in p["calibration_bins"]["interval"]
        assert "does not establish that it is reliable" in \
            p["paired_differences"]["minimum_clusters"]
        assert any("binomial reference intervals" in x for x in r["limitations"])
        assert "before looking at any result" not in json.dumps(r).lower()

    def test_sources_changing_mid_run_fail(self, w, monkeypatch):
        self._world(w)
        real = fr.build_report

        def build_and_touch(*a, **k):
            out = real(*a, **k)
            (w.snapshots / "zz_extra.json").write_text("{}")
            return out
        monkeypatch.setattr(fr, "build_report", build_and_touch)
        with pytest.raises(rel.ReportError, match="changed while the report was running"):
            report(w)

    def test_corrupt_inputs_fail_and_write_nothing(self, w, tmp_path, capsys):
        self._world(w)
        w.write_schedule()
        bad = sorted(w.snapshots.glob("*.json"))[-1]                   # the S2 snapshot
        assert "0.62" in bad.read_text()
        bad.write_text(bad.read_text().replace("0.62", "0.99"))       # checksum now wrong
        out = tmp_path / "out"
        code = rel.main(["--snapshot-dir", str(w.snapshots), "--schedule", str(w.schedule),
                         "--capture-dir", str(w.captures), "--as-of", "2026-10-14",
                         "--output-dir", str(out)])
        assert code == 1 and "failed" in capsys.readouterr().out and not out.exists()

    def test_corrupt_market_capture_fails(self, w):
        self._world(w)
        cap = next(w.captures.glob("*.json"))
        cap.write_text("{not json")
        with pytest.raises(rel.ReportError, match="market capture"):
            report(w)


# ------------------------------------------------------------ outputs --

class TestOutputs:
    def _run(self, w, tmp_path, *extra):
        w.write_schedule()
        out = tmp_path / "reports" / "rel"
        code = rel.main(["--snapshot-dir", str(w.snapshots), "--schedule", str(w.schedule),
                         "--capture-dir", str(w.captures), "--as-of", "2026-10-14",
                         "--output-dir", str(out), *extra])
        return code, out

    def test_outputs_and_text_columns(self, w, tmp_path):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.62, True)], revision=DANGEROUS_SHA)
        w.game("2026_05_CHI_GB", 24, 20)
        code, out = self._run(w, tmp_path)
        assert code == 0
        assert sorted(p.name for p in out.iterdir()) == sorted(
            [rel.JSON_NAME, rel.GAMES_CSV, rel.CALIBRATION_CSV, rel.PLOT_NAME])
        df = rel.read_games_csv(out / rel.GAMES_CSV)
        assert df.loc[0, "code_revision"] == DANGEROUS_SHA and isinstance(df.loc[0, "run_id"], str)
        assert df.loc[0, "brier_model"] == pytest.approx(0.3844)
        assert df.loc[0, "calibration_bin"] == "[0.60, 0.70)"
        cal = (out / rel.CALIBRATION_CSV).read_text().splitlines()
        assert cal[0].startswith("view,subset,sample,probability,bin")
        assert len(cal) == 1 + 2 * 3 * 5                    # 2 subsets x 3 tables x 5 bins

    def test_svg_plot(self, w, tmp_path):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.62, True), ("2026_05_DEN_LAC", 3.0, 0.45, False)])
        w.game("2026_05_CHI_GB", 24, 20)
        w.game("2026_05_DEN_LAC", 21, 20)
        _, out = self._run(w, tmp_path)
        svg = (out / rel.PLOT_NAME).read_text(encoding="utf-8")
        root = ET.fromstring(svg)                                    # well-formed
        ns = "{http://www.w3.org/2000/svg}"
        assert len(root.findall(f"{ns}circle")) == 2                 # two non-empty bins
        assert "perfect calibration" in svg and svg.count(">n=0<") == 3
        assert "n=1" in svg and "Wilson" in svg and "Not a reliability" in svg

    def test_svg_footer_is_wrapped_and_inside_the_canvas(self):
        # Regression: the one-line footer ran past the 560 px canvas and was clipped.
        assert " ".join(rel.PLOT_FOOTER) == (
            "Dashed: perfect calibration. Red: fewer than 10 games (descriptive only). "
            "Wilson bars ignore dependence. Not a reliability claim.")       # wording kept
        svg = rel.calibration_svg(rel.calibration_table([(0.55, 1)]), "t")
        root = ET.fromstring(svg)
        ns = "{http://www.w3.org/2000/svg}"
        width, height = (float(v) for v in root.get("viewBox").split()[2:])
        footer = [t for t in root.iter(f"{ns}text") if t.get("class") == "footer"]
        assert [t.text for t in footer] == list(rel.PLOT_FOOTER)
        # 11 px sans-serif averages well under 0.6 em per character, so each
        # line fits with margin; the browser check measures the real boxes.
        assert all(len(t.text) * 11 * 0.6 < width - 20 for t in footer)
        ys = [float(t.get("y")) for t in root.iter(f"{ns}text") if t.get("y")]
        assert max(ys) < height and len(set(float(t.get("y")) for t in footer)) == 2

    def test_long_title_wraps_without_changing_words(self):
        title = ("Frozen model calibration: all eligible predictions (n=1234, latest snapshot, "
                 "as of 2026-12-31)")
        lines = rel.title_lines(title)
        assert len(lines) == 2 and " ".join(lines) == title and lines[1].startswith("(n=1234")
        assert rel.title_lines("short title") == ["short title"]
        svg = rel.calibration_svg(rel.calibration_table([]), title)
        ns = "{http://www.w3.org/2000/svg}"
        shown = [t.text for t in ET.fromstring(svg).iter(f"{ns}text") if t.get("class") == "title"]
        assert shown == lines

    def test_count_label_stays_inside_near_the_right_edge(self):
        svg = rel.calibration_svg(rel.calibration_table([(0.98, 1), (0.05, 0)]), "t")
        ns = "{http://www.w3.org/2000/svg}"
        anchors = sorted(t.get("text-anchor") for t in ET.fromstring(svg).iter(f"{ns}text")
                         if t.text == "n=1")
        assert anchors == ["end", "start"]                  # right-edge point labelled leftwards

    def test_empty_plot_renders(self):
        svg = rel.calibration_svg(rel.calibration_table([]), "empty")
        root = ET.fromstring(svg)
        ns = "{http://www.w3.org/2000/svg}"
        assert root.findall(f"{ns}circle") == [] and svg.count(">n=0<") == 5
        assert all(rel.PLOT_FOOTER[i] in svg for i in range(2))

    def test_sources_unchanged(self, w, tmp_path):
        w.snapshot(S2, [("2026_05_CHI_GB", 3.0, 0.62, True)])
        w.game("2026_05_CHI_GB", 24, 20)
        w.market(WED_SLOT, [ev("2026_05_CHI_GB", UPD, betmgm_ca_on=(-3.0, -110, 3.0, -110))])
        w.write_schedule()

        def digest():
            return {p: hashlib.sha256(p.read_bytes()).hexdigest()
                    for d in (w.snapshots, w.captures, w.schedule.parent) for p in d.rglob("*")
                    if p.is_file()}
        before = digest()
        code, _ = self._run(w, tmp_path, "--earliest-sensitivity")
        assert code == 0 and digest() == before

    @pytest.mark.parametrize("where", ["snapshots", "snap_child", "captures", "schedule_dir",
                                       "parent_of_sources", "data_files", "alias"])
    def test_protected_outputs_refused(self, w, where):
        target = {"snapshots": w.snapshots, "snap_child": w.snapshots / "out",
                  "captures": w.captures, "schedule_dir": w.schedule.parent,
                  "parent_of_sources": w.tmp,
                  "data_files": ROOT / "data_files" / "reliability_out",
                  "alias": w.tmp / "sched" / ".." / "pregame_snapshots" / "x"}[where]
        if where == "alias" and os.name == "nt":
            target = Path(str(target).upper())
        with pytest.raises(rel.ReportError, match="refusing"):
            rel.write_outputs({}, target, fr.protected_roots(w.snapshots, w.schedule, w.captures))
        assert not (ROOT / "data_files" / "reliability_out").exists()

    def test_symlink_refused(self, w, tmp_path):
        link = tmp_path / "link_out"
        try:
            os.symlink(w.snapshots, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not permitted here")
        with pytest.raises(rel.ReportError, match="refusing"):
            rel.write_outputs({}, link, fr.protected_roots(w.snapshots, w.schedule, w.captures))

    @pytest.mark.skipif(os.name != "nt", reason="directory junctions are Windows-only")
    def test_junction_refused(self, w, tmp_path):
        link = tmp_path / "junction_out"
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(w.snapshots)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip(f"could not create a junction: {r.stdout}{r.stderr}")
        with pytest.raises(rel.ReportError, match="refusing"):
            rel.write_outputs({}, link, fr.protected_roots(w.snapshots, w.schedule, w.captures))
        assert sorted(p.name for p in w.snapshots.iterdir()) == []

    def test_default_output_directory_is_git_ignored(self):
        r = subprocess.run(["git", "check-ignore", "-q", "reports/frozen_reliability/x.json"],
                           cwd=ROOT, capture_output=True)
        assert r.returncode == 0
