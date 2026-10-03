"""Tests for betting_log - headless management of betting_recommendations_log.csv."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

import betting_log as bl


@pytest.mark.parametrize("prob,tier", [
    (0.49, "Lean"), (0.50, "Lean"), (0.549, "Lean"),
    (0.55, "Good"), (0.589, "Good"),
    (0.59, "Strong"), (0.649, "Strong"),
    (0.65, "Elite"), (0.90, "Elite"),
])
def test_spread_tier_cutoffs(prob, tier):
    assert bl._spread_tier(prob) == tier


def _preds(rows):
    """Minimal predictions frame with the columns grade_pending / append need."""
    cols = {
        "game_id": [], "season": [], "week": [], "gameday": [],
        "home_team": [], "away_team": [], "spread_line": [], "total_line": [],
        "home_score": [], "away_score": [], "underdogCovered": [], "spreadPush": [],
        "pred_spreadCovered_optimal": [], "prob_underdogCovered": [],
        "edge_underdog_spread": [],
    }
    for r in rows:
        for k in cols:
            cols[k].append(r.get(k, 0))
    return pd.DataFrame(cols)


class TestGradePending:
    def _log(self, tmp_path, game_id, bet_type="spread"):
        p = tmp_path / "log.csv"
        row = {c: "" for c in bl.LOG_COLUMNS}
        row.update(game_id=game_id, bet_type=bet_type, spread_line=3.0,
                   bet_result="pending", week=1, home_team="B", away_team="A", recommended_team="A")
        pd.DataFrame([row], columns=bl.LOG_COLUMNS).to_csv(p, index=False)
        return str(p)

    def test_underdog_cover_is_a_win(self, tmp_path):
        log = self._log(tmp_path, "2025_18_A_B")
        preds = _preds([dict(game_id="2025_18_A_B", gameday="2025-01-01",
                             home_score=16, away_score=14, underdogCovered=1,
                             spreadPush=0)])
        assert bl.grade_pending(preds, log) == 1
        out = pd.read_csv(log).iloc[0]
        assert out.bet_result == "win"
        assert out.bet_profit == pytest.approx(bl.WIN_PROFIT)
        assert out.actual_home_score == 16 and out.actual_away_score == 14

    def test_favorite_cover_is_a_loss(self, tmp_path):
        log = self._log(tmp_path, "2025_18_A_B")
        preds = _preds([dict(game_id="2025_18_A_B", gameday="2025-01-01",
                             home_score=30, away_score=3, underdogCovered=0,
                             spreadPush=0)])
        assert bl.grade_pending(preds, log) == 1
        out = pd.read_csv(log).iloc[0]
        assert out.bet_result == "loss"
        assert out.bet_profit == pytest.approx(bl.LOSS_PROFIT)

    def test_push_pays_zero(self, tmp_path):
        log = self._log(tmp_path, "2025_18_A_B")
        preds = _preds([dict(game_id="2025_18_A_B", gameday="2025-01-01",
                             home_score=23, away_score=20, underdogCovered=0,
                             spreadPush=1)])
        assert bl.grade_pending(preds, log) == 1
        out = pd.read_csv(log).iloc[0]
        assert out.bet_result == "push"
        assert out.bet_profit == 0.0

    def test_unplayed_zero_zero_game_is_not_graded(self, tmp_path):
        # nfl-gather-data.py fillna(0)s unplayed rows - a 0-0 "final" in the
        # future must stay pending, not be scored as a loss.
        log = self._log(tmp_path, "2026_01_A_B")
        future = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        preds = _preds([dict(game_id="2026_01_A_B", gameday=future,
                             home_score=0, away_score=0, underdogCovered=0,
                             spreadPush=0)])
        assert bl.grade_pending(preds, log) == 0
        assert pd.read_csv(log).iloc[0].bet_result == "pending"


class TestAppendRecommendations:
    def test_logs_only_near_term_signals_and_dedupes(self, tmp_path):
        log = str(tmp_path / "log.csv")
        soon = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
        far = (datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d")
        preds = _preds([
            dict(game_id="2026_01_A_B", gameday=soon, home_team="B", away_team="A",
                 spread_line=3.0, pred_spreadCovered_optimal=1,
                 prob_underdogCovered=0.61),
            dict(game_id="2026_08_C_D", gameday=far, home_team="D", away_team="C",
                 spread_line=3.0, pred_spreadCovered_optimal=1,
                 prob_underdogCovered=0.61),
        ])
        assert bl.append_recommendations(preds, log) == 1
        assert bl.append_recommendations(preds, log) == 0  # idempotent
        out = pd.read_csv(log)
        assert list(out.game_id) == ["2026_01_A_B"]
        assert out.iloc[0].confidence_tier == "Strong"  # 0.59-0.65

@pytest.mark.parametrize("team,handicap,odds,home_score,away_score,result,profit", [
    ("A", 3.5, -110, 23, 20, "win", 90.91),
    ("A", 3, -110, 23, 20, "push", 0),
    ("A", 2.5, -110, 23, 20, "loss", -100),
    ("B", -3.5, 150, 24, 20, "win", 150),
    ("B", -3.5, -175, 24, 20, "win", 57.14),
])
def test_original_bet_ignores_changed_line_and_labels(tmp_path, team, handicap, odds,
                                                     home_score, away_score, result, profit):
    path = tmp_path / "bets.csv"
    pd.DataFrame([dict(game_id="g", home_team="B", away_team="A", recommended_team=team,
                       bet_type="spread", bet_spread=handicap, bet_odds=odds,
                       spread_line=10, bet_result="pending")]).to_csv(path, index=False)
    scores = pd.DataFrame([dict(game_id="g", gameday="2025-01-01", home_score=home_score,
                               away_score=away_score, spread_line=20, spreadPush=1,
                               underdogCovered=0)])
    assert bl.grade_pending(scores, path) == 1
    out = pd.read_csv(path).iloc[0]
    assert out.bet_result == result
    assert out.bet_profit == pytest.approx(profit)
    assert bl.grade_pending(scores, path) == 0
    assert bl.grade_pending(scores, path, regrade=True) == 1


def _legacy_log(tmp_path, **overrides):
    """One legacy row (pre bet_spread/bet_odds columns), as the real log stored it."""
    row = dict(game_id="g", week=1, home_team="SEA", away_team="NE", bet_type="spread",
               recommended_team="NE", spread_line=3.5, bet_result="pending", bet_profit="")
    row.update(overrides)
    path = tmp_path / "bets.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def _final(home_score, away_score, closing_line):
    return pd.DataFrame([dict(game_id="g", gameday="2025-01-01", home_score=home_score,
                              away_score=away_score, spread_line=closing_line)])


def test_legacy_row_uses_recorded_line_not_closing_line(tmp_path):
    # Real case 2026_01_NE_SEA: logged at NE +3.5, closed at 3.0, NE lost by 3.
    # The closing-line labels said push; the recorded bet won.
    path = _legacy_log(tmp_path)
    assert bl.grade_pending(_final(13, 10, closing_line=3.0), path) == 1
    out = pd.read_csv(path).iloc[0]
    assert (out.bet_result, out.bet_profit) == ("win", pytest.approx(90.91))
    assert out.bet_odds == -110
    assert out.odds_source == bl.ASSUMED_LEGACY_ODDS


def test_legacy_home_underdog_line_converted_for_recorded_team(tmp_path):
    # spread_line -2.5 (away favored) recorded for home dog CLE -> CLE +2.5.
    path = _legacy_log(tmp_path, home_team="CLE", away_team="CAR",
                       recommended_team="CLE", spread_line=-2.5)
    bl.grade_pending(_final(20, 22, closing_line=-1.0), path)   # CLE loses by 2
    assert pd.read_csv(path).iloc[0].bet_result == "win"


def test_pick_row_is_unresolved_and_audited_once(tmp_path):
    # Real case 2026_04_PIT_CLE: logged before a line existed, no team recorded,
    # previously settled "win" from closing-line labels.
    path = _legacy_log(tmp_path, home_team="CLE", away_team="PIT", recommended_team="Pick",
                       spread_line=0.0, bet_result="win", bet_profit=90.91)
    audit = tmp_path / "audit.csv"
    assert bl.grade_pending(_final(27, 24, -2.5), path, audit_path=str(audit)) == 0  # settled rows untouched
    bl.grade_pending(_final(27, 24, -2.5), path, regrade=True, audit_path=str(audit))
    out = pd.read_csv(path).iloc[0]
    assert out.bet_result == bl.UNRESOLVED
    assert pd.isna(out.bet_profit) and pd.isna(out.bet_odds)
    assert out.recommended_team == "Pick"                    # no team invented
    log = pd.read_csv(audit)
    assert list(log.columns) == bl.AUDIT_COLUMNS
    assert (log.iloc[0].previous_result, log.iloc[0].corrected_result) == ("win", bl.UNRESOLVED)
    bl.grade_pending(_final(27, 24, -2.5), path, regrade=True, audit_path=str(audit))
    assert len(pd.read_csv(audit)) == 1                      # no-op regrade adds nothing


def test_pending_pick_row_becomes_unresolved_without_audit(tmp_path):
    path = _legacy_log(tmp_path, recommended_team="Pick", spread_line=0.0)
    audit = tmp_path / "audit.csv"
    assert bl.grade_pending(_final(13, 10, 3.0), path, audit_path=str(audit)) == 1
    assert pd.read_csv(path).iloc[0].bet_result == bl.UNRESOLVED
    assert not audit.exists()


def test_append_skips_games_without_a_line(tmp_path):
    soon = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    preds = _preds([dict(game_id="2026_01_A_B", season=2026, week=1, gameday=soon,
                         home_team="B", away_team="A", spread_line=0.0,
                         pred_spreadCovered_optimal=1, prob_underdogCovered=0.6)])
    log = tmp_path / "log.csv"
    assert bl.append_recommendations(preds, str(log)) == 0


class TestRegradeAuditTrail:
    """A changed historical settlement is never written without its audit record."""

    @staticmethod
    def _settled_wrong(tmp_path):
        # Real case 2026_01_NE_SEA: graded "push" from closing-line labels; the
        # recorded NE +3.5 bet actually won (NE lost by 3).
        return _legacy_log(tmp_path, bet_result="push", bet_profit=0.0)

    def test_regrade_without_audit_path_uses_dated_default(self, tmp_path):
        path = self._settled_wrong(tmp_path)
        assert bl.grade_pending(_final(13, 10, 3.0), path, regrade=True) == 1
        assert pd.read_csv(path).iloc[0].bet_result == "win"
        default = tmp_path / f"settlement_corrections_{datetime.now():%Y%m%d}.csv"
        assert bl.default_audit_path(str(path)) == str(default)
        audit = pd.read_csv(default)
        assert list(audit.columns) == bl.AUDIT_COLUMNS
        assert (audit.iloc[0].previous_result, audit.iloc[0].corrected_result) == ("push", "win")

    def test_audit_write_failure_leaves_log_unchanged(self, tmp_path):
        path = self._settled_wrong(tmp_path)
        before = path.read_bytes()
        unwritable = tmp_path / "missing_dir" / "audit.csv"
        with pytest.raises(OSError):
            bl.grade_pending(_final(13, 10, 3.0), path, regrade=True, audit_path=str(unwritable))
        assert path.read_bytes() == before
        assert not unwritable.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["bets.csv"]   # no temp files left

    @pytest.mark.parametrize("prior", [None, "", "game_id,previous_result\nold,loss\n"],
                             ids=["no-audit-file", "empty-audit-file", "existing-entries"])
    def test_log_write_failure_restores_audit(self, tmp_path, monkeypatch, prior):
        path = self._settled_wrong(tmp_path)
        audit = tmp_path / "audit.csv"
        if prior is not None:
            audit.write_text(prior)
        audit_before = audit.read_bytes() if prior is not None else None
        log_before = path.read_bytes()
        real_replace = bl._atomic_replace

        def fail_on_log(target, write):
            if str(target) == str(path):
                raise OSError("disk full")
            return real_replace(target, write)

        monkeypatch.setattr(bl, "_atomic_replace", fail_on_log)
        with pytest.raises(OSError):
            bl.grade_pending(_final(13, 10, 3.0), path, regrade=True, audit_path=str(audit))
        assert (audit.read_bytes() if audit.exists() else None) == audit_before
        assert path.read_bytes() == log_before
        expected = ["audit.csv", "bets.csv"] if prior is not None else ["bets.csv"]
        assert sorted(p.name for p in tmp_path.iterdir()) == expected   # no temp files left

    def test_existing_audit_entries_are_preserved(self, tmp_path):
        path = self._settled_wrong(tmp_path)
        audit = tmp_path / "audit.csv"
        earlier = pd.DataFrame([dict(game_id="2026_03_TEN_NYG", recommended_team="TEN", spread_line=5.5,
                                     previous_result="loss", previous_profit=-100.0,
                                     corrected_result="win", corrected_profit=90.91,
                                     corrected_at="2026-10-02", reason="earlier correction")],
                               columns=bl.AUDIT_COLUMNS)
        earlier.to_csv(audit, index=False)
        bl.grade_pending(_final(13, 10, 3.0), path, regrade=True, audit_path=str(audit))
        out = pd.read_csv(audit)
        assert list(out.game_id) == ["2026_03_TEN_NYG", "g"]
        assert out.iloc[0].reason == "earlier correction"

    def test_repeated_regrades_add_no_duplicate_corrections(self, tmp_path):
        path = self._settled_wrong(tmp_path)
        audit = tmp_path / "audit.csv"
        for _ in range(3):
            bl.grade_pending(_final(13, 10, 3.0), path, regrade=True, audit_path=str(audit))
        assert len(pd.read_csv(audit)) == 1
        assert pd.read_csv(path).iloc[0].bet_result == "win"

    def test_unchanged_regrade_writes_no_audit_file(self, tmp_path):
        path = _legacy_log(tmp_path, bet_result="win", bet_profit=90.91)
        bl.grade_pending(_final(13, 10, 3.0), path, regrade=True)
        assert not (tmp_path / f"settlement_corrections_{datetime.now():%Y%m%d}.csv").exists()
