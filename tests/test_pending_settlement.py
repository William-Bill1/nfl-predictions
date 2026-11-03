"""nfl-gather-data.py settles only games with valid final-score evidence.

Before: unplayed games had their missing scores fillna(0)'d into a 0-0 result,
so a flagged unplayed game (e.g. CHI-SEA on 2026-11-02) was stored with
underdogCovered = 0 and spread_bet_return = -100, a settled loss. Now every
outcome field and bet return of a game without valid evidence
(bet_journal.completion_evidence) is stored as missing.

Covers the evidence rule (future, same-day, missing, malformed, conflicting,
valid finals, shutouts), the write-time blanking, and a real pipeline run on a
temporary synthetic schedule: settled rows are graded exactly as before,
predictions don't depend on settlement, and the missing outcomes survive
save/reload and the downstream readers (betting_log, the weekly backtest,
display_status, the output checker and the real home page). Nothing under
data_files/ is read or written; no network.
"""

import contextlib
import io
import json
import subprocess
import sys
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from test_team_features import ROOT, _FastClassifier, _load, _schedule

CHECKER = ROOT / "scripts" / "check_pipeline_outputs.py"
PREDICTIONS = "nfl_games_historical_with_predictions.csv"


def _gd(name="nfl_gather_data_settlement"):
    return _load(name, "nfl-gather-data.py")


def _utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


NOW = _utc("2026-11-03T12:00:00")          # 07:00 in Toronto, 2026-11-03


def sched(*rows):
    """(game_id, gameday, home, away[, result, total, overtime]); result/total
    default to the consistent values and overtime to 0 when both scores are numbers."""
    out = []
    for r in rows:
        gid, day, h, a, *rest = r
        num = all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in (h, a))
        res = rest[0] if len(rest) > 0 else (h - a if num else np.nan)
        tot = rest[1] if len(rest) > 1 else (h + a if num else np.nan)
        ot = rest[2] if len(rest) > 2 else (0 if num else np.nan)
        out.append(dict(game_id=gid, gameday=day, home_score=h, away_score=a,
                        result=res, total=tot, overtime=ot))
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# The evidence rule
# --------------------------------------------------------------------------

class TestFinalScoreMask:
    def mask(self, frame, now=NOW):
        return dict(zip(frame["game_id"], _gd().final_score_mask(frame, now)))

    def test_valid_finals_shutouts_and_pushes(self):
        m = self.mask(sched(("F", "2026-11-01", 27, 24), ("SHUT", "2026-11-01", 27, 0),
                            ("AWAY_SHUT", "2026-11-01", 0, 13), ("PUSH", "2026-11-01", 24, 21),
                            ("OT", "2026-11-01", 23, 20, 3, 43, 1)))
        assert all(m.values()), m

    def test_chi_sea_placeholder_is_never_final(self):
        # The reproduced row: no scores, result, total or overtime in the schedule; the
        # old pipeline wrote it as 0-0 with spread_bet_return -100.
        chi = sched(("2026_08_CHI_SEA", "2026-11-02", np.nan, np.nan))
        for now in (_utc("2026-11-02T18:00:00"), NOW, _utc("2027-06-01T00:00:00")):
            assert self.mask(chi, now) == {"2026_08_CHI_SEA": False}

    def test_future_and_same_day_games_are_not_final(self):
        m = self.mask(sched(("FUTURE", "2026-11-08", np.nan, np.nan),
                            ("FUTURE_SCORED", "2026-11-08", 20, 17),
                            ("TODAY_UNSCORED", "2026-11-03", np.nan, np.nan),
                            ("TODAY_SCORED", "2026-11-03", 31, 28)))
        assert not any(m.values()), m

    def test_same_day_policy_uses_the_toronto_date(self):
        g = sched(("SUN", "2026-11-02", 24, 20))
        # 03:00 UTC on Nov 3 is still Nov 2 in Toronto: the game is dated "today".
        assert self.mask(g, _utc("2026-11-03T03:00:00")) == {"SUN": False}
        assert self.mask(g, _utc("2026-11-03T05:00:00")) == {"SUN": True}   # 00:00 Toronto

    @pytest.mark.parametrize("home,away", [
        (np.nan, 17), (24, np.nan), ("27a", 3), ("", 3), (-3, 10), (27.5, 3),
        (np.inf, 3), (True, 0), (0, 0),
    ])
    def test_missing_one_score_malformed_or_zero_zero_is_not_final(self, home, away):
        g = sched(("G", "2026-11-01", home, away, 0, 0, 0))
        assert self.mask(g) == {"G": False}

    @pytest.mark.parametrize("result,total,overtime", [
        (4, 51, 0), (3, 50, 0), (3, 51, 2), (3, 51, np.nan), (np.nan, 51, 0), (3, np.nan, 0),
    ])
    def test_conflicting_evidence_is_not_final(self, result, total, overtime):
        g = sched(("G", "2026-11-01", 27, 24, result, total, overtime))
        assert self.mask(g) == {"G": False}

    def test_bad_game_day_duplicates_and_missing_columns_are_not_final(self):
        m = self.mask(sched(("TBD", "TBD", 27, 24), ("DUP", "2026-11-01", 27, 24),
                            ("DUP", "2026-11-01", 27, 24), ("OK", "2026-11-01", 27, 24)))
        assert m == {"TBD": False, "DUP": False, "OK": True}
        no_result = sched(("OK", "2026-11-01", 27, 24)).drop(columns="result")
        assert self.mask(no_result) == {"OK": False}
        assert not _gd().final_score_mask(pd.DataFrame({"x": [1]}), NOW).any()

    def test_index_is_preserved_and_input_unchanged(self):
        frame = sched(("A", "2026-11-01", 27, 24), ("B", "2026-11-08", np.nan, np.nan))
        frame.index = [10, 3]
        before = frame.copy()
        m = _gd().final_score_mask(frame, NOW)
        assert list(m.index) == [10, 3] and list(m) == [True, False]
        pd.testing.assert_frame_equal(frame, before)


# --------------------------------------------------------------------------
# Write-time blanking
# --------------------------------------------------------------------------

class TestUnsettledAsMissing:
    def frame(self):
        return pd.DataFrame({
            "game_id": ["F", "P", "X"],
            "home_score": [27.0, 0.0, 0.0], "away_score": [24.0, 0.0, 0.0],
            "underdogCovered": [1, 0, 0], "spreadPush": [0, 0, 0], "overHit": [1, 0, 0],
            "spread_bet_return": [90.91, -100.0, 0.0], "moneyline_bet_return": [0, 0, 0],
            "prob_underdogCovered": [0.61, 0.62, 0.40], "pred_spreadCovered_optimal": [1, 1, 0],
            "spread_line": [3.0, 3.0, -2.5],
        })

    def test_non_final_outcomes_and_returns_are_missing(self):
        gd, df = _gd(), self.frame()
        out = gd.unsettled_as_missing(df, pd.Series([True, False, False]))
        cols = [c for c in gd.OUTCOME_COLUMNS + gd.BET_RETURN_COLUMNS if c in df.columns]
        assert out.loc[1:, cols].isna().all().all()
        assert out.loc[0, "spread_bet_return"] == 90.91 and out.loc[0, "underdogCovered"] == 1

    def test_everything_else_and_final_rows_are_unchanged(self, tmp_path):
        gd, df = _gd(), self.frame()
        before = df.copy()
        out = gd.unsettled_as_missing(df, pd.Series([True, False, False]))
        pd.testing.assert_frame_equal(df, before)                       # input untouched
        other = ["game_id", "prob_underdogCovered", "pred_spreadCovered_optimal", "spread_line"]
        pd.testing.assert_frame_equal(out[other], df[other])
        # A final row is written byte for byte as before (integer flags stay 1/0).
        df.to_csv(tmp_path / "a.tsv", sep="\t", index=False)
        out.to_csv(tmp_path / "b.tsv", sep="\t", index=False)
        a = (tmp_path / "a.tsv").read_text().splitlines()
        b = (tmp_path / "b.tsv").read_text().splitlines()
        assert a[:2] == b[:2]
        assert b[2].split("\t")[1:7] == ["", "", "", "", "", ""]

    def test_outcome_columns_cover_every_score_derived_field(self):
        gd = _gd()
        assert {"home_score", "away_score", "result", "total", "overtime", "underdogCovered",
                "spreadPush", "underdogWon", "overHit", "spreadCovered"} <= set(gd.OUTCOME_COLUMNS)
        assert set(gd.BET_RETURN_COLUMNS) == {"moneyline_bet_return", "spread_bet_return",
                                              "totals_bet_return"}


# --------------------------------------------------------------------------
# The real pipeline on a temporary synthetic schedule
# --------------------------------------------------------------------------

def _settlement_schedule():
    """Four synthetic seasons; 2021 weeks 10-12 unplayed. Adds the nflverse
    result/overtime columns, pushes on integer lines, and invalid evidence in
    2021 week 7: an inconsistent result, overtime 2, a scored 0-0, a fractional
    score and a single missing score."""
    raw = _schedule(seasons=(2018, 2019, 2020, 2021), weeks=12, unplayed_from=(2021, 10))
    played = raw["home_score"].notna()
    integer_line = played & (raw["spread_line"] % 1 == 0) & (raw["spread_line"] != 0)
    for k, i in enumerate(raw.index[integer_line]):
        if k % 3 == 0:                                   # the favorite wins by exactly the line
            line = abs(raw.at[i, "spread_line"])
            if raw.at[i, "spread_line"] > 0:
                raw.at[i, "home_score"] = raw.at[i, "away_score"] + line
            else:
                raw.at[i, "away_score"] = raw.at[i, "home_score"] + line
    raw["total"] = raw["home_score"] + raw["away_score"]
    raw["result"] = raw["home_score"] - raw["away_score"]
    raw["overtime"] = np.where(played, 0, np.nan)
    wk7 = raw.index[(raw["season"] == 2021) & (raw["week"] == 7)]
    invalid = {"bad_result": wk7[0], "bad_overtime": wk7[1], "zero_zero": wk7[2], "fractional": wk7[3]}
    raw.loc[invalid["bad_result"], "result"] += 1
    raw.loc[invalid["bad_overtime"], "overtime"] = 2
    raw.loc[invalid["zero_zero"], ["home_score", "away_score", "result", "total"]] = 0.0
    raw.loc[invalid["fractional"], ["home_score", "result", "total"]] = [20.5, 20.5 - raw.at[invalid["fractional"], "away_score"],
                                                                       20.5 + raw.at[invalid["fractional"], "away_score"]]
    one = raw.index[(raw["season"] == 2021) & (raw["week"] == 6)][0]
    raw.loc[one, ["away_score", "result", "total", "overtime"]] = np.nan
    invalid["one_missing"] = one
    return raw, invalid


def _run(raw, root, now, name):
    data = root / "data_files"
    data.mkdir(parents=True)
    raw.to_csv(data / "nfl_games_historical.csv", sep="\t", index=False)
    gd = _gd(name)
    gd.DATA_DIR = str(data) + "/"
    gd.XGBClassifier = _FastClassifier
    gd._LGBM_AVAILABLE = False
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        gd.main(now=now)
    out = pd.read_csv(data / PREDICTIONS, sep="\t")
    metrics = json.loads((data / "model_metrics.json").read_text(encoding="utf-8"))
    return gd, data, out, metrics


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """The same schedule run 'today' = 2021 week 9's first game day (week 9 is
    scored but dated today or later, so it is pending) and run years later."""
    raw, invalid = _settlement_schedule()
    week9 = raw[(raw["season"] == 2021) & (raw["week"] == 9)]
    now_week9 = _utc(f"{week9['gameday'].min()}T17:00:00")
    later = _utc("2030-01-01T12:00:00")
    gd, data, out, metrics = _run(raw, tmp_path_factory.mktemp("week9"), now_week9, "gd_settle_week9")
    _, _, out_later, metrics_later = _run(raw, tmp_path_factory.mktemp("later"), later, "gd_settle_later")
    return dict(gd=gd, raw=raw, invalid=invalid, data=data, out=out, metrics=metrics, now=now_week9,
                week9=week9.index, out_later=out_later, metrics_later=metrics_later)


def _expected_spread(raw, out):
    """Independent grading from the scores: underdog cover, push and the -110
    spread return of a flagged game (no bet -> 0, push -> 0)."""
    hs, as_, line = raw["home_score"], raw["away_score"], raw["spread_line"]
    fav_margin = np.where(line > 0, hs - as_, as_ - hs)
    pick = line.fillna(0) == 0
    covered = np.where(pick, 0, fav_margin < line.abs()).astype(int)
    push = np.where(pick, 0, fav_margin == line.abs()).astype(int)
    flagged = out["pred_spreadCovered_optimal"] == 1
    ret = np.where(~flagged, 0.0, np.where(push == 1, 0.0, np.where(covered == 1, 90.91, -100.0)))
    return covered, push, ret


class TestPipelineRun:
    def test_final_games_are_graded_exactly_as_before(self, runs):
        raw, out, gd = runs["raw"], runs["out"], runs["gd"]
        final = gd.final_score_mask(raw, runs["now"]).to_numpy()
        covered, push, ret = _expected_spread(raw, out)
        assert (out.loc[final, "underdogCovered"].to_numpy() == covered[final]).all()
        assert (out.loc[final, "spreadPush"].to_numpy() == push[final]).all()
        np.testing.assert_allclose(out.loc[final, "spread_bet_return"].to_numpy(), ret[final])
        np.testing.assert_array_equal(out.loc[final, ["home_score", "away_score"]].to_numpy(),
                                      raw.loc[final, ["home_score", "away_score"]].to_numpy())
        over = (raw["home_score"] + raw["away_score"] > raw["total_line"]).astype(int).to_numpy()
        assert (out.loc[final, "overHit"].to_numpy() == over[final]).all()
        flagged = out["pred_spreadCovered_optimal"] == 1
        # Not vacuous: settled wins, losses and pushes are all present.
        assert {90.91, -100.0, 0.0} <= set(out.loc[final & flagged, "spread_bet_return"])
        assert ((push == 1) & final & flagged).any()

    def test_pending_and_invalid_games_carry_no_result_or_profit(self, runs):
        raw, out, gd, inv = runs["raw"], runs["out"], runs["gd"], runs["invalid"]
        final = gd.final_score_mask(raw, runs["now"])
        cols = list(gd.OUTCOME_COLUMNS + gd.BET_RETURN_COLUMNS)
        assert set(cols) <= set(out.columns)
        assert out.loc[~final.to_numpy(), cols].isna().all().all()
        unplayed = raw["home_score"].isna() & raw["away_score"].isna()
        assert (~final[unplayed]).all() and (~final[runs["week9"]]).all()   # future and same-day
        for i in inv.values():
            assert not final[i]
            assert out.loc[i, cols].isna().all()
        # The bug: flagged games without a result used to be stored as -100 losses.
        flagged_pending = (out["pred_spreadCovered_optimal"] == 1) & ~final.to_numpy()
        assert flagged_pending.sum() >= 2
        assert out.loc[flagged_pending, "spread_bet_return"].isna().all()
        assert int(final.sum()) == len(raw) - int(unplayed.sum()) - len(runs["week9"]) - len(inv)

    def test_predictions_do_not_depend_on_settlement(self, runs):
        gd, out, later = runs["gd"], runs["out"], runs["out_later"]
        settled = list(gd.OUTCOME_COLUMNS + gd.BET_RETURN_COLUMNS)
        other = [c for c in out.columns if c not in settled]
        assert {"prob_underdogCovered", "pred_spreadCovered_optimal", "ev_spread",
                "edge_underdog_spread", "predictedSpreadCovered"} <= set(other)
        pd.testing.assert_frame_equal(out[other], later[other])
        assert runs["metrics"] == runs["metrics_later"]
        # Years later the scored week 9 settles; unplayed games still don't.
        w9 = runs["week9"]
        assert later.loc[w9, "spread_bet_return"].notna().all()
        assert out.loc[w9, "spread_bet_return"].isna().all()
        both = out["spread_bet_return"].notna()
        pd.testing.assert_frame_equal(out.loc[both, settled], later.loc[both, settled])

    def test_missing_outcomes_survive_save_and_reload(self, runs, tmp_path):
        gd, out = runs["gd"], runs["out"]
        text = (runs["data"] / PREDICTIONS).read_text(encoding="utf-8").splitlines()
        header = text[0].split("\t")
        i = header.index("spread_bet_return")
        pending_row = runs["week9"][0]
        assert text[pending_row + 1].split("\t")[i] == ""          # an empty field, not 0 or -100
        again = tmp_path / PREDICTIONS
        out.to_csv(again, sep="\t", index=False)
        pd.testing.assert_frame_equal(pd.read_csv(again, sep="\t"), out)


class TestDownstreamReaders:
    def test_betting_log_keeps_pending_bets_pending(self, runs, tmp_path):
        import betting_log as bl
        raw, out, gd = runs["raw"], runs["out"], runs["gd"]
        final = gd.final_score_mask(raw, runs["now"]).to_numpy()
        same_day = runs["week9"][0]                     # scored in the schedule, pending in the file
        unplayed = raw.index[raw["home_score"].isna()][0]
        settled = out.index[final & (out["spread_line"].abs() > 0)][0]
        rows = []
        for i in (same_day, unplayed, settled):
            r = out.loc[i]
            underdog = r["away_team"] if r["spread_line"] > 0 else r["home_team"]
            rows.append({c: "" for c in bl.LOG_COLUMNS} | {
                "game_id": r["game_id"], "gameday": r["gameday"], "season": r["season"], "week": r["week"],
                "home_team": r["home_team"], "away_team": r["away_team"], "bet_type": "spread",
                "recommended_team": underdog, "spread_line": r["spread_line"],
                "bet_spread": abs(r["spread_line"]), "bet_odds": -110.0, "bet_result": "pending"})
        log = tmp_path / "betting_recommendations_log.csv"
        pd.DataFrame(rows, columns=bl.LOG_COLUMNS).to_csv(log, index=False)
        assert bl.grade_pending(out, str(log), audit_path=str(tmp_path / "audit.csv")) == 1
        graded = pd.read_csv(log)
        assert list(graded["bet_result"][:2]) == ["pending", "pending"]
        assert graded["bet_result"][2] in ("win", "loss", "push")
        assert graded["actual_home_score"][:2].isna().all()

    def test_weekly_backtest_sees_the_pending_week_as_incomplete(self, runs):
        bt = _load("run_weekly_backtest_settle", "scripts/run_weekly_backtest.py")
        # Real today is long after these 2021 dates: only the stored file decides.
        # Week 9 is scored in the schedule but was pending at the week-9 run.
        assert bt.latest_completed_week(runs["out"], 2021) == 8
        assert bt.latest_completed_week(runs["out_later"], 2021) == 9
        # Weeks 6 and 7 hold games without valid evidence, so they never count as complete.
        week = runs["out"]["week"]
        for w in (6, 7):
            assert runs["out"].loc[(week == w) & (runs["out"]["season"] == 2021), "home_score"].isna().any()

    def test_display_status_counts_only_final_games(self, runs):
        import display_status as ds
        raw, out = runs["raw"], runs["out"]
        status = ds.result_status(out, raw, runs["now"])
        final = runs["gd"].final_score_mask(raw, runs["now"]).to_numpy()
        assert (status[final] == ds.FINAL).all()
        assert set(status[~final]) <= {ds.PENDING, ds.UNAVAILABLE}
        kept, left_out = ds.final_only(out[out["pred_spreadCovered_optimal"] == 1], raw, runs["now"])
        assert kept["spread_bet_return"].notna().all() and left_out >= 2

    def test_output_checker_passes(self, runs):
        res = subprocess.run([sys.executable, str(CHECKER)], cwd=runs["data"].parent,
                             capture_output=True, text=True)
        assert res.returncode == 0, res.stdout + res.stderr


# --------------------------------------------------------------------------
# The real home page on a predictions file with missing outcomes
# --------------------------------------------------------------------------

def _home_app(tmp_path, monkeypatch, edit=None, blank=True):
    """test_display_status's synthetic home data, with the predictions file
    written the way the pipeline now writes it (outcomes of every game without
    schedule evidence missing) unless blank=False, then edit(preds, final) if
    given, served from tmp_path with the network refused.
    Returns (predictions as written, final mask, data dir, AppTest)."""
    import socket

    import betting_log
    import dotenv
    from streamlit.testing.v1 import AppTest
    from test_display_status import build_home_fixture

    d = build_home_fixture(tmp_path)
    preds = pd.read_csv(d / PREDICTIONS, sep="\t")
    schedule = pd.read_csv(d / "nfl_games_historical.csv", sep="\t")
    aligned = schedule.set_index("game_id").reindex(preds["game_id"]).reset_index()
    gd = _gd("gd_settle_home")
    final = gd.final_score_mask(aligned, datetime.now(timezone.utc))
    final.index = preds.index
    if blank:
        preds = gd.unsettled_as_missing(preds, final)
        assert preds.loc[~final, ["spreadCovered", "underdogWon", "overHit"]].isna().all().all()
        assert (~final).sum() >= 3              # upcoming, today and missing-from-schedule games
    if edit is not None:
        preds = edit(preds, final)
    preds.to_csv(d / PREDICTIONS, sep="\t", index=False)

    monkeypatch.chdir(d.parent)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(betting_log, "append_recommendations", lambda *a, **k: 0)
    monkeypatch.setattr(betting_log, "grade_pending", lambda *a, **k: 0)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)

    def refuse(*a, **k):
        raise OSError("network blocked in this test")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    return preds, final, d, AppTest.from_file(str(ROOT / "predictions.py"), default_timeout=120)


class TestHomePageWithMissingOutcomes:
    """The home page's in-app model splits stratify on spreadCovered /
    underdogWon / overHit; with missing targets that raised "Input y contains
    NaN". Unknown targets are now left out, and the moneyline baseline counts
    only known results."""

    @pytest.fixture
    def home(self, tmp_path, monkeypatch):
        blanked, _, _, at = _home_app(tmp_path, monkeypatch)
        return blanked, at

    def test_page_renders_and_baseline_uses_known_results(self, home):
        from test_display_status import TestHomePage
        blanked, at = home
        at = TestHomePage().loaded(at)
        assert not at.exception, [e.value for e in at.exception]
        assert TestHomePage.results_table(at) is not None
        known = blanked["underdogWon"].dropna()
        assert len(known) < len(blanked)
        expected = f"{(known == 0).mean():.1%}"
        text = TestHomePage.all_text(at)
        assert f"Baseline (always pick the favorite on the moneyline)**: {expected} accuracy" in text

    def test_moneyline_figures_count_settled_bets_only(self, home):
        # The page recomputes moneyline returns itself; a missing underdogWon used to
        # become a -100 loss there, and pending bets counted in every total.
        from test_display_status import TestHomePage
        blanked, at = home
        at = TestHomePage().loaded(at)
        assert not at.exception
        flagged = blanked["pred_underdogWon_optimal"] == 1
        settled = flagged & blanked["underdogWon"].notna()
        assert (flagged & ~settled).any()                    # not vacuous: pending moneyline bets exist
        metrics = [(m.label, m.value) for m in at.metric]
        start = [label for label, _ in metrics].index("🎯 Moneyline Strategy")
        block = dict(metrics[start + 1:start + 3])           # the moneyline block's Total Bets, Win Rate
        assert block["Total Bets"] == f"{int(settled.sum()):,}"
        wins = int((blanked.loc[settled, "underdogWon"] == 1).sum())
        assert block["Win Rate"] == f"{wins / int(settled.sum()):.1%}"


# --------------------------------------------------------------------------
# Monte Carlo feature-search controls with missing targets
# --------------------------------------------------------------------------

class _StubModel:
    """Stands in for XGBClassifier: records what it is fitted on, trains nothing."""
    calls = []

    def __init__(self, *a, **k):
        pass

    def fit(self, X, y):
        _StubModel.calls.append(("fit", X, y))
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)


class _StubCalibrated(_StubModel):
    """Stands in for CalibratedClassifierCV(estimator, method=..., cv=...)."""


def _stub_cv(model, X, y, cv=3, scoring=None):
    _StubModel.calls.append(("cv", X, y))
    return np.full(cv, 0.5)


FEATURE_FILES = ("best_features_spread.txt", "best_features_moneyline.txt", "best_features_totals.txt")


def _repo_feature_files():
    return {f: (ROOT / "data_files" / f).read_bytes() if (ROOT / "data_files" / f).exists() else None
            for f in FEATURE_FILES}


class TestFeatureSearchControls:
    """The Moneyline and Totals Monte Carlo searches split, fit and score only
    games with a known target; before, train_test_split(stratify=y) on the full
    target raised "Input y contains NaN". Model fitting and CV scoring are
    stubbed; feature files are written only under tmp_path."""

    @pytest.fixture(autouse=True)
    def stubs(self, monkeypatch):
        import sklearn.calibration
        import sklearn.model_selection
        import xgboost
        _StubModel.calls = []
        monkeypatch.setattr(xgboost, "XGBClassifier", _StubModel)
        monkeypatch.setattr(sklearn.model_selection, "cross_val_score", _stub_cv)
        monkeypatch.setattr(sklearn.calibration, "CalibratedClassifierCV", _StubCalibrated)
        before = _repo_feature_files()
        yield
        assert _repo_feature_files() == before      # the repo's feature files are never touched

    @staticmethod
    def search(at, model):
        from test_display_status import TestHomePage
        at = TestHomePage().loaded(at)
        assert not at.exception, [e.value for e in at.exception]
        _StubModel.calls = []                       # only calls made by the button below
        next(n for n in at.number_input if n.label == f"Number of Iterations ({model})").set_value(10)
        next(b for b in at.button if b.label == f"Run Monte Carlo Search ({model})").click().run()
        assert not at.exception, [e.value for e in at.exception]
        return at

    @staticmethod
    def check_calls(known):
        """Every stubbed fit / CV call got labels with no missing value, drawn
        only from known rows, aligned row for row with its features."""
        assert _StubModel.calls
        for kind, X, y in _StubModel.calls:
            assert y.notna().all(), kind
            assert X.index.equals(y.index), kind
            assert set(y.index) <= set(known[known].index), kind

    @staticmethod
    def retrain_fit():
        fits = [c for c in _StubModel.calls if c[0] == "fit"]
        assert len(fits) == 1                       # the best-feature retraining
        return fits[0]

    @pytest.mark.parametrize("model, target, out", [
        ("Moneyline", "underdogWon", "best_features_moneyline.txt"),
        ("Totals", "overHit", "best_features_totals.txt"),
    ])
    def test_search_uses_known_results_only(self, tmp_path, monkeypatch, model, target, out):
        from sklearn.model_selection import train_test_split
        preds, final, d, at = _home_app(tmp_path, monkeypatch)
        known = preds[target].notna()
        assert (~known).sum() >= 3                  # not vacuous: missing targets exist
        at = self.search(at, model)
        self.check_calls(known)
        assert sum(k == "cv" for k, _, _ in _StubModel.calls) >= 10
        # The retraining fit is on the 80% split of the known rows, with labels from that split.
        y_known = preds.loc[known, target]
        train, _ = train_test_split(y_known, test_size=0.2, random_state=42, stratify=y_known)
        _, X, y = self.retrain_fit()
        assert y.index.equals(train.index) and (y.to_numpy() == train.to_numpy()).all()
        assert (d / out).read_text().strip()        # the search wrote its features under tmp_path
        assert not [w for w in at.warning if "feature search not run" in w.value]

    @pytest.mark.parametrize("model, target", [("Moneyline", "underdogWon"), ("Totals", "overHit")])
    def test_fully_known_data_keeps_the_original_split(self, tmp_path, monkeypatch, model, target):
        # With every target known, the retraining split is the page's original
        # split of all rows, so its labels equal the y_train_* used before.
        from sklearn.model_selection import train_test_split
        preds, final, d, at = _home_app(tmp_path, monkeypatch, blank=False)
        assert preds[target].notna().all()
        at = self.search(at, model)
        self.check_calls(preds[target].notna())
        y_all = preds[target]
        train, _ = train_test_split(y_all, test_size=0.2, random_state=42, stratify=y_all)
        _, X, y = self.retrain_fit()
        assert len(y) == len(train)
        assert y.index.equals(train.index) and (y.to_numpy() == train.to_numpy()).all()

    @pytest.mark.parametrize("model, target, out", [
        ("Moneyline", "underdogWon", "best_features_moneyline.txt"),
        ("Totals", "overHit", "best_features_totals.txt"),
    ])
    def test_insufficient_known_results_stop_before_any_write(self, tmp_path, monkeypatch,
                                                              model, target, out):
        def few_known(preds, final):
            # Keep 3 known games of the rarer outcome: enough for the page's own
            # split, too few for the search (5 per outcome).
            p = preds.copy()
            known = p[target].notna()
            rare = p.loc[known, target].value_counts().idxmin()
            drop = p.index[known & (p[target] == rare)][3:]
            p.loc[drop, target] = np.nan
            return p
        preds, final, d, at = _home_app(tmp_path, monkeypatch, edit=few_known)
        assert preds[target].value_counts().min() == 3
        at = self.search(at, model)
        msgs = [w.value for w in at.warning if "feature search not run" in w.value]
        assert msgs == [f"{model} feature search not run: it needs at least 5 games of each outcome "
                        f"with a final result; the rarer outcome has 3. {out} was not changed."]
        assert _StubModel.calls == []               # nothing searched, fitted or scored
        assert not (d / out).exists()


# --------------------------------------------------------------------------
# Spread performance block: pending signals vs no bet vs push
# --------------------------------------------------------------------------

class TestSpreadPendingCaption:
    """The spread block selected spread_bet_return != 0, and NaN != 0 is True,
    so every pending game counted as a left-out flagged bet. Now the left-out
    count uses the recorded signal (pred_spreadCovered_optimal); "no bet"
    (unflagged), "push" (flagged, final, return 0) and "unknown result"
    (flagged, no verified final) stay distinct."""

    @staticmethod
    def spread_rows(preds, final):
        p = preds.copy()
        p["pred_spreadCovered_optimal"] = 0
        p["spreadPush"] = np.where(final, 0.0, np.nan)
        p["spread_bet_return"] = np.where(final, 0.0, np.nan)   # pending: missing, as written now
        fin = p.index[final]
        upcoming = pd.to_datetime(p["gameday"]).dt.date > datetime.now().date()
        pending = p.index[~final & upcoming]
        wins, losses, push = fin[:3], fin[3:5], fin[5]
        p.loc[wins, "pred_spreadCovered_optimal"] = 1
        p.loc[wins, "spread_bet_return"] = 90.91
        p.loc[losses, "pred_spreadCovered_optimal"] = 1
        p.loc[losses, "spread_bet_return"] = -100.0
        p.loc[push, ["pred_spreadCovered_optimal", "spreadPush", "spread_bet_return"]] = [1, 1.0, 0.0]
        p.loc[pending[0], "pred_spreadCovered_optimal"] = 1     # the one flagged pending game
        return p

    def test_caption_counts_only_flagged_pending_games(self, tmp_path, monkeypatch):
        from test_display_status import TestHomePage
        preds, final, d, at = _home_app(tmp_path, monkeypatch, edit=self.spread_rows)
        flagged = preds["pred_spreadCovered_optimal"] == 1
        # Not vacuous: unflagged pending games with a missing return exist.
        assert (~final & ~flagged & preds["spread_bet_return"].isna()).sum() >= 3
        assert int((~final & flagged).sum()) == 1
        assert int((final & flagged & (preds["spreadPush"] == 1)).sum()) == 1
        at = TestHomePage().loaded(at)
        assert not at.exception, [e.value for e in at.exception]

        captions = [c.value for c in at.caption if "without a verified final score" in c.value]
        assert captions == ["1 flagged game(s) without a verified final score are not counted "
                            "(pending, result unavailable, or the schedule and predictions files "
                            "disagree)."]

        # Settled metrics: 3 wins and 2 losses; the push is neither a bet nor pending.
        labels = [m.label for m in at.metric]
        start = labels.index("Spread Bets")
        block = [(m.label, m.value) for m in list(at.metric)[start:start + 3]]
        assert block == [("Spread Bets", "5"), ("Win Rate", "60.0%"),
                         ("ROI", f"{(3 * 90.91 - 200) / 500:.1%}")]
        text = TestHomePage.all_text(at)
        assert "60.0% win rate" in text and "over 5 EV-qualified final games" in text
