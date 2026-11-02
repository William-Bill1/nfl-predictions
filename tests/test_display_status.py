"""Home-page display clarity (display_status.py, wired into predictions.py).

Unit tests for completion status, blanking of unsettled results, final-only
totals and neutral spread labels; source checks for removed claims; and the
real home page rendered with AppTest from temporary synthetic fixtures, with
the network blocked, .env loading disabled and recommendation logging stubbed.
Production data_files/ is never read or written.
"""
import hashlib
import json
import socket
import sys
import warnings
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import display_status as ds
from tests.test_home_downloads import PREDICTIONS, ROOT, write_home_fixture

NAN = float("nan")
NOW = datetime(2026, 11, 2, 15, 0, tzinfo=timezone.utc)     # 10:00 Toronto
PAST = "2026-10-26"
TODAY = "2026-11-02"


def sched(*rows, **override):
    """Schedule evidence rows (game_id, home, away[, gameday]). result/total are
    derived when both scores are numeric; `override` sets a column for every row."""
    out = []
    for r in rows:
        gid, h, a = r[:3]
        day = r[3] if len(r) > 3 else PAST
        num = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (h, a))
        out.append({"game_id": gid, "gameday": day, "home_score": h, "away_score": a,
                    "result": h - a if num else NAN, "total": h + a if num else NAN, "overtime": 0})
    df = pd.DataFrame(out, columns=list(ds.EVIDENCE_COLUMNS))
    for c, v in override.items():
        df[c] = v
    return df


def status_of(ev, *ids):
    return ds.completion_status(pd.Series(list(ids)), ev, NOW).tolist()


# --------------------------------------------------------------------------
# Completion status: the schedule's evidence (bet_journal.completion_evidence)
# --------------------------------------------------------------------------

class TestCompletionStatus:
    def test_valid_finals_including_shutouts(self):
        ev = sched(("G1", 27, 24), ("SHUT_H", 27, 0), ("SHUT_A", 0, 13))
        assert status_of(ev, "G1", "SHUT_H", "SHUT_A") == [ds.FINAL] * 3

    def test_missing_scores_are_pending_even_with_a_long_passed_date(self):
        ev = sched(("OLD", NAN, NAN, "2020-01-05"), ("G2", NAN, NAN))
        assert status_of(ev, "OLD", "G2") == [ds.PENDING, ds.PENDING]

    def test_scored_game_dated_today_is_pending(self):
        assert status_of(sched(("MNF", 20, 17, TODAY)), "MNF") == [ds.PENDING]

    @pytest.mark.parametrize("home,away", [
        (0, 0),                    # the 0-0 placeholder, not a final
        (0.0, 0.0),
        (27.5, 24),                # fractional
        (float("inf"), 24),        # infinite
        (27, float("-inf")),
        (-3, 24),                  # negative
        ("abc", 24),               # text
        ("27", "x"),
        (27, NAN),                 # only one score
        (NAN, 24),
        (True, 24),                # booleans are not scores
    ])
    def test_malformed_scores_are_unavailable(self, home, away):
        assert status_of(sched(("BAD", home, away)), "BAD") == [ds.UNAVAILABLE]

    def test_inconsistent_fields_are_unavailable(self):
        assert status_of(sched(("R", 27, 24), result=7), "R") == [ds.UNAVAILABLE]
        assert status_of(sched(("T", 27, 24), total=50), "T") == [ds.UNAVAILABLE]
        assert status_of(sched(("O", 27, 24), overtime=2), "O") == [ds.UNAVAILABLE]
        assert status_of(sched(("D", 27, 24), gameday="not a date"), "D") == [ds.UNAVAILABLE]

    def test_ambiguous_or_absent_evidence_is_unavailable(self):
        dup = sched(("DUP", 20, 17), ("DUP", 20, 17))
        assert status_of(dup, "DUP", "NOT_THERE") == [ds.UNAVAILABLE, ds.UNAVAILABLE]
        conflict = sched(("C", 20, 17), ("C", 24, 17))
        assert status_of(conflict, "C") == [ds.UNAVAILABLE]
        both_blank = sched(("BB", NAN, NAN), ("BB", NAN, NAN))
        assert status_of(both_blank, "BB") == [ds.UNAVAILABLE]
        assert status_of(None, "G1") == [ds.UNAVAILABLE]
        assert status_of(sched(), "G1") == [ds.UNAVAILABLE]
        assert status_of(sched(("G1", 27, 24)).drop(columns="result"), "G1") == [ds.UNAVAILABLE]

    def test_index_is_preserved(self):
        ids = pd.Series(["G1", "G2"], index=[40, 7])
        got = ds.completion_status(ids, sched(("G1", 3, 0)), NOW)
        assert list(got.index) == [40, 7] and got[40] == ds.FINAL and got[7] == ds.UNAVAILABLE

    def test_load_completion_evidence_keeps_values_as_read(self, tmp_path):
        assert ds.load_completion_evidence(str(tmp_path)) is None
        (tmp_path / ds.SCHEDULE_NAME).write_text(
            "game_id\tseason\tgameday\thome_score\taway_score\tresult\ttotal\tovertime\n"
            f"A\t2026\t{PAST}\t21\t14\t7\t35\t0\n"
            f"B\t2026\t{PAST}\t\t\t\t\t\n"
            f"Z\t2026\t{PAST}\t0\t0\t0\t0\t0\n"
            f"T\t2026\t{PAST}\tabc\t14\t\t\t0\n", encoding="utf-8")
        ev = ds.load_completion_evidence(str(tmp_path))
        assert status_of(ev, "A", "B", "Z", "T") == [ds.FINAL, ds.PENDING, ds.UNAVAILABLE, ds.UNAVAILABLE]
        (tmp_path / ds.SCHEDULE_NAME).write_text(
            "game_id\thome_score\taway_score\nA\t21\t14\n", encoding="utf-8")
        assert ds.load_completion_evidence(str(tmp_path)) is None   # lacks validation columns


# --------------------------------------------------------------------------
# Result status: schedule evidence cross-checked with the predictions file
# --------------------------------------------------------------------------

def preds(*rows):
    return pd.DataFrame(rows, columns=["game_id", "home_score", "away_score"])


class TestResultStatus:
    def test_disagreement_is_inconsistent(self):
        # The reported case: schedule 27-24, predictions file 0-0 with zero flags.
        ev = sched(("G", 27, 24))
        df = preds(("G", 0, 0)).assign(underdogCovered=0, overHit=0)
        assert ds.result_status(df, ev, NOW).tolist() == [ds.INCONSISTENT]
        assert ds.result_status(preds(("G", 24, 27)), ev, NOW).tolist() == [ds.INCONSISTENT]
        assert ds.result_status(preds(("G", 27, NAN)), ev, NOW).tolist() == [ds.INCONSISTENT]

    def test_consistent_finals_including_shutouts(self):
        ev = sched(("G", 27, 24), ("S", 27, 0))
        df = preds(("G", 27.0, 24.0), ("S", 27, 0))
        assert ds.result_status(df, ev, NOW).tolist() == [ds.FINAL, ds.FINAL]

    def test_unplayed_placeholder_is_pending_but_a_real_score_is_not(self):
        ev = sched(("P", NAN, NAN), ("Q", NAN, NAN))
        df = preds(("P", 0, 0), ("Q", 20, 17))
        assert ds.result_status(df, ev, NOW).tolist() == [ds.PENDING, ds.INCONSISTENT]

    def test_scored_today_or_missing_from_the_schedule_is_not_a_disagreement(self):
        # Same score in both files but dated today: pending, not inconsistent.
        ev = sched(("MNF", 20, 17, TODAY))
        assert ds.result_status(preds(("MNF", 20, 17)), ev, NOW).tolist() == [ds.PENDING]
        # No schedule row: nothing to disagree with, so unavailable.
        assert ds.result_status(preds(("GONE", 20, 17)), ev, NOW).tolist() == [ds.UNAVAILABLE]

    def test_malformed_schedule_stays_unavailable(self):
        df = preds(("Z", 0, 0))
        assert ds.result_status(df, sched(("Z", 0, 0)), NOW).tolist() == [ds.UNAVAILABLE]

    def test_without_ids_or_scores_nothing_is_final(self):
        ev = sched(("G", 27, 24))
        assert ds.result_status(pd.DataFrame({"x": [1]}), ev, NOW).tolist() == [ds.UNAVAILABLE]
        assert ds.result_status(pd.DataFrame({"game_id": ["G"]}), ev, NOW).tolist() == [ds.INCONSISTENT]

    def test_index_preserved_and_input_unchanged(self):
        df = preds(("G", 0, 0), ("H", 10, 3)).set_axis([9, 4])
        before = df.copy()
        got = ds.result_status(df, sched(("G", 27, 24), ("H", 10, 3)), NOW)
        assert got.to_dict() == {9: ds.INCONSISTENT, 4: ds.FINAL}
        pd.testing.assert_frame_equal(df, before)


class TestBlankUnsettled:
    def frame(self):
        return pd.DataFrame({"game_id": ["F", "P", "U"], "home_score": [31.0, 0.0, 0.0],
                             "away_score": [17.0, 0.0, 0.0], "underdogCovered": [1, 0, 0],
                             "overHit": [0, 0, 0], "spread_line": [3.5, 3.0, -2.5]})

    def test_pending_and_unavailable_scores_and_outcomes_are_blank(self):
        df = self.frame()
        status = pd.Series([ds.FINAL, ds.PENDING, ds.UNAVAILABLE])
        out = ds.blank_unsettled(df, status)
        for c in ds.RESULT_COLUMNS:
            assert out[c].iloc[1:].isna().all(), c
        assert out["spread_line"].tolist() == df["spread_line"].tolist()   # non-result column untouched

    def test_final_values_are_unchanged(self):
        df = self.frame()
        out = ds.blank_unsettled(df, pd.Series([ds.FINAL, ds.PENDING, ds.UNAVAILABLE]))
        assert out.loc[0, "home_score"] == 31 and out.loc[0, "away_score"] == 17
        assert out.loc[0, "underdogCovered"] == 1 and out.loc[0, "overHit"] == 0
        assert df["home_score"].tolist() == [31.0, 0.0, 0.0]               # input not mutated

    def test_dtypes(self):
        # Scores stay float with NaN (missing, never 0); 0/1 outcome flags become nullable
        # Int64 so a CheckboxColumn shows an empty cell rather than an unticked box.
        out = ds.blank_unsettled(self.frame(), pd.Series([ds.PENDING] * 3))
        assert str(out["home_score"].dtype) == "float64" and str(out["away_score"].dtype) == "float64"
        assert str(out["underdogCovered"].dtype) == "Int64" and str(out["overHit"].dtype) == "Int64"

    def test_column_subset(self):
        out = ds.blank_unsettled(self.frame(), pd.Series([ds.PENDING] * 3), columns=("home_score",))
        assert out["home_score"].isna().all() and out["away_score"].tolist() == [17.0, 0.0, 0.0]


class TestFinalOnly:
    def test_counts_and_keeps_only_verified_final_games(self):
        df = pd.DataFrame({"game_id": ["F1", "F2", "P", "U", "X"],
                           "home_score": [24, 10, 0, 0, 0], "away_score": [20, 13, 0, 0, 0],
                           "spread_bet_return": [90.91, -100, -100, -100, -100]})
        ev = sched(("F1", 24, 20), ("F2", 10, 13), ("P", NAN, NAN), ("X", 27, 24))
        kept, left_out = ds.final_only(df, ev, NOW)
        assert kept["game_id"].tolist() == ["F1", "F2"] and left_out == 3
        assert kept["spread_bet_return"].sum() == pytest.approx(-9.09)

    def test_no_evidence_counts_nothing(self):
        df = pd.DataFrame({"game_id": ["F1"], "home_score": [3], "away_score": [0],
                           "spread_bet_return": [90.91]})
        kept, left_out = ds.final_only(df, None, NOW)
        assert kept.empty and left_out == 1

    def test_returns_an_independent_copy(self):
        df = pd.DataFrame({"game_id": ["F1"], "home_score": [24], "away_score": [20], "r": [1.0]})
        kept, _ = ds.final_only(df, sched(("F1", 24, 20)), NOW)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            kept["r"] = 2.0
        assert df["r"].tolist() == [1.0]


# --------------------------------------------------------------------------
# Neutral labels and wording
# --------------------------------------------------------------------------

class TestNeutralLabels:
    def test_stored_tier_values_map_to_neutral_bands(self):
        assert ds.tier_label("Elite") == "Model ≥65%"
        assert ds.tier_label("🔥 Elite") == "Model ≥65%"
        assert ds.tier_label("Strong") == "Model 59–65%"
        assert ds.tier_label("Good") == "Model 55–59%"
        assert ds.tier_label("Lean") == "Model 50–55%"
        assert ds.tier_label("unresolved") == "unresolved" and ds.tier_label(None) == ""
        assert ds.tier_label(NAN) == ""

    def test_labels_make_no_quality_claim(self):
        for label in set(ds.NEUTRAL_TIER_LABELS.values()):
            for word in ("Elite", "Strong", "Good", "Lean", "edge", "value"):
                assert word.lower() not in label.lower(), (label, word)

    def test_edge_and_regenerated_wording(self):
        assert "model-implied" in ds.EDGE_HELP.lower() and "not established" in ds.EDGE_HELP
        assert "Not a demonstrated betting edge" in ds.EDGE_HELP
        note = ds.REGENERATED_NOTE
        assert "regenerates" in note and "after their results were known" in note
        assert "not the forecasts made before kickoff" in note
        assert "not evidence of historical predictive success" in note
        assert "Frozen Performance" in note
        assert (ROOT / ds.FROZEN_PAGE).is_file()

    def test_cut_points_unchanged_and_labels_follow_them(self):
        src = (ROOT / "predictions.py").read_text(encoding="utf-8")
        assert "SPREAD_TIER_CUTS = (0.65, 0.59, 0.55, 0.50)" in src
        assert ("SPREAD_TIER_LABELS = tuple(ds.NEUTRAL_TIER_LABELS[k] for k in "
                "('Elite', 'Strong', 'Good', 'Lean'))") in src


REMOVED_FROM_HOME = [
    "🔥 Elite", "⭐ Strong", "📈 Good", "⚖️ Lean", "Recommended for max units", "Solid value",
    "Strong conviction bets", "best risk-adjusted returns", "Positive edge = value bet",
    "positive = value bet", "new elite betting opportunities", "new strong betting opportunities",
    "All shown bets have positive expected value", "Positive EV = profitable long-term",
    "Elite Tier Performance", "Elite Opportunities", "Green edges indicate positive expected value bets",
    "Showing 50 most recent completed games",
]


@pytest.mark.parametrize("phrase", REMOVED_FROM_HOME)
def test_removed_claims_stay_removed(phrase):
    assert phrase not in (ROOT / "predictions.py").read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# The real home page, synthetic fixtures
# --------------------------------------------------------------------------

def tree(d):
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(d).rglob("*")) if p.is_file()}


def build_home_fixture(root):
    """The standard synthetic home data plus a schedule (nfl_games_historical.csv)
    with validated final scores for every past game, and:
    * TODAY_PENDING: dated today, 0-0 in the predictions file, no score in the
      schedule (the Monday night case);
    * the most recent past game missing from the schedule;
    * a disagreement: schedule 27-24, predictions file 0-0 with zero outcome
      flags and a -100 spread return;
    * a consistent 27-0 shutout.
    Writes the special game_ids to root/home/expected.json. Returns the
    data_files directory under root/home."""
    d = Path(root) / "home" / "data_files"
    d.mkdir(parents=True)
    write_home_fixture(d)
    preds = pd.read_csv(d / PREDICTIONS, sep="\t")
    today_row = preds.iloc[[0]].copy()
    today_row["game_id"] = "TODAY_PENDING"
    today_row["gameday"] = date.today().isoformat()
    for c in ("home_score", "away_score", "result", "total", "totalScore", "pointDiff",
              "underdogCovered", "overHit"):
        today_row[c] = 0
    today_row["spread_bet_return"] = -100.0
    today_row["prob_underdogCovered"] = 0.6
    preds = pd.concat([preds, today_row], ignore_index=True)
    # A deterministic upcoming game in the ≥65% band for the Spread Bets tab.
    up = pd.to_datetime(preds["gameday"]).dt.date > date.today()
    first_up = preds.index[up][0]
    preds.loc[first_up, "prob_underdogCovered"] = 0.70
    preds.loc[first_up, "edge_underdog_spread"] = 0.08

    past = preds[pd.to_datetime(preds["gameday"]).dt.date < date.today()]
    recent = past.sort_values("gameday", kind="stable")["game_id"].tolist()
    missing_id, disagree_id, shutout_id = recent[-1], recent[-2], recent[-3]
    s = preds["game_id"] == shutout_id
    preds.loc[s, ["home_score", "away_score", "result", "total", "totalScore", "pointDiff"]] = \
        [27, 0, 27, 27, 27, 27]

    sched = preds[["game_id", "season", "week", "gameday", "home_score", "away_score",
                   "result", "total"]].copy()
    sched["overtime"] = 0
    sched["game_type"] = "REG"
    not_played = ~pd.to_datetime(preds["gameday"]).dt.date.lt(date.today())
    sched.loc[not_played, ["home_score", "away_score", "result", "total", "overtime"]] = np.nan
    sched.loc[sched["game_id"] == disagree_id, ["home_score", "away_score", "result", "total"]] = \
        [27, 24, 3, 51]
    sched = sched[sched["game_id"] != missing_id]
    sched.to_csv(d / ds.SCHEDULE_NAME, sep="\t", index=False)

    g = preds["game_id"] == disagree_id
    preds.loc[g, ["home_score", "away_score", "result", "total", "totalScore", "pointDiff",
                  "underdogCovered", "overHit"]] = 0
    preds.loc[g, "spread_bet_return"] = -100.0
    preds.to_csv(d / PREDICTIONS, sep="\t", index=False)
    (d.parent / "expected.json").write_text(json.dumps(
        {"missing": missing_id, "disagree": disagree_id, "shutout": shutout_id}), encoding="utf-8")
    return d


class TestHomePage:
    """The real home page (predictions.py) on build_home_fixture's data."""

    @pytest.fixture
    def home_dir(self, tmp_path):
        return build_home_fixture(tmp_path)

    @pytest.fixture
    def home(self, home_dir, monkeypatch):
        import betting_log
        import dotenv

        monkeypatch.chdir(home_dir.parent)
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
        monkeypatch.setattr(betting_log, "append_recommendations", lambda *a, **k: 0)
        monkeypatch.setattr(betting_log, "grade_pending", lambda *a, **k: 0)
        monkeypatch.setattr(sys, "excepthook", sys.excepthook)

        def refuse(*a, **k):
            raise OSError("network blocked in this test")
        monkeypatch.setattr(socket.socket, "connect", refuse)
        monkeypatch.setattr(socket, "create_connection", refuse)
        return AppTest.from_file(str(ROOT / "predictions.py"), default_timeout=120)

    @staticmethod
    def results_table(at):
        for df in at.dataframe:
            if "result_status" in df.value.columns:
                return df.value
        return None

    def loaded(self, at):
        at.run()
        if self.results_table(at) is None:   # first pass may precede the background load
            at.run()
        return at

    @staticmethod
    def all_text(at):
        parts = []
        for kind in ("markdown", "caption", "info", "header", "subheader", "title", "warning", "error"):
            parts += [e.value for e in getattr(at, kind)]
        parts += [e.label for e in at.expander]
        return "\n".join(str(p) for p in parts)

    @staticmethod
    def expected(home_dir):
        return json.loads((home_dir.parent / "expected.json").read_text(encoding="utf-8"))

    @staticmethod
    def flagged_counts(home_dir):
        """(final, left out) for the spread performance block, from the fixture's own
        construction: every flagged past game is final except the missing and the
        disagreeing games and TODAY_PENDING; upcoming games are pending."""
        exp = TestHomePage.expected(home_dir)
        p = pd.read_csv(home_dir / PREDICTIONS, sep="\t")
        flagged = p[p["spread_bet_return"] != 0]
        past = pd.to_datetime(flagged["gameday"]).dt.date < date.today()
        final = past & ~flagged["game_id"].isin([exp["missing"], exp["disagree"]])
        return int(final.sum()), int((~final).sum())

    def test_pending_unavailable_inconsistent_and_final_games(self, home_dir, home):
        inputs = {n: hashlib.sha256((home_dir / n).read_bytes()).hexdigest()
                  for n in (PREDICTIONS, ds.SCHEDULE_NAME)}
        before = tree(home_dir)
        at = self.loaded(home)
        assert not at.exception, [e.message for e in at.exception]
        table = self.results_table(at)
        assert table is not None
        rows = table.set_index("game_id")
        exp = self.expected(home_dir)

        pending = rows.loc["TODAY_PENDING"]
        assert pending["result_status"] == ds.PENDING
        for c in ds.RESULT_COLUMNS:
            assert pd.isna(pending[c]), c

        assert rows.loc[exp["missing"], "result_status"] == ds.UNAVAILABLE
        assert pd.isna(rows.loc[exp["missing"], "home_score"])

        # Schedule 27-24, predictions 0-0: not shown as final, no 0-0 shown.
        bad = rows.loc[exp["disagree"]]
        assert bad["result_status"] == ds.INCONSISTENT
        for c in ds.RESULT_COLUMNS:
            assert pd.isna(bad[c]), c

        shut = rows.loc[exp["shutout"]]
        assert shut["result_status"] == ds.FINAL
        assert (shut["home_score"], shut["away_score"]) == (27, 0)

        preds = pd.read_csv(home_dir / PREDICTIONS, sep="\t").set_index("game_id")
        finals = rows[rows["result_status"] == ds.FINAL]
        assert len(finals) == len(rows) - 3
        for gid, r in finals.iterrows():   # consistent final scores and outcomes shown exactly
            for c in ds.RESULT_COLUMNS:
                assert r[c] == preds.loc[gid, c], (gid, c)

        # Performance counts include only verified finals.
        n_final, n_left_out = self.flagged_counts(home_dir)
        spread_bets = [m for m in at.metric if m.label == "Spread Bets"]
        assert [m.value for m in spread_bets] == [f"{n_final:,}"]
        assert any(f"{n_left_out:,} flagged game(s) without a verified final score" in c.value
                   for c in at.caption)

        assert tree(home_dir) == before   # nothing written
        assert {n: hashlib.sha256((home_dir / n).read_bytes()).hexdigest() for n in inputs} == inputs

    def test_wording_on_the_page(self, home_dir, home):
        at = self.loaded(home)
        assert not at.exception, [e.message for e in at.exception]
        text = self.all_text(at)
        assert "Regenerated model predictions vs results" in text
        assert ds.REGENERATED_NOTE in text
        assert ("1 pending, 1 with no result available, "
                "1 where the schedule and predictions files disagree") in text
        assert "Model ≥65% (" in text                    # neutral band label on Spread Bets
        for phrase in ("Elite", "value bet opportunity", "best risk-adjusted", "max units"):
            assert phrase not in text, phrase
        assert not at.error, [e.value for e in at.error]
        assert not at.warning, [e.value for e in at.warning]
