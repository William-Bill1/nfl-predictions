"""Pick 6 calculator on the Player Props page: a historical comparison at an
entered line, worded as such (player_props/pick6.py). Synthetic data only; no
network, data_files/ is never read."""
import re
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from player_props import pick6
from test_team_features import ROOT

PAGE = ROOT / "pages" / "2_Player_Props.py"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def original(values, line):
    """The calculator's calculation as it was, verbatim in substance."""
    s = pd.Series(values, dtype=float)
    total_games = len(s)
    games_over = len(s[s > line])
    games_under = total_games - games_over
    prob_over = (games_over + 1) / (total_games + 2)
    prob_under = 1 - prob_over
    if prob_over >= 0.60:
        recommendation, confidence = "OVER", prob_over
    elif prob_under >= 0.60:
        recommendation, confidence = "UNDER", prob_under
    else:
        recommendation = "OVER" if prob_over > prob_under else "UNDER"
        confidence = max(prob_over, prob_under)
    return games_over, games_under, prob_over, prob_under, recommendation, confidence


class TestCalculationUnchanged:
    @pytest.mark.parametrize("seed", range(200))
    def test_matches_the_original_formula(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(0, 12))
        values = rng.integers(0, 30, n).astype(float)
        line = float(rng.choice([5.0, 10.5, 15.0, 20.5]))
        over, not_over, p_over, p_under, side, rate = original(values, line)
        got = pick6.historical_hit_rate(values, line)
        assert (got["over"], got["not_over"], got["side"]) == (over, not_over, side)
        assert (got["prob_over"], got["prob_under"], got["rate"]) == (p_over, p_under, rate)

    def test_hand_values(self):
        got = pick6.historical_hit_rate([250, 300, 180, 260, 240, 90, 310, 275, 225], 242.5)
        assert (got["games"], got["over"], got["under"], got["at_line"]) == (9, 5, 4, 0)
        assert got["prob_over"] == 6 / 11 and got["side"] == "OVER" and got["rate"] == 6 / 11

    def test_ten_of_ten_over_is_smoothed(self):
        got = pick6.historical_hit_rate([300] * 10, 100.5)
        assert got["rate"] == 11 / 12 and got["side"] == "OVER"      # not 100%

    def test_no_games(self):
        got = pick6.historical_hit_rate([], 100.5)
        assert got["games"] == 0 and got["prob_over"] == 0.5 and got["side"] == "UNDER"


class TestCountsAndHits:
    def test_all_games_equal_to_the_line(self):
        # The reproduced case: never "UNDER" when no game was below the line.
        got = pick6.historical_hit_rate([10] * 10, 10)
        assert (got["over"], got["under"], got["at_line"]) == (0, 0, 10)
        assert got["rate"] == 11 / 12 and got["side"] == "UNDER"           # calculation unchanged
        assert pick6.side_label(got) == "NOT OVER (below or equal)"
        assert pick6.counts_text(got) == "10 games: 0 above the line, 0 below, 10 exactly at the line"
        assert pick6.hits_text(got) == "10 of 10 games not over the line (0 below, 10 equal)"
        assert pick6.formula_text(got) == \
            "(games not over + 1) / (games + 2) = (10 + 1) / (10 + 2)"
        assert pick6.AT_LINE_NOTE in pick6.notes(got)

    def test_mixed_below_equal_and_above(self):
        got = pick6.historical_hit_rate([10, 10, 12, 8, 7], 10.0)
        assert (got["over"], got["under"], got["at_line"], got["not_over"]) == (1, 2, 2, 4)
        assert got["rate"] == 5 / 7 and pick6.side_label(got) == "NOT OVER (below or equal)"
        assert pick6.counts_text(got) == "5 games: 1 above the line, 2 below, 2 exactly at the line"
        assert pick6.hits_text(got) == "4 of 5 games not over the line (2 below, 2 equal)"
        assert pick6.formula_text(got) == "(games not over + 1) / (games + 2) = (4 + 1) / (5 + 2)"

    def test_over_side_with_equal_games(self):
        got = pick6.historical_hit_rate([12, 13, 14, 10, 11], 10.0)
        assert got["side"] == "OVER" and pick6.side_label(got) == "OVER"
        assert pick6.hits_text(got) == "4 of 5 games above the line"
        assert pick6.formula_text(got) == "(games above + 1) / (games + 2) = (4 + 1) / (5 + 2)"
        assert pick6.AT_LINE_NOTE in pick6.notes(got)       # the equal game is still explained

    @pytest.mark.parametrize("values,line,label", [
        ([12, 12, 8, 8], 10.0, "UNDER"),                      # tie, nothing equal
        ([12, 12, 10, 8], 10.0, "NOT OVER (below or equal)"),  # tie, one equal
    ])
    def test_tied_counts(self, values, line, label):
        got = pick6.historical_hit_rate(values, line)
        assert got["tied"] and got["rate"] == 0.5 and got["side"] == "UNDER"
        assert pick6.side_label(got) == label
        assert pick6.TIE_NOTE in pick6.notes(got)
        assert "higher smoothed rate" in pick6.SIDE_LABEL

    def test_half_point_line(self):
        # This fixture has 5 of 9 games above 242.5: (5 + 1) / (9 + 2) = 54.5%.
        got = pick6.historical_hit_rate([250, 300, 180, 260, 240, 90, 310, 275, 225], 242.5)
        assert got["rate"] == 6 / 11 and f"{got['rate']:.1%}" == "54.5%"
        assert pick6.side_label(got) == "OVER"
        assert pick6.counts_text(got) == "9 games: 5 above the line, 4 below"
        assert pick6.notes(got) == []

    def test_guide_example_six_of_nine(self):
        # The how-to guide's example: 6 of 9 games over 242.5 -> (6 + 1) / (9 + 2) = 63.6%.
        got = pick6.historical_hit_rate([250, 300, 180, 260, 250, 90, 310, 275, 225], 242.5)
        assert (got["games"], got["over"], got["under"]) == (9, 6, 3)
        assert got["rate"] == 7 / 11 and f"{got['rate']:.1%}" == "63.6%"
        assert pick6.side_label(got) == "OVER" and got["side"] == "OVER"
        guide = PAGE.read_text(encoding="utf-8")
        assert "Result: **OVER**, smoothed historical hit rate 63.6% (6 of 9 games over)" in guide

    def test_under_without_equal_games(self):
        got = pick6.historical_hit_rate([10, 11, 9, 8], 10.5)
        assert pick6.side_label(got) == "UNDER"
        assert pick6.hits_text(got) == "3 of 4 games below the line"
        assert pick6.formula_text(got) == "(games below + 1) / (games + 2) = (3 + 1) / (4 + 2)"

    def test_over_hits(self):
        got = pick6.historical_hit_rate([300] * 10, 100.5)
        assert pick6.hits_text(got) == "10 of 10 games above the line"

    @pytest.mark.parametrize("seed", range(100))
    def test_formula_text_matches_the_rate(self, seed):
        rng = np.random.default_rng(seed)
        values = rng.integers(5, 15, int(rng.integers(1, 12))).astype(float)
        got = pick6.historical_hit_rate(values, 10.0)
        a, b = re.search(r"= \((\d+) \+ 1\) / \((\d+) \+ 2\)$", pick6.formula_text(got)).groups()
        assert (int(a) + 1) / (int(b) + 2) == pytest.approx(got["rate"], abs=1e-15)
        if pick6.side_label(got) == "UNDER":
            assert got["at_line"] == 0                        # UNDER only when nothing equalled

    def test_no_settlement_rule_is_claimed(self):
        note = pick6.AT_LINE_NOTE
        assert ("This calculator's counting convention does not describe sportsbook settlement "
                "rules. Check the operator's rules separately.") in note
        for text in (note, pick6.EXPLANATION, pick6.TIE_NOTE):
            assert "push" not in text.lower() and "wins" not in text.lower()


class TestWording:
    def test_explanation_says_what_the_rate_is_and_is_not(self):
        text = pick6.EXPLANATION
        assert "Laplace-smoothed" in text and "historical" in pick6.RATE_LABEL.lower()
        for claim in ("not a model prediction", "a calibrated win probability",
                      "a betting recommendation"):
            assert claim in text
        assert pick6.RATE_LABEL == "Smoothed historical hit rate"
        assert "model" not in pick6.INTRO.lower() and "line" in pick6.INTRO

    def _calculator_source(self):
        text = PAGE.read_text(encoding="utf-8")
        start = text.index("# TAB 2: DraftKings Pick 6")
        return text[start:text.index("# TAB 3", start)]

    @pytest.mark.parametrize("leftover", [
        "Model output", "model's prediction", "auto-selects tier", "See the Model Output",
        ">Recommendation<", "weighted heavily", "defense rank", "prediction_source",
        "Machine Learning", "predict_proba", "load_model(",
    ])
    def test_no_model_claims_remain_in_the_calculator(self, leftover):
        assert leftover not in self._calculator_source()

    def test_calculator_uses_the_shared_helper(self):
        src = self._calculator_source()
        assert "historical_hit_rate(recent_games[stat_col], dk_line)" in src
        assert re.search(r"games_over\s*=", src) is None


# ---------------------------------------------------------------------------
# The rendered page (AppTest) with synthetic files in a temporary directory
# ---------------------------------------------------------------------------

def _write_fixture(root: Path):
    data = root / "data_files"
    data.mkdir()
    pd.DataFrame([
        dict(gsis_id="QB1", short_name="T.Tester", display_name="Test Tester", last_name="Tester",
             position="QB"),
    ]).to_csv(data / "players.csv", index=False)
    dates = pd.date_range("2025-09-07", periods=10, freq="7D").strftime("%Y-%m-%d")
    yards = [250, 300, 180, 260, 240, 90, 310, 275, 225, 242.5]
    passing = pd.DataFrame({
        "player_id": "QB1", "player_name": "T.Tester", "game_id": [f"2025_{i:02d}_AAA_BBB" for i in range(1, 11)],
        "season": 2025, "week": range(1, 11), "team": "AAA", "opponent": "BBB",
        "passing_yards": yards, "pass_tds": 2, "completions": 20, "attempts": 30, "interceptions": 0,
        "game_date": dates, "completion_pct": 66.7})
    for c in ("passing_yards", "pass_tds", "completions", "attempts", "interceptions"):
        for w in (3, 5, 10):
            passing[f"{c}_L{w}"] = passing[c]
    passing["is_home"] = 1
    passing["days_rest"] = 7
    passing.to_csv(data / "player_passing_stats.csv", index=False)
    empty_cols = ["player_id", "player_name", "game_id", "season", "week", "team", "opponent",
                  "game_date"]
    pd.DataFrame(columns=empty_cols + ["rushing_yards", "rush_tds", "rush_attempts", "ypc"]).to_csv(
        data / "player_rushing_stats.csv", index=False)
    pd.DataFrame(columns=empty_cols + ["receiving_yards", "receptions", "rec_tds", "targets"]).to_csv(
        data / "player_receiving_stats.csv", index=False)
    return yards


def _text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [w.value for w in at.warning] + [i.value for i in at.info]
    return "\n".join(str(p) for p in parts)


@pytest.fixture
def rendered(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import streamlit as st
    yards = _write_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    st.cache_data.clear()            # the page caches its loaders by relative path
    at = AppTest.from_file(str(PAGE), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.text_input(key="dk_player_search").input("Tester").run()
    assert not at.exception, [e.value for e in at.exception]
    return at, yards


class TestRenderedCalculator:
    def test_card_shows_the_smoothed_historical_rate_with_counts(self, rendered):
        at, yards = rendered
        line = at.number_input(key=[k for k in at.session_state.filtered_state
                                    if str(k).startswith("dk_line_value_")][0]).value
        text = _text(at)
        got = pick6.historical_hit_rate(yards, line)
        [card] = [m.value for m in at.markdown if pick6.RATE_LABEL + ": <strong>" in str(m.value)]
        assert f"Smoothed historical hit rate: <strong>{got['rate']:.1%}</strong> {pick6.side_label(got)}" in card
        # the card itself carries the sample, the observed hits and what the rate is
        assert pick6.hits_text(got) in card and pick6.counts_text(got) in card
        assert pick6.EXPLANATION in card
        assert f">{pick6.side_label(got)}</p>" in card and pick6.SIDE_LABEL in card
        assert pick6.formula_text(got) in card
        assert "Model output" not in text and "model's prediction" not in text

    def test_at_line_games_are_shown_separately(self, rendered):
        at, yards = rendered
        key = [k for k in at.session_state.filtered_state if str(k).startswith("dk_line_value_")][0]
        at.number_input(key=key).set_value(242.5).run()
        assert not at.exception
        text = _text(at)
        got = pick6.historical_hit_rate(yards, 242.5)
        assert got["at_line"] == 1
        # mixed: 4 below, 1 equal, 5 above
        assert (got["under"], got["at_line"], got["over"]) == (4, 1, 5)
        assert "1 exactly at the line" in text and pick6.AT_LINE_NOTE in text
        assert f"{got['rate']:.1%}" in text and pick6.hits_text(got) in text

    def test_warnings_stay_visible(self, rendered):
        at, _ = rendered
        notice = at.warning[0].value
        assert "Do not use them to guide wagers" in notice
        assert "Stale player statistics" in notice or "Player statistics" in notice

    def test_intro_and_help_describe_a_historical_comparison(self, rendered):
        at, _ = rendered
        text = _text(at)
        assert pick6.INTRO in text
        assert "No model is scored here" in text
        assert "auto-selects tier" not in text



def _card(at):
    [card] = [m.value for m in at.markdown if pick6.RATE_LABEL + ": <strong>" in str(m.value)]
    return card


def _render(tmp_path, monkeypatch, yards, line):
    from streamlit.testing.v1 import AppTest
    _write_fixture(tmp_path)
    p = tmp_path / "data_files" / "player_passing_stats.csv"
    df = pd.read_csv(p)
    df["passing_yards"] = yards
    df.to_csv(p, index=False)
    monkeypatch.chdir(tmp_path)
    import streamlit as st
    st.cache_data.clear()            # the page caches its loaders by relative path
    at = AppTest.from_file(str(PAGE), default_timeout=60)
    at.run()
    at.text_input(key="dk_player_search").input("Tester").run()
    key = [k for k in at.session_state.filtered_state if str(k).startswith("dk_line_value_")][0]
    at.number_input(key=key).set_value(line).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


class TestRenderedEquality:
    def test_all_games_equal_to_the_line(self, tmp_path, monkeypatch):
        at = _render(tmp_path, monkeypatch, [200.0] * 10, 200.0)
        card = _card(at)
        assert ">NOT OVER (below or equal)</p>" in card and ">UNDER</p>" not in card
        assert "91.7%</strong> NOT OVER (below or equal)" in card
        assert "10 games: 0 above the line, 0 below, 10 exactly at the line" in card
        assert "(games not over + 1) / (games + 2) = (10 + 1) / (10 + 2)" in card
        assert pick6.AT_LINE_NOTE in card

    def test_tied_counts(self, tmp_path, monkeypatch):
        # 5 above, 5 below a whole-number line: 50% each way
        at = _render(tmp_path, monkeypatch, [210.0] * 5 + [190.0] * 5, 200.0)
        card = _card(at)
        assert "50.0%</strong> UNDER" in card and pick6.TIE_NOTE in card
        assert pick6.SIDE_LABEL in card and pick6.AT_LINE_NOTE not in card

    def test_half_point_line_is_unchanged(self, tmp_path, monkeypatch):
        at = _render(tmp_path, monkeypatch, [250, 300, 180, 260, 240, 90, 310, 275, 225, 230], 242.5)
        card = _card(at)
        got = pick6.historical_hit_rate([250, 300, 180, 260, 240, 90, 310, 275, 225, 230], 242.5)
        assert f"{got['rate']:.1%}</strong> {pick6.side_label(got)}" in card
        assert "exactly at the line" not in card and pick6.AT_LINE_NOTE not in card
