"""Recommendation-log display (betting_log display helpers, Betting Log tab,
alert view and RSS). Temporary fixtures only; data_files/ is never read."""
import sys
from datetime import datetime, timedelta

import pandas as pd
import pytest

import betting_log as bl
from test_team_features import ROOT


def row(**kw):
    base = dict(game_id="2026_07_DAL_PHI", season=2026, week=7, home_team="PHI",
                away_team="DAL", bet_type="spread", recommended_team="PHI", spread_line=-4.5,
                bet_spread=4.5, bet_odds=-110.0, odds_source="assumed -110 model recommendation")
    base.update(kw)
    return pd.Series(base)


class TestRecommendationText:
    def test_home_underdog_raw_negative_shows_plus(self):
        # nflverse spread_line -4.5: away DAL favoured by 4.5, so home PHI is +4.5.
        r = row()
        assert bl.recommendation_text(r) == "PHI +4.5"
        assert "-4.5" not in bl.recommendation_text(r)

    def test_raw_source_spread_is_never_attached_to_the_team(self):
        # Only bet_spread decides the handicap shown; the raw line is ignored.
        assert bl.recommendation_text(row(spread_line=99.0)) == "PHI +4.5"
        assert bl.recommendation_text(row(spread_line=None)) == "PHI +4.5"

    def test_away_underdog_with_positive_raw_line(self):
        # spread_line +3: home PHI favoured by 3, so away DAL is the dog at +3.
        r = row(recommended_team="DAL", spread_line=3.0, bet_spread=3.0)
        assert bl.recommendation_text(r) == "DAL +3"

    @pytest.mark.parametrize("handicap,text", [
        (3.0, "+3"), (7, "+7"), (4.5, "+4.5"), (0.5, "+0.5"), (10.0, "+10"),
        (-3.0, "-3"), (-6.5, "-6.5"), (0.0, "PK"), ("2.5", "+2.5"),
    ])
    def test_whole_and_half_point_formatting(self, handicap, text):
        assert bl.format_handicap(handicap) == text
        assert bl.recommendation_text(row(bet_spread=handicap)) == f"PHI {text}"

    @pytest.mark.parametrize("overrides", [
        dict(recommended_team=None),
        dict(recommended_team=float("nan")),
        dict(recommended_team="Pick"),               # legacy pick-em rows
        dict(recommended_team="Unknown"),
        dict(recommended_team="NYG"),                # not in this game
        dict(bet_spread=None),                       # legacy row: no recorded handicap
        dict(bet_spread=float("nan")),
        dict(bet_spread=""),
        dict(bet_spread="abc"),
        dict(bet_spread=float("inf")),
        dict(bet_spread=3.25),                       # not a half-point
        dict(bet_spread=61.0),                       # implausible
    ])
    def test_missing_or_invalid_is_unavailable(self, overrides):
        assert bl.recommendation_text(row(**overrides)) == bl.RECOMMENDATION_UNAVAILABLE

    def test_legacy_row_without_bet_spread_is_not_derived_from_raw_line(self):
        # Grading may convert the raw line for such rows; the display does not.
        r = pd.Series(dict(home_team="SEA", away_team="NE", recommended_team="NE",
                           spread_line=3.5, bet_type="spread"))
        assert bl.recommendation_text(r) == bl.RECOMMENDATION_UNAVAILABLE

    def test_helpers_do_not_modify_the_row(self):
        r = row()
        before = r.copy()
        bl.recommendation_text(r), bl.odds_text(r)
        pd.testing.assert_series_equal(r, before)


class TestOddsText:
    @pytest.mark.parametrize("odds,source,text", [
        (-110.0, "assumed -110 model recommendation", "-110 (assumed; not a sportsbook offer)"),
        (-110.0, "assumed -110 (legacy row, no price recorded)",
         "-110 (assumed; not a sportsbook offer)"),
        (-110.0, None, "-110 (assumed; not a sportsbook offer)"),
        (None, None, "-110 (assumed; no price recorded; not a sportsbook offer)"),
        (105.0, "betmgm_ca_on archived quote", "+105 (betmgm_ca_on archived quote)"),
    ])
    def test_labels(self, odds, source, text):
        assert bl.odds_text(row(bet_odds=odds, odds_source=source)) == text

    def test_labels_name_the_assumption(self):
        assert "not a verified sportsbook offer" in bl.ODDS_HELP
        assert bl.RAW_SPREAD_LABEL == "Raw source spread"
        assert "home-favourite positive" in bl.RAW_SPREAD_HELP


class TestSettlementUnchanged:
    """The display change touches no settlement code: the bets still grade
    exactly as recorded."""

    def _grade(self, tmp_path, rec, home_score, away_score, **extra):
        path = tmp_path / "log.csv"
        pd.DataFrame([dict(game_id="g", home_team="PHI", away_team="DAL", bet_type="spread",
                           recommended_team=rec, bet_result="pending", **extra)]
                     ).to_csv(path, index=False)
        scores = pd.DataFrame([dict(game_id="g", gameday="2025-01-01", home_score=home_score,
                                    away_score=away_score, spread_line=-4.5)])
        assert bl.grade_pending(scores, path) == 1
        return pd.read_csv(path).iloc[0]

    @pytest.mark.parametrize("home,away,result", [(20, 24, "win"), (17, 24, "loss"),
                                                  (20, 20, "win")])
    def test_recorded_phi_plus_4_5(self, tmp_path, home, away, result):
        out = self._grade(tmp_path, "PHI", home, away, spread_line=-4.5, bet_spread=4.5,
                          bet_odds=-110.0, odds_source="assumed -110 model recommendation")
        assert out.bet_result == result
        assert bl.recommendation_text(out) == "PHI +4.5"

    def test_legacy_row_still_grades_from_the_converted_raw_line(self, tmp_path):
        # PHI (home) dog: raw -4.5 -> +4.5 for PHI; PHI loses by 4 -> covers.
        out = self._grade(tmp_path, "PHI", 20, 24, spread_line=-4.5)
        assert out.bet_result == "win"
        assert bl.recommendation_text(out) == bl.RECOMMENDATION_UNAVAILABLE   # display only

    def test_append_schema_and_dedup_unchanged(self, tmp_path):
        log = str(tmp_path / "log.csv")
        soon = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
        preds = pd.DataFrame([dict(game_id="2026_07_DAL_PHI", season=2026, week=7,
                                   gameday=soon, home_team="PHI", away_team="DAL",
                                   spread_line=-4.5, total_line=44.5, home_score=0,
                                   away_score=0, underdogCovered=0, spreadPush=0,
                                   pred_spreadCovered_optimal=1, prob_underdogCovered=0.559,
                                   edge_underdog_spread=0.04)])
        assert bl.append_recommendations(preds, log) == 1
        assert bl.append_recommendations(preds, log) == 0
        out = pd.read_csv(log)
        assert list(out.columns) == bl.LOG_COLUMNS
        r = out.iloc[0]
        assert (r.recommended_team, r.spread_line, r.bet_spread) == ("PHI", -4.5, 4.5)
        assert bl.recommendation_text(r) == "PHI +4.5"


class TestRss:
    def test_rss_shows_recommendation_raw_line_and_assumed_odds(self):
        sys.path[:0] = [str(ROOT / "scripts")]
        import generate_rss as g
        df = pd.DataFrame([row(gameday="2026-10-26", model_probability=0.559, edge=0.04),
                           row(game_id="2026_01_WAS_PHI", recommended_team="WAS",
                               spread_line=5.5, bet_spread=None, gameday="2026-09-13",
                               model_probability=0.555, edge=0.02)])
        out = g.build_rss_items(df)
        assert "Recommendation: PHI +4.5" in out
        assert "Raw source spread: -4.5 (home-favourite positive)" in out
        assert "Odds: -110 (assumed; not a sportsbook offer)" in out
        assert "Spread: -4.5" not in out and "PHI -4.5" not in out
        assert "Recommendation unavailable | Raw source spread: 5.5" in out


def test_app_displays_use_the_helpers():
    src = (ROOT / "predictions.py").read_text(encoding="utf-8")
    tab = src[src.index("#### 📋 Detailed Betting Log"):src.index("**🔄 How results are updated:**")]
    assert "recommendation_text" in tab and "RAW_SPREAD_LABEL" in tab and "odds_text" in tab
    assert "'Bet On'" not in tab and "NumberColumn('Spread'" not in tab
    alert = src[src.index("# Recommendation and quick stats"):src.index("# Present the full log")]
    assert "recommendation_text" in alert and "metric('Spread'" not in alert
