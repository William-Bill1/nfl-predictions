"""Player-prop containment (player_props/containment.py and the three pages that
show prop outputs). Temporary fixtures and page sources only; data_files/ is
never read and no model or prediction is touched."""
import re
from datetime import date

import pandas as pd
import pytest

from player_props import containment as c
from test_team_features import ROOT

PAGES = {name: (ROOT / "pages" / name).read_text(encoding="utf-8")
         for name in ("2_Player_Props.py", "3_Parlay_Builder.py", "4_Model_Performance.py")}


class TestNotice:
    def test_states_both_verified_issues_and_the_wager_warning(self):
        text = c.notice_markdown(date(2026, 1, 4), 2026)
        assert "Do not use them to guide wagers" in text
        assert ("Stale player statistics" in text and "end on 2026-01-04" in text
                and "before the 2026 season began" in text and "uses no 2026 games" in text)
        assert "Random home/away training values" in text and "assigned at random" in text
        assert "not validated betting confidence" in text
        assert "no figure has been recalculated" in text

    def test_scope_separates_spread_model_and_unverified_nightly_models(self):
        text = c.notice_markdown(date(2026, 1, 4), 2026)
        assert "player props only" in text and "game spread model" in text
        assert "not covered by this notice" in text
        assert "committed model files in `player_props/models/`" in text
        assert "nightly runs are not saved, so they remain unverified" in text

    @pytest.mark.parametrize("stats_date,season,expect,absent", [
        (date(2026, 1, 4), 2025, "end on 2026-01-04. They are not refreshed", "Stale"),
        (date(2026, 10, 18), 2026, "end on 2026-10-18. They are not refreshed", "Stale"),
        (date(2026, 1, 4), None, "end on 2026-01-04. They are not refreshed", "Stale"),
        (None, 2026, "could not be checked", "end on"),
    ])
    def test_stale_claim_only_when_verified(self, stats_date, season, expect, absent):
        text = c.notice_markdown(stats_date, season)
        assert expect in text and absent not in text
        assert "Random home/away training values" in text   # always stated


class TestLatestStatsDate:
    def _write(self, d, name, dates):
        pd.DataFrame({"game_date": dates, "x": 1}).to_csv(d / name, index=False)

    def test_latest_across_files(self, tmp_path):
        self._write(tmp_path, "player_passing_stats.csv", ["2025-12-28", "2026-01-04"])
        self._write(tmp_path, "player_rushing_stats.csv", ["2025-09-07"])
        self._write(tmp_path, "player_receiving_stats.csv", ["2026-01-03", "bad"])
        assert c.latest_stats_date(tmp_path) == date(2026, 1, 4)

    def test_missing_or_unreadable_files(self, tmp_path):
        assert c.latest_stats_date(tmp_path) is None
        (tmp_path / "player_passing_stats.csv").write_text("no_date_column\n1\n")
        (tmp_path / "player_rushing_stats.csv").write_bytes(b"")
        self._write(tmp_path, "player_receiving_stats.csv", ["nope"])
        assert c.latest_stats_date(tmp_path) is None

    def test_reads_without_writing(self, tmp_path):
        self._write(tmp_path, "player_passing_stats.csv", ["2026-01-04"])
        before = {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}
        c.latest_stats_date(tmp_path)
        assert before == {p.name: p.stat().st_mtime_ns for p in tmp_path.iterdir()}


class TestLabels:
    @pytest.mark.parametrize("p,label", [
        (0.90, "≥65% band"), (0.65, "≥65% band"), (0.6499, "60–65% band"), (0.60, "60–65% band"),
        (0.5999, "55–60% band"), (0.55, "55–60% band"), (0.5499, "<55% band"), (0.50, "<55% band"),
    ])
    def test_band_boundaries_match_the_old_tiers(self, p, label):
        assert c.band_label(p) == label

    @pytest.mark.parametrize("flag,label", [
        (True, c.HOLDOUT_PASSED), ("True", c.HOLDOUT_PASSED), (1, c.HOLDOUT_PASSED),
        ("1", c.HOLDOUT_PASSED), (False, c.HOLDOUT_NOT_PASSED), ("False", c.HOLDOUT_NOT_PASSED),
        (0, c.HOLDOUT_NOT_PASSED), (float("nan"), c.HOLDOUT_NOT_PASSED), (None, c.HOLDOUT_NOT_PASSED),
    ])
    def test_holdout_label_matches_the_page_filter(self, flag, label):
        # Same truth rule as the page's "passed the holdout check" filter.
        assert c.holdout_label(flag) == label
        assert (label == c.HOLDOUT_PASSED) == (str(flag).strip().lower() in ("true", "1"))

    @pytest.mark.parametrize("seasons,expected", [
        ([2026, 2026], 2026), ([2025.0], 2025), ([2025, 2026], None), ([2026, None], None),
        (["x"], None), ([2026.5], None),
    ])
    def test_single_season(self, seasons, expected):
        assert c.single_season(pd.DataFrame({"season": seasons})) == expected

    def test_single_season_without_data(self):
        assert c.single_season(None) is None and c.single_season(pd.DataFrame()) is None


# Claims that presented prop outputs as validated betting confidence or advice.
BANNED = {
    "2_Player_Props.py": ["🔥 Elite", "💪 Strong", "✅ Good (55-60%)", "🔥 ELITE", "💪 STRONG",
                          "✅ GOOD", "tested (reliable)", "'✅ tested'", "Top Player Prop Recommendations",
                          "⭐ Top Picks", "above breakeven", "ROC-AUC scores: Passing 0.72",
                          "ROC-AUC: 0.72-0.84", "data-driven recommendations",
                          "bigger is always better"],
    "3_Parlay_Builder.py": ["Higher confidence = better value", "Winning Parlay", "Elite tier",
                            "Top Recommendations", "Our Edge", 'st.metric("Edge"', "good value",
                            "Betting Strategy Tips", "high-confidence props"],
    "4_Model_Performance.py": ["Best Strategy", "Bet on {best_threshold", "Profitable Thresholds",
                               "actually working"],
}


class TestPages:
    @pytest.mark.parametrize("page", sorted(BANNED))
    def test_unvalidated_claims_are_gone(self, page):
        for phrase in BANNED[page]:
            assert phrase not in PAGES[page], (page, phrase)

    @pytest.mark.parametrize("page,first_output", [
        ("2_Player_Props.py", "stats_available = all("),
        ("3_Parlay_Builder.py", "if predictions_df is None or predictions_df.empty:"),
        ("4_Model_Performance.py", "def get_dataframe_height"),
    ])
    def test_notice_shown_before_any_prop_output(self, page, first_output):
        src = PAGES[page]
        call = src.index("st.warning(notice_markdown(latest_stats_date(Path('data_files'))")
        assert call < src.index(first_output)

    def test_parlay_export_carries_the_warning(self):
        parlay = PAGES["3_Parlay_Builder.py"]
        export = parlay[parlay.index('if st.button("📄 Export Parlay Summary"):'):
                        parlay.index("st.download_button(")]
        assert ("NOT VALIDATED FOR BETTING: player prop outputs have unresolved data and "
                "training problems.") in export
        assert "Edge" not in export and "assumes independent legs" in export

    def test_figures_and_filters_are_unchanged(self):
        props, parlay = PAGES["2_Player_Props.py"], PAGES["3_Parlay_Builder.py"]
        # Band counts, the probability floor and the holdout filter default as before.
        for expr in ("predictions['confidence'] >= 0.65",
                     "(predictions['confidence'] >= 0.60) & (predictions['confidence'] < 0.65)",
                     "(predictions['confidence'] >= 0.55) & (predictions['confidence'] < 0.60)",
                     'st.slider("Minimum Confidence", 0.50, 0.90, 0.60, 0.05, key="min_conf_top")',
                     "key=\"reliable_only_top\""):
            assert expr in props, expr
        checkbox = props[props.index("tested_only = st.checkbox("):]
        assert re.search(r"value=True,\s*\n\s*key=\"reliable_only_top\"", checkbox)
        # Parlay maths unchanged; only the always-zero "Edge" display was removed.
        assert "combined_prob = selected_props['confidence'].prod()" in parlay
        assert "predictions_df[predictions_df['confidence'] >= 0.55]" in parlay
