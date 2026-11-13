"""Conditional spread display labels; synthetic data only, no probability edits."""
import pandas as pd
import pytest
import display_status as ds


@pytest.mark.parametrize("basis", ["legacy", "conditional", "mixed", "unknown", "missing"])
def test_notice_classifies_without_changing_values(basis):
    frame = pd.DataFrame({"prob_underdogCovered": [0.55, 0.65], "ev_spread": [5.0, 24.0]})
    if basis != "legacy":
        frame["spread_probability_basis"] = {
            "conditional": ["conditional_no_push"]*2,
            "mixed": ["conditional_no_push", "legacy"],
            "unknown": ["invented"]*2,
            "missing": ["conditional_no_push", None],
        }[basis]
        frame["spread_ev_basis"] = "per_100_non_push_stakes_assumed_minus110"
    before = frame.copy(deep=True)
    text = ds.spread_basis_notice(frame)
    pd.testing.assert_frame_equal(frame, before)
    if basis == "legacy":
        assert "legacy file (not declared)" in text
        assert "Do not assume" in text
    elif basis == "conditional":
        assert "P(underdog covers | no push)" in text
        assert "per $100 of non-push stakes" in text and "not unconditional EV" in text
    else:
        assert "mixed, missing or unrecognized" in text
        assert "P(underdog covers | no push)" not in text


@pytest.mark.parametrize("ev", [None, "unknown", "mixed"])
def test_conditional_probability_does_not_infer_ev_basis(ev):
    frame = pd.DataFrame({"spread_probability_basis": ["conditional_no_push"]*2})
    if ev is not None:
        frame["spread_ev_basis"] = ["per_100_non_push_stakes_assumed_minus110", ev]
    text = ds.spread_basis_notice(frame)
    assert "P(underdog covers | no push)" in text
    assert "EV basis is unavailable or unrecognized" in text
    assert "Displayed spread EV is per $100" not in text


def test_empty_data_has_no_claim():
    assert ds.spread_basis_notice(None) == ""
    assert ds.spread_basis_notice(pd.DataFrame()) == ""


# Reuse the existing real-home-page fixture and guarded AppTest setup.
from test_display_status import TestHomePage as _Home


class TestRenderedBasis:
    home_dir = _Home.home_dir
    home = _Home.home
    loaded = _Home.loaded
    results_table = staticmethod(_Home.results_table)
    all_text = staticmethod(_Home.all_text)
    @pytest.mark.parametrize("basis", ["legacy", "conditional", "mixed"])
    def test_basis_notice_on_real_home_page(self, home_dir, home, basis):
        from test_home_downloads import PREDICTIONS
        file = home_dir / PREDICTIONS
        frame = pd.read_csv(file, sep="\t")
        if basis != "legacy":
            frame["spread_probability_basis"] = "conditional_no_push"
            frame["spread_ev_basis"] = "per_100_non_push_stakes_assumed_minus110"
            if basis == "mixed":
                frame.loc[frame.index[0], "spread_probability_basis"] = "legacy"
            frame.to_csv(file, sep="\t", index=False)
        expected_values = frame[["game_id", "prob_underdogCovered"]].copy()
        import streamlit as st
        st.cache_data.clear()
        at = self.loaded(home)
        assert not at.exception, [e.message for e in at.exception]
        text = self.all_text(at)
        expected = ds.spread_basis_notice(frame)
        assert expected in text
        assert "Current predictions file." in text
        assert "Historical recommendation-log and frozen records retain their own recorded meaning." in text
        pd.testing.assert_frame_equal(pd.read_csv(file, sep="\t")[["game_id", "prob_underdogCovered"]], expected_values)


    def test_game_detail_retains_conditional_notice(self, home_dir, home):
        import streamlit as st
        from test_home_downloads import PREDICTIONS
        file = home_dir / PREDICTIONS
        frame = pd.read_csv(file, sep="\t")
        frame["spread_probability_basis"] = "conditional_no_push"
        frame["spread_ev_basis"] = "per_100_non_push_stakes_assumed_minus110"
        frame.to_csv(file, sep="\t", index=False)
        st.cache_data.clear()
        home.query_params["game"] = str(frame.iloc[0].game_id)
        home.run()
        assert not home.exception, [e.message for e in home.exception]
        assert "Game Details" in self.all_text(home)
        assert ds.spread_basis_notice(frame.iloc[[0]]) in self.all_text(home)


@pytest.mark.parametrize("conditional", [False, True])
def test_pdf_retains_basis_notice(conditional):
    import home_downloads as hd
    from test_home_downloads import home_predictions_fixture, pdf_text
    frame = home_predictions_fixture(n_past=3)
    if conditional:
        frame["spread_probability_basis"] = "conditional_no_push"
        frame["spread_ev_basis"] = "per_100_non_push_stakes_assumed_minus110"
    before = frame.copy(deep=True)
    text = pdf_text(hd.generate_predictions_pdf(frame))
    assert (b"conditional on no push" if conditional else b"legacy file") in text
    if conditional:
        assert b"not unconditional EV" in text
    pd.testing.assert_frame_equal(frame, before)
