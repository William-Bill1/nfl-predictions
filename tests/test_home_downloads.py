"""Home-page sidebar export controls (home_downloads.py, wired in home_page()).

Rendered with Streamlit's AppTest against temporary fixtures: the download links
are decoded and compared with the fixture data, the PDF is generated, and the
fixture files are checked byte for byte afterwards. One test renders the real
home page (predictions.py) from a temporary data directory, with the network
blocked, .env loading disabled and the automatic recommendation logging stubbed.
"""
import base64
import hashlib
import html
import re
import shutil
import socket
import sys
import zlib
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import home_downloads

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = "nfl_games_historical_with_predictions.csv"
LINK = re.compile(r'<a download="([^"]+)" href="data:([^;]+);base64,([^"]+)"')


def predictions_fixture():
    today = date.today()
    rows = []
    for i, (home, away, day) in enumerate([("KC", "BUF", today + timedelta(days=1)),
                                           ("DAL", "PHI", today + timedelta(days=2)),
                                           ("NYJ", "MIA", today - timedelta(days=30))]):
        rows.append(dict(game_id=f"2026_07_{away}_{home}", season=2026, week=7, gameday=day.isoformat(),
                         home_team=home, away_team=away, spread_line=3.5 - i, total_line=44.5 + i,
                         home_moneyline=-160 + i, away_moneyline=135 + i, prob_underdogCovered=0.56 + i / 100,
                         prob_underdogWon=0.41, prob_overHit=0.52, home_score=float("nan"),
                         away_score=float("nan")))
    return pd.DataFrame(rows)


# The predictions table's schema (data_files/nfl_games_historical_with_predictions.csv),
# by kind, so the whole home page can render from synthetic rows.
SCHEMA = {
    "str": "game_id game_type gameday weekday gametime away_team home_team location nfl_detail_id pfr roof "
           "surface away_qb_id home_qb_id away_qb_name home_qb_name away_coach home_coach referee stadium_id stadium",
    "int": "season week away_score home_score result total old_game_id gsis pff espn ftn away_rest home_rest "
           "away_moneyline home_moneyline away_spread_odds home_spread_odds under_odds over_odds temp wind "
           "totalScore pointDiff homeTeamGamesPlayed awayTeamGamesPlayed restDaysDiff",
    "bin": "overtime div_game overUnderAccuracy homeWin awayWin isCloseGame isBlowout homeFavored awayFavored "
           "spreadCovered favoriteCovered spreadPush underdogCovered underdogWon overHit underHit totalHit "
           "isCloseSpread isMediumSpread isLargeSpread homeTeamWellRested awayTeamWellRested homeTeamShortRest "
           "awayTeamShortRest isColdWeather isWindy isExtremeWeather pred_underdogWon_optimal "
           "pred_spreadCovered_optimal pred_overHit_optimal pred_over moneyline_bet_return totals_bet_return "
           "predictedSpreadCovered predictedOverHit edge_underdog_ml edge_over",
    "float": "spread_line total_line gameLineAccuracy total_line_diff spreadSize homeTeamWinPct awayTeamWinPct "
             "homeTeamCloseGamePct awayTeamCloseGamePct homeTeamBlowoutPct awayTeamBlowoutPct homeTeamAvgScore "
             "awayTeamAvgScore homeTeamAvgScoreAllowed awayTeamAvgScoreAllowed homeTeamAvgPointDiff "
             "awayTeamAvgPointDiff homeTeamAvgTotalScore awayTeamAvgTotalScore homeTeamAvgPointSpread "
             "awayTeamAvgPointSpread homeTeamAvgTotal awayTeamAvgTotal homeTeamFavoredPct awayTeamFavoredPct "
             "homeTeamSpreadCoveredPct awayTeamSpreadCoveredPct homeTeamOverHitPct awayTeamOverHitPct "
             "homeTeamUnderHitPct awayTeamUnderHitPct homeTeamTotalHitPct awayTeamTotalHitPct homeTeamLast3WinPct "
             "awayTeamLast3WinPct homeTeamLast3AvgScore awayTeamLast3AvgScore homeTeamLast3AvgScoreAllowed "
             "awayTeamLast3AvgScoreAllowed homeTeamPointDiffTrend awayTeamPointDiffTrend prob_underdogCovered "
             "prob_overHit prob_underdogWon ev_spread ev_moneyline ev_totals spread_bet_return "
             "implied_prob_underdog_ml implied_prob_underdog_spread edge_underdog_spread implied_prob_over "
             "implied_prob_under edge_under",
}
TEAMS = "ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC PHI".split()


UPCOMING = [("KC", "BUF"), ("DAL", "PHI")]          # (home, away) on today + 1 and today + 2 days


def home_predictions_fixture(n_past=240, seed=11):
    """Synthetic past games plus the UPCOMING games, with every column."""
    import numpy as np
    rng = np.random.default_rng(seed)
    start = date.today() - timedelta(days=7 * n_past // 8 + 60)
    rows = []
    for i in range(n_past + len(UPCOMING)):
        upcoming = i >= n_past
        day = (date.today() + timedelta(days=i - n_past + 1)) if upcoming else start + timedelta(days=7 * (i // 8))
        h, a = rng.choice(len(TEAMS), 2, replace=False)
        if upcoming:
            h, a = (TEAMS.index(t) for t in UPCOMING[i - n_past])
        r = {c: str(rng.integers(1, 99)) for c in SCHEMA["str"].split()}
        r.update({c: int(rng.integers(0, 30)) for c in SCHEMA["int"].split()})
        r.update({c: int(rng.integers(0, 2)) for c in SCHEMA["bin"].split()})
        r.update({c: float(rng.uniform(0, 1)) for c in SCHEMA["float"].split()})
        hs, as_ = (0, 0) if upcoming else (int(rng.integers(3, 38)), int(rng.integers(3, 38)))
        r.update(game_id=f"{day.year}_{i:03d}_{TEAMS[a]}_{TEAMS[h]}", game_type="REG", gameday=day.isoformat(),
                 weekday=day.strftime("%A"), gametime="13:00", home_team=TEAMS[h], away_team=TEAMS[a],
                 location="Home", roof="outdoors", surface="grass", season=day.year if day.month > 2 else day.year - 1,
                 week=i % 18 + 1, home_score=hs, away_score=as_, result=hs - as_, total=hs + as_,
                 totalScore=hs + as_, pointDiff=hs - as_, spread_line=float(rng.choice([-6.5, -3, -1.5, 2.5, 3.5, 7])),
                 total_line=float(rng.choice([38.5, 42.5, 44, 47.5, 51])), home_moneyline=-150, away_moneyline=130,
                 home_spread_odds=-110, away_spread_odds=-110, over_odds=-110, under_odds=-110,
                 prob_underdogCovered=float(rng.uniform(0.40, 0.70)), prob_underdogWon=float(rng.uniform(0.2, 0.5)),
                 prob_overHit=float(rng.uniform(0.4, 0.6)))
        rows.append(r)
    return pd.DataFrame(rows)


def write_home_fixture(d):
    """Everything the full home page reads, synthetic."""
    home_predictions_fixture().to_csv(d / PREDICTIONS, sep="\t", index=False)
    # betting_log.LOG_COLUMNS: one win, one loss, one pending.
    (d / home_downloads.RECOMMENDATION_LOG).write_bytes(
        b"log_date,season,week,game_id,gameday,home_team,away_team,bet_type,recommended_team,spread_line,"
        b"total_line,moneyline_odds,model_probability,edge,confidence_tier,actual_home_score,actual_away_score,"
        b"bet_result,bet_profit,bet_spread,bet_odds,odds_source\r\n"
        b"2026-09-06 10:09:22,2026,1,2026_01_NE_SEA,2026-09-09,SEA,NE,spread,NE,3.5,44.5,,0.578,0.043,Good,"
        b"13.0,10.0,win,90.91,3.5,-110.0,fixture\r\n"
        b"2026-09-06 10:09:22,2026,1,2026_01_SF_LA,2026-09-10,LA,SF,spread,SF,3.5,48.5,,0.552,0.024,Good,"
        b"27.0,7.0,loss,-100.0,3.5,-110.0,fixture\r\n"
        b"2026-10-20 08:00:00,2026,7,2026_07_BUF_KC,2026-10-26,KC,BUF,spread,BUF,3.5,47.5,,0.601,0.081,Strong,"
        b",,,,3.5,-110.0,fixture\r\n")
    shutil.copyfile(ROOT / "data_files" / "csv_icon.png", d / "csv_icon.png")
    # load_schedule() reads nfl_schedule_<season>.csv (previous year in Jan-Feb).
    today = date.today()
    season = today.year - 1 if today.month <= 2 else today.year
    lines = ["week,date,home_team,away_team,venue,status"]
    for i, (home, away) in enumerate(UPCOMING):
        lines.append(f"7,{(today + timedelta(days=i + 1)).isoformat()},{home},{away},Fixture Stadium,REG")
    (d / f"nfl_schedule_{season}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data_files"
    d.mkdir()
    predictions_fixture().to_csv(d / PREDICTIONS, sep="\t", index=False)
    (d / home_downloads.RECOMMENDATION_LOG).write_bytes(
        b"log_date,season,week,game_id,bet_type,recommended_team\r\n"
        b"2026-10-20,2026,7,2026_07_BUF_KC,spread,BUF\r\n")
    shutil.copyfile(ROOT / "data_files" / "csv_icon.png", d / "csv_icon.png")
    return d


def tree(d):
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(d).rglob("*")) if p.is_file()}


def links(at):
    """{filename: (mime, bytes)} for every rendered download link in the sidebar."""
    out = {}
    for m in at.sidebar.markdown:
        for name, mime, b64 in LINK.findall(m.value):
            out[html.unescape(name)] = (mime, base64.b64decode(b64))
    return out


def pdf_text(body):
    """The decoded page content of a reportlab PDF (ASCII85 + Flate streams)."""
    out = b""
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", body, re.S):
        data = m.group(1).strip()
        try:
            out += zlib.decompress(base64.a85decode(data.rstrip(b"~>").replace(b"\n", b""), adobe=False))
        except Exception:
            out += data
    return out


def pdf_teams(body):
    return {t.decode() for t in re.findall(rb"\(([A-Z]{2,3})\) Tj", pdf_text(body))}


def texts(elements):
    return [e.value for e in elements]


def sidebar_app(data_dir, with_predictions=True):
    import pandas as pd
    import streamlit as st
    import home_downloads

    with st.sidebar:
        p1, p2, p3 = st.empty(), st.empty(), st.empty()
    df = (pd.read_csv(f"{data_dir}/nfl_games_historical_with_predictions.csv", sep="\t")
          if with_predictions else None)
    home_downloads.render_home_downloads(p1, p2, p3, df, data_dir)


def media_app(data_dir):
    """sidebar_app, then records the stored download files by URL (AppTest's mock
    runtime is cleared after each run, so they're captured during it)."""
    import pandas as pd
    import streamlit as st
    from streamlit.runtime import Runtime
    import home_downloads

    with st.sidebar:
        p1, p2, p3 = st.empty(), st.empty(), st.empty()
    df = pd.read_csv(f"{data_dir}/nfl_games_historical_with_predictions.csv", sep="\t")
    home_downloads.render_home_downloads(p1, p2, p3, df, data_dir)
    storage = Runtime.instance().media_file_mgr._storage
    st.session_state["media"] = {storage.get_url(fid): (f.filename, f.mimetype, f.content)
                                 for fid, f in storage._files_by_id.items()}


def run_sidebar(data_dir, **kw):
    at = AppTest.from_function(sidebar_app, args=(str(data_dir),), kwargs=kw, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def pdf_button(at):
    return [b for b in at.sidebar.button if b.label == "📄 Generate Predictions PDF"]


class TestRenderedControls:
    def test_all_three_controls_render_with_exact_contents(self, data_dir):
        before = tree(data_dir)
        at = run_sidebar(data_dir)
        got = links(at)
        stamp = date.today().strftime("%Y%m%d")
        assert set(got) == {f"nfl_predictions_{stamp}.csv", f"betting_recommendations_log_{stamp}.csv"}
        mime, preds = got[f"nfl_predictions_{stamp}.csv"]
        fixture = pd.read_csv(data_dir / PREDICTIONS, sep="\t")
        assert mime == "text/csv"
        assert preds == fixture.to_csv(index=False).encode("utf-8")
        pd.testing.assert_frame_equal(pd.read_csv(pd.io.common.BytesIO(preds)), fixture)
        mime, log = got[f"betting_recommendations_log_{stamp}.csv"]
        assert mime == "text/csv" and log == (data_dir / home_downloads.RECOMMENDATION_LOG).read_bytes()
        assert "Download Predictions" in " ".join(texts(at.sidebar.markdown))
        assert "Download Betting Log" in " ".join(texts(at.sidebar.markdown))
        assert len(pdf_button(at)) == 1
        assert tree(data_dir) == before

    @pytest.mark.skipif(bool(home_downloads.PDF_UNAVAILABLE), reason="reportlab not installed")
    def test_pdf_is_generated_on_demand(self, data_dir):
        before = tree(data_dir)
        at = run_sidebar(data_dir)
        pdf_button(at)[0].click().run()
        assert not at.exception
        pdfs = {n: v for n, v in links(at).items() if n.endswith(".pdf")}
        assert len(pdfs) == 1
        (name, (mime, body)), = pdfs.items()
        assert re.fullmatch(r"nfl_predictions_\d{8}_\d{6}\.pdf", name)
        assert mime == "application/pdf" and body.startswith(b"%PDF-") and body.rstrip().endswith(b"%%EOF")
        # Saved under exports/ exactly as offered; nothing else changed.
        assert (data_dir / "exports" / name).read_bytes() == body
        after = tree(data_dir)
        assert {k: v for k, v in after.items() if not k.startswith("exports/")} == before
        assert [k for k in after if k.startswith("exports/")] == [f"exports/{name}"]

    @pytest.mark.skipif(bool(home_downloads.PDF_UNAVAILABLE), reason="reportlab not installed")
    def test_pdf_holds_only_upcoming_games(self):
        body = home_downloads.generate_predictions_pdf(predictions_fixture())
        assert body.startswith(b"%PDF-")
        assert pdf_teams(body) == {"KC", "BUF", "DAL", "PHI"}           # the past NYJ/MIA game is left out
        empty = home_downloads.generate_predictions_pdf(predictions_fixture().iloc[[2]])
        assert pdf_teams(empty) == set() and b"No upcoming games found" in pdf_text(empty)


class TestNativeFallback:
    """If the HTML link can't be rendered, render_download falls back to st.download_button."""

    @pytest.fixture
    def no_html(self, monkeypatch):
        from streamlit.delta_generator import DeltaGenerator
        real = DeltaGenerator.markdown

        def markdown(self, body, *a, unsafe_allow_html=False, **k):
            if unsafe_allow_html:
                raise RuntimeError("HTML rendering disabled for this test")
            return real(self, body, *a, unsafe_allow_html=unsafe_allow_html, **k)
        monkeypatch.setattr(DeltaGenerator, "markdown", markdown)

    def buttons(self, at):
        media = at.session_state["media"]
        return {b.label: media[b.proto.url] for b in at.sidebar.download_button}

    def test_csv_downloads_fall_back_to_download_button(self, data_dir, no_html):
        before = tree(data_dir)
        at = AppTest.from_function(media_app, args=(str(data_dir),), default_timeout=60)
        at.run()
        assert not at.exception
        stamp = date.today().strftime("%Y%m%d")
        preds_name, log_name = f"nfl_predictions_{stamp}.csv", f"betting_recommendations_log_{stamp}.csv"
        assert not links(at)
        got = self.buttons(at)
        assert set(got) == {f"Download {preds_name}", f"Download {log_name}"}
        filename, mime, body = got[f"Download {preds_name}"]
        assert (filename, mime) == (preds_name, "text/csv")
        assert body == pd.read_csv(data_dir / PREDICTIONS, sep="\t").to_csv(index=False).encode("utf-8")
        filename, mime, body = got[f"Download {log_name}"]
        assert (filename, mime) == (log_name, "text/csv")
        assert body == (data_dir / home_downloads.RECOMMENDATION_LOG).read_bytes()
        assert tree(data_dir) == before

    @pytest.mark.skipif(bool(home_downloads.PDF_UNAVAILABLE), reason="reportlab not installed")
    def test_pdf_falls_back_to_download_button(self, data_dir, no_html):
        at = AppTest.from_function(media_app, args=(str(data_dir),), default_timeout=60)
        at.run()
        pdf_button(at)[0].click().run()
        assert not at.exception
        (label, (filename, mime, body)), = [(k, v) for k, v in self.buttons(at).items() if k.endswith(".pdf")]
        assert re.fullmatch(r"nfl_predictions_\d{8}_\d{6}\.pdf", filename) and label == f"Download {filename}"
        assert mime == "application/pdf" and body.startswith(b"%PDF-")
        assert (data_dir / "exports" / filename).read_bytes() == body
        assert pdf_teams(body) == {"KC", "BUF", "DAL", "PHI"}


class TestUnavailableAndFailures:
    def test_predictions_not_loaded(self, data_dir):
        at = run_sidebar(data_dir, with_predictions=False)
        info = texts(at.sidebar.info)
        assert "Predictions download unavailable: predictions haven't loaded yet." in info
        assert "PDF export unavailable: predictions haven't loaded yet." in info
        assert not pdf_button(at)
        assert any(n.startswith("betting_recommendations_log_") for n in links(at))

    def test_recommendation_log_missing(self, data_dir):
        (data_dir / home_downloads.RECOMMENDATION_LOG).unlink()
        at = run_sidebar(data_dir)
        assert any("Betting log download unavailable: no recommendation log "
                   "(betting_recommendations_log.csv) yet." == t for t in texts(at.sidebar.info))
        assert not any(n.startswith("betting_recommendations_log_") for n in links(at))
        assert any(n.startswith("nfl_predictions_") for n in links(at))

    def test_pdf_library_unavailable(self, data_dir, monkeypatch):
        monkeypatch.setattr(home_downloads, "PDF_UNAVAILABLE", "the PDF library (reportlab) could not be imported")
        at = run_sidebar(data_dir)
        assert "PDF export unavailable: the PDF library (reportlab) could not be imported." in texts(at.sidebar.info)
        assert not pdf_button(at)
        with pytest.raises(RuntimeError, match="PDF export unavailable"):
            home_downloads.generate_predictions_pdf(predictions_fixture())

    def test_csv_failure_is_shown(self, data_dir, monkeypatch):
        def boom(df):
            raise ValueError("cannot encode")
        monkeypatch.setattr(home_downloads, "convert_df_to_csv", boom)
        at = run_sidebar(data_dir)
        assert any("Could not prepare nfl_predictions_" in t and "cannot encode" in t for t in texts(at.sidebar.error))

    def test_log_read_failure_is_shown(self, data_dir, monkeypatch):
        real_open = open

        def guarded(file, mode="r", *a, **k):
            if str(file).endswith(home_downloads.RECOMMENDATION_LOG):
                raise PermissionError("locked")
            return real_open(file, mode, *a, **k)
        monkeypatch.setattr("builtins.open", guarded)
        at = run_sidebar(data_dir)
        assert any("Could not prepare betting_recommendations_log_" in t and "locked" in t
                   for t in texts(at.sidebar.error))

    def test_pdf_generation_failure_is_shown(self, data_dir, monkeypatch):
        def boom(df):
            raise RuntimeError("layout failed")
        monkeypatch.setattr(home_downloads, "generate_predictions_pdf", boom)
        at = run_sidebar(data_dir)
        pdf_button(at)[0].click().run()
        assert "PDF generation failed: layout failed" in texts(at.sidebar.error)
        assert not (data_dir / "exports").exists()

    def test_empty_pdf_is_reported(self, data_dir, monkeypatch):
        monkeypatch.setattr(home_downloads, "generate_predictions_pdf", lambda df: b"")
        at = run_sidebar(data_dir)
        pdf_button(at)[0].click().run()
        assert "Failed to generate PDF." in texts(at.sidebar.error)


class TestHomePageWiring:
    """The real home page: the bug was that home_page() never passed its own
    placeholders on, so no control rendered."""

    @pytest.fixture
    def home_dir(self, tmp_path):
        d = tmp_path / "home" / "data_files"
        d.mkdir(parents=True)
        write_home_fixture(d)
        return d

    @pytest.fixture
    def home(self, home_dir, monkeypatch):
        import betting_log
        import dotenv

        monkeypatch.chdir(home_dir.parent)
        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)   # never read the real .env
        monkeypatch.setattr(betting_log, "append_recommendations", lambda *a, **k: 0)
        monkeypatch.setattr(betting_log, "grade_pending", lambda *a, **k: 0)
        monkeypatch.setattr(sys, "excepthook", sys.excepthook)           # restored after the test

        def refuse(*a, **k):
            raise OSError("network blocked in this test")
        monkeypatch.setattr(socket.socket, "connect", refuse)
        monkeypatch.setattr(socket, "create_connection", refuse)
        return AppTest.from_file(str(ROOT / "predictions.py"), default_timeout=120)

    def test_a_failing_export_call_is_shown(self, home, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("placeholder gone")
        monkeypatch.setattr(home_downloads, "render_home_downloads", boom)
        home.run()
        assert not home.exception, [e.message for e in home.exception]
        assert "Export controls failed: placeholder gone" in texts(home.sidebar.error)

    def loaded(self, at):
        at.run()
        if f"nfl_predictions_{date.today():%Y%m%d}.csv" not in links(at):   # first pass may precede the background load
            at.run()
        return at

    def test_whole_page_renders_without_exceptions(self, home):
        at = self.loaded(home)
        assert not at.exception, [e.message for e in at.exception]
        assert not at.error, [e.value for e in at.error]
        assert not at.warning, [e.value for e in at.warning]
        assert "NFL Game Outcome Predictor" in [h.value for h in at.header]

    def test_home_page_renders_the_three_controls(self, home_dir, home):
        before = tree(home_dir)
        at = self.loaded(home)
        assert not at.exception, [e.message for e in at.exception]
        stamp = date.today().strftime("%Y%m%d")
        got = links(at)
        assert got[f"nfl_predictions_{stamp}.csv"] == (
            "text/csv", pd.read_csv(home_dir / PREDICTIONS, sep="\t").to_csv(index=False).encode("utf-8"))
        assert got[f"betting_recommendations_log_{stamp}.csv"] == (
            "text/csv", (home_dir / home_downloads.RECOMMENDATION_LOG).read_bytes())
        assert len(pdf_button(at)) == 1
        assert not at.sidebar.error
        assert tree(home_dir) == before

    @pytest.mark.skipif(bool(home_downloads.PDF_UNAVAILABLE), reason="reportlab not installed")
    def test_home_page_generates_the_pdf(self, home_dir, home):
        before = tree(home_dir)
        at = self.loaded(home)
        pdf_button(at)[0].click().run()
        assert not at.exception, [e.message for e in at.exception]
        (name, (mime, body)), = [(n, v) for n, v in links(at).items() if n.endswith(".pdf")]
        assert mime == "application/pdf" and pdf_teams(body) == {"KC", "BUF", "DAL", "PHI"}
        after = tree(home_dir)
        assert {k: v for k, v in after.items() if not k.startswith("exports/")} == before
        assert (home_dir / "exports" / name).read_bytes() == body
