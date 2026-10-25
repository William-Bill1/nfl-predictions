"""Historical Data exports: "Export current page" and "Export all filtered rows".

The real page (pages/1_Historical_Data.py) runs under AppTest against a
temporary synthetic play-by-play file spanning several pages. The full data is
loaded with the page's own button. Each download is captured as Streamlit
registers it (the current page as bytes; all filtered rows as the deferred
callable Streamlit runs on click) and its CSV is checked against the rows the
page displays and an independent filter of the fixture.
"""
import gzip
import io
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import historical_exports

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "pages" / "1_Historical_Data.py"
PBP = "nfl_play_by_play_historical.csv.gz"
STAMP = date.today().strftime("%Y%m%d")
N_2025, N_2024 = 100, 130                      # 230 played plays: 5 pages of 50


def plays():
    """One play per day, newest first: 2025 plays, then 2024 plays, plus one
    future-dated play that the page's own date filter drops. DAL only plays
    offense in 2025, so 2024 + DAL matches nothing."""
    today = date.today()
    rows = []
    for i in range(N_2025 + N_2024 + 1):
        future = i == N_2025 + N_2024
        season = 2025 if (i < N_2025 or future) else 2024
        offense = ["KC", "BUF", "DAL"][i % 3] if season == 2025 else ["KC", "BUF"][i % 2]
        rows.append(dict(
            game_date=(today + timedelta(days=5) if future else today - timedelta(days=10 + i)).isoformat(),
            week=i % 18 + 1, season=season, home_team=offense, away_team="NYJ", posteam=offense, defteam="NYJ",
            game_seconds_remaining=3600 - i, qtr=i % 4 + 1, down=i % 4 + 1, ydstogo=i % 10 + 1,
            yardline_100=i % 90 + 5, play_type="pass" if i % 2 else "run", yards_gained=i % 12,
            desc=f"Play {i:04d}", epa=round((i % 7) / 7 - 0.4, 3), wp=0.5, posteam_score=i % 30,
            defteam_score=(i * 3) % 30, score_differential=(i % 30) - ((i * 3) % 30), pass_attempt=i % 2,
            rush_attempt=1 - i % 2, complete_pass=0, interception=0, fumble_lost=0, td_prob=0.1,
            touchdown=0, field_goal_result="", drive=i // 8))     # drive: loaded, never displayed
    # Shuffled in the file, so only the page's own sort gives display order.
    return pd.DataFrame(rows).sample(frac=1, random_state=3).reset_index(drop=True)


def expected_descs(season=None, offense=None):
    """The page's order (newest first) for played plays matching the filters."""
    d = plays()
    d = d[pd.to_datetime(d["game_date"]) <= pd.Timestamp(date.today())]
    if season is not None:
        d = d[d["season"].isin(season)]
    if offense is not None:
        d = d[d["posteam"].isin(offense)]
    return d.sort_values("game_date", ascending=False)["desc"].tolist()


@pytest.fixture
def capture(monkeypatch):
    """Every download Streamlit registers: {url: (name, mime, bytes)} and
    {deferred id: (name, mime, callable)}."""
    from streamlit.runtime.media_file_manager import MediaFileManager
    eager, deferred = {}, {}
    real_add, real_deferred = MediaFileManager.add, MediaFileManager.add_deferred

    def add(self, data, mimetype, coordinates, file_name=None, is_for_static_download=False):
        url = real_add(self, data, mimetype, coordinates, file_name=file_name,
                       is_for_static_download=is_for_static_download)
        eager[url] = (file_name, mimetype, data)
        return url

    def add_deferred(self, fn, mimetype, coordinates, file_name=None):
        fid = real_deferred(self, fn, mimetype, coordinates, file_name=file_name)
        deferred[fid] = (file_name, mimetype, fn)
        return fid
    monkeypatch.setattr(MediaFileManager, "add", add)
    monkeypatch.setattr(MediaFileManager, "add_deferred", add_deferred)
    return eager, deferred


@pytest.fixture
def sample_page(tmp_path, monkeypatch, capture):
    """The page as first opened: only its sample is loaded (the most recent
    season's plays, here 2025's 100 played plays)."""
    d = tmp_path / "data_files"
    d.mkdir()
    with gzip.open(d / PBP, "wt", encoding="utf-8", newline="") as fh:
        plays().to_csv(fh, sep="\t", index=False)
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(str(PAGE), default_timeout=60)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert "historical_data" not in at.session_state
    at.capture = capture
    at.source = d / PBP
    return at


def load_full(at):
    next(b for b in at.button if b.label == "Load full historical play-by-play data").click().run()
    assert not at.exception, [e.message for e in at.exception]
    assert len(at.session_state["historical_data"]) == N_2025 + N_2024 + 1
    return at


@pytest.fixture
def page(sample_page):
    """The page after "Load full historical play-by-play data"."""
    return load_full(sample_page)


def downloads(at):
    """{"page": (label, name, mime, bytes), "all": (label, name, mime, callable)} for the
    buttons in the latest run."""
    eager, deferred = at.capture
    out = {}
    for b in at.get("download_button"):
        if b.proto.deferred_file_id:
            name, mime, fn = deferred[b.proto.deferred_file_id]
            out["all"] = (b.label, name, mime, fn)
        else:
            name, mime, data = eager[b.proto.url]
            out["page"] = (b.label, name, mime, data)
    return out


def descs(csv_bytes):
    return pd.read_csv(io.BytesIO(csv_bytes))["desc"].tolist()


def shown(at):
    """The play-by-play rows the page displays."""
    return next(df.value for df in at.dataframe if "desc" in df.value.columns)


def set_page(at, n):
    next(w for w in at.number_input if w.label == "Page").set_value(n).run()
    assert not at.exception, [e.message for e in at.exception]


def ok(at):
    assert not at.exception, [e.message for e in at.exception]
    return at


class TestCurrentPage:
    def test_first_page_is_exactly_the_displayed_rows(self, page):
        label, name, mime, data = downloads(page)["page"]
        assert label == "📥 Export current page (50 rows)"
        assert name == f"nfl_historical_data_{STAMP}_page1_of_5.csv" and mime == "text/csv"
        assert descs(data) == shown(page)["desc"].tolist() == expected_descs()[:50]
        csv = pd.read_csv(io.BytesIO(data))
        assert list(csv.columns) == list(shown(page).columns)          # same columns, no index column
        assert data.decode("utf-8").splitlines()[0].startswith("game_date,week,season,")

    def test_a_later_page(self, page):
        set_page(page, 3)
        label, name, _, data = downloads(page)["page"]
        assert label == "📥 Export current page (50 rows)"
        assert name == f"nfl_historical_data_{STAMP}_page3_of_5.csv"
        assert descs(data) == shown(page)["desc"].tolist() == expected_descs()[100:150]

    def test_the_last_partial_page(self, page):
        set_page(page, 5)
        label, name, _, data = downloads(page)["page"]
        assert label == "📥 Export current page (30 rows)" and name.endswith("_page5_of_5.csv")
        assert descs(data) == expected_descs()[200:]

    def test_rows_per_page(self, page):
        next(w for w in page.selectbox if w.label == "Rows per page").set_value(100).run()
        ok(page)
        label, name, _, data = downloads(page)["page"]
        assert label == "📥 Export current page (100 rows)" and name.endswith("_page1_of_3.csv")
        assert descs(data) == expected_descs()[:100]


class TestAllFilteredRows:
    def test_every_filtered_row_across_pages(self, page):
        label, name, mime, fn = downloads(page)["all"]
        assert label == "📥 Export all filtered rows (230 rows)"
        assert name == f"nfl_historical_data_{STAMP}_all_filtered_230_rows.csv" and mime == "text/csv"
        data = fn()
        assert descs(data) == expected_descs()                    # every page, in display order
        assert list(pd.read_csv(io.BytesIO(data)).columns) == list(shown(page).columns)   # display columns only
        assert "Play 0230" not in descs(data)                       # the future play, dropped by the page
        page_data = downloads(page)["page"][3]
        # Same conversion: the first page of the full file is the page file, byte for byte.
        assert data.splitlines()[:51] == page_data.splitlines()

    def test_click_does_not_rerun(self, page):
        # In a real browser a rerun on click re-registers the deferred download
        # under a new id and Streamlit drops the one being fetched, so the
        # download silently fails; AppTest can't show that, so pin the setting.
        buttons = {b.label: b.proto for b in page.get("download_button")}
        assert buttons["📥 Export all filtered rows (230 rows)"].ignore_rerun is True
        assert buttons["📥 Export current page (50 rows)"].ignore_rerun is False

    def test_unchanged_by_the_page_shown(self, page):
        before = downloads(page)["all"][3]()
        set_page(page, 4)
        label, name, _, fn = downloads(page)["all"]
        assert label == "📥 Export all filtered rows (230 rows)" and name.endswith("_all_filtered_230_rows.csv")
        assert fn() == before

    def test_follows_changed_filters(self, page):
        page.multiselect(key="season").set_value([2024]).run()
        ok(page)
        label, name, _, fn = downloads(page)["all"]
        assert label == "📥 Export all filtered rows (130 rows)" and name.endswith("_all_filtered_130_rows.csv")
        rows = descs(fn())
        assert rows == expected_descs(season=[2024]) and len(rows) == 130
        assert not set(rows) & set(expected_descs(season=[2025]))   # non-matching rows excluded
        assert downloads(page)["page"][1].endswith("_page1_of_3.csv")

        page.multiselect(key="posteam").set_value(["KC"]).run()
        ok(page)
        label, _, _, fn = downloads(page)["all"]
        assert label == "📥 Export all filtered rows (65 rows)"
        assert descs(fn()) == expected_descs(season=[2024], offense=["KC"])

    def test_source_data_change(self, page):
        page.multiselect(key="season").set_value([2024]).run()
        ok(page)
        trimmed = page.session_state["historical_data"].iloc[::2]
        page.session_state["historical_data"] = trimmed
        page.run()
        ok(page)
        label, _, _, fn = downloads(page)["all"]
        want = [d for d in expected_descs(season=[2024]) if d in set(trimmed["desc"])]
        assert label == f"📥 Export all filtered rows ({len(want)} rows)"
        assert descs(fn()) == want

    def test_built_only_when_clicked(self, page, monkeypatch):
        sizes = []
        real = pd.DataFrame.to_csv

        def to_csv(self, *a, **k):
            sizes.append(len(self))
            return real(self, *a, **k)
        monkeypatch.setattr(pd.DataFrame, "to_csv", to_csv)
        set_page(page, 2)
        page.multiselect(key="season").set_value([2024]).run()
        ok(page)
        assert sizes and max(sizes) <= 50                            # reruns only build page-sized files
        downloads(page)["all"][3]()
        assert sizes[-1] == 130


FULL_NOTE = "Every row matching the current filters, across all {pages}. Built when clicked."
SAMPLE_NOTE = ("Only the loaded sample is included: the {rows} sample rows matching the current filters, "
               "across all {pages}. Use “Load full historical play-by-play data” to export the "
               "full filtered dataset.")


def captions(at):
    return [c.value for c in at.caption]


class TestSampleScope:
    """Before the full dataset is loaded the page works from a sample; the
    whole-view export says so in its label, caption and file name."""

    def test_initial_sample(self, sample_page):
        at = sample_page
        got = downloads(at)
        label, name, mime, fn = got["all"]
        assert label == "📥 Export filtered sample (100 rows)"
        assert name == f"nfl_historical_data_{STAMP}_sample_filtered_100_rows.csv" and mime == "text/csv"
        assert SAMPLE_NOTE.format(rows=100, pages="2 pages") in captions(at)
        assert not any("Every row matching" in c for c in captions(at))
        data = fn()
        assert descs(data) == expected_descs(season=[2025])                 # exactly the sample's rows
        assert list(pd.read_csv(io.BytesIO(data)).columns) == list(shown(at).columns)
        assert data.splitlines()[:51] == got["page"][3].splitlines()        # page 1 of the same file
        assert got["page"][0] == "📥 Export current page (50 rows)"
        assert descs(got["page"][3]) == shown(at)["desc"].tolist() == expected_descs(season=[2025])[:50]
        buttons = {b.label: b.proto for b in at.get("download_button")}
        assert buttons[label].ignore_rerun is True and buttons[label].deferred_file_id   # still deferred

    def test_filtered_sample(self, sample_page):
        at = sample_page
        at.multiselect(key="posteam").set_value(["DAL"]).run()
        ok(at)
        want = expected_descs(season=[2025], offense=["DAL"])
        label, name, _, fn = downloads(at)["all"]
        assert label == f"📥 Export filtered sample ({len(want)} rows)"
        assert name == f"nfl_historical_data_{STAMP}_sample_filtered_{len(want)}_rows.csv"
        assert SAMPLE_NOTE.format(rows=len(want), pages="1 page") in captions(at)
        assert descs(fn()) == want

    def test_transition_to_full_data(self, sample_page):
        at = sample_page
        at.multiselect(key="posteam").set_value(["KC"]).run()
        ok(at)
        sample_kc = expected_descs(season=[2025], offense=["KC"])
        assert downloads(at)["all"][0] == f"📥 Export filtered sample ({len(sample_kc)} rows)"
        # Loading reruns before the filters render, so they start cleared on the
        # full data (existing page behaviour): the export is then every row.
        load_full(at)
        label, name, _, fn = downloads(at)["all"]
        assert label == "📥 Export all filtered rows (230 rows)"
        assert name == f"nfl_historical_data_{STAMP}_all_filtered_230_rows.csv"
        assert FULL_NOTE.format(pages="5 pages") in captions(at)
        assert not any("Only the loaded sample" in c for c in captions(at))
        assert descs(fn()) == expected_descs()
        buttons = {b.label: b.proto for b in at.get("download_button")}
        assert buttons[label].ignore_rerun is True and buttons[label].deferred_file_id
        # Filtering the full data: rows the sample never had are included.
        at.multiselect(key="posteam").set_value(["KC"]).run()
        ok(at)
        want = expected_descs(offense=["KC"])
        assert len(want) > len(sample_kc)
        label, name, _, fn = downloads(at)["all"]
        assert label == f"📥 Export all filtered rows ({len(want)} rows)"
        assert name == f"nfl_historical_data_{STAMP}_all_filtered_{len(want)}_rows.csv"
        assert FULL_NOTE.format(pages="2 pages") in captions(at)
        assert descs(fn()) == want

    def test_full_data_caption(self, page):
        assert FULL_NOTE.format(pages="5 pages") in captions(page)

    def test_scope_is_explicit(self):
        with pytest.raises(ValueError, match="unknown data scope"):
            historical_exports.render_exports(pd.DataFrame({"a": [1]}), ["a"], start_idx=0, end_idx=1, page=1,
                                              total_pages=1, convert=lambda df: b"", scope="partial")
        with pytest.raises(TypeError):
            historical_exports.render_exports(pd.DataFrame({"a": [1]}), ["a"], start_idx=0, end_idx=1, page=1,
                                              total_pages=1, convert=lambda df: b"")


class TestEmpty:
    def test_no_matching_rows(self, page):
        page.multiselect(key="season").set_value([2024]).run()
        page.multiselect(key="posteam").set_value(["DAL"]).run()
        ok(page)
        assert downloads(page) == {}
        captions = [c.value for c in page.caption]
        assert "Nothing to export: no rows match the current filters." in captions
        assert "No rows match the current filters. Adjust filters or load the full dataset." in \
            [i.value for i in page.info]

    def test_back_to_matching_rows(self, page):
        page.multiselect(key="season").set_value([2024]).run()
        page.multiselect(key="posteam").set_value(["DAL"]).run()
        page.multiselect(key="posteam").set_value([]).run()
        ok(page)
        assert downloads(page)["all"][0] == "📥 Export all filtered rows (130 rows)"


def test_an_export_failure_is_shown(page, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("cannot build")
    monkeypatch.setattr(historical_exports, "render_exports", boom)
    page.run()
    ok(page)
    assert "Export unavailable: cannot build" in [e.value for e in page.error]
    assert shown(page)["desc"].tolist() == expected_descs()[:50]      # the table still renders


class TestSourceUnchanged:
    def test_exports_do_not_touch_the_source(self, page):
        before = page.source.read_bytes()
        downloads(page)["all"][3]()
        set_page(page, 2)
        downloads(page)["all"][3]()
        assert page.source.read_bytes() == before
        assert sorted(p.name for p in page.source.parent.iterdir()) == [PBP]


def test_filenames_are_distinct():
    assert historical_exports.page_filename("20261025", 2, 5) == "nfl_historical_data_20261025_page2_of_5.csv"
    assert historical_exports.all_filtered_filename("20261025", 230) == \
        "nfl_historical_data_20261025_all_filtered_230_rows.csv"
    assert historical_exports.sample_filtered_filename("20261025", 100) == \
        "nfl_historical_data_20261025_sample_filtered_100_rows.csv"
