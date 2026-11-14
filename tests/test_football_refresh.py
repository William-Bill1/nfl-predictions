"""Refresh uses only synthetic sources and temporary research directories."""
import csv
from datetime import date
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import pytest
from scripts import football_refresh as refresh
from tests.test_football_repository import game, COLUMNS


def source(path, rows, delimiter=","):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def setup(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    monkeypatch.setattr(refresh, "REPORTS", reports)
    old = source(tmp_path / "old.tsv", [game("2026_01_BBB_AAA", "AAA", "BBB", None, None, season=2026)], "\t")
    new = source(tmp_path / "new.csv", [game("2026_01_BBB_AAA", "AAA", "BBB", 20, 7, season=2026, surface=""),
                                        game("2026_02_CCC_AAA", "AAA", "CCC", None, None, season=2026, gameday="2026-11-20")])
    args = dict(source=new, previous_source=old, output_root=reports/"refreshes", season=2026, evaluation_date=date(2026,11,14))
    return args


def test_refresh_deltas_completion_missing_fields_and_preservation(setup):
    old_hash = hashlib.sha256(setup["previous_source"].read_bytes()).hexdigest()
    run, report = refresh.refresh(**setup)
    assert report["status"] == "captured_not_audited"
    assert report["newly_completed"] == ["2026_01_BBB_AAA"]
    assert report["added_games"] == ["2026_02_CCC_AAA"]
    assert report["removed_games"] == []
    assert report["completed_count"] == 1
    assert report["completed_missing_conditions"] == [{"game_id":"2026_01_BBB_AAA", "fields":["surface"]}]
    assert report["changed_games"][0]["fields"]["home_score"] == {"before":"", "after":"20.0"}
    assert report["pending_or_unverifiable"][0]["game_id"] == "2026_02_CCC_AAA"
    assert (run/"source.raw").read_bytes() == setup["source"].read_bytes()
    assert json.loads((run/"report.json").read_text()) == report
    assert hashlib.sha256(setup["previous_source"].read_bytes()).hexdigest() == old_hash
    with sqlite3.connect(run/"repository.sqlite3") as conn:
        assert conn.execute("SELECT COUNT(*) FROM source_imports").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='query_coverage_games'").fetchone()[0] == 0


def test_repeated_refresh_creates_new_attempt_without_changing_old(setup):
    first, _ = refresh.refresh(**setup)
    before={p.name: p.read_bytes() for p in first.iterdir()}
    second, _ = refresh.refresh(**setup)
    assert first != second
    assert before == {p.name: p.read_bytes() for p in first.iterdir()}


def test_unchanged_source_has_no_deltas_but_new_assessment(setup):
    setup["source"]=setup["previous_source"]
    _, report=refresh.refresh(**setup)
    assert not report["added_games"] and not report["changed_games"] and not report["newly_completed"]
    assert report["import_id"] == 1


def test_removed_and_lost_completion_are_explicit(setup):
    setup["previous_source"], setup["source"] = setup["source"], setup["previous_source"]
    _, report=refresh.refresh(**setup)
    assert report["removed_games"] == ["2026_02_CCC_AAA"]
    assert report["no_longer_completed"] == ["2026_01_BBB_AAA"]


@pytest.mark.parametrize("kind", ["duplicate", "missing_column", "wrong_season", "bad_row"])
def test_invalid_source_refused_without_attempt_directory(setup, kind):
    if kind == "duplicate":
        row=game("same", "AAA", "BBB", 20, 7, season=2026)
        source(setup["source"], [row,row])
    elif kind == "wrong_season":
        source(setup["source"], [game("other", "AAA", "BBB", 20,7,season=2025)])
    elif kind == "missing_column":
        setup["source"].write_text("game_id,season\nX,2026\n")
    else:
        with setup["source"].open("a") as handle: handle.write("too,few,fields\n")
    with pytest.raises(ValueError): refresh.refresh(**setup)
    assert not setup["output_root"].exists()


def test_output_outside_research_root_refused_before_download(setup, monkeypatch, tmp_path):
    setup.update(source=None, download=True, output_root=tmp_path/"production")
    monkeypatch.setattr(refresh.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Must not download"))
    with pytest.raises(ValueError, match="Output must"): refresh.refresh(**setup)


def test_download_uses_fixed_public_schedule_and_keeps_exact_bytes(setup, monkeypatch):
    data=setup["source"].read_bytes()
    seen=[]
    def opened(url, timeout):
        seen.append((url,timeout))
        return io.BytesIO(data)
    monkeypatch.setattr(refresh.urllib.request, "urlopen", opened)
    setup.update(source=None, download=True)
    run, report=refresh.refresh(**setup)
    assert seen == [(refresh.SOURCE_URL,45)]
    assert (run/"source.raw").read_bytes()==data
    assert report["source_sha256"]==hashlib.sha256(data).hexdigest()


def test_oversized_download_refused_without_output(setup, monkeypatch):
    monkeypatch.setattr(refresh.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"x"*101))
    monkeypatch.setattr(refresh, "MAX_BYTES", 100)
    setup.update(source=None, download=True)
    with pytest.raises(ValueError,match="size limit"): refresh.refresh(**setup)
    assert not setup["output_root"].exists()


def test_cli_failure_returns_actual_nonzero(setup, capsys):
    assert refresh.main(["--source",str(setup["source"]),"--previous-source",str(setup["previous_source"]),"--output-root",str(setup["output_root"]),"--season","2024","--date","2026-11-14"])==2
    assert json.loads(capsys.readouterr().out)["status"]=="failed"
