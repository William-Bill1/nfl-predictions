import csv
import sqlite3

import pytest

from scripts import football_epa as epa
from scripts.football_repository import open_read_only


@pytest.fixture
def inputs(tmp_path):
    audit = tmp_path / "audit.sqlite3"
    with sqlite3.connect(audit) as conn:
        conn.execute("CREATE TABLE query_coverage_games(assessment_id,season,game_type,completion_status,scores_verified,game_id,gameday,home_team,away_team)")
        conn.executemany("INSERT INTO query_coverage_games VALUES(1,2026,'REG','final',1,?,?,?,?)",
                         [("g1", "2026-09-01", "KC", "BUF"), ("g2", "2026-09-08", "BUF", "KC")])
    source = tmp_path / "pbp.csv"
    fields = sorted(epa.REQUIRED)
    rows = []
    for gid in ("g1", "g2"):
        for pid, off, defense, drop, rush, val in [(1,"KC","BUF",1,0,2), (2,"KC","BUF",1,1,-1),
                                                   (3,"BUF","KC",0,1,-2), (4,"BUF","KC",1,0,1)]:
            rows.append(dict(game_id=gid,play_id=pid,play_type="pass" if drop else "run",posteam=off,defteam=defense,qb_dropback=drop,
                             rush_attempt=rush,epa=val,qb_kneel=0,qb_spike=0))
    def write():
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    write()
    return source, audit, tmp_path / "epa.sqlite3", rows, write


def build(inputs):
    source, audit, output, _, _ = inputs
    return epa.build(source, audit, output, assessment_id=1, season=2026)


def test_weighted_metrics_and_scramble(inputs):
    assert build(inputs)["status"] == "covered"
    result = epa.team_summary(inputs[2], "KC", "2026-09-08")
    assert result["games"] == 1
    off = result["metrics"][0]
    assert off["epa_per_play"] == .5
    assert off["success_rate"] == .5
    assert result["metrics"][1]["eligible_plays"] == 2  # scramble stays dropback
    assert result["metrics"][2]["epa_per_play"] is None
    assert result["metrics"][3]["epa_per_play"] == -.5  # defense allowed, not sign flipped
    assert epa.team_summary(inputs[2], "KC", "2026-09-09", 1)["game_ids"] == ["g2"]


def test_missing_epa_is_not_zero(inputs):
    inputs[3][0]["epa"] = ""
    inputs[4]()
    assert build(inputs)["eligible_plays_missing_epa"] == 1
    assert epa.team_summary(inputs[2], "KC", "2026-09-08")["metrics"][0]["measured_plays"] == 1


@pytest.mark.parametrize("change", ["duplicate", "team", "infinity", "flag"])
def test_refusal_before_output(inputs, change):
    rows = inputs[3]
    if change == "duplicate":
        rows.append(rows[0].copy())
    elif change == "team":
        rows[0]["defteam"] = "NYJ"
    elif change == "infinity":
        rows[0]["epa"] = "inf"
    else:
        rows[0]["qb_dropback"] = ""
    inputs[4]()
    with pytest.raises(ValueError):
        build(inputs)
    assert not inputs[2].exists()


def test_existing_snapshot_not_replaced(inputs):
    build(inputs)
    before = inputs[2].read_bytes()
    with pytest.raises(FileExistsError):
        build(inputs)
    assert inputs[2].read_bytes() == before


def test_missing_game_and_side(inputs):
    inputs[3][:] = [r for r in inputs[3] if r["game_id"] == "g1" and r["posteam"] == "KC"]
    inputs[4]()
    report = build(inputs)
    assert report["games_without_eligible_plays"] == ["g2"]
    assert report["games_without_both_offenses"] == ["g1"]


@pytest.mark.parametrize("statement", ["DELETE FROM plays", "ATTACH DATABASE 'bad.db' AS bad", "PRAGMA user_version=3"])
def test_read_only_sql(inputs, statement):
    build(inputs)
    before = inputs[2].read_bytes()
    with open_read_only(inputs[2], require_current_schema=False) as conn:
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(statement)
        assert conn.execute("SELECT COUNT(*) FROM team_game_epa").fetchone()[0] == 12
    assert inputs[2].read_bytes() == before


def test_kneel_spike_excluded(inputs):
    inputs[3][0]["qb_kneel"] = 1
    inputs[3][1]["qb_spike"] = 1
    inputs[4]()
    build(inputs)
    assert epa.team_summary(inputs[2], "KC", "2026-09-08")["metrics"][0]["eligible_plays"] == 0


def test_partial_game_withholds_rates_and_does_not_replace_recent_game(inputs):
    inputs[3][:] = [r for r in inputs[3] if r["game_id"] != "g2"]
    inputs[4]()
    build(inputs)
    recent = epa.team_summary(inputs[2], "KC", "2026-09-09", 1)
    assert recent["game_ids"] == ["g2"]
    assert recent["coverage"]["status"] == "incomplete"
    assert recent["net_epa_per_play"] is None
    assert all(m["epa_per_play"] is None for m in recent["metrics"])
    earlier = epa.team_summary(inputs[2], "KC", "2026-09-08")
    assert earlier["coverage"]["status"] == "covered"
    assert earlier["net_epa_per_play"] == 1


def test_missing_epa_withholds_rates(inputs):
    inputs[3][0]["epa"] = ""
    inputs[4]()
    build(inputs)
    result = epa.team_summary(inputs[2], "KC", "2026-09-08")
    assert result["coverage"]["status"] == "incomplete"
    assert result["metrics"][0]["measured_plays"] == 1
    assert result["metrics"][0]["epa_per_play"] is None


def test_unknown_team_and_empty_cutoff(inputs):
    build(inputs)
    with pytest.raises(ValueError, match="Unknown team"):
        epa.team_summary(inputs[2], "TYPO", "2026-09-09")
    result = epa.team_summary(inputs[2], "KC", "2026-09-01")
    assert result["coverage"]["status"] == "no_data"
    assert result["net_epa_per_play"] is None
    with pytest.raises(ValueError):
        epa.team_summary(inputs[2], "KC", "20260909")
