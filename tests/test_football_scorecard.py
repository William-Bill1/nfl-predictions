from scripts import football_epa as epa
from scripts import football_scorecard as scorecard
from tests.test_football_epa import inputs  # shared synthetic source/audited database fixture

import pytest


def test_readable_matchup_matches_existing_metrics(inputs):
    source, audit, output, _, _ = inputs
    epa.build(source, audit, output, assessment_id=1, season=2026)
    before = output.read_bytes()
    report = scorecard.matchup(output, ["KC", "BUF"], "2026-09-08")
    assert report["comparisons"]["KC"]["season"] == epa.team_summary(output, "KC", "2026-09-08")
    text = scorecard.markdown(report)
    assert "| Net EPA/play | +1.000 | -1.000 |" in text
    assert "| Games in selection | 1 | 1 |" in text
    assert "50.0%" in text
    assert "pbp_sha256" in text and "two-point attempts are included" in text
    assert output.read_bytes() == before


def test_incomplete_report_withholds_values(inputs):
    source, audit, output, rows, write = inputs
    rows[0]["epa"] = ""
    write()
    epa.build(source, audit, output, assessment_id=1, season=2026)
    text = scorecard.markdown(scorecard.matchup(output, ["KC", "BUF"], "2026-09-08"))
    assert "| Net EPA/play | Unavailable | Unavailable |" in text
    assert "rates withheld" in text
    assert "eligible plays missing EPA: 1" in text


def test_same_team_refused(inputs):
    with pytest.raises(ValueError, match="different teams"):
        scorecard.matchup(inputs[2], ["KC", "KC"], "2026-09-08")


def test_verified_scores_and_measurement_edges(inputs):
    import sqlite3
    source, audit, output, _, _ = inputs
    with sqlite3.connect(audit) as conn:
        conn.execute("ALTER TABLE query_coverage_games ADD COLUMN home_score INTEGER")
        conn.execute("ALTER TABLE query_coverage_games ADD COLUMN away_score INTEGER")
        conn.execute("UPDATE query_coverage_games SET home_score=21,away_score=14")
    epa.build(source, audit, output, assessment_id=1, season=2026)
    report = scorecard.matchup(output, ["KC", "BUF"], "2026-09-08", audit)
    assert report["score_results"]["KC:season"]["point_differential_per_game"] == 7
    assert report["score_results"]["BUF:season"]["point_differential_per_game"] == -7
    text = scorecard.markdown(report)
    assert "| Point differential/game | +7.000 | -7.000 | KC |" in text
    assert "No weights or combined total" in text
    assert "| Turnover differential (season total) | Unavailable | Unavailable | Not assessed |" in text
    assert "Passing matchup: KC offense" in text
    with sqlite3.connect(audit) as conn:
        conn.execute("UPDATE query_coverage_games SET home_score=99")
    with pytest.raises(ValueError, match="pinned audit hash"):
        scorecard.matchup(output, ["KC", "BUF"], "2026-09-08", audit)


@pytest.mark.parametrize("location,expected", [("Home","away"),("Neutral","neutral"),("", "unknown"),("unexpected","unknown")])
def test_site_records_streak_and_matchup_highlight(inputs,location,expected):
    import sqlite3
    source,audit,output,_,_=inputs
    with sqlite3.connect(audit) as conn:
        conn.execute("ALTER TABLE query_coverage_games ADD COLUMN home_score INTEGER")
        conn.execute("ALTER TABLE query_coverage_games ADD COLUMN away_score INTEGER")
        conn.execute("UPDATE query_coverage_games SET home_score=14,away_score=21 WHERE game_id='g1'")
        conn.execute("UPDATE query_coverage_games SET home_score=21,away_score=14 WHERE game_id='g2'")
        conn.execute("CREATE TABLE completion_assessments(assessment_id,import_id)")
        conn.execute("INSERT INTO completion_assessments VALUES(1,5)")
        conn.execute("CREATE TABLE game_observations(import_id,game_id,location)")
        conn.executemany("INSERT INTO game_observations VALUES(5,?,?)",[("g1","Home"),("g2",location),("future",location)])
        conn.execute("INSERT INTO query_coverage_games VALUES(1,2026,'REG','pending',0,'future','2026-09-15','BUF','KC',NULL,NULL)")
    epa.build(source,audit,output,assessment_id=1,season=2026)
    before=output.read_bytes(),audit.read_bytes()
    report=scorecard.matchup(output,["KC","BUF"],"2026-09-09",audit,"future")
    kc=report['score_results']['KC:season']
    assert (kc['wins'],kc['losses'],kc['ties'])==(0,2,0)
    assert kc['site_records']['home']['losses']==1
    assert kc['site_records'][expected]['losses']==1
    assert sum(v['games'] for v in kc['site_records'].values())==2
    assert kc['streak']=={'outcome':'L','games':2}
    assert kc['through']=='2026-09-08'
    assert kc['latest_result']['opponent']=='BUF'
    assert report['score_results']['BUF:season']['streak']=={'outcome':'W','games':2}
    text=scorecard.markdown(report)
    assert f'Selected matchup: KC {expected} record:' in text
    assert 'Lost 2' in text and 'Won 2' in text
    cutoff=scorecard.matchup(output,["KC","BUF"],"2026-09-08",audit)
    assert cutoff['score_results']['KC:season']['games']==1
    assert cutoff['score_results']['KC:season']['through']=='2026-09-01'
    assert before==(output.read_bytes(),audit.read_bytes())


def test_tie_ends_streak_and_missing_locations_stay_unknown(inputs):
    import sqlite3
    source,audit,output,_,_=inputs
    with sqlite3.connect(audit) as conn:
        conn.execute("ALTER TABLE query_coverage_games ADD COLUMN home_score INTEGER")
        conn.execute("ALTER TABLE query_coverage_games ADD COLUMN away_score INTEGER")
        conn.execute("UPDATE query_coverage_games SET home_score=21,away_score=14")
        conn.execute("UPDATE query_coverage_games SET home_score=14,away_score=14 WHERE game_id='g2'")
    epa.build(source,audit,output,assessment_id=1,season=2026)
    report=scorecard.matchup(output,["KC","BUF"],"2026-09-09",audit)
    kc=report['score_results']['KC:season']
    assert (kc['wins'],kc['losses'],kc['ties'])==(1,0,1)
    assert kc['site_records']['unknown']['games']==2
    assert kc['site_records']['home']['games']==0
    assert kc['streak']=={'outcome':'T','games':1}
    assert 'Tied last game' in scorecard.markdown(report)


def test_unverified_game_is_not_counted_and_report_says_snapshot_only(inputs):
    import sqlite3
    source,audit,output,_,_=inputs
    with sqlite3.connect(audit) as conn:
        conn.execute("ALTER TABLE query_coverage_games ADD COLUMN home_score INTEGER")
        conn.execute("ALTER TABLE query_coverage_games ADD COLUMN away_score INTEGER")
        conn.execute("UPDATE query_coverage_games SET home_score=21,away_score=14")
        conn.execute("UPDATE query_coverage_games SET scores_verified=0 WHERE game_id='g2'")
    conn.close()
    epa.build(source,audit,output,assessment_id=1,season=2026)
    report=scorecard.matchup(output,["KC","BUF"],"2026-09-30",audit)
    assert report['score_results']['KC:season']['games']==1
    assert 'retained audited snapshot' in report['score_results']['KC:season']['record_basis']
    text=scorecard.markdown(report)
    assert "regular-season records in the retained audited snapshot" in text
    assert "a game missing from it is not counted" in text


@pytest.mark.parametrize("day", [None, "2026-9-15"])
def test_selected_game_without_iso_date_is_refused(inputs, day):
    import sqlite3
    source,audit,output,_,_=inputs
    with sqlite3.connect(audit) as conn:
        conn.execute("ALTER TABLE query_coverage_games ADD COLUMN home_score INTEGER")
        conn.execute("ALTER TABLE query_coverage_games ADD COLUMN away_score INTEGER")
        conn.execute("UPDATE query_coverage_games SET home_score=21,away_score=14")
        conn.execute("INSERT INTO query_coverage_games VALUES(1,2026,'REG','pending',0,'future',?,'BUF','KC',NULL,NULL)",(day,))
    conn.close()
    epa.build(source,audit,output,assessment_id=1,season=2026)
    with pytest.raises(ValueError, match="on/after the cutoff"):
        scorecard.matchup(output,["KC","BUF"],"2026-09-09",audit,"future")


def test_opponent_records_exclude_subject_and_repeat_opponents():
    finals=[('g1','2026-09-01','KC','BUF',21,14),
            ('g2','2026-09-08','BUF','KC',7,14),
            ('g3','2026-09-02','BUF','NYJ',10,10),
            ('g4','2026-09-03','MIA','BUF',0,7)]
    result=scorecard.opponent_records(finals,'KC',['g1','g2'])
    assert result['subject_games']==2
    assert (result['opponent_wins'],result['opponent_losses'],result['opponent_ties'])==(2,0,2)
    assert result['opponent_game_observations']==4
    assert result['opponent_win_fraction']==.75
    assert [d['opponent'] for d in result['details']]==['BUF','BUF']
    assert result['status']=='covered_for_snapshot'


def test_opponent_record_missing_sample_does_not_become_zero():
    finals=[('g1','2026-09-01','KC','BUF',21,14)]
    result=scorecard.opponent_records(finals,'KC',['g1'])
    assert result['status']=='incomplete'
    assert result['opponent_win_fraction'] is None
    assert result['games_without_opponent_sample']==['g1']
    assert scorecard.opponent_records(finals,'KC',[])['status']=='no_data'
    with pytest.raises(ValueError,match='duplicate'):
        scorecard.opponent_records(finals+finals,'KC',['g1'])


def test_opponent_cutoff_and_season_filter_integration(inputs):
    import sqlite3
    source,audit,output,_,_=inputs
    with sqlite3.connect(audit) as conn:
        conn.execute('ALTER TABLE query_coverage_games ADD COLUMN home_score INTEGER')
        conn.execute('ALTER TABLE query_coverage_games ADD COLUMN away_score INTEGER')
        conn.execute('UPDATE query_coverage_games SET home_score=21,away_score=14')
        conn.executemany("INSERT INTO query_coverage_games VALUES(1,?,'REG','final',1,?,?,'BUF','NYJ',?,?)",
                         [(2026,'early','2026-09-02',21,7),(2026,'same_day','2026-09-08',0,7),
                          (2026,'future','2026-09-09',0,7),(2025,'old','2025-09-01',0,7)])
    epa.build(source,audit,output,assessment_id=1,season=2026)
    report=scorecard.matchup(output,['KC','BUF'],'2026-09-08',audit)
    result=report['opponent_records']['KC']['season']
    assert result['opponent_win_fraction']==1
    assert result['opponent_game_observations']==1
    assert 'Schedule context: opponent win fraction | 100.0%' in scorecard.markdown(report)
    assert 'Not a team edge' in scorecard.markdown(report)


def test_opponent_records_say_snapshot_omissions_are_not_live_standings(inputs):
    import sqlite3
    source,audit,output,_,_=inputs
    with sqlite3.connect(audit) as conn:
        conn.execute('ALTER TABLE query_coverage_games ADD COLUMN home_score INTEGER')
        conn.execute('ALTER TABLE query_coverage_games ADD COLUMN away_score INTEGER')
        conn.execute('UPDATE query_coverage_games SET home_score=21,away_score=14')
        conn.executemany("INSERT INTO query_coverage_games VALUES(1,2026,'REG','final',?,?,?,?,?,?,?)",
                         [(1,'g3','2026-09-02','BUF','NYJ',21,7),(0,'g4','2026-09-03','MIA','BUF',30,0)])
    conn.close()
    epa.build(source,audit,output,assessment_id=1,season=2026)
    report=scorecard.matchup(output,['KC','BUF'],'2026-09-30',audit)
    result=report['opponent_records']['KC']['season']
    assert (result['opponent_wins'],result['opponent_losses'])==(2,0)  # BUF 1-0 per meeting; unverified g4 not counted
    assert 'not complete or live standings' in result['basis']
    assert 'not counted, so these are not complete or live standings' in scorecard.markdown(report)
