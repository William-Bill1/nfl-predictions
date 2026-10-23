"""Player-prop Stage 2 source freshness and provenance (player_props/provenance.py)
and the gate in predict.py. A synthetic four-team league is built through the
real Stage 1 pipeline into tmp_path; no network, and data_files/ is never read
or written."""
import gzip
import hashlib
import json
import os
import socket
from datetime import date

import pandas as pd
import pytest

from player_props import aggregators as ag
from player_props import provenance as pv


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


PAIRS = {1: [("KC", "BUF"), ("DEN", "LV")], 2: [("BUF", "DEN"), ("LV", "KC")],
         3: [("KC", "DEN"), ("BUF", "LV")], 19: [("KC", "BUF")]}
WEEK_DATES = {(2025, 1): "2025-09-07", (2025, 2): "2025-09-14", (2025, 3): "2025-09-21",
              (2025, 19): "2026-01-11",
              (2026, 1): "2026-09-13", (2026, 2): "2026-09-20", (2026, 3): "2026-09-27"}
ORDER = list(WEEK_DATES)                         # chronological (season, week)
SCORES = [(24, 17), (20, 23)]                    # (home, away) for a week's 1st and 2nd game


def gid(season, week, home, away):
    return f"{season}_{week:02d}_{away}_{home}"


def games(completed_through=(2026, 2)):
    """Game schedule as nfl_games_historical.csv: scores for every game up to
    and including `completed_through`, none after."""
    rows = []
    for season, week in ORDER:
        for i, (home, away) in enumerate(PAIRS[week]):
            done = ORDER.index((season, week)) <= ORDER.index(completed_through)
            home_score, away_score = SCORES[i]
            rows.append(dict(game_id=gid(season, week, home, away), season=season, week=week,
                             game_type="REG" if week < 19 else "WC",
                             gameday=WEEK_DATES[(season, week)], home_team=home, away_team=away,
                             home_score=float(home_score) if done else None,
                             away_score=float(away_score) if done else None,
                             result=float(home_score - away_score) if done else None,
                             total=float(home_score + away_score) if done else None))
    return pd.DataFrame(rows)


def pbp(through=(2026, 2), skip=(), unended=()):
    """Plays for every game up to `through`: each team's QB->WR pass and RB run,
    then the END GAME play with the recorded score (left out for `unended`)."""
    plays = []
    for season, week in ORDER[:ORDER.index(through) + 1]:
        for i, (home, away) in enumerate(PAIRS[week]):
            game = gid(season, week, home, away)
            if game in skip:
                continue
            if game not in unended:
                plays.append(dict(game_id=game, season=season, week=week,
                                  game_date=WEEK_DATES[(season, week)], desc="END GAME",
                                  total_home_score=SCORES[i][0], total_away_score=SCORES[i][1]))
            for team, opp in ((home, away), (away, home)):
                base = dict(game_id=game, season=season, week=week, posteam=team, defteam=opp,
                            game_date=WEEK_DATES[(season, week)], interception=0, desc="play",
                            total_home_score=None, total_away_score=None,
                            passer_player_id=None, passer_player_name=None,
                            rusher_player_id=None, rusher_player_name=None,
                            receiver_player_id=None, receiver_player_name=None,
                            passing_yards=0, rushing_yards=0, receiving_yards=0,
                            pass_touchdown=0, rush_touchdown=0, complete_pass=0,
                            pass_attempt=0, rush_attempt=0)
                plays.append({**base, 'passer_player_id': f"{team}-QB", 'passer_player_name': f"{team}.QB",
                              'receiver_player_id': f"{team}-WR", 'receiver_player_name': f"{team}.WR",
                              'passing_yards': 20 + week, 'receiving_yards': 20 + week,
                              'complete_pass': 1, 'pass_attempt': 1})
                plays.append({**base, 'rusher_player_id': f"{team}-RB", 'rusher_player_name': f"{team}.RB",
                              'rushing_yards': 5 + week, 'rush_attempt': 1})
    return pd.DataFrame(plays)


def write_source(path, df):
    with gzip.open(path, "wt", newline="") as f:
        df.to_csv(f, sep="\t", index=False)
    return path


def league(tmp_path, through=(2026, 2), completed=(2026, 2), skip=(), sched=None, plays=None,
           unended=()):
    """Build stats + manifest from a synthetic source, the Stage 1 way. Returns
    the data directory (stats, manifest, nfl_games_historical.csv)."""
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    sched = games(completed) if sched is None else sched
    sched.to_csv(data / ag.SCHEDULE_FILE, sep="\t", index=False)
    src = write_source(tmp_path / pv.SOURCE_FILE, pbp(through, skip, unended) if plays is None else plays)
    plays, source = pv.read_source(src)
    tables, _ = ag.build_player_stats(plays, ag.load_matchup_schedule(data))
    ag.write_player_stats(tables, data, manifest_for=pv.stats_manifest(source))
    return data


def problems(result):
    return result.provenance_problems + result.freshness_problems


def snapshot(directory):
    return {p.name: p.read_bytes() for p in sorted(directory.iterdir()) if p.is_file()}


# ---------------------------------------------------------------------------
# Source observation and manifest
# ---------------------------------------------------------------------------

class TestSourceObservation:
    def test_hash_and_coverage_come_from_the_content(self, tmp_path):
        src = write_source(tmp_path / pv.SOURCE_FILE, pbp((2026, 2), skip={gid(2025, 2, "BUF", "DEN")}))
        df, obs = pv.read_source(src)
        assert obs["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
        assert obs["bytes"] == src.stat().st_size
        cov = obs["coverage"]
        assert set(cov["seasons"]) == {"2025", "2026"}
        assert cov["seasons"]["2025"]["games"] == 6           # 7 scheduled, one absent
        assert cov["seasons"]["2025"]["weeks"] == [1, 19]
        assert cov["seasons"]["2026"]["last_date"] == "2026-09-20"
        assert (cov["first_date"], cov["last_date"]) == ("2025-09-07", "2026-09-20")
        assert cov["games"] == df["game_id"].nunique() == 10

    def test_observation_time_is_not_the_file_modification_time(self, tmp_path):
        src = write_source(tmp_path / pv.SOURCE_FILE, pbp())
        os.utime(src, (946684800, 946684800))                  # mtime 2000-01-01
        _, obs = pv.read_source(src)
        assert not obs["observed_at"].startswith("2000")
        assert "mtime" not in json.dumps(obs) and "generated" not in json.dumps(obs)

    def test_manifest_links_every_file(self, tmp_path):
        data = league(tmp_path)
        m = json.loads((data / pv.MANIFEST_FILE).read_text())
        src = tmp_path / pv.SOURCE_FILE
        assert m["kind"] == pv.MANIFEST_KIND and m["manifest_version"] == pv.MANIFEST_VERSION
        assert m["source"]["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
        assert m["source"]["coverage"]["seasons"]["2026"]["games"] == 4
        assert m["aggregation"]["feature_definition"] == ag.FEATURE_DEFINITION
        assert m["aggregation"]["rolling_windows"] == ag.ROLLING_WINDOWS
        assert m["aggregation"]["code"]["sha256"] == hashlib.sha256(
            (pv.ROOT / pv.AGGREGATION_CODE).read_bytes()).hexdigest()
        for stat_type, name in ag.STATS_FILES.items():
            entry = m["stats"][stat_type]
            assert entry["sha256"] == hashlib.sha256((data / name).read_bytes()).hexdigest()
            assert entry["columns"] == ag.output_columns(stat_type)
            assert entry["coverage"]["games"] == 11
            assert entry["coverage"]["last_date"] == "2026-09-20"
        assert not list(data.glob("*.tmp")) and not list(data.glob("*.bak"))

    def test_manifest_needs_a_source_observation(self):
        with pytest.raises(ValueError, match="source observation"):
            pv.stats_manifest({})

    def test_writing_without_a_manifest_leaves_the_old_one_unmatched(self, tmp_path):
        data = league(tmp_path)
        plays, _ = pv.read_source(tmp_path / pv.SOURCE_FILE)
        tables, _ = ag.build_player_stats(plays.iloc[:-2], ag.load_matchup_schedule(data))
        ag.write_player_stats(tables, data)
        assert any("does not match its manifest hash" in p
                   for p in pv.check_inputs(2026, 3, data).provenance_problems)

    def test_failed_manifest_replacement_restores_stats_and_manifest(self, tmp_path, monkeypatch):
        data = league(tmp_path)
        before = snapshot(data)
        plays, source = pv.read_source(tmp_path / pv.SOURCE_FILE)
        tables, _ = ag.build_player_stats(plays.iloc[:-2], ag.load_matchup_schedule(data))
        real = os.replace

        def fail_on_manifest(src, dst):
            if str(dst).endswith(pv.MANIFEST_FILE) and not str(src).endswith(".bak"):
                raise OSError("disk full")
            return real(src, dst)
        monkeypatch.setattr(ag.os, "replace", fail_on_manifest)
        with pytest.raises(OSError, match="disk full"):
            ag.write_player_stats(tables, data, manifest_for=pv.stats_manifest(source))
        assert snapshot(data) == before


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_valid_complete_coverage_passes(self, tmp_path):
        result = pv.check_inputs(2026, 3, league(tmp_path))
        assert result.ok, problems(result)
        assert result.summary["expected_games"] == {"2025": 7, "2026": 4}
        assert result.summary["feature_cutoff"] == "2026-09-27"
        assert "RESULT: PASS" in pv.format_report(result)

    def test_missing_manifest_means_unverified(self, tmp_path):
        data = league(tmp_path)
        (data / pv.MANIFEST_FILE).unlink()
        result = pv.check_inputs(2026, 3, data)
        assert not result.ok and result.feature_tables == {}
        [p] = result.provenance_problems
        assert "no provenance manifest" in p and "generation time are unknown" in p
        # coverage is complete; only the source corroboration of completion is missing
        [f] = result.freshness_problems
        assert pv.COMPLETION_REASONS["no_source"] in f

    def test_no_manifest_is_created_for_existing_files(self, tmp_path):
        data = league(tmp_path)
        (data / pv.MANIFEST_FILE).unlink()
        before = snapshot(data)
        pv.check_inputs(2026, 3, data)
        assert pv.main(["--season", "2026", "--week", "3", "--data-dir", str(data)]) == 1
        assert snapshot(data) == before

    @pytest.mark.parametrize("name", list(ag.STATS_FILES.values()))
    def test_stats_changed_after_the_manifest_fail(self, tmp_path, name):
        data = league(tmp_path)
        path = data / name
        path.write_bytes(path.read_bytes().replace(b",1,", b",0,", 1))
        result = pv.check_inputs(2026, 3, data)
        assert any(name in p and "does not match its manifest hash" in p
                   for p in result.provenance_problems)
        assert not result.ok

    def test_missing_stats_file_fails(self, tmp_path):
        data = league(tmp_path)
        (data / ag.STATS_FILES["rushing"]).unlink()
        assert "player_rushing_stats.csv is missing" in pv.check_inputs(2026, 3, data).provenance_problems

    @pytest.mark.parametrize("edit,message", [
        (lambda m: m.update(kind="other"), "not a version-1"),
        (lambda m: m.update(manifest_version=2), "not a version-1"),
        (lambda m: m["source"].pop("sha256"), "source.sha256 must be"),
        (lambda m: m["aggregation"].update(feature_definition="stage0"), "feature definition"),
        (lambda m: m["aggregation"].update(rolling_windows=[3, 5]), "rolling windows"),
        (lambda m: m["aggregation"]["code"].pop("sha256"), "aggregation.code.sha256 must be"),
        (lambda m: m["stats"].pop("passing"), "no entry for player_passing_stats.csv"),
        (lambda m: m["stats"]["receiving"]["columns"].pop(), "manifest columns differ"),
    ])
    def test_manifest_contents_are_checked(self, tmp_path, edit, message):
        data = league(tmp_path)
        m = json.loads((data / pv.MANIFEST_FILE).read_text())
        edit(m)
        (data / pv.MANIFEST_FILE).write_text(json.dumps(m))
        result = pv.check_inputs(2026, 3, data)
        assert any(message in p for p in result.provenance_problems), result.provenance_problems

    def test_forged_manifest_for_off_contract_files_fails(self, tmp_path):
        # A hand-made manifest whose hash matches a file with an extra column.
        data = league(tmp_path)
        name = ag.STATS_FILES["passing"]
        df = pd.read_csv(data / name).assign(opponent_def_rank=1)
        df.to_csv(data / name, index=False)
        m = json.loads((data / pv.MANIFEST_FILE).read_text())
        m["stats"]["passing"]["sha256"] = hashlib.sha256((data / name).read_bytes()).hexdigest()
        (data / pv.MANIFEST_FILE).write_text(json.dumps(m))
        result = pv.check_inputs(2026, 3, data)
        assert f"{name}: columns differ from the current contract" in result.provenance_problems

    def test_unreadable_manifest_fails(self, tmp_path):
        data = league(tmp_path)
        (data / pv.MANIFEST_FILE).write_text("{not json")
        assert any("could not be read" in p for p in pv.check_inputs(2026, 3, data).provenance_problems)

    def test_changed_source_and_code_are_reported_not_trusted(self, tmp_path, monkeypatch):
        data = league(tmp_path)
        write_source(data / pv.SOURCE_FILE, pbp((2026, 3)))       # a newer source arrives
        m = json.loads((data / pv.MANIFEST_FILE).read_text())
        m["aggregation"]["code"].update(sha256="0" * 64, git_dirty=True)
        (data / pv.MANIFEST_FILE).write_text(json.dumps(m))
        result = pv.check_inputs(2026, 3, data)
        assert result.ok
        notes = " ".join(result.notes)
        assert "is not the source these stats were built from" in notes
        assert "has changed since the stats were built" in notes
        assert "uncommitted changes" in notes


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------

class TestFreshness:
    def test_missing_game_behind_a_recent_maximum_date(self, tmp_path):
        missing = gid(2026, 1, "DEN", "LV")
        data = league(tmp_path, skip={missing})
        result = pv.check_inputs(2026, 3, data)
        # the latest date is the most recent completed week, yet a game is absent
        assert result.summary["feature_last_date"]["passing"] == "2026-09-20"
        assert not result.ok and result.provenance_problems == []
        for name in ag.STATS_FILES.values():
            assert any(name in p and "1 completed earlier game(s) missing (2026: 1)" in p
                       and missing in p for p in result.freshness_problems)

    def test_stale_stats_with_fresh_metadata_fail(self, tmp_path):
        data = league(tmp_path, through=(2026, 1), completed=(2026, 2))
        (data / "pbp_metadata.json").write_text(json.dumps(
            {"last_update": "2099-01-01T00:00:00", "total_plays": 10 ** 6, "seasons": "2020-2026"}))
        result = pv.check_inputs(2026, 3, data)
        assert result.provenance_problems == []            # the manifest itself is fresh and valid
        assert any("2 completed earlier game(s) missing (2026: 2)" in p for p in result.freshness_problems)
        assert any("pbp_metadata.json is not used as evidence" in n for n in result.notes)
        assert not result.ok

    def test_season_boundary_requires_the_whole_previous_season(self, tmp_path):
        postseason = gid(2025, 19, "KC", "BUF")
        result = pv.check_inputs(2026, 1, league(tmp_path, skip={postseason}))
        assert any("(2025: 1)" in p and postseason in p for p in result.freshness_problems)
        assert result.summary["expected_games"] == {"2025": 7}

    def test_season_boundary_features_are_the_previous_season_only(self, tmp_path):
        result = pv.check_inputs(2026, 1, league(tmp_path))
        assert result.ok, problems(result)
        for t in result.feature_tables.values():
            assert set(t["season"]) == {2025}
            assert gid(2025, 19, "KC", "BUF") in set(t["game_id"])
            assert t["game_date"].max() == "2026-01-11"

    def test_postseason_target_needs_the_regular_season(self, tmp_path):
        result = pv.check_inputs(2025, 19, league(tmp_path, skip={gid(2025, 3, "BUF", "LV")}))
        assert any("(2025: 1)" in p for p in result.freshness_problems)

    def test_earlier_seasons_in_the_stats_are_checked_too(self, tmp_path):
        # Target 2026 week 3 checks 2025 as well as 2026.
        result = pv.check_inputs(2026, 3, league(tmp_path, skip={gid(2025, 1, "KC", "BUF")}))
        assert any("(2025: 1)" in p for p in result.freshness_problems)

    def test_seasons_older_than_the_previous_one_are_checked(self, tmp_path):
        # Relabel 2025 as 2023, so the stats hold a season before S-1 (2025 has no games).
        def to_2023(df, date_col):
            df = df.copy()
            old = df["season"] == 2025
            df.loc[old, "season"] = 2023
            df.loc[old, "game_id"] = df.loc[old, "game_id"].str.replace("2025_", "2023_", n=1)
            df.loc[old, date_col] = df.loc[old, date_col].str.replace("2025-", "2023-").str.replace("2026-01", "2024-01")
            return df
        missing = "2023_02_DEN_BUF"
        plays = to_2023(pbp(), "game_date")
        data = league(tmp_path, sched=to_2023(games(), "gameday"),
                      plays=plays[plays["game_id"] != missing])
        result = pv.check_inputs(2026, 3, data)
        assert result.summary["first_checked_season"] == 2023
        assert any("(2023: 1)" in p and missing in p for p in result.freshness_problems)

    def test_stats_dated_differently_from_the_schedule_fail(self, tmp_path):
        data = league(tmp_path)
        sched = pd.read_csv(data / ag.SCHEDULE_FILE, sep="	")
        sched.loc[sched["game_id"] == gid(2026, 1, "KC", "BUF"), "gameday"] = "2026-09-12"
        sched.to_csv(data / ag.SCHEDULE_FILE, sep="	", index=False)
        result = pv.check_inputs(2026, 3, data)
        assert any("dated differently from their schedule game" in p for p in result.freshness_problems)

    def test_future_data_never_enters_the_feature_source(self, tmp_path):
        # Stats hold results through 2026 week 3; predicting week 2.
        data = league(tmp_path, through=(2026, 3), completed=(2026, 3))
        result = pv.check_inputs(2026, 2, data)
        assert result.ok, problems(result)
        for t in result.feature_tables.values():
            assert (t["game_date"] < "2026-09-20").all()
            later = (t["season"] > 2026) | ((t["season"] == 2026) & (t["week"] >= 2))
            assert not later.any()
        assert result.summary["excluded_rows"] == {"passing": 8, "rushing": 8, "receiving": 8}
        assert any("2 target-week game(s) already have recorded scores" in n for n in result.notes)

    def test_target_week_game_already_played_is_excluded(self, tmp_path):
        # Thursday game of the target week is final and in the stats (week 3, one game).
        sched = games((2026, 2))
        thursday = sched["game_id"] == gid(2026, 3, "KC", "DEN")
        sched.loc[thursday, ["home_score", "away_score", "result", "total"]] = [24.0, 17.0, 7.0, 41.0]
        sched.loc[thursday, "gameday"] = "2026-09-24"
        plays = pbp((2026, 3), skip={gid(2026, 3, "BUF", "LV")})
        plays.loc[plays["game_id"] == gid(2026, 3, "KC", "DEN"), "game_date"] = "2026-09-24"
        data = tmp_path / "data"
        data.mkdir()
        sched.to_csv(data / ag.SCHEDULE_FILE, sep="\t", index=False)
        plays, source = pv.read_source(write_source(tmp_path / pv.SOURCE_FILE, plays))
        tables, _ = ag.build_player_stats(plays, ag.load_matchup_schedule(data))
        ag.write_player_stats(tables, data, manifest_for=pv.stats_manifest(source))
        result = pv.check_inputs(2026, 3, data)
        assert result.ok, problems(result)
        assert result.summary["feature_cutoff"] == "2026-09-24"
        for t in result.feature_tables.values():
            assert gid(2026, 3, "KC", "DEN") not in set(t["game_id"])
            assert t["game_date"].max() == "2026-09-20"

    def test_rescheduled_game_after_the_target_start_is_not_expected(self, tmp_path):
        sched = games((2026, 2))
        moved = sched["game_id"] == gid(2026, 2, "BUF", "DEN")
        sched.loc[moved, ["gameday", "home_score", "away_score", "result", "total"]] = [
            "2026-10-04", None, None, None, None]
        data = league(tmp_path, skip={gid(2026, 2, "BUF", "DEN")}, sched=sched)
        result = pv.check_inputs(2026, 3, data)
        assert result.ok, problems(result)
        assert any("not expected" in n and gid(2026, 2, "BUF", "DEN") in n for n in result.notes)


class TestAmbiguousScheduleEvidence:
    def test_earlier_game_without_a_final_score(self, tmp_path):
        data = league(tmp_path, completed=(2026, 1))     # week 2 played, no scores recorded
        result = pv.check_inputs(2026, 3, data)
        assert any("2 earlier game(s) have no recorded score" in p
                   for p in result.freshness_problems)
        # their stats rows can't be shown complete either
        assert any("without completion evidence" in p for p in result.freshness_problems)
        assert not result.ok

    def test_only_one_score(self, tmp_path):
        sched = games()
        sched.loc[sched["game_id"] == gid(2026, 1, "KC", "BUF"), "away_score"] = None
        result = pv.check_inputs(2026, 3, league(tmp_path, sched=sched))
        assert any("only one team's score" in p for p in result.freshness_problems)

    def test_duplicate_game_ids(self, tmp_path):
        data = league(tmp_path)
        sched = pd.read_csv(data / ag.SCHEDULE_FILE, sep="\t")
        pd.concat([sched, sched.iloc[[0]]]).to_csv(data / ag.SCHEDULE_FILE, sep="\t", index=False)
        assert any("more than once" in p for p in pv.check_inputs(2026, 3, data).freshness_problems)

    def test_team_twice_in_one_week(self, tmp_path):
        data = league(tmp_path)
        sched = pd.read_csv(data / ag.SCHEDULE_FILE, sep="\t")
        extra = sched.iloc[[0]].assign(game_id="2025_01_KC_LV", away_team="KC", home_team="LV")
        pd.concat([sched, extra]).to_csv(data / ag.SCHEDULE_FILE, sep="\t", index=False)
        assert any("two games in one week" in p for p in pv.check_inputs(2026, 3, data).freshness_problems)

    def test_no_games_for_the_target_week(self, tmp_path):
        result = pv.check_inputs(2026, 9, league(tmp_path))
        assert result.freshness_problems == ["schedule has no games for 2026 week 9"]

    def test_invalid_target_week_date(self, tmp_path):
        sched = games()
        sched.loc[sched["game_id"] == gid(2026, 3, "KC", "DEN"), "gameday"] = "TBD"
        result = pv.check_inputs(2026, 3, league(tmp_path, sched=sched))
        assert any("invalid dates for target-week" in p for p in result.freshness_problems)

    def test_invalid_earlier_date(self, tmp_path):
        data = league(tmp_path)
        sched = pd.read_csv(data / ag.SCHEDULE_FILE, sep="\t")
        sched.loc[sched["game_id"] == gid(2025, 2, "BUF", "DEN"), "gameday"] = "2025-13-01"
        sched.to_csv(data / ag.SCHEDULE_FILE, sep="\t", index=False)
        assert any("no valid date" in p for p in pv.check_inputs(2026, 3, data).freshness_problems)

    def test_stats_game_missing_from_the_schedule(self, tmp_path):
        data = league(tmp_path)
        sched = pd.read_csv(data / ag.SCHEDULE_FILE, sep="\t")
        sched[sched["game_id"] != gid(2025, 1, "DEN", "LV")].to_csv(
            data / ag.SCHEDULE_FILE, sep="\t", index=False)
        assert any("not in the schedule" in p for p in pv.check_inputs(2026, 3, data).freshness_problems)

    def test_prediction_schedule_disagrees(self, tmp_path):
        result = pv.check_inputs(2026, 3, league(tmp_path),
                                 prediction_matchups={("KC", "DEN"), ("LV", "BUF")})
        assert any("disagree" in p and "BUF@LV" in p for p in result.freshness_problems)

    def test_unreadable_schedule(self, tmp_path):
        data = league(tmp_path)
        (data / ag.SCHEDULE_FILE).unlink()
        result = pv.check_inputs(2026, 3, data)
        assert any("could not be read" in p for p in result.freshness_problems) and not result.ok

    def test_game_with_stats_for_only_one_team(self, tmp_path):
        data = league(tmp_path)
        for stat_type, name in ag.STATS_FILES.items():
            df = pd.read_csv(data / name)
            df = df[~((df["game_id"] == gid(2026, 1, "KC", "BUF")) & (df["team"] == "BUF"))]
            df.to_csv(data / name, index=False)
        result = pv.check_inputs(2026, 3, data)
        assert any("only one team" in p for p in result.freshness_problems)


# ---------------------------------------------------------------------------
# CLI and prediction gate
# ---------------------------------------------------------------------------

class TestCommand:
    def test_exit_codes_and_diagnostics(self, tmp_path, capsys):
        data = league(tmp_path)
        assert pv.main(["--season", "2026", "--week", "3", "--data-dir", str(data)]) == 0
        assert "RESULT: PASS" in capsys.readouterr().out
        (data / pv.MANIFEST_FILE).unlink()
        assert pv.main(["--season", "2026", "--week", "3", "--data-dir", str(data)]) == 1
        out = capsys.readouterr().out
        assert "PROVENANCE: NOT ESTABLISHED" in out and "RESULT: FAIL" in out

    def test_check_is_read_only(self, tmp_path):
        data = league(tmp_path, completed=(2026, 1))
        before = snapshot(data)
        assert pv.main(["--season", "2026", "--week", "3", "--data-dir", str(data)]) == 1
        assert snapshot(data) == before

    @pytest.mark.parametrize("argv", [["--season", "2026"], ["--data-dir", "x"]])
    def test_incomplete_arguments(self, argv):
        with pytest.raises(SystemExit) as exc:
            pv.main(argv)
        assert exc.value.code == 2


def _prediction_files(data):
    pd.DataFrame({"week": [2], "season": [2026], "player_name": ["KC.QB"]}).to_csv(
        data / "player_props_predictions.csv", index=False)
    pd.DataFrame({"week": [2], "season": [2026], "player_name": ["old"]}).to_csv(
        data / "player_props_predictions_week2_2026.csv", index=False)
    sched = games()
    sched["date"] = sched["gameday"]
    sched[sched["season"] == 2026][["week", "date", "home_team", "away_team"]].to_csv(
        data / "nfl_schedule_2026.csv", index=False)


class TestPredictionGate:
    @pytest.fixture
    def predict(self, monkeypatch):
        from player_props import predict
        monkeypatch.setattr(predict, "fetch_market_odds",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError("odds fetched")))
        return predict

    @pytest.mark.parametrize("breakage", ["no_manifest", "stale", "tampered"])
    def test_invalid_inputs_leave_predictions_byte_identical(self, tmp_path, monkeypatch, predict, breakage):
        data = league(tmp_path, through=(2026, 1) if breakage == "stale" else (2026, 2))
        if breakage == "no_manifest":
            (data / pv.MANIFEST_FILE).unlink()
        if breakage == "tampered":
            p = data / ag.STATS_FILES["passing"]
            p.write_bytes(p.read_bytes() + b"\n")
        _prediction_files(data)
        before = snapshot(data)
        monkeypatch.setattr(predict, "DATA_DIR", data)
        monkeypatch.setattr(predict, "load_models",
                            lambda: (_ for _ in ()).throw(AssertionError("models loaded")))
        code = predict.generate_predictions(season=2026, week=3, skip_injuries=True,
                                            skip_weather=True, skip_market_odds=False)
        assert code == 1
        assert snapshot(data) == before                    # nothing new, nothing changed
        assert not (data / "player_props_predictions_week3_2026.csv").exists()

    def test_valid_inputs_feed_only_earlier_games(self, tmp_path, monkeypatch, predict):
        data = league(tmp_path, through=(2026, 3), completed=(2026, 3))
        _prediction_files(data)
        monkeypatch.setattr(predict, "DATA_DIR", data)
        monkeypatch.setattr(predict, "load_models", lambda: {"m": object()})
        seen = []

        def capture(game_row, all_stats, models, **kw):
            seen.append(all_stats)
            return []
        monkeypatch.setattr(predict, "predict_props_for_game", capture)
        predict.generate_predictions(season=2026, week=2, skip_injuries=True, skip_weather=True,
                                     skip_market_odds=True)
        assert len(seen) == 2
        for t in seen[0].values():
            assert (t["game_date"] < "2026-09-20").all()
            assert not ((t["season"] == 2026) & (t["week"] >= 2)).any()


def test_training_entry_point_writes_the_manifest(tmp_path, monkeypatch):
    from player_props import train_models as tm
    games().to_csv(tmp_path / ag.SCHEDULE_FILE, sep="\t", index=False)
    write_source(tmp_path / pv.SOURCE_FILE, pbp())
    monkeypatch.setattr(tm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(tm, "PBP_FILE", tmp_path / pv.SOURCE_FILE)
    plays, source = tm._load_pbp()
    tm._aggregate_and_roll(plays, source)
    assert pv.check_inputs(2026, 3, tmp_path).ok


# ---------------------------------------------------------------------------
# Completion evidence
# ---------------------------------------------------------------------------

PRIOR = gid(2026, 1, "KC", "BUF")                 # week 1, first game: 24-17 in the fixture


def _with_schedule(tmp_path, **values):
    sched = games()
    for col, value in values.items():
        sched[col] = sched[col].astype(object)
        sched.loc[sched["game_id"] == PRIOR, col] = value
    return league(tmp_path, sched=sched)


class TestCompletionEvidence:
    def test_valid_completed_games_pass(self, tmp_path):
        result = pv.check_inputs(2026, 3, league(tmp_path))
        assert result.ok, problems(result)
        assert PRIOR in set(result.feature_tables["passing"]["game_id"])

    @pytest.mark.parametrize("values,reason", [
        (dict(home_score=0.0, away_score=0.0, result=0.0, total=0.0), "placeholder"),
        (dict(result=None, total=None), "no_result_total"),          # 24-17 with nothing else
        (dict(home_score=7.0, away_score=0.0, result=None, total=None), "no_result_total"),
        (dict(result=None), "no_result_total"),
        (dict(total=None), "no_result_total"),
        (dict(result=3.0), "inconsistent"),
        (dict(total=40.0), "inconsistent"),
        (dict(result="x"), "inconsistent"),
        (dict(home_score=1.0, result=-16.0, total=18.0), "invalid_score"),
        (dict(home_score=-3.0, result=-20.0, total=14.0), "invalid_score"),
        (dict(home_score=24.5, result=7.5, total=41.5), "invalid_score"),
        (dict(home_score="TBD"), "invalid_score"),
        (dict(away_score=None), "one_score"),
        (dict(home_score=None, away_score=None, result=None, total=None), "no_score"),
        # recorded scores that agree with each other but not with the play-by-play END GAME
        (dict(home_score=31.0, result=14.0, total=48.0), "source_mismatch"),
    ])
    def test_insufficient_or_inconsistent_evidence_fails(self, tmp_path, values, reason):
        result = pv.check_inputs(2026, 3, _with_schedule(tmp_path, **values))
        assert not result.ok and result.feature_tables == {}
        text = pv.COMPLETION_REASONS[reason]
        assert any(text in p and PRIOR in p for p in result.freshness_problems), result.freshness_problems

    def test_zero_zero_is_never_completed(self, tmp_path):
        result = pv.check_inputs(2026, 3, _with_schedule(
            tmp_path, home_score=0.0, away_score=0.0, result=0.0, total=0.0))
        assert any("show 0-0" in p for p in result.freshness_problems)

    def test_shutout_is_a_valid_result(self, tmp_path):
        sched = games()
        sched.loc[sched["game_id"] == PRIOR, ["home_score", "away_score", "result", "total"]] = [24.0, 0.0, 24.0, 24.0]
        plays = pbp()
        plays.loc[(plays["game_id"] == PRIOR) & (plays["desc"] == "END GAME"), "total_away_score"] = 0
        assert pv.check_inputs(2026, 3, league(tmp_path, sched=sched, plays=plays)).ok

    def test_missing_end_game_play_fails(self, tmp_path):
        result = pv.check_inputs(2026, 3, league(tmp_path, unended={PRIOR}))
        assert any(pv.COMPLETION_REASONS["source_unended"] in p and PRIOR in p
                   for p in result.freshness_problems)

    def test_game_dated_today_is_not_complete_yet(self, tmp_path):
        data = league(tmp_path)
        result = pv.check_inputs(2026, 3, data, today=date(2026, 9, 20))
        assert any(pv.COMPLETION_REASONS["not_past"] in p and gid(2026, 2, "BUF", "DEN") in p
                   for p in result.freshness_problems)
        assert pv.check_inputs(2026, 3, data, today=date(2026, 9, 21)).ok

    def test_without_a_manifest_completion_is_not_corroborated(self, tmp_path):
        data = league(tmp_path)
        (data / pv.MANIFEST_FILE).unlink()
        result = pv.check_inputs(2026, 3, data)
        assert any(pv.COMPLETION_REASONS["no_source"] in p for p in result.freshness_problems)
        # coverage is still checked game by game
        assert result.summary["expected_games"] == {"2025": 7, "2026": 4}

    @pytest.mark.parametrize("column", ["result", "total", "home_score"])
    def test_missing_completion_column_fails(self, tmp_path, column):
        data = league(tmp_path)
        sched = pd.read_csv(data / ag.SCHEDULE_FILE, sep="\t").drop(columns=[column])
        sched.to_csv(data / ag.SCHEDULE_FILE, sep="\t", index=False)
        result = pv.check_inputs(2026, 3, data)
        assert any("could not be read" in p for p in result.freshness_problems) and not result.ok

    def test_source_without_end_game_columns_corroborates_nothing(self, tmp_path):
        plays = pbp()
        plays = plays[plays["desc"] != "END GAME"].drop(columns=["desc", "total_home_score", "total_away_score"])
        result = pv.check_inputs(2026, 3, league(tmp_path, plays=plays))
        assert any(pv.COMPLETION_REASONS["not_in_source"] in p for p in result.freshness_problems)


# ---------------------------------------------------------------------------
# Malformed manifests
# ---------------------------------------------------------------------------

_DROP = object()


def _set(m, path, value):
    target = m
    for key in path[:-1]:
        target = target[key]
    if value is _DROP:
        target.pop(path[-1])
    else:
        target[path[-1]] = value


MALFORMED = [
    (("source",), [1, 2], "source must be an object"),
    (("source",), "nfl.csv.gz", "source must be an object"),
    (("source", "coverage"), [1], "source.coverage must be an object"),
    (("source", "coverage", "seasons"), ["2025"], "source.coverage.seasons must be an object"),
    (("source", "coverage", "seasons", "2025"), [], "source.coverage.seasons has malformed entries"),
    (("source", "coverage", "rows"), "many", "source.coverage.rows"),
    (("source", "coverage", "last_date"), "2026-02-30", "source.coverage.last_date"),
    (("source", "completion"), [], "source.completion must be an object"),
    (("source", "completion", PRIOR), {"end_game": "yes"}, "source.completion has malformed entries"),
    (("source", "sha256"), "abc", "source.sha256 must be"),
    (("source", "sha256"), _DROP, "source.sha256 must be"),
    (("source", "bytes"), True, "source.bytes"),
    (("source", "observed_at"), "2026-10-01T00:00:00", "source.observed_at must be"),
    (("source", "observed_at"), "2999-01-01T00:00:00+00:00", "source.observed_at is in the future"),
    (("written_at",), _DROP, "written_at must be"),
    (("written_at",), "yesterday", "written_at must be"),
    (("written_at",), "2999-01-01T00:00:00+00:00", "written_at is in the future"),
    (("written_at",), "2000-01-01T00:00:00+00:00", "observed_at is after written_at"),
    (("aggregation",), [], "aggregation must be an object"),
    (("aggregation", "code"), [1], "aggregation.code must be an object"),
    (("aggregation", "code", "file"), "other.py", "aggregation.code.file"),
    (("aggregation", "code", "git_commit"), 5, "aggregation.code.git_commit"),
    (("aggregation", "code", "git_dirty"), "no", "aggregation.code.git_dirty"),
    (("aggregation", "rolling_windows"), "3,5,10", "aggregation.rolling_windows"),
    (("aggregation", "feature_definition"), 1, "aggregation.feature_definition"),
    (("stats",), [], "stats must be an object"),
    (("stats", "passing"), [1], "stats.passing must be an object"),
    (("stats", "rushing", "file"), "other.csv", "stats.rushing.file"),
    (("stats", "receiving", "sha256"), 12, "stats.receiving.sha256"),
    (("stats", "receiving", "bytes"), -1, "stats.receiving.bytes"),
    (("stats", "passing", "columns"), "player_id", "stats.passing.columns"),
    (("stats", "passing", "coverage"), None, "stats.passing.coverage must be an object"),
]


class TestMalformedManifest:
    @pytest.mark.parametrize("path,value,message", MALFORMED,
                             ids=[f"{'.'.join(p)}-{i}" for i, (p, _, _) in enumerate(MALFORMED)])
    def test_malformed_manifest_is_a_clear_failure(self, tmp_path, capsys, path, value, message):
        data = league(tmp_path)
        m = json.loads((data / pv.MANIFEST_FILE).read_text())
        _set(m, path, value)
        (data / pv.MANIFEST_FILE).write_text(json.dumps(m))
        result = pv.check_inputs(2026, 3, data)           # never raises
        assert not result.ok and result.manifest is None and result.feature_tables == {}
        assert any("is malformed" in p and message in p for p in result.provenance_problems), \
            result.provenance_problems
        assert pv.main(["--season", "2026", "--week", "3", "--data-dir", str(data)]) == 1
        out = capsys.readouterr().out
        assert "Traceback" not in out and "PROVENANCE: NOT ESTABLISHED" in out

    @pytest.mark.parametrize("content", ["[]", "null", "42", '"text"'])
    def test_manifest_that_is_not_an_object(self, tmp_path, content):
        data = league(tmp_path)
        (data / pv.MANIFEST_FILE).write_text(content)
        result = pv.check_inputs(2026, 3, data)
        assert any("must be a JSON object" in p for p in result.provenance_problems)

    def test_forged_coverage_fails(self, tmp_path):
        data = league(tmp_path)
        m = json.loads((data / pv.MANIFEST_FILE).read_text())
        m["stats"]["passing"]["coverage"]["last_date"] = "2026-10-18"
        (data / pv.MANIFEST_FILE).write_text(json.dumps(m))
        result = pv.check_inputs(2026, 3, data)
        assert "player_passing_stats.csv: manifest coverage does not match the file's content" \
            in result.provenance_problems

    def test_malformed_manifest_blocks_predictions(self, tmp_path, monkeypatch):
        from player_props import predict
        data = league(tmp_path)
        m = json.loads((data / pv.MANIFEST_FILE).read_text())
        m["source"] = ["not", "an", "object"]
        (data / pv.MANIFEST_FILE).write_text(json.dumps(m))
        _prediction_files(data)
        before = snapshot(data)
        monkeypatch.setattr(predict, "DATA_DIR", data)
        assert predict.generate_predictions(season=2026, week=3, skip_injuries=True,
                                            skip_weather=True, skip_market_odds=True) == 1
        assert snapshot(data) == before

    def test_unexpected_check_error_blocks_predictions(self, tmp_path, monkeypatch, capsys):
        from player_props import predict
        data = league(tmp_path)
        _prediction_files(data)
        before = snapshot(data)
        monkeypatch.setattr(predict, "DATA_DIR", data)
        monkeypatch.setattr(predict, "check_inputs",
                            lambda *a, **k: (_ for _ in ()).throw(KeyError("boom")))
        assert predict.generate_predictions(season=2026, week=3, skip_injuries=True,
                                            skip_weather=True, skip_market_odds=True) == 1
        assert snapshot(data) == before
        assert "could not complete" in capsys.readouterr().out

    def test_unexpected_check_error_in_the_command(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(pv, "check_inputs", lambda *a, **k: (_ for _ in ()).throw(KeyError("boom")))
        assert pv.main(["--season", "2026", "--week", "3", "--data-dir", str(tmp_path)]) == 2
        assert "not established" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Backtest prediction identity
# ---------------------------------------------------------------------------

class TestBacktestIdentity:
    @pytest.fixture
    def workdir(self, tmp_path, monkeypatch):
        from player_props import backtest
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data_files").mkdir()
        scored = []
        monkeypatch.setattr(backtest, "collect_actual_results",
                            lambda *a: scored.append(a) or (pd.DataFrame(), "stop here"))
        return tmp_path / "data_files", scored

    @staticmethod
    def write(path, **cols):
        base = {"player_name": ["A", "B"], "prop_type": ["passing_yards"] * 2,
                "line_value": [200.5, 50.5], "recommendation": ["OVER", "UNDER"],
                "confidence": [0.6, 0.6]}
        pd.DataFrame({**cols, **base}).to_csv(path, index=False)

    @pytest.mark.parametrize("name", ["player_props_predictions.csv",
                                      "player_props_predictions_week8.csv",
                                      "player_props_predictions_week8_2026.csv"])
    @pytest.mark.parametrize("cols,message", [
        ({}, "has no week/season column"),
        ({"week": [8, 8]}, "has no season column"),
        ({"season": [2026, 2026]}, "has no week column"),
        ({"week": [7, 7], "season": [2026, 2026]}, "holds predictions for Week 7 2026"),
        ({"week": [8, 7], "season": [2026, 2026]}, "Week 7 2026, Week 8 2026"),
        ({"week": [8, 8], "season": [2026, 2025]}, "Week 8 2025, Week 8 2026"),
        ({"week": [8.5, 8], "season": [2026, 2026]}, "Week 8.5 2026"),
        ({"week": [8, None], "season": [2026, 2026]}, "missing or non-numeric"),
        ({"week": ["8", "eight"], "season": [2026, 2026]}, "missing or non-numeric"),
    ])
    def test_rows_must_all_be_the_requested_week(self, workdir, capsys, name, cols, message):
        from player_props import backtest
        data, scored = workdir
        self.write(data / name, **cols)
        assert backtest.run_weekly_accuracy_check(8, 2026) == {}
        out = capsys.readouterr().out
        assert message in out and "not scoring it as Week 8 2026" in out
        assert scored == []

    def test_empty_file_is_refused(self, workdir, capsys):
        from player_props import backtest
        data, scored = workdir
        pd.DataFrame(columns=["week", "season", "player_name"]).to_csv(
            data / "player_props_predictions_week8_2026.csv", index=False)
        assert backtest.run_weekly_accuracy_check(8, 2026) == {}
        assert "has no rows" in capsys.readouterr().out and scored == []

    @pytest.mark.parametrize("name", ["player_props_predictions_week8_2026.csv",
                                      "player_props_predictions.csv"])
    def test_matching_rows_are_scored(self, workdir, capsys, name):
        from player_props import backtest
        data, scored = workdir
        self.write(data / name, week=[8, 8], season=[2026, 2026])
        backtest.run_weekly_accuracy_check(8, 2026)
        assert scored == [(8, 2026)]
        assert "Loaded 2 predictions" in capsys.readouterr().out

    def test_frozen_snapshot_is_preferred_over_a_matching_feed(self, workdir, capsys):
        from player_props import backtest
        data, scored = workdir
        self.write(data / "player_props_predictions_week8_2026.csv", week=[8, 8], season=[2026, 2026])
        self.write(data / "player_props_predictions.csv", week=[9, 9], season=[2026, 2026])
        backtest.run_weekly_accuracy_check(8, 2026)
        out = capsys.readouterr().out
        assert scored == [(8, 2026)] and "WARNING" not in out


class TestSourceCompletionEvidence:
    def test_end_game_flag_is_required_not_just_a_score(self, tmp_path):
        data = league(tmp_path)
        m = json.loads((data / pv.MANIFEST_FILE).read_text())
        m["source"]["completion"][PRIOR]["end_game"] = False
        (data / pv.MANIFEST_FILE).write_text(json.dumps(m))
        result = pv.check_inputs(2026, 3, data)
        assert any(pv.COMPLETION_REASONS["source_unended"] in p and PRIOR in p
                   for p in result.freshness_problems)

    def test_uncorroborated_games_never_enter_the_feature_source(self, tmp_path):
        data = league(tmp_path)
        m = json.loads((data / pv.MANIFEST_FILE).read_text())
        completion = dict(m["source"]["completion"])
        completion.pop(PRIOR)
        tables = {s: pd.read_csv(data / n) for s, n in ag.STATS_FILES.items()}
        problems_, _, features, _ = pv.check_freshness(
            tables, pv.load_games(data), 2026, 3, source_completion=completion)
        assert any(PRIOR in p for p in problems_)
        for t in features.values():
            assert PRIOR not in set(t["game_id"]) and len(t)

    def test_end_game_evidence_is_read_with_any_usecols(self, tmp_path):
        src = write_source(tmp_path / pv.SOURCE_FILE, pbp())
        df, obs = pv.read_source(src, usecols=["game_id", "season", "week", "game_date"])
        assert obs["completion"][PRIOR] == {"end_game": True, "home_total": 24, "away_total": 17}
        assert len(obs["completion"]) == 11
