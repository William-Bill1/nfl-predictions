"""Player-prop Stage 1 feature generation (player_props/aggregators.py): schedule-
derived is_home, prior-only rolling stats, days_rest, duplicate resolution,
validation and atomic writes. Synthetic play-by-play and schedules only; no
network, and data_files/ is never written."""
import json
import socket

import numpy as np
import pandas as pd
import pytest

from player_props import aggregators as ag
from test_team_features import ROOT


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


GAMES = {  # game_id: (season, week, date, home, away)
    "2025_01_BUF_KC": (2025, 1, "2025-09-07", "KC", "BUF"),
    "2025_02_KC_LV": (2025, 2, "2025-09-11", "LV", "KC"),        # Thursday: 4 days later
    "2025_04_KC_DEN": (2025, 4, "2025-09-25", "DEN", "KC"),      # after a bye: 14
    "2025_08_KC_NE": (2025, 8, "2025-10-23", "KC", "NE"),        # missed games: 28 -> 21
    "2025_09_KC_NYJ": (2025, 9, "2025-10-29", "NYJ", "KC"),      # 6 days
    "2026_01_KC_SF": (2026, 1, "2026-09-10", "KC", "SF"),        # next season: capped 21
    "2026_02_LAR_SF": (2026, 2, "2026-09-20", "SF", "LAR"),      # traded to LAR: 10
}


def schedule(games=GAMES):
    return pd.DataFrame([dict(game_id=g, season=s, week=w, gameday=d, home_team=h, away_team=a)
                         for g, (s, w, d, h, a) in games.items()])


def play(game, team, *, passer=None, rusher=None, receiver=None, name=None, yards=0, td=0,
         complete=1, attempt=1, intercept=0, rush=0, games=GAMES):
    season, week, date, home, away = games[game]
    row = dict(game_id=game, season=season, week=week, game_date=date, posteam=team,
               defteam=away if team == home else home,
               passer_player_id=passer, passer_player_name=name if passer else None,
               rusher_player_id=rusher, rusher_player_name=name if rusher else None,
               receiver_player_id=receiver, receiver_player_name=name if receiver else None,
               passing_yards=yards if passer else 0, rushing_yards=yards if rusher else 0,
               receiving_yards=yards if receiver else 0,
               pass_touchdown=td if (passer or receiver) else 0, rush_touchdown=td if rusher else 0,
               complete_pass=complete if (passer or receiver) else 0,
               pass_attempt=attempt if (passer or receiver) else 0,
               interception=intercept, rush_attempt=rush if rusher else 0)
    return row


def qb_season(yards=(100, 200, 300, 250, 280, 310, 220)):
    """One QB (P1) through every game; KC until the 2026 trade to LAR. Each game
    also has a running back (R1) and a receiver (W1) so all tables build."""
    plays = []
    for (game, (_, _, _, home, away)), y in zip(GAMES.items(), yards):
        team = "LAR" if game == "2026_02_LAR_SF" else "KC"
        plays.append(play(game, team, passer="P1", name="P.Mahomes", yards=y))
        plays.append(play(game, team, rusher="R1", name="R.Back", yards=y // 10, rush=1))
        plays.append(play(game, team, receiver="W1", name="W.Out", yards=y // 2))
    return pd.DataFrame(plays)


def build(pbp=None, sched=None):
    return ag.build_player_stats(qb_season() if pbp is None else pbp,
                                 schedule() if sched is None else sched)


class TestIsHome:
    def test_known_home_and_away_games(self):
        tables, _ = build()
        p = tables["passing"].set_index("game_id")
        assert p.loc["2025_01_BUF_KC", "is_home"] == 1          # KC home
        assert p.loc["2025_02_KC_LV", "is_home"] == 0           # KC at LV
        assert p.loc["2025_08_KC_NE", "is_home"] == 1
        assert p.loc["2026_02_LAR_SF", "is_home"] == 0          # LAR at SF after the trade
        assert p["is_home"].dtype.kind == "i"

    def test_missing_schedule_game_fails(self):
        sched = schedule().query("game_id != '2025_04_KC_DEN'")
        with pytest.raises(ag.FeatureValidationError, match="no schedule game"):
            build(sched=sched)

    def test_duplicate_schedule_rows_fail(self):
        sched = pd.concat([schedule(), schedule().iloc[[0]]])
        with pytest.raises(ag.FeatureValidationError, match="more than one row"):
            build(sched=sched)

    @pytest.mark.parametrize("change,message", [
        (dict(home_team="NYG", away_team="BUF"), "not exactly one side"),     # team absent
        (dict(home_team="KC", away_team="KC"), "not exactly one side"),       # both sides
        (dict(home_team="KC", away_team="NYG"), "not the other side"),        # wrong opponent
        (dict(week=3), "season/week"),
        (dict(home_team=None), "missing game_id, home_team or away_team"),
    ])
    def test_conflicting_schedule_evidence_fails(self, change, message):
        sched = schedule()
        for k, v in change.items():
            sched.loc[sched.game_id == "2025_01_BUF_KC", k] = v
        with pytest.raises(ag.FeatureValidationError, match=message):
            build(sched=sched)

    def test_missing_schedule_file_is_a_failure_not_a_default(self, tmp_path):
        with pytest.raises(ag.FeatureValidationError, match="could not be read"):
            ag.load_matchup_schedule(tmp_path)

    def test_no_random_values(self):
        a, _ = build()
        b, _ = build()
        pd.testing.assert_frame_equal(a["passing"], b["passing"])
        assert "random" not in (ROOT / "player_props" / "aggregators.py").read_text(encoding="utf-8")


class TestRollingAverages:
    def test_first_game_has_no_history_and_target_game_is_excluded(self):
        p = build()[0]["passing"].sort_values("game_date").reset_index(drop=True)
        assert p.loc[0, "passing_yards_L3"] != p.loc[0, "passing_yards_L3"]   # NaN
        assert p.loc[1, "passing_yards_L3"] == 100                           # only game 1
        assert p.loc[2, "passing_yards_L3"] == pytest.approx(150)            # ewm(100, 200), span 3

    def test_target_result_never_enters_its_own_features(self):
        base = build()[0]["passing"].sort_values("game_date").reset_index(drop=True)
        pbp = qb_season(yards=(100, 200, 9999, 250, 280, 310, 220))
        changed = build(pbp)[0]["passing"].sort_values("game_date").reset_index(drop=True)
        rolling = [c for c in base.columns if "_L" in c]
        pd.testing.assert_frame_equal(base.loc[[2], rolling], changed.loc[[2], rolling])
        assert changed.loc[3, "passing_yards_L3"] != base.loc[3, "passing_yards_L3"]

    def test_changing_later_games_never_changes_earlier_rows(self):
        base = build()[0]
        later = qb_season(yards=(100, 200, 300, 250, 1, 2, 3))
        changed = build(later)[0]
        for stat_type in base:
            b = base[stat_type].sort_values(["player_id", "game_date"]).reset_index(drop=True)
            c = changed[stat_type].sort_values(["player_id", "game_date"]).reset_index(drop=True)
            early = b["game_date"] <= "2025-10-23"     # games 1-4 unchanged, game 5+ changed
            pd.testing.assert_frame_equal(b[early], c[early])

    def test_order_is_calendar_date_not_game_id(self):
        games = {"A_LATE": (2025, 2, "2025-09-14", "KC", "BUF"),
                 "Z_EARLY": (2025, 1, "2025-09-07", "KC", "LV")}
        pbp = pd.DataFrame([play(g, "KC", **kw, games=games) for g in games for kw in (
            dict(passer="P1", name="Q", yards=100 if g == "Z_EARLY" else 300),
            dict(rusher="R1", name="R", yards=5, rush=1), dict(receiver="W1", name="W", yards=9))])
        p = build(pbp, schedule(games))[0]["passing"].set_index("game_id")
        assert np.isnan(p.loc["Z_EARLY", "passing_yards_L3"]) and p.loc["A_LATE", "passing_yards_L3"] == 100
        assert p.loc["Z_EARLY", "days_rest"] == 7 and p.loc["A_LATE", "days_rest"] == 7

    def test_history_carries_across_seasons_and_teams(self):
        p = build()[0]["passing"].set_index("game_id")
        assert p.loc["2026_01_KC_SF", "passing_yards_L3"] == p.loc["2026_01_KC_SF", "passing_yards_L3"]
        assert not np.isnan(p.loc["2026_02_LAR_SF", "passing_yards_L3"])     # after the trade


class TestDaysRest:
    def test_cases(self):
        p = build()[0]["passing"].set_index("game_id")["days_rest"]
        assert p.to_dict() == {
            "2025_01_BUF_KC": 7,     # first-ever appearance
            "2025_02_KC_LV": 4,      # short week
            "2025_04_KC_DEN": 14,    # bye
            "2025_08_KC_NE": 21,     # missed appearances: 28 days, capped
            "2025_09_KC_NYJ": 6,
            "2026_01_KC_SF": 21,     # season boundary, capped
            "2026_02_LAR_SF": 10,    # traded: history carries across teams (a reset would give 7)
        }
        assert p.dtype.kind == "i"

    def test_missing_appearance_in_this_table_lengthens_the_gap(self):
        pbp = qb_season()
        pbp = pbp[~((pbp.receiver_player_id == "W1") & (pbp.game_id == "2025_02_KC_LV"))]
        w = build(pbp)[0]["receiving"].set_index("game_id")["days_rest"]
        assert w["2025_04_KC_DEN"] == 18      # Sep 7 -> Sep 25: no row for the Sep 11 game


class TestDuplicates:
    def test_name_variants_in_one_game_become_one_summed_row(self):
        pbp = qb_season()
        extra = play("2025_01_BUF_KC", "KC", passer="P1", name="Pat Mahomes", yards=50, td=1)
        pbp = pd.concat([pbp, pd.DataFrame([extra, extra])], ignore_index=True)
        tables, reports = build(pbp)
        p = tables["passing"]
        g = p[p.game_id == "2025_01_BUF_KC"]
        assert len(g) == 1 and not p.duplicated(["player_id", "game_id"]).any()
        assert g.iloc[0]["passing_yards"] == 200 and g.iloc[0]["pass_tds"] == 2   # all plays summed
        assert g.iloc[0]["player_name"] == "Pat Mahomes"      # most frequent spelling (2 vs 1)
        assert reports["passing"]["name_merges"] == 1

    def test_player_on_two_teams_in_one_game_fails(self):
        pbp = qb_season()
        bad = play("2025_01_BUF_KC", "BUF", passer="P1", name="P.Mahomes", yards=10)
        with pytest.raises(ag.FeatureValidationError, match="more than one team"):
            build(pd.concat([pbp, pd.DataFrame([bad])], ignore_index=True))

    def test_two_games_on_one_date_fails(self):
        games = {**GAMES, "2025_01_KC_XX": (2025, 1, "2025-09-07", "XX", "KC")}
        pbp = pd.concat([qb_season(), pd.DataFrame([play("2025_01_KC_XX", "KC", passer="P1",
                                                          name="P.Mahomes", yards=5, games=games)])])
        with pytest.raises(ag.FeatureValidationError, match="two games on one date"):
            build(pbp, schedule(games))

    def test_plays_missing_labels_are_excluded_and_counted(self):
        pbp = qb_season()
        bad = play("2025_01_BUF_KC", "KC", passer="P1", name="P.Mahomes", yards=999)
        bad["posteam"] = None
        tables, reports = build(pd.concat([pbp, pd.DataFrame([bad])], ignore_index=True))
        assert reports["passing"]["excluded_plays"] == 1
        assert tables["passing"].set_index("game_id").loc["2025_01_BUF_KC", "passing_yards"] == 100


class TestValidationAndWriting:
    def test_contract_columns_and_committed_model_features(self):
        tables, _ = build()
        for stat_type, df in tables.items():
            assert list(df.columns) == ag.output_columns(stat_type)
            assert ag.validate_player_stats(df, stat_type) == []
        prefix = {"passing": "passing", "rushing": "rushing", "receiving": "receiving",
                  "receptions": "receiving"}
        for model in (ROOT / "player_props" / "models").glob("*.json"):
            if model.name.startswith("model_metrics"):
                continue
            names = json.loads(model.read_text(encoding="utf-8"))["learner"]["feature_names"]
            stat_type = prefix[model.stem.split("_")[0]]
            assert set(names) <= set(ag.output_columns(stat_type)), model.name

    def test_ratios_are_nan_not_inf(self):
        pbp = qb_season()
        pbp.loc[pbp.passer_player_id == "P1", "pass_attempt"] = 0
        p = build(pbp)[0]["passing"]
        assert p["completion_pct"].isna().all() and not np.isinf(p.select_dtypes("number")).any(axis=None)

    def test_save_and_reload_keep_features_and_values(self, tmp_path):
        tables, _ = build()
        written = ag.write_player_stats(tables, tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(ag.STATS_FILES.values())
        for stat_type, path in written.items():
            back = pd.read_csv(path)
            assert list(back.columns) == ag.output_columns(stat_type)
            assert ag.validate_player_stats(back, stat_type) == []
            pd.testing.assert_frame_equal(back, tables[stat_type], check_dtype=False)

    @pytest.mark.parametrize("corrupt,message", [
        (lambda t: t.__setitem__("passing", pd.concat([t["passing"], t["passing"].iloc[[0]]])),
         "duplicate player/game"),
        (lambda t: t["rushing"].__setitem__("is_home", 2), "is_home must be 0 or 1"),
        (lambda t: t["receiving"].__setitem__("days_rest", 0), "days_rest must be"),
        (lambda t: t["passing"].__setitem__("passing_yards_L3", np.inf), "infinite"),
        (lambda t: t["passing"].drop(columns="is_home", inplace=True), "columns differ"),
        (lambda t: t["passing"].__setitem__("game_date", "not a date"), "invalid game_date"),
        (lambda t: t["passing"].__setitem__("passing_yards_L3", np.nan), "missing for"),
        (lambda t: t.pop("rushing"), "missing table"),
    ])
    def test_invalid_tables_write_nothing(self, tmp_path, corrupt, message):
        for name in ag.STATS_FILES.values():
            (tmp_path / name).write_text("existing,content\n1,2\n", encoding="utf-8")
        before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        tables, _ = build()
        tables = {k: v.copy() for k, v in tables.items()}
        corrupt(tables)
        with pytest.raises(ag.FeatureValidationError, match=message):
            ag.write_player_stats(tables, tmp_path)
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before

    def test_successful_overwrite_replaces_all_and_leaves_no_temp_or_backup(self, tmp_path):
        for name in ag.STATS_FILES.values():
            (tmp_path / name).write_text("old\n", encoding="utf-8")
        tables, _ = build()
        ag.write_player_stats(tables, tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(ag.STATS_FILES.values())
        for stat_type, name in ag.STATS_FILES.items():
            pd.testing.assert_frame_equal(pd.read_csv(tmp_path / name), tables[stat_type],
                                          check_dtype=False)

    def test_failure_while_writing_temps_leaves_existing_files_and_no_temps(self, tmp_path, monkeypatch):
        for name in ag.STATS_FILES.values():
            (tmp_path / name).write_text("existing\n", encoding="utf-8")
        tables, _ = build()
        calls = {"n": 0}
        real = pd.DataFrame.to_csv

        def flaky(self, *a, **k):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("disk full")
            return real(self, *a, **k)
        monkeypatch.setattr(pd.DataFrame, "to_csv", flaky)
        with pytest.raises(OSError):
            ag.write_player_stats(tables, tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(ag.STATS_FILES.values())
        assert all(p.read_text() == "existing\n" for p in tmp_path.iterdir())


NAMES = list(ag.STATS_FILES.values())       # replacement order: passing, rushing, receiving


def seed(tmp_path, present=NAMES):
    for name in present:
        (tmp_path / name).write_text(f"previous {name}\n", encoding="utf-8")
    return {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def fail_replace(monkeypatch, forward_call=None, restore=False, unlink=False):
    """Make the Nth forward replacement (temp -> destination) raise, and
    optionally every restore (backup -> destination) or rollback unlink too."""
    real_replace, real_unlink = ag.os.replace, ag.Path.unlink
    calls = {"forward": 0}

    def replace(src, dst, *a, **k):
        if str(src).endswith(".tmp"):
            calls["forward"] += 1
            if calls["forward"] == forward_call:
                raise OSError(f"replace {calls['forward']} failed")
        elif str(src).endswith(".bak") and restore:
            raise OSError("restore failed")
        return real_replace(src, dst, *a, **k)

    def path_unlink(self, *a, **k):
        if unlink and self.name in NAMES:
            raise OSError("remove failed")
        return real_unlink(self, *a, **k)
    monkeypatch.setattr(ag.os, "replace", replace)
    monkeypatch.setattr(ag.Path, "unlink", path_unlink)


class TestReplacementRollback:
    @pytest.mark.parametrize("failing", [2, 3])
    def test_failed_replacement_restores_the_previous_set(self, tmp_path, monkeypatch, failing):
        before = seed(tmp_path)
        tables, _ = build()
        fail_replace(monkeypatch, forward_call=failing)
        with pytest.raises(OSError, match=f"replace {failing} failed"):
            ag.write_player_stats(tables, tmp_path)
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before   # no temps/backups

    @pytest.mark.parametrize("failing", [2, 3])
    def test_destination_absent_before_is_removed_again(self, tmp_path, monkeypatch, failing):
        before = seed(tmp_path, present=NAMES[1:])           # passing file did not exist
        tables, _ = build()
        fail_replace(monkeypatch, forward_call=failing)
        with pytest.raises(OSError):
            ag.write_player_stats(tables, tmp_path)
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
        assert not (tmp_path / NAMES[0]).exists()

    def test_all_absent_leaves_an_empty_directory(self, tmp_path, monkeypatch):
        tables, _ = build()
        fail_replace(monkeypatch, forward_call=3)
        with pytest.raises(OSError):
            ag.write_player_stats(tables, tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_rollback_failure_keeps_the_backups_and_says_where(self, tmp_path, monkeypatch):
        before = seed(tmp_path)
        tables, _ = build()
        fail_replace(monkeypatch, forward_call=2, restore=True)
        with pytest.raises(ag.PersistenceRollbackError) as err:
            ag.write_player_stats(tables, tmp_path)
        e = err.value
        passing = tmp_path / NAMES[0]
        assert e.unrestored == [passing]
        assert isinstance(e.__cause__, OSError) and "replace 2 failed" in str(e)
        # The only copy of the old passing file is retained, intact, and named in the error.
        bak = e.recovery_files[passing]
        assert bak.exists() and bak.read_bytes() == before[NAMES[0]] and str(bak) in str(e)
        # Untouched destinations still hold their previous content; no temp files remain.
        assert (tmp_path / NAMES[1]).read_bytes() == before[NAMES[1]]
        assert (tmp_path / NAMES[2]).read_bytes() == before[NAMES[2]]
        assert not list(tmp_path.glob("*.tmp"))
        for b in e.recovery_files.values():
            assert b.exists()

    def test_rollback_failure_on_an_absent_destination_is_reported(self, tmp_path, monkeypatch):
        seed(tmp_path, present=NAMES[1:])
        tables, _ = build()
        fail_replace(monkeypatch, forward_call=2, unlink=True)
        with pytest.raises(ag.PersistenceRollbackError) as err:
            ag.write_player_stats(tables, tmp_path)
        assert err.value.unrestored == [tmp_path / NAMES[0]]
        assert tmp_path / NAMES[0] not in err.value.recovery_files    # it had no previous copy

    def test_backup_copy_failure_replaces_nothing(self, tmp_path, monkeypatch):
        before = seed(tmp_path)
        tables, _ = build()

        def broken_copy(*a, **k):
            raise OSError("copy failed")
        monkeypatch.setattr(ag.shutil, "copy2", broken_copy)
        with pytest.raises(OSError, match="copy failed"):
            ag.write_player_stats(tables, tmp_path)
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_training_entry_point_saves_features_before_models_read_them(tmp_path, monkeypatch):
    from player_props import train_models as tm
    schedule().to_csv(tmp_path / ag.SCHEDULE_FILE, sep="\t", index=False)
    monkeypatch.setattr(tm, "DATA_DIR", tmp_path)
    tables = tm._aggregate_and_roll(qb_season())
    for stat_type, name in ag.STATS_FILES.items():
        saved = pd.read_csv(tmp_path / name)
        assert {"is_home", "days_rest"} <= set(saved.columns)
        assert len(saved) == len(tables[stat_type])
