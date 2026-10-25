"""Real-input path of the Player Props rehearsal (player_props/rehearsal_real.py).

Offline: download() runs with injected fetchers that return a synthetic league
as nflverse-shaped bytes (sockets blocked), and the full rehearsal then runs
through the real-input code path. No real nflverse data is used here.
"""
import json
import socket
from pathlib import Path

import pandas as pd
import pytest

from player_props import rehearsal as rh
from player_props import rehearsal_fixtures as fixtures
from player_props import rehearsal_real as real

ROOT = Path(__file__).resolve().parents[1]
SEASONS = [2022, 2023, 2024, 2025, 2026]


@pytest.fixture
def blocked(monkeypatch):
    def refuse(*a, **k):
        raise OSError("network blocked in this test")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def league():
    played = {s: fixtures.WEEKS for s in SEASONS[:-1]}
    played[2026] = 2
    return fixtures.league(SEASONS, played, 7)


def fake_fetchers(plays, schedule):
    games_csv = schedule.assign(stadium="Synthetic Field").to_csv(index=False).encode("utf-8")
    calls = []

    def fetch_pbp(season):
        calls.append(season)
        return fixtures.season_download(plays[season])

    def fetch_schedule(url):
        assert url == real.SCHEDULE_URL
        calls.append("schedule")
        return games_csv
    return fetch_pbp, fetch_schedule, calls


@pytest.fixture(scope="module")
def downloaded(tmp_path_factory):
    ws = tmp_path_factory.mktemp("real") / "ws"
    plays, schedule = league()
    fetch_pbp, fetch_schedule, calls = fake_fetchers(plays, schedule)
    dest = real.download(ws, SEASONS, fetch_pbp=fetch_pbp, fetch_schedule=fetch_schedule)
    players = ws / "players.csv"
    fixtures.players().to_csv(players, index=False)
    return ws, dest, players, calls


class TestDownload:
    def test_writes_every_file_with_its_record(self, downloaded):
        ws, dest, _, calls = downloaded
        assert calls == ["schedule"] + SEASONS
        assert dest.parent == Path(ws).resolve() / "downloads"
        doc = json.loads((dest / real.DOWNLOADS_NAME).read_text())
        assert doc["kind"] == real.DOWNLOADS_KIND and doc["seasons"] == SEASONS
        assert set(doc["files"]) == {real.SCHEDULE_NAME} | {real.pbp_name(s) for s in SEASONS}
        for name, entry in doc["files"].items():
            data = (dest / name).read_bytes()
            assert entry["bytes"] == len(data) and entry["sha256"] == real._sha256(data)
        assert doc["files"]["play_by_play_2024.csv.gz"]["url"].endswith("/pbp/play_by_play_2024.csv.gz")

    @pytest.mark.parametrize("where", [lambda: ROOT / "data_files" / "dl", lambda: ROOT])
    def test_refuses_the_repository(self, where, blocked):
        with pytest.raises(rh.RehearsalError, match="overlaps protected"):
            real.download(where(), [2025], fetch_pbp=lambda s: b"", fetch_schedule=lambda u: b"")

    def test_refuses_gaps(self, tmp_path):
        with pytest.raises(rh.RehearsalError, match="consecutive"):
            real.download(tmp_path / "ws", [2024, 2026], fetch_pbp=lambda s: b"", fetch_schedule=lambda u: b"")

    def test_failed_download_leaves_no_record(self, tmp_path):
        def fail(season):
            raise OSError("404")
        with pytest.raises(OSError):
            real.download(tmp_path / "ws", [2025], fetch_pbp=fail, fetch_schedule=lambda u: b"x")
        assert not list((tmp_path / "ws").rglob(real.DOWNLOADS_NAME))


def tree(d):
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(Path(d).rglob("*")) if p.is_file()}


@pytest.fixture
def frozen_clock(monkeypatch):
    """Every download attempt sees the same UTC second."""
    from datetime import datetime, timezone

    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 25, 20, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(real, "datetime", Frozen)


class TestAttemptDirectories:
    """Download attempts never share, reuse or overwrite a directory, even in
    the same second."""

    def fetchers(self):
        plays, schedule = league()
        return fake_fetchers(plays, schedule)[:2]

    def test_two_successes_in_one_second(self, tmp_path, frozen_clock, blocked):
        fetch_pbp, fetch_schedule = self.fetchers()
        players = tmp_path / "players.csv"
        fixtures.players().to_csv(players, index=False)
        first = real.download(tmp_path / "ws", [2025, 2026], fetch_pbp=fetch_pbp, fetch_schedule=fetch_schedule)
        second = real.download(tmp_path / "ws", [2025, 2026], fetch_pbp=fetch_pbp, fetch_schedule=fetch_schedule)
        assert first != second and first.name.startswith("20261025T200000Z-") \
            and second.name.startswith("20261025T200000Z-")
        for d in (first, second):                       # each independently hash-valid
            inputs = real.RealInputs(d, players)
            assert inputs.fetch(2025) and inputs.fetch(2026) and len(inputs.schedule())

    def test_failed_attempt_leaves_the_previous_one_untouched(self, tmp_path, frozen_clock, blocked):
        fetch_pbp, fetch_schedule = self.fetchers()
        first = real.download(tmp_path / "ws", [2025, 2026], fetch_pbp=fetch_pbp, fetch_schedule=fetch_schedule)
        before = tree(first)

        def schedule_changed(url):
            return b"different schedule bytes"

        def fail_on_2026(season):
            if season == 2026:
                raise OSError("connection reset")
            return fetch_pbp(season)
        with pytest.raises(OSError, match="connection reset"):
            real.download(tmp_path / "ws", [2025, 2026], fetch_pbp=fail_on_2026, fetch_schedule=schedule_changed)
        assert tree(first) == before                    # byte-identical, record included
        failed = [d for d in first.parent.iterdir() if d != first]
        assert len(failed) == 1
        assert (failed[0] / real.SCHEDULE_NAME).read_bytes() == b"different schedule bytes"

    def test_failed_attempt_has_no_completion_record(self, tmp_path, frozen_clock, blocked):
        fetch_pbp, fetch_schedule = self.fetchers()

        def fail(season):
            raise OSError("404")
        with pytest.raises(OSError):
            real.download(tmp_path / "ws", [2025], fetch_pbp=fail, fetch_schedule=fetch_schedule)
        (attempt,) = list((tmp_path / "ws" / "downloads").iterdir())
        assert (attempt / real.SCHEDULE_NAME).exists() and not (attempt / real.DOWNLOADS_NAME).exists()
        assert not list(attempt.glob("*.tmp"))
        with pytest.raises(rh.RehearsalError, match="no readable"):
            real.RealInputs(attempt, tmp_path / "unused.csv")

    def test_existing_destination_is_never_overwritten(self, tmp_path, frozen_clock, blocked, monkeypatch):
        fetch_pbp, fetch_schedule = self.fetchers()
        taken = tmp_path / "ws" / "downloads" / "20261025T200000Z-aaaa0000"
        taken.mkdir(parents=True)
        (taken / real.DOWNLOADS_NAME).write_text("earlier record")
        (taken / real.SCHEDULE_NAME).write_bytes(b"earlier schedule")
        before = tree(taken)
        tokens = iter(["aaaa0000", "bbbb1111"])
        monkeypatch.setattr(real, "_token", lambda: next(tokens, "cccc2222"))
        dest = real.download(tmp_path / "ws", [2025], fetch_pbp=fetch_pbp, fetch_schedule=fetch_schedule)
        assert dest.name == "20261025T200000Z-bbbb1111"
        assert tree(taken) == before

    def test_gives_up_rather_than_reuse(self, tmp_path, frozen_clock, blocked, monkeypatch):
        fetch_pbp, fetch_schedule = self.fetchers()
        taken = tmp_path / "ws" / "downloads" / "20261025T200000Z-aaaa0000"
        taken.mkdir(parents=True)
        monkeypatch.setattr(real, "_token", lambda: "aaaa0000")
        with pytest.raises(rh.RehearsalError, match="no unused download directory"):
            real.download(tmp_path / "ws", [2025], fetch_pbp=fetch_pbp, fetch_schedule=fetch_schedule)
        assert list(taken.iterdir()) == []

    def test_exclusive_safe_dir_keeps_path_checks(self, tmp_path):
        ws = rh.check_workspace(tmp_path / "ws")
        ws.mkdir()
        made = rh.safe_dir(ws / "x", ws, exclusive=True)
        with pytest.raises(FileExistsError):
            rh.safe_dir(made, ws, exclusive=True)
        with pytest.raises(rh.RehearsalError, match="outside the workspace"):
            rh.safe_dir(tmp_path / "elsewhere", ws, exclusive=True)


class TestRealInputs:
    def test_reads_back_verified(self, downloaded):
        _, dest, players, _ = downloaded
        inputs = real.RealInputs(dest, players)
        assert inputs.seasons == SEASONS
        assert list(inputs.schedule().columns) == real.GAME_COLUMNS
        assert inputs.fetch(2024) == (dest / real.pbp_name(2024)).read_bytes()
        sched = inputs.prediction_schedule(2026)
        assert list(sched.columns) == ["week", "date", "home_team", "away_team", "venue", "status"]
        assert set(sched["status"]) == {"REG"} and set(sched["venue"]) == {"Synthetic Field"}
        assert inputs.describe()["players"]["sha256"] == real._sha256(players.read_bytes())

    def test_tampered_download_is_refused(self, downloaded, tmp_path):
        _, dest, players, _ = downloaded
        copy = tmp_path / "dl"
        copy.mkdir()
        for f in dest.iterdir():
            copy.joinpath(f.name).write_bytes(f.read_bytes())
        p = copy / real.pbp_name(2025)
        p.write_bytes(p.read_bytes() + b"x")
        inputs = real.RealInputs(copy, players)
        with pytest.raises(rh.RehearsalError, match="doesn't match its recorded download hash"):
            inputs.fetch(2025)

    def test_season_not_downloaded(self, downloaded):
        _, dest, players, _ = downloaded
        with pytest.raises(OSError, match="was not downloaded"):
            real.RealInputs(dest, players).fetch(2019)

    def test_players_columns_required(self, downloaded, tmp_path):
        _, dest, _, _ = downloaded
        bad = tmp_path / "players.csv"
        pd.DataFrame({"gsis_id": ["x"]}).to_csv(bad, index=False)
        with pytest.raises(rh.RehearsalError, match="lacks column"):
            real.RealInputs(dest, bad)

    def test_not_a_downloads_dir(self, tmp_path, downloaded):
        with pytest.raises(rh.RehearsalError, match="no readable"):
            real.RealInputs(tmp_path, downloaded[2])


class TestPlayerIdentity:
    """The verifier's name -> ID resolution (outputs carry names, not IDs)."""

    def expect(self, ids, team_ids):
        e = object.__new__(rh._Expectations)
        e.ids, e.team_ids = ids, team_ids
        return e

    def test_unique_name(self):
        assert self.expect({"A B": ["1"]}, {}).player_id({"display_name": "A B", "team": "KC"}) == ("1", None)

    def test_namesake_resolved_by_team(self):
        e = self.expect({"A B": ["old", "1"]}, {"KC": {"1"}, "BUF": {"old"}})
        assert e.player_id({"display_name": "A B", "team": "KC"}) == ("1", None)

    def test_namesakes_on_the_same_team_are_refused(self):
        e = self.expect({"A B": ["1", "2"]}, {"KC": {"1", "2"}})
        pid, problem = e.player_id({"display_name": "A B", "team": "KC"})
        assert pid is None and "ambiguous" in problem and "2 of them have stats for KC" in problem

    def test_no_namesake_on_the_team_is_refused(self):
        pid, problem = self.expect({"A B": ["1", "2"]}, {"KC": set()}).player_id(
            {"display_name": "A B", "team": "KC"})
        assert pid is None and "0 of them" in problem

    def test_unknown(self):
        assert self.expect({}, {}).player_id({"display_name": "Nobody", "team": "KC"})[1] == "unknown player 'Nobody'"


@pytest.fixture(scope="module")
def result(downloaded):
    ws, dest, players, _ = downloaded
    summary = rh.run_rehearsal(rh.RehearsalConfig(workspace=ws, real_inputs=dest, players_file=players))
    return ws, summary


class TestRehearsalWithRealInputs:
    def test_every_stage_succeeds_offline(self, result):
        _, s = result
        assert s["status"] == "succeeded", s["stages"]
        assert all(x["status"] == "ok" for x in s["stages"])

    def test_inputs_and_limitations_are_recorded(self, result, downloaded):
        _, s = result
        _, dest, players, _ = downloaded
        assert s["limitations"][0] == rh.REAL_LIMITATION
        doc = json.loads((dest / real.DOWNLOADS_NAME).read_text())
        assert {n: e["sha256"] for n, e in s["inputs"]["files"].items()} == \
            {n: e["sha256"] for n, e in doc["files"].items()}
        assert s["inputs"]["players"]["sha256"] == real._sha256(players.read_bytes())
        # The run reads the downloaded bytes: the training source records their hashes.
        run_dir = Path(result[0]).resolve() / "runs" / s["run_id"]
        manifest = json.loads((run_dir / "train" / "source" / "player_props_source_manifest.json").read_text())
        doc = json.loads((dest / real.DOWNLOADS_NAME).read_text())
        for season in ("2022", "2025"):
            assert manifest["downloads"][season]["sha256"] == doc["files"][real.pbp_name(int(season))]["sha256"]

    def test_missing_season_is_refused(self, downloaded, tmp_path):
        _, dest, players, _ = downloaded
        s = rh.run_rehearsal(rh.RehearsalConfig(workspace=tmp_path / "ws", real_inputs=dest, players_file=players,
                                                train_seasons=[2020, 2021, 2022]))
        err = next(x for x in s["stages"] if x["stage"] == "workspace")["error"]["message"]
        assert s["status"] == "failed" and "lack season(s) [2020, 2021]" in err

    def test_players_file_required(self, downloaded, tmp_path):
        _, dest, _, _ = downloaded
        s = rh.run_rehearsal(rh.RehearsalConfig(workspace=tmp_path / "ws", real_inputs=dest))
        assert s["status"] == "failed"
        assert "explicit players file" in next(x for x in s["stages"] if x["stage"] == "workspace")["error"]["message"]

    def test_shared_display_name_is_resolved_by_team(self, downloaded, tmp_path):
        # Real players.csv has retired namesakes. One listed AFTER the real player
        # (so a name -> last ID lookup picks it) must not break verification.
        _, dest, players, _ = downloaded
        p = pd.read_csv(players)
        namesake = p.iloc[[0]].assign(gsis_id="00-0000001", position="DE")
        with_namesake = tmp_path / "players.csv"
        pd.concat([p, namesake]).to_csv(with_namesake, index=False)
        s = rh.run_rehearsal(rh.RehearsalConfig(workspace=tmp_path / "ws", real_inputs=dest,
                                                players_file=with_namesake))
        assert s["status"] == "succeeded", s["stages"]

    def test_synthetic_default_unchanged(self, tmp_path):
        s = rh.run_rehearsal(rh.RehearsalConfig(workspace=tmp_path / "ws"))
        assert s["status"] == "succeeded" and s["limitations"][0] == rh.SYNTHETIC_LIMITATION
        assert s["inputs"] == "synthetic"
