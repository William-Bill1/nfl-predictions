"""Player-prop Stage 5A: complete-source refresh (player_props/source_refresh.py)
and bundle storage (player_props/bundle_store.py, model_bundle.publisher_lock).
Synthetic data only, downloads injected, no network; data_files/ and
player_props/models/ are never written."""
import gzip
import io
import json
import os
import socket
import tarfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from player_props import bundle_store as bs
from player_props import features as fx
from player_props import model_bundle as mb
from player_props import source_refresh as sr
from prop_bundle_helpers import publish_test_bundle, trained_entry, training_source
from test_player_props_stage2 import league

TODAY = date(2026, 10, 24)
PAIRS = [("KC", "BUF"), ("DEN", "LV")]          # (home, away)
SCORES = [(24, 17), (20, 23)]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def snapshot(directory):
    return {str(p.relative_to(directory)): p.read_bytes()
            for p in sorted(Path(directory).rglob("*")) if p.is_file()}


# ---------------------------------------------------------------------------
# Synthetic seasons and schedule
# ---------------------------------------------------------------------------

def gid(season, week, home, away):
    return f"{season}_{week:02d}_{away}_{home}"


def season_plays(season, weeks=(1, 2), skip=(), unended=(), scores=None):
    """nflverse-style plays for `season`: each team's pass and run, then END GAME."""
    rows = []
    for week in weeks:
        day = (date(season, 9, 7) + timedelta(days=7 * (week - 1))).isoformat()
        for i, (home, away) in enumerate(PAIRS):
            game = gid(season, week, home, away)
            if game in skip:
                continue
            play = 0
            for team, opp in ((home, away), (away, home)):
                base = dict(game_id=game, season=season, week=week, game_date=day, posteam=team,
                            defteam=opp, desc="play", total_home_score=None, total_away_score=None,
                            passer_player_id=None, passer_player_name=None, rusher_player_id=None,
                            rusher_player_name=None, receiver_player_id=None,
                            receiver_player_name=None, passing_yards=0, rushing_yards=0,
                            receiving_yards=0, pass_touchdown=0, rush_touchdown=0, complete_pass=0,
                            pass_attempt=0, interception=0, rush_attempt=0, season_type="REG")
                play += 1
                rows.append({**base, "play_id": play, "passer_player_id": f"{team}-QB",
                             "passer_player_name": f"{team}.QB", "receiver_player_id": f"{team}-WR",
                             "receiver_player_name": f"{team}.WR", "passing_yards": 20,
                             "receiving_yards": 20, "complete_pass": 1, "pass_attempt": 1})
                play += 1
                rows.append({**base, "play_id": play, "rusher_player_id": f"{team}-RB",
                             "rusher_player_name": f"{team}.RB", "rushing_yards": 5, "rush_attempt": 1})
            if game not in unended:
                h, a = (scores or {}).get(game, SCORES[i])
                rows.append({**base, "play_id": play + 1, "posteam": None, "defteam": None,
                             "desc": "END GAME", "total_home_score": h, "total_away_score": a})
    return pd.DataFrame(rows)


def to_bytes(df, gz=True):
    raw = df.to_csv(index=False).encode()
    return gzip.compress(raw) if gz else raw


def schedule(seasons, weeks=(1, 2), future_week=None):
    rows = []
    for season in seasons:
        for week in list(weeks) + ([future_week] if future_week else []):
            day = (date(season, 9, 7) + timedelta(days=7 * (week - 1))).isoformat()
            for i, (home, away) in enumerate(PAIRS):
                done = week != future_week
                h, a = SCORES[i]
                rows.append(dict(game_id=gid(season, week, home, away), season=season, week=week,
                                 game_type="REG", gameday=day, home_team=home, away_team=away,
                                 home_score=float(h) if done else None,
                                 away_score=float(a) if done else None,
                                 result=float(h - a) if done else None,
                                 total=float(h + a) if done else None))
    return pd.DataFrame(rows)


@pytest.fixture
def setup(tmp_path):
    sched = tmp_path / "nfl_games_historical.csv"
    schedule([2024, 2025, 2026]).to_csv(sched, sep="\t", index=False)
    out = tmp_path / "prop_source"
    downloads = {s: to_bytes(season_plays(s)) for s in (2024, 2025, 2026)}
    return out, sched, downloads


def run(out, sched, downloads, seasons=(2024, 2025, 2026)):
    return sr.refresh_source(out, sched, fetch=lambda s: downloads[s], seasons=list(seasons), today=TODAY)


# ---------------------------------------------------------------------------
# Source refresh
# ---------------------------------------------------------------------------

class TestRequiredSeasons:
    def test_every_intervening_season_is_required(self):
        assert sr.required_seasons(2020, 2026) == [2020, 2021, 2022, 2023, 2024, 2025, 2026]
        assert sr.required_seasons(today=date(2026, 10, 24))[-1] == 2026
        assert sr.required_seasons(today=date(2026, 3, 1))[-1] == 2025

    def test_gaps_are_refused(self, setup):
        out, sched, downloads = setup
        with pytest.raises(sr.SourceRefreshError, match="gaps"):
            run(out, sched, downloads, seasons=(2024, 2026))
        with pytest.raises(sr.SourceRefreshError, match="no seasons"):
            run(out, sched, downloads, seasons=())


class TestValidRefresh:
    def test_multi_season_refresh_installs_verified_content(self, setup):
        out, sched, downloads = setup
        m = run(out, sched, downloads)
        assert m["required_seasons"] == [2024, 2025, 2026]
        assert sorted(m["source"]["coverage"]["seasons"]) == ["2024", "2025", "2026"]
        assert m["source"]["coverage"]["games"] == 12
        for s in ("2024", "2025", "2026"):
            assert m["downloads"][s]["sha256"] == sr._sha256(downloads[int(s)])
            assert m["coverage"][s]["completed_and_present"] == 4
        data = (out / sr.SOURCE_NAME).read_bytes()
        assert m["source"]["sha256"] == sr._sha256(data)
        assert sr.verify_installed(out)["source"]["sha256"] == sr._sha256(data)
        assert m["schedule"]["sha256"] == sr._sha256(sched.read_bytes())
        # tab-separated gzip that the Stage 2 reader accepts, with END GAME evidence
        from player_props.provenance import read_source
        _, obs = read_source(out / sr.SOURCE_NAME)
        assert obs["sha256"] == m["source"]["sha256"] and len(obs["completion"]) == 12
        assert sorted(p.name for p in out.iterdir()) == sorted([sr.SOURCE_NAME, sr.MANIFEST_NAME])

    def test_same_content_hashes_the_same(self, setup, tmp_path):
        out, sched, downloads = setup
        a = run(out, sched, downloads)["source"]["sha256"]
        b = run(tmp_path / "other", sched, downloads)["source"]["sha256"]
        assert a == b

    def test_future_and_in_progress_games_are_not_required(self, setup, tmp_path):
        out, _, downloads = setup
        sched = tmp_path / "sched2.csv"
        schedule([2024, 2025, 2026], future_week=10).to_csv(sched, sep="\t", index=False)
        # 2026 week 10 is unplayed and absent from the download: fine.
        m = run(out, sched, downloads)
        assert m["coverage"]["2026"]["not_yet_completed"] == 2
        # A game in the download whose schedule score isn't recorded yet is kept and listed.
        s = schedule([2024, 2025, 2026])
        s.loc[s["game_id"] == gid(2026, 2, "KC", "BUF"), ["home_score", "away_score", "result", "total"]] = None
        s.to_csv(sched, sep="\t", index=False)
        m = run(tmp_path / "o2", sched, downloads)
        assert m["coverage"]["2026"]["present_without_completion_evidence"] == [gid(2026, 2, "KC", "BUF")]

    def test_lock_is_released(self, setup):
        out, sched, downloads = setup
        run(out, sched, downloads)
        assert not list(out.glob(".*.lock"))
        run(out, sched, downloads)                      # a second refresh isn't blocked


@pytest.fixture
def installed(setup):
    out, sched, downloads = setup
    run(out, sched, downloads)
    return out, sched, downloads, snapshot(out)


class TestFailurePreservesThePreviousSource:
    def _fails(self, installed, downloads, match, **kw):
        out, sched, _, before = installed
        with pytest.raises(sr.SourceRefreshError, match=match) as exc:
            run(out, sched, downloads, **kw)
        assert snapshot(out) == before                     # previous source + manifest untouched
        assert not [p for p in out.iterdir() if p.name.startswith(".")]   # no staging, lock or backup
        return exc.value.problems

    def test_missing_intervening_season(self, installed):
        downloads = dict(installed[2])

        def fetch(season):
            if season == 2025:
                raise OSError("404 Not Found")
            return downloads[season]
        out, sched, _, before = installed
        with pytest.raises(sr.SourceRefreshError, match="season 2025: download failed"):
            sr.refresh_source(out, sched, fetch=fetch, seasons=[2024, 2025, 2026], today=TODAY)
        assert snapshot(out) == before

    def test_wrong_season_response(self, installed):
        d = dict(installed[2])
        d[2025] = d[2024]
        self._fails(installed, d, "labelled season")

    def test_game_ids_from_another_season(self, installed):
        d = dict(installed[2])
        df = season_plays(2025)
        df["game_id"] = df["game_id"].str.replace("2025_", "2024_", n=1)
        d[2025] = to_bytes(df)
        self._fails(installed, d, "another season")

    @pytest.mark.parametrize("make,match", [
        (lambda b: b"", "empty"),
        (lambda b: b[: len(b) // 2], "truncated or corrupt"),
        (lambda b: gzip.compress(b"play_id,game_id\n"), "missing column"),
        (lambda b: gzip.compress(season_plays(2025).head(0).to_csv(index=False).encode()), "no plays"),
        (lambda b: b"\x00\x01garbage" * 50, "not readable|missing column"),
    ])
    def test_truncated_or_empty_data(self, installed, make, match):
        d = dict(installed[2])
        d[2025] = make(d[2025])
        self._fails(installed, d, match)

    @pytest.mark.parametrize("gz", [False, True])
    def test_lfs_pointer(self, installed, gz):
        pointer = (b"version https://git-lfs.github.com/spec/v1\n"
                   b"oid sha256:" + b"0" * 64 + b"\nsize 116114629\n")
        d = dict(installed[2])
        d[2025] = gzip.compress(pointer) if gz else pointer
        self._fails(installed, d, "Git LFS pointer")

    def test_duplicate_plays(self, installed):
        df = season_plays(2025)
        d = dict(installed[2])
        d[2025] = to_bytes(pd.concat([df, df.iloc[[0]]]))
        self._fails(installed, d, "duplicated play")

    def test_invalid_dates(self, installed):
        df = season_plays(2025)
        df.loc[0, "game_date"] = "2025-13-40"
        d = dict(installed[2])
        d[2025] = to_bytes(df)
        self._fails(installed, d, "invalid game_date")

    def test_missing_completed_game(self, installed):
        d = dict(installed[2])
        d[2025] = to_bytes(season_plays(2025, skip={gid(2025, 2, "DEN", "LV")}))
        problems = self._fails(installed, d, "missing from the download")
        assert any(gid(2025, 2, "DEN", "LV") in p for p in problems)

    def test_partial_game(self, installed):
        d = dict(installed[2])
        d[2025] = to_bytes(season_plays(2025, unended={gid(2025, 1, "KC", "BUF")}))
        self._fails(installed, d, "without a single END GAME score")

    def test_score_correction_not_yet_in_the_schedule(self, installed):
        d = dict(installed[2])
        d[2025] = to_bytes(season_plays(2025, scores={gid(2025, 1, "KC", "BUF"): (27, 17)}))
        self._fails(installed, d, "differs from the recorded score")

    def test_game_not_in_the_schedule(self, installed):
        df = season_plays(2025)
        extra = season_plays(2025, weeks=(3,))
        d = dict(installed[2])
        d[2025] = to_bytes(pd.concat([df, extra]))
        self._fails(installed, d, "not in the schedule")

    def test_every_problem_is_reported(self, installed):
        d = dict(installed[2])
        d[2024] = b""
        d[2026] = to_bytes(season_plays(2026, skip={gid(2026, 1, "KC", "BUF")}))
        problems = self._fails(installed, d, "season 2024")
        assert any("season 2026" in p for p in problems)

    def test_schedule_without_the_season(self, installed, tmp_path):
        out, _, downloads, before = installed
        sched = tmp_path / "short.csv"
        schedule([2024, 2025]).to_csv(sched, sep="\t", index=False)
        with pytest.raises(sr.SourceRefreshError, match="no games for this season"):
            sr.refresh_source(out, sched, fetch=lambda s: downloads[s], seasons=[2024, 2025, 2026],
                              today=TODAY)
        assert snapshot(out) == before

    def test_failed_install_rolls_back(self, installed, monkeypatch):
        out, sched, downloads, before = installed
        real = os.replace

        def fail_manifest(src, dst):
            if str(dst).endswith(sr.MANIFEST_NAME) and ".staging-" in str(src):
                raise OSError("disk full")
            return real(src, dst)
        monkeypatch.setattr(sr.os, "replace", fail_manifest)
        d = dict(downloads)
        changed = season_plays(2026)
        changed.loc[changed["passing_yards"] > 0, "passing_yards"] = 31   # different, still valid
        d[2026] = to_bytes(changed)
        with pytest.raises(OSError, match="disk full"):
            run(out, sched, d)
        assert snapshot(out) == before          # the already-moved source was rolled back

    def test_concurrent_refresh_is_refused(self, installed):
        out, sched, downloads, before = installed
        from ontario_spreads import SlotLock
        with SlotLock(out, "prop-source-refresh", "other-run", what="prop source refresh"):
            with pytest.raises(sr.SourceRefreshError, match="holds lock file"):
                run(out, sched, downloads)
            assert (out / ".prop-source-refresh.lock").exists()     # not removed by the loser
        assert {k: v for k, v in snapshot(out).items() if not k.endswith(".lock")} == before


class TestSourceCommand:
    def test_out_dir_is_required(self):
        with pytest.raises(SystemExit) as exc:
            sr.main([])
        assert exc.value.code == 2

    def test_verify_installed_rejects_changed_content(self, installed):
        out = installed[0]
        f = out / sr.SOURCE_NAME
        f.write_bytes(f.read_bytes() + b"\x00")
        with pytest.raises(sr.SourceRefreshError, match="doesn't match its manifest hash"):
            sr.verify_installed(out)

    def test_verify_installed_rejects_changes(self, installed):
        out = installed[0]
        (out / sr.SOURCE_NAME).write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        with pytest.raises(sr.SourceRefreshError, match="LFS pointer"):
            sr.verify_installed(out)

    def test_spread_files_are_never_touched(self, setup):
        out, sched, downloads = setup
        run(out, sched, downloads)
        names = {p.name for p in out.iterdir()}
        assert "nfl_play_by_play_historical.csv.gz" not in names and "pbp_metadata.json" not in names


# ---------------------------------------------------------------------------
# Bundle storage
# ---------------------------------------------------------------------------

@pytest.fixture
def bundle(tmp_path):
    data = league(tmp_path)
    models = tmp_path / "models"
    path = publish_test_bundle(models, data)
    return models, path, data


def _archive(tmp_path, entries, index=None, bundle_id="20261024T000000Z-0000abcd"):
    """A hand-built tar: entries = [(name, bytes or TarInfo-tweak)]."""
    path = tmp_path / "hand.tar"
    with tarfile.open(path, "w") as tar:
        if index is not None:
            raw = json.dumps(index).encode()
            info = tarfile.TarInfo(bs.EXPORT_FILE)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
        for name, data, kind in entries:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                info.type = {"symlink": tarfile.SYMTYPE, "dir": tarfile.DIRTYPE,
                             "hardlink": tarfile.LNKTYPE, "fifo": tarfile.FIFOTYPE}[kind]
                info.linkname = "../../etc/passwd" if kind in ("symlink", "hardlink") else ""
                tar.addfile(info)
    return path


class TestExportImport:
    def test_round_trip_gives_identical_raw_scores(self, bundle, tmp_path):
        models, path, _ = bundle
        archive = tmp_path / "b.tar"
        index = bs.export_bundle(models, archive)
        target = tmp_path / "restored"
        r = bs.import_bundle(target, archive)
        assert r == {"bundle_id": path.name, "installed": True, "activated": True}
        assert snapshot(target / mb.BUNDLES_DIR) == snapshot(models / mb.BUNDLES_DIR)
        a = mb.load_bundle(models, fx.PROP_LINES)["models"]["passing_yards_starter"]["model"]
        b = mb.load_bundle(target, fx.PROP_LINES)["models"]["passing_yards_starter"]["model"]
        _, rows = trained_entry("passing_yards", "starter")
        X = fx.model_matrix(rows, "passing_yards")
        assert (mb.score_probabilities(a, X, "passing_yards") ==
                mb.score_probabilities(b, X, "passing_yards")).all()
        assert set(index["files"]) == {f"{path.name}/{p.name}" for p in path.iterdir()}

    def test_export_is_deterministic(self, bundle, tmp_path):
        models, _, _ = bundle
        bs.export_bundle(models, tmp_path / "1.tar")
        bs.export_bundle(models, tmp_path / "2.tar")
        assert (tmp_path / "1.tar").read_bytes() == (tmp_path / "2.tar").read_bytes()

    def test_import_without_activation_keeps_the_current_bundle(self, bundle, tmp_path):
        models, old, data = bundle
        new = publish_test_bundle(tmp_path / "elsewhere", data, seed=3)
        archive = tmp_path / "new.tar"
        bs.export_bundle(tmp_path / "elsewhere", archive)
        r = bs.import_bundle(models, archive, activate=False)
        assert r["installed"] and not r["activated"]
        assert mb.load_bundle(models, fx.PROP_LINES)["manifest"]["bundle_id"] == old.name
        bs.import_bundle(models, archive)                           # now activate
        assert mb.load_bundle(models, fx.PROP_LINES)["manifest"]["bundle_id"] == new.name

    def test_reimporting_the_same_bundle_is_idempotent(self, bundle, tmp_path):
        models, path, _ = bundle
        archive = tmp_path / "b.tar"
        bs.export_bundle(models, archive)
        before = snapshot(models / mb.BUNDLES_DIR)
        assert bs.import_bundle(models, archive)["installed"] is False
        assert snapshot(models / mb.BUNDLES_DIR) == before

    def test_conflicting_bundle_id_is_refused(self, bundle, tmp_path):
        models, path, data = bundle
        archive = tmp_path / "b.tar"
        bs.export_bundle(models, archive)
        target = tmp_path / "t"
        publish_test_bundle(target, data, seed=4)
        current = mb.load_bundle(target, fx.PROP_LINES)["manifest"]["bundle_id"]
        clash = target / mb.BUNDLES_DIR / path.name
        clash.mkdir()
        (clash / "manifest.json").write_text("{}")
        before = snapshot(target)
        with pytest.raises(mb.BundleError, match="already installed with different content"):
            bs.import_bundle(target, archive)
        assert snapshot(target) == before
        assert mb.load_bundle(target, fx.PROP_LINES)["manifest"]["bundle_id"] == current

    def test_exporting_a_tampered_bundle_is_refused(self, bundle, tmp_path):
        models, path, _ = bundle
        f = path / "passing_yards_starter.json"
        f.write_bytes(f.read_bytes() + b" ")
        with pytest.raises(mb.BundleError):
            bs.export_bundle(models, tmp_path / "x.tar")                 # current: load_bundle refuses
        with pytest.raises(mb.BundleError, match="doesn't match its manifest hash"):
            bs.export_bundle(models, tmp_path / "x.tar", bundle_id=path.name)   # by id: verified too
        assert not (tmp_path / "x.tar").exists()


class TestUnsafeArchives:
    @pytest.fixture
    def target(self, bundle, tmp_path):
        models, path, data = bundle
        archive = tmp_path / "good.tar"
        bs.export_bundle(models, archive)
        return models, path, archive

    def _refused(self, models, archive, match):
        before = snapshot(models)
        with pytest.raises(mb.BundleError, match=match):
            bs.import_bundle(models, archive)
        assert snapshot(models) == before

    @pytest.mark.parametrize("content", [b"", b"not a tar at all" * 20])
    def test_corrupt_archive(self, target, tmp_path, content):
        models, _, _ = target
        bad = tmp_path / "bad.tar"
        bad.write_bytes(content)
        self._refused(models, bad, "can't be read|no export.json|truncated")

    def test_truncated_archive(self, target, tmp_path):
        models, _, archive = target
        raw = archive.read_bytes()
        bad = tmp_path / "cut.tar"
        bad.write_bytes(raw[: len(raw) // 2])
        self._refused(models, bad, "truncated|corrupt|can't be read|differ|no ")

    @pytest.mark.parametrize("name,kind", [
        ("../evil.json", "file"), ("/abs.json", "file"), ("a\\b.json", "file"), ("C:x.json", "file"),
        ("20261024T000000Z-0000abcd/../../x.json", "file"), ("notes.txt", "file"),
        ("20261024T000000Z-0000abcd/x.exe", "file"), ("20261024T000000Z-0000abcd/link.json", "symlink"),
        ("20261024T000000Z-0000abcd/hard.json", "hardlink"), ("20261024T000000Z-0000abcd", "dir"),
        ("20261024T000000Z-0000abcd/pipe.json", "fifo"), ("a/b/c.json", "file"),
    ])
    def test_unsafe_entries(self, target, tmp_path, name, kind):
        models, _, _ = target
        bad = _archive(tmp_path, [(name, b"{}", kind)], index={"kind": bs.EXPORT_KIND})
        self._refused(models, bad, "not a regular file|not a safe relative path|unexpected entry")

    def test_duplicate_entries(self, target, tmp_path):
        models, path, archive = target
        index, contents = bs.read_archive(archive)
        name = next(iter(contents))
        bad = _archive(tmp_path, [(name, contents[name], "file"), (name, contents[name], "file")],
                       index=index)
        self._refused(models, bad, "duplicate entry")

    def test_oversized_entry(self, target, monkeypatch):
        models, _, archive = target
        monkeypatch.setattr(bs, "MAX_ENTRY_BYTES", 100)
        self._refused(models, archive, "larger than")

    @pytest.mark.parametrize("edit,match", [
        (lambda i, c: c.__setitem__(next(k for k in c if k.endswith("starter.json")),
                                    b"x" + c[next(k for k in c if k.endswith("starter.json"))]),
         "doesn't match its export hash"),
        (lambda i, c: c.pop(next(k for k in c if k.endswith("starter.json"))), "differ from the export index"),
        (lambda i, c: c.__setitem__(f"{i['bundle_id']}/extra.json", b"{}"), "differ from the export index"),
        (lambda i, c: i.update(manifest_sha256="0" * 64), "manifest hash"),
        (lambda i, c: i.update(kind="other"), "not a version-1 bundle export"),
        (lambda i, c: i.update(bundle_id="../../x"), "bundle_id is malformed"),
        (lambda i, c: i.update(files={}), "lists no files"),
    ])
    def test_index_and_hash_mismatches(self, target, tmp_path, edit, match):
        models, _, archive = target
        index, contents = bs.read_archive(archive)
        edit(index, contents)
        bad = _archive(tmp_path, [(k, v, "file") for k, v in contents.items()], index=index)
        self._refused(models, bad, match)

    def test_forged_archive_with_consistent_hashes_still_verifies_the_bundle(self, target, tmp_path):
        # Rehashed model bytes pass the export index but fail bundle verification.
        models, path, archive = target
        index, contents = bs.read_archive(archive)
        name = next(k for k in contents if k.endswith("starter.json"))
        contents[name] = contents[name] + b" "
        index["files"][name] = {"sha256": bs._sha256(contents[name]), "bytes": len(contents[name])}
        bad = _archive(tmp_path, [(k, v, "file") for k, v in contents.items()], index=index)
        target_dir = tmp_path / "fresh"
        with pytest.raises(mb.BundleError, match="doesn't match its manifest hash"):
            bs.import_bundle(target_dir, bad)
        assert not (target_dir / mb.POINTER_FILE).exists()
        assert not [p for p in (target_dir / mb.BUNDLES_DIR).iterdir()]   # staging removed


class TestInterruptedInstallation:
    def test_failed_verification_leaves_nothing(self, bundle, tmp_path, monkeypatch):
        models, path, data = bundle
        archive = tmp_path / "b.tar"
        bs.export_bundle(models, archive)
        target = tmp_path / "t"
        publish_test_bundle(target, data, seed=5)
        before = snapshot(target)
        monkeypatch.setattr(bs, "verify_bundle_dir",
                            lambda *a, **k: (_ for _ in ()).throw(mb.BundleError(["power cut"])))
        with pytest.raises(mb.BundleError, match="power cut"):
            bs.import_bundle(target, archive)
        monkeypatch.undo()
        assert snapshot(target) == before
        assert mb.load_bundle(target, fx.PROP_LINES)["manifest"]["bundle_id"] != path.name

    def test_failed_pointer_switch_keeps_the_previous_current(self, bundle, tmp_path, monkeypatch):
        models, path, data = bundle
        archive = tmp_path / "b.tar"
        bs.export_bundle(models, archive)
        target = tmp_path / "t"
        old = publish_test_bundle(target, data, seed=6)
        real = os.replace

        def fail_pointer(src, dst):
            if str(dst).endswith(mb.POINTER_FILE):
                raise OSError("disk full")
            return real(src, dst)
        monkeypatch.setattr(mb.os, "replace", fail_pointer)
        with pytest.raises(OSError, match="disk full"):
            bs.import_bundle(target, archive)
        monkeypatch.undo()
        assert mb.load_bundle(target, fx.PROP_LINES)["manifest"]["bundle_id"] == old.name
        assert not list(target.glob(".*.tmp"))
        # the new bundle was installed (verified) but not made current; a retry activates it
        assert bs.import_bundle(target, archive) == {"bundle_id": path.name, "installed": False,
                                                    "activated": True}

    def test_switch_pointer_refuses_a_tampered_bundle(self, bundle, tmp_path):
        models, old, data = bundle
        new = publish_test_bundle(models, data, seed=7)
        f = old / "passing_yards_starter.json"
        f.write_bytes(f.read_bytes() + b" ")
        with mb.publisher_lock(models, "t"):
            with pytest.raises(mb.BundleError):
                mb.switch_pointer(models, old.name, fx.PROP_LINES)
        assert mb.load_bundle(models, fx.PROP_LINES)["manifest"]["bundle_id"] == new.name


class TestPublisherLock:
    def test_concurrent_publish_is_refused(self, bundle):
        models, old, data = bundle
        with mb.publisher_lock(models, "other-publisher"):
            with pytest.raises(mb.BundleError, match="holds lock file"):
                publish_test_bundle(models, data, seed=8)
            assert (models / ".model-bundle.lock").exists()
        assert mb.load_bundle(models, fx.PROP_LINES)["manifest"]["bundle_id"] == old.name
        assert not (models / ".model-bundle.lock").exists()
        publish_test_bundle(models, data, seed=8)                     # free again

    def test_concurrent_import_is_refused(self, bundle, tmp_path):
        models, path, data = bundle
        archive = tmp_path / "b.tar"
        bs.export_bundle(models, archive)
        target = tmp_path / "t"
        target.mkdir()
        with mb.publisher_lock(target, "other"):
            with pytest.raises(mb.BundleError, match="holds lock file"):
                bs.import_bundle(target, archive)
        assert not (target / mb.POINTER_FILE).exists()

    def test_an_old_lock_is_not_removed_on_age(self, bundle):
        models, old, data = bundle
        lock = models / ".model-bundle.lock"
        started = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
        lock.write_text(json.dumps({"what": "model bundle publication", "id": "crashed",
                                    "pid": 999999, "host": "elsewhere", "started_at": started,
                                    "token": "theirs"}))
        with pytest.raises(mb.BundleError, match="older than expected") as exc:
            publish_test_bundle(models, data, seed=9)
        assert "only after confirming" in str(exc.value)
        assert lock.exists() and json.loads(lock.read_text())["token"] == "theirs"

    def test_a_taken_over_lock_is_not_deleted_by_the_previous_holder(self, bundle):
        models, _, _ = bundle
        lock = models / ".model-bundle.lock"
        with mb.publisher_lock(models, "first"):
            info = json.loads(lock.read_text())
            lock.write_text(json.dumps({**info, "token": "someone-else"}))
        assert lock.exists() and json.loads(lock.read_text())["token"] == "someone-else"


class TestInventory:
    def test_lists_without_deleting(self, bundle, tmp_path):
        models, old, data = bundle
        new = publish_test_bundle(models, data, seed=10)
        (models / mb.BUNDLES_DIR / ".leftover.partial").mkdir()
        f = old / "passing_yards_starter.json"
        f.write_bytes(f.read_bytes() + b" ")
        before = snapshot(models)
        inv = bs.inventory(models)
        assert snapshot(models) == before
        assert (models / mb.BUNDLES_DIR / ".leftover.partial").is_dir()     # not removed
        assert inv["current"] == new.name
        by_id = {b["bundle_id"]: b for b in inv["bundles"]}
        assert by_id[new.name]["current"] and by_id[new.name]["verified"]
        assert not by_id[old.name]["current"] and not by_id[old.name]["verified"]
        assert inv["leftovers"] == [".leftover.partial"]


def test_production_models_and_data_are_untouched():
    root = Path(mb.ROOT)
    assert not (root / "player_props" / "models" / mb.BUNDLES_DIR).exists()
    assert not (root / "player_props" / "models" / mb.POINTER_FILE).exists()
    assert not (root / "data_files" / sr.SOURCE_NAME).exists()
    assert not (root / "data_files" / sr.MANIFEST_NAME).exists()


# ---------------------------------------------------------------------------
# Findings: schedule ambiguity, game dates, rollback recovery tracking
# ---------------------------------------------------------------------------

def _sched_with(tmp_path, edit):
    s = schedule([2024, 2025, 2026])
    edit(s)
    path = tmp_path / "edited_schedule.csv"
    s.to_csv(path, sep="\t", index=False)
    return path


class TestScheduleFailsClosed:
    G = gid(2025, 2, "DEN", "LV")

    def _fails(self, installed, tmp_path, edit, match, downloads=None):
        out, _, d, before = installed
        sched = _sched_with(tmp_path, edit)
        with pytest.raises(sr.SourceRefreshError, match=match) as exc:
            sr.refresh_source(out, sched, fetch=lambda s: (downloads or d)[s],
                              seasons=[2024, 2025, 2026], today=TODAY)
        assert snapshot(out) == before
        return exc.value.problems

    def test_invalid_date_for_a_missing_game_is_not_unfinished(self, installed, tmp_path):
        # The reproduced case: the game is absent and its schedule date is invalid.
        d = dict(installed[2])
        d[2025] = to_bytes(season_plays(2025, skip={self.G}))
        problems = self._fails(installed, tmp_path,
                               lambda s: s.__setitem__("gameday", s["gameday"].where(s["game_id"] != self.G, "2025-02-30")),
                               "invalid date", downloads=d)
        assert any(self.G in p for p in problems)

    def test_invalid_date_for_a_present_game(self, installed, tmp_path):
        self._fails(installed, tmp_path,
                    lambda s: s.__setitem__("gameday", s["gameday"].where(s["game_id"] != self.G, "TBD")),
                    "invalid date")

    @pytest.mark.parametrize("column", ["gameday", "home_team", "week"])
    def test_missing_schedule_values(self, installed, tmp_path, column):
        def edit(s):
            s[column] = s[column].astype(object)
            s.loc[s["game_id"] == self.G, column] = None
        self._fails(installed, tmp_path, edit, "missing")

    def test_duplicate_schedule_rows(self, installed, tmp_path):
        def edit(s):
            s.loc[len(s)] = s[s["game_id"] == self.G].iloc[0]
        self._fails(installed, tmp_path, edit, "listed more than once")

    def test_conflicting_schedule_rows(self, installed, tmp_path):
        # Another game puts DEN in week 2 a second time.
        def edit(s):
            extra = s[s["game_id"] == self.G].iloc[0].copy()
            extra["game_id"], extra["away_team"] = "2025_02_NYG_DEN", "NYG"
            s.loc[len(s)] = extra
        self._fails(installed, tmp_path, edit, "two games in one week")

    def test_other_season_game_id(self, installed, tmp_path):
        self._fails(installed, tmp_path,
                    lambda s: s.__setitem__("game_id", s["game_id"].where(s["game_id"] != self.G, "2024_02_LV_DEN")),
                    "listed more than once|other-season")

    @pytest.mark.parametrize("values,reason", [
        (dict(away_score=None), "one_score"),
        (dict(home_score=0.0, away_score=0.0, result=0.0, total=0.0), "placeholder"),
        (dict(result=3.0), "inconsistent"),
        (dict(result=None, total=None), "no_result_total"),
        (dict(home_score=1.0, result=-22.0, total=24.0), "invalid_score"),
    ])
    @pytest.mark.parametrize("missing", [False, True])
    def test_malformed_completion_evidence_fails(self, installed, tmp_path, values, reason, missing):
        def edit(s):
            for k, v in values.items():
                s[k] = s[k].astype(object)
                s.loc[s["game_id"] == self.G, k] = v
        d = dict(installed[2])
        if missing:
            d[2025] = to_bytes(season_plays(2025, skip={self.G}))
        problems = self._fails(installed, tmp_path, edit, "malformed or contradictory", downloads=d)
        assert any(f"{self.G} ({reason})" in p for p in problems)

    def test_genuinely_unfinished_games_are_allowed(self, installed, tmp_path):
        out, _, d, _ = installed
        # past-dated with no score recorded yet (postponed / not yet scored), absent from the download
        def unscored(s):
            for col in ("home_score", "away_score", "result", "total"):
                s[col] = s[col].where(s["game_id"] != self.G)
        sched = _sched_with(tmp_path, unscored)
        d2 = dict(d)
        d2[2025] = to_bytes(season_plays(2025, skip={self.G}))
        m = sr.refresh_source(out, sched, fetch=lambda s: d2[s], seasons=[2024, 2025, 2026], today=TODAY)
        assert m["coverage"]["2025"]["not_yet_completed"] == 1
        # dated in the future relative to the check date
        m = sr.refresh_source(out, installed[1], fetch=lambda s: d[s], seasons=[2024, 2025, 2026],
                              today=date(2026, 9, 14))
        assert m["coverage"]["2026"]["not_yet_completed"] == 2


class TestDownloadMatchesSchedule:
    def test_shifted_dates_fail(self, installed):
        # The reproduced case: every 2026 game moved to 2026-10-01.
        out, sched, d, before = installed
        df = season_plays(2026)
        df["game_date"] = "2026-10-01"
        d2 = dict(d)
        d2[2026] = to_bytes(df)
        with pytest.raises(sr.SourceRefreshError, match="dated differently from the schedule") as exc:
            sr.refresh_source(out, sched, fetch=lambda s: d2[s], seasons=[2024, 2025, 2026], today=TODAY)
        assert snapshot(out) == before
        assert any("2026-10-01 vs scheduled 2026-09-07" in p for p in exc.value.problems)

    def test_one_game_on_the_wrong_date(self, installed):
        out, sched, d, before = installed
        df = season_plays(2026)
        df.loc[df["game_id"] == gid(2026, 1, "KC", "BUF"), "game_date"] = "2026-09-08"
        d2 = dict(d)
        d2[2026] = to_bytes(df)
        with pytest.raises(sr.SourceRefreshError, match="1 downloaded game"):
            sr.refresh_source(out, sched, fetch=lambda s: d2[s], seasons=[2024, 2025, 2026], today=TODAY)
        assert snapshot(out) == before

    def test_wrong_week_label(self, installed):
        out, sched, d, before = installed
        df = season_plays(2026)
        df.loc[df["game_id"] == gid(2026, 2, "KC", "BUF"), "week"] = 3
        d2 = dict(d)
        d2[2026] = to_bytes(df)
        with pytest.raises(sr.SourceRefreshError, match="season or week differs"):
            sr.refresh_source(out, sched, fetch=lambda s: d2[s], seasons=[2024, 2025, 2026], today=TODAY)
        assert snapshot(out) == before


class TestRollbackRecovery:
    def _two(self, tmp_path, a_exists=True):
        dest_a, dest_b = tmp_path / "a.csv.gz", tmp_path / "b.json"
        if a_exists:
            dest_a.write_bytes(b"old A")
        dest_b.write_bytes(b"old B")
        st = tmp_path / "st"
        st.mkdir()
        (st / "a").write_bytes(b"new A")
        (st / "b").write_bytes(b"new B")
        return dest_a, dest_b, {dest_a: st / "a", dest_b: st / "b"}

    def test_failed_second_install_and_failed_restore(self, tmp_path, monkeypatch):
        dest_a, dest_b, staged = self._two(tmp_path)
        real = os.replace

        def flaky(src, dst):
            if Path(dst) == dest_b:
                raise OSError("disk full")                    # second install fails
            if Path(dst) == dest_a and str(src).endswith(".bak"):
                raise OSError("locked")                       # restoring the first fails
            return real(src, dst)
        monkeypatch.setattr(sr.os, "replace", flaky)
        with pytest.raises(sr.SourceRefreshError) as exc:
            sr.install_files(staged)
        msg = str(exc.value)
        backups = list(tmp_path.glob(".a.csv.gz.*.bak"))
        assert len(backups) == 1 and backups[0].read_bytes() == b"old A"      # recoverable
        assert f"{dest_a}: previous content kept at {backups[0]}" in msg      # names the real file
        assert "disk full" in msg and str(dest_b) not in msg
        assert dest_a.read_bytes() == b"new A" and dest_b.read_bytes() == b"old B"
        assert not list(tmp_path.glob(".b.json.*.bak"))                      # not needed: B untouched
        monkeypatch.undo()
        os.replace(backups[0], dest_a)                                        # manual recovery works
        assert dest_a.read_bytes() == b"old A"

    def test_initially_absent_destination_that_cannot_be_removed(self, tmp_path, monkeypatch):
        dest_a, dest_b, staged = self._two(tmp_path, a_exists=False)
        real = os.replace

        def fail_b(src, dst):
            if Path(dst) == dest_b:
                raise OSError("disk full")
            return real(src, dst)
        monkeypatch.setattr(sr.os, "replace", fail_b)
        real_unlink = Path.unlink
        monkeypatch.setattr(Path, "unlink", lambda self, *a, **k: (_ for _ in ()).throw(OSError("busy"))
                            if self == dest_a else real_unlink(self, *a, **k))
        with pytest.raises(sr.SourceRefreshError) as exc:
            sr.install_files(staged)
        assert f"{dest_a}: did not exist before, and the new file could not be removed" in str(exc.value)
        assert "previous content kept" not in str(exc.value)

    def test_successful_restore_is_dropped_from_the_recovery_list(self, tmp_path, monkeypatch):
        # Three files: A and B installed, C fails; B restores fine, A doesn't.
        dests = [tmp_path / n for n in ("a", "b", "c")]
        for d in dests:
            d.write_bytes(b"old " + d.name.encode())
        st = tmp_path / "st"
        st.mkdir()
        staged = {}
        for d in dests:
            (st / d.name).write_bytes(b"new " + d.name.encode())
            staged[d] = st / d.name
        real = os.replace

        def flaky(src, dst):
            if Path(dst) == dests[2]:
                raise OSError("disk full")
            if Path(dst) == dests[0] and str(src).endswith(".bak"):
                raise OSError("locked")
            return real(src, dst)
        monkeypatch.setattr(sr.os, "replace", flaky)
        with pytest.raises(sr.SourceRefreshError) as exc:
            sr.install_files(staged)
        msg = str(exc.value)
        assert f"{dests[0]}:" in msg and f"{dests[1]}:" not in msg
        assert dests[1].read_bytes() == b"old b"                                 # restored
        assert [p.name for p in tmp_path.glob(".*.bak")] == [p.name for p in tmp_path.glob(".a.*.bak")]

    def test_refresh_reports_the_recovery_file(self, installed, monkeypatch):
        out, sched, d, before = installed
        real = os.replace

        def flaky(src, dst):
            if str(dst).endswith(sr.MANIFEST_NAME) and ".staging-" in str(src):
                raise OSError("disk full")
            if str(dst).endswith(sr.SOURCE_NAME) and str(src).endswith(".bak"):
                raise OSError("locked")
            return real(src, dst)
        monkeypatch.setattr(sr.os, "replace", flaky)
        changed = season_plays(2026)
        changed.loc[changed["passing_yards"] > 0, "passing_yards"] = 31
        d2 = dict(d)
        d2[2026] = to_bytes(changed)
        with pytest.raises(sr.SourceRefreshError, match="previous content kept at") as exc:
            sr.refresh_source(out, sched, fetch=lambda s: d2[s], seasons=[2024, 2025, 2026], today=TODAY)
        monkeypatch.undo()
        [bak] = [p for p in out.iterdir() if p.name.endswith(".bak")]
        assert str(bak) in str(exc.value) and bak.read_bytes() == before[sr.SOURCE_NAME]
        assert (out / sr.MANIFEST_NAME).read_bytes() == before[sr.MANIFEST_NAME]
        os.replace(bak, out / sr.SOURCE_NAME)                     # recovery restores a verifiable source
        assert sr.verify_installed(out)["source"]["sha256"] == sr._sha256(before[sr.SOURCE_NAME])
