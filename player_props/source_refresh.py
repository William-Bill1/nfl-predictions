"""
Player-prop play-by-play source refresh (Stage 5A).

A prop-specific updater, separate from update_pbp_smart.py: the spread pipeline's
shared file (data_files/nfl_play_by_play_historical.csv.gz) and its
pbp_metadata.json are never read or written here.

* The required seasons are explicit (FIRST_SEASON through the current season,
  every intervening season included), never "a cache plus the current year".
* Each season is downloaded (``fetch`` is injectable, so tests run offline),
  checked for Git LFS pointers, parsed, and validated: season identity, game
  IDs, dates, duplicate plays, required columns, and game-level coverage of
  every completed game in the schedule (Stage 2 completion evidence: recorded
  scores corroborated by the play-by-play's own END GAME play).
* A complete candidate is staged and validated in a staging directory, then
  installed with its manifest as one set (replace, with rollback on a
  catchable failure). Any problem leaves the previous source in place and
  reports what is missing.
* The manifest records what was actually read: per-season download hashes,
  the installed file's hash and coverage (provenance.read_source), and the
  schedule file's hash. Requested seasons or fresh timestamps are never
  evidence on their own.

Score corrections and incomplete games: each refresh replaces every season
wholesale from a fresh download, so a corrected score arrives with the next
successful refresh. If the play-by-play's END GAME score and the recorded
schedule score disagree (one corrected before the other), the refresh fails and
the previous source stays until they agree. A game present without completion
evidence (in progress, or not yet scored in the schedule) is kept in the file
but listed in the manifest; Stage 2 excludes it from features until its
completion is corroborated.

Nothing here runs on a schedule. CLI (writes only to --out-dir):
    python player_props/source_refresh.py --out-dir DIR [--seasons 2020-2026]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from player_props.backtest import pbp_game_completion  # noqa: E402
from player_props.provenance import (  # noqa: E402
    GAME_COLUMNS, completion_status, read_source, toronto_today,
)
from season_utils import FIRST_SEASON, latest_pbp_season  # noqa: E402

SOURCE_NAME = 'nfl_play_by_play_props.csv.gz'
MANIFEST_NAME = 'player_props_source_manifest.json'
MANIFEST_KIND = 'player_props_source'
MANIFEST_VERSION = 1
NFLVERSE_URL = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.csv.gz'
LFS_PREFIX = b'version https://git-lfs'
# What the prop pipeline reads (train_models.REQUIRED_PBP_COLS, rush_attempt),
# what completion evidence needs, and play_id to find duplicate plays.
REQUIRED_COLUMNS = ['play_id', 'game_id', 'season', 'week', 'posteam', 'defteam', 'game_date',
                    'passer_player_id', 'passer_player_name', 'rusher_player_id', 'rusher_player_name',
                    'receiver_player_id', 'receiver_player_name', 'passing_yards', 'rushing_yards',
                    'receiving_yards', 'pass_touchdown', 'rush_touchdown', 'complete_pass',
                    'pass_attempt', 'interception', 'rush_attempt', 'desc', 'total_home_score',
                    'total_away_score']
OPTIONAL_COLUMNS = ['season_type']
_GAME_ID = re.compile(r'(\d{4})_(\d{2})_([A-Z]{2,3})_([A-Z]{2,3})')
EXAMPLES = 5


class SourceRefreshError(RuntimeError):
    """The refresh couldn't produce a complete, valid source. ``problems``
    lists every reason; the previous source was left in place."""

    def __init__(self, problems: List[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _examples(items) -> str:
    items = sorted(str(i) for i in items)
    more = f" (+{len(items) - EXAMPLES} more)" if len(items) > EXAMPLES else ""
    return ", ".join(items[:EXAMPLES]) + more


def required_seasons(first: int = FIRST_SEASON, last: Optional[int] = None,
                     today: Optional[date] = None) -> List[int]:
    """Every season from ``first`` through ``last`` (default: the newest season
    with play-by-play), with no gaps."""
    last = latest_pbp_season(today) if last is None else last
    if last < first:
        raise ValueError(f"no seasons between {first} and {last}")
    return list(range(first, last + 1))


def nflverse_fetch(season: int) -> bytes:
    """Download one season from nflverse (network). Never used by the tests."""
    import requests
    resp = requests.get(NFLVERSE_URL.format(season=season), timeout=120)
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# One season
# ---------------------------------------------------------------------------

def parse_season(data: bytes, season: int) -> pd.DataFrame:
    """Parse one downloaded season (gzipped or plain CSV). Raises
    SourceRefreshError for an LFS pointer, empty or unreadable data, or
    missing columns."""
    where = f"season {season}"
    if not data:
        raise SourceRefreshError([f"{where}: download is empty"])
    raw = data
    if data[:2] == b'\x1f\x8b':
        try:
            raw = gzip.decompress(data)
        except (OSError, EOFError) as exc:
            raise SourceRefreshError([f"{where}: gzip data is truncated or corrupt ({exc})"]) from exc
    # Checked after any decompression, so a gzipped pointer is caught too.
    if raw.lstrip().startswith(LFS_PREFIX):
        raise SourceRefreshError([f"{where}: download is a Git LFS pointer, not play-by-play data"])
    try:
        df = pd.read_csv(io.BytesIO(raw), low_memory=False,
                         usecols=lambda c: c in set(REQUIRED_COLUMNS + OPTIONAL_COLUMNS))
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise SourceRefreshError([f"{where}: not readable as play-by-play CSV ({exc})"]) from exc
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SourceRefreshError([f"{where}: missing column(s) {missing}"])
    if df.empty:
        raise SourceRefreshError([f"{where}: no plays"])
    return df[[c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if c in df.columns]]


def season_problems(df: pd.DataFrame, season: int) -> List[str]:
    """Content checks for one season's plays (identity, IDs, dates, duplicates)."""
    where = f"season {season}"
    problems = []
    seasons = pd.to_numeric(df['season'], errors='coerce')
    if seasons.isna().any() or set(seasons.astype(int)) != {season}:
        found = sorted(set(seasons.dropna().astype(int)))
        problems.append(f"{where}: plays are labelled season {found}, not {season}")
    ids = df['game_id'].astype(str)
    parsed = ids.str.fullmatch(_GAME_ID.pattern)
    if not parsed.all():
        problems.append(f"{where}: malformed game_id(s) {_examples(set(ids[~parsed]))}")
    elif set(ids.str[:4]) != {str(season)}:
        problems.append(f"{where}: game_id(s) from another season {_examples(set(ids[ids.str[:4] != str(season)]))}")
    days = pd.to_datetime(df['game_date'], format='%Y-%m-%d', errors='coerce')
    if days.isna().any():
        problems.append(f"{where}: {int(days.isna().sum())} play(s) with a missing or invalid game_date")
    else:
        per_game = df.assign(_d=days).groupby('game_id')['_d'].nunique()
        if (per_game > 1).any():
            problems.append(f"{where}: games with more than one date {_examples(per_game[per_game > 1].index)}")
        lo, hi = pd.Timestamp(f"{season}-08-01"), pd.Timestamp(f"{season + 1}-03-01")
        outside = df.loc[(days < lo) | (days > hi), 'game_id'].unique()
        if len(outside):
            problems.append(f"{where}: games dated outside the {season} season {_examples(outside)}")
    play = pd.to_numeric(df['play_id'], errors='coerce')
    if play.isna().any():
        problems.append(f"{where}: {int(play.isna().sum())} play(s) without a play_id")
    dup = df.assign(_p=play).duplicated(['game_id', '_p'], keep=False)
    if dup.any():
        problems.append(f"{where}: {int(dup.sum())} duplicated play(s) in "
                        f"{_examples(set(df.loc[dup, 'game_id']))}")
    return problems


# Completion statuses (provenance.completion_status) meaning a game genuinely
# hasn't finished: no score recorded yet, or dated today or later. Any other
# non-completed status is malformed or contradictory evidence and fails.
INCOMPLETE = {'no_score', 'not_past'}
SOURCE_FAILURES = {
    'not_in_source': 'completed game(s) missing from the download',
    'source_unended': 'completed game(s) without a single END GAME score (partial)',
    'source_mismatch': 'game(s) whose END GAME score differs from the recorded score '
                       '(a correction not yet in both sources?)',
}


def schedule_problems(sched: pd.DataFrame, season: int) -> List[str]:
    """Malformed or ambiguous schedule rows for one season. Coverage can't be
    established from a schedule with any of these."""
    where = f"season {season} schedule"
    problems = []
    needed = ['game_id', 'week', 'gameday', 'home_team', 'away_team']
    blank = sched[needed].isna().any(axis=1)
    if blank.any():
        problems.append(f"{where}: {int(blank.sum())} row(s) missing {', '.join(needed)} values")
    ids = sched['game_id'].astype(str)
    bad_id = ~ids.str.fullmatch(_GAME_ID.pattern) | (ids.str[:4] != str(season))
    if bad_id.any():
        problems.append(f"{where}: malformed or other-season game_id(s) {_examples(set(ids[bad_id]))}")
    dup = ids[ids.duplicated(keep=False)].unique()
    if len(dup):
        problems.append(f"{where}: game_id(s) listed more than once {_examples(dup)}")
    bad_day = sched['parsed_day'].isna() & sched['gameday'].notna()
    if bad_day.any():
        problems.append(f"{where}: invalid date(s) for {_examples(set(ids[bad_day]))}")
    sides = pd.concat([sched[['week', 'home_team']].rename(columns={'home_team': 'team'}),
                       sched[['week', 'away_team']].rename(columns={'away_team': 'team'})]).dropna()
    twice = sides[sides.duplicated(keep=False)].drop_duplicates()
    if len(twice):
        problems.append(f"{where}: a team in two games in one week "
                        f"{_examples(f'wk{r.week} {r.team}' for r in twice.itertuples())}")
    return problems


def coverage_problems(df: pd.DataFrame, games: pd.DataFrame, season: int,
                      today: date) -> tuple[List[str], dict]:
    """Game-level coverage of one season against the schedule. Returns
    (problems, summary); coverage is established only with no problems.

    * The schedule itself must be well formed (schedule_problems).
    * Every downloaded game must match one schedule row, with the same date,
      season and week.
    * Every completed game (Stage 2 completion predicate) must be present with
      an END GAME play matching the recorded score.
    * A game without completion evidence counts as unfinished only when no
      score is recorded or it's dated today or later. Malformed evidence (one
      score, invalid or 0-0 scores, a missing or inconsistent result/total, an
      invalid date) is a problem, never an unfinished game."""
    where = f"season {season}"
    sched = games[games['season'] == season].copy()
    if sched.empty:
        return [f"{where}: the schedule has no games for this season, so coverage can't be checked"], {}
    sched['parsed_day'] = pd.to_datetime(sched['gameday'], format='%Y-%m-%d', errors='coerce')
    problems = schedule_problems(sched, season)
    if problems:
        return problems, {}

    # Downloaded games against their unique schedule row.
    by_id = sched.set_index('game_id')
    per_game = df.groupby('game_id').agg(game_date=('game_date', 'first'), week=('week', 'first'),
                                         season=('season', 'first'))
    unknown = sorted(set(per_game.index) - set(by_id.index))
    if unknown:
        problems.append(f"{where}: {len(unknown)} game(s) not in the schedule {_examples(unknown)}")
    known = per_game.loc[per_game.index.isin(by_id.index)]
    row = by_id.loc[known.index]
    off_date = known.index[known['game_date'].astype(str).to_numpy() != row['gameday'].astype(str).to_numpy()]
    off_week = known.index[(pd.to_numeric(known['week'], errors='coerce').to_numpy()
                            != pd.to_numeric(row['week'], errors='coerce').to_numpy())
                           | (pd.to_numeric(known['season'], errors='coerce').to_numpy()
                              != pd.to_numeric(row['season'], errors='coerce').to_numpy())]
    if len(off_date):
        examples = [f"{g} ({known.at[g, 'game_date']} vs scheduled {by_id.at[g, 'gameday']})"
                    for g in sorted(off_date)]
        problems.append(f"{where}: {len(off_date)} downloaded game(s) dated differently from the "
                        f"schedule: {_examples(examples)}")
    if len(off_week):
        problems.append(f"{where}: {len(off_week)} downloaded game(s) whose season or week differs "
                        f"from the schedule: {_examples(off_week)}")

    completion = pbp_game_completion(df)
    status = pd.Series([completion_status(r, completion, today) for r in sched.itertuples()],
                       index=sched.index, dtype=object)
    for reason, text in SOURCE_FAILURES.items():
        ids = sched.loc[status == reason, 'game_id']
        if len(ids):
            problems.append(f"{where}: {len(ids)} {text}: {_examples(ids)}")
    malformed = sched[~status.isin({'completed', *SOURCE_FAILURES, *INCOMPLETE})]
    if len(malformed):
        detail = [f"{g} ({status[i]})" for i, g in zip(malformed.index, malformed['game_id'])]
        problems.append(f"{where}: {len(malformed)} game(s) with malformed or contradictory completion "
                        f"evidence in the schedule, so coverage can't be established: {_examples(detail)}")
    present = set(per_game.index)
    unfinished = status.isin(INCOMPLETE)
    summary = {'scheduled': int(len(sched)),
               'completed_and_present': int((status == 'completed').sum()),
               'games_in_download': int(len(present)),
               'not_yet_completed': int(unfinished.sum()),
               'present_without_completion_evidence': sorted(set(sched.loc[unfinished, 'game_id']) & present)}
    return problems, summary


# ---------------------------------------------------------------------------
# Installing a set of files
# ---------------------------------------------------------------------------

def install_files(staged: Dict[Path, Path]) -> None:
    """Move staged files to their destinations as a set.

    Each existing destination is copied to a backup first. If a move raises,
    the destinations already replaced are rolled back in reverse: restored
    from their backup, or removed if they didn't exist before. A backup leaves
    the recovery list only once its restore has succeeded. If any restore
    fails, SourceRefreshError names every destination left wrong and, for
    each, the backup still holding its previous content (or that it didn't
    exist before); those backups are kept. Otherwise the original error is
    re-raised and all backups are removed. Not crash-atomic."""
    backups: Dict[Path, Path] = {}
    moved: List[Path] = []
    unrestored: List[Path] = []
    try:
        for dest in staged:
            if dest.exists():
                bak = dest.with_name(f".{dest.name}.{secrets.token_hex(4)}.bak")
                backups[dest] = bak        # registered first, so a failed copy is cleaned up
                shutil.copy2(dest, bak)
        try:
            for dest, src in staged.items():
                os.replace(src, dest)
                moved.append(dest)
        except Exception as exc:
            for dest in reversed(moved):
                try:
                    if dest in backups:
                        os.replace(backups[dest], dest)
                        del backups[dest]          # restored: no longer a recovery copy
                    else:
                        dest.unlink()
                except Exception:      # noqa: BLE001
                    unrestored.append(dest)
            if unrestored:
                detail = [f"{d}: previous content kept at {backups[d]}" if d in backups
                          else f"{d}: did not exist before, and the new file could not be removed"
                          for d in unrestored]
                raise SourceRefreshError([
                    f"installing the source failed ({exc!r}) and the rollback also failed. "
                    + "; ".join(detail)]) from exc
            raise
    finally:
        for dest, bak in backups.items():
            if dest not in unrestored and bak.exists():
                bak.unlink()


# ---------------------------------------------------------------------------
# The refresh
# ---------------------------------------------------------------------------

def _write_tsv_gz(df: pd.DataFrame, path: Path) -> None:
    """Tab-separated, gzipped, with a fixed gzip timestamp so the same content
    hashes the same."""
    text = df.to_csv(sep='\t', index=False).encode('utf-8')
    with open(path, 'wb') as f, gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as gz:
        gz.write(text)


def refresh_source(out_dir: Path | str, schedule_path: Path | str,
                   fetch: Callable[[int], bytes] = nflverse_fetch,
                   seasons: Optional[List[int]] = None, today: Optional[date] = None) -> dict:
    """Download, validate and install a complete prop source in ``out_dir``.

    Returns the installed manifest. On any problem raises SourceRefreshError
    listing every problem; the previous source and manifest in ``out_dir`` are
    untouched and the staging directory is removed."""
    from ontario_spreads import CaptureError, SlotLock, new_id
    out_dir = Path(out_dir)
    today = today or toronto_today()
    seasons = required_seasons(today=today) if seasons is None else list(seasons)
    if not seasons:
        raise SourceRefreshError(["no seasons requested"])
    if seasons != list(range(seasons[0], seasons[-1] + 1)):
        raise SourceRefreshError([f"required seasons {seasons} have gaps"])
    try:
        schedule_bytes = Path(schedule_path).read_bytes()
        games = pd.read_csv(io.BytesIO(schedule_bytes), sep='\t', low_memory=False, usecols=GAME_COLUMNS)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise SourceRefreshError([f"schedule {schedule_path} could not be read: {exc}"]) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        lock = SlotLock(out_dir, 'prop-source-refresh', new_id(datetime.now(timezone.utc)),
                        what='prop source refresh')
        lock.__enter__()
    except CaptureError as exc:
        raise SourceRefreshError([str(exc)]) from exc
    staging = out_dir / f".staging-{secrets.token_hex(6)}"
    try:
        staging.mkdir()
        problems, frames, downloads, coverage = [], [], {}, {}
        for season in seasons:
            try:
                data = fetch(season)
            except Exception as exc:  # noqa: BLE001 - any download failure is reported
                problems.append(f"season {season}: download failed ({type(exc).__name__}: {exc})")
                continue
            data = bytes(data or b'')
            (staging / f"season_{season}.download").write_bytes(data)
            downloads[str(season)] = {'sha256': _sha256(data), 'bytes': len(data)}
            try:
                df = parse_season(data, season)
            except SourceRefreshError as exc:
                problems += exc.problems
                continue
            found = season_problems(df, season)
            problems += found
            if found:
                continue
            found, summary = coverage_problems(df, games, season, today)
            problems += found
            coverage[str(season)] = summary
            downloads[str(season)].update(rows=int(len(df)), games=int(df['game_id'].nunique()))
            frames.append(df)
        if problems:
            raise SourceRefreshError(problems)

        candidate = staging / SOURCE_NAME
        _write_tsv_gz(pd.concat(frames, ignore_index=True), candidate)
        _, observation = read_source(candidate)          # hash + coverage of the exact bytes
        got = sorted(int(s) for s in observation['coverage']['seasons'])
        if got != seasons:
            raise SourceRefreshError([f"candidate covers seasons {got}, required {seasons}"])
        manifest = {
            'kind': MANIFEST_KIND, 'manifest_version': MANIFEST_VERSION,
            # When this refresh ran - the data's own coverage is in observation.
            'refreshed_at': _now(), 'required_seasons': seasons,
            'source': {k: v for k, v in observation.items() if k != 'completion'},
            'downloads': downloads, 'coverage': coverage,
            'schedule': {'file': Path(schedule_path).name, 'sha256': _sha256(schedule_bytes)},
            'checked_through': today.isoformat(),
        }
        manifest_tmp = staging / MANIFEST_NAME
        manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        install_files({out_dir / SOURCE_NAME: candidate, out_dir / MANIFEST_NAME: manifest_tmp})
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        lock.__exit__(None, None, None)


def verify_installed(out_dir: Path | str) -> dict:
    """Re-check an installed source against its manifest (hash, size, not an
    LFS pointer). Returns the manifest; raises SourceRefreshError otherwise."""
    out_dir = Path(out_dir)
    try:
        manifest = json.loads((out_dir / MANIFEST_NAME).read_text(encoding='utf-8'))
        data = (out_dir / SOURCE_NAME).read_bytes()
    except (OSError, ValueError) as exc:
        raise SourceRefreshError([f"installed source can't be read: {exc}"]) from exc
    if not isinstance(manifest, dict) or manifest.get('kind') != MANIFEST_KIND \
            or manifest.get('manifest_version') != MANIFEST_VERSION:
        raise SourceRefreshError([f"{MANIFEST_NAME} is not a version-{MANIFEST_VERSION} source manifest"])
    if data.startswith(LFS_PREFIX):
        raise SourceRefreshError([f"{SOURCE_NAME} is a Git LFS pointer, not play-by-play data"])
    src = manifest.get('source') or {}
    if src.get('sha256') != _sha256(data) or src.get('bytes') != len(data):
        raise SourceRefreshError([f"{SOURCE_NAME} doesn't match its manifest hash"])
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Refresh the player-prop play-by-play source")
    ap.add_argument('--out-dir', type=Path, required=True,
                    help="directory to install the prop source in (required; nothing is written elsewhere)")
    ap.add_argument('--schedule', type=Path, default=ROOT / 'data_files' / 'nfl_games_historical.csv')
    ap.add_argument('--seasons', help="FIRST-LAST, e.g. 2020-2026 (default: FIRST_SEASON to current)")
    args = ap.parse_args(argv)
    seasons = None
    if args.seasons:
        first, last = (int(x) for x in args.seasons.split('-'))
        seasons = required_seasons(first, last)
    try:
        manifest = refresh_source(args.out_dir, args.schedule, seasons=seasons)
    except SourceRefreshError as exc:
        print("Source refresh failed; the previous source was left in place:")
        for p in exc.problems:
            print(f"  - {p}")
        return 1
    cov = manifest['source']['coverage']
    print(f"Installed {SOURCE_NAME}: seasons {', '.join(cov['seasons'])}, {cov['games']} games, "
          f"last game {cov['last_date']}, sha256 {manifest['source']['sha256'][:12]}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
