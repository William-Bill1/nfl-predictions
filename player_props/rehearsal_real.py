"""
Real nflverse inputs for the isolated Player Props rehearsal.

Two separate steps, so the rehearsal run itself stays offline:

1. ``download`` (network; nothing else uses it): fetches the public nflverse
   play-by-play for each season, with the source refresh's own
   ``nflverse_fetch``, and the nflverse schedule (``games.csv``) into
   ``<workspace>/downloads/<UTC time>-<token>/``, a directory created
   exclusively for that attempt (never reused or overwritten).
   ``downloads.json`` records each file's URL, size, SHA-256 and fetch time,
   and is written only once the attempt has succeeded. The workspace is
   checked exactly as the rehearsal checks it (no overlap with the repository
   or production paths).
2. ``RealInputs`` (offline): re-reads a downloads directory, verifying every
   file against ``downloads.json``, and gives the rehearsal what its synthetic
   fixtures give it: the schedule in ``provenance.GAME_COLUMNS`` form, a
   per-season fetch function for ``source_refresh.refresh_source``, the players
   reference, and the ``nfl_schedule_<season>.csv`` prediction schedule (built
   as update_schedule.py builds it).

The players reference (``players.csv``: IDs, names, positions) is not part of
the source-refresh contract and is not downloaded; the caller names the file
explicitly (normally the repository's committed copy) and its hash is recorded.

CLI (network, writes only under the workspace):
    python player_props/rehearsal_real.py --workspace DIR --seasons 2020-2026
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from player_props.provenance import GAME_COLUMNS  # noqa: E402
from player_props.rehearsal import RehearsalError, check_workspace, safe_dir  # noqa: E402
from player_props.source_refresh import NFLVERSE_URL, nflverse_fetch  # noqa: E402

SCHEDULE_URL = 'https://github.com/nflverse/nfldata/raw/master/data/games.csv'
SCHEDULE_NAME = 'games.csv'
DOWNLOADS_NAME = 'downloads.json'
DOWNLOADS_KIND = 'player_props_rehearsal_downloads'
PLAYER_COLUMNS = ['gsis_id', 'short_name', 'display_name', 'last_name', 'position', 'position_group']
ATTEMPT_TRIES = 20


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _token() -> str:
    return secrets.token_hex(4)


def pbp_name(season: int) -> str:
    return f'play_by_play_{season}.csv.gz'


def fetch_url(url: str) -> bytes:
    """One public download (network)."""
    import requests
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.content


def use_system_trust_store() -> str:
    """Verify TLS against the operating system's trust store (via truststore)
    instead of certifi's bundle. Needed where a local security product
    re-signs HTTPS with a root only the OS trusts; verification stays on.
    Returns the trust source used."""
    try:
        import truststore
    except ImportError:
        return 'certifi'
    truststore.inject_into_ssl()
    return 'system (truststore)'


def download(workspace: Path | str, seasons: List[int], *,
             fetch_pbp: Callable[[int], bytes] = nflverse_fetch,
             fetch_schedule: Callable[[str], bytes] = fetch_url,
             protected: Optional[List[Path]] = None, tls_trust: str = 'not recorded') -> Path:
    """Download the schedule and every season into a new directory under the
    workspace and return it.

    Each attempt gets its own directory (UTC time plus a random token),
    created exclusively: an existing directory is never reused, so a previous
    attempt's files can't be overwritten even within the same second. Files
    are created exclusively too. downloads.json, the completion record, is
    written last, only after every download succeeded, and never replaces an
    existing one. Any failure raises; files already written stay in that
    attempt's directory, which has no downloads.json and so can't be used."""
    if seasons != list(range(seasons[0], seasons[-1] + 1)):
        raise RehearsalError(f"seasons {seasons} must be consecutive")
    ws = check_workspace(Path(workspace), protected)
    ws.mkdir(parents=True, exist_ok=True)
    ws = check_workspace(ws, protected)
    parent = safe_dir(ws / 'downloads', ws, protected)
    dest = None
    for _ in range(ATTEMPT_TRIES):
        candidate = parent / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{_token()}"
        try:
            dest = safe_dir(candidate, ws, protected, exclusive=True)
            break
        except FileExistsError:
            continue
    if dest is None:
        raise RehearsalError(f"no unused download directory under {parent} after {ATTEMPT_TRIES} tries")
    files: Dict[str, dict] = {}

    def keep(name: str, url: str, get: Callable[[], bytes]) -> None:
        t0 = time.monotonic()
        data = get()
        with open(dest / name, 'xb') as fh:       # exclusive: never replaces a file
            fh.write(data)
        files[name] = {'url': url, 'bytes': len(data), 'sha256': _sha256(data),
                       'fetched_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                       'seconds': round(time.monotonic() - t0, 1)}

    keep(SCHEDULE_NAME, SCHEDULE_URL, lambda: fetch_schedule(SCHEDULE_URL))
    for season in seasons:
        keep(pbp_name(season), NFLVERSE_URL.format(season=season), lambda s=season: fetch_pbp(s))
    doc = {'kind': DOWNLOADS_KIND, 'version': 1, 'seasons': seasons, 'files': files,
           'tls_trust': tls_trust, 'completed_at': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    # Complete file first, then a hard link into place: the record appears
    # whole or not at all, and an existing one is never replaced.
    tmp = dest / f".{DOWNLOADS_NAME}.{_token()}.tmp"
    with open(tmp, 'x', encoding='utf-8') as fh:
        fh.write(json.dumps(doc, indent=2))
    try:
        os.link(tmp, dest / DOWNLOADS_NAME)
    finally:
        tmp.unlink()
    return dest


class RealInputs:
    """A verified downloads directory, offline. Every file's bytes are checked
    against downloads.json when read."""

    def __init__(self, downloads_dir: Path | str, players_file: Path | str):
        self.dir = Path(downloads_dir)
        try:
            self.record = json.loads((self.dir / DOWNLOADS_NAME).read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise RehearsalError(f"{self.dir} has no readable {DOWNLOADS_NAME}: {exc}") from exc
        if self.record.get('kind') != DOWNLOADS_KIND:
            raise RehearsalError(f"{self.dir / DOWNLOADS_NAME} is not a rehearsal downloads record")
        self.seasons = list(self.record['seasons'])
        games = pd.read_csv(io.BytesIO(self._bytes(SCHEDULE_NAME)), low_memory=False)
        missing = [c for c in GAME_COLUMNS + ['stadium'] if c not in games.columns]
        if missing:
            raise RehearsalError(f"{SCHEDULE_NAME} lacks column(s) {missing}")
        self.games = games[games['season'].isin(self.seasons)].reset_index(drop=True)
        players_file = Path(players_file)
        raw = players_file.read_bytes()
        self.players = pd.read_csv(io.BytesIO(raw), low_memory=False)
        missing = [c for c in PLAYER_COLUMNS if c not in self.players.columns]
        if missing:
            raise RehearsalError(f"players file {players_file} lacks column(s) {missing}")
        self.players_info = {'file': str(players_file), 'sha256': _sha256(raw), 'bytes': len(raw),
                             'rows': int(len(self.players))}

    def _bytes(self, name: str) -> bytes:
        entry = self.record.get('files', {}).get(name)
        if entry is None:
            raise OSError(f"{name} was not downloaded into {self.dir}")
        data = (self.dir / name).read_bytes()
        if _sha256(data) != entry['sha256'] or len(data) != entry['bytes']:
            raise RehearsalError(f"{self.dir / name} doesn't match its recorded download hash")
        return data

    def schedule(self) -> pd.DataFrame:
        """The schedule for the downloaded seasons, in the form source_refresh reads."""
        return self.games[GAME_COLUMNS].copy()

    def fetch(self, season: int) -> bytes:
        """source_refresh's fetch: the downloaded season's exact bytes, no network."""
        return self._bytes(pbp_name(season))

    def prediction_schedule(self, season: int) -> pd.DataFrame:
        """nfl_schedule_<season>.csv as update_schedule.py writes it."""
        s = self.games[self.games['season'] == season]
        return pd.DataFrame({'week': s['week'], 'date': pd.to_datetime(s['gameday']).dt.strftime('%Y-%m-%d'),
                             'home_team': s['home_team'], 'away_team': s['away_team'],
                             'venue': s['stadium'], 'status': 'REG'})

    def describe(self) -> dict:
        return {'downloads_dir': str(self.dir), 'seasons': self.seasons,
                'files': {n: {k: e[k] for k in ('url', 'bytes', 'sha256', 'fetched_at')}
                          for n, e in self.record['files'].items()},
                'players': self.players_info}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Download real nflverse inputs for the Player Props "
                                             "rehearsal (network; writes only under the workspace)")
    ap.add_argument('--workspace', type=Path, required=True)
    ap.add_argument('--seasons', required=True, help="FIRST-LAST, e.g. 2020-2026")
    args = ap.parse_args(argv)
    first, last = (int(x) for x in args.seasons.split('-'))
    trust = use_system_trust_store()
    try:
        dest = download(args.workspace, list(range(first, last + 1)), tls_trust=trust)
    except RehearsalError as exc:
        print(json.dumps({'status': 'refused', 'error': str(exc)}))
        return 2
    print(json.dumps({'status': 'downloaded', 'dir': str(dest)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
