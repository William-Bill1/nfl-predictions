"""Reviewed kickoff corrections for NFL-announced 2026 schedule changes (research context only).

The historical schedule (nfl_games_historical.csv and the predictions table built
from it) can list a pre-change time for a game the NFL moved: two Week 11 games
(announced November 9, 2026) and three Week 12 games (November 29, 2026). Injury
collection, the stored-context panel and the weekly market context apply these
corrections in memory so that their kickoff matching agrees with the real games.
Each game keeps the NFL document that announced its change as its authority.
Nothing here changes a schedule file, a prediction input, a probability or a
ranking, and it is not general schedule management: only the listed games are
known, and a later change that leaves the source at a recognized time can't be
detected.

Every correction requires the exact game ID, teams, season, week and date, and a
source time that is either the known old time or the reviewed time. Anything
else raises KickoffReviewError ("review required") so callers fail closed.
"""
from __future__ import annotations

from typing import NamedTuple

import pandas as pd

import pregame_snapshots as ps

WEEK11_AUTHORITY = ("https://media.nfl.com/content/dam/communications/documents/2026/"
                    "Week%2011%20Schedule%20Change.pdf")
WEEK12_AUTHORITY = ("https://media.nfl.com/content/dam/communications/documents/2026/"
                    "Week%2012%20Schedule%20Change.pdf")
# Version 1 blocks (written before the Week 12 change) carry one block authority,
# the Week 11 document, and may list only Week 11 games. Version 2 blocks list the
# distinct authorities of their records under "authorities".
REVIEW_VERSION = 2
V1_AUTHORITY = WEEK11_AUTHORITY
_BLOCK_FIELDS = {1: {"review_version", "authority", "games"},
                 2: {"review_version", "authorities", "games"}}
TIMEZONE = ps.SOURCE_TZ.key           # nflverse local times; the same rule the scope uses

APPLIED = "applied"                                   # source had the old time; corrected in memory
SOURCE_ALREADY_CORRECTED = "source_already_corrected"  # source already had the reviewed time
STATUSES = (APPLIED, SOURCE_ALREADY_CORRECTED)
RECORD_FIELDS = ("game_id", "season", "week", "away_team", "home_team", "gameday", "timezone",
                 "source_gametime", "effective_gametime", "source_kickoff_utc",
                 "effective_kickoff_utc", "status", "authority")


class Reviewed(NamedTuple):
    game_id: str
    season: int
    week: int
    away_team: str
    home_team: str
    gameday: str
    old_gametime: str
    effective_gametime: str
    authority: str


REVIEWED = {r.game_id: r for r in (
    Reviewed("2026_11_PIT_PHI", 2026, 11, "PIT", "PHI", "2026-11-22", "16:25", "13:00", WEEK11_AUTHORITY),
    Reviewed("2026_11_JAX_NYG", 2026, 11, "JAX", "NYG", "2026-11-22", "13:00", "16:25", WEEK11_AUTHORITY),
    Reviewed("2026_12_TEN_JAX", 2026, 12, "TEN", "JAX", "2026-11-29", "16:05", "13:00", WEEK12_AUTHORITY),
    Reviewed("2026_12_NE_LAC", 2026, 12, "NE", "LAC", "2026-11-29", "20:20", "16:05", WEEK12_AUTHORITY),
    Reviewed("2026_12_LV_CLE", 2026, 12, "LV", "CLE", "2026-11-29", "13:00", "20:20", WEEK12_AUTHORITY),
)}


class KickoffReviewError(ValueError):
    """A reviewed game no longer matches its reviewed identity or times."""


def _iso(gameday: str, gametime: str) -> str:
    kickoff, problem = ps.kickoff_utc(gameday, gametime)
    if kickoff is None:                       # can't happen for the fixed table; fail closed anyway
        raise KickoffReviewError(f"reviewed kickoff {gameday} {gametime} unusable ({problem})")
    return kickoff.strftime("%Y-%m-%dT%H:%M:%SZ")


def _int(value):
    """Exact integer or None (bools, NaN, fractions and other text are not
    integers; a plain digit string, as in a table read as text, is)."""
    if isinstance(value, str):
        text = value.strip()
        return int(text) if text.isascii() and text.isdigit() else None
    if isinstance(value, bool) or value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        return int(value) if float(value) == int(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _text(value) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def expected_record(game_id: str, source_gametime: str) -> dict:
    """The record for a reviewed game whose source lists `source_gametime`."""
    r = REVIEWED[game_id]
    status = SOURCE_ALREADY_CORRECTED if source_gametime == r.effective_gametime else APPLIED
    return {"game_id": r.game_id, "season": r.season, "week": r.week, "away_team": r.away_team,
            "home_team": r.home_team, "gameday": r.gameday, "timezone": TIMEZONE,
            "source_gametime": source_gametime, "effective_gametime": r.effective_gametime,
            "source_kickoff_utc": _iso(r.gameday, source_gametime),
            "effective_kickoff_utc": _iso(r.gameday, r.effective_gametime),
            "status": status, "authority": r.authority}


def resolve(game_id, *, season, week, away_team, home_team, gameday, gametime) -> dict | None:
    """None for a game that isn't reviewed; otherwise its record, after checking
    the exact identity and that the source time is the old or reviewed time."""
    gid = _text(game_id)
    if gid not in REVIEWED:
        return None
    r = REVIEWED[gid]
    day = _text(gameday)
    found = (_int(season), _int(week), _text(away_team), _text(home_team),
             day[:10] if day else None, _text(gametime))
    if found[:5] != (r.season, r.week, r.away_team, r.home_team, r.gameday) \
            or (day and len(day) > 10 and day[10:].strip() not in ("", "00:00:00")) \
            or found[5] not in (r.old_gametime, r.effective_gametime):
        raise KickoffReviewError(
            f"reviewed kickoff no longer matches schedule: {gid} (found season/week {found[0]}/{found[1]}, "
            f"{found[2]} @ {found[3]}, {found[4]} {found[5]}); review required")
    return expected_record(gid, found[5])


def apply_to_frame(schedule: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """(copy with reviewed effective times, one record per reviewed game present).
    The input frame is never modified. Duplicated or incomplete identities raise."""
    out = schedule.copy()
    records = []
    if "game_id" not in out.columns:
        raise KickoffReviewError("schedule has no game_id column; review required")
    required = {"away_team", "home_team", "season", "week", "gameday", "gametime"}
    for gid in REVIEWED:
        rows = out[out["game_id"].astype(str) == gid]
        if rows.empty:
            continue
        if len(rows) != 1 or not required.issubset(out.columns):
            raise KickoffReviewError(f"reviewed kickoff identity incomplete or duplicated: {gid}; review required")
        row = rows.iloc[0]
        rec = resolve(gid, season=row["season"], week=row["week"], away_team=row["away_team"],
                      home_team=row["home_team"], gameday=row["gameday"], gametime=row["gametime"])
        if rec["status"] == APPLIED:
            out.loc[rows.index, "gametime"] = rec["effective_gametime"]
        records.append(rec)
    return out, records


def known_to_block(block: dict, game_id: str) -> bool:
    """Whether a (checked) review block could have recorded this reviewed game:
    version 1 blocks were written before the Week 12 change and know only Week 11."""
    return block["review_version"] >= 2 or REVIEWED[game_id].authority == V1_AUTHORITY


def _authorities(records) -> list[str]:
    return sorted({rec["authority"] for rec in records})


def review_block(records: list[dict]) -> dict:
    return {"review_version": REVIEW_VERSION, "authorities": _authorities(records), "games": list(records)}


def check_review_block(block) -> dict:
    """{game_id: record} for a well-formed review block; raises otherwise. Every
    record must equal the one this module derives from its source time, including
    its game's own authority. Version 1 blocks (Week 11 only) stay valid as written."""
    if not isinstance(block, dict):
        raise KickoffReviewError("kickoff_review fields")
    version = block.get("review_version")
    if isinstance(version, bool) or version not in _BLOCK_FIELDS:
        raise KickoffReviewError(f"unsupported kickoff review_version {version!r}")
    if set(block) != _BLOCK_FIELDS[version]:
        raise KickoffReviewError("kickoff_review fields")
    if version == 1 and block["authority"] != V1_AUTHORITY:
        raise KickoffReviewError("kickoff_review authority differs")
    if not isinstance(block["games"], list):
        raise KickoffReviewError("kickoff_review games must be a list")
    out = {}
    for i, rec in enumerate(block["games"]):
        if not isinstance(rec, dict) or set(rec) != set(RECORD_FIELDS):
            raise KickoffReviewError(f"kickoff_review games[{i}] fields")
        gid = rec["game_id"]
        if not isinstance(gid, str) or gid not in REVIEWED:
            raise KickoffReviewError(f"kickoff_review games[{i}]: {gid!r} is not a reviewed game")
        if gid in out:
            raise KickoffReviewError(f"kickoff_review lists {gid} twice")
        r = REVIEWED[gid]
        if version == 1 and r.authority != V1_AUTHORITY:
            raise KickoffReviewError(f"kickoff_review games[{i}]: {gid} isn't a version 1 (Week 11) game")
        if rec["source_gametime"] not in (r.old_gametime, r.effective_gametime):
            raise KickoffReviewError(f"kickoff_review {gid}: source time {rec['source_gametime']!r} isn't reviewed")
        if rec != expected_record(gid, rec["source_gametime"]):
            raise KickoffReviewError(f"kickoff_review {gid}: record differs from the reviewed correction")
        out[gid] = rec
    if version == 2 and block["authorities"] != _authorities(out.values()):
        raise KickoffReviewError("kickoff_review authority list differs from its records")
    return out
