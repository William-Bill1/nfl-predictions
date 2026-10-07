"""Actual-bet journal: single-game spread wagers placed at Ontario sportsbooks.

Records what *you* bet, on the terms the sportsbook accepted. It is separate
from the model's simulated recommendations (betting_recommendations_log.csv),
from sportsbook observations (data_files/ontario_spreads/) and from frozen
predictions, and it never places a bet or creates a recommendation.

Storage: data_files/bet_journal/, one immutable JSON file per record, written
once (temp file + exclusive link) and never edited or deleted. Five kinds:

* wager        - the accepted terms of one placed wager;
* amendment    - corrected terms; links to the record it replaces (previous);
* void         - marks a wager void; links to its effective terms record;
* grade        - a settlement of the effective terms against conservative
                 evidence of completion, with provenance;
* invalidation - withdraws a grade the schedule no longer supports.

History is explicit. Amendments and the void form one chain per wager
(wager -> amendment -> ... [-> void]) through `previous`; grades and
invalidations form a second chain through `supersedes`. A save must link to
the current tip (checked under the journal lock), and loading rejects forks,
cycles, dangling links, duplicate voids and out-of-order timestamps. Nothing
is ever resolved by timestamp or file order.

Every record carries schema_version, kind, a stable record_id, recorded_at,
code_revision and payload_sha256 (SHA-256 of the canonical record without that
field). The checksum is unkeyed: it detects accidental damage, not a deliberate
edit followed by recomputing it.

The journal is personal data, so data_files/bet_journal/ is git-ignored: it
stays on this machine (back it up yourself) and isn't deployed with the app.

Money is exact decimal CAD, stored as strings (e.g. "25.00").
See docs/BET_JOURNAL.md.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

import betting_log
import ontario_spreads as on
import pregame_snapshots as ps

ROOT = Path(__file__).resolve().parent
JOURNAL_DIR = ROOT / "data_files" / "bet_journal"
SCHEDULE_PATH = ROOT / "data_files" / ps.SCHEDULE_NAME

SCHEMA_VERSION = 1
KIND_WAGER = "bet_journal_wager"
KIND_AMENDMENT = "bet_journal_amendment"
KIND_VOID = "bet_journal_void"
KIND_GRADE = "bet_journal_grade"
KIND_INVALIDATION = "bet_journal_invalidation"
SETTLEMENT_KINDS = (KIND_GRADE, KIND_INVALIDATION)
PREFIX = {KIND_WAGER: "W", KIND_AMENDMENT: "A", KIND_VOID: "V", KIND_GRADE: "G",
          KIND_INVALIDATION: "X"}
CHECKSUM_FIELD = ps.CHECKSUM_FIELD
CENT = Decimal("0.01")
MAX_STAKE = Decimal("100000.00")
MAX_TEXT = 200

# Ontario-licensed sportsbooks offered in the form. "other" requires a name.
SPORTSBOOKS = {b.key: b.title for b in on.ONTARIO_BOOKS + (on.BET99,)}
SPORTSBOOKS["fanduel_on"] = "FanDuel (CA - ON)"
SPORTSBOOKS["other_on"] = "Other Ontario-licensed sportsbook"

_ID_RE = re.compile(r"^[WAVGX]-\d{8}T\d{6}Z-[0-9a-f]{12}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_GAME_ID_RE = re.compile(r"^(?P<season>\d{4})_(?P<week>\d{2})_(?P<away>[A-Z]{2,3})_(?P<home>[A-Z]{2,3})$")
_MONEY_RE = re.compile(r"^-?\d+\.\d{2}$")
TERM_FIELDS = ("sportsbook_key", "sportsbook_name", "jurisdiction", "game_id", "season", "week",
               "home_team", "away_team", "kickoff_utc", "team", "team_side", "handicap", "odds",
               "stake_cad", "placed_at", "reference", "note")
# Two wagers are "identical" when all of these match (see docs/BET_JOURNAL.md).
IDENTITY_FIELDS = ("sportsbook_key", "sportsbook_name", "game_id", "team", "handicap", "odds",
                   "stake_cad", "placed_at")
RESULTS = ("win", "loss", "push")
EVIDENCE_FIELDS = ("gameday", "home_score", "away_score", "result", "total", "overtime")


class JournalError(ValueError):
    """Invalid input or a refused write."""


class DuplicateWager(JournalError):
    """The same wager appears to be recorded already."""


class JournalBusy(JournalError):
    """Another journal write holds the lock."""


class IntegrityError(RuntimeError):
    """A stored journal record is unreadable or fails validation."""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def now_utc() -> datetime:
    return ps.utc_now()


def iso(dt: datetime) -> str:
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="microseconds" if dt.microsecond else "seconds").replace(
        "+00:00", "Z")


def money(value) -> Decimal:
    """Exact CAD amount with two decimals; rejects anything else."""
    try:
        d = Decimal(str(value).strip().removeprefix("$").strip())
    except (InvalidOperation, AttributeError):
        raise JournalError(f"{value!r} is not an amount") from None
    if not d.is_finite() or d != d.quantize(CENT):
        raise JournalError(f"{value!r} must be dollars and cents (at most 2 decimals)")
    return d.quantize(CENT)


def win_profit(odds: int, stake: Decimal) -> Decimal:
    """Net winnings on a win at American `odds` for `stake`, rounded to the cent."""
    ratio = Decimal(100) / Decimal(-odds) if odds < 0 else Decimal(odds) / Decimal(100)
    return (stake * ratio).quantize(CENT, rounding=ROUND_HALF_UP)


def settle(terms: dict, home_score: float, away_score: float) -> tuple[str, Decimal]:
    """(result, net profit in CAD) for the recorded team and handicap at the
    accepted odds and stake. Uses the reviewed betting_log.spread_result."""
    result = betting_log.spread_result(terms["team"], terms["home_team"], terms["away_team"],
                                       terms["handicap"], home_score, away_score)
    stake = money(terms["stake_cad"])
    profit = win_profit(int(terms["odds"]), stake) if result == "win" else \
        (-stake if result == "loss" else Decimal("0.00"))
    return result, profit


def _seal(doc: dict) -> dict:
    doc[CHECKSUM_FIELD] = ps.payload_checksum(doc)
    return doc


def _new_id(kind: str, at: datetime) -> str:
    return f"{PREFIX[kind]}-{at.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(6)}"


def terms_sha256(terms: dict) -> str:
    return ps.sha256_bytes(ps._canonical(terms).encode())


# --------------------------------------------------------------------------
# Validation of single records
# --------------------------------------------------------------------------

def _need(cond, where, msg):
    if not cond:
        raise IntegrityError(f"{where}: {msg}")


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_id(v, kinds) -> bool:
    return isinstance(v, str) and bool(_ID_RE.match(v)) and v[0] in {PREFIX[k] for k in kinds}


def _check_terms(t: dict, where: str) -> None:
    _need(isinstance(t, dict) and set(t) == set(TERM_FIELDS), where, "terms fields differ from schema")
    _need(t["sportsbook_key"] in SPORTSBOOKS, where, "unknown sportsbook")
    _need(isinstance(t["sportsbook_name"], str) and t["sportsbook_name"].strip(), where,
          "sportsbook name")
    _need(t["jurisdiction"] == "CA-ON", where, "jurisdiction must be CA-ON")
    m = _GAME_ID_RE.match(t["game_id"]) if isinstance(t["game_id"], str) else None
    _need(m is not None, where, "game_id")
    _need(_is_int(t["season"]) and 1999 <= t["season"] <= 2100 and str(t["season"]) == m["season"],
          where, "season must be an integer matching game_id")
    _need(_is_int(t["week"]) and 1 <= t["week"] <= 25 and t["week"] == int(m["week"]), where,
          "week must be an integer matching game_id")
    _need((t["away_team"], t["home_team"]) == (m["away"], m["home"]), where,
          "teams don't match game_id")
    _need(t["team"] in (t["home_team"], t["away_team"]), where, "team not in game")
    _need(t["team_side"] == ("home" if t["team"] == t["home_team"] else "away"), where, "team side")
    _need(on.valid_handicap(t["handicap"]) and isinstance(t["handicap"], float), where, "handicap")
    _need(_is_int(t["odds"]) and on.valid_american(t["odds"]), where, "odds")
    _need(isinstance(t["stake_cad"], str) and _MONEY_RE.match(t["stake_cad"])
          and Decimal("0") < Decimal(t["stake_cad"]) <= MAX_STAKE, where, "stake")
    placed, kickoff = _ts(t["placed_at"], where), _ts(t["kickoff_utc"], where)
    _need(placed < kickoff, where, "placed at or after kickoff")
    for f in ("reference", "note"):
        _need(t[f] is None or (isinstance(t[f], str) and 0 < len(t[f]) <= MAX_TEXT), where, f)


def _ts(text, where) -> datetime:
    try:
        return on.parse_utc(text)
    except ValueError as exc:
        raise IntegrityError(f"{where}: {exc}") from None


def _check_schedule(s, where):
    _need(isinstance(s, dict) and set(s) == {"path", "sha256"} and s["path"] == ps.SCHEDULE_NAME
          and isinstance(s["sha256"], str) and _SHA_RE.match(s["sha256"]), where,
          "schedule provenance")


def _check_reason(r, where):
    _need(isinstance(r, str) and r.strip() and len(r) <= MAX_TEXT, where, "reason required")


def _check_confirmed(c, where):
    _need(isinstance(c, list) and all(_is_id(x, (KIND_WAGER,)) for x in c)
          and c == sorted(set(c)), where, "confirmed_existing must be sorted, unique wager IDs")


def check_evidence(ev, where="evidence") -> tuple[int, int]:
    """Validate stored completion evidence; returns (home_score, away_score)."""
    _need(isinstance(ev, dict) and set(ev) == set(EVIDENCE_FIELDS), where,
          "evidence fields differ from schema")
    for f in ("home_score", "away_score", "result", "total", "overtime"):
        _need(_is_int(ev[f]), where, f"{f} must be an integer")
    h, a = ev["home_score"], ev["away_score"]
    _need(h >= 0 and a >= 0, where, "negative score")
    _need((h, a) != (0, 0), where, "0-0 is a placeholder, not a final score")
    _need(ev["result"] == h - a and ev["total"] == h + a, where,
          "result/total inconsistent with the scores")
    _need(ev["overtime"] in (0, 1), where, "overtime must be 0 or 1")
    _need(isinstance(ev["gameday"], str) and _DATE_RE.match(ev["gameday"]), where, "gameday")
    return h, a


_FIELDS = {
    KIND_WAGER: ("terms", "confirmed_existing", "schedule"),
    KIND_AMENDMENT: ("amends", "previous", "reason", "terms", "confirmed_existing", "schedule"),
    KIND_VOID: ("voids", "previous", "reason"),
    KIND_GRADE: ("wager_id", "terms_record_id", "terms_sha256", "supersedes", "result",
                 "profit_cad", "evidence", "schedule", "reason"),
    KIND_INVALIDATION: ("wager_id", "terms_record_id", "supersedes", "schedule", "reason"),
}
_COMMON = ("schema_version", "kind", "record_id", "recorded_at", "code_revision", CHECKSUM_FIELD)


def validate_record(doc, where: str = "record") -> None:
    """Schema, checksum and value checks for one record on its own. Links
    between records are checked by current_state()."""
    _need(isinstance(doc, dict), where, "not an object")
    kind = doc.get("kind")
    _need(kind in _FIELDS, where, f"unknown kind {kind!r}")
    _need(set(doc) == set(_COMMON + _FIELDS[kind]), where,
          f"fields differ from schema: {sorted(set(doc) ^ set(_COMMON + _FIELDS[kind]))}")
    _need(doc["schema_version"] == SCHEMA_VERSION, where, "unsupported schema_version")
    try:
        ok = doc[CHECKSUM_FIELD] == ps.payload_checksum(doc)
    except ValueError:                          # NaN/Infinity can't be canonical JSON
        ok = False
    _need(ok, where, "checksum mismatch or non-finite number")
    _need(_is_id(doc["record_id"], (kind,)), where, "bad record_id")
    recorded = _ts(doc["recorded_at"], where)
    _need(doc["record_id"][2:18] == f"{recorded:%Y%m%dT%H%M%SZ}", where,
          "record_id doesn't match recorded_at")
    _need(isinstance(doc["code_revision"], str) and doc["code_revision"], where, "code_revision")
    if kind in (KIND_WAGER, KIND_AMENDMENT):
        _check_terms(doc["terms"], where)
        _need(_ts(doc["terms"]["placed_at"], where) <= recorded, where, "placed after recorded")
        _check_confirmed(doc["confirmed_existing"], where)
        _check_schedule(doc["schedule"], where)
    if kind == KIND_AMENDMENT:
        _need(_is_id(doc["amends"], (KIND_WAGER,)), where, "amends must name a wager")
        _need(_is_id(doc["previous"], (KIND_WAGER, KIND_AMENDMENT)), where,
              "previous must name the wager or an amendment")
        _need(doc["amends"] not in doc["confirmed_existing"], where, "confirms itself")
        _check_reason(doc["reason"], where)
    if kind == KIND_WAGER:
        _need(doc["record_id"] not in doc["confirmed_existing"], where, "confirms itself")
    if kind == KIND_VOID:
        _need(_is_id(doc["voids"], (KIND_WAGER,)), where, "voids must name a wager")
        _need(_is_id(doc["previous"], (KIND_WAGER, KIND_AMENDMENT)), where,
              "previous must name the wager or an amendment")
        _check_reason(doc["reason"], where)
    if kind in (KIND_GRADE, KIND_INVALIDATION):
        _need(_is_id(doc["wager_id"], (KIND_WAGER,)), where, "wager_id")
        _need(_is_id(doc["terms_record_id"], (KIND_WAGER, KIND_AMENDMENT)), where,
              "terms_record_id")
        _check_schedule(doc["schedule"], where)
        _check_reason(doc["reason"], where)
    if kind == KIND_GRADE:
        _need(doc["supersedes"] is None or _is_id(doc["supersedes"], SETTLEMENT_KINDS), where,
              "supersedes must be null or a grade/invalidation")
        _need(isinstance(doc["terms_sha256"], str) and _SHA_RE.match(doc["terms_sha256"]),
              where, "terms_sha256")
        _need(doc["result"] in RESULTS, where, "result")
        _need(isinstance(doc["profit_cad"], str) and _MONEY_RE.match(doc["profit_cad"]), where,
              "profit")
        check_evidence(doc["evidence"], where)
    if kind == KIND_INVALIDATION:
        _need(_is_id(doc["supersedes"], (KIND_GRADE,)), where, "supersedes must name a grade")


# --------------------------------------------------------------------------
# Loading and the history graph
# --------------------------------------------------------------------------

def load_records(journal_dir: Path = JOURNAL_DIR) -> list[dict]:
    """Every record, validated on its own. Any unreadable or invalid file
    raises IntegrityError naming it - it never silently drops out."""
    out = []
    d = Path(journal_dir)
    for path in sorted(d.iterdir()) if d.exists() else []:
        if path.name.startswith("."):
            continue        # the lock file and in-flight temp files, never records
        if path.suffix != ".json" or not _ID_RE.match(path.stem) or not path.is_file():
            raise IntegrityError(f"{path.name}: unexpected entry in the journal directory")
        try:
            doc = json.loads(path.read_text(encoding="utf-8"), parse_constant=_no_constant)
        except (OSError, ValueError) as exc:
            raise IntegrityError(f"{path.name}: unreadable ({exc})") from None
        validate_record(doc, path.name)
        _need(path.stem == doc["record_id"], path.name, "file name doesn't match record_id")
        out.append({**doc, "_file": path.name})
    return out


def _no_constant(name):
    raise ValueError(f"non-finite number {name}")


@dataclass
class WagerState:
    wager: dict
    terms: dict                   # effective terms (end of the terms chain)
    terms_record_id: str          # the wager or amendment holding them
    amendments: list              # in chain order
    void: dict | None
    settlements: list             # grades and invalidations, in chain order
    grade: dict | None            # the chain tip, if it is a grade of the effective terms

    @property
    def wager_id(self) -> str:
        return self.wager["record_id"]

    @property
    def tip(self) -> dict | None:
        return self.settlements[-1] if self.settlements else None

    @property
    def status(self) -> str:
        """void / win / loss / push / unverified (the grade of these terms was
        invalidated) / pending (never graded, or graded on earlier terms)."""
        if self.void:
            return "void"
        if self.grade:
            return self.grade["result"]
        tip = self.tip
        if tip and tip["kind"] == KIND_INVALIDATION and tip["terms_record_id"] == self.terms_record_id:
            return "unverified"
        return "pending"

    @property
    def status_note(self) -> str:
        tip = self.tip
        if self.void or self.grade or tip is None:
            return ""
        if self.status == "unverified":
            return f"earlier grade {tip['supersedes']} invalidated: {tip['reason']}"
        return "terms amended since the last grade; grade again"


def _chain(root_id: str, children: dict, members: set, where: str) -> list[str]:
    """Walk single-child links from root; any fork, or a member the walk
    doesn't reach (missing link or cycle), is an integrity error."""
    order, node, seen = [], root_id, set()
    while True:
        kids = children.get(node, [])
        _need(len(kids) <= 1, where, f"history forks after {node}: {', '.join(sorted(kids))}")
        if not kids:
            break
        node = kids[0]
        _need(node not in seen, where, f"cycle at {node}")
        seen.add(node)
        order.append(node)
    stray = members - set(order)
    _need(not stray, where, f"records not on the chain (missing or circular links): "
                            f"{', '.join(sorted(stray))}")
    return order


def current_state(records: list[dict]) -> dict[str, WagerState]:
    """wager_id -> current state, from explicit links only (never from
    timestamps or file order). Forks, cycles, dangling or cross-wager links,
    duplicate voids, out-of-order timestamps, and grades that don't match
    their terms or evidence all raise IntegrityError."""
    by_id = {r["record_id"]: r for r in records}
    wagers = {k: r for k, r in by_id.items() if r["kind"] == KIND_WAGER}

    def ref(r, field, kinds):
        target = by_id.get(r[field])
        _need(target is not None, r["_file"], f"{field} {r[field]} doesn't exist")
        _need(target["kind"] in kinds, r["_file"], f"{field} {r[field]} has the wrong kind")
        _need(_ts(target["recorded_at"], r["_file"]) <= _ts(r["recorded_at"], r["_file"]),
              r["_file"], f"recorded before its {field} {r[field]}")
        return target

    owner = {}
    for r in records:
        k = r["kind"]
        if k in (KIND_WAGER, KIND_AMENDMENT):
            for c in r["confirmed_existing"]:
                ref({**r, "_c": c}, "_c", (KIND_WAGER,))
        if k == KIND_AMENDMENT:
            ref(r, "amends", (KIND_WAGER,))
            owner[r["record_id"]] = r["amends"]
        elif k == KIND_VOID:
            ref(r, "voids", (KIND_WAGER,))
            owner[r["record_id"]] = r["voids"]
        elif k in SETTLEMENT_KINDS:
            ref(r, "wager_id", (KIND_WAGER,))
            ref(r, "terms_record_id", (KIND_WAGER, KIND_AMENDMENT))
            if r["supersedes"]:
                ref(r, "supersedes", SETTLEMENT_KINDS if k == KIND_GRADE else (KIND_GRADE,))
    for r in records:
        if r["kind"] in (KIND_AMENDMENT, KIND_VOID):
            prev = ref(r, "previous", (KIND_WAGER, KIND_AMENDMENT))
            _need(owner.get(prev["record_id"], prev["record_id"]) == owner[r["record_id"]],
                  r["_file"], "previous belongs to another wager")

    states = {}
    for wid, w in sorted(wagers.items()):
        where = f"history of wager {wid}"
        mods = {r["record_id"]: r for r in records
                if r["kind"] in (KIND_AMENDMENT, KIND_VOID) and owner[r["record_id"]] == wid}
        voids = [r for r in mods.values() if r["kind"] == KIND_VOID]
        _need(len(voids) <= 1, where, f"voided more than once: "
                                      f"{', '.join(sorted(v['record_id'] for v in voids))}")
        children = {}
        for r in mods.values():
            children.setdefault(r["previous"], []).append(r["record_id"])
        chain = [by_id[x] for x in _chain(wid, children, set(mods), where)]
        _need(all(r["kind"] == KIND_AMENDMENT for r in chain[:-1]), where,
              "a record follows the void")
        void = chain[-1] if chain and chain[-1]["kind"] == KIND_VOID else None
        amendments = [r for r in chain if r["kind"] == KIND_AMENDMENT]
        terms_chain = [w, *amendments]
        terms_records = {r["record_id"]: r for r in terms_chain}
        effective = amendments[-1] if amendments else w
        for a in amendments:
            _need(a["terms"]["game_id"] == w["terms"]["game_id"], a["_file"],
                  "an amendment can't change the game")

        sets = {r["record_id"]: r for r in records
                if r["kind"] in SETTLEMENT_KINDS and r["wager_id"] == wid}
        roots = [r for r in sets.values() if r["supersedes"] is None]
        _need(len(roots) <= 1 and (roots or not sets), where,
              "settlement history needs exactly one first grade")
        settlements = []
        if roots:
            schildren = {}
            for r in sets.values():
                if r["supersedes"]:
                    schildren.setdefault(r["supersedes"], []).append(r["record_id"])
            rest = _chain(roots[0]["record_id"], schildren, set(sets) - {roots[0]["record_id"]},
                          where)
            settlements = [roots[0]] + [by_id[x] for x in rest]
        for s in settlements:
            if void:
                _need(_ts(s["recorded_at"], s["_file"]) < _ts(void["recorded_at"], void["_file"]),
                    s["_file"], "settlement is not recorded strictly before void")
            t = terms_records.get(s["terms_record_id"])
            _need(t is not None, s["_file"], "terms_record_id isn't in this wager's terms chain")
            terms_index = next(i for i, r in enumerate(terms_chain)
                         if r["record_id"] == s["terms_record_id"])
            settled_at = _ts(s["recorded_at"], s["_file"])
            _need(not any(_ts(r["recorded_at"], r["_file"]) < settled_at
                      for r in terms_chain[terms_index + 1:]), s["_file"],
                "settlement recorded after its terms were superseded")
            if s["kind"] == KIND_GRADE:
                _need(s["terms_sha256"] == terms_sha256(t["terms"]), s["_file"],
                      "terms_sha256 doesn't match the referenced terms")
                h, a = check_evidence(s["evidence"], s["_file"])
                result, profit = settle(t["terms"], h, a)
                _need((result, f"{profit:.2f}") == (s["result"], s["profit_cad"]), s["_file"],
                      "result/profit don't follow from the terms and scores")
            else:
                _need(by_id[s["supersedes"]]["terms_record_id"] == s["terms_record_id"],
                      s["_file"], "invalidation names other terms than the grade it voids")
        tip = settlements[-1] if settlements else None
        current = tip if tip and tip["kind"] == KIND_GRADE and \
            tip["terms_record_id"] == effective["record_id"] else None
        states[wid] = WagerState(w, effective["terms"], effective["record_id"], amendments,
                                 void, settlements, current)

    for r in records:
        if r["kind"] not in (KIND_WAGER, KIND_AMENDMENT):
            continue
        where, terms = r["_file"], r["terms"]
        at = _ts(r["recorded_at"], where)
        exclude = r["record_id"] if r["kind"] == KIND_WAGER else r["amends"]
        confirmed = set(r["confirmed_existing"])
        expected = []
        for s in states.values():
            wager_at = _ts(s.wager["recorded_at"], where)
            if (s.wager_id == exclude or wager_at > at
                    or (wager_at == at and s.wager_id not in confirmed)):
                continue
            if s.void and _ts(s.void["recorded_at"], where) <= at:
                continue
            other_terms = s.wager["terms"]
            for amendment in s.amendments:
                if _ts(amendment["recorded_at"], where) <= at:
                    other_terms = amendment["terms"]
                else:
                    break
            same_book = (terms["sportsbook_key"], terms["sportsbook_name"], terms["reference"])
            other_book = (other_terms["sportsbook_key"], other_terms["sportsbook_name"],
                          other_terms["reference"])
            if terms["reference"] and same_book == other_book:
                _need(False, where, "confirmed wager reuses an existing sportsbook reference")
            if (tuple(terms[f] for f in IDENTITY_FIELDS) ==
                    tuple(other_terms[f] for f in IDENTITY_FIELDS)
                    and _refs_compatible(terms, other_terms)):
                expected.append(s.wager_id)
        _need(sorted(expected) == r["confirmed_existing"], where,
              "confirmed_existing doesn't match the active duplicate wagers at record time")
    return states


def totals(states: dict[str, WagerState]) -> dict:
    """Current (non-void) wagers only. ROI = net profit / stake of graded
    wagers (wins, losses and pushes). Pending and unverified stake is shown
    separately and excluded; voided wagers are excluded entirely."""
    live = [s for s in states.values() if not s.void]
    graded = [s for s in live if s.grade]
    stake_graded = sum((money(s.terms["stake_cad"]) for s in graded), Decimal("0.00"))
    profit = sum((money(s.grade["profit_cad"]) for s in graded), Decimal("0.00"))
    counts = {k: sum(1 for s in graded if s.grade["result"] == k) for k in RESULTS}
    return {
        "wagers": len(live), "graded": len(graded), "pending": len(live) - len(graded),
        "unverified": sum(1 for s in live if s.status == "unverified"),
        "voided": sum(1 for s in states.values() if s.void), **counts,
        "stake_graded_cad": stake_graded,
        "stake_pending_cad": sum((money(s.terms["stake_cad"]) for s in live if not s.grade),
                                 Decimal("0.00")),
        "net_profit_cad": profit,
        "roi_pct": None if not stake_graded else (profit / stake_graded * 100).quantize(CENT),
    }


# --------------------------------------------------------------------------
# Building records
# --------------------------------------------------------------------------

def _schedule_game(game_id: str, schedule_path: Path) -> tuple[dict, str]:
    schedule, sha = on.load_schedule(schedule_path)
    rows = schedule[schedule["game_id"] == game_id]
    if len(rows) != 1:
        raise JournalError(f"game {game_id!r} not found in the schedule")
    g = rows.iloc[0]
    kickoff, problem = ps.kickoff_utc(g["gameday"], g["gametime"])
    if kickoff is None:
        raise JournalError(f"game {game_id} has no usable kickoff ({problem})")
    return {"game_id": game_id, "season": int(g["season"]), "week": int(g["week"]),
            "home_team": g["home_team"], "away_team": g["away_team"], "kickoff": kickoff}, sha


def build_terms(*, sportsbook_key: str, sportsbook_name: str | None, game_id: str, team: str,
                handicap, odds, stake, placed_at: datetime, reference: str | None,
                note: str | None, now: datetime, schedule_path: Path) -> tuple[dict, str]:
    """Validated accepted terms of a wager, plus the schedule's sha256.

    placed_at must be timezone-aware, not in the future (vs `now`) and
    strictly before kickoff. Recording after kickoff (a late entry of a
    pregame wager) is allowed."""
    if sportsbook_key not in SPORTSBOOKS:
        raise JournalError(f"unknown sportsbook {sportsbook_key!r}")
    name = (sportsbook_name or "").strip() if sportsbook_key == "other_on" else \
        SPORTSBOOKS[sportsbook_key]
    if not name:
        raise JournalError("name the Ontario sportsbook")
    if placed_at.tzinfo is None:
        raise JournalError("placement time needs a timezone")
    placed = placed_at.astimezone(timezone.utc).replace(second=0, microsecond=0)
    if placed > now:
        raise JournalError("the placement time is in the future")
    game, sha = _schedule_game(game_id, schedule_path)
    if placed >= game["kickoff"]:
        raise JournalError(f"placed at {iso(placed)}, not before kickoff {iso(game['kickoff'])}; "
                           "only pregame wagers are recorded")
    team = str(team).strip().upper()
    if team not in (game["home_team"], game["away_team"]):
        raise JournalError(f"team {team} is not playing in {game_id}")
    try:
        handicap = float(handicap) + 0.0
    except (TypeError, ValueError):
        raise JournalError("enter the spread") from None
    if not on.valid_handicap(handicap):
        raise JournalError(f"spread {handicap:g} isn't a half-point value within 60")
    if odds is None or not on.valid_american(odds):
        raise JournalError("odds must be American odds (e.g. -110, +105)")
    stake_d = money(stake)
    if not (Decimal("0") < stake_d <= MAX_STAKE):
        raise JournalError(f"the stake must be more than $0.00 and at most ${MAX_STAKE}")
    clean = lambda s: (s or "").strip()[:MAX_TEXT] or None  # noqa: E731
    terms = {
        "sportsbook_key": sportsbook_key, "sportsbook_name": name, "jurisdiction": "CA-ON",
        "game_id": game_id, "season": game["season"], "week": game["week"],
        "home_team": game["home_team"], "away_team": game["away_team"],
        "kickoff_utc": iso(game["kickoff"]), "team": team,
        "team_side": "home" if team == game["home_team"] else "away",
        "handicap": handicap, "odds": int(odds), "stake_cad": f"{stake_d:.2f}",
        "placed_at": iso(placed), "reference": clean(reference), "note": clean(note),
    }
    try:
        _check_terms(terms, "terms")
    except IntegrityError as exc:
        raise JournalError(str(exc)) from None
    return terms, sha


def identical_wagers(terms: dict, states: dict[str, WagerState],
                     exclude: str | None = None) -> list[str]:
    """Active wagers (other than `exclude`) whose effective terms have the
    same identity as `terms`."""
    key = tuple(terms[f] for f in IDENTITY_FIELDS)
    return sorted(s.wager_id for s in states.values() if not s.void and s.wager_id != exclude
                  and tuple(s.terms[f] for f in IDENTITY_FIELDS) == key
                  and _refs_compatible(s.terms, terms))


def _refs_compatible(a: dict, b: dict) -> bool:
    """Different, both-present sportsbook references mean different wagers."""
    return not (a["reference"] and b["reference"] and a["reference"] != b["reference"])


def same_reference(terms: dict, states: dict[str, WagerState], exclude: str | None = None) -> list[str]:
    """Active wagers at the same sportsbook with the same non-empty reference."""
    if not terms["reference"]:
        return []
    return sorted(s.wager_id for s in states.values() if not s.void and s.wager_id != exclude
                  and s.terms["sportsbook_key"] == terms["sportsbook_key"]
                  and s.terms["sportsbook_name"] == terms["sportsbook_name"]
                  and s.terms["reference"] == terms["reference"])


def _base(kind: str, now: datetime, repo_dir: Path) -> dict:
    return {"schema_version": SCHEMA_VERSION, "kind": kind, "record_id": _new_id(kind, now),
            "recorded_at": iso(now), "code_revision": ps.code_revision(repo_dir)[0]}


def _finish(doc: dict, what: str) -> dict:
    _seal(doc)
    try:
        validate_record(doc, what)
    except IntegrityError as exc:
        raise JournalError(str(exc)) from None
    return doc


def prepare_wager(terms: dict, schedule_sha: str, *, confirmed_existing: list[str],
                  now: datetime, repo_dir: Path = ROOT) -> dict:
    doc = {**_base(KIND_WAGER, now, repo_dir), "terms": terms,
           "confirmed_existing": sorted(set(confirmed_existing)),
           "schedule": {"path": ps.SCHEDULE_NAME, "sha256": schedule_sha}}
    return _finish(doc, "new wager")


def prepare_amendment(wager_id: str, previous: str, terms: dict, schedule_sha: str, reason: str,
                      *, confirmed_existing: list[str] = (), now: datetime,
                      repo_dir: Path = ROOT) -> dict:
    """`previous` is the wager's effective terms record when the correction
    was previewed; saving refuses if it is no longer current."""
    if not (reason or "").strip():
        raise JournalError("a reason is required")
    doc = {**_base(KIND_AMENDMENT, now, repo_dir), "amends": wager_id, "previous": previous,
           "reason": reason.strip()[:MAX_TEXT], "terms": terms,
           "confirmed_existing": sorted(set(confirmed_existing)),
           "schedule": {"path": ps.SCHEDULE_NAME, "sha256": schedule_sha}}
    return _finish(doc, "new amendment")


def prepare_void(wager_id: str, previous: str, reason: str, *, now: datetime,
                 repo_dir: Path = ROOT) -> dict:
    if not (reason or "").strip():
        raise JournalError("a reason is required")
    doc = {**_base(KIND_VOID, now, repo_dir), "voids": wager_id, "previous": previous,
           "reason": reason.strip()[:MAX_TEXT]}
    return _finish(doc, "new void")


# --------------------------------------------------------------------------
# Writing (append-only, exclusive, locked)
# --------------------------------------------------------------------------

def _write(doc: dict, journal_dir: Path) -> Path:
    journal_dir.mkdir(parents=True, exist_ok=True)
    path = journal_dir / f"{doc['record_id']}.json"
    text = json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n"
    if not ps._exclusive_write(path, text):
        raise JournalError(f"{path.name} already exists; nothing was overwritten")
    return path


@contextmanager
def _lock(journal_dir: Path, run_id: str):
    """The journal-wide write lock (ontario_spreads.SlotLock semantics)."""
    lock = on.SlotLock(Path(journal_dir), "journal_write", run_id, what="bet-journal write")
    try:
        lock.__enter__()
    except on.CaptureError as exc:
        raise JournalBusy(str(exc)) from None
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


def _check_new(records: list[dict], doc: dict) -> None:
    """Refuse a record that would make the history invalid (the loader would
    then reject the whole journal)."""
    try:
        current_state(records + [{**doc, "_file": "new record"}])
    except IntegrityError as exc:
        raise JournalError(f"refused: the record would make the history invalid ({exc})") \
            from None


def save_wager(doc: dict, journal_dir: Path = JOURNAL_DIR) -> Path:
    """Write a new wager. Under the journal lock, re-checks for identical
    wagers: the set found now must equal the set the user confirmed at
    preview (normally none), and a reused sportsbook reference is refused.
    So a rerun, double click or concurrent save can't add an unconfirmed
    duplicate."""
    journal_dir = Path(journal_dir)
    with _lock(journal_dir, doc["record_id"]):
        records = load_records(journal_dir)
        states = current_state(records)
        same_ref = same_reference(doc["terms"], states)
        if same_ref:
            raise DuplicateWager(f"sportsbook reference {doc['terms']['reference']!r} is already "
                                 f"recorded as {', '.join(same_ref)}; amend that wager instead")
        _check_confirmed_set(identical_wagers(doc["terms"], states), doc["confirmed_existing"])
        _check_new(records, doc)
        return _write(doc, journal_dir)


def _check_confirmed_set(found: list[str], confirmed: list[str]) -> None:
    if found == confirmed:
        return
    if not confirmed:
        raise DuplicateWager(
            f"identical wager(s) already recorded: {', '.join(found)}; if this is a "
            "separate wager, confirm that and preview again")
    raise DuplicateWager(
        f"the identical wagers changed since preview (confirmed: {', '.join(confirmed)}; "
        f"now: {', '.join(found) or 'none'}); preview again")


def _current_for(states: dict, doc: dict, wager_id: str) -> WagerState:
    s = states.get(wager_id)
    if s is None:
        raise JournalError(f"wager {wager_id} not found")
    if s.void:
        raise JournalError(f"wager {wager_id} is void; it can't be changed")
    if doc["previous"] != s.terms_record_id:
        raise JournalError(f"wager {wager_id} changed since preview (its current terms are "
                           f"{s.terms_record_id}, not {doc['previous']}); preview again")
    return s


def save_amendment(doc: dict, journal_dir: Path = JOURNAL_DIR) -> Path:
    journal_dir = Path(journal_dir)
    with _lock(journal_dir, doc["record_id"]):
        records = load_records(journal_dir)
        states = current_state(records)
        s = _current_for(states, doc, doc["amends"])
        if doc["terms"]["game_id"] != s.terms["game_id"]:
            raise JournalError("an amendment can't change the game; void the wager and record "
                               "a new one")
        if doc["terms"] == s.terms:
            raise JournalError("the corrected terms are the same as the current terms")
        if same_reference(doc["terms"], states, exclude=doc["amends"]):
            raise DuplicateWager("that sportsbook reference belongs to another wager")
        _check_confirmed_set(identical_wagers(doc["terms"], states, exclude=doc["amends"]),
                             doc["confirmed_existing"])
        _check_new(records, doc)
        return _write(doc, journal_dir)


def save_void(doc: dict, journal_dir: Path = JOURNAL_DIR) -> Path:
    journal_dir = Path(journal_dir)
    with _lock(journal_dir, doc["record_id"]):
        records = load_records(journal_dir)
        states = current_state(records)
        if (s := states.get(doc["voids"])) is not None and s.void:
            raise JournalError(f"wager {doc['voids']} is already void")
        _current_for(states, doc, doc["voids"])
        _check_new(records, doc)
        return _write(doc, journal_dir)


# --------------------------------------------------------------------------
# Grading (explicit action only)
# --------------------------------------------------------------------------

def _int_field(value):
    """A finite, integer-valued schedule number as int, else None."""
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return int(f) if math.isfinite(f) and f.is_integer() else None


def completion_evidence(schedule: pd.DataFrame, game_id: str, now: datetime) -> tuple[dict | None, str]:
    """(evidence, "") when the schedule row conservatively indicates a
    completed game, else (None, why it stays pending).

    nflverse's schedule has no final-status flag. Evidence of completion is:
    exactly one row; both scores finite non-negative integers, not 0-0;
    `result` == home - away and `total` == home + away; `overtime` 0 or 1; and
    the game day strictly before today (America/Toronto). This is
    conservative evidence, not an authoritative final status."""
    rows = schedule[schedule["game_id"] == game_id]
    if len(rows) != 1:
        return None, "game not found" if rows.empty else "game listed more than once"
    g = rows.iloc[0]
    vals = {f: _int_field(g.get(f)) for f in ("home_score", "away_score", "result", "total",
                                               "overtime")}
    missing = [f for f, v in vals.items() if v is None]
    if missing:
        return None, f"no final score yet ({', '.join(missing)} missing or not an integer)"
    h, a = vals["home_score"], vals["away_score"]
    if h < 0 or a < 0:
        return None, "negative score"
    if (h, a) == (0, 0):
        return None, "0-0 placeholder"
    if vals["result"] != h - a or vals["total"] != h + a or vals["overtime"] not in (0, 1):
        return None, "score fields are inconsistent"
    day = str(g.get("gameday", "")).strip()[:10]
    try:
        gameday = date.fromisoformat(day)
    except ValueError:
        return None, "no valid game day"
    if gameday >= now.astimezone(on.TORONTO).date():
        return None, "game day is not before today"
    return {"gameday": day, **vals}, ""


def _settlement(kind: str, s: WagerState, prev: dict | None, schedule_sha: str, reason: str,
                now: datetime, code: str) -> dict:
    doc = {"schema_version": SCHEMA_VERSION, "kind": kind, "record_id": _new_id(kind, now),
           "recorded_at": iso(now), "code_revision": code, "wager_id": s.wager_id,
           "terms_record_id": s.terms_record_id if kind == KIND_GRADE else prev["terms_record_id"],
           "supersedes": prev["record_id"] if prev else None,
           "schedule": {"path": ps.SCHEDULE_NAME, "sha256": schedule_sha}, "reason": reason}
    return doc


def grade(journal_dir: Path = JOURNAL_DIR, schedule_path: Path = SCHEDULE_PATH, *,
          now: datetime | None = None, repo_dir: Path = ROOT) -> list[dict]:
    """Settle every current, non-void wager against the schedule.

    For each wager, compared with the tip of its settlement chain:
    * evidence of completion and no grade of the effective terms -> grade
      ("initial grade", "terms amended..." or "evidence restored");
    * a grade of the effective terms whose evidence changed -> superseding
      grade ("score correction");
    * a grade of the effective terms that the schedule no longer supports
      -> invalidation (status "unverified", out of the totals);
    * otherwise nothing. Running it again with unchanged data writes nothing.
    The schedule hash is of the exact bytes parsed. Returns the new records."""
    now = now or now_utc()
    journal_dir = Path(journal_dir)
    written = []
    with _lock(journal_dir, f"grade-{secrets.token_hex(4)}"):
        states = current_state(load_records(journal_dir))
        schedule, sha = on.load_schedule(schedule_path)
        code = ps.code_revision(repo_dir)[0]
        for s in sorted(states.values(), key=lambda x: x.wager_id):
            if s.void:
                continue
            evidence, why = completion_evidence(schedule, s.terms["game_id"], now)
            tip = s.tip
            if evidence is None:
                if s.grade is None:
                    continue                    # pending, unverified or old-terms grade
                doc = _settlement(KIND_INVALIDATION, s, tip, sha,
                                  f"schedule no longer supports the grade: {why}", now, code)
            else:
                if s.grade is not None and s.grade["evidence"] == evidence:
                    continue
                result, profit = settle(s.terms, evidence["home_score"], evidence["away_score"])
                if tip is None:
                    reason = "initial grade"
                elif s.grade is not None:
                    old = s.grade["evidence"]
                    reason = (f"score correction: {old['away_score']}-{old['home_score']} -> "
                              f"{evidence['away_score']}-{evidence['home_score']} (away-home)"
                              if (old["home_score"], old["away_score"]) !=
                              (evidence["home_score"], evidence["away_score"])
                              else "completion evidence changed")
                elif tip["terms_record_id"] != s.terms_record_id:
                    reason = "terms amended since the last grade"
                else:
                    reason = "evidence restored after invalidation"
                doc = _settlement(KIND_GRADE, s, tip, sha, reason, now, code)
                doc.update(terms_sha256=terms_sha256(s.terms), result=result,
                           profit_cad=f"{profit:.2f}", evidence=evidence)
            _seal(doc)
            validate_record(doc, "new settlement")
            _write(doc, journal_dir)
            written.append(doc)
        if written:
            current_state(load_records(journal_dir))       # the result must load cleanly
    return written


# --------------------------------------------------------------------------
# Toronto time (shared with the page)
# --------------------------------------------------------------------------

def toronto_to_utc(day: date, at: time) -> datetime:
    """America/Toronto wall-clock time -> UTC; rejects DST-ambiguous and
    nonexistent times (reuses ontario_spreads.toronto_local_to_utc)."""
    try:
        return on.toronto_local_to_utc(day, at)
    except on.ValidationError as exc:
        raise JournalError(str(exc)) from None


# --------------------------------------------------------------------------
# Page support (read-only)
# --------------------------------------------------------------------------

def fingerprint(journal_dir: Path = JOURNAL_DIR) -> tuple:
    """Name + SHA-256 of every entry in the journal directory (the page's cache
    key): any new, changed or stray file gives a new fingerprint."""
    d = Path(journal_dir)
    out = []
    for p in sorted(d.iterdir()) if d.exists() else []:
        if p.name.startswith("."):
            continue
        out.append((p.name, hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "dir"))
    return tuple(out)


def selectable_games(schedule: pd.DataFrame, now: datetime, past_days: int = 180,
                     ahead_days: int = 14) -> list[dict]:
    """Games with a usable kickoff from `past_days` ago to `ahead_days` ahead,
    newest first - past games are offered so an earlier pregame wager can be
    recorded late."""
    out = []
    for g in schedule.itertuples(index=False):
        kickoff, _ = ps.kickoff_utc(g.gameday, g.gametime)
        if kickoff is None or not (now - timedelta(days=past_days) <= kickoff
                                   <= now + timedelta(days=ahead_days)):
            continue
        when = kickoff.astimezone(on.TORONTO)
        out.append({"game_id": g.game_id, "season": int(g.season), "week": int(g.week),
                    "away_team": g.away_team, "home_team": g.home_team,
                    "kickoff_utc": iso(kickoff),
                    "label": f"{int(g.season)} wk {int(g.week)}: {team_name(g.away_team)} @ "
                             f"{team_name(g.home_team)} ({when:%a %b} {when.day}, "
                             f"{when:%H:%M %Z})"})
    return sorted(out, key=lambda x: (x["kickoff_utc"], x["game_id"]), reverse=True)


MINUS = "−"


def cad(amount) -> str:
    """'$1,234.50', '-$25.00'."""
    d = Decimal(str(amount))
    return f"{'-' if d < 0 else ''}${abs(d):,.2f}"


def team_name(abbr: str) -> str:
    return on.TEAM_FULL_NAME.get(abbr, abbr)


def toronto_text(ts: str) -> str:
    dt = on.parse_utc(ts).astimezone(on.TORONTO)
    return f"{dt:%Y-%m-%d %H:%M %Z}"


def bet_text(terms: dict) -> str:
    """'Chicago Bears +3.5 at −110' (the bettor-facing line)."""
    h = terms["handicap"]
    hs = "PK" if h == 0 else (f"{h:+.1f}"[:-2] if float(h).is_integer() else f"{h:+.1f}")
    return f"{team_name(terms['team'])} {hs} at {terms['odds']:+d}".replace("-", MINUS)

def current_rows(states: dict[str, WagerState], include_void: bool = False) -> list[dict]:
    rows = []
    for s in sorted(states.values(), key=lambda x: x.terms["placed_at"], reverse=True):
        if s.void and not include_void:
            continue
        t = s.terms
        ev = s.grade["evidence"] if s.grade else None
        rows.append({
            "Wager ID": s.wager_id, "Status": s.status, "Status note": s.status_note,
            "Sportsbook": t["sportsbook_name"],
            "Game": f"{t['season']} wk {t['week']}: {t['away_team']} @ {t['home_team']}",
            "Bet": bet_text(t), "Stake (CAD)": float(t["stake_cad"]),
            "Net profit (CAD)": float(s.grade["profit_cad"]) if s.grade and not s.void else None,
            "Final (away-home)": f"{ev['away_score']}-{ev['home_score']}" if ev else "",
            "Placed (Toronto)": toronto_text(t["placed_at"]),
            "Amended": len(s.amendments), "Reference": t["reference"] or "",
            "Note": t["note"] or "",
        })
    return rows


def _terms_detail(terms: dict) -> str:
    return (f"{terms['sportsbook_name']}: {bet_text(terms)}, ${terms['stake_cad']} CAD, "
            f"placed {toronto_text(terms['placed_at'])}")


_KIND_RANK = {KIND_WAGER: 0, KIND_AMENDMENT: 1, KIND_VOID: 2, KIND_GRADE: 3,
              KIND_INVALIDATION: 3}


def history_rows(records: list[dict], states: dict[str, WagerState] | None = None) -> list[dict]:
    """Every record (the audit trail), oldest first. Within the same second,
    a wager comes before its corrections and settlements, each in chain order.
    The links are shown explicitly; the display order is for reading only."""
    position = {}
    for s in (states or {}).values():
        for i, r in enumerate(s.amendments + ([s.void] if s.void else [])):
            position[r["record_id"]] = i
        for i, r in enumerate(s.settlements):
            position[r["record_id"]] = i
    rows = []
    for r in sorted(records, key=lambda r: (r["recorded_at"], _KIND_RANK[r["kind"]],
                                            position.get(r["record_id"], 0), r["record_id"])):
        k = r["kind"]
        confirmed = (f"; confirmed separate from {', '.join(r['confirmed_existing'])}"
                     if r.get("confirmed_existing") else "")
        if k == KIND_WAGER:
            target, link, detail = r["record_id"], "", _terms_detail(r["terms"]) + confirmed
        elif k == KIND_AMENDMENT:
            target, link = r["amends"], f"after {r['previous']}"
            detail = "terms now " + _terms_detail(r["terms"]) + confirmed
        elif k == KIND_VOID:
            target, link, detail = r["voids"], f"after {r['previous']}", "wager voided"
        elif k == KIND_GRADE:
            ev = r["evidence"]
            target = r["wager_id"]
            link = f"supersedes {r['supersedes']}" if r["supersedes"] else "first settlement"
            detail = (f"{r['result']} {r['profit_cad']} CAD on terms {r['terms_record_id']}, "
                      f"final {ev['away_score']}-{ev['home_score']} (away-home), "
                      f"game day {ev['gameday']}, schedule {r['schedule']['sha256'][:12]}")
        else:
            target, link = r["wager_id"], f"supersedes {r['supersedes']}"
            detail = (f"grade invalidated (terms {r['terms_record_id']}), "
                      f"schedule {r['schedule']['sha256'][:12]}")
        rows.append({"Recorded (UTC)": r["recorded_at"], "Record": k.replace("bet_journal_", ""),
                     "Record ID": r["record_id"], "Wager": target, "Link": link,
                     "Detail": detail, "Reason": r.get("reason") or "",
                     "Code": r["code_revision"][:10]})
    return rows


# --------------------------------------------------------------------------
# CLI: check a journal directory (e.g. after restoring a backup)
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Validate the bet journal (read-only).")
    ap.add_argument("command", choices=["validate"])
    ap.add_argument("--dir", default=str(JOURNAL_DIR), help="journal directory")
    args = ap.parse_args(argv)
    try:
        records = load_records(Path(args.dir))
        states = current_state(records)
    except IntegrityError as exc:
        print(f"INTEGRITY ERROR: {exc}")
        return 1
    t = totals(states)
    print(f"OK: {len(records)} records, {len(states)} wagers ({t['wagers']} current, "
          f"{t['voided']} voided, {t['graded']} graded, {t['unverified']} unverified, "
          f"{t['pending'] - t['unverified']} pending); net {cad(t['net_profit_cad'])} CAD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
