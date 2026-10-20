"""On-demand Excel export of the Bet Journal - a viewing copy.

The JSON records in data_files/bet_journal/ remain the authoritative
journal. This module builds an .xlsx workbook in memory, only when asked,
from one validated read of the journal: no spreadsheet file is kept beside
the journal, editing the workbook never changes the journal, and the
workbook is not a complete backup (it omits checksums and other record
fields).

Every figure comes from bet_journal itself - totals(), current_state()
(statuses, effective terms, grades) and history_rows() (audit order and
links) - so nothing here re-implements settlement, grading or ROI.

Sheets:
  Summary - export time (America/Toronto), currency, the journal totals and
            which wagers enter each, plus the wagers left out of profit/ROI;
  Wagers  - one row per wager on its current effective terms;
  History - every record (wagers, amendments, voids, grades, invalidations).

Identifiers, references, notes and precise timestamps are written as text
cells, never formulas: text beginning with =, +, - or @ stays literal.
"""
from __future__ import annotations

import io
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import bet_journal as bj
import ontario_spreads as on

MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
CAD_FMT = '"$"#,##0.00;-"$"#,##0.00'
PCT_FMT = "0.00%"
HANDICAP_FMT = '+0.0;-0.0;"PK"'          # numeric, sign always shown
ODDS_FMT = "+0;-0"
INT_FMT = "0"
FORMULA_LEADS = ("=", "+", "-", "@")

NOTICE = ("Viewing export only. The JSON records in data_files/bet_journal/ are the "
          "authoritative journal. Editing this workbook never changes the journal, and it "
          "is not a complete backup (checksums and other record fields are omitted) - back "
          "up the JSON files themselves.")

_HEADER_FONT = Font(bold=True)
_HEADER_FILL = PatternFill("solid", fgColor="DDE5F0")
_TITLE_FONT = Font(bold=True, size=13)


class ExportError(RuntimeError):
    """The journal couldn't be read consistently for export."""


# --------------------------------------------------------------------------
# One consistent, validated view of the journal
# --------------------------------------------------------------------------

def read_journal(journal_dir: Path = None) -> tuple[list[dict], dict, tuple]:
    """(records, states, fingerprint) from a single validated read.

    load_records() validates every record and current_state() every link;
    any problem raises bet_journal.IntegrityError and nothing is exported.
    If files change while being read (a concurrent save), the read is
    repeated, so the three sheets always describe the same journal."""
    journal_dir = Path(journal_dir or bj.JOURNAL_DIR)
    for _ in range(3):
        try:
            before = bj.fingerprint(journal_dir)
            records = bj.load_records(journal_dir)
            states = bj.current_state(records)
            after = bj.fingerprint(journal_dir)
        except OSError as exc:
            raise bj.IntegrityError(f"journal can't be read: {exc}") from None
        if before == after:
            return records, states, after
    raise ExportError("the journal kept changing while it was being read; try again")


def file_name(exported_at: datetime) -> str:
    """bet_journal_YYYYMMDD_HHMMSS.xlsx, in America/Toronto time."""
    return f"bet_journal_{exported_at.astimezone(on.TORONTO):%Y%m%d_%H%M%S}.xlsx"


def export(journal_dir: Path = None, now: datetime | None = None) -> dict:
    """Build the workbook now, in memory. Returns
    {data, file_name, exported_at, fingerprint, records, wagers}."""
    now = now or bj.now_utc()
    records, states, fp = read_journal(journal_dir)
    return {"data": build_workbook(records, states, now), "file_name": file_name(now),
            "exported_at": now, "fingerprint": fp, "records": len(records),
            "wagers": len(states)}


# --------------------------------------------------------------------------
# Cell writing
# --------------------------------------------------------------------------

def _text(value: str) -> str:
    """XML can't hold control characters; replace them (visibly) rather than
    fail. Everything else is kept literally."""
    return ILLEGAL_CHARACTERS_RE.sub("�", value)


def _put(ws, row: int, col: int, value, fmt: str | None = None):
    cell = ws.cell(row=row, column=col)
    if value is None:
        return cell
    if isinstance(value, str):
        cell.value = _text(value)
        cell.data_type = "s"                     # literal text, never a formula
        if value.startswith(FORMULA_LEADS):
            cell.quotePrefix = True              # Excel keeps it as text when edited
    elif isinstance(value, Decimal):
        cell.value = float(value)
    else:
        cell.value = value
    if fmt:
        cell.number_format = fmt
    return cell


def _table(ws, columns: list[tuple[str, str | None]], rows: list[list], *,
           first_row: int = 1, wrap: tuple[str, ...] = ()) -> None:
    """Header + rows; freeze below the header, filter, readable widths."""
    for c, (name, _) in enumerate(columns, 1):
        cell = _put(ws, first_row, c, name)
        cell.font, cell.fill = _HEADER_FONT, _HEADER_FILL
        cell.alignment = Alignment(vertical="top", wrap_text=True)
    for r, values in enumerate(rows, first_row + 1):
        for c, ((name, fmt), value) in enumerate(zip(columns, values), 1):
            cell = _put(ws, r, c, value, fmt)
            cell.alignment = Alignment(vertical="top", wrap_text=name in wrap)
    last = first_row + max(len(rows), 1)
    ws.freeze_panes = ws.cell(row=first_row + 1, column=1)
    ws.auto_filter.ref = f"A{first_row}:{get_column_letter(len(columns))}{last}"
    for c, (name, fmt) in enumerate(columns, 1):
        # +3 for the header's filter arrow
        lengths = [len(name) + 3] + [len(str(v)) for v in (row[c - 1] for row in rows)
                                     if v is not None]
        width = max(lengths) + 2
        if fmt == CAD_FMT:
            width = max(width, 13)
        cap = 60 if name in wrap else 48
        ws.column_dimensions[get_column_letter(c)].width = min(max(width, 9), cap)


def _toronto(ts: str) -> str:
    dt = bj.parse_ts(ts).astimezone(on.TORONTO)
    return f"{dt:%Y-%m-%d %H:%M:%S %Z}"


# --------------------------------------------------------------------------
# Sheets
# --------------------------------------------------------------------------

def _verification(s: bj.WagerState) -> tuple[str, str]:
    """(grade verification, exclusion / status note) from the journal's own
    status and status_note."""
    if s.void:
        return ("not applicable: voided",
                f"voided by {s.void['record_id']}: {s.void['reason']} (excluded from all totals)")
    if s.grade:
        return f"verified grade of the current terms ({s.grade['record_id']})", ""
    if s.status == "unverified":
        return ("unverified: grade withdrawn by " + s.tip["record_id"],
                s.status_note + " (excluded from profit and ROI; counted as pending)")
    if s.tip is not None:
        return ("pending: the last grade is of earlier terms",
                s.status_note + " (excluded from profit and ROI; counted as pending)")
    return ("pending: not graded yet",
            "no grade yet - no completion evidence, or grading not run "
            "(excluded from profit and ROI; counted as pending)")


def _profit(s: bj.WagerState) -> tuple[Decimal | None, str]:
    """Settled cash result, or None (blank) when there is no verified grade -
    an ungraded wager is never shown as a settled $0.00."""
    if s.void:
        return None, "excluded: void"
    if s.grade:
        bonus = bj.terms_view(s.terms)["funding"] == "bonus"
        result = s.grade["result"]
        if bonus:
            return Decimal(s.grade["profit_cad"]), {
                "win": "settled: bonus bet won (cash winnings)",
                "loss": "settled: bonus bet lost ($0.00 cash; bonus used)",
                "push": "settled: bonus bet pushed ($0.00 cash; bonus outcome unresolved)",
            }[result]
        if result == "push":
            return Decimal(s.grade["profit_cad"]), "settled: push (stake returned, $0.00)"
        return Decimal(s.grade["profit_cad"]), f"settled: {result}"
    if s.status == "unverified":
        return None, "unavailable: grade withdrawn (unverified)"
    return None, "unavailable: not graded (pending)"


WAGER_COLUMNS = [
    ("Wager ID", None), ("Status", None), ("Result", None), ("Profit (CAD)", CAD_FMT),
    ("Profit basis", None), ("Season", INT_FMT), ("Week", INT_FMT), ("Game ID", None),
    ("Game", None), ("Kickoff (America/Toronto)", None), ("Team", None), ("Team name", None),
    ("Sportsbook", None), ("Market", None), ("Handicap", HANDICAP_FMT),
    ("Odds (American)", ODDS_FMT), ("Funding", None), ("Cash stake (CAD)", CAD_FMT),
    ("Bonus stake (CAD)", CAD_FMT), ("Bonus stake returned as cash on win", None),
    ("Moneyline tie rule", None), ("Placed at (America/Toronto)", None), ("Placed at (UTC)", None),
    ("Recorded at (UTC, wager record)", None), ("Current terms record", None),
    ("Terms recorded at (UTC)", None), ("Amendments", INT_FMT), ("Grade verification", None),
    ("Exclusion / status note", None), ("Grade record", None),
    ("Final score (away-home)", None), ("Reference", None), ("Note", None),
]


def wager_rows(states: dict) -> list[list]:
    rows = []
    for s in sorted(states.values(), key=lambda x: (bj.parse_ts(x.terms["placed_at"]),
                                                     x.wager_id)):
        t = bj.terms_view(s.terms)
        bonus = t["funding"] == "bonus"
        terms_record = s.amendments[-1] if s.amendments else s.wager
        verification, note = _verification(s)
        profit, basis = _profit(s)
        ev = s.grade["evidence"] if s.grade and not s.void else None
        rows.append([
            s.wager_id, s.status, s.grade["result"] if s.grade and not s.void else
            ("not graded" if s.status in ("pending", "unverified") else "void"),
            profit, basis, t["season"], t["week"], t["game_id"],
            f"{t['away_team']} @ {t['home_team']}", _toronto(t["kickoff_utc"]), t["team"],
            bj.team_name(t["team"]), t["sportsbook_name"], t["market"], t["handicap"], t["odds"],
            "bonus bet" if bonus else "cash", Decimal(t["cash_stake_cad"]),
            Decimal(t["bonus_stake_cad"]) if bonus else None,
            ("yes" if t["bonus_stake_returned"] else "no") if bonus else None,
            {"push": "tie refunded (push)", "loss": "tie settles as a loss"}.get(t["tie_rule"]),
            _toronto(t["placed_at"]), t["placed_at"],
            s.wager["recorded_at"], s.terms_record_id, terms_record["recorded_at"],
            len(s.amendments), verification, note,
            s.grade["record_id"] if s.grade and not s.void else None,
            f"{ev['away_score']}-{ev['home_score']}" if ev else None,
            t["reference"], t["note"],
        ])
    return rows


HISTORY_COLUMNS = [
    ("Recorded at (UTC, precise)", None), ("Recorded at (America/Toronto)", None),
    ("Kind", None), ("Schema", INT_FMT), ("Record ID", None), ("Wager ID", None),
    ("Follows record", None), ("Terms record", None), ("Link", None), ("Market", None),
    ("Funding", None), ("Result", None), ("Profit (CAD)", CAD_FMT), ("Bonus outcome", None),
    ("Cash stake (CAD)", CAD_FMT), ("Bonus stake (CAD)", CAD_FMT), ("Detail", None),
    ("Reason", None), ("Code revision", None),
]


def history_rows(records: list[dict], states: dict) -> list[list]:
    """bet_journal.history_rows (its audit order, links and details), with
    the precise timestamps and structured fields of each record."""
    by_id = {r["record_id"]: r for r in records}
    rows = []
    for h in bj.history_rows(records, states):
        r = by_id[h["Record ID"]]
        follows = r.get("previous") or r.get("supersedes")
        view = bj.terms_view(r["terms"]) if r.get("terms") else None
        if view:
            market, funding = view["market"], view["funding"]
        elif r["kind"] == bj.KIND_GRADE:          # schema-1 grades settle cash spreads
            market, funding = r.get("market", "spread"), r.get("funding", "cash")
        else:
            market = funding = None
        bonus = view is not None and view["funding"] == "bonus"
        rows.append([
            r["recorded_at"], _toronto(r["recorded_at"]), h["Record"], r["schema_version"],
            r["record_id"], h["Wager"], follows, r.get("terms_record_id"), h["Link"], market,
            None if funding is None else ("bonus bet" if funding == "bonus" else "cash"),
            r.get("result"),
            Decimal(r["profit_cad"]) if r.get("profit_cad") is not None else None,
            r.get("bonus_outcome"),
            Decimal(view["cash_stake_cad"]) if view else None,
            Decimal(view["bonus_stake_cad"]) if bonus else None,
            h["Detail"], r.get("reason"), r["code_revision"],
        ])
    return rows


SUMMARY_COLUMNS = [("Item", None), ("Value", None), ("Which wagers / notes", None)]


def summary_rows(records: list[dict], states: dict, exported_at: datetime) -> list[list]:
    """The journal's own totals() with what enters each figure."""
    t = bj.totals(states)
    local = exported_at.astimezone(on.TORONTO)
    roi = None if t["roi_pct"] is None else t["roi_pct"] / 100
    return [
        ["Exported at (America/Toronto)", f"{local:%Y-%m-%d %H:%M:%S %Z}", NOTICE],
        ["Exported at (UTC)", bj.iso(exported_at.replace(microsecond=0)), ""],
        ["Currency", "CAD", "All amounts are Canadian dollars."],
        ["Journal records (all kinds)", len(records),
         "Wagers, amendments, voids, grades and invalidations (see History)."],
        ["Current wagers", t["wagers"], "Every wager not voided; an amended wager counts once, "
                                        "on its current terms."],
        ["Graded (verified)", t["graded"], "Current wagers whose current terms have a verified "
                                           "grade (wins + losses + pushes)."],
        ["Wins", t["win"], "Graded wagers that won."],
        ["Losses", t["loss"], "Graded wagers that lost."],
        ["Pushes", t["push"], "Graded wagers that pushed: stake returned, $0.00 profit, "
                              "counted in graded stake."],
        ["Pending", t["pending"], "Current wagers without a verified grade: not graded yet, "
                                  "graded on earlier terms, or unverified. Not in profit or ROI."],
        ["Pending exposure (CAD)", t["stake_pending_cad"], "Cash stake of the pending wagers "
                                                           "(pending bonus stake is below)."],
        ["Unverified (included in Pending)", t["unverified"],
         "Wagers whose grade was withdrawn by an invalidation; excluded from profit and ROI "
         "until a new grade."],
        ["Voided", t["voided"], "Voided wagers: kept in History, excluded from every total."],
        ["Cash-funded wagers", t["cash_wagers"], "Current wagers staked with cash."],
        ["Bonus-bet wagers", t["bonus_wagers"], "Current wagers placed with one sportsbook "
                                                "bonus bet (never mixed with cash)."],
        ["Moneyline wagers", t["moneyline_wagers"], "Current moneyline wagers (the rest are "
                                                    "spread wagers)."],
        ["Graded stake (CAD)", t["stake_graded_cad"], "Cash stake of graded cash-funded "
                                                      "wagers (wins, losses and pushes)."],
        ["Net profit (CAD)", t["net_profit_cad"], "Cash-funded wagers only: sum of their "
                                                  "graded profit."],
        ["ROI", roi if roi is not None else "n/a (no graded cash wagers)",
         "Cash-funded wagers only: net profit / graded cash stake (bet_journal.totals). "
         "Bonus bets never enter it."],
        ["Bonus stake used (CAD, face value)", t["bonus_stake_used_cad"],
         "Bonus stake of graded bonus bets that won or lost. Not cash staked."],
        ["Cash winnings from bonus bets (CAD)", t["bonus_cash_winnings_cad"],
         "Cash paid by winning bonus bets (winnings only, plus the stake where the "
         "promotion returns it as cash; reissued bonus credit is not counted). Not part of "
         "cash ROI."],
        ["Unresolved bonus outcomes", t["bonus_unresolved"],
         "Bonus bets that pushed or tied: whether the bonus is reissued isn't recorded, so "
         "it is neither cash nor counted as used."],
        ["Unresolved bonus stake (CAD, face value)", t["bonus_stake_unresolved_cad"],
         "Bonus stake of those pushes/ties."],
        ["Pending bonus stake (CAD, face value)", t["bonus_stake_pending_cad"],
         "Bonus stake of bonus bets without a verified grade."],
        ["Net cash incl. bonus winnings (CAD)", t["net_cash_cad"],
         "Cash-bet net profit + cash winnings from bonus bets."],
    ]


_SUMMARY_FORMATS = {"Pending exposure (CAD)": CAD_FMT, "Graded stake (CAD)": CAD_FMT,
                    "Net profit (CAD)": CAD_FMT, "ROI": PCT_FMT,
                    "Bonus stake used (CAD, face value)": CAD_FMT,
                    "Cash winnings from bonus bets (CAD)": CAD_FMT,
                    "Unresolved bonus stake (CAD, face value)": CAD_FMT,
                    "Pending bonus stake (CAD, face value)": CAD_FMT,
                    "Net cash incl. bonus winnings (CAD)": CAD_FMT}


def _summary_sheet(ws, records, states, exported_at):
    rows = summary_rows(records, states, exported_at)
    _table(ws, SUMMARY_COLUMNS, rows, wrap=("Which wagers / notes",))
    for r, row in enumerate(rows, 2):
        fmt = _SUMMARY_FORMATS.get(row[0])
        if fmt and not isinstance(row[1], str):
            ws.cell(row=r, column=2).number_format = fmt
        if isinstance(row[1], int) and not isinstance(row[1], bool):
            ws.cell(row=r, column=2).number_format = INT_FMT
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 26
    ws.column_dimensions["C"].width = 90
    # Wagers left out of profit/ROI, by reason (after the totals table).
    excluded = [s for s in sorted(states.values(), key=lambda x: x.wager_id) if not s.grade
                or s.void]
    start = len(rows) + 4
    title = ws.cell(row=start, column=1, value="Wagers not in profit or ROI")
    title.font = _TITLE_FONT
    if not excluded:
        _put(ws, start + 1, 1, "None: every current wager has a verified grade."
             if states else "None: the journal has no wagers.")
        return
    for c, name in enumerate(("Wager ID", "Status", "Reason"), 1):
        cell = _put(ws, start + 1, c, name)
        cell.font, cell.fill = _HEADER_FONT, _HEADER_FILL
    for i, s in enumerate(excluded, start + 2):
        _put(ws, i, 1, s.wager_id)
        _put(ws, i, 2, s.status)
        _put(ws, i, 3, _verification(s)[1]).alignment = Alignment(wrap_text=True,
                                                                  vertical="top")


def build_workbook(records: list[dict], states: dict, exported_at: datetime) -> bytes:
    """The workbook as .xlsx bytes, built in memory (nothing written to disk)."""
    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"
    _summary_sheet(summary, records, states, exported_at)
    wagers = wb.create_sheet("Wagers")
    _table(wagers, WAGER_COLUMNS, wager_rows(states),
           wrap=("Exclusion / status note", "Note", "Grade verification"))
    history = wb.create_sheet("History")
    _table(history, HISTORY_COLUMNS, history_rows(records, states), wrap=("Detail", "Reason"))
    wb.properties.title = "Bet journal export (viewing copy)"
    wb.properties.description = NOTICE
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
