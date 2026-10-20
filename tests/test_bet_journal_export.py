"""Bet Journal Excel export (bet_journal_export.py and the page's export section).

Temporary journals only (the bet-journal tests' `jr` fixture: temp journal and
schedule, fixed clock, network blocked). Workbooks are checked with an
independent reader: the .xlsx package is parsed directly with zipfile and
ElementTree (not openpyxl, which wrote it), so cell types, formulas, number
formats and sheet settings are verified as Excel will see them.
"""
import hashlib
import io
import re
import zipfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from xml.etree import ElementTree as ET

import pytest

import bet_journal as bj
import bet_journal_export as bjx
from test_bet_journal import (CHI_GB, KC_BUF, NOW, fill, jr, prep_amend, prep_void,  # noqa: F401
                              record, render, state, terms, text)

UTC = timezone.utc
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "rel": "http://schemas.openxmlformats.org/package/2006/relationships"}
BUILTIN_FMT = {0: "General", 1: "0", 2: "0.00", 9: "0%", 10: "0.00%", 14: "mm-dd-yy"}
REAL = datetime(2026, 10, 19, 11, 25, 38, 86319, tzinfo=UTC)


# ------------------------------------------------- independent .xlsx reader --

def _col_index(ref: str) -> int:
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def read_xlsx(data: bytes) -> dict:
    """{sheet name: {"rows": [[cell, ...]], "freeze": ref|None, "filter": ref|None,
    "widths": {col: width}, "formulas": n}}; each cell is
    {"t": type, "v": value, "fmt": number format, "quote": bool} or None."""
    z = zipfile.ZipFile(io.BytesIO(data))
    styles = ET.fromstring(z.read("xl/styles.xml"))
    custom = {int(n.get("numFmtId")): n.get("formatCode")
              for n in styles.findall("m:numFmts/m:numFmt", NS)}
    xfs = [(int(x.get("numFmtId", 0)), x.get("quotePrefix") == "1")
           for x in styles.findall("m:cellXfs/m:xf", NS)]
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
    wb = ET.fromstring(z.read("xl/workbook.xml"))
    rels = {r.get("Id"): r.get("Target") for r in
            ET.fromstring(z.read("xl/_rels/workbook.xml.rels")).findall("rel:Relationship", NS)}
    out = {}
    for sh in wb.findall("m:sheets/m:sheet", NS):
        target = rels[sh.get(f"{{{NS['r']}}}id")].lstrip("/")
        path = target if target.startswith("xl/") else f"xl/{target}"
        root = ET.fromstring(z.read(path))
        rows = []
        for row in root.findall("m:sheetData/m:row", NS):
            r = int(row.get("r"))
            while len(rows) < r:
                rows.append([])
            cells = rows[r - 1]
            for c in row.findall("m:c", NS):
                i = _col_index(c.get("r"))
                while len(cells) < i:
                    cells.append(None)
                t = c.get("t", "n")
                fmt_id, quote = xfs[int(c.get("s", 0))]
                if t == "inlineStr":
                    v = "".join(x.text or "" for x in c.iter(f"{{{NS['m']}}}t"))
                elif t == "s":
                    v = shared[int(c.find("m:v", NS).text)]
                elif c.find("m:v", NS) is not None:
                    v = c.find("m:v", NS).text
                    v = float(v) if t == "n" else v
                else:
                    v = None
                cells[i - 1] = {"t": t, "v": v, "quote": quote,
                                "fmt": custom.get(fmt_id, BUILTIN_FMT.get(fmt_id, str(fmt_id))),
                                "f": c.find("m:f", NS) is not None}
        pane = root.find("m:sheetViews/m:sheetView/m:pane", NS)
        af = root.find("m:autoFilter", NS)
        out[sh.get("name")] = {
            "rows": rows, "freeze": pane.get("topLeftCell") if pane is not None else None,
            "filter": af.get("ref") if af is not None else None,
            "widths": {int(c.get("min")): float(c.get("width"))
                       for c in root.findall("m:cols/m:col", NS)},
            "formulas": len(root.findall(".//m:f", NS)),
        }
    return out


def table(sheet: dict) -> list[dict]:
    """Rows of a header-first table as {header: cell}."""
    head = [c["v"] if c else None for c in sheet["rows"][0]]
    return [{h: (row[i] if i < len(row) else None) for i, h in enumerate(head)}
            for row in sheet["rows"][1:] if any(row)]


def summary(book: dict) -> dict:
    out = {}
    for row in book["Summary"]["rows"][1:]:
        if row and row[0] and row[0]["v"] == "Wagers not in profit or ROI":
            break
        if row and row[0]:
            out[row[0]["v"]] = row[1]
    return out


def val(cell):
    return None if cell is None else cell["v"]


def tree(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob("*")) if p.is_file()}


def export(jr, now=NOW):
    return read_xlsx(bjx.export(jr.dir, now)["data"])


# ------------------------------------------------------------- fixtures --

def mixed_journal(jr):
    """Win, loss, push (KC @ BUF, final 24-27), an amended wager, a pending
    wager (CHI @ GB, not played) and a voided one; graded once."""
    win = record(jr, team="BUF", handicap=-2.5, odds=-110, stake="50.00")          # BUF by 3
    loss = record(jr, team="KC", handicap=2.5, odds=150, stake="20.00", reference="L-1")
    push = record(jr, team="BUF", handicap=-3.0, odds=-105, stake="10.00", reference="P-1")
    amended = record(jr, team="KC", handicap=5.5, odds=-120, stake="30.00", reference="A-1")
    bj.save_amendment(prep_amend(jr, amended["record_id"],
                                 *terms(jr, team="KC", handicap=5.5, odds=-120, stake="35.00",
                                        reference="A-1"), "stake typo"), jr.dir)
    pending = record(jr, game_id=CHI_GB, team="GB", handicap=-1.5, odds=-110, stake="25.00",
                     placed_at=datetime(2026, 10, 7, 16, 0, tzinfo=UTC))
    voided = record(jr, game_id=CHI_GB, team="CHI", handicap=1.5, odds=-110, stake="40.00",
                    placed_at=datetime(2026, 10, 7, 16, 0, tzinfo=UTC), reference="V-1")
    bj.save_void(prep_void(jr, voided["record_id"], "entered by mistake"), jr.dir)
    bj.grade(jr.dir, jr.schedule, now=NOW)
    return {"win": win, "loss": loss, "push": push, "amended": amended, "pending": pending,
            "voided": voided}


# ----------------------------------------------------------------- tests --

class TestWorkbook:
    def test_empty_journal(self, jr):
        book = export(jr)
        assert list(book) == ["Summary", "Wagers", "History"]
        s = summary(book)
        assert (val(s["Current wagers"]), val(s["Graded (verified)"]), val(s["Voided"])) == (0, 0, 0)
        assert val(s["ROI"]) == "n/a (no graded cash wagers)"
        assert val(s["Currency"]) == "CAD"
        assert table(book["Wagers"]) == [] and table(book["History"]) == []
        cells = [c["v"] for row in book["Summary"]["rows"] for c in row if c]
        assert "None: the journal has no wagers." in cells
        assert not jr.dir.exists()                                     # nothing created

    def test_summary_matches_the_journal_totals_exactly(self, jr):
        mixed_journal(jr)
        t = bj.totals(state(jr))
        s = summary(export(jr))
        expected = {"Current wagers": t["wagers"], "Graded (verified)": t["graded"],
                    "Wins": t["win"], "Losses": t["loss"], "Pushes": t["push"],
                    "Pending": t["pending"], "Unverified (included in Pending)": t["unverified"],
                    "Voided": t["voided"], "Pending exposure (CAD)": float(t["stake_pending_cad"]),
                    "Graded stake (CAD)": float(t["stake_graded_cad"]),
                    "Net profit (CAD)": float(t["net_profit_cad"]),
                    "ROI": float(t["roi_pct"] / 100)}
        assert {k: val(s[k]) for k in expected} == expected
        assert (t["win"], t["loss"], t["push"], t["pending"], t["voided"]) == (2, 1, 1, 1, 1)
        for k in ("Pending exposure (CAD)", "Graded stake (CAD)", "Net profit (CAD)"):
            assert s[k]["t"] == "n" and s[k]["fmt"] == bjx.CAD_FMT
        assert s["ROI"]["t"] == "n" and s["ROI"]["fmt"] == bjx.PCT_FMT
        assert "America/Toronto" in book_text(export(jr), "Summary")
        assert bjx.NOTICE in book_text(export(jr), "Summary")

    def test_one_current_row_per_wager_and_complete_history(self, jr):
        w = mixed_journal(jr)
        st_ = state(jr)
        records = bj.load_records(jr.dir)
        book = export(jr)
        rows = {val(r["Wager ID"]): r for r in table(book["Wagers"])}
        assert set(rows) == set(st_) and len(table(book["Wagers"])) == len(st_)
        for wid, s in st_.items():
            assert val(rows[wid]["Status"]) == s.status
        assert val(rows[w["amended"]["record_id"]]["Cash stake (CAD)"]) == 35.0     # current terms
        assert val(rows[w["amended"]["record_id"]]["Amendments"]) == 1
        hist = table(book["History"])
        assert [val(r["Record ID"]) for r in hist] == \
            [h["Record ID"] for h in bj.history_rows(records, st_)]
        assert len(hist) == len(records)
        kinds = [val(r["Kind"]) for r in hist]
        assert {"wager", "amendment", "void", "grade"} <= set(kinds)
        amend = next(r for r in hist if val(r["Kind"]) == "amendment")
        assert val(amend["Follows record"]) == w["amended"]["record_id"]

    def test_win_loss_push_pending_and_void_rows(self, jr):
        w = mixed_journal(jr)
        rows = {val(r["Wager ID"]): r for r in table(export(jr)["Wagers"])}
        win, loss, push = (rows[w[k]["record_id"]] for k in ("win", "loss", "push"))
        pend, void = rows[w["pending"]["record_id"]], rows[w["voided"]["record_id"]]
        assert (val(win["Result"]), val(win["Profit (CAD)"])) == ("win", 45.45)
        assert (val(loss["Result"]), val(loss["Profit (CAD)"])) == ("loss", -20.0)
        assert (val(push["Result"]), val(push["Profit (CAD)"])) == ("push", 0.0)
        assert push["Profit (CAD)"]["t"] == "n" and "push" in val(push["Profit basis"])
        # Ungraded: no profit value at all (not a settled 0.00), and labelled so.
        assert val(pend["Profit (CAD)"]) is None and val(pend["Result"]) == "not graded"   # blank, not 0
        assert val(pend["Profit basis"]) == "unavailable: not graded (pending)"
        assert val(void["Profit (CAD)"]) is None and val(void["Status"]) == "void"
        assert "entered by mistake" in val(void["Exclusion / status note"])
        # Numeric, signed displays; amounts in CAD format.
        assert (win["Handicap"]["t"], win["Handicap"]["v"], win["Handicap"]["fmt"]) == \
            ("n", -2.5, bjx.HANDICAP_FMT)
        assert (loss["Odds (American)"]["v"], loss["Odds (American)"]["fmt"]) == \
            (150.0, bjx.ODDS_FMT)
        assert win["Cash stake (CAD)"]["fmt"] == bjx.CAD_FMT
        for col in ("Wager ID", "Game ID", "Recorded at (UTC, wager record)", "Placed at (UTC)"):
            assert win[col]["t"] == "inlineStr"

    def test_superseded_grade_kept_in_history(self, jr):
        w = mixed_journal(jr)
        jr.set_scores((24, 28))                                         # score correction
        new = bj.grade(jr.dir, jr.schedule, now=NOW + timedelta(hours=1))
        assert new and all(g["supersedes"] for g in new)
        book = export(jr, NOW + timedelta(hours=1))
        rows = {val(r["Wager ID"]): r for r in table(book["Wagers"])}
        assert val(rows[w["push"]["record_id"]]["Result"]) == "win"         # -3 now covers
        hist = table(book["History"])
        grades = [r for r in hist if val(r["Kind"]) == "grade"]
        assert len(grades) == 8 and sum(1 for r in grades if val(r["Follows record"])) == 4
        assert summary(book)["Wins"]["v"] == bj.totals(state(jr))["win"]

    def test_invalidated_grade_is_unverified_not_zero(self, jr):
        w = mixed_journal(jr)
        jr.set_scores((None, None))                                     # evidence withdrawn
        bj.grade(jr.dir, jr.schedule, now=NOW + timedelta(hours=1))
        book = export(jr, NOW + timedelta(hours=1))
        row = {val(r["Wager ID"]): r for r in table(book["Wagers"])}[w["win"]["record_id"]]
        assert val(row["Status"]) == "unverified" and val(row["Profit (CAD)"]) is None
        assert val(row["Profit basis"]) == "unavailable: grade withdrawn (unverified)"
        assert val(row["Grade verification"]).startswith("unverified: grade withdrawn by X-")
        t = bj.totals(state(jr))
        s = summary(book)
        assert (val(s["Unverified (included in Pending)"]), val(s["Net profit (CAD)"])) == \
            (t["unverified"], float(t["net_profit_cad"]))
        assert any(val(r["Kind"]) == "invalidation" for r in table(book["History"]))
        excluded = book_text(book, "Summary")
        assert w["win"]["record_id"] in excluded

    def test_whole_second_and_microsecond_timestamps_kept_exactly(self, jr):
        old = record(jr)                                                 # ...17:00:00Z
        new = record(jr, at=REAL, game_id=CHI_GB, team="GB",
                     placed_at=datetime(2026, 10, 7, 16, 0, tzinfo=UTC), now=REAL)
        book = export(jr, REAL)
        hist = {val(r["Record ID"]): r for r in table(book["History"])}
        assert val(hist[old["record_id"]]["Recorded at (UTC, precise)"]) == "2026-10-07T17:00:00Z"
        assert val(hist[new["record_id"]]["Recorded at (UTC, precise)"]) == \
            "2026-10-19T11:25:38.086319Z"
        assert hist[new["record_id"]]["Recorded at (UTC, precise)"]["t"] == "inlineStr"
        rows = {val(r["Wager ID"]): r for r in table(book["Wagers"])}
        assert val(rows[new["record_id"]]["Recorded at (UTC, wager record)"]) == \
            "2026-10-19T11:25:38.086319Z"

    def test_user_text_stays_literal_never_a_formula(self, jr):
        nasty = {"note": "=HYPERLINK(\"http://evil\",\"x\")", "reference": "+SUM(A1:A9)"}
        w = record(jr, **nasty)
        w2 = record(jr, sportsbook_key="other_on", sportsbook_name="-Book @ home", team="KC",
                    handicap=2.5, note="@cmd", reference="-1")
        book = export(jr)
        assert all(sheet["formulas"] == 0 for sheet in book.values())
        rows = {val(r["Wager ID"]): r for r in table(book["Wagers"])}
        r1, r2 = rows[w["record_id"]], rows[w2["record_id"]]
        for cell, expected in ((r1["Note"], nasty["note"]), (r1["Reference"], nasty["reference"]),
                               (r2["Sportsbook"], "-Book @ home"), (r2["Note"], "@cmd"),
                               (r2["Reference"], "-1")):
            assert (cell["t"], cell["v"], cell["quote"], cell["f"]) == \
                ("inlineStr", expected, True, False)

    def test_sheet_presentation(self, jr):
        mixed_journal(jr)
        book = export(jr)
        for name, sheet in book.items():
            assert sheet["freeze"] == "A2", name
            assert sheet["filter"] and sheet["filter"].startswith("A1:"), name
            assert sheet["widths"] and all(8 <= w <= 100 for w in sheet["widths"].values()), name

    def test_file_name_is_timestamped_in_toronto(self):
        assert bjx.file_name(REAL) == "bet_journal_20261019_072538.xlsx"


class TestIntegrityAndFreshness:
    def test_fresh_export_after_each_change(self, jr):
        w = record(jr)
        assert val(summary(export(jr))["Current wagers"]) == 1
        bj.save_amendment(prep_amend(jr, w["record_id"], *terms(jr, stake="75.00"), "typo"),
                          jr.dir)
        rows = table(export(jr)["Wagers"])
        assert val(rows[0]["Cash stake (CAD)"]) == 75.0
        bj.grade(jr.dir, jr.schedule, now=NOW)
        assert val(table(export(jr)["Wagers"])[0]["Result"]) == "win"
        jr.set_scores((None, None))
        bj.grade(jr.dir, jr.schedule, now=NOW + timedelta(hours=1))
        assert val(table(export(jr)["Wagers"])[0]["Status"]) == "unverified"

    @pytest.mark.parametrize("damage", ["corrupt", "tampered", "stray", "dangling_link"])
    def test_invalid_journal_refuses_export(self, jr, damage):
        w = record(jr)
        path = jr.dir / f"{w['record_id']}.json"
        if damage == "dangling_link":           # each file valid; the history isn't
            bj.save_amendment(prep_amend(jr, w["record_id"], *terms(jr, stake="60.00"), "x"),
                              jr.dir)
            path.unlink()
        elif damage == "corrupt":
            path.write_text("{not json", encoding="utf-8")
        elif damage == "tampered":
            path.write_text(path.read_text(encoding="utf-8").replace('"50.00"', '"55.00"'),
                            encoding="utf-8")
        else:
            (jr.dir / "notes.txt").write_text("x", encoding="utf-8")
        with pytest.raises(bj.IntegrityError):
            bjx.export(jr.dir, NOW)

    def test_export_writes_nothing_and_uses_no_network(self, jr, tmp_path):
        mixed_journal(jr)
        before = tree(tmp_path)
        bjx.export(jr.dir, NOW)                  # the jr fixture blocks every connection
        assert tree(tmp_path) == before


class TestPage:
    def test_prepare_then_download(self, jr, render):
        mixed_journal(jr)
        at = render()
        at.button(key="bj_export_prepare").click()
        at.run()
        assert not at.exception and not at.error
        prepared = at.session_state["bj_export"]
        assert prepared["file_name"] == "bet_journal_20261007_130000.xlsx"
        assert len(at.get("download_button")) == 1
        assert "bet_journal_20261007_130000.xlsx" in text(at, "caption")
        book = read_xlsx(prepared["data"])
        assert val(summary(book)["Current wagers"]) == bj.totals(state(jr))["wagers"]
        assert "viewing" in text(at, "caption") and "not a complete backup" in text(at, "caption")

    def test_prepared_export_is_dropped_when_the_journal_changes(self, jr, render):
        record(jr)
        at = render()
        at.button(key="bj_export_prepare").click()
        at.run()
        assert len(at.get("download_button")) == 1
        record(jr, team="KC", handicap=2.5, reference="NEW")             # journal changes
        at = render(at=at)
        assert len(at.get("download_button")) == 0
        assert "changed since the export was prepared" in text(at, "info")
        at.button(key="bj_export_prepare").click()
        at.run()
        assert at.session_state["bj_export"]["wagers"] == 2

    def test_corruption_after_render_refuses_the_export(self, jr, render):
        w = record(jr)
        at = render()
        (jr.dir / f"{w['record_id']}.json").write_text("{bad", encoding="utf-8")
        at.button(key="bj_export_prepare").click()
        at.run()
        assert "bj_export" not in at.session_state
        assert len(at.get("download_button")) == 0
        assert any("Integrity error" in e.value for e in at.error)

    def test_page_export_writes_nothing(self, jr, render, tmp_path):
        mixed_journal(jr)
        before = tree(tmp_path)
        at = render()
        at.button(key="bj_export_prepare").click()
        at.run()
        assert tree(tmp_path) == before


def book_text(book: dict, sheet: str) -> str:
    return " ".join(str(c["v"]) for row in book[sheet]["rows"] for c in row if c)
