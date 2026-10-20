"""Bet Journal: moneyline wagers and bonus bets (schema 2), legacy compatibility.

Temporary journals and synthetic schedules only (the bet-journal tests' `jr`
fixture: temp journal and schedule, fixed clock, network blocked). The
fixture's finished game is KC @ BUF, final KC 24 - BUF 27 (BUF by 3).
"""
import hashlib
import json
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

import bet_journal as bj
import bet_journal_export as bjx
from test_bet_journal import (CHI_GB, KC_BUF, NOW, jr, prep_amend, prep_void,  # noqa: F401
                              record, render, state, terms, text, label)
from test_bet_journal_export import read_xlsx, summary, table, val

UTC = timezone.utc
REAL = datetime(2026, 10, 19, 11, 25, 38, 86319, tzinfo=UTC)


def ml(**kw):
    """Moneyline terms kwargs (no handicap)."""
    return {"market": "moneyline", "handicap": None, "tie_rule": "push", **kw}


def bonus(returned=False, **kw):
    return {"funding": "bonus", "bonus_stake_returned": returned, **kw}


def grade(jr, at=NOW):
    return bj.grade(jr.dir, jr.schedule, now=at)


def files(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.iterdir())}


def reseal(doc, **changes):
    d = {k: v for k, v in doc.items() if k not in (bj.CHECKSUM_FIELD, "_file")}
    d.update(changes)
    return bj._seal(d)


# ------------------------------------------------------------ legacy -------

class TestLegacyCompatibility:
    def test_cash_spread_is_still_written_as_schema_1(self, jr):
        doc = record(jr)
        assert doc["schema_version"] == 1 and set(doc["terms"]) == set(bj.TERM_FIELDS)
        assert "market" not in doc["terms"] and doc["terms"]["stake_cad"] == "50.00"
        [g] = grade(jr)
        assert g["schema_version"] == 1 and "market" not in g          # schema-1 grade

    def test_legacy_defaults_are_explicit(self, jr):
        v = bj.terms_view(record(jr)["terms"])
        assert (v["market"], v["funding"], v["cash_stake_cad"], v["bonus_stake_cad"],
                v["bonus_stake_returned"], v["tie_rule"]) == \
            ("spread", "cash", "50.00", "0.00", None, None)

    def test_cash_spread_totals_unchanged(self, jr):
        record(jr)                                                   # BUF -2.5 -110 $50: win
        record(jr, team="KC", handicap=2.5, odds=150, stake="20.00", reference="L")   # loss
        grade(jr)
        t = bj.totals(state(jr))
        assert (t["stake_graded_cad"], t["net_profit_cad"], t["roi_pct"]) == \
            (Decimal("70.00"), Decimal("25.45"), Decimal("36.36"))
        assert (t["bonus_wagers"], t["bonus_cash_winnings_cad"], t["net_cash_cad"]) == \
            (0, Decimal("0.00"), Decimal("25.45"))

    def test_existing_files_are_never_rewritten(self, jr):
        record(jr)
        before = files(jr.dir)
        grade(jr)
        bj.current_state(bj.load_records(jr.dir))
        after = files(jr.dir)
        assert all(after[k] == v for k, v in before.items())          # only new files added


# --------------------------------------------------------- moneyline -------

class TestMoneyline:
    def test_cash_moneyline_win_and_loss(self, jr):
        w = record(jr, **ml(team="BUF", odds=-110, stake="50.00"))
        lo = record(jr, **ml(team="KC", odds=150, stake="20.00", reference="K"))
        assert w["schema_version"] == 2 and w["terms"]["handicap"] is None
        assert (w["terms"]["cash_stake_cad"], w["terms"]["bonus_stake_cad"]) == ("50.00", "0.00")
        grade(jr)
        s = state(jr)
        assert (s[w["record_id"]].grade["result"], s[w["record_id"]].grade["profit_cad"]) == \
            ("win", "45.45")
        assert (s[lo["record_id"]].grade["result"], s[lo["record_id"]].grade["profit_cad"]) == \
            ("loss", "-20.00")
        g = s[w["record_id"]].grade
        assert (g["schema_version"], g["market"], g["funding"], g["bonus_outcome"]) == \
            (2, "moneyline", "cash", None)

    def test_moneyline_ignores_the_margin(self, jr):
        # BUF won by 3: a BUF -3.5 spread loses, a BUF moneyline wins.
        sp = record(jr, team="BUF", handicap=-3.5, odds=-110, stake="10.00")
        mlw = record(jr, **ml(team="BUF", odds=-200, stake="10.00", reference="M"))
        grade(jr)
        s = state(jr)
        assert s[sp["record_id"]].grade["result"] == "loss"
        assert (s[mlw["record_id"]].grade["result"], s[mlw["record_id"]].grade["profit_cad"]) == \
            ("win", "5.00")

    @pytest.mark.parametrize("rule,result,profit", [("push", "push", "0.00"),
                                                    ("loss", "loss", "-20.00")])
    def test_tie_follows_the_recorded_rule(self, jr, rule, result, profit):
        jr.set_scores((20, 20))
        w = record(jr, **ml(team="KC", odds=150, stake="20.00", tie_rule=rule))
        grade(jr)
        g = state(jr)[w["record_id"]].grade
        assert (g["result"], g["profit_cad"]) == (result, profit)

    @pytest.mark.parametrize("kw,msg", [
        ({"handicap": 0.0}, "no handicap"),
        ({"handicap": -1.5}, "no handicap"),
        ({"tie_rule": None}, "tie"),
        ({"tie_rule": "refund"}, "tie"),
    ])
    def test_moneyline_input_rules(self, jr, kw, msg):
        with pytest.raises(bj.JournalError, match=msg):
            terms(jr, **{**ml(), **kw})

    def test_spread_rejects_a_tie_rule(self, jr):
        with pytest.raises(bj.JournalError, match="only to moneyline"):
            terms(jr, tie_rule="push")

    def test_display_never_shows_a_handicap_for_moneyline(self, jr):
        w = record(jr, **ml(team="BUF", odds=-110, stake="50.00"))
        assert bj.bet_text(w["terms"]) == "Buffalo Bills moneyline at −110"
        assert "a tie is refunded (push)" in bj.stake_text(w["terms"])
        [row] = bj.current_rows(state(jr))
        assert (row["Market"], row["Bet"]) == ("moneyline", "Buffalo Bills moneyline at −110")


# ------------------------------------------------------------ bonus --------

class TestBonusBets:
    def test_the_25_plus_320_example(self, jr):
        win = record(jr, **ml(**bonus(team="BUF", odds=320, stake="25.00")))
        loss = record(jr, **ml(**bonus(team="KC", odds=320, stake="25.00", reference="B2")))
        grade(jr)
        s = state(jr)
        assert s[win["record_id"]].grade["profit_cad"] == "80.00"          # not 105
        assert s[loss["record_id"]].grade["profit_cad"] == "0.00"          # not -25
        assert {s[x["record_id"]].grade["bonus_outcome"] for x in (win, loss)} == {"used"}
        assert win["terms"]["cash_stake_cad"] == "0.00" and win["terms"]["bonus_stake_cad"] == "25.00"

    def test_stake_returned_with_winnings(self, jr):
        w = record(jr, **ml(**bonus(returned=True, team="BUF", odds=320, stake="25.00")))
        assert "stake returned as cash with winnings" in bj.stake_text(w["terms"])
        grade(jr)
        assert state(jr)[w["record_id"]].grade["profit_cad"] == "105.00"

    def test_bonus_spread_win_loss_and_push(self, jr):
        win = record(jr, **bonus(team="BUF", handicap=-2.5, odds=-110, stake="25.00"))
        loss = record(jr, **bonus(team="KC", handicap=2.5, odds=-110, stake="25.00",
                                  reference="L"))
        push = record(jr, **bonus(team="BUF", handicap=-3.0, odds=-110, stake="25.00",
                                  reference="P"))
        grade(jr)
        s = state(jr)
        assert s[win["record_id"]].grade["profit_cad"] == "22.73"
        assert (s[loss["record_id"]].grade["result"], s[loss["record_id"]].grade["profit_cad"]) \
            == ("loss", "0.00")
        g = s[push["record_id"]].grade
        assert (g["result"], g["profit_cad"], g["bonus_outcome"]) == ("push", "0.00", "unresolved")

    @pytest.mark.parametrize("odds,stake,expected", [(-115, "10.00", "8.70"), (105, "33.33", "35.00"),
                                                    (-110, "1.00", "0.91"), (101, "0.99", "1.00"),
                                                    (-333, "10.00", "3.00")])
    def test_cent_rounding_positive_and_negative_odds(self, jr, odds, stake, expected):
        w = record(jr, **ml(**bonus(team="BUF", odds=odds, stake=stake)))
        grade(jr)
        assert state(jr)[w["record_id"]].grade["profit_cad"] == expected

    @pytest.mark.parametrize("kw,msg", [
        ({"funding": "bonus"}, "returned as cash with winnings"),
        ({"funding": "cash", "bonus_stake_returned": False}, "only to bonus"),
        ({"funding": "credit"}, "funding"),
    ])
    def test_bonus_input_rules(self, jr, kw, msg):
        with pytest.raises(bj.JournalError, match=msg):
            terms(jr, **kw)


# ------------------------------------------------------------ totals -------

class TestTotals:
    def test_bonus_only_portfolio_has_no_cash_roi(self, jr):
        record(jr, **ml(**bonus(team="BUF", odds=320, stake="25.00")))
        record(jr, **ml(**bonus(team="KC", odds=320, stake="25.00", reference="B2")))
        grade(jr)
        t = bj.totals(state(jr))
        assert (t["stake_graded_cad"], t["net_profit_cad"], t["roi_pct"]) == \
            (Decimal("0.00"), Decimal("0.00"), None)
        assert (t["bonus_stake_used_cad"], t["bonus_cash_winnings_cad"], t["net_cash_cad"]) == \
            (Decimal("50.00"), Decimal("80.00"), Decimal("80.00"))
        assert (t["win"], t["loss"], t["bonus_graded"], t["cash_graded"]) == (1, 1, 2, 0)

    def test_mixed_portfolio_keeps_cash_roi_separate(self, jr):
        record(jr)                                                         # cash win +45.45
        record(jr, team="KC", handicap=2.5, odds=150, stake="20.00", reference="L")  # cash -20
        record(jr, **ml(**bonus(team="BUF", odds=320, stake="25.00", reference="B")))  # +80 cash
        push = record(jr, **bonus(team="BUF", handicap=-3.0, odds=-110, stake="10.00",
                                  reference="P"))                           # unresolved
        record(jr, game_id=CHI_GB, team="GB", placed_at=datetime(2026, 10, 7, 16, 0, tzinfo=UTC),
               **ml(**bonus(odds=200, stake="15.00", reference="PEND")))   # pending bonus
        grade(jr)
        t = bj.totals(state(jr))
        assert (t["stake_graded_cad"], t["net_profit_cad"], t["roi_pct"]) == \
            (Decimal("70.00"), Decimal("25.45"), Decimal("36.36"))         # cash only
        assert (t["bonus_cash_winnings_cad"], t["bonus_stake_used_cad"]) == \
            (Decimal("80.00"), Decimal("25.00"))
        assert (t["bonus_unresolved"], t["bonus_stake_unresolved_cad"]) == (1, Decimal("10.00"))
        assert (t["bonus_stake_pending_cad"], t["stake_pending_cad"]) == \
            (Decimal("15.00"), Decimal("0.00"))
        assert t["net_cash_cad"] == Decimal("105.45")
        assert state(jr)[push["record_id"]].grade["result"] == "push"

    def test_void_bonus_and_pending_stay_out_of_settled_totals(self, jr):
        v = record(jr, **ml(**bonus(team="BUF", odds=320, stake="25.00")))
        bj.save_void(prep_void(jr, v["record_id"], "entered by mistake"), jr.dir)
        grade(jr)
        t = bj.totals(state(jr))
        assert (t["voided"], t["bonus_wagers"], t["bonus_cash_winnings_cad"], t["net_cash_cad"]) \
            == (1, 0, Decimal("0.00"), Decimal("0.00"))


# --------------------------------------------------- lifecycle -------------

class TestLifecycle:
    def test_amend_cash_spread_into_bonus_moneyline_and_regrade(self, jr):
        w = record(jr)                                                    # schema 1
        [g1] = grade(jr)
        a = prep_amend(jr, w["record_id"], *terms(jr, **ml(**bonus(team="BUF", odds=320,
                                                                    stake="25.00"))),
                       "it was a bonus bet on the moneyline", now=NOW + timedelta(minutes=1))
        assert a["schema_version"] == 2
        bj.save_amendment(a, jr.dir)
        s = state(jr)[w["record_id"]]
        assert s.status == "pending" and s.grade is None                 # old grade no longer applies
        [g2] = grade(jr, NOW + timedelta(minutes=2))
        assert (g2["schema_version"], g2["market"], g2["funding"], g2["profit_cad"],
                g2["supersedes"], g2["reason"]) == \
            (2, "moneyline", "bonus", "80.00", g1["record_id"], "terms amended since the last grade")
        assert bj.totals(state(jr))["net_profit_cad"] == Decimal("0.00")

    def test_invalidation_and_score_correction_keep_market_and_funding(self, jr):
        w = record(jr, **ml(**bonus(team="BUF", odds=320, stake="25.00")))
        [g1] = grade(jr)
        jr.set_scores((None, None))
        [x] = grade(jr, NOW + timedelta(hours=1))
        assert x["kind"] == bj.KIND_INVALIDATION and x["schema_version"] == 1
        s = state(jr)[w["record_id"]]
        assert s.status == "unverified"
        assert bj.totals(state(jr))["bonus_cash_winnings_cad"] == Decimal("0.00")
        jr.set_scores((20, 20))                                          # now a tie
        [g2] = grade(jr, NOW + timedelta(hours=2))
        assert (g2["result"], g2["profit_cad"], g2["bonus_outcome"], g2["market"], g2["funding"]) \
            == ("push", "0.00", "unresolved", "moneyline", "bonus")
        jr.set_scores((24, 27))
        [g3] = grade(jr, NOW + timedelta(hours=3))
        assert (g3["result"], g3["profit_cad"], g3["supersedes"]) == ("win", "80.00", g2["record_id"])
        assert [r["kind"] for r in state(jr)[w["record_id"]].settlements] == \
            ["bet_journal_grade", "bet_journal_invalidation", "bet_journal_grade", "bet_journal_grade"]

    def test_microsecond_timestamps_and_late_entry(self, jr):
        t, sha = terms(jr, now=REAL, **ml(**bonus(team="BUF", odds=320, stake="25.00")))
        doc = bj.prepare_wager(t, sha, confirmed_existing=[], now=REAL)
        bj.save_wager(doc, jr.dir)
        assert doc["recorded_at"] == "2026-10-19T11:25:38.086319Z"
        assert t["placed_at"] == "2026-10-04T16:00:00Z"                  # pregame, recorded later
        [loaded] = bj.load_records(jr.dir)
        assert loaded["recorded_at"] == doc["recorded_at"]


# ---------------------------------------------------------- duplicates ------

class TestDuplicates:
    def test_market_and_funding_are_part_of_identity(self, jr):
        base = dict(team="BUF", odds=-110, stake="50.00")
        record(jr, **ml(**base))
        t_cash_ml, _ = terms(jr, **ml(**base))
        t_bonus_ml, _ = terms(jr, **ml(**bonus(**base)))
        t_spread, _ = terms(jr, handicap=-2.5, **base)
        s = state(jr)
        assert bj.identical_wagers(t_cash_ml, s)                         # genuinely identical
        assert not bj.identical_wagers(t_bonus_ml, s)
        assert not bj.identical_wagers(t_spread, s)

    def test_identical_bonus_moneyline_needs_confirmation(self, jr):
        first = record(jr, **ml(**bonus(team="BUF", odds=320, stake="25.00")))
        with pytest.raises(bj.DuplicateWager):
            record(jr, **ml(**bonus(team="BUF", odds=320, stake="25.00")))
        second = record(jr, confirmed=[first["record_id"]],
                        **ml(**bonus(team="BUF", odds=320, stake="25.00")))
        assert second["confirmed_existing"] == [first["record_id"]]

    def test_amendment_into_an_identical_wager_is_flagged(self, jr):
        a = record(jr, **ml(**bonus(team="BUF", odds=320, stake="25.00")))
        b = record(jr, **ml(team="BUF", odds=320, stake="25.00", reference="CASH"))
        t, _ = terms(jr, **ml(**bonus(team="BUF", odds=320, stake="25.00")))
        assert bj.identical_wagers(t, state(jr), exclude=b["record_id"]) == [a["record_id"]]


# ----------------------------------------------------------- integrity ------

def _v2(jr):
    return record(jr, **ml(**bonus(team="BUF", odds=320, stake="25.00")))


class TestIntegrity:
    @pytest.mark.parametrize("change,msg", [
        ({"schema_version": 3}, "unsupported schema_version"),
        ({"schema_version": True}, "unsupported schema_version"),
        ({"schema_version": 1}, "fields differ from schema 1"),
    ])
    def test_unknown_or_wrong_versions_rejected(self, jr, change, msg):
        doc = _v2(jr)
        with pytest.raises(bj.IntegrityError, match=msg):
            bj.validate_record(reseal(doc, **change), "x")

    @pytest.mark.parametrize("terms_change,msg", [
        ({"market": "spread", "handicap": -2.5, "tie_rule": None, "cash_stake_cad": "25.00",
          "bonus_stake_cad": "0.00", "bonus_stake_returned": None}, "schema-1 record"),
        ({"cash_stake_cad": "5.00"}, "mixed funding"),
        ({"bonus_stake_cad": "0.00"}, "entirely"),
        ({"handicap": 0.0}, "no handicap"),
        ({"tie_rule": None}, "tie rule"),
        ({"bonus_stake_returned": None}, "returned as cash with"),
        ({"market": "parlay"}, "unknown market"),
    ])
    def test_invalid_v2_terms_rejected(self, jr, terms_change, msg):
        doc = _v2(jr)
        with pytest.raises(bj.IntegrityError, match=msg):
            bj.validate_record(reseal(doc, terms={**doc["terms"], **terms_change}), "x")

    def test_void_and_invalidation_exist_only_in_schema_1(self, jr):
        w = _v2(jr)
        v = prep_void(jr, w["record_id"], "x")
        assert v["schema_version"] == 1
        with pytest.raises(bj.IntegrityError, match="only in schema 1"):
            bj.validate_record(reseal(v, schema_version=2), "x")

    @pytest.mark.parametrize("change,msg", [
        ({"bonus_outcome": "unresolved"}, "bonus_outcome"),
        ({"funding": "cash"}, "bonus outcome|market/funding"),
        ({"profit_cad": "105.00"}, "result/profit"),
    ])
    def test_grade_must_follow_from_its_terms(self, jr, change, msg):
        _v2(jr)
        [g] = grade(jr)
        path = jr.dir / f"{g['record_id']}.json"
        doc = {k: v for k, v in json.loads(path.read_text()).items()}
        path.write_text(json.dumps(reseal(doc, **change)), encoding="utf-8")
        with pytest.raises(bj.IntegrityError, match=msg):
            state(jr)

    def test_schema_1_grade_of_schema_2_terms_rejected(self, jr):
        _v2(jr)
        [g] = grade(jr)
        path = jr.dir / f"{g['record_id']}.json"
        doc = json.loads(path.read_text())
        for f in ("market", "funding", "bonus_outcome"):
            doc.pop(f)
        path.write_text(json.dumps(reseal(doc, schema_version=1)), encoding="utf-8")
        with pytest.raises(bj.IntegrityError, match="schema version of the terms"):
            state(jr)

    def test_corrupt_v2_record_fails_closed(self, jr):
        w = _v2(jr)
        p = jr.dir / f"{w['record_id']}.json"
        p.write_text(p.read_text().replace('"25.00"', '"26.00"', 1), encoding="utf-8")
        with pytest.raises(bj.IntegrityError, match="checksum"):
            bj.load_records(jr.dir)
        with pytest.raises(bj.IntegrityError):
            bjx.export(jr.dir, NOW)


# --------------------------------------------------------------- Excel ------

class TestExcel:
    def test_workbook_reconciles_markets_funding_and_bonus_totals(self, jr, tmp_path):
        cash = record(jr)
        mlb = record(jr, **ml(**bonus(team="BUF", odds=320, stake="25.00", reference="B",
                                      note="=1+1")))
        push = record(jr, **bonus(team="BUF", handicap=-3.0, odds=-110, stake="10.00",
                                  reference="P"))
        grade(jr)
        before = files(jr.dir)
        book = read_xlsx(bjx.export(jr.dir, NOW)["data"])
        assert files(jr.dir) == before                                   # nothing written
        t = bj.totals(state(jr))
        s = summary(book)
        for item, key in (("Bonus stake used (CAD, face value)", "bonus_stake_used_cad"),
                          ("Cash winnings from bonus bets (CAD)", "bonus_cash_winnings_cad"),
                          ("Unresolved bonus stake (CAD, face value)", "bonus_stake_unresolved_cad"),
                          ("Net cash incl. bonus winnings (CAD)", "net_cash_cad"),
                          ("Net profit (CAD)", "net_profit_cad")):
            assert val(s[item]) == float(t[key]) and s[item]["fmt"] == bjx.CAD_FMT, item
        assert (val(s["Unresolved bonus outcomes"]), val(s["Bonus-bet wagers"]),
                val(s["Moneyline wagers"])) == (1, 2, 1)
        assert val(s["ROI"]) == float(t["roi_pct"] / 100)
        rows = {val(r["Wager ID"]): r for r in table(book["Wagers"])}
        r = rows[mlb["record_id"]]
        assert (val(r["Market"]), val(r["Funding"]), val(r["Handicap"]), val(r["Cash stake (CAD)"]),
                val(r["Bonus stake (CAD)"]), val(r["Bonus stake returned as cash on win"]),
                val(r["Profit (CAD)"])) == ("moneyline", "bonus bet", None, 0.0, 25.0, "no", 80.0)
        assert "bonus bet won" in val(r["Profit basis"])
        assert (r["Note"]["t"], r["Note"]["v"], r["Note"]["quote"]) == ("inlineStr", "=1+1", True)
        p = rows[push["record_id"]]
        assert (val(p["Result"]), val(p["Profit (CAD)"])) == ("push", 0.0)
        assert "unresolved" in val(p["Profit basis"])
        c = rows[cash["record_id"]]
        assert (val(c["Market"]), val(c["Funding"]), val(c["Handicap"])) == ("spread", "cash", -2.5)
        hist = table(book["History"])
        grades = {val(h["Wager ID"]): h for h in hist if val(h["Kind"]) == "grade"}
        assert (val(grades[mlb["record_id"]]["Schema"]), val(grades[mlb["record_id"]]["Bonus outcome"])) \
            == (2, "used")
        assert val(grades[cash["record_id"]]["Schema"]) == 1
        assert all(sheet["formulas"] == 0 for sheet in book.values())


# ---------------------------------------------------------------- page ------

def fill_ml_bonus(at, jr, *, tie="push", ret="not_returned", odds=320, stake="25.00",
                  preview=True):
    box = at.selectbox(key="bj_game")
    box.select_index(box.options.index(label(jr, KC_BUF)))
    at.run()
    at.radio(key="bj_market").set_value("moneyline")
    at.radio(key="bj_funding").set_value("bonus")
    at.run()
    at.selectbox(key="bj_book").set_value("betmgm_ca_on")
    at.selectbox(key="bj_team").set_value("BUF")
    at.number_input(key="bj_odds").set_value(odds)
    at.text_input(key="bj_stake").set_value(stake)
    at.date_input(key="bj_date").set_value(date(2026, 10, 4))
    at.time_input(key="bj_time").set_value(time(12, 0))
    if tie:
        at.radio(key="bj_tie").set_value(tie)
    if ret:
        at.radio(key="bj_bonus_ret").set_value(ret)
    if preview:
        at.button(key="bj_preview").click()
        at.run()
    return at


class TestPage:
    def test_preview_save_reload_bonus_moneyline(self, jr, render):
        at = fill_ml_bonus(render(), jr)
        md = text(at, "markdown")
        assert "Buffalo Bills moneyline at +320" in md
        assert "bonus bet (stake not returned with winnings)" in md and "refunded (push)" in md
        assert at.button(key="bj_save").disabled is False
        assert not any(w.key == "bj_spread" for w in at.number_input)       # no handicap field
        at.button(key="bj_save").click()
        at.run()
        assert "Saved wager" in text(at, "success"), text(at, "error")
        [doc] = [json.loads(p.read_text()) for p in jr.files()]
        assert (doc["schema_version"], doc["terms"]["market"], doc["terms"]["handicap"],
                doc["terms"]["bonus_stake_cad"], doc["terms"]["bonus_stake_returned"],
                doc["terms"]["tie_rule"]) == (2, "moneyline", None, "25.00", False, "push")
        bj.grade(jr.dir, jr.schedule, now=NOW)
        at = render(at=at)                                                   # reload
        assert not at.exception and not at.error
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Cash won from bonus bets"] == "$80.00"
        assert metrics["Net profit (CAD)"] == "$0.00" and metrics["ROI"] == "n/a"

    def test_tie_rule_is_required(self, jr, render):
        at = fill_ml_bonus(render(), jr, tie=None)
        assert "tie" in text(at, "error") and jr.files() == []

    def test_bonus_return_terms_are_required(self, jr, render):
        at = fill_ml_bonus(render(), jr, ret=None)
        assert "returned as cash with winnings" in text(at, "error") and jr.files() == []

    def test_changing_the_tie_rule_after_preview_requires_a_new_preview(self, jr, render):
        at = fill_ml_bonus(render(), jr)
        at.radio(key="bj_tie").set_value("loss")
        at.button(key="bj_save").click()
        at.run()
        assert "form changed after Preview" in text(at, "error") and jr.files() == []
