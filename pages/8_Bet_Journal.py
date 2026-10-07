"""Bet Journal - single-game spread wagers you actually placed at Ontario sportsbooks.

Records the accepted terms of real wagers in data_files/bet_journal/ through
bet_journal.py (append-only records: wagers, amendments, voids and grades).
It never places a bet, creates no model recommendation, makes no API calls,
and doesn't touch betting_recommendations_log.csv or the Ontario captures.
Grading runs only when you click "Grade settled wagers". See docs/BET_JOURNAL.md.
"""

import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

import bet_journal as bj  # noqa: E402
import ontario_spreads as on  # noqa: E402
from footer import add_betting_oracle_footer  # noqa: E402

st.title("Bet journal")
st.caption("Wagers **you placed** at Ontario sportsbooks, recorded on the terms the sportsbook "
           "accepted. This page **records** wagers; it never places them. Journal entries are "
           "kept separate from the model's simulated recommendations, sportsbook line "
           "observations and frozen predictions.")


@st.cache_data(max_entries=8, show_spinner="Reading the journal...")
def load_journal(fingerprint: tuple, journal_dir: str) -> list[dict]:
    """Cached on the content fingerprint of every journal file, so a new,
    changed or stray file always re-reads (and re-validates) the journal."""
    return bj.load_records(Path(journal_dir))


@st.cache_data(max_entries=4)
def load_schedule(path: str, mtime_ns: int, size: int):
    return on.load_schedule(Path(path))[0]


try:
    records = load_journal(bj.fingerprint(bj.JOURNAL_DIR), str(bj.JOURNAL_DIR))
    states = bj.current_state(records)
except bj.IntegrityError as exc:
    st.error(
        "**Integrity error: a bet-journal record failed validation.** No totals are shown "
        "rather than totals that silently leave a wager out, and saving and grading are "
        "refused. Restore the journal from your backup and check it with "
        "`python bet_journal.py validate` (see docs/BET_JOURNAL.md), then reload."
        f"\n\n`{exc}`",
        icon=":material/gpp_bad:",
    )
    st.stop()

now = bj.now_utc()
local_now = now.astimezone(on.TORONTO)


def _md_safe(text: str) -> str:
    """Escape $ so Streamlit markdown doesn't treat a pair of amounts as a
    LaTeX math span."""
    return text.replace("$", "\\$")


def _show_result(key: str) -> None:
    result = st.session_state.pop(key, None)
    if result:
        getattr(st, result[0])(_md_safe(result[1]))


def _games_by_id() -> dict[str, dict]:
    return {g["game_id"]: g for g in _games()}


def _games() -> list[dict]:
    try:
        stat = bj.SCHEDULE_PATH.stat()
        schedule = load_schedule(str(bj.SCHEDULE_PATH), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return []
    return bj.selectable_games(schedule, now)


def _fresh_states() -> dict:
    """Current state read straight from disk (not the cache) for checks."""
    return bj.current_state(bj.load_records(bj.JOURNAL_DIR))


def _placed_utc(prefix: str):
    s = st.session_state
    return bj.toronto_to_utc(s[f"{prefix}_date"], s[f"{prefix}_time"])


def _terms_args(prefix: str, game_id: str) -> dict:
    """The wager terms described by a form's submitted values."""
    s = st.session_state
    if s.get(f"{prefix}_spread") is None or s.get(f"{prefix}_odds") is None:
        raise bj.JournalError("enter the spread and the American odds")
    if not (s.get(f"{prefix}_stake") or "").strip():
        raise bj.JournalError("enter the stake in CAD")
    return {"sportsbook_key": s[f"{prefix}_book"],
            "sportsbook_name": (s.get(f"{prefix}_book_name") or "").strip() or None,
            "game_id": game_id, "team": s[f"{prefix}_team"],
            "handicap": float(s[f"{prefix}_spread"]), "odds": int(s[f"{prefix}_odds"]),
            "stake": s[f"{prefix}_stake"].strip(),
            "placed_at": on.iso(_placed_utc(prefix)),
            "reference": (s.get(f"{prefix}_ref") or "").strip() or None,
            "note": (s.get(f"{prefix}_note") or "").strip() or None}


def _build(args: dict) -> tuple[dict, str]:
    return bj.build_terms(**{**args, "placed_at": on.parse_utc(args["placed_at"])},
                          now=bj.now_utc(), schedule_path=bj.SCHEDULE_PATH)


def _summary(terms: dict) -> str:
    return (f"**{bj.bet_text(terms)}** - stake **\\${terms['stake_cad']} CAD** at "
            f"{terms['sportsbook_name']}, placed {bj.toronto_text(terms['placed_at'])} "
            f"({terms['placed_at']} UTC)")


# ---- callbacks: grading ---------------------------------------------------

def _on_grade() -> None:
    try:
        written = bj.grade(bj.JOURNAL_DIR, bj.SCHEDULE_PATH, now=bj.now_utc())
    except (bj.JournalError, bj.IntegrityError, OSError, ValueError) as exc:
        st.session_state["bj_grade_result"] = ("error", f"Grading stopped: {exc}")
        return
    pending = sum(1 for s in _fresh_states().values() if not s.void and not s.grade)
    msg = (f"Wrote {len(written)} grade record(s)." if written else
           "No new grades: every result is unchanged.") + \
        (f" {pending} wager(s) still pending (no final score yet)." if pending else "")
    st.session_state["bj_grade_result"] = ("success" if written else "info", msg)


# ---- callbacks: new wager -------------------------------------------------
# Preview and Save are submit buttons of one form, handled in click callbacks
# that read the values submitted with that click. Save writes only if those
# values equal the previewed ones (otherwise: preview again), at most once per
# preview (token), and bet_journal re-checks duplicates under its lock.

def _new_args() -> dict:
    game_id = st.session_state.get("bj_game")
    if game_id not in _games_by_id():          # none, or a label that's no longer current
        raise bj.JournalError("choose the game again")
    args = _terms_args("bj", game_id)
    args["second"] = bool(st.session_state.get("bj_second"))
    return args


def _on_preview() -> None:
    st.session_state.pop("bj_pending", None)
    try:
        args = _new_args()
        terms, _ = _build({k: v for k, v in args.items() if k != "second"})
        states_now = _fresh_states()
        same_ref = bj.same_reference(terms, states_now)
        if same_ref:
            raise bj.JournalError(f"sportsbook reference {terms['reference']!r} is already "
                                  f"recorded ({', '.join(same_ref)}); correct that wager instead")
        identical = bj.identical_wagers(terms, states_now)
        if identical and not args["second"]:
            raise bj.DuplicateWager(
                f"this wager is already recorded ({', '.join(identical)}). If you really placed "
                "a separate second wager on identical terms, tick the separate-wager box and "
                "preview again")
    except (ValueError, bj.IntegrityError) as exc:
        st.session_state["bj_result"] = ("error", f"Not valid: {exc}")
        return
    st.session_state["bj_pending"] = {"token": uuid.uuid4().hex, "args": args, "terms": terms,
                                      "confirmed_existing": identical}


def _on_save() -> None:
    pending = st.session_state.get("bj_pending")
    if not pending:
        st.session_state["bj_result"] = ("error", "Nothing to save: preview the wager first.")
        return
    saved = st.session_state.setdefault("bj_saved_tokens", set())
    if pending["token"] in saved:
        st.session_state.pop("bj_pending", None)
        st.session_state["bj_result"] = ("info", "This wager was already saved.")
        return
    try:
        args, error = _new_args(), None
    except ValueError as exc:
        args, error = None, exc
    if args != pending["args"]:
        st.session_state.pop("bj_pending", None)
        st.session_state["bj_result"] = (
            "error", "Not saved: the form changed after Preview. Check the values and preview "
                     "again." + (f" ({error})" if error else ""))
        return
    try:
        terms, sha = _build({k: v for k, v in args.items() if k != "second"})
        if terms != pending["terms"]:
            raise bj.JournalError("the game's schedule entry changed since Preview; preview again")
        doc = bj.prepare_wager(terms, sha, confirmed_existing=pending["confirmed_existing"],
                               now=bj.now_utc())
        path = bj.save_wager(doc, bj.JOURNAL_DIR)
    except (ValueError, bj.IntegrityError, OSError) as exc:
        st.session_state.pop("bj_pending", None)
        st.session_state["bj_result"] = ("error", f"Not saved: {exc}")
        return
    saved.add(pending["token"])
    st.session_state.pop("bj_pending", None)
    st.session_state["bj_second"] = False
    st.session_state["bj_result"] = ("success", f"Saved wager {doc['record_id']}: "
                                                f"{bj.bet_text(terms)}, ${terms['stake_cad']} CAD "
                                                f"({path.name}).")


def _on_discard() -> None:
    st.session_state.pop("bj_pending", None)


# ---- callbacks: amend / void ----------------------------------------------

def _fix_args() -> dict:
    s = st.session_state
    wid, action = s.get("bjf_wager"), s.get("bjf_action")
    if not (isinstance(wid, str) and bj._ID_RE.match(wid)):
        raise bj.JournalError("choose the wager again")
    args = {"wager_id": wid, "action": action, "reason": (s.get("bjf_reason") or "").strip()}
    if not args["reason"]:
        raise bj.JournalError("a reason is required")
    if action == "amend":
        args["terms"] = _terms_args("bjf", s["bjf_game_id"])
        args["second"] = bool(s.get("bjf_second"))
    return args


def _prepare_fix(args: dict, previous: str, confirmed: list[str]) -> dict:
    """The amendment or void record. `previous` is the wager's effective
    terms record at Preview; saving refuses if the wager changed since."""
    now_ = bj.now_utc()
    if args["action"] == "amend":
        terms, sha = _build(args["terms"])
        return bj.prepare_amendment(args["wager_id"], previous, terms, sha, args["reason"],
                                    confirmed_existing=confirmed, now=now_)
    return bj.prepare_void(args["wager_id"], previous, args["reason"], now=now_)


def _on_fix_preview() -> None:
    st.session_state.pop("bjf_pending", None)
    try:
        args = _fix_args()
        states_now = _fresh_states()
        current = states_now.get(args["wager_id"])
        if current is None or current.void:
            raise bj.JournalError(f"wager {args['wager_id']} is not an active wager")
        confirmed = []
        if args["action"] == "amend":
            terms, _ = _build(args["terms"])
            if terms == current.terms:
                raise bj.JournalError("the corrected terms are the same as the current terms")
            if bj.same_reference(terms, states_now, exclude=args["wager_id"]):
                raise bj.DuplicateWager(f"sportsbook reference {terms['reference']!r} belongs "
                                        "to another wager")
            confirmed = bj.identical_wagers(terms, states_now, exclude=args["wager_id"])
            if confirmed and not args["second"]:
                raise bj.DuplicateWager(
                    f"the corrected terms are identical to another current wager "
                    f"({', '.join(confirmed)}). If both really are separate wagers, tick the "
                    "separate-wager box and preview again; if this is the same wager entered "
                    "twice, void one instead")
        doc = _prepare_fix(args, current.terms_record_id, confirmed)
    except (ValueError, bj.IntegrityError) as exc:
        st.session_state["bjf_result"] = ("error", f"Not valid: {exc}")
        return
    st.session_state["bjf_pending"] = {"token": uuid.uuid4().hex, "args": args,
                                       "previous": current.terms_record_id,
                                       "confirmed_existing": confirmed,
                                       "terms": doc.get("terms"), "before": current.terms}


def _on_fix_save() -> None:
    pending = st.session_state.get("bjf_pending")
    if not pending:
        st.session_state["bjf_result"] = ("error", "Nothing to save: preview the correction.")
        return
    saved = st.session_state.setdefault("bj_saved_tokens", set())
    if pending["token"] in saved:
        st.session_state.pop("bjf_pending", None)
        st.session_state["bjf_result"] = ("info", "This correction was already saved.")
        return
    try:
        args, error = _fix_args(), None
    except ValueError as exc:
        args, error = None, exc
    if args != pending["args"]:
        st.session_state.pop("bjf_pending", None)
        st.session_state["bjf_result"] = (
            "error", "Not saved: the form changed after Preview. Preview again."
                     + (f" ({error})" if error else ""))
        return
    try:
        doc = _prepare_fix(args, pending["previous"], pending["confirmed_existing"])
        if doc.get("terms") != pending["terms"]:
            raise bj.JournalError("the game's schedule entry changed since Preview; preview again")
        save = bj.save_amendment if args["action"] == "amend" else bj.save_void
        path = save(doc, bj.JOURNAL_DIR)
    except (ValueError, bj.IntegrityError, OSError) as exc:
        st.session_state.pop("bjf_pending", None)
        st.session_state["bjf_result"] = ("error", f"Not saved: {exc}")
        return
    saved.add(pending["token"])
    st.session_state.pop("bjf_pending", None)
    st.session_state.pop("bjf_loaded", None)
    what = "Amended" if args["action"] == "amend" else "Voided"
    st.session_state["bjf_result"] = ("success", f"{what} wager {args['wager_id']} "
                                                 f"({doc['record_id']}, {path.name}).")


def _on_fix_discard() -> None:
    st.session_state.pop("bjf_pending", None)


# ---- form fields shared by the new-wager and amendment forms --------------

def _book_label(key: str) -> str:
    return bj.SPORTSBOOKS[key]


def _term_fields(prefix: str, game: dict) -> None:
    st.selectbox("Ontario sportsbook", list(bj.SPORTSBOOKS), format_func=_book_label,
                 key=f"{prefix}_book")
    st.text_input("Sportsbook name (only for \"Other Ontario-licensed\")", max_chars=80,
                  key=f"{prefix}_book_name")
    st.selectbox("Team you bet on", [game["away_team"], game["home_team"]],
                 format_func=lambda t: f"{bj.team_name(t)} ({t})", key=f"{prefix}_team")
    with st.container(horizontal=True):
        st.number_input("Spread (bettor-facing, signed)", min_value=-60.0, max_value=60.0,
                        step=0.5, value=None, format="%.1f", placeholder="e.g. +3.5 or -7",
                        key=f"{prefix}_spread",
                        help="Your team's handicap as the ticket shows it: -3.5 lays 3.5, "
                             "+3.5 gets 3.5, 0 is pick'em.")
        st.number_input("Accepted American odds", step=1, value=None,
                        placeholder="e.g. -110 or +105", key=f"{prefix}_odds")
        st.text_input("Stake (CAD)", max_chars=12, placeholder="e.g. 25.00",
                      key=f"{prefix}_stake")
    with st.container(horizontal=True):
        st.date_input("Placed on (Toronto)", key=f"{prefix}_date")
        st.time_input("Placed at (Toronto)", step=60, key=f"{prefix}_time")
    with st.container(horizontal=True):
        st.text_input("Sportsbook reference (optional)", max_chars=bj.MAX_TEXT,
                      key=f"{prefix}_ref", help="Bet ID or ticket number from the sportsbook.")
        st.text_input("Note (optional)", max_chars=bj.MAX_TEXT, key=f"{prefix}_note")


# ---- summary and grading --------------------------------------------------

tot = bj.totals(states)
cols = st.columns(5)
cols[0].metric("Current wagers", tot["wagers"],
               help="Not voided. Amended wagers count once, on their current terms.")
cols[1].metric("Record (W-L-P)", f"{tot['win']}-{tot['loss']}-{tot['push']}")
cols[2].metric("Net profit (CAD)", bj.cad(tot["net_profit_cad"]),
               help="Graded current wagers only.")
cols[3].metric("ROI", "n/a" if tot["roi_pct"] is None else f"{tot['roi_pct']}%",
               help="Net profit / total stake of graded current wagers (wins, losses and "
                    "pushes). Pending and voided wagers are excluded.")
cols[4].metric("Pending", f"{tot['pending']} ({bj.cad(tot['stake_pending_cad'])})",
               help="Current wagers without a verified grade (no completion evidence yet, "
                    "amended since grading, or unverified); not in profit or ROI.")
st.button("Grade settled wagers", icon=":material/sports_score:", on_click=_on_grade,
          key="bj_grade",
          help="Settles every current wager whose game has a final score in "
               "data_files/nfl_games_historical.csv. Nothing is graded until you click this.")
_show_result("bj_grade_result")
if tot["unverified"]:
    st.warning(f"**{tot['unverified']} unverified settlement(s).** The schedule no longer "
               "supports an earlier grade (e.g. a score was removed or became inconsistent). "
               "Those wagers are excluded from profit and ROI until the evidence supports a "
               "new grade; see Status note and History.", icon=":material/report:")
if tot["voided"]:
    st.caption(f"{tot['voided']} voided wager(s) excluded from totals; see History.")

# Stateful tabs: the selected tab survives reruns even when the messages
# above it change (otherwise the browser falls back to the first tab).
tab_current, tab_new, tab_fix, tab_hist, tab_help = st.tabs(
    ["Current wagers", "Record a wager", "Correct or void", "History (audit)", "Definitions"],
    key="bj_tab", on_change="rerun")

with tab_current:
    show_void = st.toggle("Include voided wagers", key="bj_show_void")
    rows = bj.current_rows(states, include_void=show_void)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Stake (CAD)": st.column_config.NumberColumn(format="$%.2f"),
            "Net profit (CAD)": st.column_config.NumberColumn(format="$%.2f")})
    else:
        st.info("No wagers recorded yet. Use **Record a wager** to add one you've placed.")

with tab_new:
    st.caption("Record a single-game spread wager **you have already placed**, exactly as the "
               "sportsbook accepted it. A wager placed before kickoff can be recorded later.")
    _show_result("bj_result")
    games = _games()
    if not games:
        st.info("No games with a usable kickoff in the schedule window.")
    else:
        by_id = {g["game_id"]: g for g in games}
        if st.session_state.get("bj_game") not in by_id:
            st.session_state.pop("bj_game", None)
        game = by_id[st.selectbox("Game", list(by_id), format_func=lambda g: by_id[g]["label"],
                                  key="bj_game")]
        pending = st.session_state.get("bj_pending")
        if pending and pending["args"]["game_id"] != game["game_id"]:
            st.session_state.pop("bj_pending", None)     # preview belongs to another game
            pending = None
        st.session_state.setdefault("bj_date", local_now.date())
        st.session_state.setdefault("bj_time", local_now.time().replace(second=0, microsecond=0))
        with st.form("bj_form", border=False):
            _term_fields("bj", game)
            st.checkbox("This is a separate, second wager on terms identical to one already "
                        "recorded (not a re-entry of the same wager)", key="bj_second")
            if pending:
                with st.container(border=True):
                    st.markdown(f"**Preview:** {_summary(pending['terms'])}")
                    st.caption("**User-confirmed placed wager.** Save writes exactly these "
                               "accepted terms to the journal; it does not place a bet. Change "
                               "anything and you'll need to preview again.")
                    if pending["terms"]["reference"]:
                        st.caption(f"Sportsbook reference: {pending['terms']['reference']}")
                    if pending["confirmed_existing"]:
                        st.warning("Recorded as a **separate second wager** alongside "
                                   + ", ".join(pending["confirmed_existing"]) + ".")
            with st.container(horizontal=True):
                st.form_submit_button("Preview", icon=":material/visibility:", key="bj_preview",
                                      on_click=_on_preview)
                st.form_submit_button("Save wager", type="primary", icon=":material/save:",
                                      key="bj_save", on_click=_on_save, disabled=not pending)
                st.form_submit_button("Discard preview", key="bj_discard",
                                      on_click=_on_discard, disabled=not pending)

with tab_fix:
    st.caption("Wagers are never edited or deleted. A correction adds an **amendment** (new "
               "terms for the same game) or a **void** record, linked to the wager and with a "
               "reason; the original stays in History.")
    _show_result("bjf_result")
    active = {s.wager_id: s for s in states.values() if not s.void}
    if not active:
        st.info("No current wagers to correct.")
    else:
        if st.session_state.get("bjf_wager") not in active:
            st.session_state.pop("bjf_wager", None)
        # The label must not change when the wager is amended (see _games_by_id),
        # so it shows only the ID and the game; the current terms follow below.
        wid = st.selectbox("Wager", sorted(active, reverse=True), key="bjf_wager",
                           format_func=lambda w: f"{w}: {active[w].terms['season']} wk "
                                                 f"{active[w].terms['week']}, "
                                                 f"{active[w].terms['away_team']} @ "
                                                 f"{active[w].terms['home_team']}")
        st.caption(f"Current terms: {_summary(active[wid].terms)}")
        st.radio("Action", ["amend", "void"], horizontal=True, key="bjf_action",
                 format_func={"amend": "Amend terms", "void": "Void wager"}.get)
        fpending = st.session_state.get("bjf_pending")
        if fpending and fpending["args"]["wager_id"] != wid:
            st.session_state.pop("bjf_pending", None)
            fpending = None
        t = active[wid].terms
        if st.session_state.get("bjf_loaded") != wid:       # another wager: start afresh
            st.session_state.update({"bjf_reason": "", "bjf_loaded": wid})
            st.session_state.pop("bjf_stake", None)
        if st.session_state.get("bjf_action") == "amend" and "bjf_stake" not in st.session_state:
            # prefill with the current terms (again if the fields were hidden by "void")
            placed = on.parse_utc(t["placed_at"]).astimezone(on.TORONTO)
            st.session_state.update({
                "bjf_book": t["sportsbook_key"],
                "bjf_book_name": t["sportsbook_name"] if t["sportsbook_key"] == "other_on" else "",
                "bjf_team": t["team"], "bjf_spread": t["handicap"], "bjf_odds": t["odds"],
                "bjf_stake": t["stake_cad"], "bjf_date": placed.date(),
                "bjf_time": placed.time().replace(tzinfo=None), "bjf_ref": t["reference"] or "",
                "bjf_note": t["note"] or "", "bjf_second": False})
        st.session_state["bjf_game_id"] = t["game_id"]
        with st.form("bjf_form", border=False):
            if st.session_state.get("bjf_action") == "amend":
                st.caption(f"Game (fixed): {t['away_team']} @ {t['home_team']}, "
                           f"{t['season']} week {t['week']}. To change the game, void this "
                           "wager and record a new one.")
                _term_fields("bjf", {"away_team": t["away_team"], "home_team": t["home_team"]})
                st.checkbox("The corrected terms match another current wager, and both are "
                            "separate wagers I placed", key="bjf_second")
            st.text_input("Reason (required)", max_chars=bj.MAX_TEXT, key="bjf_reason")
            if fpending:
                with st.container(border=True):
                    if fpending["args"]["action"] == "amend":
                        st.markdown(f"**Before:** {_summary(fpending['before'])}  \n"
                                    f"**After:** {_summary(fpending['terms'])}")
                    else:
                        st.markdown(f"**Void:** {_summary(fpending['before'])}")
                    if fpending["confirmed_existing"]:
                        st.warning("Confirmed as separate from identical current wager(s) "
                                   + ", ".join(fpending["confirmed_existing"]) + ".")
                    st.caption(f"Reason: {fpending['args']['reason']}. Save appends this record; "
                               "the original wager stays in History.")
            with st.container(horizontal=True):
                st.form_submit_button("Preview", icon=":material/visibility:",
                                      key="bjf_preview", on_click=_on_fix_preview)
                st.form_submit_button("Save correction", type="primary",
                                      icon=":material/save:", key="bjf_save",
                                      on_click=_on_fix_save, disabled=not fpending)
                st.form_submit_button("Discard preview", key="bjf_discard",
                                      on_click=_on_fix_discard, disabled=not fpending)

with tab_hist:
    st.caption("Every journal record, oldest first: wagers, amendments, voids, grades and "
               "invalidations, including grades superseded by a score correction or an "
               "amendment. **Link** shows the record each one follows. Nothing here is ever "
               "edited or deleted.")
    hist = bj.history_rows(records, states)
    if hist:
        st.dataframe(pd.DataFrame(hist), hide_index=True, width="stretch")
    else:
        st.info("The journal is empty.")

with tab_help:
    st.markdown(f"""
- **Placed wager:** a single-game spread bet you confirm you placed, with the
  sportsbook's accepted team, signed handicap, American odds and CAD stake.
- **Settlement:** your team's final margin plus your handicap. Above zero wins,
  zero pushes, below zero loses. A **win** earns stake x 100/|odds| (negative
  odds) or stake x odds/100 (positive odds), rounded to the cent. A **loss**
  loses the stake, and a **push** earns \\$0.
- **Grading evidence:** the schedule has no official "final" flag, so a game
  is graded only on conservative evidence of completion: exactly one schedule
  row; whole, non-negative scores that aren't 0-0; `result` = home - away,
  `total` = home + away and `overtime` 0 or 1; and a game day before today
  (Toronto). Anything else stays **pending**.
- **Unverified:** a wager whose grade the schedule no longer supports. An
  invalidation record withdraws the grade (both stay in History), and the
  wager leaves profit and ROI until new evidence supports a fresh grade.
- **Pending** and unverified wagers count toward *Pending* stake only,
  never profit or ROI.
- **ROI** = net profit / total stake of **graded** current wagers. Wins, losses
  and pushes are included, so a push adds stake and \\$0 profit. Pending and
  voided wagers are excluded.
- **Voided wagers** stay in History and are excluded from every total. An
  **amended** wager counts once, on its latest terms; a grade of earlier terms
  no longer applies.
- **Corrections** link to the record they replace. If the wager changed after
  your Preview (e.g. in another tab), Save refuses and asks you to preview
  again. Corrected terms identical to another current wager need the same
  explicit separate-wager confirmation as a new wager.
- **Duplicates:** two wagers are *identical* when sportsbook, game, team,
  handicap, odds, stake and placement minute all match (and their sportsbook
  references don't differ). An identical entry is refused as a duplicate
  unless you tick *separate second wager*. A reused sportsbook reference at
  the same sportsbook is always refused.
- **Integrity:** each record has an unkeyed SHA-256 checksum. It detects
  accidental damage, not a deliberate edit with a recomputed checksum.
- **Storage:** `data_files/bet_journal/`, one file per record. It is
  git-ignored (personal data), so it stays on this machine; back it up.
- Stake limit: \\${bj.MAX_STAKE:,.2f} CAD per wager. Times are America/Toronto
  and recorded to the minute.
""")

add_betting_oracle_footer()
