"""Frozen Performance - the spread model scored only on predictions frozen
before kickoff.

Read-only view of the frozen-pregame spread report (scripts/
frozen_spread_report.py; docs/FROZEN_SPREAD_REPORT.md), built in memory from
the validated pregame snapshots, the schedule and the archived Ontario
captures. It writes nothing, makes no API calls and never substitutes
regenerated probabilities, current lines or an earlier snapshot.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

import frozen_performance as fp  # noqa: E402
from footer import add_betting_oracle_footer  # noqa: E402

fr = fp.fr

st.title("Frozen Performance")
st.caption("Evaluates the frozen pregame report's predictions and bet signals: each game "
           "scored with the spread probability and handicap captured before kickoff. "
           "It is not a record of any hand-picked list of picks, and not of placed wagers.")


@st.cache_data(max_entries=8, show_spinner="Reading frozen snapshots...")
def load_sources(fingerprint: tuple, snapshot_dir: str, schedule_path: str, capture_dir: str,
                 which: str) -> dict:
    """File-dependent inputs only, cached on the content fingerprint of every
    snapshot, capture and the schedule: any added, changed or deleted file
    gives a new key. Clock-dependent evaluation happens outside this cache."""
    return fp.load(Path(snapshot_dir), Path(schedule_path), Path(capture_dir), which)


def integrity_stop(exc: Exception) -> None:
    st.error("**Integrity error: a frozen snapshot, market capture or the schedule is missing "
             "or failed validation.** Nothing is shown rather than partial or older data. Fix "
             f"or restore the file, then reload.\n\n`{exc}`", icon=":material/gpp_bad:")
    st.stop()


snapshot_dir, schedule_path, capture_dir = fp.SNAPSHOT_DIR, fp.SCHEDULE_PATH, fp.CAPTURE_DIR

c1, c2, c3 = st.columns([1, 1, 2])
which = c3.radio(
    "Snapshot used per game", list(fp.SELECTIONS), format_func=fp.SELECTIONS.get,
    help="Latest: the last prediction captured strictly before kickoff (default). Earliest: "
         "the first one, usually days earlier at an older line. Either way the snapshot is "
         "chosen first; a game is never re-scored with another snapshot because it had a "
         "probability or a bet signal.")
try:
    fingerprint = fp.source_fingerprint(snapshot_dir, schedule_path, capture_dir)
    src = load_sources(fingerprint, str(snapshot_dir), str(schedule_path), str(capture_dir),
                       which)
except fp.IntegrityError as exc:
    integrity_stop(exc)

seasons = fp.season_options(src["schedule"])
if not seasons:
    st.info("The schedule has no completed regular-season games yet, so there is nothing to "
            "evaluate.")
    st.stop()
season = c1.selectbox("Season", seasons, index=0)
weeks, default_week = fp.week_options(src["schedule"], season)
ALL_WEEKS = 0                                   # option value for "All weeks" (no week filter)
week_labels = {ALL_WEEKS: "All weeks"}
week_labels.update({w: f"Week {w}" + ("" if full else " (provisional)") for w, full in weeks})
choices = [ALL_WEEKS] + [w for w, _ in weeks]
week = c2.selectbox("Week", choices, format_func=week_labels.get,
                    index=choices.index(default_week) if default_week in choices else 0)
week = None if week == ALL_WEEKS else week

as_of = fp.today_toronto()
try:
    report = fp.build(src, season, week, as_of)
except fp.IntegrityError as exc:
    integrity_stop(exc)

week_complete = None if week is None else dict(weeks).get(week)
if fp.provisional(report, week_complete):
    st.warning("**Provisional:** this view includes games without a final result yet (or a "
               "week not fully played). Figures will change as results arrive.",
               icon=":material/schedule:")
st.caption(f"Outcomes counted for games dated before {as_of:%a %b %d, %Y} (Toronto). "
           f"{report['as_of_semantics']}")

history = report["scope"]["snapshot_history"]
if history == "none":
    st.info("**No snapshot history exists**, so no frozen prediction can be evaluated. "
            "Completed games are listed under Coverage below.", icon=":material/info:")

# ---------------------------------------------------------------- summaries --
summ = fp.summaries(report)
t_all, t_sig, t_mkt, t_sim, t_actual = st.tabs(
    ["All frozen predictions", "Frozen bet signals", "Market comparison", "Simulated returns",
     "Actual wagers"])


def prediction_summary(tab, s: dict, what: str) -> None:
    m = s["metrics"]["overall"]
    with tab:
        st.markdown(f"**{what}** - frozen probability that the underdog covers its frozen "
                    "handicap. Pushes are excluded from the probability metrics.")
        a, b, c, d = st.columns(4)
        a.metric("Evaluated (win/loss)", s["evaluated"])
        b.metric("Pushes", s["push"])
        c.metric("Pending", s["pending"])
        d.metric("Excluded", s["excluded"])
        a, b, c, d = st.columns(4)
        a.metric("Model Brier", fp.metric_text(m["brier"]))
        b.metric("50% baseline Brier", fp.metric_text(m["brier_baseline_50"]))
        c.metric("Model log loss", fp.metric_text(m["log_loss"]))
        d.metric("50% baseline log loss", fp.metric_text(m["log_loss_baseline_50"]))
        if m["n"]:
            st.caption(f"Sample: {m['n']} game(s); underdog covered {m['covers']} "
                       f"({m['observed_cover_rate']:.0%}), mean frozen probability "
                       f"{m['mean_probability']:.1%}. Lower Brier/log loss is better. Small "
                       "samples say little; no significance is claimed.")
        else:
            st.caption("No evaluated games in this view.")
        if s["excluded_by_reason"]:
            st.caption("Excluded: " + ", ".join(f"{fp.STATUS_LABELS[k]} ({v})"
                                                 for k, v in s["excluded_by_reason"].items()))


prediction_summary(t_all, summ["all"], "All eligible frozen predictions")
prediction_summary(t_sig, summ["signal"], "Only games whose frozen prediction carried a bet signal")

with t_mkt:
    mc = report["market_comparison"]
    n_all = summ["all"]["metrics"]["overall"]["n"]
    st.markdown("Automated Ontario sportsbook feeds only, at the **exact** frozen handicap, "
                "devigged; manual Ontario observations and the US reference feed are never "
                "used. Model, market and 50% are scored on the same matched games.")
    if mc["status"] != "available":
        reason = mc["reason"][:1].upper() + mc["reason"][1:]
        st.info(f"**Market comparison unavailable.** {reason}", icon=":material/info:")
    else:
        st.caption(f"Matched sample: **{mc['matched_games']}** game(s), of {n_all} evaluated "
                   "overall.")
        a, b, c = st.columns(3)
        a.metric("Model Brier (matched)", fp.metric_text(mc["model"]["brier"]))
        b.metric("Ontario market Brier", fp.metric_text(mc["market"]["brier"]))
        c.metric("50% baseline Brier (matched)", fp.metric_text(mc["baseline_50"]["brier"]))
        a, b, c = st.columns(3)
        a.metric("Model log loss (matched)", fp.metric_text(mc["model"]["log_loss"]))
        b.metric("Ontario market log loss", fp.metric_text(mc["market"]["log_loss"]))
        c.metric("50% baseline log loss (matched)", fp.metric_text(mc["baseline_50"]["log_loss"]))
    st.caption("Market status per game: " + ", ".join(
        f"{k.replace('_', ' ')} {v}" for k, v in mc["market_status_counts"].items() if v))

with t_sim:
    sr = report["simulated_returns"]
    st.markdown(
        "**Simulated, not placed.** Frozen bet signals settled at the frozen underdog and "
        f"handicap, {fr.STAKE} units per bet. The frozen handicap is nflverse's line and is not "
        "necessarily a sportsbook offer. A push returns the stake (profit 0) and counts as a "
        "bet; ROI = net units / units staked on settled bets.")
    for key, title in (("archived_price", "Observed archived Ontario price"),
                       ("assumed_minus_110", "Assumed -110 (scenario only)")):
        x = sr[key]
        st.markdown(f"**{title}** - {x['label']}")
        a, b, c, d = st.columns(4)
        a.metric("Bets settled", x["bets"])
        b.metric("W-L-P", f"{x['wins']}-{x['losses']}-{x['pushes']}")
        c.metric("Net units", x["net_units"])
        d.metric("ROI", "n/a" if x["roi"] is None else f"{x['roi']:.1%}")
        extra = f"Signals selected: {x['signals_selected']}; not settled yet or excluded: " \
                f"{x['signals_not_settled']}"
        if x.get("settled_without_archived_price") is not None:
            extra += f"; settled but no archived exact-line price: " \
                     f"{x['settled_without_archived_price']}"
        st.caption(extra + ".")

with t_actual:
    st.info("Actual placed wagers are recorded only in the **Bet Journal** page. Nothing on "
            "this page reads or summarizes them; the simulated returns above are not betting "
            "results.", icon=":material/receipt_long:")

# ----------------------------------------------------------------- coverage --
st.subheader("Coverage")
st.caption(report["coverage"]["sample_note"][:1].upper() + report["coverage"]["sample_note"][1:]
           + ". Completed games without a selected frozen snapshot are counted below, never "
           "evaluated or backfilled.")
st.dataframe(fp.coverage_rows(report), hide_index=True, width="content")
gaps = report["coverage"]["completed_without_snapshot"]["after_first_snapshot_games"]
if gaps:
    st.caption("Missing a snapshot after tracking began: " + ", ".join(gaps))

# ------------------------------------------------------------------ results --
st.subheader("Results by game")
rows = fp.results_rows(report)
if rows.empty:
    st.info("No frozen predictions are selected for this view.", icon=":material/info:")
else:
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption("Team and handicap are the frozen underdog and its signed handicap as captured "
               "(bettor-facing, e.g. +3.5), never today's line. Spread result: win/loss/push for "
               "that team at that handicap.")

with st.expander("Limitations", icon=":material/info:"):
    for item in report["limitations"]:
        st.markdown(f"- {item}")

add_betting_oracle_footer()
