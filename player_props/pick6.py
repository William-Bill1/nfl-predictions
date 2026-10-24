"""
DraftKings Pick 6 calculator on the Player Props page: a historical comparison
at a line the user enters. No model is scored here (see docs/architecture.md,
"Stage 4 model provenance"): the calculator has no target game to build model
features for, and each prop model is trained for its own tier line.

The percentage is Laplace-smoothed from the player's recent games:
P(over) = (games above the line + 1) / (games + 2), and the other side is the
complement, (games not over + 1) / (games + 2), where "not over" is below or
equal to the line. It is not a model prediction, a calibrated win probability
or a betting recommendation. Its counting convention for results equal to the line
does not describe sportsbook settlement rules.
"""
from __future__ import annotations

import pandas as pd

RATE_LABEL = "Smoothed historical hit rate"
INTRO = ("Enter a player and a DraftKings Pick 6 line to compare it with that player's "
         "recent games: how often they went above, below or exactly at that line.")
EXPLANATION = ("Laplace-smoothed from this player's recent games. It is not a model "
               "prediction, a calibrated win probability or a betting recommendation.")
SIDE_LABEL = "Side with the higher smoothed rate"
AT_LINE_NOTE = ("Games exactly at the line are counted as not over here. This calculator's "
                "counting convention does not describe sportsbook settlement rules. Check the "
                "operator's rules separately.")
TIE_NOTE = "The two sides are tied; the calculator shows the not-over side."


def historical_hit_rate(values: pd.Series, line: float) -> dict:
    """Counts and the smoothed rate for one player's recent values at ``line``.

    A game counts as over only when the value is strictly above the line; a
    game exactly at the line is "not over" and adds to the other side, as the
    calculator has always counted it. The side and its rate follow the
    calculator's original rule: a side at 60% or more, otherwise the larger,
    with a tie going to the not-over side."""
    values = pd.Series(values, dtype=float)
    games = int(len(values))
    over = int((values > line).sum())
    under = int((values < line).sum())
    at_line = games - over - under
    prob_over = (over + 1) / (games + 2)
    prob_under = 1 - prob_over
    if prob_over >= 0.60:
        side, rate = "OVER", prob_over
    elif prob_under >= 0.60:
        side, rate = "UNDER", prob_under
    else:
        side = "OVER" if prob_over > prob_under else "UNDER"
        rate = max(prob_over, prob_under)
    return {"games": games, "over": over, "under": under, "at_line": at_line,
            "not_over": games - over, "prob_over": prob_over, "prob_under": prob_under,
            "side": side, "rate": rate, "tied": prob_over == prob_under}


def side_label(summary: dict) -> str:
    """What the displayed side actually counts. The complement of OVER is only
    "UNDER" when no game equalled the line; otherwise it is below-or-equal."""
    if summary["side"] == "OVER":
        return "OVER"
    return "NOT OVER (below or equal)" if summary["at_line"] else "UNDER"


def counts_text(summary: dict) -> str:
    """Games strictly above, strictly below and exactly at the line."""
    text = f"{summary['games']} games: {summary['over']} above the line, {summary['under']} below"
    if summary["at_line"]:
        text += f", {summary['at_line']} exactly at the line"
    return text


def hits_text(summary: dict) -> str:
    """The observed games behind the side shown."""
    if summary["side"] == "OVER":
        return f"{summary['over']} of {summary['games']} games above the line"
    if summary["at_line"]:
        return (f"{summary['not_over']} of {summary['games']} games not over the line "
                f"({summary['under']} below, {summary['at_line']} equal)")
    return f"{summary['under']} of {summary['games']} games below the line"


def formula_text(summary: dict) -> str:
    """The displayed side's formula with its numbers."""
    if summary["side"] == "OVER":
        return f"(games above + 1) / (games + 2) = ({summary['over']} + 1) / ({summary['games']} + 2)"
    name = "games not over" if summary["at_line"] else "games below"
    return f"({name} + 1) / (games + 2) = ({summary['not_over']} + 1) / ({summary['games']} + 2)"


def notes(summary: dict) -> list:
    """Notes the card must show for this result (equality, ties)."""
    found = []
    if summary["tied"]:
        found.append(TIE_NOTE)
    if summary["at_line"]:
        found.append(AT_LINE_NOTE)
    return found
