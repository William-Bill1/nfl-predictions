"""Price-adjusted value finder: which side of a book's own line is actually
mispriced, once you account for BOTH the point number and the price.

Reads data_files/spread_tracker_log.csv (see spread_tracker.py). For one
target book, each game it quotes is compared against the field median line
from every OTHER tracked book that same week - not to see whether the book's
POINT number is generous (a book can move the points and shade the price to
compensate, netting out to a fair bet - see the CLE/TB case in
docs/MARKET_SPREAD_TRACKER_PLAN.md's chat history), but whether the
resulting bet is priced favorably against a fair-value estimate.

"Fair" win probability comes from a standard normal approximation of NFL
margin of victory (mean = field's median line, std dev = MOV_SIGMA) evaluated
at the target book's own point number - NOT scipy, just math.erf, since
scipy isn't a project dependency and this is one line of math.

All output uses standard bettor-facing spread notation (favorite negative,
e.g. "NE -2.5"), read directly from the log's own home_point/away_point
columns - never re-derived from spread_tracker.py's internal
home-favorite-positive convention, which is what caused a labeling bug
during a manual version of this analysis (2026-09-17). See
TestComputeBookEdges::test_labels_match_raw_bettor_facing_columns.

CLI: python scripts/spread_value_finder.py --book playnow_ca --week 2
"""
from __future__ import annotations

import argparse
import math
import os

import pandas as pd

DATA_DIR = "data_files"
LOG_PATH = os.path.join(DATA_DIR, "spread_tracker_log.csv")

# Commonly-cited NFL margin-of-victory standard deviation, used to convert a
# point spread into a win/cover probability. Not fitted on this project's own
# data - a standard industry approximation, not a validated model.
MOV_SIGMA = 13.5

# A game needs at least this many OTHER tracked books to trust their median
# as a "field" estimate, rather than one or two books that could themselves
# be outliers.
MIN_FIELD_BOOKS = 2

EDGE_COLUMNS = [
    "season", "week", "game_id", "side", "team", "opponent",
    "line", "price", "fair_prob", "required_prob", "edge_pts",
]


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via math.erf - no scipy dependency for one line of math."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def _fair_prob_home_covers(mu_field: float, book_home_spread_normalized: float,
                            sigma: float = MOV_SIGMA) -> float:
    """P(home team covers the book's own line), given the true home-margin
    distribution is Normal(mu_field, sigma) - mu_field and
    book_home_spread_normalized both in nflverse's home-favorite-positive
    convention (spread_tracker.py's home_spread_normalized column).

    Sanity check: when book_home_spread_normalized == mu_field (the book's
    line matches the field exactly), this returns exactly 0.5, as it should -
    the book's own line is by definition the 50/50 point of its own
    distribution.
    """
    return _normal_cdf((mu_field - book_home_spread_normalized) / sigma)


def compute_book_edges(log_df: pd.DataFrame, book_key: str, season: int | None = None,
                        week: int | None = None, sigma: float = MOV_SIGMA,
                        min_field_books: int = MIN_FIELD_BOOKS) -> pd.DataFrame:
    """One row per side (home/away) per game the target book quoted, ranked
    nowhere here (caller sorts) - fair_prob vs required_prob (the target
    book's own devigged break-even probability for that exact side), and
    edge_pts = (fair_prob - required_prob) * 100. Positive edge_pts means
    that side is priced favorably relative to the field-implied fair value;
    negative means the book's price already more than compensates for
    however generous the points look.

    Games where fewer than `min_field_books` other books quoted the same
    (season, week, game_id) are skipped - not enough of a field to trust a
    median. Rows where the target book's own price can't be devigged
    (missing/invalid odds) are skipped too.
    """
    if log_df is None or log_df.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    target = log_df[log_df["book_key"] == book_key]
    if season is not None:
        target = target[target["season"] == season]
    if week is not None:
        target = target[target["week"] == week]
    if target.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    rows = []
    for _, t in target.iterrows():
        field = log_df[
            (log_df["season"] == t["season"]) & (log_df["week"] == t["week"])
            & (log_df["game_id"] == t["game_id"]) & (log_df["book_key"] != book_key)
        ]
        if len(field) < min_field_books:
            continue

        req_home = t.get("home_implied_prob_devigged")
        if req_home is None or pd.isna(req_home):
            continue

        mu_field = field["home_spread_normalized"].median()
        l_book = t["home_spread_normalized"]
        fair_home = _fair_prob_home_covers(mu_field, l_book, sigma)
        fair_away = 1.0 - fair_home
        req_away = 1.0 - req_home

        common = {"season": int(t["season"]), "week": int(t["week"]), "game_id": t["game_id"]}
        rows.append({
            **common, "side": "home", "team": t["home_team"], "opponent": t["away_team"],
            "line": float(t["home_point"]), "price": int(t["home_price"]),
            "fair_prob": fair_home, "required_prob": req_home,
            "edge_pts": (fair_home - req_home) * 100,
        })
        rows.append({
            **common, "side": "away", "team": t["away_team"], "opponent": t["home_team"],
            "line": float(t["away_point"]), "price": int(t["away_price"]),
            "fair_prob": fair_away, "required_prob": req_away,
            "edge_pts": (fair_away - req_away) * 100,
        })

    return pd.DataFrame(rows, columns=EDGE_COLUMNS)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Price-adjusted value finder for one tracked sportsbook's spread lines "
                    "(reads data_files/spread_tracker_log.csv - run spread_tracker.py first)."
    )
    ap.add_argument("--book", required=True, help="book_key, e.g. playnow_ca, draftkings")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None, help="default: all tracked weeks for this book")
    ap.add_argument("--sigma", type=float, default=MOV_SIGMA,
                     help=f"NFL margin-of-victory std dev used for the fair-value estimate (default {MOV_SIGMA})")
    ap.add_argument("--top", type=int, default=None, help="only show the top N edges (default: all)")
    args = ap.parse_args()

    if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
        print(f"[spread_value_finder] {LOG_PATH} not found or empty - run spread_tracker.py first")
        raise SystemExit(1)

    log_df = pd.read_csv(LOG_PATH)
    edges = compute_book_edges(log_df, args.book, season=args.season, week=args.week, sigma=args.sigma)

    if edges.empty:
        print(f"[spread_value_finder] no priceable rows for book '{args.book}' "
              f"(season={args.season}, week={args.week}) - check the book_key is right "
              f"and at least {MIN_FIELD_BOOKS} other books quoted the same games")
        raise SystemExit(0)

    edges = edges.sort_values("edge_pts", ascending=False).reset_index(drop=True)
    if args.top:
        edges = edges.head(args.top)

    print(f"[spread_value_finder] {args.book}: {len(edges)} priced sides "
          f"(sigma={args.sigma}, fair value vs. field median of other tracked books)\n")
    for _, r in edges.iterrows():
        sign = "+" if r["line"] >= 0 else ""
        print(f"  wk{r['week']:>2}  {r['team']:>3} {sign}{r['line']:.1f} ({r['price']:+.0f}) "
              f"vs {r['opponent']:<3}  fair {r['fair_prob']:.1%} / req {r['required_prob']:.1%}"
              f"  ->  edge {r['edge_pts']:+.1f}pt")

    print(
        "\n[spread_value_finder] CAVEATS: fair-value uses a standard normal "
        f"margin-of-victory approximation (sigma={args.sigma}), not a validated "
        "model; assumes the field median is close to true probability; does "
        "NOT account for parlay-specific pricing (real multi-leg odds usually "
        "carry more hold than multiplying these single-leg no-vig prices "
        "would suggest); and is only as good as however many weeks are in "
        f"{os.path.basename(LOG_PATH)} - no track record of this signal "
        "holding up over a season."
    )


if __name__ == "__main__":
    main()
