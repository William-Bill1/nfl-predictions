"""Model-vs-market line shopping: extend the spread model's own probability
(evaluated only against nflverse's consensus spread_line) out to every
tracked sportsbook's ACTUAL posted line for the same game, and compare
against what each book's own price requires.

The model never sees individual sportsbook lines - prob_underdogCovered is
computed once, against nflverse's spread_line. A book offering the underdog
MORE points than that is strictly easier to cover (monotonic - more cushion
never hurts), so the model's implied probability at that book's specific
line is always at least as high as at nflverse's own number, and can be
estimated with the same normal margin-of-victory approximation used
elsewhere in this project (see spread_value_finder.py). This is an
EXTRAPOLATION of the model's read onto a line it never actually scored, not
literally the model's output - every result is labeled to make that clear.

Distinct from spread_value_finder.py, which asks "is this book's price good
relative to the OTHER TRACKED BOOKS" (field median). This asks "is this
book's price good relative to what OUR OWN MODEL thinks" - a different
baseline, only computed for games the model actually has a prediction for.

CLI: python scripts/model_line_shop.py --game 2026_02_SEA_ARI
     python scripts/model_line_shop.py --week 2 --season 2026   (all current picks)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spread_value_finder import MOV_SIGMA, _normal_cdf, _normal_ppf  # noqa: E402

DATA_DIR = "data_files"
LOG_PATH = os.path.join(DATA_DIR, "spread_tracker_log.csv")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "nfl_games_historical_with_predictions.csv")

EDGE_COLUMNS = [
    "season", "week", "game_id", "underdog_team", "favorite_team",
    "book_key", "book_title", "region",
    "book_line", "book_price",
    "nflverse_line", "model_prob_at_nflverse_line", "extrapolated_prob",
    "required_prob", "edge_pts",
]


def extrapolate_prob(known_prob: float, known_points: float, target_points: float,
                      sigma: float = MOV_SIGMA) -> float:
    """Given the model's probability of the underdog covering `known_points`
    (nflverse's own line, in standard plus-points notation), estimate the
    probability of covering `target_points` (a specific book's own line) -
    same normal-margin-of-victory approximation as spread_value_finder.py,
    calibrated to the model's own read instead of the field median.

    Identity check: extrapolate_prob(p, L, L, sigma) == p for any p, L, sigma
    (asking about the exact same line the model was evaluated on).
    """
    z = _normal_ppf(known_prob)
    fair_points = known_points - z * sigma
    return _normal_cdf((target_points - fair_points) / sigma)


def compute_model_edges_for_game(game_id: str, predictions_df: pd.DataFrame,
                                  log_df: pd.DataFrame, sigma: float = MOV_SIGMA) -> pd.DataFrame:
    """One row per tracked book that quoted this game, ranked nowhere here
    (caller sorts). Returns an empty DataFrame - never raises - when the game
    isn't in predictions_df, has no spread_line (pick'em or not yet posted),
    or isn't in the tracker log.
    """
    pred = predictions_df[predictions_df["game_id"] == game_id]
    if pred.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    pred = pred.iloc[0]

    spread_line = pred.get("spread_line")
    if spread_line is None or pd.isna(spread_line) or spread_line == 0:
        # Pick'em or not yet posted - no clear underdog to anchor the
        # extrapolation on. Not an error, just nothing to compute.
        return pd.DataFrame(columns=EDGE_COLUMNS)

    underdog_is_home = spread_line < 0  # nflverse convention: negative = away favored = home is the dog
    underdog_team = pred["home_team"] if underdog_is_home else pred["away_team"]
    favorite_team = pred["away_team"] if underdog_is_home else pred["home_team"]
    nflverse_line = abs(float(spread_line))
    model_prob = pred.get("prob_underdogCovered")
    if model_prob is None or pd.isna(model_prob):
        return pd.DataFrame(columns=EDGE_COLUMNS)

    books = log_df[log_df["game_id"] == game_id]
    if books.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    rows = []
    for _, b in books.iterrows():
        if underdog_is_home:
            book_line, book_price = b["home_point"], b["home_price"]
            required = b["home_implied_prob_devigged"]
        else:
            book_line, book_price = b["away_point"], b["away_price"]
            required = b["away_implied_prob_devigged"]
        if required is None or pd.isna(required):
            continue

        extrapolated = extrapolate_prob(model_prob, nflverse_line, float(book_line), sigma)
        rows.append({
            "season": int(b["season"]), "week": int(b["week"]), "game_id": game_id,
            "underdog_team": underdog_team, "favorite_team": favorite_team,
            "book_key": b["book_key"], "book_title": b["book_title"], "region": b["region"],
            "book_line": float(book_line), "book_price": int(book_price),
            "nflverse_line": nflverse_line, "model_prob_at_nflverse_line": float(model_prob),
            "extrapolated_prob": extrapolated, "required_prob": float(required),
            "edge_pts": (extrapolated - required) * 100,
        })

    return pd.DataFrame(rows, columns=EDGE_COLUMNS)


def _load_predictions() -> pd.DataFrame:
    # Despite the .csv extension this file is TAB-separated (see
    # betting_log.py's own PREDICTIONS_PATH read) - a plain pd.read_csv()
    # silently parses it as one unsplit column with no error.
    return pd.read_csv(PREDICTIONS_PATH, sep="\t")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extend the spread model's own probability to every tracked book's actual "
                    "line, for one game or every current model pick in a week."
    )
    ap.add_argument("--game", default=None, help="a single game_id, e.g. 2026_02_SEA_ARI")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--include-non-picks", action="store_true",
                     help="also show games where pred_spreadCovered_optimal != 1 "
                          "(default: only the model's actual current picks)")
    ap.add_argument("--sigma", type=float, default=MOV_SIGMA)
    ap.add_argument("--top", type=int, default=None)
    args = ap.parse_args()

    if not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0:
        print(f"[model_line_shop] {LOG_PATH} not found or empty - run spread_tracker.py first")
        raise SystemExit(1)
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"[model_line_shop] {PREDICTIONS_PATH} not found")
        raise SystemExit(1)

    predictions_df = _load_predictions()
    log_df = pd.read_csv(LOG_PATH)

    if args.game:
        game_ids = [args.game]
    else:
        candidates = predictions_df
        if args.season is not None:
            candidates = candidates[candidates["season"] == args.season]
        if args.week is not None:
            candidates = candidates[candidates["week"] == args.week]
        if not args.include_non_picks:
            candidates = candidates[candidates.get("pred_spreadCovered_optimal", 0) == 1]
        game_ids = candidates["game_id"].dropna().unique().tolist()

    if not game_ids:
        scope = args.game or f"season={args.season} week={args.week}"
        pick_note = "" if args.include_non_picks else " with pred_spreadCovered_optimal==1"
        print(f"[model_line_shop] no games found for {scope}{pick_note}")
        raise SystemExit(0)

    all_edges = pd.concat(
        [compute_model_edges_for_game(gid, predictions_df, log_df, sigma=args.sigma) for gid in game_ids],
        ignore_index=True,
    )
    if all_edges.empty:
        print("[model_line_shop] no priceable rows (pick'em games, missing lines, "
              "or games not yet tracked by spread_tracker.py)")
        raise SystemExit(0)

    all_edges = all_edges.sort_values("edge_pts", ascending=False).reset_index(drop=True)
    if args.top:
        all_edges = all_edges.head(args.top)

    print(f"[model_line_shop] {len(all_edges)} book quotes across "
          f"{all_edges['game_id'].nunique()} game(s), sigma={args.sigma}\n")
    for gid, grp in all_edges.groupby("game_id", sort=False):
        first = grp.iloc[0]
        print(f"{first['underdog_team']} vs {first['favorite_team']} "
              f"({gid}) - model: {first['underdog_team']} +{first['nflverse_line']:.1f} -> "
              f"{first['model_prob_at_nflverse_line']:.1%} (nflverse's line)")
        for _, r in grp.sort_values("edge_pts", ascending=False).iterrows():
            sign = "+" if r["book_line"] >= 0 else ""
            print(f"   {r['book_title']:<24} {sign}{r['book_line']:.1f} ({r['book_price']:+.0f})  "
                  f"extrapolated {r['extrapolated_prob']:.1%} / required {r['required_prob']:.1%}"
                  f"  ->  edge {r['edge_pts']:+.1f}pt")
        print()

    print(
        "[model_line_shop] CAVEATS: 'extrapolated' is NOT the model's actual output - "
        "it's the model's nflverse-line probability extended to each book's specific "
        f"line via a normal margin-of-victory approximation (sigma={args.sigma}), which "
        "assumes the model's read scales smoothly across nearby lines. The underlying "
        "spread model is only ~break-even out-of-sample (see model_metrics.json -> "
        "Spread_OOS_Test) - this ranks books given the model's read, it does not "
        "validate that the model's read is correct."
    )


if __name__ == "__main__":
    main()
