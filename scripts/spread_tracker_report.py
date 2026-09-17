"""Roll up `spread_tracker_log.csv` into a season-to-date sportsbook comparison.

Writes `data_files/spread_tracker_report.json`: per-book season-to-date mean/
mean-abs deviation from nflverse's `spread_line` (the "which book is closest
to nflverse" ranking), the best available line per side for each game, and an
anomaly list (a book's normalized spread more than `ANOMALY_THRESHOLD_PTS`
away from that game's field median that week - the generalized version of the
PlayNow divergence first spotted by hand). Run after `spread_tracker.py`, by
`spread-tracker.yml`; also fine to run by hand.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

DATA_DIR = "data_files"
LOG_PATH = os.path.join(DATA_DIR, "spread_tracker_log.csv")
OUT_PATH = os.path.join(DATA_DIR, "spread_tracker_report.json")

# The PlayNow divergence that motivated this report (CLE +11.5 / NYJ +6.0 vs.
# a ~+8.5 / ~+3.5 field) was 3+ points off the field - this floor catches that
# scale of anomaly without flagging ordinary half-point book-to-book noise.
ANOMALY_THRESHOLD_PTS = 1.5


def _per_book_bucket(df: pd.DataFrame) -> dict:
    """book_key -> {book_title, region, n_quotes, n_with_comparison,
    mean_deviation_pts, mean_abs_deviation_pts}.

    mean_deviation_pts' sign shows whether a book systematically shades
    toward the home side (+) or away side (-) relative to nflverse;
    mean_abs_deviation_pts is the direction-agnostic "how far off" ranking
    metric - the direct answer to "which book is closest to nflverse."
    """
    out: dict = {}
    for book_key, grp in df.groupby("book_key"):
        valid = pd.to_numeric(grp["deviation_pts"], errors="coerce").dropna()
        out[str(book_key)] = {
            "book_title": grp["book_title"].iloc[0],
            "region": grp["region"].iloc[0],
            "n_quotes": int(len(grp)),
            "n_with_comparison": int(len(valid)),
            "mean_deviation_pts": round(float(valid.mean()), 3) if len(valid) else None,
            "mean_abs_deviation_pts": round(float(valid.abs().mean()), 3) if len(valid) else None,
        }
    return out


def _best_lines_per_game(df: pd.DataFrame) -> list[dict]:
    """Per game: which book gives the most points to the home side (best
    number for a home-side bettor) and which gives the most to the away
    side (best for an away-side bettor)."""
    rows = []
    for (season, week, game_id), grp in df.groupby(["season", "week", "game_id"]):
        home_row = grp.loc[grp["home_point"].idxmax()]
        away_row = grp.loc[grp["away_point"].idxmax()]
        rows.append({
            "season": int(season), "week": int(week), "game_id": game_id,
            "home_team": grp["home_team"].iloc[0], "away_team": grp["away_team"].iloc[0],
            "best_home_book": home_row["book_key"], "best_home_book_title": home_row["book_title"],
            "best_home_point": float(home_row["home_point"]),
            "best_away_book": away_row["book_key"], "best_away_book_title": away_row["book_title"],
            "best_away_point": float(away_row["away_point"]),
        })
    rows.sort(key=lambda r: (r["season"], r["week"], r["game_id"]))
    return rows


def _anomalies(df: pd.DataFrame, threshold: float = ANOMALY_THRESHOLD_PTS) -> list[dict]:
    """Per (season, week, game_id), the field median `home_spread_normalized`
    across all books that week; flag any book more than `threshold` points
    away from it - sorted worst-first."""
    rows = []
    for (season, week, game_id), grp in df.groupby(["season", "week", "game_id"]):
        median = grp["home_spread_normalized"].median()
        for _, r in grp.iterrows():
            diff = float(r["home_spread_normalized"]) - float(median)
            if abs(diff) > threshold:
                rows.append({
                    "season": int(season), "week": int(week), "game_id": game_id,
                    "book_key": r["book_key"], "book_title": r["book_title"], "region": r["region"],
                    "book_home_spread": float(r["home_spread_normalized"]),
                    "field_median_home_spread": float(median),
                    "diff_from_field_median": round(diff, 3),
                })
    rows.sort(key=lambda r: -abs(r["diff_from_field_median"]))
    return rows


def build_report(log_path: str = LOG_PATH) -> dict:
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": os.path.basename(log_path),
        "note": ("deviation_pts = a book's home-side spread minus nflverse's "
                 "spread_line, in nflverse's home-favorite-positive convention. "
                 "A positive per-book mean means that book systematically gives "
                 "the home side more points than nflverse; mean_abs_deviation_pts "
                 "is the direction-agnostic 'how far off nflverse' ranking metric."),
        "overall": {"n_weeks": 0, "n_games": 0, "n_quotes": 0, "per_book": {}},
        "by_week": [],
        "best_line_per_game": [],
        "anomalies": [],
    }
    if not (os.path.exists(log_path) and os.path.getsize(log_path) > 0):
        return report

    df = pd.read_csv(log_path)
    if df.empty:
        return report

    report["overall"] = {
        "n_weeks": int(df["week"].nunique()),
        "n_games": int(df["game_id"].nunique()),
        "n_quotes": int(len(df)),
        "per_book": _per_book_bucket(df),
    }
    for week, grp in df.groupby("week"):
        report["by_week"].append({
            "week": int(week),
            "n_games": int(grp["game_id"].nunique()),
            "per_book": _per_book_bucket(grp),
        })
    report["by_week"].sort(key=lambda r: r["week"])

    report["best_line_per_game"] = _best_lines_per_game(df)
    report["anomalies"] = _anomalies(df)
    return report


def main() -> None:
    report = build_report()
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[spread_tracker_report] wrote {OUT_PATH}")

    o = report["overall"]
    if not o["per_book"]:
        print("[spread_tracker_report] no spread-tracker data yet")
        return

    print(f"[spread_tracker_report] {o['n_weeks']} week(s), {o['n_games']} games, "
          f"{o['n_quotes']} quotes")
    ranked = sorted(
        o["per_book"].items(),
        key=lambda kv: (kv[1]["mean_abs_deviation_pts"] is None, kv[1]["mean_abs_deviation_pts"] or 0),
    )
    print("[spread_tracker_report] closest to nflverse's line (season to date):")
    for book_key, stats in ranked:
        if stats["mean_abs_deviation_pts"] is None:
            continue
        print(f"   {stats['book_title']:<24} ({stats['region']})  "
              f"mean|dev|={stats['mean_abs_deviation_pts']:.2f}pt  "
              f"mean_dev={stats['mean_deviation_pts']:+.2f}pt  n={stats['n_with_comparison']}")

    if report["anomalies"]:
        print(f"[spread_tracker_report] {len(report['anomalies'])} anomalies flagged "
              f"(>{ANOMALY_THRESHOLD_PTS}pt off field median)")
        for a in report["anomalies"][:5]:
            print(f"   {a['game_id']}: {a['book_title']} at {a['book_home_spread']:+.1f} "
                  f"vs. field median {a['field_median_home_spread']:+.1f} "
                  f"({a['diff_from_field_median']:+.1f}pt)")


if __name__ == "__main__":
    main()
