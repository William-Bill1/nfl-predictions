# Rolling spread backtest

`scripts/rolling_spread_backtest.py` scores the spread model the way it would
have been used: retrained every week on games played before that week, then
asked about that week's games. Use it to test any change to the spread model
before trusting the model's picks. It is read-only. It never changes training,
prediction CSVs, feature files, recommendations or the app.

## Run it

```bash
python scripts/rolling_spread_backtest.py --smoke              # 3 weeks, 25 trees, ~5 s
python scripts/rolling_spread_backtest.py                      # 2023 wk 1 -> latest completed week
python scripts/rolling_spread_backtest.py --start 2021-1 --end 2025-22 --n-boot 2000 --seed 7
```

| Option | Default | Meaning |
|---|---|---|
| `--start`, `--end` | `2023-1`, latest | Evaluated range, as `season-week` |
| `--cal-min-games` | 256 | Size of the calibration period: the most recent whole weeks holding at least this many games |
| `--min-fit-games` | 300 | Weeks with less fitting history are skipped and listed in `windows` |
| `--n-boot`, `--seed` | 1000, 42 | Bootstrap draws and seed. The same seed gives identical output |
| `--smoke` | off | 3 weeks, 25 trees, 50 bootstrap draws (for checks, not for conclusions) |
| `--output-dir` | `backtest_output/` | Git-ignored output folder |

A full run from 2021 takes a few minutes, single-threaded for reproducibility.

## Outputs (`backtest_output/`)

| File | Contents |
|---|---|
| `retrospective_report.json` | Per-season and overall metrics for every model, calibration bins, bootstrap intervals, and per-week sample counts (`windows`) |
| `retrospective_predictions.csv` | One row per game and model: probability, outcome, bet, price, price source, profit |
| `frozen_pregame_report.json` | The probabilities the nightly job actually logged before kickoff, settled on the recorded line |

The two reports are never combined:

- **Retrospective** is rebuilt from today's nflverse schedule using
  **closing** lines and odds. It estimates what the model would have said.
- **Frozen pregame** is what the model did say, but only for games that
  triggered a bet signal. Each probability is the first value logged for that
  game, not the last one before kickoff.

## Method

For each evaluated week W:

1. **Features.** Production's selected features (`best_features_spread.txt`)
   are built by `team_features.py`, the same code the production pipeline
   uses since Oct 2026. Only completed games from strictly earlier weeks count.
   Team rates such as win %, blowout % and cover % use each team's earlier
   home games, or earlier away games. A game's own result, later results and
   same-week results never affect its features. Score-derived columns such as
   `total` are refused. Before Oct 2026, production's cover, favored and
   over/under rates included every game, the predicted one too. This backtest
   was always leak-free, so its results don't measure that production fix.
2. **Training pool.** Completed games with a line from weeks before W. The
   script stops with an error if any of them kicked off after W's first game.
3. **Fit and calibration periods.** The most recent whole weeks with at least
   `--cal-min-games` games form the calibration period. Everything earlier is
   the fit period. Each week's actual counts are in `windows` (`fit_games`,
   `cal_games`, `fit_weeks`, `cal_weeks`, `eval_games`, `cutoff`).
4. **Models.**

   | Name | What it is |
   |---|---|
   | `production_config` | Production's estimator unchanged: XGBoost and LightGBM, each wrapped in `CalibratedClassifierCV(isotonic, cv=5)`, averaged, refit on the whole pool. Its calibration folds are not in time order, as in production |
   | `production_holdout_platt` | The same two models, uncalibrated, fit on the fit period only. A Platt (logistic) calibrator is then fit on their predictions for the calibration period. The models are never refit afterwards, so the calibrator always matches the model it was fit on |
   | `logistic_abs_spread` | Logistic regression of "underdog covers" on the absolute spread |
   | `constant_50` | 50% for every game |
   | `devig_closing` | The underdog's nflverse closing spread price, with the margin removed using the favorite's price. Used only where both prices exist |

5. **Bets.** A model bets the underdog when its probability is at least
   0.5438, production's rule: the −110 break-even of 52.38% plus 2 points.
   Profit is per $100 risked, at the underdog's closing price
   (`closing_odds_nflverse`), or at an assumed −110 when that price is
   missing (`assumed_-110`).
6. **Exclusions.** Games with a missing line, or a line of 0 (pick'em, so no
   underdog), are left out and counted in `excluded_games`. Pushes are left
   out of Brier score, log loss, accuracy and calibration, earn $0, and are
   counted separately. ROI = profit ÷ ($100 × bets that weren't pushes).
7. **Uncertainty.** 95% percentile intervals from a bootstrap that resamples
   whole weeks. The same weeks are drawn for every model, so
   `brier_minus_constant50_95ci` is a direct comparison with the 50% guess.

Frozen pregame prices: rows whose `odds_source` says "assumed" (all legacy
rows, priced at −110) are labelled `assumed_-110`. These are not real
sportsbook prices. Only rows with a recorded source count as `recorded`.

## Reading the results

- **Brier score and log loss** measure probability accuracy; lower is better.
  The 50% guess scores 0.2500 and 0.6931. A model is only adding information
  if it beats `constant_50`, and especially `devig_closing`, with a
  `brier_minus_constant50_95ci` interval entirely below zero.
- **Calibration bins** compare predicted probability with the actual cover
  rate. Probabilities above 0.6 should cover more than 60% of the time.
  Larger gaps, especially in bins with many games, mean the probabilities
  shouldn't be shown as confidence.
- **ROI** is very noisy at a few hundred bets. Read `roi_pct_95ci` before the
  point estimate. Retrospective ROI is at closing prices, which are usually
  harder to beat than the earlier lines bets are placed at.
- **The frozen report** contains logged bets only, so its accuracy can't be
  compared directly with the retrospective numbers, which cover every game.

## Dependency on `betting_log.py`

The script changes no production files. The frozen report imports
`betting_log._settle` and `betting_log.UNRESOLVED` and calls them without
modification. That ties the frozen pregame numbers to the dashboard's own
settlement rules: the recorded team and line, the recorded price or an
assumed −110, and "Pick" rows marked unresolved. Both names come from the
pricing and settlement change (`70d812a`, branch
`codex/fix-spread-pricing-settlement`). This script requires that change:
without it, `frozen_report` raises `AttributeError`. `_settle` is private, so a rename in
`betting_log.py` has to be made here too. The frozen-report tests cover it.

## Known limitations

- Lines and odds are nflverse closing values, not the line available when a
  pick was made. Weather, rest and similar inputs are as finally recorded.
- Team rates copy production's quirks: home-only or away-only splits, all
  history back to 2020, and 0 when a team has no history.
- Rows in the frozen report come from several pipeline versions.
- The nightly git snapshots of `nfl_games_historical_with_predictions.csv`
  (since Oct 2025) are not used yet.
