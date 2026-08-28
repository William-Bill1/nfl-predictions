# Spread Model: "Inversion Fix" Investigation

**Status:** RESOLVED (Aug 2026, branch `refactor/spread-single-convention`).
**Scope:** `nfl-gather-data.py` spread model — label definitions, the `1 - p`
"inversion fix", and the threshold/EV code that consumes it.
**Question:** is the model genuinely "backwards", and is the current fix correct?

## Resolution

Single convention adopted end to end, without renaming the CSV columns:

- **P1** — the split-then-invert (`prob = blend(); prob = 1 - prob`) is now one
  line with an honest comment: `prob_underdogCovered = 1 - P(favorite covers)`.
  The "model is backwards" prints are gone.
- **P2** — `spread_ev_threshold()` now takes `P(underdog covers)` and
  `underdogCovered` labels directly (pushes removed), so the threshold and the
  bets it gates live in the same space.
- **P3** — `predictedSpreadCovered` is now the underdog-covers prediction
  (`prob_underdogCovered >= 0.5`); the dashboard's "Actual Spread" column now
  reads `underdogCovered`.
- **P4** — `Spread Accuracy` / `Spread MAE` measure underdog-covers on the
  held-out test set, push games excluded. A `"Spread Note"` in
  `model_metrics.json` states the convention.
- **P5** — new `spreadPush` column; `underdogCovered` uses strict `>` (a push is
  no longer an underdog cover); `spread_bet_return` refunds pushes (returns 0).
- **P6** — temporal split landed earlier (`refactor/temporal-split`).

Model training is unchanged (target is still `spreadCovered`, whose definition
did not change), verified byte-identical `model_feature_importances.csv` /
`best_features_spread.txt`. The rest of this doc is the original analysis.

---

---

## TL;DR

The model is **not** backwards. It is trained correctly to predict
`P(favorite covers the spread)`. The bug the "inversion fix" actually corrected
was a **variable mix-up**: the model's output was stored in a column named
`prob_underdogCovered` and fed — unchanged — into EV, threshold and bet-selection
logic that expects the underdog probability. `prob = 1 - prob` is the
mathematically correct conversion between two complementary events, so it works.

But the surrounding narrative ("model predicts backwards", "low confidence = high
accuracy") is misleading, and the code still **mixes the two conventions**:

- Bet predictions use the *inverted* probability (`prob_underdogCovered`).
- The EV-threshold routine that produces `optimal_spread_threshold` is computed
  in the *non-inverted* ("favorite covers") space.
- `model_metrics.json` "Spread Accuracy" and the CSV column
  `predictedSpreadCovered` are also *non-inverted*.

A future "cleanup" that removes the `1 - p` line, or renames things "for clarity",
will silently re-introduce the sign error. This doc records what a correct,
single-convention design looks like so that refactor can be done safely.

---

## What the code actually does

### 1. Label definitions (`nfl-gather-data.py:47-66`)

```
spread_line > 0  => home favored          (nflverse convention)      # :48-49
```

`spreadCovered` (`:52-53`) — **1 = the favorite covered**:
- home favored & `(home_score - away_score) >  spread_line`, or
- away favored & `(away_score - home_score) >  abs(spread_line)`
- strict `>`, so a **push = 0**.

`favoriteCovered` (`:54-57`) — identical formula to `spreadCovered`.

`underdogCovered` (`:58-61`) — **1 = the underdog covered**:
- home favored & `(away_score - home_score) + spread_line >= 0`, or
- away favored & `(home_score - away_score) - spread_line >= 0`
- `>=`, so a **push = 1** (assigned to the underdog).

So `underdogCovered == 1 - spreadCovered` in every case, with pushes landing on
the underdog side.

### 2. Training (`nfl-gather-data.py:242, 251-256, 285-290`)

```python
target_spread = 'spreadCovered'          # favorite covers = 1        # :242
y_spread = historical_game_level_data[target_spread]                  # :254
model_spread = CalibratedClassifierCV(XGBClassifier(...), 'isotonic', cv=5)
model_spread.fit(X_train_spread, y_spread_train)                      # :290
```

The model learns `P(favorite covers)`. This is fine.

### 3. The "inversion fix" (`nfl-gather-data.py:457-467`)

```python
historical_game_level_data['prob_underdogCovered'] = _blend_proba(model_spread, lgbm_spread, X_spread)  # :457
# ^ this value is P(favorite covers) -- but the column is named prob_underdogCovered
...
historical_game_level_data['prob_underdogCovered'] = 1 - historical_game_level_data['prob_underdogCovered']  # :465
# ^ now it is 1 - P(favorite covers) = P(favorite does NOT cover) ~= P(underdog covers)
```

After `:465` the column name is **correct**. `1 - p` is the right conversion
because `favorite covers` and `underdog covers` are complementary (pushes aside).

### 4. Why the metrics jumped (-90% -> +66% ROI, 3.6% -> 91.9% win rate)

Before `:465`, downstream code treated `P(favorite covers)` as
`P(underdog covers)`. Every spread bet was therefore selected on an inverted
signal — the pipeline bet the underdog exactly when the model was *most sure the
favorite would cover*. Flipping the probability flips every bet, which is why the
headline numbers invert too. It is a real fix; it is just mislabeled as "the
model is backwards" when the model was never the problem.

---

## Problems that remain

| # | Location | Issue |
|---|----------|-------|
| P1 | `:457` / `:465` | The model is trained on `spreadCovered` (favorite) but its output is **only ever used** as `prob_underdogCovered`. The name is wrong for one line and right for the next. Any edit that drops the `1 - p`, or "renames for clarity", reintroduces the sign bug. Docs calling the model "backwards" actively invite this. |
| P2 | `calculate_spread_ev_threshold` (`:385-434`), called at `:439` | `probs = _blend_proba(model_spread, ...)` here is **non-inverted** `P(favorite covers)`, and `y_test` passed in is `y_spread_test` = `spreadCovered` (favorite). So `optimal_spread_threshold` and its "historical accuracy on +EV bets" are computed in *favorite-covers* space. That threshold is then applied at `:538-541` to `prob_underdogCovered`, which is *underdog-covers* space. The two are mixed; the threshold may not mean what the bet-selection code assumes. |
| P3 | `:741-744` | `predictedSpreadCovered` is written from `y_spread_pred` (non-inverted, `>= 0.5` on favorite-covers). The CSV then carries both `predictedSpreadCovered` (favorite space) and `prob_underdogCovered` (underdog space) — two spread columns in opposite conventions. |
| P4 | `model_metrics.json` (`:787-797`) | "Spread Accuracy" / "Spread MAE" are on non-inverted `spreadCovered`. The dashboard's spread bets are on the inverted probability. The reported accuracy does not describe the bets being shown. |
| P5 | `:52-53` vs `:58-61` | Pushes: `spreadCovered` uses `>` (push = loss), `underdogCovered` uses `>=` (push = win). In real betting a push is a stake refund, not a win. This biases `underdogCovered` rate and every ROI figure derived from it upward slightly. |
| P6 | `:255-256`, `:262-263`, `:269-270` | `train_test_split(..., random_state=42, stratify=...)` — a **random** split. The "91.9% win rate" / "76.7% simulation ROI" headline numbers are measured on a test set chronologically interleaved with training games; they are optimistic. Rolling *features* are leak-free (`:84`, `:97`, `:108`), but the split is not temporal. |

---

## Recommended target design (for the future change — not done here)

Pick **one** convention and carry it end to end.

1. **Rename** `spreadCovered` -> `favoriteCovered` everywhere (the column
   `favoriteCovered` already exists and is identical — collapse to one). Keep the
   training target as `favoriteCovered`. Drop the word "backwards" from every doc.
2. **One derived probability, one place:**
   ```python
   prob_favorite_covers = _blend_proba(model_spread, lgbm_spread, X_spread)
   # underdog covering is the complement (pushes handled separately, see 4)
   prob_underdog_covers = 1.0 - prob_favorite_covers
   ```
   Feed **only** `prob_underdog_covers` to EV, thresholding and bet selection.
3. **Fix `calculate_spread_ev_threshold`** to operate on the underdog quantity:
   pass `prob_underdog_covers` and `y_test = underdogCovered` so the threshold
   lives in the same space as the bets it gates (resolves P2).
4. **Handle pushes consistently** in labels and ROI: exclude pushed games from
   accuracy/ROI, or score them as 0.5 (stake returned). Use the same comparator
   (`>`) for both `favoriteCovered` and `underdogCovered` and add an explicit
   `spreadPush` column (resolves P5).
5. **Align the CSV + metrics:** write a single `pred_underdog_covers` column and
   report "Spread Accuracy" against `underdogCovered`, so the reported number
   describes the bets the dashboard actually makes (resolves P3, P4).
6. **Switch to a temporal split** (`train: season <= N-1`, `test: season N`, or
   `TimeSeriesSplit`) and re-measure. Expect the 91.9% / 76.7% figures to fall —
   record the honest post-split numbers (resolves P6).

### How to validate the change

- Golden test: `underdogCovered == 1 - favoriteCovered` for all non-push rows;
  `spreadPush` rows excluded from both.
- Before/after backtest table in the PR: bets placed, win rate, ROI, calibration
  error — on a **temporal** holdout, current pipeline vs proposed.
- The proposed design must not depend on any `1 - p` line outside step 2.
- Merge only if the temporal-holdout ROI/calibration is at least as good; revert
  otherwise.

---

## Files/docs that repeat the "backwards" framing (update when the code changes)

- `docs/architecture.md` — "Critical: Spread Model Inversion Fix"
- `.github/copilot-instructions.md` — "Model Training - Spread Prediction Inversion"
- `README.md` — "Major Performance Breakthrough", "Spread Model Fixed ... 3.6% to 91.9%"
- `docs/MODEL_FIX_PLAN.md`, `docs/SPREAD_THRESHOLD_CHANGE.md`,
  `scripts/test_inversion_fix.py`, `scripts/verify_spread_fix.py`
