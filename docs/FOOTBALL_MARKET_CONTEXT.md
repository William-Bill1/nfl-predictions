# Manual FanDuel Ontario history beside weekly scorecards

The existing weekly generator accepts four optional arguments together:

- `--manual-quotes`: directory of original manual Ontario quote JSON files.
- `--market-schedule`: explicit reviewed schedule used to reassess kickoff times.
- `--market-as-of`: observation AND entry cutoff, ISO timestamp with timezone.
- `--model-snapshot`: one explicitly selected, validated frozen pregame snapshot.

Without these arguments the prior output is unchanged. With them, the generator
reuses fanduel_ontario_timeline.timeline and appends a read-only history section to
each card. No collector, request, schema migration or model calculation is added.
The single-game generator remains unchanged; the optional section is in weekly
batch outputs, including a batch containing a selected single game.

Rows show both sides, Toronto timestamps with offsets, handicaps, American odds,
points changes and price-only one-side implied-probability changes in percentage
points. Price-only changes are withheld when the handicap changes. Margin labels
include touching/crossing +/-3 and +/-7, not a claim about market causes.

Each card states: "Dates shown are observed dates across this week plus the
applicable as-of date—not a complete daily or planned-checkpoint calendar."
The dates are the Toronto dates of included observations for any game in the
selected season/week, plus the as-of Toronto date, limited to dates on or before
this game's Toronto kickoff date. A date nobody observed is not listed, so the
table does not show that every planned daily or near-kickoff checkpoint was
collected. A listed date without an observation for a team shows No observation;
earlier prices are never copied into it. Changes compare successive observations
even if a day is missing. All raw observations remain available in card JSON.
Conflicting same-time terms are retained and comparisons withheld.

The 2026 games moved by the NFL (Week 11: `2026_11_PIT_PHI` to 13:00 ET and
`2026_11_JAX_NYG` to 16:25 ET; Week 12: `2026_12_TEN_JAX` to 13:00 ET,
`2026_12_NE_LAC` to 16:05 ET and `2026_12_LV_CLE` to 20:20 ET) are checked with
`kickoff_corrections.resolve`. The
market schedule must already list the reviewed time; a schedule with the old time
is refused before any output is created and is never corrected in memory. Any
other changed identity, date or time for these games is refused as needing
review. Quote filtering and the model-snapshot timing check then use the reviewed
kickoff.

The model column is the snapshot's signed line for each side, not a recalculated
probability. A different available handicap flags LINE MISMATCH. A probability at
+3.5 cannot be transferred to +3. A pick'em model line shows as 0 for both sides;
no underdog, signal or probability is shown or inferred for it. Missing model
lines remain unavailable. Snapshot
capture must precede both the as-of cutoff and the reviewed game kickoff; team,
season/week and schedule date must match the scorecard. Default current files are
not silently substituted for any of the explicit inputs.

The index and card JSON retain hashes of the schedule, frozen snapshot and quote
inventory. Inputs are rechecked before writing. Existing output is refused and
INDEX.json remains the final completion marker. A failed write may leave partial
output without that marker. Original quotes, inputs and model outputs are read-only.

These are first/last observed prices, not established opening/closing prices or
live quotes. No fair probabilities, injury/weather attribution, ROI, staking or
validated betting value are claimed. API US FanDuel prices are never substituted.
