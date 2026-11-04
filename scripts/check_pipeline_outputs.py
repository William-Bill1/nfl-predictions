"""Output checks for `python nfl-gather-data.py`.

Used by the `pipeline-smoke` CI job and, with --require-current-run, as the
nightly workflow's publication gate (the step before tonight's outputs are
uploaded, committed or pushed). Exits non-zero (with a message) if any check
fails. Not a correctness proof of the models - it checks that the outputs are
this run's, are well formed, and settle games consistently:

- provenance: pipeline_run_manifest.json is valid and the predictions CSV and
  schedule on disk are byte-for-byte the files it recorded (SHA-256);
- with --require-current-run: the manifest was written by THIS workflow run
  (GITHUB_RUN_ID/GITHUB_RUN_ATTEMPT, GITHUB_SHA) from the code checked out now
  (code_sha256), so an older run's consistent pair can't pass as tonight's;
- structure: one predictions row per schedule row, same games in the same order;
- source fields: home/away team and the posted lines and odds equal the
  matching schedule row, after the pipeline's own normalization (a missing or
  infinite line is written as 0);
- probabilities finite and in [0, 1], and present wherever their market is
  valid (a game without a posted spread line, or a pick'em, has none); bet
  signals 0 or 1;
- settlement, decided exactly as the pipeline decides it
  (nfl-gather-data.final_score_mask, at the manifest's settled_as_of): pending
  or unverifiable games carry NO outcome or bet return (missing is correct);
  final games carry their scores, as in the schedule, and outcome fields and
  returns that follow from them.

model_metrics.json, model_feature_importances.csv and best_features_spread.txt
are not recorded in the manifest, so for them only structure is checked.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import pregame_snapshots as ps  # noqa: E402

DATA = Path("data_files")
PRED_NAME, SRC_NAME, METRICS_NAME = ps.PREDICTIONS_NAME, ps.SCHEDULE_NAME, "model_metrics.json"
FEATS_NAME = "best_features_spread.txt"
CODE_FILES = ("nfl-gather-data.py", "team_features.py")   # what the pipeline hashes as code_sha256
# The manifest is written at the end of the run and settled_as_of is taken at the
# start; the pipeline step's timeout bounds the gap.
MAX_RUN_SECONDS = 30 * 60

REQUIRED_PRED_COLS = {
    "prob_underdogCovered", "pred_spreadCovered_optimal", "spreadPush",
    "ev_spread", "edge_underdog_spread", "gameday", "season", "week",
}
REQUIRED_METRIC_KEYS = {
    "Spread Accuracy", "Spread MAE", "Optimal Spread Threshold",
    "Spread_EV_Analysis", "Spread_OOS_Test",
}
PROB_COLS = ("prob_underdogCovered", "prob_overHit", "prob_underdogWon")
SIGNAL_COLS = ("pred_spreadCovered_optimal", "pred_underdogWon_optimal", "pred_overHit_optimal")
# Outcome fields whose value can legitimately be missing or infinite for a final
# game (ratios over a 0 or missing line): settled but not re-derived here.
RATIO_COLS = ("gameLineAccuracy", "overUnderAccuracy", "total_line_diff")
# Outcome fields taken straight from the schedule (the final-score evidence).
SCHEDULE_FIELDS = ("home_score", "away_score", "result", "total", "overtime")
# Copied unchanged from the schedule row: team identities (never renamed by the
# pipeline) and the posted market, which the pipeline writes after its
# whole-frame fillna(0) and +/-inf -> 0 (so an unposted line is written as 0).
TEAM_COLS = ("home_team", "away_team")
LINE_COLS = ("spread_line", "total_line", "home_moneyline", "away_moneyline",
             "home_spread_odds", "away_spread_odds", "over_odds", "under_odds")
EXAMPLES = 5


def _pipeline():
    spec = importlib.util.spec_from_file_location("nfl_gather_data_for_checks", REPO / "nfl-gather-data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)            # import only: the pipeline runs under main()
    return mod


def _ids(df: pd.DataFrame, mask) -> str:
    ids = df.loc[mask, "game_id"].astype(str).tolist() if "game_id" in df.columns else []
    more = f" (+{len(ids) - EXAMPLES} more)" if len(ids) > EXAMPLES else ""
    return ", ".join(ids[:EXAMPLES]) + more


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _differs(a: pd.Series, b, atol=1e-6) -> np.ndarray:
    a, b = np.asarray(_num(a), dtype=float), np.asarray(b, dtype=float)
    same = np.isclose(a, b, rtol=0, atol=atol) | (np.isnan(a) & np.isnan(b))
    return ~same


def _as_written(s: pd.Series) -> np.ndarray:
    """A schedule market column as the pipeline writes it: fillna(0), then +/-inf -> 0."""
    v = _num(s).to_numpy(dtype=float)
    return np.where(np.isfinite(v), v, 0.0)


def _finite_prob(s: pd.Series) -> np.ndarray:
    p = _num(s).to_numpy(dtype=float)
    return np.isfinite(p) & (p >= 0.0) & (p <= 1.0)


class Checker:
    def __init__(self, data_dir: Path, require_current_run: bool, env: dict, repo: Path):
        self.data, self.require, self.env, self.repo = Path(data_dir), require_current_run, env, repo
        self.errors: list[str] = []
        self.info: dict = {}

    def check(self, cond, msg: str) -> bool:
        if not cond:
            self.errors.append(msg)
        return bool(cond)

    # ------------------------------------------------------------ inputs --
    def load(self):
        """(manifest, schedule, predictions). The frames come from the bytes whose
        hashes were checked against the manifest. If provenance fails, the files
        are still read (unverified) so the remaining checks can report too."""
        try:
            manifest, sched, pred = ps.load_verified_inputs(self.data)
            self.info["provenance"] = "verified"
            return manifest, sched, pred
        except (ps.ProvenanceError, OSError, ValueError, KeyError) as exc:
            self.check(False, f"run provenance: {exc}")
            self.info["provenance"] = "FAILED"
        manifest = None
        try:
            manifest = json.loads((self.data / ps.MANIFEST_NAME).read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                manifest = None
        except (OSError, ValueError):
            pass
        frames = []
        for name in (SRC_NAME, PRED_NAME):
            try:
                frames.append(ps.parse_tsv((self.data / name).read_bytes()))
            except FileNotFoundError:
                self.check(False, f"missing {self.data / name}")
                frames.append(None)
            except Exception as exc:                          # malformed / truncated
                self.check(False, f"{name} could not be parsed: {type(exc).__name__}: {exc}")
                frames.append(None)
        return manifest, frames[0], frames[1]

    # ------------------------------------------------------ current run --
    def current_run(self, m: dict | None):
        if not self.require:
            return
        if not self.check(m is not None, "current run: no readable manifest"):
            return
        want = {k: self.env.get(k) for k in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA")}
        missing = [k for k, v in want.items() if not v]
        if not self.check(not missing, f"current run: {', '.join(missing)} not set - "
                                       "--require-current-run is for the CI workflow"):
            return
        ci = m.get("ci_run") if isinstance(m.get("ci_run"), dict) else {}
        for key in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
            self.check(ci.get(key) == want[key],
                       f"current run: manifest {key} {ci.get(key)!r} != this run's {want[key]!r} "
                       "(outputs from another run)")
        self.check(m.get("code_revision") == want["GITHUB_SHA"],
                   f"current run: manifest code_revision {m.get('code_revision')!r} != GITHUB_SHA {want['GITHUB_SHA']!r}")
        self.check(m.get("code_dirty") is False, f"current run: code_dirty is {m.get('code_dirty')!r}, expected false")
        code = hashlib.sha256()
        try:
            for name in sorted(CODE_FILES):
                code.update(name.encode() + b"\0" + (self.repo / name).read_bytes())
            self.check(m.get("code_sha256") == code.hexdigest(),
                       "current run: code_sha256 doesn't match the checked-out "
                       f"{' + '.join(CODE_FILES)}")
        except OSError as exc:
            self.check(False, f"current run: can't hash the pipeline code: {exc}")
        try:
            generated = ps._parse_utc(m.get("generated_at"))
            settled = ps._parse_utc(m.get("settled_as_of"))
        except ValueError as exc:
            self.check(False, f"current run: {exc}")
            return
        now = datetime.now(timezone.utc)
        self.check(generated <= now + timedelta(minutes=5), f"current run: generated_at {m['generated_at']} is in the future")
        self.check(settled <= generated <= settled + timedelta(seconds=MAX_RUN_SECONDS),
                   f"current run: settled_as_of {m['settled_as_of']} is not within "
                   f"{MAX_RUN_SECONDS // 60} min before generated_at {m['generated_at']}")

    # ---------------------------------------------------------- frames --
    def structure(self, sched: pd.DataFrame, df: pd.DataFrame) -> bool:
        """True when the two frames line up row for row."""
        missing = REQUIRED_PRED_COLS - set(df.columns)
        self.check(not missing, f"{PRED_NAME} missing columns: {sorted(missing)}")
        self.check(len(df) == len(sched), f"{PRED_NAME} has {len(df)} rows, expected {len(sched)} (same as source)")
        if not self.check("game_id" in df.columns and "game_id" in sched.columns, "game_id column missing"):
            return False
        if len(df) != len(sched):
            return False
        self.check(df["game_id"].is_unique, f"{PRED_NAME} lists a game more than once: "
                                            f"{_ids(df, df['game_id'].duplicated())}")
        same = df["game_id"].astype(str).to_numpy() == sched["game_id"].astype(str).to_numpy()
        if not self.check(same.all(), f"{PRED_NAME} games differ from the schedule (row order or IDs): "
                                      f"{_ids(df, ~same)}"):
            return False
        for col in ("season", "week", "gameday"):
            if col in df.columns and col in sched.columns:
                bad = df[col].astype(str).to_numpy() != sched[col].astype(str).to_numpy()
                self.check(not bad.any(), f"{PRED_NAME} {col} differs from the schedule: {_ids(df, bad)}")
        return True

    def lines(self, sched, df):
        # Games without a valid line (missing or 0 in the raw schedule) must
        # carry no spread probability and no spread signal - the CSV's own
        # spread_line is fillna(0)'d, so check against the source.
        if {"prob_underdogCovered", "pred_spreadCovered_optimal"} <= set(df.columns):
            no_line = (sched["spread_line"].isna() | (sched["spread_line"] == 0)).to_numpy()
            self.check(df.loc[no_line, "prob_underdogCovered"].isna().all(),
                       "games without a spread line have a prob_underdogCovered")
            self.check((df.loc[no_line, "pred_spreadCovered_optimal"] == 0).all(),
                       "games without a spread line carry a spread bet signal")

    def source_fields(self, sched, df):
        """Teams and the posted market must be the matching schedule row's, as the
        pipeline writes them - checked, never filled in from the schedule."""
        cols = TEAM_COLS + LINE_COLS
        for frame, name in ((df, PRED_NAME), (sched, SRC_NAME)):
            missing = [c for c in cols if c not in frame.columns]
            self.check(not missing, f"{name} missing team/line columns: {missing}")
        for col in TEAM_COLS:
            if col not in df.columns or col not in sched.columns:
                continue
            src, out = sched[col], df[col]
            no_team = src.isna().to_numpy()
            self.check(not no_team.any(), f"{SRC_NAME} has no {col} for {int(no_team.sum())} game(s): "
                                          f"{_ids(sched, no_team)}")
            bad = ~no_team & (out.isna().to_numpy() | (out.astype(str).to_numpy() != src.astype(str).to_numpy()))
            self.check(not bad.any(), f"{PRED_NAME} {col} differs from the schedule on {int(bad.sum())} game(s): "
                                      f"{_ids(df, bad)}")
        for col in LINE_COLS:
            if col not in df.columns or col not in sched.columns:
                continue
            junk = (sched[col].notna() & _num(sched[col]).isna()).to_numpy()
            self.check(not junk.any(), f"{SRC_NAME} {col} has non-numeric values: {_ids(sched, junk)}")
            out = _num(df[col]).to_numpy(dtype=float)
            bad = ~np.isfinite(out) | ~np.isclose(out, _as_written(sched[col]), rtol=0, atol=1e-9)
            self.check(not bad.any(), f"{PRED_NAME} {col} differs from the schedule (missing/inf written as 0) "
                                      f"on {int(bad.sum())} game(s): {_ids(df, bad)}")

    def probabilities(self, sched, df):
        """Each probability is present exactly where its market is valid, by the
        pipeline's own rules: spread - a posted, non-zero spread_line in the raw
        schedule (missing lines and pick'ems have none); totals - over_odds;
        moneyline - the underdog's moneyline (the higher of the two)."""
        if not set(PROB_COLS) <= set(df.columns):
            return                                            # values() reports the missing column
        need = {"spread_line", "over_odds", "home_moneyline", "away_moneyline"}
        if not need <= set(sched.columns):
            return                                            # source_fields() reports it
        line = _num(sched["spread_line"])
        away, home = _as_written(sched["away_moneyline"]), _as_written(sched["home_moneyline"])
        valid = {
            "prob_underdogCovered": (line.notna() & (line != 0)).to_numpy(),
            "prob_overHit": _as_written(sched["over_odds"]) != 0,
            "prob_underdogWon": np.where(away > home, away, home) != 0,
        }
        flagged = (df["pred_spreadCovered_optimal"] == 1).to_numpy() if "pred_spreadCovered_optimal" in df.columns \
            else np.zeros(len(df), bool)
        for col, market in valid.items():
            absent = market & ~_finite_prob(df[col])
            note = f", {int((absent & flagged).sum())} of them flagged" if col == "prob_underdogCovered" else ""
            self.check(not absent.any(), f"{int(absent.sum())} game(s) with a valid market have no finite {col} "
                                         f"in [0,1]{note}: {_ids(df, absent)}")
            if col != "prob_underdogCovered":                 # lines() covers the spread with its own message
                extra = ~market & df[col].notna().to_numpy()
                self.check(not extra.any(), f"{int(extra.sum())} game(s) without odds for {col} carry one: "
                                            f"{_ids(df, extra)}")

    def values(self, df):
        for col in PROB_COLS:
            if not self.check(col in df.columns, f"{PRED_NAME} missing column {col}"):
                continue
            raw, p = df[col], _num(df[col])
            self.check(not (raw.notna() & p.isna()).any(), f"{col} has non-numeric values: {_ids(df, raw.notna() & p.isna())}")
            bad = p.notna() & ~(np.isfinite(p) & p.between(0.0, 1.0))
            self.check(not bad.any(), f"{col} out of [0,1] or non-finite: min={p.min()}, max={p.max()} "
                                      f"({_ids(df, bad)})")
        if "prob_underdogCovered" in df.columns:
            p = _num(df["prob_underdogCovered"]).dropna()
            self.check(len(p) > 0, "prob_underdogCovered is all-NaN")
            self.check(p.nunique() > 10, f"prob_underdogCovered nearly constant ({p.nunique()} unique values)")
        for col in SIGNAL_COLS:
            if self.check(col in df.columns, f"{PRED_NAME} missing column {col}"):
                bad = ~df[col].isin([0, 1])
                self.check(not bad.any(), f"{col} must be 0 or 1: {_ids(df, bad)}")
        if "pred_spreadCovered_optimal" in df.columns:
            sig = int((df["pred_spreadCovered_optimal"] == 1).sum())
            self.check(sig > 0, "no spread bet signals at all (pred_spreadCovered_optimal)")
            self.check(sig < len(df), "every game is a spread signal (threshold broken?)")

    # ------------------------------------------------------ settlement --
    def settlement(self, m: dict | None, sched, df):
        gd = _pipeline()
        # The pipeline writes the outcome fields it can derive: a schedule without
        # nflverse's result/overtime columns has none of those (and then no game
        # has valid final evidence). Anything the schedule provides must be there.
        cols = [c for c in gd.OUTCOME_COLUMNS + gd.BET_RETURN_COLUMNS
                if c in df.columns or c in sched.columns or c not in SCHEDULE_FIELDS]
        absent = [c for c in cols if c not in df.columns]
        if not self.check(not absent, f"{PRED_NAME} missing outcome/return columns: {absent}"):
            return
        try:
            as_of = ps._parse_utc((m or {}).get("settled_as_of"))
        except ValueError:
            self.check(False, "manifest has no valid settled_as_of (the clock the run settled games with); "
                              "it was written by an older pipeline - can't check settlement")
            return
        final = gd.final_score_mask(sched, as_of).to_numpy()
        realized = df[cols].notna().to_numpy()
        self.info.update(final=int(final.sum()), pending=int((~final).sum()), settled_as_of=m["settled_as_of"],
                         flagged_pending=int(((df["pred_spreadCovered_optimal"] == 1).to_numpy() & ~final).sum()))

        # Pending or unverifiable: every outcome and return missing (a 0-0, a 0 or -100 is not a result).
        bad = ~final & realized.any(axis=1)
        if bad.any():
            which = sorted({c for c, hit in zip(cols, realized[bad].any(axis=0)) if hit})
            self.check(False, f"{int(bad.sum())} pending or unverifiable game(s) carry realized outcomes or "
                              f"returns ({', '.join(which)}): {_ids(df, bad)}")

        # Final: settled fields present, equal to the schedule's evidence, and consistent.
        core = [c for c in cols if c not in RATIO_COLS]
        holes = final & ~df[core].notna().to_numpy().all(axis=1)
        self.check(not holes.any(), f"{int(holes.sum())} final game(s) are missing outcome or return fields: "
                                    f"{_ids(df, holes)}")
        f = pd.Series(final, index=df.index)
        if not f.any():
            return
        s, d = sched.loc[f], df.loc[f]
        hs, as_ = _num(s["home_score"]), _num(s["away_score"])
        for col in SCHEDULE_FIELDS:            # all present: final evidence requires them
            self._expect(d, col, _num(s[col]), "the schedule's final-score evidence")
        line = _num(s["spread_line"])
        home_fav, away_fav = line > 0, line < 0
        pick = ~(home_fav | away_fav)
        margin = np.where(home_fav, hs - as_, as_ - hs)
        fav_line = line.abs()
        total_line = _num(s["total_line"]) if "total_line" in s.columns else pd.Series(np.nan, index=s.index)
        derived = {
            "homeWin": hs > as_, "awayWin": as_ > hs,
            "isCloseGame": (hs - as_).abs() <= 3, "isBlowout": (hs - as_).abs() >= 20,
            "totalScore": hs + as_, "pointDiff": (hs - as_).abs(),
            "spreadCovered": ~pick & (margin > fav_line), "favoriteCovered": ~pick & (margin > fav_line),
            "spreadPush": ~pick & (margin == fav_line), "underdogCovered": ~pick & (margin < fav_line),
            "underdogWon": (home_fav & (as_ > hs)) | (away_fav & (hs > as_)),
            "overHit": hs + as_ > total_line, "underHit": total_line > hs + as_,
            "totalHit": total_line == hs + as_,
        }
        for col, exp in derived.items():
            self._expect(d, col, np.asarray(exp, dtype=float), "the final score")
        # Spread return (the live signal): -110 on the underdog; push refunds; no bet -> 0.
        flag = d["pred_spreadCovered_optimal"] == 1
        ud = np.asarray(derived["underdogCovered"], bool)
        push = np.asarray(derived["spreadPush"], bool)
        self._expect(d, "spread_bet_return", np.where(~flag, 0.0, np.where(push, 0.0, np.where(ud, 90.91, -100.0))),
                     "the signal and the final score", atol=1e-6)
        # Moneyline / totals (disabled models): no bet -> 0; a bet is -100, 0 or a positive payout.
        for col, sig in (("moneyline_bet_return", "pred_underdogWon_optimal"), ("totals_bet_return", "pred_overHit_optimal")):
            r, bet = _num(d[col]), d[sig] == 1
            wrong = (~bet & (r != 0)) | (bet & ~((r == -100) | (r >= 0)))
            self.check(not wrong.any(), f"{col} inconsistent with {sig} on {int(wrong.sum())} final game(s): "
                                        f"{_ids(d, wrong)}")

    def _expect(self, d, col, expected, source, atol=1e-6):
        bad = _differs(d[col], expected, atol)
        self.check(not bad.any(), f"{col} disagrees with {source} on {int(bad.sum())} final game(s): {_ids(d, bad)}")

    # --------------------------------------------------- other outputs --
    def metrics(self):
        path = self.data / METRICS_NAME
        if not self.check(path.exists(), f"missing {path}"):
            return
        try:
            m = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            self.check(False, f"{METRICS_NAME} is not valid JSON: {exc}")
            return
        missing = REQUIRED_METRIC_KEYS - set(m)
        self.check(not missing, f"{METRICS_NAME} missing keys: {sorted(missing)}")
        acc = m.get("Spread Accuracy")
        self.check(isinstance(acc, (int, float)) and 0.0 <= acc <= 1.0, f"Spread Accuracy not a probability: {acc!r}")
        thr = m.get("Optimal Spread Threshold")
        self.check(isinstance(thr, (int, float)) and 0.4 <= thr <= 0.95, f"Optimal Spread Threshold implausible: {thr!r}")
        oos = m.get("Spread_OOS_Test") or {}
        self.check(oos.get("slice") == "test", f"Spread_OOS_Test.slice != 'test': {oos!r}")
        ev = m.get("Spread_EV_Analysis") or {}
        self.check(ev.get("slice") == "validation", f"Spread_EV_Analysis.slice != 'validation': {ev!r}")
        feats = self.data / FEATS_NAME
        self.check(feats.exists() and feats.stat().st_size > 0, f"missing/empty {feats}")

    # ------------------------------------------------------------- run --
    def run(self) -> list[str]:
        manifest, sched, pred = self.load()
        if manifest is not None:
            self.info.update({k: manifest.get(k) for k in ("run_id", "generated_at", "code_revision")})
            self.info["ci_run"] = manifest.get("ci_run")
            self.check(bool(manifest.get("run_id")) and bool(manifest.get("feature_set_id"))
                       and bool(manifest.get("config_id")), "pipeline_run_manifest.json missing run/config/feature ids")
        self.current_run(manifest)
        if sched is not None and pred is not None:
            self.info["rows"] = len(pred)
            if self.structure(sched, pred):
                self.lines(sched, pred)
                self.source_fields(sched, pred)
                self.probabilities(sched, pred)
                self.values(pred)
                self.settlement(manifest, sched, pred)
        self.metrics()
        return self.errors


def check_outputs(data_dir=DATA, *, require_current_run=False, env=None, repo=REPO) -> tuple[list[str], dict]:
    c = Checker(data_dir, require_current_run, dict(os.environ if env is None else env), Path(repo))
    return c.run(), c.info


def report(errors: list[str], info: dict) -> str:
    """Markdown summary for the job log / step summary: identities and counts only."""
    lines = [f"### Pipeline output gate: {'FAILED' if errors else 'passed'}", ""]
    for key in ("run_id", "generated_at", "settled_as_of", "code_revision", "ci_run", "provenance",
                "rows", "final", "pending", "flagged_pending"):
        if key in info:
            lines.append(f"- {key}: `{info[key]}`")
    if errors:
        lines += ["", f"{len(errors)} problem(s):"] + [f"- {e}" for e in errors]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--require-current-run", action="store_true",
                    help="also require the manifest to come from this CI run and the checked-out code")
    ap.add_argument("--report", type=Path, help="also write a Markdown summary here")
    args = ap.parse_args(argv)
    errors, info = check_outputs(args.data_dir, require_current_run=args.require_current_run)
    if args.report:
        args.report.write_text(report(errors, info), encoding="utf-8")
    if errors:
        print("PIPELINE SMOKE FAILED:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("pipeline smoke OK:", PRED_NAME, "+", METRICS_NAME, "look sane;",
          f"{info.get('final', 0)} final, {info.get('pending', 0)} pending (outcomes missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
