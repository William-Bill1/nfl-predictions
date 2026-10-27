"""
Player Props players-reference candidate builder and validator (offline).

Builds a CANDIDATE players reference from the existing reference
(data_files/players.csv) and a fresh nflverse players file that was
downloaded beforehand with recorded hashes. Nothing is adopted: the
candidate and its reports are written to an isolated output directory, and
data_files/players.csv is never touched.

Policy (approved 2026-10-26):
 1. Fresh source records are taken as they are, in the existing reference's
    column order, all values kept as text.
 2. Historical IDs absent from the fresh source are retained unchanged.
 3. A placeholder ID (a gsis_id that is not GSIS-format; in nflverse it
    equals the record's esb_id) is corrected to a GSIS ID only when its
    esb_id and its smart_id each match exactly one fresh record and both
    match the same record, whose gsis_id is GSIS-format, is not already an
    existing ID, and is claimed by no other placeholder. Anything else is
    rejected, reported, and the placeholder retained.
 4. Identity records are never filtered by roster membership or status:
    the weekly roster is evidence only (for one explicit season and week).
 5. People are never merged by name. Records sharing an esb_id or smart_id
    are reported as source aliases (one person, several records), not merged.

Invalid input (a missing required column; a missing, blank, padded,
malformed or repeated gsis_id; a missing, blank or padded position; a hash
that differs from the downloads manifest) raises ReferenceBuildError. A
gsis_id is valid only as a GSIS ID (00-00NNNNN) or a supported legacy
placeholder (its record's own esb_id, AAANNNNNN). Values are checked as
written and never trimmed. On any refusal an existing output directory is
left as it was: output is written to a staging directory and swapped in only
when complete. An output directory that overlaps the downloads directory, or
would replace the existing reference or candidates file, is refused before
anything is read (paths compared fully resolved, links and junctions
included).

CLI (offline; writes only to --out, which must be outside the repository):
    python player_props/reference_builder.py --existing data_files/players.csv \\
        --fresh-dir DOWNLOADS_DIR --out OUT_DIR \\
        [--candidates CANDIDATES_CSV] [--roster-week 2026:7]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from player_props.predict import POSITION_MAP  # noqa: E402
from player_props.rehearsal import RehearsalError, _overlaps, _real, check_workspace  # noqa: E402

PLAYERS_NAME = 'players.csv'
ROSTER_NAME = 'roster_weekly_{season}.csv'
MANIFEST_NAME = 'manifest.json'
REQUIRED_COLUMNS = ('gsis_id', 'display_name', 'short_name', 'last_name', 'position', 'esb_id', 'smart_id')
ROSTER_COLUMNS = ('season', 'week', 'gsis_id', 'team', 'position', 'status')
GSIS = re.compile(r'^00-00\d{5}$')
# nflverse's legacy placeholder: the record's own esb_id (three capitals, six
# digits) stored as gsis_id before a GSIS ID was assigned. Every non-GSIS ID
# in the 2025 and 2026 nflverse players files has this form.
PLACEHOLDER = re.compile(r'^[A-Z]{3}\d{6}$')
ELIGIBLE = frozenset(POSITION_MAP)
EXAMPLES = 10

# Who reads which reference field (for the field-change report).
CONSUMERS = {
    'gsis_id': ['player_props/predict.py (lookup key)', 'player_props/rehearsal*.py (lookup key)',
                'pages/2_Player_Props.py (player search)'],
    'position': ['player_props/predict.py (prop eligibility; output position)',
                 'player_props/rehearsal_expected.py (expected-row eligibility)',
                 'pages/2_Player_Props.py', 'pages/3_Parlay_Builder.py'],
    'display_name': ['player_props/predict.py (output display_name)', 'pages/2_Player_Props.py',
                     'pages/3_Parlay_Builder.py'],
    'short_name': ['pages/2_Player_Props.py (name merge key)', 'pages/3_Parlay_Builder.py (name merge key)'],
    'last_name': ['player_props/predict.py and pages/2_Player_Props.py (read, not used)'],
}


class ReferenceBuildError(RuntimeError):
    def __init__(self, problems: List[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_text_csv(path: Path) -> pd.DataFrame:
    """Every value as text, exactly as written; only empty cells are missing."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[''], low_memory=False)


def is_gsis(values: pd.Series) -> pd.Series:
    return values.fillna('').str.match(GSIS.pattern)


def _examples(values) -> list:
    return sorted(map(str, values))[:EXAMPLES]


def blank(values: pd.Series) -> pd.Series:
    """Missing, empty or whitespace-only. A check only: values are never trimmed."""
    return values.isna() | values.astype(str).str.strip().eq('')


def padded(values: pd.Series) -> pd.Series:
    """Non-blank, with leading or trailing whitespace. A check only."""
    s = values.astype(str)
    return ~blank(values) & s.ne(s.str.strip())


def id_kind(df: pd.DataFrame) -> pd.Series:
    """'gsis', 'placeholder' (a supported legacy placeholder: PLACEHOLDER form
    and equal to the record's own esb_id), or 'malformed'. Blank and padded
    IDs are reported separately and are never valid."""
    ids = df['gsis_id'].fillna('')
    kind = pd.Series('malformed', index=df.index)
    kind[ids.str.match(PLACEHOLDER.pattern) & ids.eq(df['esb_id'].fillna(''))] = 'placeholder'
    kind[ids.str.match(GSIS.pattern)] = 'gsis'
    return kind


def validate(df: pd.DataFrame, label: str) -> List[str]:
    """Problems that make a reference unusable. Values are checked as written:
    nothing is trimmed, so a padded value is refused rather than repaired."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return [f"{label}: missing required column(s) {missing}"]
    problems = []
    no_id, pad_id = blank(df['gsis_id']), padded(df['gsis_id'])
    if no_id.any():
        problems.append(f"{label}: {int(no_id.sum())} record(s) with a missing or blank gsis_id")
    if pad_id.any():
        problems.append(f"{label}: {int(pad_id.sum())} gsis_id(s) with surrounding whitespace: "
                        f"{', '.join(_examples(df.loc[pad_id, 'gsis_id'].map(repr)))}")
    bad = (id_kind(df) == 'malformed') & ~no_id & ~pad_id
    if bad.any():
        problems.append(f"{label}: {int(bad.sum())} gsis_id(s) neither GSIS (00-00NNNNN) nor a supported "
                        f"placeholder (its own esb_id, AAANNNNNN): {', '.join(_examples(df.loc[bad, 'gsis_id'].map(repr)))}")
    ids = df.loc[~no_id, 'gsis_id']
    dup = ids[ids.duplicated(keep=False)]
    if len(dup):
        problems.append(f"{label}: {dup.nunique()} gsis_id(s) on more than one record: {_examples(set(dup))}")
    nopos = df.loc[blank(df['position']), 'gsis_id']
    if len(nopos):
        problems.append(f"{label}: {len(nopos)} record(s) with a missing or blank position: "
                        f"{_examples(nopos.fillna('?'))}")
    pad_pos = padded(df['position'])
    if pad_pos.any():
        problems.append(f"{label}: {int(pad_pos.sum())} position(s) with surrounding whitespace: "
                        f"{_examples(df.loc[pad_pos, 'gsis_id'].fillna('?'))}")
    return problems


def check_output_location(out_dir: Path, fresh_dir: Path, inputs: List[Path]) -> None:
    """Refuse an output directory that would overwrite or delete an input.
    Paths are compared fully resolved (links and junctions). The downloads
    directory may be neither the output, inside it, nor contain it; an input
    file may be neither the output nor inside it (a replaced output is
    removed)."""
    out = _real(out_dir)
    problems = []
    fresh = _real(fresh_dir)
    if _overlaps(out, fresh):
        problems.append(f"output {out_dir} (resolves to {out}) overlaps the downloads directory "
                        f"{fresh_dir} (resolves to {fresh})")
    for p in inputs:
        r = _real(p)
        if r == out or r.is_relative_to(out):
            problems.append(f"output {out_dir} (resolves to {out}) would replace input {p} (resolves to {r})")
    if problems:
        raise ReferenceBuildError(problems)


def verify_downloads(fresh_dir: Path, names: List[str]) -> Dict[str, dict]:
    """Each named file against the downloads manifest (size and SHA-256)."""
    path = Path(fresh_dir) / MANIFEST_NAME
    if not path.exists():
        raise ReferenceBuildError([f"{path} is missing; only recorded downloads are used"])
    files = json.loads(path.read_text(encoding='utf-8')).get('files', {})
    out, problems = {}, []
    for name in names:
        rec, f = files.get(name), Path(fresh_dir) / name
        if rec is None or not f.exists():
            problems.append(f"{name}: not recorded in {MANIFEST_NAME} or missing")
            continue
        digest, size = sha256_file(f), f.stat().st_size
        if digest != rec.get('sha256') or size != rec.get('bytes'):
            problems.append(f"{name}: differs from {MANIFEST_NAME} (sha256 {digest}, {size} bytes)")
            continue
        out[name] = {'sha256': digest, 'bytes': size, 'url': rec.get('url'),
                     'retrieved': rec.get('retrieved_finished') or rec.get('fetched_at')}
    if problems:
        raise ReferenceBuildError(problems)
    return out


# --- placeholder crosswalk ---------------------------------------------------

def crosswalk(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """One row per existing placeholder ID that the fresh source no longer has."""
    fresh_ids, existing_ids = set(fresh['gsis_id']), set(existing['gsis_id'])
    by_esb = fresh.dropna(subset=['esb_id']).groupby('esb_id')['gsis_id'].apply(list).to_dict()
    by_smart = fresh.dropna(subset=['smart_id']).groupby('smart_id')['gsis_id'].apply(list).to_dict()
    rows = []
    for r in existing[~is_gsis(existing['gsis_id'])].to_dict('records'):
        pid = r['gsis_id']
        if pid in fresh_ids:
            continue                                  # still a source record: kept as it is
        esb = by_esb.get(r['esb_id'], []) if isinstance(r['esb_id'], str) else []
        smart = by_smart.get(r['smart_id'], []) if isinstance(r['smart_id'], str) else []
        row = {'old_id': pid, 'esb_id': r['esb_id'], 'smart_id': r['smart_id'],
               'esb_matches': ';'.join(esb), 'smart_matches': ';'.join(smart), 'new_id': None}
        if not esb and not smart:
            row['result'] = 'no_match'
        elif len(esb) > 1 or len(smart) > 1:
            row['result'] = 'rejected_ambiguous'
        elif not esb or not smart:
            row['result'] = 'rejected_single_id_match'
        elif esb != smart:
            row['result'] = 'rejected_conflicting'
        elif not GSIS.match(esb[0]):
            row['result'] = 'rejected_target_not_gsis'
        elif esb[0] in existing_ids:
            row['result'] = 'rejected_target_already_existing'
        else:
            row.update(result='corrected', new_id=esb[0])
        rows.append(row)
    cols = ['old_id', 'new_id', 'result', 'esb_id', 'smart_id', 'esb_matches', 'smart_matches']
    out = pd.DataFrame(rows, columns=cols)
    taken = out.loc[out['result'] == 'corrected', 'new_id']
    shared = set(taken[taken.duplicated(keep=False)])
    out.loc[out['new_id'].isin(shared), 'result'] = 'rejected_shared_target'
    out.loc[out['result'] != 'corrected', 'new_id'] = None
    return out


def alias_groups(df: pd.DataFrame) -> pd.Series:
    """A person key per record: records linked by a shared esb_id or smart_id
    (union-find). Names are never used."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for col in ('esb_id', 'smart_id'):
        first: Dict[str, int] = {}
        for i, v in enumerate(df[col].tolist()):
            if isinstance(v, str):
                if v in first:
                    parent[find(i)] = find(first[v])
                else:
                    first[v] = i
    return pd.Series([find(i) for i in range(len(df))], index=df.index)


# --- field changes -------------------------------------------------------------

# Representations of the same value. nflverse headshot URLs moved from a
# template placeholder to the filled-in default transformation.
FORMAT_EQUIVALENTS = [('{formatInstructions}', 'f_auto,q_auto')]


def _format_only(a: str, b: str) -> bool:
    """Same value, different representation: whitespace, numeric form
    ('2023.0' / '2023') or a FORMAT_EQUIVALENTS substitution."""
    if a.strip() == b.strip():
        return True
    for x, y in FORMAT_EQUIVALENTS:
        if a.replace(x, y) == b.replace(x, y):
            return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def field_changes(pairs: List[Tuple[str, str, str]], existing: pd.DataFrame, fresh: pd.DataFrame,
                  columns: List[str]) -> pd.DataFrame:
    """(identity, existing_id, fresh_id) pairs -> one row per changed field,
    classified 'meaningful' or 'format_only'."""
    old = existing.set_index('gsis_id')
    new = fresh.set_index('gsis_id')
    rows = []
    for kind, old_id, new_id in pairs:
        o, n = old.loc[old_id], new.loc[new_id]
        for col in columns:
            if col == 'gsis_id':
                continue
            a, b = o[col], n[col]
            if (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and a == b):
                continue
            fmt = not pd.isna(a) and not pd.isna(b) and _format_only(a, b)
            rows.append({'gsis_id': new_id, 'old_id': old_id, 'identity': kind, 'field': col,
                         'old': a, 'new': b, 'change': 'format_only' if fmt else 'meaningful'})
    return pd.DataFrame(rows, columns=['gsis_id', 'old_id', 'identity', 'field', 'old', 'new', 'change'])


# --- roster evidence -------------------------------------------------------------

def roster_week(roster: pd.DataFrame, season: int, week: int) -> Tuple[pd.DataFrame, dict]:
    """Rows for exactly (season, week); other weeks are counted and not used."""
    missing = [c for c in ROSTER_COLUMNS if c not in roster.columns]
    if missing:
        raise ReferenceBuildError([f"roster: missing column(s) {missing}"])
    s = pd.to_numeric(roster['season'], errors='coerce')
    w = pd.to_numeric(roster['week'], errors='coerce')
    use = (s == season) & (w == week)
    if not use.any():
        raise ReferenceBuildError([f"roster has no rows for {season} week {week}"])
    rows = roster[use]
    info = {'season': season, 'week': week, 'rows_used': int(use.sum()),
            'rows_other_weeks_not_used': int((~use).sum()),
            'later_weeks_present': sorted(int(x) for x in w[(s == season) & (w > week)].dropna().unique()),
            'ids_on_more_than_one_row': int(rows['gsis_id'].dropna().duplicated().sum())}
    return rows, info


def _join(values) -> str:
    return ';'.join(sorted({str(v) for v in values if isinstance(v, str)}))


# --- build -----------------------------------------------------------------------------

def _assemble(fresh: pd.DataFrame, retained_hist: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Fresh records, then the retained historical ones, in the existing column order."""
    return pd.concat([fresh[columns], retained_hist[columns]], ignore_index=True)


def build(existing_path: Path, fresh_dir: Path, out_dir: Path, candidates_path: Optional[Path] = None,
          season_week: Optional[Tuple[int, int]] = None, protected: Optional[List[Path]] = None) -> dict:
    """Build and validate the candidate reference; write it and its reports to
    out_dir (replacing a previous output only when everything succeeded)."""
    out_dir = Path(out_dir)
    try:
        check_workspace(out_dir, protected)
    except RehearsalError as exc:
        raise ReferenceBuildError([str(exc)]) from exc
    fresh_dir = Path(fresh_dir)
    check_output_location(out_dir, fresh_dir, [Path(existing_path)] + ([Path(candidates_path)]
                                                                       if candidates_path else []))
    names = [PLAYERS_NAME] + ([ROSTER_NAME.format(season=season_week[0])] if season_week else [])
    sources = verify_downloads(fresh_dir, names)
    existing_path = Path(existing_path)
    sources['existing'] = {'path': str(existing_path), 'sha256': sha256_file(existing_path),
                           'bytes': existing_path.stat().st_size}
    existing, fresh = read_text_csv(existing_path), read_text_csv(fresh_dir / PLAYERS_NAME)
    problems = validate(existing, 'existing reference') + validate(fresh, 'fresh source')
    if problems:
        raise ReferenceBuildError(problems)
    columns = list(existing.columns)
    absent = [c for c in columns if c not in fresh.columns]
    if absent:
        raise ReferenceBuildError([f"fresh source lacks existing column(s) {absent}"])
    extra = [c for c in fresh.columns if c not in columns]

    walk = crosswalk(existing, fresh)
    corrected = walk[walk['result'] == 'corrected']
    fresh_ids, existing_ids = set(fresh['gsis_id']), set(existing['gsis_id'])
    retained_hist = existing[~existing['gsis_id'].isin(fresh_ids)
                             & ~existing['gsis_id'].isin(set(corrected['old_id']))]
    candidate = _assemble(fresh, retained_hist, columns)
    # Backstop: whatever the assembly did, no existing ID may disappear
    # except a documented placeholder correction.
    problems = validate(candidate, 'candidate')
    lost = existing_ids - set(candidate['gsis_id']) - set(corrected['old_id'])
    if lost:
        problems.append(f"candidate loses {len(lost)} existing ID(s): {_examples(lost)}")
    if problems:
        raise ReferenceBuildError(problems)

    # Identity categories, as records and as people (alias groups).
    target = set(corrected['new_id'])
    cat = pd.Series('added', index=candidate.index)
    cat[candidate['gsis_id'].isin(existing_ids)] = 'retained_in_source'
    cat[candidate['gsis_id'].isin(target)] = 'corrected_placeholder'
    cat[candidate.index >= len(fresh)] = 'retained_historical'
    person = alias_groups(candidate)
    alias = person.map(person.value_counts()) > 1
    identities = {}
    for name in ('retained_in_source', 'added', 'corrected_placeholder', 'retained_historical'):
        m = cat == name
        identities[name] = {'records': int(m.sum()), 'people': int(person[m].nunique()),
                            'gsis_format': int(is_gsis(candidate.loc[m, 'gsis_id']).sum())}
    identities['source_aliases'] = {'records': int(alias.sum()), 'people': int(person[alias].nunique()),
                                    'examples': [sorted(candidate.loc[person == p, 'gsis_id'])
                                                 for p in person[alias].unique()[:EXAMPLES]]}
    identities['total'] = {'records': len(candidate), 'people': int(person.nunique())}
    hist_gsis = retained_hist[is_gsis(retained_hist['gsis_id'])]
    # Retained historical GSIS IDs that a fresh record shares a crosswalk ID with:
    # possible re-identification, reported for review, never applied.
    shared = {}
    for col in ('esb_id', 'smart_id'):
        fv = set(fresh[col].dropna())
        shared[col] = int(hist_gsis[col].isin(fv).sum())

    pairs = ([('retained_in_source', i, i) for i in sorted(existing_ids & fresh_ids)]
             + [('corrected_placeholder', o, n) for o, n in zip(corrected['old_id'], corrected['new_id'])])
    changes = field_changes(pairs, existing, fresh, columns)
    summary_changes = {}
    for col, g in changes.groupby('field'):
        summary_changes[col] = {'meaningful': int((g['change'] == 'meaningful').sum()),
                                'format_only': int((g['change'] == 'format_only').sum()),
                                'consumers': CONSUMERS.get(col, [])}
    pos = changes[(changes['field'] == 'position') & (changes['change'] == 'meaningful')]
    flips = pos[pos['old'].isin(ELIGIBLE) != pos['new'].isin(ELIGIBLE)]

    report = {
        'kind': 'player_reference_candidate', 'version': 1,
        'built_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'policy': [line.strip() for line in __doc__.split('Policy')[1].split('Invalid input')[0]
                   .splitlines()[1:] if line.strip()],
        'sources': sources, 'columns': columns, 'fresh_columns_not_kept': extra,
        'identities': identities,
        'crosswalk': walk['result'].value_counts().to_dict(),
        'retained_historical_gsis_sharing_a_crosswalk_id_with_a_fresh_record': shared,
        'field_changes': summary_changes,
        'eligibility_flips': flips[['gsis_id', 'old', 'new']].to_dict('records'),
    }
    files = {'players_candidate.csv': candidate, 'crosswalk.csv': walk, 'field_changes.csv': changes,
             'identities.csv': candidate[['gsis_id']].assign(identity=cat.values, person=person.values,
                                                             source_alias=alias.values)}

    if candidates_path or season_week:
        roster_rows, roster_info = (None, None)
        if season_week:
            roster = read_text_csv(fresh_dir / ROSTER_NAME.format(season=season_week[0]))
            roster_rows, roster_info = roster_week(roster, *season_week)
            report['roster'] = roster_info
        if candidates_path:
            report['candidates'], files['candidates.csv'] = _candidates(
                read_text_csv(candidates_path), existing, candidate, cat, roster_rows, flips)

    _write(out_dir, files, report)
    return report


def _candidates(cands: pd.DataFrame, existing: pd.DataFrame, candidate: pd.DataFrame, cat: pd.Series,
                roster_rows: Optional[pd.DataFrame], flips: pd.DataFrame) -> Tuple[dict, pd.DataFrame]:
    """Coverage and eligibility of the week's candidates, before and after."""
    if 'player_id' not in cands.columns or 'team' not in cands.columns:
        raise ReferenceBuildError(["candidates file needs player_id and team columns"])
    before = existing.set_index('gsis_id')['position']
    after = candidate.set_index('gsis_id')['position']
    category = pd.Series(cat.values, index=candidate['gsis_id'])
    c = cands[['player_id', 'team']].copy()
    c['before_position'] = c['player_id'].map(before)
    c['after_position'] = c['player_id'].map(after)
    c['identity'] = c['player_id'].map(category)
    c['before_in_reference'] = c['before_position'].notna()
    c['after_in_reference'] = c['after_position'].notna()
    c['before_eligible'] = c['before_position'].isin(ELIGIBLE)
    c['after_eligible'] = c['after_position'].isin(ELIGIBLE)
    if roster_rows is not None:
        g = roster_rows.groupby('gsis_id')
        c['roster_team'] = c['player_id'].map(g['team'].agg(_join))
        c['roster_status'] = c['player_id'].map(g['status'].agg(_join))
        c['roster_position'] = c['player_id'].map(g['position'].agg(_join))
        if 'depth_chart_position' in roster_rows.columns:
            c['roster_depth_chart_position'] = c['player_id'].map(g['depth_chart_position'].agg(_join))
        c['roster_position_disagrees'] = (c['roster_position'].notna() & c['after_position'].notna()
                                          & (c['roster_position'] != c['after_position']))
        c['roster_team_differs'] = c['roster_team'].notna() & (c['roster_team'] != c['team'])
    newly = c[~c['before_in_reference'] & c['after_in_reference']]
    out = {
        'candidates': len(c),
        'in_reference': {'before': int(c['before_in_reference'].sum()), 'after': int(c['after_in_reference'].sum())},
        'eligible': {'before': int(c['before_eligible'].sum()), 'after': int(c['after_eligible'].sum())},
        'newly_covered': {'total': len(newly), 'by_identity': newly['identity'].value_counts().to_dict(),
                          'eligible': int(newly['after_eligible'].sum())},
        'eligibility_lost': c.loc[c['before_eligible'] & ~c['after_eligible'], 'player_id'].tolist(),
        'eligibility_gained_existing': c.loc[c['before_in_reference'] & ~c['before_eligible']
                                             & c['after_eligible'], 'player_id'].tolist(),
        'lost_coverage': c.loc[c['before_in_reference'] & ~c['after_in_reference'], 'player_id'].tolist(),
    }
    out['eligible']['explained'] = (out['eligible']['before'] + out['newly_covered']['eligible']
                                    - len(out['eligibility_lost']) + len(out['eligibility_gained_existing']))
    if roster_rows is not None:
        out['roster_status'] = c['roster_status'].fillna('not on roster that week').value_counts().to_dict()
        out['roster_team_differs'] = c.loc[c['roster_team_differs'], 'player_id'].tolist()
        out['roster_position_disagrees'] = c.loc[c['roster_position_disagrees'], 'player_id'].tolist()
        flip_ids = set(flips['gsis_id'])
        cols = [x for x in ('player_id', 'team', 'before_position', 'after_position', 'roster_position',
                            'roster_depth_chart_position', 'roster_status', 'roster_team')
                if x in c.columns]
        out['candidate_eligibility_flips'] = c.loc[c['player_id'].isin(flip_ids), cols].fillna('').to_dict('records')
    return out, c


def _write(out_dir: Path, files: Dict[str, pd.DataFrame], report: dict) -> None:
    """Write to a staging directory, then swap it in; a failure leaves any
    previous output untouched."""
    parent = out_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".{out_dir.name}.staging-{secrets.token_hex(4)}"
    os.mkdir(staging)
    try:
        for name, df in files.items():
            df.to_csv(staging / name, index=False)
        report['outputs'] = {n: sha256_file(staging / n) for n in files}
        (staging / 'report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    previous = None
    if out_dir.exists():
        previous = parent / f".{out_dir.name}.previous-{secrets.token_hex(4)}"
        os.rename(out_dir, previous)
    try:
        os.rename(staging, out_dir)
    except BaseException:
        if previous is not None:
            os.rename(previous, out_dir)
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if previous is not None:
        shutil.rmtree(previous, ignore_errors=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build and validate a candidate players reference (offline)")
    ap.add_argument('--existing', type=Path, required=True, help="current reference, e.g. data_files/players.csv")
    ap.add_argument('--fresh-dir', type=Path, required=True,
                    help="downloads directory with manifest.json and players.csv")
    ap.add_argument('--out', type=Path, required=True, help="output directory outside the repository")
    ap.add_argument('--candidates', type=Path, help="CSV of week candidates (player_id, team)")
    ap.add_argument('--roster-week', help="SEASON:WEEK of the weekly-roster evidence, e.g. 2026:7")
    args = ap.parse_args(argv)
    sw = tuple(int(x) for x in args.roster_week.split(':')) if args.roster_week else None
    try:
        report = build(args.existing, args.fresh_dir, args.out, args.candidates, sw)
    except ReferenceBuildError as exc:
        print(json.dumps({'status': 'refused', 'problems': exc.problems}, indent=2))
        return 2
    print(json.dumps({k: report[k] for k in ('identities', 'crosswalk', 'field_changes')
                      if k in report}, indent=2, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
