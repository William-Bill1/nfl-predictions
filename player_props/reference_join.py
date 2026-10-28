"""
Player-reference joins for the Player Props and Parlay Builder pages.

The pages attach a player's display name and position from the players
reference (data_files/players.csv). They used to merge on short_name, which
is not unique (several "J.Smith"), so a merge could multiply rows or attach
another player's details. These helpers join on the stable ID instead.

Rules:
- IDs are text. Rows are matched player_id -> gsis_id, many-to-one: a row
  never multiplies, and row count, order and existing values are kept.
- A reference ID on more than one record is ambiguous: it is reported and
  never used, not resolved by taking the first record.
- A row with an ID uses only that ID. An ID missing from the reference stays
  unenriched; it never falls back to the row's name.
- Only a row with no ID at all (older files) may fall back to its name, and
  only when exactly one reference record has that short_name and its ID is
  not ambiguous. Such a match is marked 'name_unique' (inferred), never as
  verified.
- Fields are only filled where the input lacks them; stored values are kept.
  Where nothing can be attached, display_name falls back to the row's own
  player_name, so the original name stays visible.

Every row gets a `reference_match` value: 'id', 'name_unique',
'unknown_id', 'duplicate_reference_id', 'ambiguous_name', 'name_not_found',
'no_identity', or 'reference_unavailable' (no usable reference: absent,
unreadable or missing columns - not the same as a name that isn't in it).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

REFERENCE_COLUMNS = ('gsis_id', 'short_name', 'display_name', 'last_name', 'position')
VERIFIED, INFERRED = 'id', 'name_unique'
MATCHED = (VERIFIED, INFERRED)
EXAMPLES = 5


@dataclass
class Reference:
    """The players reference as text, with its problems. `records` keeps every
    record (duplicates included); `by_id` holds only unambiguous IDs."""
    records: pd.DataFrame
    problems: List[str] = field(default_factory=list)
    duplicate_ids: frozenset = frozenset()
    usable: bool = True

    @property
    def by_id(self) -> pd.DataFrame:
        ok = self.records[self.records['gsis_id'].notna() & ~self.records['gsis_id'].isin(self.duplicate_ids)]
        return ok.set_index('gsis_id', verify_integrity=True)

    @property
    def unique_names(self) -> pd.DataFrame:
        """short_name -> record, only for names on exactly one record whose ID is unambiguous."""
        r = self.records[self.records['short_name'].notna()]
        r = r[~r['short_name'].duplicated(keep=False)]
        r = r[r['gsis_id'].notna() & ~r['gsis_id'].isin(self.duplicate_ids)]
        return r.set_index('short_name', verify_integrity=True)


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype(str).str.strip().eq('')


def _examples(values) -> str:
    v = sorted(map(str, values))
    return ", ".join(v[:EXAMPLES]) + (f" (+{len(v) - EXAMPLES} more)" if len(v) > EXAMPLES else "")


def build_reference(df: Optional[pd.DataFrame]) -> Optional[Reference]:
    """Validate an already-read reference; None when it can't be used at all."""
    if df is None:
        return None
    missing = [c for c in REFERENCE_COLUMNS if c not in df.columns]
    if missing:
        return Reference(pd.DataFrame(columns=list(REFERENCE_COLUMNS)),
                         [f"players reference lacks column(s) {missing}; player details are not attached"],
                         usable=False)
    records = df[list(REFERENCE_COLUMNS)].astype(object).where(df[list(REFERENCE_COLUMNS)].notna(), None)
    records = records.apply(lambda s: s.map(lambda v: v if v is None else str(v)))
    problems = []
    blank = _blank(records['gsis_id'])
    if blank.any():
        problems.append(f"{int(blank.sum())} players-reference record(s) have no gsis_id and are not used")
        records.loc[blank, 'gsis_id'] = None
    ids = records['gsis_id'].dropna()
    dup = frozenset(ids[ids.duplicated(keep=False)])
    if dup:
        problems.append(f"{len(dup)} gsis_id(s) appear on more than one players-reference record and are not "
                        f"used: {_examples(dup)}")
    return Reference(records.reset_index(drop=True), problems, dup)


def load_reference(path: Path | str = Path('data_files/players.csv')) -> Optional[Reference]:
    """Read the reference with every value as text; None if the file is absent.
    An unreadable file is reported as a problem, not raised."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[''], low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        return Reference(pd.DataFrame(columns=list(REFERENCE_COLUMNS)),
                         [f"players reference could not be read ({exc}); player details are not attached"],
                         usable=False)
    return build_reference(df)


def match_rows(df: pd.DataFrame, ref: Optional[Reference], id_col: str = 'player_id',
               name_col: str = 'player_name') -> pd.DataFrame:
    """Per input row (same index, same order): `reference_match` and the
    matched gsis_id (None unless matched)."""
    out = pd.DataFrame({'reference_match': 'no_identity', 'reference_id': None}, index=df.index, dtype=object)
    if ref is None or not ref.usable:
        out['reference_match'] = 'reference_unavailable'
        return out
    by_id, by_name = ref.by_id, ref.unique_names
    has_id = (~_blank(df[id_col])) if id_col in df.columns else pd.Series(False, index=df.index)
    if has_id.any():
        ids = df.loc[has_id, id_col].astype(str)
        known = ids.isin(by_id.index)
        dup = ids.isin(ref.duplicate_ids)
        out.loc[known[known].index, 'reference_match'] = VERIFIED
        out.loc[known[known].index, 'reference_id'] = ids[known]
        out.loc[dup[dup].index, 'reference_match'] = 'duplicate_reference_id'
        out.loc[(~known & ~dup)[~known & ~dup].index, 'reference_match'] = 'unknown_id'
    no_id = ~has_id
    if no_id.any() and name_col in df.columns:
        names = df.loc[no_id, name_col]
        named = ~_blank(names)
        names = names[named].astype(str)
        recs = ref.records[ref.records['short_name'].notna()]
        counts = recs['short_name'].value_counts()
        dup_id_names = set(recs.loc[recs['gsis_id'].isin(ref.duplicate_ids), 'short_name'])
        uniq = names.isin(by_name.index)
        out.loc[uniq[uniq].index, 'reference_match'] = INFERRED
        out.loc[uniq[uniq].index, 'reference_id'] = names[uniq].map(by_name['gsis_id'])
        shared = ~uniq & (names.map(counts).fillna(0) > 1)
        out.loc[shared[shared].index, 'reference_match'] = 'ambiguous_name'
        # One record by that name, but its ID is on other records too.
        dup = ~uniq & ~shared & names.isin(dup_id_names)
        out.loc[dup[dup].index, 'reference_match'] = 'duplicate_reference_id'
        rest = ~uniq & ~shared & ~dup
        out.loc[rest[rest].index, 'reference_match'] = 'name_not_found'
    return out


def enrich(df: Optional[pd.DataFrame], ref: Optional[Reference], fields: Sequence[str] = ('display_name', 'position'),
           id_col: str = 'player_id', name_col: str = 'player_name') -> Optional[pd.DataFrame]:
    """A copy of df with `fields` filled from the reference where the input
    lacks them, plus `reference_match`. Rows are never added, dropped or
    reordered; stored values are never overwritten."""
    if df is None:
        return None
    out = df.copy()
    m = match_rows(out, ref, id_col, name_col)
    out['reference_match'] = m['reference_match']
    out['reference_id'] = m['reference_id']
    by_id = ref.by_id if ref is not None else pd.DataFrame(columns=list(REFERENCE_COLUMNS[1:]))
    for f in fields:
        attached = m['reference_id'].map(by_id[f]) if len(by_id) else pd.Series(None, index=out.index, dtype=object)
        if f not in out.columns:
            out[f] = attached
        else:
            out[f] = out[f].where(~_blank(out[f]), attached)
    if 'display_name' in fields and name_col in out.columns:
        out['display_name'] = out['display_name'].where(~_blank(out['display_name']), out[name_col])
    assert len(out) == len(df) and out.index.equals(df.index)
    return out


def enrichment_summary(df: Optional[pd.DataFrame], ref: Optional[Reference]) -> List[str]:
    """Plain-language notes for the page: reference problems, then rows whose
    details are inferred or missing."""
    notes = list(ref.problems) if ref is not None else ["players reference not found; player identities can't be checked"]
    if df is None or 'reference_match' not in df.columns:
        return notes
    counts = df['reference_match'].value_counts().to_dict()
    labels = {
        INFERRED: "matched by a unique name only (no player ID in the file; inferred, not verified)",
        'unknown_id': "have a player ID that is not in the players reference",
        'duplicate_reference_id': "have a player ID that appears more than once in the players reference",
        'ambiguous_name': "have no player ID and a name shared by several players",
        'name_not_found': "have no player ID and a name not in the players reference",
        'no_identity': "have neither a player ID nor a name",
        'reference_unavailable': "can't be checked because the players reference is unavailable",
    }
    for k, text in labels.items():
        if counts.get(k):
            notes.append(f"{counts[k]} row(s) {text}"
                         + ("" if k in (INFERRED, 'reference_unavailable')
                            else "; no reference details attached (stored values are kept)"))
    return notes


# Short labels for a table column; verified rows show nothing.
STATUS_LABELS = {
    VERIFIED: '', INFERRED: 'inferred by name', 'unknown_id': 'unknown ID',
    'duplicate_reference_id': 'repeated reference ID', 'ambiguous_name': 'ambiguous name',
    'name_not_found': 'not in reference', 'no_identity': 'no identity',
    'reference_unavailable': 'reference unavailable',
}


def status_label(match: pd.Series) -> pd.Series:
    return match.map(STATUS_LABELS).fillna('')


def season_totals(stats: pd.DataFrame, agg: dict, observed: bool = True) -> pd.DataFrame:
    """Per-player season totals that never combine players by name alone.

    Rows with a player_id are grouped by (player_id, player_name, team) as
    before; rows without one are grouped by (player_name, team) on their own,
    keyed apart from every identified player. The result keeps player_id
    (None for name-only groups), so enrich() can verify ID groups and only
    infer name groups when the name is unique."""
    key = player_key(stats)
    g = stats.assign(_player_key=key).groupby(['_player_key', 'player_name', 'team'], observed=observed,
                                              dropna=False).agg(agg).reset_index()
    named = g['_player_key'].str.startswith('name:')
    g['player_id'] = g['_player_key'].where(~named, None)
    return g.drop(columns='_player_key')


def player_rows(df: pd.DataFrame, ref: Optional[Reference], gsis_id: str,
                id_col: str = 'player_id') -> pd.Series:
    """Mask of df's rows for one reference player. A row with an ID matches
    only on that ID. A row without one matches through an earlier unique-name
    enrichment (`reference_id`), or, failing that, when its display_name
    belongs to exactly one reference record and that record is this player."""
    gsis_id = str(gsis_id)
    has_id = (~_blank(df[id_col])) if id_col in df.columns else pd.Series(False, index=df.index)
    mask = has_id & (df[id_col].astype(str) == gsis_id) if id_col in df.columns else has_id
    legacy = ~has_id
    if 'reference_id' in df.columns:
        via = legacy & df['reference_id'].notna()
        mask |= via & (df['reference_id'].astype(str) == gsis_id)
        legacy &= ~via
    if legacy.any() and ref is not None and 'display_name' in df.columns:
        r = ref.records[ref.records['display_name'].notna()]
        r = r[~r['display_name'].duplicated(keep=False) & r['gsis_id'].notna()
              & ~r['gsis_id'].isin(ref.duplicate_ids)]
        owner = r.set_index('display_name')['gsis_id']
        mask |= legacy & (df['display_name'].map(owner) == gsis_id)
    return mask


def player_key(df: pd.DataFrame, id_col: str = 'player_id', name_col: str = 'player_name') -> pd.Series:
    """One key per row for "same player" checks: the ID where present,
    otherwise the name (marked, so a name never equals an ID)."""
    if id_col in df.columns:
        ids = df[id_col]
        return ids.where(~_blank(ids), 'name:' + df[name_col].astype(str)).astype(str)
    return 'name:' + df[name_col].astype(str)


def unique_labels(labels: Iterable[str], keys: Iterable[str]) -> List[str]:
    """Labels made unique by appending the key to any label shared by
    different keys (a Streamlit select maps its label back to the value)."""
    labels, keys = list(labels), list(map(str, keys))
    owners: Dict[str, set] = {}
    for lab, k in zip(labels, keys):
        owners.setdefault(lab, set()).add(k)
    out = [f"{lab} [{k}]" if len(owners[lab]) > 1 else lab for lab, k in zip(labels, keys)]
    seen: Dict[str, int] = {}
    final = []
    for lab in out:                       # identical label and key twice: number them
        seen[lab] = seen.get(lab, 0) + 1
        final.append(lab if seen[lab] == 1 else f"{lab} ({seen[lab]})")
    return final
