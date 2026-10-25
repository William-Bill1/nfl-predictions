"""Historical Data page CSV exports: the page being shown, or every filtered row.

``render_exports`` is called by pages/1_Historical_Data.py after filtering and
pagination, with the filtered view (already sorted as displayed) and the
display columns. Both exports use the page's own CSV conversion, so columns,
encoding (UTF-8), row order and the missing index are as before.

``scope`` says what the page has loaded, as the page knows it: ``'sample'``
until "Load full historical play-by-play data" is used (the page then shows a
small sample), ``'full'`` afterwards. It is passed explicitly, never inferred
from row counts. With a sample, the whole-view export is labelled and named
as a sample.

The whole-view file is built only when its button is clicked: Streamlit runs
the callable passed as ``data`` at download time, so a rerun (a filter, page
or any other change) never builds it. Nothing is kept in session state; every
rerun registers a fresh callable over the filters then in effect, and
Streamlit drops the superseded ones.
"""
from datetime import datetime

import streamlit as st

FILE_PREFIX = 'nfl_historical_data'
SCOPES = ('sample', 'full')
LOAD_BUTTON = 'Load full historical play-by-play data'


def page_filename(stamp: str, page: int, total_pages: int) -> str:
    return f'{FILE_PREFIX}_{stamp}_page{page}_of_{total_pages}.csv'


def all_filtered_filename(stamp: str, rows: int) -> str:
    return f'{FILE_PREFIX}_{stamp}_all_filtered_{rows}_rows.csv'


def sample_filtered_filename(stamp: str, rows: int) -> str:
    return f'{FILE_PREFIX}_{stamp}_sample_filtered_{rows}_rows.csv'


def render_exports(filtered, display_cols, *, start_idx: int, end_idx: int, page: int,
                   total_pages: int, convert, scope: str, now=None) -> None:
    """The two export buttons for the current filters and page."""
    if scope not in SCOPES:
        raise ValueError(f"unknown data scope {scope!r} (expected one of {SCOPES})")
    total = len(filtered)
    if total == 0:
        st.caption("Nothing to export: no rows match the current filters.")
        return
    stamp = (now or datetime.now()).strftime("%Y%m%d")
    view = filtered[display_cols]
    page_rows = view.iloc[start_idx:end_idx]
    pages = f"{total_pages:,} page{'s' if total_pages != 1 else ''}"
    if scope == 'full':
        label = f"📥 Export all filtered rows ({total:,} rows)"
        file_name = all_filtered_filename(stamp, total)
        note = f"Every row matching the current filters, across all {pages}. Built when clicked."
    else:
        label = f"📥 Export filtered sample ({total:,} rows)"
        file_name = sample_filtered_filename(stamp, total)
        note = (f"Only the loaded sample is included: the {total:,} sample rows matching the current "
                f"filters, across all {pages}. Use “{LOAD_BUTTON}” to export the full "
                f"filtered dataset.")
    left, right = st.columns(2)
    with left:
        st.download_button(
            label=f"📥 Export current page ({len(page_rows):,} rows)",
            data=convert(page_rows),
            file_name=page_filename(stamp, page, total_pages),
            mime='text/csv',
            key='hist_export_page',
        )
        st.caption(f"Rows {start_idx + 1:,} to {end_idx:,}: the rows shown above.")
    with right:
        st.download_button(
            label=label,
            data=lambda: convert(view),
            file_name=file_name,
            mime='text/csv',
            key='hist_export_all',
            # No rerun on click: a rerun registers a new deferred id for this
            # button and Streamlit drops the one being downloaded, so the
            # download can silently fail (seen in a real browser).
            on_click='ignore',
        )
        st.caption(note)
