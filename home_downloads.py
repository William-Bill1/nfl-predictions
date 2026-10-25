"""Home-page sidebar export controls: predictions CSV, recommendation-log CSV
and the on-demand predictions PDF.

`home_page()` creates the three sidebar placeholders and passes them here, with
the loaded predictions and the data directory. Nothing here regenerates
predictions or rewrites a source file: the CSVs are built from what is already
loaded (or read as stored), and the PDF is written as a new file under
``<data_dir>/exports/``.

"Betting log" is the model's automatic spread recommendation log
(``betting_recommendations_log.csv``, see betting_log.py), not the private
Bet Journal.
"""
import os
import traceback
from datetime import datetime
from io import BytesIO
from os import path
import base64

import numpy as np
import pandas as pd
import streamlit as st

try:
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    PDF_UNAVAILABLE = None
except ImportError as _e:  # the app still runs; only the PDF export is unavailable
    PDF_UNAVAILABLE = f"the PDF library (reportlab) could not be imported: {_e}"

RECOMMENDATION_LOG = 'betting_recommendations_log.csv'


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to a CSV bytes object suitable for `st.download_button`.

    Keeps encoding explicit to UTF-8 to avoid platform issues.
    """
    try:
        return df.to_csv(index=False).encode('utf-8')
    except Exception:
        # Fallback: coerce to strings then export
        df2 = df.copy()
        for c in df2.columns:
            try:
                df2[c] = df2[c].astype(str)
            except Exception:
                pass
        return df2.to_csv(index=False).encode('utf-8')


def generate_predictions_pdf(df: pd.DataFrame, max_rows: int = 200) -> bytes:
    """Generate a simple PDF summary of the predictions DataFrame and return bytes.

    Keeps layout compact and avoids embedding large images. Uses a landscape
    letter page with a table of core fields.
    """
    if PDF_UNAVAILABLE:
        raise RuntimeError(f"PDF export unavailable: {PDF_UNAVAILABLE}")
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), topMargin=0.4 * inch)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=16, alignment=1)

    story = []
    story.append(Paragraph("NFL Predictions", title_style))
    story.append(Spacer(1, 8))

    date_text = f"Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}"
    story.append(Paragraph(date_text, styles['Normal']))
    story.append(Spacer(1, 8))

    # Table header and column selection - keep columns compact for readability
    # Short, compact headers to improve fit on the PDF
    header = ['Gameday', 'Home', 'Away', 'Spread', 'Total', 'Home ML', 'Away ML', 'Pr Cover', 'Pr Win', 'Pr Over']
    table_data = [header]
    # Only include games that are today or later (no past games)
    try:
        today_dt = pd.to_datetime(datetime.now().date())
        if 'gameday' in df.columns:
            gamedays = pd.to_datetime(df['gameday'], errors='coerce')
            upcoming_mask = gamedays >= today_dt
            df_to_use = df[upcoming_mask]
        else:
            df_to_use = df.copy()
    except Exception:
        df_to_use = df.copy()

    # If there are no upcoming games, emit a short message instead of a table
    if df_to_use is None or len(df_to_use) == 0:
        story.append(Spacer(1, 6))
        story.append(Paragraph("No upcoming games found for the selected predictions.", styles['Normal']))
        try:
            doc.build(story)
        except Exception:
            pass

        buffer.seek(0)
        return buffer.getvalue()

    # Fill rows (limit to max_rows for performance)
    for _, row in df_to_use.head(max_rows).iterrows():
        try:
            gameday_val = row.get('gameday', '')
            gameday = ''
            if pd.notna(gameday_val):
                try:
                    dt = pd.to_datetime(gameday_val)
                    # If time is exactly midnight, only show the date (hide 00:00:00)
                    if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
                        gameday = dt.strftime('%Y-%m-%d')
                    else:
                        gameday = dt.strftime('%Y-%m-%d %H:%M')
                except Exception:
                    gameday = str(gameday_val)

            def fmt(v, digits=2):
                try:
                    if pd.isna(v):
                        return ''
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        return f"{v:.{digits}f}"
                    return str(v)
                except Exception:
                    return str(v)

            row_vals = [
                gameday,
                row.get('home_team', ''),
                row.get('away_team', ''),
                fmt(row.get('spread_line', '')),
                fmt(row.get('total_line', '')),
                str(row.get('home_moneyline', '')),
                str(row.get('away_moneyline', '')),
                fmt(row.get('prob_underdogCovered', ''), 2),
                fmt(row.get('prob_underdogWon', ''), 2),
                fmt(row.get('prob_overHit', ''), 2),
            ]
            table_data.append(row_vals)
        except Exception:
            # Skip problematic rows but continue building the PDF
            continue

    # Create table with reasonable column widths for landscape
    # Adjust column widths for a tighter, print-friendly layout
    col_widths = [1.0 * inch, 1.0 * inch, 1.0 * inch, 0.6 * inch, 0.6 * inch, 0.8 * inch, 0.8 * inch, 0.6 * inch, 0.6 * inch, 0.6 * inch]
    table = Table(table_data, colWidths=col_widths)

    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2f4f4f')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        # Smaller header and body fonts to fit more content
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        # Tighter paddings for compact layout
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f7f7f7')]),
        # Alignments: header centered, numeric columns right-aligned for readability
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (3, 1), (6, -1), 'RIGHT'),
        ('ALIGN', (7, 1), (9, -1), 'RIGHT')
    ])
    table.setStyle(table_style)

    story.append(table)

    # Build PDF
    try:
        doc.build(story)
    except Exception:
        # If building fails, return an empty bytes object instead of crashing the app
        return b''

    buffer.seek(0)
    return buffer.getvalue()


def render_download(placeholder, filename, data_bytes, mime='application/octet-stream', display_text=None,
                    icons_dir='data_files/'):
    """An HTML download link with an embedded icon; st.download_button if that fails."""
    csv_icon = path.join(icons_dir, 'csv_icon.png')
    pdf_icon = path.join(icons_dir, 'pdf_icon.png')
    fallback_icon = path.join(icons_dir, 'favicon.png')
    try:
        b64_file = base64.b64encode(data_bytes).decode('ascii')
        if mime == 'text/csv':
            file_data_uri = f"data:text/csv;base64,{b64_file}"
        elif mime == 'application/pdf':
            file_data_uri = f"data:application/pdf;base64,{b64_file}"
        else:
            file_data_uri = f"data:application/octet-stream;base64,{b64_file}"

        # Choose icon based on mime
        chosen_icon_path = None
        if mime == 'text/csv' and os.path.exists(csv_icon):
            chosen_icon_path = csv_icon
        elif mime == 'application/pdf' and os.path.exists(pdf_icon):
            chosen_icon_path = pdf_icon
        elif os.path.exists(fallback_icon):
            chosen_icon_path = fallback_icon

        img_tag = ''
        if chosen_icon_path is not None:
            try:
                with open(chosen_icon_path, 'rb') as ifh:
                    img_b64 = base64.b64encode(ifh.read()).decode('ascii')
                img_tag = f'<img src="data:image/png;base64,{img_b64}" style="width:36px;height:36px;margin-right:10px;vertical-align:middle;border-radius:6px;">'
            except Exception:
                img_tag = ''

        if display_text is None:
            display_text = filename
        html = (
            f'<div style="display:flex;align-items:center;margin:6px 0;">'
            f'<a download="{filename}" href="{file_data_uri}" '
            f'style="display:flex;align-items:center;padding:6px 12px;background:#1976d2;color:#fff;border-radius:6px;text-decoration:none;font-weight:600;">'
            f'{img_tag}'
            f'<span style="color:#fff;">{display_text}</span>'
            f'</a></div>'
        )
        placeholder.markdown(html, unsafe_allow_html=True)
    except Exception:
        try:
            placeholder.download_button(label=f"Download {filename}", data=data_bytes, file_name=filename, mime=mime)
        except Exception as e:
            placeholder.error(f'Could not prepare download for {filename}: {e}')


def render_home_downloads(predictions_placeholder, log_placeholder, pdf_placeholder, predictions_df,
                          data_dir) -> None:
    """Fill the three sidebar placeholders created by home_page().

    A placeholder that is None (its creation failed) is skipped. A control
    whose data or dependency is missing says so; a failure is shown, not
    swallowed."""
    # Predictions CSV
    if predictions_placeholder is not None:
        if predictions_df is None:
            predictions_placeholder.info("Predictions download unavailable: predictions haven't loaded yet.")
        else:
            filename = f'nfl_predictions_{datetime.now().strftime("%Y%m%d")}.csv'
            try:
                csv_bytes = convert_df_to_csv(predictions_df)
                render_download(predictions_placeholder, filename, csv_bytes, mime='text/csv',
                                display_text='Download Predictions', icons_dir=data_dir)
            except Exception as e:
                predictions_placeholder.error(f"Could not prepare {filename}: {e}")

    # Recommendation log CSV, exactly as stored
    if log_placeholder is not None:
        log_path = path.join(data_dir, RECOMMENDATION_LOG)
        if not os.path.exists(log_path):
            log_placeholder.info(f"Betting log download unavailable: no recommendation log "
                                 f"({RECOMMENDATION_LOG}) yet.")
        else:
            filename = f'betting_recommendations_log_{datetime.now().strftime("%Y%m%d")}.csv'
            try:
                with open(log_path, 'rb') as _f:
                    log_bytes = _f.read()
                render_download(log_placeholder, filename, log_bytes, mime='text/csv',
                                display_text='Download Betting Log', icons_dir=data_dir)
            except Exception as e:
                log_placeholder.error(f"Could not prepare {filename}: {e}")

    # PDF export: generate on demand to avoid pre-building large binary at load time
    if pdf_placeholder is not None:
        if PDF_UNAVAILABLE:
            pdf_placeholder.info(f"PDF export unavailable: {PDF_UNAVAILABLE}.")
        elif predictions_df is None:
            pdf_placeholder.info("PDF export unavailable: predictions haven't loaded yet.")
        else:
            try:
                if pdf_placeholder.button("📄 Generate Predictions PDF"):
                    try:
                        pdf_bytes = generate_predictions_pdf(predictions_df)
                    except Exception as e:
                        st.sidebar.error(f"PDF generation failed: {e}")
                        st.sidebar.text(traceback.format_exc())
                        pdf_bytes = b''

                    if pdf_bytes:
                        try:
                            exports_dir = path.join(data_dir, 'exports')
                            os.makedirs(exports_dir, exist_ok=True)
                            filename = f'nfl_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
                            file_path = path.join(exports_dir, filename)
                            with open(file_path, 'wb') as f:
                                f.write(pdf_bytes)

                                try:
                                    render_download(pdf_placeholder, filename, pdf_bytes, mime='application/pdf',
                                                    display_text='Download Predictions (PDF)', icons_dir=data_dir)
                                except Exception as e:
                                    st.sidebar.error(f"Saved PDF but failed to create download button: {e}")
                        except Exception as e:
                            st.sidebar.error(f"Failed to save generated PDF: {e}")
                    else:
                        st.sidebar.error("Failed to generate PDF.")
            except Exception as e:
                st.sidebar.error(f"PDF export handler error: {e}")
