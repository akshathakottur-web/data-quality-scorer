"""
app.py

Streamlit web interface for the Automatic Data Quality Scoring System.

Visual identity: "instrument console" — the dashboard reads like a
calibration panel for a dataset rather than a generic admin page.
Palette is graphite + brass, numbers are set in mono, and the overall
score renders as a dial gauge rather than a plain st.metric.
"""

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib.styles import getSampleStyleSheet

import streamlit as st
import pandas as pd
import tempfile
import os

from main import run_pipeline, run_pipeline_on_df


# =====================================================================
# PDF REPORT
# =====================================================================

def generate_pdf_report(report_text):
    """
    Generate the data quality report as a PDF.
    """
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "Automatic Data Quality Report",
            styles["Title"]
        )
    )

    story.append(Spacer(1, 20))

    story.append(
        Preformatted(
            report_text,
            styles["Code"]
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()


# =====================================================================
# DESIGN SYSTEM
#
# Palette   : graphite console + brass accent (a calibration-instrument
#             read, deliberately avoiding the cream/terracotta and
#             near-black/neon defaults).
#   --bg          #14181C   base console
#   --surface     #1B2126   panel
#   --surface-2   #232A31   raised / hover panel
#   --border      #2E363D   hairline
#   --text        #ECEDEA   primary text
#   --text-muted  #9BA3AA   secondary text
#   --accent      #C6A15B   brass — the one signature color
#   --success     #7FAE8E   sage
#   --warning     #D8AE5C   amber
#   --critical    #C1705F   rust
#
# Type      : Fraunces (display / the dial numerals) · IBM Plex Sans
#             (body copy) · IBM Plex Mono (every measured value).
#
# Signature : the brass calibration dial for the overall score, with a
#             tick ring behind it — every other surface stays quiet.
# =====================================================================

def inject_theme():
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">

        <style>
        :root{
            --bg:#14181C; --surface:#1B2126; --surface-2:#232A31;
            --border:#2E363D; --text:#ECEDEA; --text-muted:#9BA3AA;
            --accent:#C6A15B; --accent-soft:rgba(198,161,91,0.14);
            --success:#7FAE8E; --warning:#D8AE5C; --critical:#C1705F;
            --font-display:'Fraunces', serif;
            --font-body:'IBM Plex Sans', sans-serif;
            --font-mono:'IBM Plex Mono', monospace;
        }

        html, body, [class*="css"]{ font-family: var(--font-body); }

        .stApp{
            background: var(--bg);
            color: var(--text);
        }

        /* ---- layout width & rhythm ---- */
        .block-container{ padding-top: 2.5rem; max-width: 1180px; }

        h1, h2, h3{ font-family: var(--font-display); color: var(--text); letter-spacing: 0.01em; }
        p, span, div, label{ color: var(--text); }

        /* ---- console header ---- */
        .console-eyebrow{
            font-family: var(--font-mono);
            font-size: 0.72rem;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            color: var(--accent);
            margin-bottom: 0.4rem;
        }
        .console-title{
            font-family: var(--font-display);
            font-size: 2.3rem;
            font-weight: 500;
            margin: 0 0 0.15rem 0;
            color: var(--text);
        }
        .console-sub{
            font-family: var(--font-body);
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-bottom: 1.6rem;
        }
        .console-divider{
            height: 1px;
            background: linear-gradient(90deg, var(--border), transparent);
            margin: 0.4rem 0 2rem 0;
        }

        /* ---- section labels ---- */
        .section-label{
            font-family: var(--font-mono);
            font-size: 0.72rem;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
            margin: 2.2rem 0 1.1rem 0;
        }

        /* ---- readout cards (hover surface) ---- */
        .readout-card{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            transition: transform 0.15s ease, border-color 0.15s ease, background 0.15s ease;
        }
        .readout-card:hover{
            transform: translateY(-2px);
            border-color: var(--accent);
            background: var(--surface-2);
        }
        .readout-label{
            font-family: var(--font-mono);
            font-size: 0.68rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }
        .readout-value{
            font-family: var(--font-mono);
            font-size: 1.7rem;
            font-weight: 500;
            color: var(--text);
        }
        .readout-value.accent{ color: var(--accent); }

        /* ---- calibration gauge (signature element) ---- */
        .gauge-wrap{
            position: relative;
            width: 220px; height: 220px;
            margin: 0.4rem auto 0.2rem auto;
        }
        .gauge-ticks{
            position: absolute; inset: 0; border-radius: 50%;
            background: repeating-conic-gradient(var(--border) 0deg 1.4deg, transparent 1.4deg 9deg);
            -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 13px), #000 calc(100% - 12px));
                    mask: radial-gradient(farthest-side, transparent calc(100% - 13px), #000 calc(100% - 12px));
        }
        .gauge-ring{
            position: absolute; inset: 15px; border-radius: 50%;
            background: conic-gradient(var(--accent) calc(var(--pct) * 3.6deg), var(--surface-2) 0deg);
        }
        .gauge-ring::before{
            content:""; position:absolute; inset:14px; border-radius:50%;
            background: var(--surface);
        }
        .gauge-center{
            position:absolute; inset:0;
            display:flex; flex-direction:column; align-items:center; justify-content:center;
        }
        .gauge-value{
            font-family: var(--font-display);
            font-size: 2.6rem;
            font-weight: 500;
            color: var(--text);
            line-height: 1;
        }
        .gauge-max{ font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-muted); }
        .gauge-status{
            margin-top: 0.85rem; text-align:center;
            font-family: var(--font-mono); font-size: 0.78rem;
            letter-spacing: 0.14em; text-transform: uppercase;
        }

        /* ---- sub-score gauge bars ---- */
        .subscore-row{
            display:flex; align-items:center; gap: 0.9rem;
            padding: 0.55rem 0.2rem;
            border-bottom: 1px solid var(--border);
            transition: background 0.15s ease;
        }
        .subscore-row:hover{ background: var(--accent-soft); }
        .subscore-label{
            width: 190px; flex-shrink:0;
            font-size: 0.86rem; color: var(--text);
        }
        .subscore-track{
            flex-grow:1; height: 7px; border-radius: 4px;
            background: var(--surface-2); overflow:hidden;
        }
        .subscore-fill{
            height:100%; border-radius:4px;
            background: linear-gradient(90deg, var(--accent), #DDBF86);
        }
        .subscore-value{
            width: 46px; text-align:right; flex-shrink:0;
            font-family: var(--font-mono); font-size:0.86rem; color: var(--accent);
        }

        /* ---- recommendations ---- */
        .rec-item{
            display:flex; gap: 0.9rem; align-items:flex-start;
            padding: 0.75rem 0.9rem;
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 2px solid var(--accent);
            border-radius: 6px;
            margin-bottom: 0.55rem;
            transition: transform 0.12s ease, background 0.12s ease;
        }
        .rec-item:hover{ transform: translateX(3px); background: var(--surface-2); }
        .rec-index{
            font-family: var(--font-mono); font-size: 0.72rem;
            color: var(--accent); margin-top: 0.15rem; flex-shrink:0;
        }
        .rec-text{ font-size: 0.92rem; color: var(--text); }

        /* ---- visualization frame ---- */
        .viz-caption{
            font-family: var(--font-mono);
            font-size: 0.7rem; letter-spacing: 0.12em; text-transform: uppercase;
            color: var(--text-muted); margin-top: 0.4rem; text-align:center;
        }

        /* ---- streamlit widget overrides ---- */
        [data-testid="stFileUploader"]{
            background: var(--surface);
            border: 1px dashed var(--border);
            border-radius: 10px;
            padding: 0.6rem;
        }
        [data-testid="stFileUploader"]:hover{ border-color: var(--accent); }

        .stButton>button, [data-testid="stDownloadButton"] button{
            background: var(--surface-2);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 7px;
            font-family: var(--font-mono);
            font-size: 0.82rem;
            letter-spacing: 0.04em;
            transition: border-color 0.15s ease, color 0.15s ease;
        }
        .stButton>button:hover, [data-testid="stDownloadButton"] button:hover{
            border-color: var(--accent);
            color: var(--accent);
        }

        [data-testid="stExpander"]{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
        }

        [data-testid="stChatMessage"]{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
        }

        [data-testid="stChatInput"] textarea{
            background: var(--surface) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
        }

        [data-testid="stMetricValue"]{ font-family: var(--font-mono); color: var(--accent); }

        ::-webkit-scrollbar{ width: 8px; }
        ::-webkit-scrollbar-thumb{ background: var(--border); border-radius: 4px; }

        @media (prefers-reduced-motion: reduce){
            .readout-card, .rec-item, .subscore-row, .stButton>button{ transition: none !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_color(score):
    if score >= 85:
        return "var(--success)"
    if score >= 65:
        return "var(--warning)"
    return "var(--critical)"


def render_gauge(score, status_text):
    pct = max(0, min(100, score))
    color = status_color(pct)
    st.markdown(
        f"""
        <div class="gauge-wrap">
            <div class="gauge-ticks"></div>
            <div class="gauge-ring" style="--pct:{pct};"></div>
            <div class="gauge-center">
                <div><span class="gauge-value">{pct}</span><span class="gauge-max">/100</span></div>
            </div>
        </div>
        <div class="gauge-status" style="color:{color};">{status_text}</div>
        """,
        unsafe_allow_html=True,
    )


def render_readout(label, value, accent=False):
    cls = "readout-value accent" if accent else "readout-value"
    st.markdown(
        f"""
        <div class="readout-card">
            <div class="readout-label">{label}</div>
            <div class="{cls}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_subscore(name, score):
    st.markdown(
        f"""
        <div class="subscore-row">
            <div class="subscore-label">{name}</div>
            <div class="subscore-track"><div class="subscore-fill" style="width:{score}%;"></div></div>
            <div class="subscore-value">{score}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendation(index, text):
    st.markdown(
        f"""
        <div class="rec-item">
            <span class="rec-index">{index:02d}</span>
            <span class="rec-text">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# APP
# =====================================================================

if "GROQ_API_KEY" not in st.secrets:
    st.error("Groq API key not configured. Add it to .streamlit/secrets.toml")
    st.stop()

# Imported here (after the secrets check) since chatbot.py constructs the
# Groq client at import time and would fail immediately without a key.
from src.dqs.chatbot import chat_turn

st.set_page_config(page_title="Data Quality Instrument", layout="wide", page_icon="🎛️")
inject_theme()

st.markdown('<div class="console-eyebrow">Automatic Assessment System</div>', unsafe_allow_html=True)
st.markdown('<div class="console-title">Data Quality Instrument</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="console-sub">Upload a dataset to calibrate its readiness for machine learning.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="console-divider"></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop a dataset to begin calibration",
    type=["csv"],
)

if uploaded_file is not None:
    # Detect a NEW file (different from the last one processed) and reset ALL stale state --
    # including cached results, so a new upload can't inherit the old file's analysis.
    if st.session_state.get("last_uploaded_filename") != uploaded_file.name:
        st.session_state.pop("working_df", None)
        st.session_state.pop("chat_history", None)
        st.session_state.pop("results", None)
        st.session_state["last_uploaded_filename"] = uploaded_file.name

    # Only run the full pipeline once per file. On every subsequent Streamlit
    # rerun (chat messages, widget interactions, etc.) we reuse the cached
    # results instead of re-reading the original upload from disk -- which
    # is what was silently discarding every chatbot fix before.
    if "results" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        with st.spinner("Calibrating instrument..."):
            st.session_state.results = run_pipeline(tmp_path)

        st.session_state.working_df = pd.read_csv(tmp_path)
        os.unlink(tmp_path)

    results = st.session_state.results
    scoring = results["scoring"]
    profile = results["profile"]

    # -----------------------------------------------------------
    # OVERVIEW: gauge + readouts
    # -----------------------------------------------------------
    st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)

    gauge_col, readout_col = st.columns([1, 1.6], gap="large")

    with gauge_col:
        render_gauge(scoring["overall_score"], scoring["status"])

    with readout_col:
        r1, r2 = st.columns(2, gap="medium")
        with r1:
            render_readout("Rows", f'{profile["n_rows"]:,}')
        with r2:
            render_readout("Columns", f'{profile["n_cols"]:,}')
        st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)
        r3, r4 = st.columns(2, gap="medium")
        with r3:
            render_readout("Status", scoring["status"], accent=True)
        with r4:
            render_readout("Overall Score", f'{scoring["overall_score"]}/100', accent=True)

    # -----------------------------------------------------------
    # SUB-SCORES
    # -----------------------------------------------------------
    st.markdown('<div class="section-label">Sub-Scores</div>', unsafe_allow_html=True)
    for detector, score in scoring["individual_scores"].items():
        render_subscore(detector, score)

    # -----------------------------------------------------------
    # RECOMMENDATIONS
    # -----------------------------------------------------------
    st.markdown('<div class="section-label">Recommendations</div>', unsafe_allow_html=True)
    for i, rec in enumerate(results["recommendations"], start=1):
        render_recommendation(i, rec)

    # -----------------------------------------------------------
    # VISUALIZATIONS
    # -----------------------------------------------------------
    st.markdown('<div class="section-label">Visualizations</div>', unsafe_allow_html=True)
    img_col1, img_col2 = st.columns(2)

    if os.path.exists("outputs/reports/missing_values_heatmap.png"):
        with img_col1:
            st.image("outputs/reports/missing_values_heatmap.png")
            st.markdown('<div class="viz-caption">Missing Values Heatmap</div>', unsafe_allow_html=True)

    if os.path.exists("outputs/reports/score_breakdown.png"):
        with img_col2:
            st.image("outputs/reports/score_breakdown.png")
            st.markdown('<div class="viz-caption">Score Breakdown</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------
    # OUTLIER VISUALIZATIONS
    # -----------------------------------------------------------
    st.markdown('<div class="section-label">Outlier Analysis</div>', unsafe_allow_html=True)

    if os.path.exists("outputs/reports/outlier_boxplots.png"):
        st.image("outputs/reports/outlier_boxplots.png")
        st.markdown('<div class="viz-caption">Box Plot of Numeric Features</div>', unsafe_allow_html=True)

    if os.path.exists("outputs/reports/outlier_percentage.png"):
        st.image("outputs/reports/outlier_percentage.png")
        st.markdown('<div class="viz-caption">Percentage of Outliers by Column</div>', unsafe_allow_html=True)

    if os.path.exists("outputs/reports/isolation_forest_outliers.png"):
        st.image("outputs/reports/isolation_forest_outliers.png")
        st.markdown('<div class="viz-caption">Isolation Forest Detected Outliers</div>', unsafe_allow_html=True)

    # -----------------------------------------------------------
    # FEATURE DISTRIBUTIONS
    # -----------------------------------------------------------
    st.markdown('<div class="section-label">Feature Distributions</div>', unsafe_allow_html=True)

    for col in profile["numeric_cols"]:
        safe_name = str(col).replace(" ", "_").replace("/", "_")
        histogram_path = f"outputs/reports/histogram_{safe_name}.png"

        if os.path.exists(histogram_path):
            st.image(histogram_path)
            st.markdown(f'<div class="viz-caption">Distribution of {col}</div>', unsafe_allow_html=True)

    with st.expander("View full text report"):
        st.text(results["report_text"])

    # -----------------------------------------------------------
    # DOWNLOAD DATA QUALITY REPORT
    # -----------------------------------------------------------
    st.markdown('<div class="section-label">Download Report</div>', unsafe_allow_html=True)

    pdf_bytes = generate_pdf_report(results["report_text"])

    st.download_button(
        label="Download report as PDF",
        data=pdf_bytes,
        file_name="data_quality_report.pdf",
        mime="application/pdf",
    )

    # -----------------------------------------------------------
    # DATA QUALITY CHATBOT
    # -----------------------------------------------------------
    st.markdown('<div class="section-label">Assistant</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display past messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input(
        "Ask a question or request a fix (e.g. 'fix missing values in Age')"
    )

    if user_input:
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("Thinking..."):
            result = chat_turn(
                user_input,
                st.session_state.chat_history[:-1],
                st.session_state.working_df,
                results["profile"],
                results["all_reports"],
                results["scoring"],
            )

        with st.chat_message("assistant"):
            st.write(result["reply"])

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result["reply"]
            }
        )

        if result["fix_applied"]:
            st.session_state.working_df = result["updated_df"]

            # THE ACTUAL FIX: re-run the full pipeline on the UPDATED
            # dataframe (not the original upload) and replace the cached
            # results. Everything downstream -- metrics, sub-scores,
            # recommendations, visualizations, PDF report, and the next
            # chatbot turn's grounding -- now reflects the fix.
            with st.spinner("Re-calibrating instrument..."):
                st.session_state.results = run_pipeline_on_df(st.session_state.working_df)

            st.success("Fix applied. Instrument re-calibrated.")

            st.rerun()

    # -----------------------------------------------------------
    # DOWNLOAD CLEANED DATASET
    # -----------------------------------------------------------
    st.markdown('<div class="section-label">Download Dataset</div>', unsafe_allow_html=True)

    csv_bytes = st.session_state.working_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download updated CSV",
        data=csv_bytes,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )

else:
    st.markdown(
        """
        <div class="readout-card" style="text-align:center; padding: 2.4rem 1rem;">
            <div class="readout-label">Awaiting Input</div>
            <div style="color:var(--text-muted); font-size:0.92rem; margin-top:0.4rem;">
                Upload a CSV above to run the full calibration — score, sub-scores,
                outlier analysis, distributions, and the assistant will appear here.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )