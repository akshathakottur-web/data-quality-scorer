"""
app.py

Streamlit web interface for the Automatic Data Quality Scoring System.
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


if "GROQ_API_KEY" not in st.secrets:
    st.error("Groq API key not configured. Add it to .streamlit/secrets.toml")
    st.stop()

# Imported here (after the secrets check) since chatbot.py constructs the
# Groq client at import time and would fail immediately without a key.
from src.dqs.chatbot import chat_turn

st.set_page_config(page_title="Data Quality Scorer", layout="wide")

st.title("🔍 Automatic Data Quality Scoring System")
st.write("Upload a CSV file to automatically analyze its quality for machine learning.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

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

        with st.spinner("Analyzing dataset..."):
            st.session_state.results = run_pipeline(tmp_path)

        st.session_state.working_df = pd.read_csv(tmp_path)
        os.unlink(tmp_path)

    results = st.session_state.results
    scoring = results["scoring"]
    profile = results["profile"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Score", f"{scoring['overall_score']}/100")
    col2.metric("Status", scoring["status"])
    col3.metric("Rows x Columns", f"{profile['n_rows']} x {profile['n_cols']}")

    st.subheader("Sub-Scores")
    scores_df = pd.DataFrame(
        list(scoring["individual_scores"].items()),
        columns=["Detector", "Score"]
    )
    st.bar_chart(scores_df.set_index("Detector"))

    st.subheader("Recommendations")
    for rec in results["recommendations"]:
        st.write(f"- {rec}")

    st.subheader("Visualizations")
    img_col1, img_col2 = st.columns(2)

    if os.path.exists("outputs/reports/missing_values_heatmap.png"):
        img_col1.image(
            "outputs/reports/missing_values_heatmap.png",
            caption="Missing Values Heatmap"
        )

    if os.path.exists("outputs/reports/score_breakdown.png"):
        img_col2.image(
            "outputs/reports/score_breakdown.png",
            caption="Score Breakdown"
        )

    # ---------------------------------------------
    # OUTLIER VISUALIZATIONS
    # ---------------------------------------------

    st.subheader("📊 Outlier Analysis")

    if os.path.exists("outputs/reports/outlier_boxplots.png"):
        st.image(
            "outputs/reports/outlier_boxplots.png",
            caption="Box Plot of Numeric Features"
        )

    if os.path.exists("outputs/reports/outlier_percentage.png"):
        st.image(
            "outputs/reports/outlier_percentage.png",
            caption="Percentage of Outliers by Column"
        )

    if os.path.exists("outputs/reports/isolation_forest_outliers.png"):
        st.image(
            "outputs/reports/isolation_forest_outliers.png",
            caption="Isolation Forest Detected Outliers"
        )

    st.subheader("📈 Feature Distributions")

    for col in profile["numeric_cols"]:

        safe_name = str(col).replace(" ", "_").replace("/", "_")

        histogram_path = (
            f"outputs/reports/histogram_{safe_name}.png"
        )

        if os.path.exists(histogram_path):

            st.image(
                histogram_path,
                caption=f"Distribution of {col}"
            )

    with st.expander("View full text report"):
        st.text(results["report_text"])

    # ---------------------------------------------
    # DOWNLOAD DATA QUALITY REPORT
    # ---------------------------------------------

    st.subheader("📄 Download Data Quality Report")

    pdf_bytes = generate_pdf_report(
        results["report_text"]
    )

    st.download_button(
        label="Download Report as PDF",
        data=pdf_bytes,
        file_name="data_quality_report.pdf",
        mime="application/pdf",
    )

    # ---------------------------------------------
    # DATA QUALITY CHATBOT
    # ---------------------------------------------

    st.subheader("💬 Chat with your Data Quality Assistant")

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
            with st.spinner("Re-analyzing updated dataset..."):
                st.session_state.results = run_pipeline_on_df(st.session_state.working_df)

            st.success("Fix applied! Analysis updated.")

            st.rerun()

    # ---------------------------------------------
    # DOWNLOAD CLEANED DATASET
    # ---------------------------------------------

    st.subheader("📥 Download Cleaned Dataset")

    csv_bytes = st.session_state.working_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download updated CSV",
        data=csv_bytes,
        file_name="cleaned_dataset.csv",
        mime="text/csv",
    )

else:
    st.info("Upload a CSV file to get started.")