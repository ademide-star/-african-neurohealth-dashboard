"""
utils.py — African NeuroHealth AI
Helper functions for data fetching, cleaning, and export.
Imported by app.py — must live in the same directory.
"""

import pandas as pd
import streamlit as st
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ======================================================
# 1. FETCH DATA FROM SUPABASE
# ======================================================
def fetch_all_data(supabase):
    """
    Fetch all four tables from Supabase.
    Returns four DataFrames: nutrition, stroke, alzheimer, stress.
    Any table that fails returns an empty DataFrame.
    """
    def _safe_fetch(table_name):
        try:
            data = supabase.table(table_name).select("*").execute().data
            return pd.DataFrame(data) if data else pd.DataFrame()
        except Exception as e:
            logger.warning(f"Could not fetch '{table_name}': {e}")
            return pd.DataFrame()

    df_nutrition  = _safe_fetch("nutrition_tracker")
    df_stroke     = _safe_fetch("stroke_predictions")
    df_alzheimer  = _safe_fetch("alzheimer_predictions")
    df_stress     = _safe_fetch("stress_assessments")

    return df_nutrition, df_stroke, df_alzheimer, df_stress


# ======================================================
# 2. CLEAN DATA
# ======================================================
def clean_dataframe(df, source_name):
    """
    Standardise a raw Supabase DataFrame:
    - Add a 'source' column
    - Parse 'created_at' to datetime
    - Fill NaN with 'N/A'
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["source"] = source_name

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    df = df.fillna("N/A")
    return df


# ======================================================
# 3. COMBINE DATASETS
# ======================================================
def combine_all(df_list):
    """
    Concatenate a list of DataFrames into one master DataFrame.
    Non-overlapping columns are filled with NaN (sort=False keeps column order).
    """
    non_empty = [df for df in df_list if df is not None and not df.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True, sort=False)


# ======================================================
# 4. EXPORT CSV (RESEARCH DATASET)
# ======================================================
def export_csv(supabase):
    """
    Fetch, clean, combine all tables and render a Streamlit download button
    for the master research CSV.
    """
    df_nutrition, df_stroke, df_alzheimer, df_stress = fetch_all_data(supabase)

    df_nutrition  = clean_dataframe(df_nutrition,  "nutrition")
    df_stroke     = clean_dataframe(df_stroke,     "stroke")
    df_alzheimer  = clean_dataframe(df_alzheimer,  "alzheimers")
    df_stress     = clean_dataframe(df_stress,     "stress")

    master_df = combine_all([df_nutrition, df_stroke, df_alzheimer, df_stress])

    if master_df.empty:
        st.warning("No data found across any table.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_bytes = master_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Master Research Dataset (CSV)",
        data=csv_bytes,
        file_name=f"african_neurohealth_dataset_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(f"Dataset contains {len(master_df)} records across all tables.")


# ======================================================
# 5. BASIC OVERVIEW DASHBOARD
# ======================================================
def show_dashboard(supabase):
    """
    Render a simple overview dashboard with record counts,
    a combined dataset preview, and a risk-level bar chart.
    """
    st.title("📊 Research Analytics Dashboard")

    df_nutrition, df_stroke, df_alzheimer, df_stress = fetch_all_data(supabase)

    df_nutrition  = clean_dataframe(df_nutrition,  "nutrition")
    df_stroke     = clean_dataframe(df_stroke,     "stroke")
    df_alzheimer  = clean_dataframe(df_alzheimer,  "alzheimers")
    df_stress     = clean_dataframe(df_stress,     "stress")

    # ── Record counts ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nutrition",  len(df_nutrition))
    col2.metric("Stroke",     len(df_stroke))
    col3.metric("Alzheimer",  len(df_alzheimer))
    col4.metric("Stress",     len(df_stress))

    st.markdown("---")

    master_df = combine_all([df_nutrition, df_stroke, df_alzheimer, df_stress])

    st.subheader("📋 Combined Dataset Preview (first 50 rows)")
    st.dataframe(master_df.head(50), use_container_width=True)

    if "risk_level" in master_df.columns:
        st.subheader("🧠 Risk Level Distribution (all tables)")
        st.bar_chart(master_df["risk_level"].value_counts())


# ======================================================
# 6. DESCRIPTIVE STATISTICS HELPER
# ======================================================
def descriptive_stats(df, numeric_cols):
    """
    Return a concise descriptive-statistics table for the given numeric columns.
    Only columns that actually exist in df are included.
    """
    available = [c for c in numeric_cols if c in df.columns]
    if not available or df.empty:
        return pd.DataFrame()

    df_num = df[available].apply(pd.to_numeric, errors="coerce")
    stats = df_num.describe().T[["mean", "std", "min", "50%", "max"]].rename(
        columns={"mean": "Mean", "std": "Std Dev", "min": "Min",
                 "50%": "Median", "max": "Max"}
    )
    return stats.round(3)


# ======================================================
# 7. SPSS-STYLE DASHBOARD
# ======================================================
def show_spss_dashboard(supabase):
    """
    Render an SPSS-style research output with:
    - Descriptive statistics tables
    - Frequency / distribution charts
    - Cross-tabulation analysis
    - Combined dataset preview
    """
    st.title("📊 SPSS-Style Research Output Dashboard")

    df_nutrition, df_stroke, df_alzheimer, df_stress = fetch_all_data(supabase)

    df_stroke     = clean_dataframe(df_stroke,     "stroke")
    df_alzheimer  = clean_dataframe(df_alzheimer,  "alzheimers")
    df_stress     = clean_dataframe(df_stress,     "stress")

    # ── 1. Descriptive Statistics ─────────────────────────────────────────
    st.header("📈 Descriptive Statistics")

    if not df_stroke.empty:
        stroke_stats = descriptive_stats(
            df_stroke,
            ["age", "bmi", "avg_glucose_level", "systolic_bp", "diastolic_bp", "risk_score"]
        )
        if not stroke_stats.empty:
            st.subheader("Stroke Dataset")
            st.dataframe(stroke_stats, use_container_width=True)

    if not df_alzheimer.empty:
        alz_stats = descriptive_stats(
            df_alzheimer,
            ["age", "bmi", "mmse", "SystolicBP", "DiastolicBP", "risk_score"]
        )
        if not alz_stats.empty:
            st.subheader("Alzheimer Dataset")
            st.dataframe(alz_stats, use_container_width=True)

    if not df_stress.empty:
        stress_stats = descriptive_stats(
            df_stress,
            ["total_score", "financial_stress", "family_stress",
             "work_stress", "caregiver_stress"]
        )
        if not stress_stats.empty:
            st.subheader("Stress Dataset")
            st.dataframe(stress_stats, use_container_width=True)

    # ── 2. Frequency Tables ───────────────────────────────────────────────
    st.header("📊 Frequency Tables")

    if "risk_level" in df_stroke.columns and not df_stroke.empty:
        st.subheader("Stroke Risk Levels")
        st.bar_chart(df_stroke["risk_level"].value_counts())

    if "risk_level" in df_alzheimer.columns and not df_alzheimer.empty:
        st.subheader("Alzheimer Risk Levels")
        st.bar_chart(df_alzheimer["risk_level"].value_counts())

    if "stress_level" in df_stress.columns and not df_stress.empty:
        st.subheader("Stress Levels")
        st.bar_chart(df_stress["stress_level"].value_counts())

    # ── 3. Cross-Tab Analysis ─────────────────────────────────────────────
    st.header("🔄 Cross-Tab Analysis")

    if not df_stroke.empty and "age" in df_stroke.columns and "risk_level" in df_stroke.columns:
        st.subheader("Stroke: Risk Level × Age Group")
        df_stroke["age"] = pd.to_numeric(df_stroke["age"], errors="coerce")
        df_stroke["age_group"] = pd.cut(
            df_stroke["age"],
            bins=[0, 30, 45, 60, 75, 120],
            labels=["<30", "30-45", "45-60", "60-75", "75+"]
        )
        ctab = pd.crosstab(df_stroke["age_group"], df_stroke["risk_level"])
        st.dataframe(ctab, use_container_width=True)

    if not df_alzheimer.empty and "age" in df_alzheimer.columns and "risk_level" in df_alzheimer.columns:
        st.subheader("Alzheimer: Risk Level × Age Group")
        df_alzheimer["age"] = pd.to_numeric(df_alzheimer["age"], errors="coerce")
        df_alzheimer["age_group"] = pd.cut(
            df_alzheimer["age"],
            bins=[0, 30, 45, 60, 75, 120],
            labels=["<30", "30-45", "45-60", "60-75", "75+"]
        )
        ctab2 = pd.crosstab(df_alzheimer["age_group"], df_alzheimer["risk_level"])
        st.dataframe(ctab2, use_container_width=True)

    # ── 4. Combined Dataset Preview ───────────────────────────────────────
    st.header("📋 Combined Dataset Preview")

    master_df = combine_all([df_stroke, df_alzheimer, df_stress])
    if not master_df.empty:
        st.dataframe(master_df.head(100), use_container_width=True)
    else:
        st.info("No data available yet.")
