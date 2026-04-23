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
# COLUMN DEFINITIONS  (all collected parameters)
# ======================================================

STROKE_NUMERIC_COLS = [
    "age", "height", "weight", "bmi",
    "avg_glucose_level", "systolic_bp", "diastolic_bp",
    "sleep_hours", "risk_score",
]

STROKE_CATEGORICAL_COLS = [
    "gender", "blood_group", "genotype", "marital_status",
    "work_type", "residence_type", "smoking_status",
    "physical_activity", "salt_intake", "depression_level",
    "ptsd", "chronic_pain", "diabetes_type",
    "hypertension", "hypertension_treatment", "heart_disease",
    "noise_sources", "pollution_level_air",
    "pollution_level_water", "pollution_level_environmental",
    "risk_level",
]

ALZ_NUMERIC_COLS = [
    "age", "height", "weight", "bmi", "mmse",
    "SystolicBP", "DiastolicBP",
    "AlcoholConsumption", "PhysicalActivity",
    "DietQuality", "SleepQuality",
    "FunctionalAssessment", "ADL",
    "PollutionScore", "CustomStressScore",
    "risk_score",
]

ALZ_CATEGORICAL_COLS = [
    "gender", "blood_group", "genotype",
    "Smoking", "FamilyHistoryAlzheimers",
    "CardiovascularDisease", "Diabetes",
    "Depression", "Hypertension",
    "BehavioralProblems", "Confusion",
    "Disorientation", "PersonalityChanges",
    "DifficultyCompletingTasks", "Forgetfulness",
    "MemoryComplaints", "risk_level",
]

STRESS_NUMERIC_COLS = [
    "total_score", "financial_stress", "family_stress",
    "work_stress", "safety_stress", "caregiver_stress",
    "migration_stress", "family_expectations", "spiritual_stress",
]

STRESS_CATEGORICAL_COLS = ["stress_level"]

NUTRITION_NUMERIC_COLS = [
    "fruit_intake", "vegetable_intake",
    "hydration_liters", "nutritional_score",
]

NUTRITION_CATEGORICAL_COLS = ["lifestyle_choices"]


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

    df_nutrition = _safe_fetch("nutrition_tracker")
    df_stroke    = _safe_fetch("stroke_predictions")
    df_alzheimer = _safe_fetch("alzheimer_predictions")
    df_stress    = _safe_fetch("stress_assessments")

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
    Non-overlapping columns are filled with NaN.
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
    Fetch, clean, combine all tables and render a Streamlit
    download button for the master research CSV.
    """
    df_nutrition, df_stroke, df_alzheimer, df_stress = fetch_all_data(supabase)

    df_nutrition = clean_dataframe(df_nutrition, "nutrition")
    df_stroke    = clean_dataframe(df_stroke,    "stroke")
    df_alzheimer = clean_dataframe(df_alzheimer, "alzheimers")
    df_stress    = clean_dataframe(df_stress,    "stress")

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
# 5. DESCRIPTIVE STATISTICS HELPER
# ======================================================
def descriptive_stats(df, numeric_cols):
    """
    Return a concise descriptive-statistics table.
    Only columns that actually exist in df are included.
    """
    available = [c for c in numeric_cols if c in df.columns]
    if not available or df.empty:
        return pd.DataFrame()

    df_num = df[available].apply(pd.to_numeric, errors="coerce")
    stats = df_num.describe().T[["mean", "std", "min", "50%", "max"]].rename(
        columns={"mean": "Mean", "std": "Std Dev",
                 "min": "Min", "50%": "Median", "max": "Max"}
    )
    return stats.round(3)


# ======================================================
# 6. FREQUENCY TABLE HELPER
# ======================================================
def frequency_table(df, col):
    """
    Return a frequency + percentage table for a categorical column.
    """
    if col not in df.columns or df.empty:
        return pd.DataFrame()
    vc = df[col].value_counts(dropna=False)
    pct = (vc / len(df) * 100).round(1)
    return pd.DataFrame({"Count": vc, "Percent (%)": pct})


# ======================================================
# 7. RENDER FREQUENCY TABLES FOR A DATASET
# ======================================================
def _render_freq_tables(df, cat_cols, label):
    """Render st.expander blocks with frequency tables for each categorical col."""
    available = [c for c in cat_cols if c in df.columns]
    if not available:
        st.info(f"No categorical columns available for {label}.")
        return
    for col in available:
        with st.expander(f"📊 {col}", expanded=False):
            ft = frequency_table(df, col)
            if not ft.empty:
                st.dataframe(ft, use_container_width=True)
                st.bar_chart(ft["Count"])


# ======================================================
# 8. BASIC OVERVIEW DASHBOARD
# ======================================================
def show_dashboard(supabase):
    """
    Simple overview dashboard with counts, preview, and risk chart.
    """
    st.title("📊 Research Analytics Dashboard")

    df_nutrition, df_stroke, df_alzheimer, df_stress = fetch_all_data(supabase)

    df_nutrition = clean_dataframe(df_nutrition, "nutrition")
    df_stroke    = clean_dataframe(df_stroke,    "stroke")
    df_alzheimer = clean_dataframe(df_alzheimer, "alzheimers")
    df_stress    = clean_dataframe(df_stress,    "stress")

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
# 9. SPSS-STYLE DASHBOARD  (FULL — ALL PARAMETERS)
# ======================================================
def show_spss_dashboard(supabase):
    """
    Full SPSS-style research output dashboard covering ALL collected
    parameters across stroke, dementia, stress and nutrition datasets.
    """
    st.title("📊 SPSS-Style Research Output Dashboard")

    df_nutrition, df_stroke, df_alzheimer, df_stress = fetch_all_data(supabase)

    df_stroke    = clean_dataframe(df_stroke,    "stroke")
    df_alzheimer = clean_dataframe(df_alzheimer, "alzheimers")
    df_stress    = clean_dataframe(df_stress,    "stress")
    df_nutrition = clean_dataframe(df_nutrition, "nutrition")

    t1, t2, t3, t4 = st.tabs([
        "📈 Descriptive Stats",
        "📊 Frequency Tables",
        "🔄 Cross-Tab Analysis",
        "📋 Dataset Preview",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — DESCRIPTIVE STATISTICS
    # ══════════════════════════════════════════════════════════════════════
    with t1:
        st.header("📈 Descriptive Statistics")

        if not df_stroke.empty:
            st.subheader("🩺 Stroke Dataset")
            s = descriptive_stats(df_stroke, STROKE_NUMERIC_COLS)
            st.dataframe(s, use_container_width=True) if not s.empty else st.info("No numeric data.")

        if not df_alzheimer.empty:
            st.subheader("🧠 Alzheimer / Dementia Dataset")
            s = descriptive_stats(df_alzheimer, ALZ_NUMERIC_COLS)
            st.dataframe(s, use_container_width=True) if not s.empty else st.info("No numeric data.")

        if not df_stress.empty:
            st.subheader("😌 Stress Dataset")
            s = descriptive_stats(df_stress, STRESS_NUMERIC_COLS)
            st.dataframe(s, use_container_width=True) if not s.empty else st.info("No numeric data.")

        if not df_nutrition.empty:
            st.subheader("🥗 Nutrition Dataset")
            s = descriptive_stats(df_nutrition, NUTRITION_NUMERIC_COLS)
            st.dataframe(s, use_container_width=True) if not s.empty else st.info("No numeric data.")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — FREQUENCY TABLES
    # ══════════════════════════════════════════════════════════════════════
    with t2:
        st.header("📊 Frequency Tables")

        st.subheader("🩺 Stroke — Categorical Variables")
        _render_freq_tables(df_stroke, STROKE_CATEGORICAL_COLS, "Stroke")

        st.subheader("🧠 Alzheimer — Categorical Variables")
        _render_freq_tables(df_alzheimer, ALZ_CATEGORICAL_COLS, "Alzheimer")

        st.subheader("😌 Stress — Stress Level")
        _render_freq_tables(df_stress, STRESS_CATEGORICAL_COLS, "Stress")

        st.subheader("🥗 Nutrition — Lifestyle Choices")
        _render_freq_tables(df_nutrition, NUTRITION_CATEGORICAL_COLS, "Nutrition")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — CROSS-TAB ANALYSIS
    # ══════════════════════════════════════════════════════════════════════
    with t3:
        st.header("🔄 Cross-Tab Analysis")

        age_bins   = [0, 30, 45, 60, 75, 120]
        age_labels = ["<30", "30-45", "45-60", "60-75", "75+"]

        def _crosstab_section(df, row_col, col_col, label,
                              bins=None, bin_labels=None):
            if df.empty or row_col not in df.columns or col_col not in df.columns:
                return
            df = df.copy()
            if bins:
                df[row_col] = pd.to_numeric(df[row_col], errors="coerce")
                df[row_col] = pd.cut(df[row_col], bins=bins, labels=bin_labels)
            ctab = pd.crosstab(df[row_col], df[col_col],
                               margins=True, margins_name="Total")
            with st.expander(label, expanded=False):
                st.dataframe(ctab, use_container_width=True)

        # Stroke cross-tabs
        st.subheader("🩺 Stroke Cross-Tabs")
        _crosstab_section(df_stroke, "age",                 "risk_level",
                          "Risk Level × Age Group",
                          bins=age_bins, bin_labels=age_labels)
        _crosstab_section(df_stroke, "gender",              "risk_level",
                          "Risk Level × Gender")
        _crosstab_section(df_stroke, "blood_group",         "risk_level",
                          "Risk Level × Blood Group")
        _crosstab_section(df_stroke, "genotype",            "risk_level",
                          "Risk Level × Genotype")
        _crosstab_section(df_stroke, "marital_status",      "risk_level",
                          "Risk Level × Marital Status")
        _crosstab_section(df_stroke, "work_type",           "risk_level",
                          "Risk Level × Work Type")
        _crosstab_section(df_stroke, "residence_type",      "risk_level",
                          "Risk Level × Residence Type")
        _crosstab_section(df_stroke, "smoking_status",      "risk_level",
                          "Risk Level × Smoking Status")
        _crosstab_section(df_stroke, "physical_activity",   "risk_level",
                          "Risk Level × Physical Activity")
        _crosstab_section(df_stroke, "salt_intake",         "risk_level",
                          "Risk Level × Salt Intake")
        _crosstab_section(df_stroke, "depression_level",    "risk_level",
                          "Risk Level × Depression Level")
        _crosstab_section(df_stroke, "ptsd",                "risk_level",
                          "Risk Level × PTSD")
        _crosstab_section(df_stroke, "chronic_pain",        "risk_level",
                          "Risk Level × Chronic Pain")
        _crosstab_section(df_stroke, "diabetes_type",       "risk_level",
                          "Risk Level × Diabetes Type")
        _crosstab_section(df_stroke, "hypertension",        "risk_level",
                          "Risk Level × Hypertension")
        _crosstab_section(df_stroke, "hypertension_treatment","risk_level",
                          "Risk Level × Hypertension Treatment")
        _crosstab_section(df_stroke, "heart_disease",       "risk_level",
                          "Risk Level × Heart Disease")
        _crosstab_section(df_stroke, "noise_sources",       "risk_level",
                          "Risk Level × Noise Sources")
        _crosstab_section(df_stroke, "pollution_level_air", "risk_level",
                          "Risk Level × Air Pollution")
        _crosstab_section(df_stroke, "pollution_level_water","risk_level",
                          "Risk Level × Water Pollution")
        _crosstab_section(df_stroke, "pollution_level_environmental","risk_level",
                          "Risk Level × Environmental Pollution")

        st.markdown("---")

        # Alzheimer cross-tabs
        st.subheader("🧠 Alzheimer / Dementia Cross-Tabs")
        _crosstab_section(df_alzheimer, "age",                       "risk_level",
                          "Risk Level × Age Group",
                          bins=age_bins, bin_labels=age_labels)
        _crosstab_section(df_alzheimer, "gender",                    "risk_level",
                          "Risk Level × Gender")
        _crosstab_section(df_alzheimer, "blood_group",               "risk_level",
                          "Risk Level × Blood Group")
        _crosstab_section(df_alzheimer, "genotype",                  "risk_level",
                          "Risk Level × Genotype")
        _crosstab_section(df_alzheimer, "Smoking",                   "risk_level",
                          "Risk Level × Smoking")
        _crosstab_section(df_alzheimer, "FamilyHistoryAlzheimers",   "risk_level",
                          "Risk Level × Family History of Alzheimer's")
        _crosstab_section(df_alzheimer, "CardiovascularDisease",     "risk_level",
                          "Risk Level × Cardiovascular Disease")
        _crosstab_section(df_alzheimer, "Diabetes",                  "risk_level",
                          "Risk Level × Diabetes")
        _crosstab_section(df_alzheimer, "Depression",                "risk_level",
                          "Risk Level × Depression")
        _crosstab_section(df_alzheimer, "Hypertension",              "risk_level",
                          "Risk Level × Hypertension")
        _crosstab_section(df_alzheimer, "BehavioralProblems",        "risk_level",
                          "Risk Level × Behavioral Problems")
        _crosstab_section(df_alzheimer, "Confusion",                 "risk_level",
                          "Risk Level × Confusion")
        _crosstab_section(df_alzheimer, "Disorientation",            "risk_level",
                          "Risk Level × Disorientation")
        _crosstab_section(df_alzheimer, "PersonalityChanges",        "risk_level",
                          "Risk Level × Personality Changes")
        _crosstab_section(df_alzheimer, "DifficultyCompletingTasks", "risk_level",
                          "Risk Level × Difficulty Completing Tasks")
        _crosstab_section(df_alzheimer, "Forgetfulness",             "risk_level",
                          "Risk Level × Forgetfulness")
        _crosstab_section(df_alzheimer, "MemoryComplaints",          "risk_level",
                          "Risk Level × Memory Complaints")

        st.markdown("---")

        # Stress cross-tabs
        st.subheader("😌 Stress Cross-Tabs")
        _crosstab_section(df_stress, "stress_level", "user_id",
                          "Count by Stress Level")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — DATASET PREVIEW WITH COLUMN FILTER + PER-TABLE CSV
    # ══════════════════════════════════════════════════════════════════════
    with t4:
        st.header("📋 Dataset Previews")

        dataset_choice = st.selectbox(
            "Choose dataset to preview",
            ["Stroke", "Alzheimer / Dementia", "Stress", "Nutrition", "All Combined"],
            key="spss_preview_choice"
        )

        preview_map = {
            "Stroke":               df_stroke,
            "Alzheimer / Dementia": df_alzheimer,
            "Stress":               df_stress,
            "Nutrition":            df_nutrition,
            "All Combined":         combine_all([df_stroke, df_alzheimer,
                                                 df_stress, df_nutrition]),
        }

        chosen_df = preview_map[dataset_choice]

        if chosen_df.empty:
            st.info("No records found for this dataset.")
        else:
            st.caption(f"{len(chosen_df)} total records")

            all_cols = list(chosen_df.columns)
            selected_cols = st.multiselect(
                "Filter columns to display",
                options=all_cols,
                default=all_cols,
                key="spss_col_filter"
            )
            display_df = chosen_df[selected_cols] if selected_cols else chosen_df
            st.dataframe(display_df, use_container_width=True)

            csv_bytes = display_df.to_csv(index=False).encode("utf-8")
            fname = dataset_choice.lower().replace(" / ", "_").replace(" ", "_")
            st.download_button(
                label=f"📥 Download {dataset_choice} CSV",
                data=csv_bytes,
                file_name=f"{fname}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="spss_download_btn"
            )
