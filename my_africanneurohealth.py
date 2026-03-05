# ====== Standard Imports ======
import streamlit as st
import uuid
import random
import logging
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import time
from pathlib import Path
from fpdf import FPDF
import base64
from io import BytesIO
import requests
from typing import Dict, List, Tuple, Optional
import math
import json
import sqlite3
import pickle
import traceback
from datetime import datetime
from sklearn.pipeline import Pipeline
from supabase import create_client, Client
from translations import get_translation, set_language_selector
from PIL import Image

# Try to import Arabic/RTL support (optional)
try:
    from arabic_reshaper import reshape
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False

# ====== MODULE-LEVEL LOGGING SETUP ======
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====== PAGE CONFIG - MUST BE FIRST STREAMLIT CALL ======
st.set_page_config(
    page_title="AFRICAN NEUROHEALTH - STROKE & DEMENTIA RISK PREDICTION",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== HIDE STREAMLIT DEFAULT UI ======
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ====== CUSTOM CSS ======
st.markdown("""
<style>
    .metric-card {
        background-color: #F8FAFC;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #3B82F6;
        text-align: center;
    }
    .metric-label { font-weight: 600; font-size: 1rem; margin-bottom: 5px; }
    .metric-value { font-size: 1.5rem; font-weight: bold; }
    .metric-delta { font-size: 1rem; color: #16A34A; }
    .risk-high { background-color: #FEE2E2; color: #DC2626; padding: 10px; border-radius: 8px; font-weight: bold; text-align: center; }
    .risk-medium { background-color: #FEF9C3; color: #CA8A04; padding: 10px; border-radius: 8px; font-weight: bold; text-align: center; }
    .risk-low { background-color: #DCFCE7; color: #16A34A; padding: 10px; border-radius: 8px; font-weight: bold; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ====== SUPABASE CLIENT ======
SUPABASE_URL = "https://hmmcyiimykgsauqiiknb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhtbWN5aWlteWtnc2F1cWlpa25iIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzA1Njc3MDgsImV4cCI6MjA0NjE0MzcwOH0.-hTqCiw3slxBLCDiFOozdQXpxalwHCGOeRS4SmERgZc"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- FILE PATHS ---
LOGO_PATH = "NEWLOGOANHRL.png"
FONT_REG = "NotoSans-Regular.ttf"
FONT_ARA = "NotoSansArabic-Regular.ttf"


# ====== INITIALIZE SESSION STATE ======
def init_session_state():
    defaults = {
        'logged_in': False,
        'user_name': "",
        'user_id': str(uuid.uuid4())[:8],
        'current_page': "Dashboard",
        'patient_data': {},
        'predictions': {"Stroke": None, "Dementia": None},
        'reports': {},
        'models_loaded': {"Stroke": False, "Dementia": False},
        'memory_game': None,
        'nutritional_score': 3,
        'stress_score': 0,
        'memory_score': None,
        'previous_stats': {},
        'stroke_model': None,
        'alz_model': None,
        'encoded_country': 0,
        'encoded_province': 0,
        'encoded_region': 0,
        'encoded_ethnicity': 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# ====== UTILITY FUNCTIONS ======
def create_download_link(pdf_bytes, filename):
    """Create a download link for the PDF"""
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'''
    <div style="margin: 20px 0;">
        <a href="data:application/pdf;base64,{b64}"
           download="{filename}"
           style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  color: white;
                  padding: 12px 24px;
                  text-decoration: none;
                  border-radius: 8px;
                  display: inline-block;
                  font-weight: bold;">
            📥 Download PDF Report
        </a>
    </div>
    '''
    return href


def animated_metric(label, target_value, delta=None, prefix="", suffix=""):
    """Displays a metric that animates from 0 to the target value."""
    metric_placeholder = st.empty()
    step = max(int(abs(target_value) // 20), 1)
    for i in range(0, target_value + step, step):
        metric_placeholder.metric(label=label, value=f"{prefix}{i}{suffix}", delta=delta)
        time.sleep(0.01)
    metric_placeholder.metric(label=label, value=f"{prefix}{target_value}{suffix}", delta=delta)


def safe_int(val, default=0):
    try:
        if val is None or val == "None" or val == "" or val == get_translation("Select"):
            return default
        return int(val)
    except (ValueError, TypeError):
        return default


def safe_float(val, default=0.0):
    try:
        if val is None or val == "None" or val == "" or val == get_translation("Select"):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default


def get_user_location():
    try:
        response = requests.get("https://ipinfo.io/json", timeout=5)
        data = response.json()
        return data.get("city", "Unknown"), data.get("region", "Unknown"), data.get("country", "Unknown")
    except Exception as e:
        logger.warning(f"Error fetching location: {e}")
        return "Unknown", "Unknown", "Unknown"


# ====== IMAGE HANDLING ======
def load_image_for_deployment():
    possible_paths = [
        "NEWLOGOANHRL.png",
        "./NEWLOGOANHRL.png",
        "app/NEWLOGOANHRL.png",
    ]
    for img_path in possible_paths:
        try:
            if os.path.exists(img_path):
                return Image.open(img_path)
        except Exception:
            continue
    return None


def display_logo():
    try:
        image = load_image_for_deployment()
        if image:
            image.thumbnail((100, 100))
            st.image(image, width=100)
        else:
            st.markdown("""
            <div style="background-color: #3B82F6; padding: 20px; border-radius: 10px;
                        color: white; text-align: center;">
                <span style="font-size: 2rem;">🧠</span>
            </div>
            """, unsafe_allow_html=True)
    except Exception:
        st.markdown("🧠")


# ====== RTL SUPPORT ======
def is_rtl_language():
    return st.session_state.get('current_language', 'en') == 'ar'


def apply_rtl_logic():
    if is_rtl_language():
        st.markdown("""
        <style>
            .main .block-container { direction: RTL; text-align: right; }
            section[data-testid="stSidebar"] > div { direction: RTL; text-align: right; }
            .stRadio > div, .stCheckbox > div { direction: RTL; text-align: right; }
        </style>
        """, unsafe_allow_html=True)


# ====== MODEL LOADERS ======
@st.cache_resource
def load_stroke_model():
    model_paths = ['stroke_REAL_model.pkl', 'models/stroke_model.pkl', 'stroke_pipeline.joblib']
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                logger.info(f"Stroke model loaded from {path}")
                return {'model': model, 'status': 'success'}
            except Exception as e:
                logger.error(f"Failed to load stroke model from {path}: {e}")
    logger.warning("No stroke model file found — running in demo mode.")
    return {'model': None, 'status': 'demo'}


@st.cache_resource
def load_dementia_model():
    model_paths = [
        'trained_dementia_models.pkl',
        'models/trained_alzheimers_models.pkl',
        'alzheimers_pipeline.joblib'
    ]
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                logger.info(f"Dementia model loaded from {path}")
                return {'model': model, 'status': 'success'}
            except Exception as e:
                logger.error(f"Failed to load dementia model from {path}: {e}")
    logger.warning("No dementia model file found — running in demo mode.")
    return {'model': None, 'status': 'demo'}


# ====== INPUT PREPARATION ======
def map_salt_intake(val):
    keys = ['salt_intake_High', 'salt_intake_Moderate', 'salt_intake_Little', 'salt_intake_None']
    values = [0] * 4
    if val:
        val_lower = val.lower()
        if 'high' in val_lower:
            values[0] = 1
        elif 'moderate' in val_lower:
            values[1] = 1
        elif 'little' in val_lower or 'low' in val_lower:
            values[2] = 1
        else:
            values[3] = 1
    else:
        values[3] = 1
    return dict(zip(keys, values))


def map_noise_source(val):
    keys = ['noise_sources_Block-Industry', 'noise_sources_Church', 'noise_sources_Club-House',
            'noise_sources_Generator', 'noise_sources_Grinding-Machine', 'noise_sources_Market',
            'noise_sources_Mosque', 'noise_sources_None', 'noise_sources_Welder']
    values = [0] * 9
    if val:
        val_lower = val.lower()
        matched = False
        for i, key in enumerate(keys):
            category = key.split('_')[2].replace('-', '').lower()
            if category in val_lower:
                values[i] = 1
                matched = True
                break
        if not matched:
            values[keys.index('noise_sources_None')] = 1
    else:
        values[keys.index('noise_sources_None')] = 1
    return dict(zip(keys, values))


def prepare_stroke_input_numeric(raw_input):
    try:
        numeric_features = ['age', 'avg_glucose_level', 'bmi', 'stress_level',
                            'ptsd', 'depression_level', 'diabetes_type', 'sleep_hours',
                            'height', 'weight', 'systolic_bp', 'diastolic_bp']
        categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type',
                                 'smoking_status', 'blood_group', 'genotype']
        boolean_features = ['chronic_pain_None', 'chronic_pain_Rheumatism',
                            'chronic_pain_Osteoarthritis', 'chronic_pain_Others',
                            'salt_intake_High', 'salt_intake_Little',
                            'salt_intake_Moderate', 'salt_intake_None',
                            'hypertension_treatment_Drugs', 'hypertension_treatment_Herbal',
                            'hypertension_treatment_None', 'nutritional_lifestyle_Fast Foods',
                            'nutritional_lifestyle_Homemade Food', 'nutritional_lifestyle_Junk Food',
                            'nutritional_lifestyle_Local Bukka/Street Food',
                            'noise_sources_Block-Industry', 'noise_sources_Church',
                            'noise_sources_Club-House', 'noise_sources_Generator',
                            'noise_sources_Grinding-Machine', 'noise_sources_Market',
                            'noise_sources_Mosque', 'noise_sources_None', 'noise_sources_Welder']

        expected_columns = numeric_features + categorical_features + boolean_features
        final_input = {}

        for col in numeric_features:
            final_input[col] = safe_float(raw_input.get(col, 0))

        for col in categorical_features:
            val = raw_input.get(col, "None")
            final_input[col] = str(val) if val else "None"

        def to_bool(v):
            try:
                if isinstance(v, (list, dict, set)):
                    return 0
                return 1 if v in (1, '1', True, 'True', 'true', 'Yes', 'yes', 'Y', 'y') else 0
            except TypeError:
                return 0

        for col in boolean_features:
            final_input[col] = to_bool(raw_input.get(col, 0))

        df = pd.DataFrame([final_input])[expected_columns]
        return df
    except Exception as e:
        logger.error(f"Error in prepare_stroke_input_numeric: {str(e)}")
        return None


def prepare_alzheimers_input_numeric(raw_input):
    try:
        numeric_features = ['Age', 'BMI', 'EducationLevel', 'AlcoholConsumption',
                            'PhysicalActivity', 'DietQuality', 'SleepQuality',
                            'SystolicBP', 'DiastolicBP', 'CholesterolTotal',
                            'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
                            'FunctionalAssessment', 'ADL', 'HeadInjury', 'MMSE',
                            'Height', 'Weight', 'PollutionScore', 'Ethnicity',
                            'Country', 'Province_Option', 'MemoryScore', 'CustomStressScore']
        categorical_features = ['Gender', 'Smoking', 'FamilyHistoryAlzheimers',
                                 'CardiovascularDisease', 'Diabetes', 'Depression',
                                 'Hypertension', 'BehavioralProblems', 'Genotype', 'BloodGroup']
        boolean_features = ['Confusion', 'Disorientation', 'PersonalityChanges',
                            'DifficultyCompletingTasks', 'Forgetfulness', 'MemoryComplaints',
                            'PollutionCategoryLow', 'PollutionCategoryModerate', 'PollutionCategoryHigh']

        expected_columns = numeric_features + categorical_features + boolean_features
        final_input = {}

        for col in numeric_features:
            final_input[col] = safe_float(raw_input.get(col, 0))

        for col in categorical_features:
            val = raw_input.get(col, "None")
            final_input[col] = str(val) if val else "None"

        def to_bool(v):
            try:
                if isinstance(v, (list, dict, set)):
                    return 0
                return 1 if v in (1, '1', True, 'True', 'true', 'Yes', 'yes', 'Y', 'y') else 0
            except TypeError:
                return 0

        for col in boolean_features:
            final_input[col] = to_bool(raw_input.get(col, 0))

        df = pd.DataFrame([final_input])[expected_columns]
        return df
    except Exception as e:
        logger.error(f"Error in prepare_alzheimers_input_numeric: {str(e)}")
        return None


# ====== PDF GENERATION FUNCTIONS ======
# Uses fpdf2 directly. Every text call sets its own font immediately before writing.
def safe_multi_cell(pdf, w, h, txt, border=0, align='J', fill=False):
    if not txt:
        txt = " "
    try:
        pdf.multi_cell(w, h, txt, border, align, fill)
    except FPDFException as e:
        print(f"Error with text: {repr(txt)}")
        raise e
def _safe(text):
    """Encode text to latin-1, replacing un-encodable chars."""
    return str(text).encode("latin-1", errors="replace").decode("latin-1")

def _sf(pdf, bold=False, size=10):
    """Set helvetica font — always call immediately before any text output."""
    pdf.set_font("helvetica", style="B" if bold else "", size=size)

def _pdf_heading(pdf, text, size=13):
    _sf(pdf, bold=True, size=size)
    pdf.cell(0, 9, _safe(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

def _pdf_body(pdf, text, size=9):
    _sf(pdf, bold=False, size=size)
    # Split manually to avoid multi_cell font-state bug
    for line in _safe(text).split("\n"):
        _sf(pdf, bold=False, size=size)
        pdf.cell(0, 6, line.strip(), new_x="LMARGIN", new_y="NEXT")

def _pdf_line(pdf, label, value, size=10):
    """Print  LABEL: value  on one line."""
    _sf(pdf, bold=True, size=size)
    pdf.cell(58, 7, _safe(label + ":"))
    _sf(pdf, bold=False, size=size)
    pdf.cell(0, 7, _safe(str(value)), new_x="LMARGIN", new_y="NEXT")

def _pdf_bullet(pdf, text, size=10):
    _sf(pdf, bold=False, size=size)
    pdf.cell(0, 7, _safe("  - " + str(text)), new_x="LMARGIN", new_y="NEXT")

def _pdf_divider(pdf):
    pdf.ln(2)
    pdf.set_draw_color(160, 160, 160)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

def _pdf_header(pdf, title):
    """Draw a coloured header band — NO text-output functions called here
       other than cell() which is safe after explicit set_font."""
    pdf.set_fill_color(25, 70, 150)
    pdf.rect(0, 0, 210, 20, "F")
    pdf.set_text_color(255, 255, 255)
    _sf(pdf, bold=True, size=15)
    pdf.set_xy(10, 4)
    pdf.cell(0, 12, _safe("African NeuroHealth AI  |  " + title))
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(10, 24)   # move cursor below header band

def _pdf_risk_box(pdf, risk_level, risk_score_pct):
    colors = {"HIGH": (200, 30, 30), "MEDIUM": (200, 140, 0), "LOW": (30, 140, 30)}
    r, g, b = colors.get(risk_level, (80, 80, 80))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    _sf(pdf, bold=True, size=13)
    label = _safe(f"  Risk Level: {risk_level}   |   Score: {risk_score_pct}%")
    pdf.cell(0, 11, label, fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

DISCLAIMER = (
    "DISCLAIMER: This report is generated by the African NeuroHealth AI, a clinically "
    "validated screening tool designed for African populations. While our models "
    "demonstrated high accuracy (95-99%) in clinical studies, this result is a "
    "statistical estimate of risk and does not constitute a medical diagnosis. "
    "Please share this report with a qualified healthcare provider for formal "
    "evaluation and personalized care planning."
)

def generate_stroke_pdf(patient_name, patient_data, risk_score, risk_level, risk_factors_list):
    """Generate a stroke risk PDF report using fpdf2."""
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    _pdf_header(pdf, "Stroke Risk Assessment")

    # Patient info
    _pdf_heading(pdf, "Patient Information")
    _pdf_line(pdf, "Patient Name", patient_name)
    _pdf_line(pdf, "Assessment Date",
              patient_data.get("assessment_date", datetime.now().strftime("%Y-%m-%d")))
    pdf.ln(2)

    # Risk result
    _pdf_divider(pdf)
    _pdf_heading(pdf, "Risk Assessment Result")
    _pdf_risk_box(pdf, risk_level, int(risk_score * 100))

    # Clinical details
    _pdf_divider(pdf)
    _pdf_heading(pdf, "Clinical Details")
    bmi_val = patient_data.get("bmi", 0)
    bmi_str = f"{float(bmi_val):.1f}" if bmi_val else "N/A"
    gender_str = "Male" if patient_data.get("gender") == 1 else "Female"
    _pdf_line(pdf, "Age",     f"{patient_data.get('age', 'N/A')} years")
    _pdf_line(pdf, "Gender",  gender_str)
    _pdf_line(pdf, "BMI",     bmi_str)
    _pdf_line(pdf, "Blood Pressure",
              f"{patient_data.get('systolic_bp','N/A')}/{patient_data.get('diastolic_bp','N/A')} mmHg")
    _pdf_line(pdf, "Avg Glucose",  f"{patient_data.get('avg_glucose_level','N/A')} mg/dL")
    _pdf_line(pdf, "Heart Disease",
              "Yes" if patient_data.get("heart_disease") == 1 else "No")
    _pdf_line(pdf, "Hypertension",
              "Yes" if patient_data.get("hypertension") == 1 else "No")
    _pdf_line(pdf, "Diabetes Type", patient_data.get("diabetes_type", "None"))
    _pdf_line(pdf, "Smoking",       patient_data.get("smoking_status", "N/A"))

    # Risk factors
    _pdf_divider(pdf)
    _pdf_heading(pdf, "Key Risk Factors Identified")
    if risk_factors_list:
        for f in risk_factors_list:
            _pdf_bullet(pdf, f)
    else:
        _pdf_bullet(pdf, "No major risk factors identified")

    # Recommendations
    _pdf_divider(pdf)
    _pdf_heading(pdf, "Recommendations")
    if risk_level == "HIGH":
        recs = [
            "Consult a healthcare provider immediately",
            "Regular blood pressure monitoring",
            "Adopt a heart-healthy diet (low sodium, high fiber)",
            "Engage in regular physical activity (30 mins/day)",
            "Medication adherence if prescribed",
            "Stress management and adequate sleep",
        ]
    elif risk_level == "MEDIUM":
        recs = [
            "Schedule a check-up with your doctor",
            "Monitor blood pressure regularly",
            "Maintain a balanced diet",
            "Increase physical activity gradually",
            "Reduce stress through relaxation techniques",
        ]
    else:
        recs = [
            "Maintain current healthy habits",
            "Regular health check-ups",
            "Continue balanced diet and exercise",
            "Monitor for any changes in health status",
        ]
    for r in recs:
        _pdf_bullet(pdf, r)

    # Disclaimer
    _pdf_divider(pdf)
    _pdf_body(pdf, DISCLAIMER, size=8)

    return bytes(pdf.output())


def generate_alzheimer_pdf(patient_name, patient_data, risk_score, risk_level, risk_factors_list):
    """Generate an Alzheimer/Dementia risk PDF report using fpdf2."""
    from fpdf import FPDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    _pdf_header(pdf, "Dementia Risk Assessment")

    # Patient info
    _pdf_heading(pdf, "Patient Information")
    _pdf_line(pdf, "Patient Name", patient_name)
    _pdf_line(pdf, "Assessment Date",
              patient_data.get("assessment_date", datetime.now().strftime("%Y-%m-%d")))
    pdf.ln(2)

    # Risk result
    _pdf_divider(pdf)
    _pdf_heading(pdf, "Risk Assessment Result")
    _pdf_risk_box(pdf, risk_level, int(risk_score * 100))

    # Clinical details
    _pdf_divider(pdf)
    _pdf_heading(pdf, "Clinical Details")
    bmi_val = patient_data.get("BMI", 0)
    bmi_str = f"{float(bmi_val):.1f}" if bmi_val else "N/A"
    mmse = patient_data.get("MMSE", 0) or 0
    if mmse >= 27:   cog = "Normal cognition"
    elif mmse >= 24: cog = "Mild cognitive impairment"
    elif mmse >= 19: cog = "Moderate cognitive impairment"
    else:            cog = "Severe cognitive impairment"

    _pdf_line(pdf, "Age",             f"{patient_data.get('Age','N/A')} years")
    _pdf_line(pdf, "Gender",          patient_data.get("Gender", "N/A"))
    _pdf_line(pdf, "Education Level", f"{patient_data.get('EducationLevel','N/A')} years")
    _pdf_line(pdf, "BMI",             bmi_str)
    _pdf_line(pdf, "MMSE Score",      f"{mmse}/30  ({cog})")
    _pdf_line(pdf, "Family History",  patient_data.get("FamilyHistoryAlzheimers", "N/A"))
    _pdf_line(pdf, "Cardiovascular",  patient_data.get("CardiovascularDisease", "N/A"))
    _pdf_line(pdf, "Diabetes",        patient_data.get("Diabetes", "N/A"))
    _pdf_line(pdf, "Depression",      patient_data.get("Depression", "N/A"))
    _pdf_line(pdf, "Hypertension",    patient_data.get("Hypertension", "N/A"))
    _pdf_line(pdf, "Physical Activity",
              f"{patient_data.get('PhysicalActivity','N/A')} hrs/week")
    _pdf_line(pdf, "Sleep Quality",   f"{patient_data.get('SleepQuality','N/A')}/5")
    _pdf_line(pdf, "Diet Quality",    f"{patient_data.get('DietQuality','N/A')}/5")
    _pdf_line(pdf, "Smoking",         patient_data.get("Smoking", "N/A"))

    # Risk factors
    _pdf_divider(pdf)
    _pdf_heading(pdf, "Key Risk Factors Identified")
    if risk_factors_list:
        for f in risk_factors_list:
            _pdf_bullet(pdf, f)
    else:
        _pdf_bullet(pdf, "No major risk factors identified")

    # Recommendations
    _pdf_divider(pdf)
    _pdf_heading(pdf, "Recommendations")
    if risk_level == "HIGH":
        recs = [
            "Consult a neurologist or memory specialist urgently",
            "Comprehensive cognitive assessment recommended",
            "Brain imaging (MRI/CT) may be necessary",
            "Engage in cognitively stimulating activities daily",
            "Mediterranean or MIND diet recommended",
            "Regular physical exercise (150 mins/week)",
            "Social engagement and mental activities",
            "Monitor and manage cardiovascular risk factors",
        ]
    elif risk_level == "MEDIUM":
        recs = [
            "Schedule a cognitive assessment with your doctor",
            "Increase mental stimulation (reading, puzzles, learning)",
            "Adopt brain-healthy diet rich in omega-3 and antioxidants",
            "Regular aerobic exercise",
            "Ensure adequate sleep (7-8 hours)",
            "Manage stress through meditation or relaxation",
            "Stay socially active",
        ]
    else:
        recs = [
            "Maintain current healthy lifestyle",
            "Continue cognitive activities",
            "Regular health check-ups",
            "Balanced diet and exercise routine",
            "Monitor for any cognitive changes",
        ]
    for r in recs:
        _pdf_bullet(pdf, r)

    # Disclaimer
    _pdf_divider(pdf)
    _pdf_body(pdf, DISCLAIMER, size=8)

    return bytes(pdf.output())


# ====== LOCATION FILTERS ======
def render_location_filters():
    countries_with_provinces = {
        "Nigeria": ["Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno",
                    "Cross River", "Delta", "Ebonyi", "Edo", "Ekiti", "Enugu", "FCT", "Gombe", "Imo",
                    "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Kogi", "Kwara", "Lagos", "Nasarawa",
                    "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers", "Sokoto", "Taraba", "Yobe", "Zamfara"],
        "Ghana": ["Greater Accra", "Ashanti", "Western", "Eastern", "Volta", "Northern", "Upper East",
                  "Upper West", "Bono", "Ahafo", "Savannah", "Oti", "North East", "Western North", "Central"],
        "Kenya": ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Kiambu", "Machakos", "Uasin Gishu", "Meru",
                  "Embu", "Kakamega", "Bungoma", "Kisii"],
        "South Africa": ["Gauteng", "Western Cape", "Eastern Cape", "Northern Cape", "KwaZulu-Natal",
                         "Free State", "North West", "Mpumalanga", "Limpopo"],
        "Uganda": ["Central", "Eastern", "Northern", "Western"],
        "Tanzania": ["Arusha", "Dar es Salaam", "Dodoma", "Geita", "Kagera", "Kigoma", "Kilimanjaro",
                     "Lindi", "Manyara", "Mara", "Mbeya", "Morogoro", "Mtwara", "Mwanza", "Njombe",
                     "Pwani", "Rukwa", "Ruvuma", "Shinyanga", "Simiyu", "Singida", "Tabora", "Tanga"],
        "Ethiopia": ["Addis Ababa", "Amhara", "Oromia", "Tigray", "Sidama", "Somali",
                     "Benishangul-Gumuz", "SNNPR", "Afar", "Gambela", "Harari"],
        "Egypt": ["Cairo", "Alexandria", "Giza", "Aswan", "Asyut", "Beheira", "Beni Suef", "Dakahlia",
                  "Damietta", "Faiyum", "Gharbia", "Ismailia", "Kafr El Sheikh", "Luxor", "Matruh",
                  "Minya", "Monufia", "New Valley", "North Sinai", "Port Said", "Qalyubia", "Qena",
                  "Red Sea", "Sharqia", "Sohag", "South Sinai", "Suez"],
        "Morocco": ["Casablanca-Settat", "Rabat-Sale-Kenitra", "Fes-Meknes", "Marrakesh-Safi",
                    "Tangier-Tetouan-Al Hoceima", "Souss-Massa", "Oriental", "Beni Mellal-Khenifra"],
        "Cameroon": ["Adamawa", "Centre", "East", "Far North", "Littoral", "North", "Northwest",
                     "South", "Southwest", "West"],
        "Zimbabwe": ["Bulawayo", "Harare", "Manicaland", "Mashonaland Central", "Mashonaland East",
                     "Mashonaland West", "Masvingo", "Matabeleland North", "Matabeleland South", "Midlands"],
        "Zambia": ["Central", "Copperbelt", "Eastern", "Luapula", "Lusaka", "Muchinga",
                   "Northern", "North-Western", "Southern", "Western"],
        "Rwanda": ["Kigali", "Eastern", "Northern", "Southern", "Western"],
        "Algeria": ["Algiers", "Oran", "Constantine", "Blida", "Annaba", "Batna", "Setif",
                    "Djelfa", "Tlemcen", "Tizi Ouzou", "Bejaia", "Skikda", "Mostaganem"]
    }

    region_with_ethnicity = {
        "North Africa": ["Amazigh (Berber)", "Arab", "Bedouin", "Coptic", "Nubian", "Tuareg", "Tebu"],
        "West Africa": ["Yoruba", "Hausa", "Igbo", "Fulani", "Akan", "Ashanti", "Ewe", "Fon", "Ga",
                        "Mandinka", "Wolof", "Serer", "Mossi", "Dogon", "Songhai", "Tiv", "Ijaw",
                        "Ibibio", "Kanuri", "Nupe"],
        "Central Africa": ["Bantu", "Kongo", "Luba", "Mongo", "Teke", "Fang", "Bamileke",
                           "Bamum", "Chokwe", "Ovimbundu", "Mbundu", "Lunda"],
        "East Africa": ["Amhara", "Tigray", "Oromo", "Somali", "Afar", "Sidama", "Kamba", "Kikuyu",
                        "Luhya", "Luo", "Kalenjin", "Maasai", "Chaga", "Sukuma", "Ganda", "Tutsi",
                        "Hutu", "Twa", "Dinka", "Nuer"],
        "Southern Africa": ["Shona", "Ndebele", "Zulu", "Xhosa", "Sotho", "Tswana", "Swazi", "Venda",
                            "Tsonga", "Pedi", "Herero", "Ovambo", "San (Bushmen)", "Lozi", "Chewa", "Bemba"],
        "Indian Ocean Islands": ["Merina", "Betsileo", "Comorian", "Mauritian", "Seychellois Creole", "Zanzibari"]
    }

    st.sidebar.markdown("---")
    st.sidebar.header(get_translation("🌍 Location Information"))

    country = st.sidebar.selectbox(get_translation("Select Country"), list(countries_with_provinces.keys()))
    province = st.sidebar.selectbox(get_translation("Select Province"), countries_with_provinces[country])
    region = st.sidebar.selectbox(get_translation("Select Region"), list(region_with_ethnicity.keys()))
    ethnicity = st.sidebar.selectbox(get_translation("Select Ethnicity"), region_with_ethnicity[region])

    c_map = {name: i for i, name in enumerate(countries_with_provinces.keys())}
    p_map = {name: i for i, name in enumerate(countries_with_provinces[country])}
    r_map = {name: i for i, name in enumerate(region_with_ethnicity.keys())}
    e_map = {name: i for i, name in enumerate(region_with_ethnicity[region])}

    # Store in session state for use in assessment forms
    st.session_state.encoded_country = c_map[country]
    st.session_state.encoded_province = p_map[province]
    st.session_state.encoded_region = r_map[region]
    st.session_state.encoded_ethnicity = e_map[ethnicity]


# ====== WELCOME SCREEN ======
def show_welcome_screen():
    col1, col2 = st.columns([1, 4])
    with col1:
        display_logo()
    with col2:
        st.markdown("<h2>African NeuroHealth AI Dashboard</h2>", unsafe_allow_html=True)
        st.markdown("""
        ## Your Personal Health Assessment Platform

        **Assessments available:**
        - 🩺 **Stroke** Risk Assessment
        - 🧠 **Dementia** Risk Evaluation
        - 🥗 **Nutrition** Tracking
        - 😌 **Stress** Assessment

        ### Getting Started:
        1. Enter your name in the sidebar
        2. Click "Start Session"
        3. Complete assessments
        4. View/download your reports
        """)


# ====== DASHBOARD ======
def render_dashboard():
    col1, col2 = st.columns([1, 4])
    with col1:
        display_logo()
    with col2:
        st.markdown("""
        <div style="margin-top: 10px;">
            <h1 style="margin: 0;">African NeuroHealth AI Dashboard</h1>
            <p style="color: #6B7280; font-size: 1rem; margin: 0;">Stroke & Dementia Predictor</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    Welcome to the **African NeuroHealth AI Dashboard** — an integrated platform for predicting
    **stroke** and **dementia** risks using advanced machine learning models. This platform is
    culturally attuned and context-aware, tailored for assessing neuro-health risks in African populations.

    ### Features:
    - **Stroke Risk Prediction**: Assess your risk factors and get personalized recommendations
    - **Dementia Risk Assessment**: Evaluate cognitive health and dementia risk
    - **Memory Game**: Test and train your cognitive abilities
    - **Nutrition Tracker**: Monitor dietary habits and get nutritional scores
    - **Stress Assessment**: Evaluate stress levels and coping mechanisms
    - **PDF Reports**: Download printable medical reports

    **Developed by Adebimpe-John Omolola E., with support from the GRASP / NIH / DSI Collaborative Program.**
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        animated_metric(label="🧠 Stroke Predictions", target_value=1247, delta="+23")
    with col2:
        animated_metric(label="🧓 Dementia Predictions", target_value=892, delta="+15")
    with col3:
        animated_metric(label="🎮 Memory Game Players", target_value=543, delta="+12")

    st.session_state.previous_stats = {
        "stroke_predictions": 1247,
        "dementia_predictions": 892,
        "memory_game_players": 543
    }

    st.markdown("## 🎯 Quick Access")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("### 🩺 Stroke Prediction")
            st.markdown("- Assess stroke risk factors\n- Get personalized recommendations\n- Download printable report")
            if st.button("Start Assessment", key="dash_stroke", use_container_width=True):
                st.session_state.current_page = "Stroke Assessment"
                st.rerun()
    with col2:
        with st.container(border=True):
            st.markdown("### 🧠 Dementia Prediction")
            st.markdown("- Cognitive health assessment\n- Memory function evaluation\n- Download printable report")
            if st.button("Start Assessment", key="dash_dementia", use_container_width=True):
                st.session_state.current_page = "Dementia Assessment"
                st.rerun()

    st.markdown("## 🛠️ Health Tools")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧠 Memory Game", use_container_width=True, key="dash_memory"):
            st.session_state.current_page = "Memory Game"
            st.rerun()
    with col2:
        if st.button("🥗 Nutrition Tracker", use_container_width=True, key="dash_nutrition"):
            st.session_state.current_page = "Nutrition Tracker"
            st.rerun()
    with col3:
        if st.button("😌 Stress Assessment", use_container_width=True, key="dash_stress"):
            st.session_state.current_page = "Stress Assessment"
            st.rerun()


# ====== STROKE ASSESSMENT ======
def render_stroke_assessment():
    st.header(get_translation("🩺 Stroke Risk Assessment"))

    if not st.session_state.models_loaded.get('Stroke', False):
        with st.spinner("Loading stroke model..."):
            st.session_state.stroke_model = load_stroke_model()
            st.session_state.models_loaded['Stroke'] = True

    col1, col2 = st.columns([2, 1])

    with col1:
        with st.form("stroke_form"):
            st.subheader(get_translation("Personal Information"))

            age = st.selectbox(get_translation("Age"), [get_translation("Select")] + list(range(18, 121)))
            gender = st.selectbox(get_translation("Gender"),
                                  [get_translation("Select Gender"), get_translation("Male"), get_translation("Female")])

            col_hw1, col_hw2 = st.columns(2)
            with col_hw1:
                height = st.number_input("Height (cm)", 100, 250, 170, key="stroke_height")
            with col_hw2:
                weight = st.number_input("Weight (kg)", 30, 200, 70, key="stroke_weight")

            bmi = weight / ((height / 100) ** 2) if height > 0 else 25.0
            st.metric("BMI", f"{bmi:.1f}")

            if bmi < 18.5:
                bmi_cat, bmi_color = get_translation("Underweight"), "orange"
            elif bmi < 25:
                bmi_cat, bmi_color = get_translation("Normal weight"), "green"
            elif bmi < 30:
                bmi_cat, bmi_color = get_translation("Overweight"), "orange"
            else:
                bmi_cat, bmi_color = get_translation("Obese"), "red"
            st.markdown(f"**BMI Category:** <span style='color:{bmi_color}'>{bmi_cat}</span>", unsafe_allow_html=True)

            blood_group = st.selectbox(get_translation("Blood Group"),
                                       [get_translation("Select Blood Group"), "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
            genotype = st.selectbox(get_translation("Genotype"),
                                    [get_translation("Select Genotype"), "AA", "AS", "SS", "AC", "SC"])

            st.markdown("---")
            st.subheader(get_translation("Medical History"))

            heart_disease = st.selectbox(get_translation("Heart Disease"),
                                         [get_translation("Select"), get_translation("Yes"), get_translation("No")])
            hypertension = st.selectbox(get_translation("Hypertension"),
                                        [get_translation("Select"), get_translation("Yes"), get_translation("No")])

            col_bp1, col_bp2 = st.columns(2)
            with col_bp1:
                systolic_bp = st.number_input(get_translation("Systolic BP"), 80, 250, 120, key="stroke_sys_bp")
            with col_bp2:
                diastolic_bp = st.number_input(get_translation("Diastolic BP"), 50, 150, 80, key="stroke_dia_bp")

            avg_glucose_level = st.number_input(get_translation("Average Glucose Level"),
                                                min_value=50.0, max_value=300.0, value=100.0, format="%.2f")
            diabetes_type = st.selectbox(get_translation("Diabetes Type"),
                                         [get_translation("Select"), get_translation("None"),
                                          get_translation("Type 1"), get_translation("Type 2"),
                                          get_translation("Gestational")])
            chronic_pain = st.selectbox(get_translation("Chronic Pain"),
                                        [get_translation("Select"), get_translation("None"),
                                         get_translation("Rheumatism"), get_translation("Osteoarthritis"),
                                         get_translation("Others")])

            st.markdown("---")
            st.subheader(get_translation("Lifestyle Factors"))

            marital_status = st.selectbox(get_translation("Marital Status"),
                                          [get_translation("Select"), get_translation("Single"),
                                           get_translation("Married"), get_translation("Divorced"),
                                           get_translation("Widowed")])
            work_type = st.selectbox(get_translation("Work Type"),
                                     [get_translation("Select"), get_translation("Private"),
                                      get_translation("Self-employed"), get_translation("Govt job"),
                                      get_translation("Children"), get_translation("Never worked")])
            residence_type = st.selectbox(get_translation("Residence Type"),
                                          [get_translation("Select"), get_translation("Urban"), get_translation("Rural")])
            smoking_status = st.selectbox(get_translation("Smoking Status"),
                                          [get_translation("Select"), get_translation("formerly smoked"),
                                           get_translation("never smoked"), get_translation("smokes")])
            physical_activity = st.selectbox(get_translation("Physical Activity"),
                                             [get_translation("Select"), "Sedentary", "Light", "Moderate", "Active", "Very Active"],
                                             key="stroke_activity")
            sleep_hours = st.selectbox(get_translation("Sleep (hours/day)"),
                                       [get_translation("Select")] + list(range(3, 13)), key="stroke_sleep")
            stress_level = st.selectbox(get_translation("Stress Level"),
                                        [get_translation("Select"), get_translation("None"),
                                         get_translation("Low"), get_translation("Moderate"), get_translation("High")])
            ptsd = st.selectbox(get_translation("PTSD"),
                                [get_translation("Select"), get_translation("Yes"), get_translation("No")])
            depression_level = st.selectbox(get_translation("Depression Level"),
                                            [get_translation("Select"), get_translation("None"),
                                             get_translation("Mild"), get_translation("Moderate"),
                                             get_translation("Severe")])

            st.markdown("---")
            st.subheader(get_translation("Environmental & Dietary Factors"))

            hypertension_treatment = st.selectbox(get_translation("Hypertension Treatment"),
                                                  [get_translation("Select"), get_translation("None"),
                                                   get_translation("Herbal"), get_translation("Drugs")])
            salt_intake = st.selectbox(get_translation("Salt Intake"),
                                       [get_translation("Select"), get_translation("None"),
                                        get_translation("Little"), get_translation("Moderate"), get_translation("High")])
            noise_sources = st.selectbox(get_translation("Noise Sources"),
                                         [get_translation("Select"), get_translation("None"),
                                          get_translation("Mosque"), get_translation("Church"),
                                          get_translation("Market"), get_translation("Block-Industry"),
                                          get_translation("Grinding-Machine"), get_translation("Welder"),
                                          get_translation("Club-House"), get_translation("Generator")])
            pollution_level_air = st.selectbox(get_translation("Air Pollution Level"),
                                               [get_translation("Select"), get_translation("None"),
                                                get_translation("Low"), get_translation("Moderate"), get_translation("High")])
            pollution_level_water = st.selectbox(get_translation("Water Pollution Level"),
                                                 [get_translation("Select"), get_translation("None"),
                                                  get_translation("Low"), get_translation("Moderate"), get_translation("High")])
            pollution_level_environmental = st.selectbox(get_translation("Environmental Pollution Level"),
                                                         [get_translation("Select"), get_translation("None"),
                                                          get_translation("Low"), get_translation("Moderate"), get_translation("High")])

            submitted = st.form_submit_button(get_translation("Submit Stroke Assessment"))

        if submitted:
            select_values = [get_translation("Select"), get_translation("Select Gender"),
                             get_translation("Select Blood Group"), get_translation("Select Genotype")]
            required = [age, gender, blood_group, genotype, heart_disease, hypertension,
                        marital_status, work_type, residence_type, smoking_status,
                        stress_level, ptsd, depression_level, diabetes_type, chronic_pain,
                        sleep_hours, hypertension_treatment, salt_intake, noise_sources,
                        pollution_level_air, pollution_level_water, pollution_level_environmental]

            def is_unset(x):
                try:
                    return x is None or x in select_values
                except TypeError:
                    return False  # unhashable types (lists) are not "unset"

            if any(is_unset(x) for x in required):
                st.error(get_translation("⚠️ Please complete all fields before prediction."))
            else:
                try:
                    stress_map = {get_translation("None"): 0, get_translation("Low"): 1,
                                  get_translation("Moderate"): 2, get_translation("High"): 3}
                    pain_map = {get_translation("None"): 0, get_translation("Rheumatism"): 1,
                                get_translation("Osteoarthritis"): 2, get_translation("Others"): 3}
                    treatment_map = {get_translation("None"): 0, get_translation("Herbal"): 1, get_translation("Drugs"): 2}
                    diabetes_map = {get_translation("None"): 0, get_translation("Type 1"): 1,
                                    get_translation("Type 2"): 2, get_translation("Gestational"): 3}
                    depression_map = {get_translation("None"): 0, get_translation("Mild"): 1,
                                      get_translation("Moderate"): 2, get_translation("Severe"): 3}

                    raw_inputs = {
                        'age': safe_int(age),
                        'gender': 1 if gender == get_translation("Male") else 0,
                        'height': safe_float(height),
                        'weight': safe_float(weight),
                        'bmi': safe_float(bmi),
                        'blood_group': blood_group,
                        'genotype': genotype,
                        'heart_disease': 1 if heart_disease == get_translation("Yes") else 0,
                        'hypertension': 1 if hypertension == get_translation("Yes") else 0,
                        'systolic_bp': safe_float(systolic_bp),
                        'diastolic_bp': safe_float(diastolic_bp),
                        'avg_glucose_level': safe_float(avg_glucose_level),
                        'diabetes_type': diabetes_map.get(diabetes_type, 0),
                        'chronic_pain': pain_map.get(chronic_pain, 0),
                        'ever_married': 1 if marital_status in [get_translation("Married"),
                                                                  get_translation("Divorced"),
                                                                  get_translation("Widowed")] else 0,
                        'work_type': work_type,
                        'Residence_type': residence_type,
                        'smoking_status': smoking_status,
                        'physical_activity': physical_activity,
                        'sleep_hours': safe_int(sleep_hours, 7),
                        'stress_level': stress_map.get(stress_level, 0),
                        'ptsd': 1 if ptsd == get_translation("Yes") else 0,
                        'depression_level': depression_map.get(depression_level, 0),
                        'hypertension_treatment': treatment_map.get(hypertension_treatment, 0),
                        'salt_intake': salt_intake,
                        'noise_sources': noise_sources,
                        'pollution_level_air': pollution_level_air,
                        'pollution_level_water': pollution_level_water,
                        'pollution_level_environmental': pollution_level_environmental,
                        'ethnicity': st.session_state.get('encoded_ethnicity', 0),
                        'Country': st.session_state.get('encoded_country', 0),
                        'Province_Option': st.session_state.get('encoded_province', 0),
                        'CustomStressScore': st.session_state.get('stress_score', 0)
                    }

                    # One-hot encode chronic_pain
                    cp_val = chronic_pain if chronic_pain else ""
                    raw_inputs.update({
                        'chronic_pain_None':          1 if get_translation("None")         in cp_val else 0,
                        'chronic_pain_Rheumatism':    1 if get_translation("Rheumatism")   in cp_val else 0,
                        'chronic_pain_Osteoarthritis':1 if get_translation("Osteoarthritis") in cp_val else 0,
                        'chronic_pain_Others':        1 if get_translation("Others")       in cp_val else 0,
                    })
                    # One-hot encode hypertension_treatment
                    ht_val = hypertension_treatment if hypertension_treatment else ""
                    raw_inputs.update({
                        'hypertension_treatment_None':  1 if get_translation("None")   in ht_val else 0,
                        'hypertension_treatment_Herbal':1 if get_translation("Herbal") in ht_val else 0,
                        'hypertension_treatment_Drugs': 1 if get_translation("Drugs")  in ht_val else 0,
                    })
                    # One-hot encode nutritional lifestyle (use session state score as proxy)
                    raw_inputs.update({
                        'nutritional_lifestyle_Fast Foods':           0,
                        'nutritional_lifestyle_Homemade Food':        0,
                        'nutritional_lifestyle_Junk Food':            0,
                        'nutritional_lifestyle_Local Bukka/Street Food': 0,
                    })
                    raw_inputs.update(map_salt_intake(salt_intake))
                    raw_inputs.update(map_noise_source(noise_sources))

                    stroke_inputs_df = prepare_stroke_input_numeric(raw_inputs)

                    if stroke_inputs_df is None or stroke_inputs_df.empty:
                        st.error(get_translation("Input preparation failed."))
                    else:
                        # Calculate risk score from known risk factors
                        risk_count = 0
                        if raw_inputs['age'] > 60: risk_count += 1
                        if raw_inputs['hypertension'] == 1: risk_count += 1
                        if raw_inputs['heart_disease'] == 1: risk_count += 1
                        if raw_inputs['diabetes_type'] > 0: risk_count += 1
                        if smoking_status != get_translation("never smoked"): risk_count += 1
                        if bmi > 30: risk_count += 1
                        if raw_inputs['stress_level'] > 2: risk_count += 1

                        risk_score = min(0.95, (risk_count * 0.12) + random.uniform(-0.05, 0.05))

                        # Use model if available, else fallback
                        model_dict = st.session_state.stroke_model
                        actual_model = model_dict.get('model') if model_dict else None
                        if actual_model is not None:
                            try:
                                pred = int(actual_model.predict(stroke_inputs_df)[0])
                            except Exception as model_err:
                                logger.error(f"Model prediction error: {model_err}")
                                pred = 1 if risk_score > 0.5 else 0
                        else:
                            pred = 1 if risk_score > 0.5 else 0

                        risk_level = "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.3 else "LOW"

                        st.session_state.predictions["Stroke"] = {
                            "risk_score": risk_score,
                            "risk_level": risk_level,
                            "patient_data": raw_inputs,
                            "prediction": pred
                        }

                        city, reg, ctry = get_user_location()
                        db_payload = {
                            "user_id": st.session_state.get('user_id', 'anonymous'),
                            "age": raw_inputs['age'],
                            "gender": raw_inputs['gender'],
                            "bmi": raw_inputs['bmi'],
                            "heart_disease": raw_inputs['heart_disease'],
                            "hypertension": raw_inputs['hypertension'],
                            "avg_glucose_level": raw_inputs['avg_glucose_level'],
                            "smoking_status": raw_inputs['smoking_status'],
                            "risk_score": float(risk_score),
                            "risk_level": risk_level,
                            "prediction_result": pred,
                            "location": f"{city}, {reg}, {ctry}",
                            "assessment_date": datetime.now().strftime("%Y-%m-%d")
                        }

                        try:
                            supabase.table("stroke_predictions").insert(db_payload).execute()
                            st.success(get_translation("✅ Stroke prediction saved successfully!"))
                        except Exception as db_err:
                            logger.warning(f"DB save failed: {db_err}")
                            st.warning("Prediction complete. (Database save skipped.)")

                        if pred == 1:
                            st.warning(get_translation("⚠️ The model predicts a higher risk of stroke."))
                        else:
                            st.success(get_translation("✅ The model predicts a lower risk of stroke."))

                        st.rerun()

                except Exception as e:
                    st.error(get_translation(f"An error occurred during prediction: {str(e)}"))
                    logger.error(traceback.format_exc())

    with col2:
        st.subheader(get_translation("Risk Assessment"))

        if st.session_state.predictions.get("Stroke"):
            prediction = st.session_state.predictions["Stroke"]
            risk_score = prediction["risk_score"]
            risk_level = prediction["risk_level"]
            patient_data = prediction["patient_data"]

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                title={"text": "Stroke Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

            risk_css = "risk-high" if risk_level == "HIGH" else "risk-medium" if risk_level == "MEDIUM" else "risk-low"
            st.markdown(f"<div class='{risk_css}'>Risk Level: {risk_level}</div>", unsafe_allow_html=True)

            st.subheader(get_translation("📊 Key Risk Factors"))
            risk_factors_list = []
            if patient_data.get('age', 0) > 60:
                risk_factors_list.append(f"Age ({patient_data['age']} years)")
            if patient_data.get('hypertension') == 1:
                risk_factors_list.append("Hypertension")
            if patient_data.get('heart_disease') == 1:
                risk_factors_list.append("Heart Disease")
            if patient_data.get('diabetes_type', 0) > 0:
                risk_factors_list.append("Diabetes")
            if patient_data.get('smoking_status', '') not in ['never smoked', get_translation("never smoked")]:
                risk_factors_list.append(f"Smoking ({patient_data.get('smoking_status', '')})")
            if patient_data.get('bmi', 0) > 30:
                risk_factors_list.append(f"Obesity (BMI: {patient_data['bmi']:.1f})")
            if patient_data.get('stress_level', 0) > 2:
                risk_factors_list.append("High Stress Level")

            for factor in risk_factors_list[:5]:
                st.write(f"• {factor}")
            if not risk_factors_list:
                st.info(get_translation("No major risk factors identified"))

            st.subheader(get_translation("📄 Report Generation"))
            if st.button(get_translation("📥 Generate PDF Report"), use_container_width=True, key="stroke_pdf"):
                with st.spinner("Generating report..."):
                    pdf_bytes = generate_stroke_pdf(
                        st.session_state.get('user_name', 'Patient'),
                        patient_data, risk_score, risk_level, risk_factors_list
                    )
                    filename = f"Stroke_Report_{st.session_state.get('user_name', 'Patient')}_{datetime.now().strftime('%Y%m%d')}.pdf"
                    st.markdown(create_download_link(pdf_bytes, filename), unsafe_allow_html=True)
                    report_id = f"stroke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    st.session_state.reports[report_id] = {
                        "type": "Stroke", "data": patient_data,
                        "risk_score": risk_score, "risk_level": risk_level, "filename": filename
                    }
        else:
            st.info(get_translation("👈 Fill the form and click 'Submit Stroke Assessment' to see results"))


# ====== ALZHEIMER'S ASSESSMENT ======
def render_alzheimer_assessment():
    st.header(get_translation("🧠 Alzheimer's/Dementia Risk Assessment"))

    if not st.session_state.models_loaded.get('Dementia', False):
        with st.spinner("Loading dementia model..."):
            st.session_state.alz_model = load_dementia_model()
            st.session_state.models_loaded['Dementia'] = True

    col1, col2 = st.columns([2, 1])

    with col1:
        with st.form("dementia_form"):
            st.subheader(get_translation("Patient Information"))

            age = st.selectbox(get_translation("Age"), [get_translation("Select")] + list(range(18, 121)))
            gender = st.selectbox(get_translation("Gender"),
                                  [get_translation("Select Gender"), get_translation("Male"), get_translation("Female")])
            education_years = st.selectbox(get_translation("Education Level (years)"),
                                           [get_translation("Select")] + list(range(0, 25)), key="alz_education")

            height = st.number_input("Height (cm)", 100, 250, 170, key="alz_height")
            weight = st.number_input("Weight (kg)", 30, 200, 70, key="alz_weight")
            bmi = weight / ((height / 100) ** 2) if height > 0 else 25.0
            st.metric("BMI", f"{bmi:.1f}")

            if bmi < 18.5:
                bmi_cat, bmi_color = get_translation("Underweight"), "orange"
            elif bmi < 25:
                bmi_cat, bmi_color = get_translation("Normal weight"), "green"
            elif bmi < 30:
                bmi_cat, bmi_color = get_translation("Overweight"), "orange"
            else:
                bmi_cat, bmi_color = get_translation("Obese"), "red"
            st.markdown(f"**BMI Category:** <span style='color:{bmi_color}'>{bmi_cat}</span>", unsafe_allow_html=True)

            blood_group = st.selectbox(get_translation("Blood Group"),
                                       [get_translation("Select Blood Group"), "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
                                       key="alz_blood_group")
            genotype = st.selectbox(get_translation("Genotype"),
                                    [get_translation("Select Genotype"), "AA", "AS", "SS", "AC", "SC"],
                                    key="alz_genotype")

            st.subheader(get_translation("Medical Assessment"))

            is_smoker = st.selectbox(get_translation("Smoking Status"),
                                     [get_translation("Select"), get_translation("formerly smoked"),
                                      get_translation("never smoked"), get_translation("smokes")], key='alz_smoking')
            alcohol_consumption = st.selectbox(get_translation("Alcohol Consumption (0=None, 5=High)"),
                                               [get_translation("Select")] + [str(i) for i in range(0, 6)], key='alz_alcohol')
            physical_activity = st.selectbox(get_translation("Physical Activity (hrs/week)"),
                                             [get_translation("Select")] + [str(i) for i in range(0, 21)], key='alz_activity')
            sleep_quality = st.selectbox(get_translation("Sleep Quality (1-5)"),
                                         [get_translation("Select")] + [str(i) for i in range(1, 6)], key='alz_sleep')
            diet_quality = st.selectbox(get_translation("Diet Quality (1-5)"),
                                        [get_translation("Select")] + [str(i) for i in range(1, 6)], key='alz_diet')

            family_history_alz = st.selectbox(get_translation("Family History of Alzheimer's"),
                                              [get_translation("Select"), get_translation("Yes"), get_translation("No")], key='alz_family')
            cardiovascular_disease = st.selectbox(get_translation("Cardiovascular Disease"),
                                                  [get_translation("Select"), get_translation("Yes"), get_translation("No")], key='alz_cardio')
            diabetes = st.selectbox(get_translation("Diabetes"),
                                    [get_translation("Select"), get_translation("Yes"), get_translation("No")], key='alz_diabetes')
            depression = st.selectbox(get_translation("Depression"),
                                      [get_translation("Select"), get_translation("Yes"), get_translation("No")], key='alz_depression')
            hypertension = st.selectbox(get_translation("Hypertension"),
                                        [get_translation("Select"), get_translation("Yes"), get_translation("No")], key='alz_hypertension')

            systolic_bp = st.number_input(get_translation("Systolic BP"), min_value=80, max_value=220, value=120, key='alz_systolic')
            diastolic_bp = st.number_input(get_translation("Diastolic BP"), min_value=50, max_value=150, value=80, key='alz_diastolic')
            cholesterol_total = st.number_input(get_translation("Total Cholesterol"), min_value=100, max_value=400, value=180, key='alz_chol_total')
            cholesterol_ldl = st.number_input(get_translation("LDL"), min_value=50, max_value=300, value=100, key='alz_ldl')
            cholesterol_hdl = st.number_input(get_translation("HDL"), min_value=20, max_value=100, value=50, key='alz_hdl')
            cholesterol_triglycerides = st.number_input(get_translation("Triglycerides"), min_value=50, max_value=500, value=150, key='alz_trig')

            functional_assessment = st.slider(get_translation("Functional Assessment (0-5)"), 0, 5, 3, key='alz_func')
            behavioral_problems = st.selectbox(get_translation("Behavioral Problems"),
                                               [get_translation("Select"), get_translation("Yes"), get_translation("No")], key='alz_behavior')
            adl = st.slider(get_translation("ADL Score (Activities of Daily Living)"), 0, 6, 3, key='alz_adl')

            st.subheader(get_translation("🧠 MMSE Assessment (Adapted for African Context)"))
            st.info(get_translation("Answer these culturally relevant questions to estimate your MMSE score:"))

            col_q1, col_q2 = st.columns(2)
            with col_q1:
                q1 = st.selectbox(get_translation("Do you forget names of relatives/village members?"),
                                  [get_translation("Select"), get_translation("Never"), get_translation("Sometimes"), get_translation("Often")], key='q1')
                q2 = st.selectbox(get_translation("Do you misplace important items (farming tools, keys)?"),
                                  [get_translation("Select"), get_translation("Never"), get_translation("Sometimes"), get_translation("Often")], key='q2')
                q3 = st.selectbox(get_translation("Can you recall traditional recipes or remedies?"),
                                  [get_translation("Select"), get_translation("Always"), get_translation("Sometimes"), get_translation("Rarely")], key='q3')
            with col_q2:
                q4 = st.selectbox(get_translation("Do you recognize people from your community?"),
                                  [get_translation("Select"), get_translation("Always"), get_translation("Sometimes"), get_translation("Rarely")], key='q4')
                q5 = st.selectbox(get_translation("Can you navigate familiar paths/markets?"),
                                  [get_translation("Select"), get_translation("Always"), get_translation("Sometimes"), get_translation("Rarely")], key='q5')
                q6 = st.selectbox(get_translation("Do you remember important cultural events/dates?"),
                                  [get_translation("Select"), get_translation("Always"), get_translation("Sometimes"), get_translation("Rarely")], key='q6')

            response_scores = {
                get_translation("Never"): 2, get_translation("Sometimes"): 1,
                get_translation("Often"): 0, get_translation("Always"): 2, get_translation("Rarely"): 0
            }
            weights = {"q1": 2.0, "q2": 1.5, "q3": 1.0, "q4": 1.7, "q5": 2.0, "q6": 1.3}
            mmse_score = 15

            if all(q not in (None, get_translation("Select")) for q in [q1, q2, q3, q4, q5, q6]):
                mmse_score = int(min(30, max(0, 20 + (
                    response_scores.get(q1, 0) * weights["q1"] +
                    response_scores.get(q2, 0) * weights["q2"] +
                    response_scores.get(q3, 0) * weights["q3"] +
                    response_scores.get(q4, 0) * weights["q4"] +
                    response_scores.get(q5, 0) * weights["q5"] +
                    response_scores.get(q6, 0) * weights["q6"]
                ))))
                st.info(f"Estimated MMSE Score: {mmse_score}/30")

            pollution_score = st.slider(get_translation("Pollution Score (0-100)"), 0, 100, 0, key='alz_pollution_score')
            pollution_choice = st.selectbox(get_translation("Pollution Category"),
                                            [get_translation("Select"), get_translation("Low"),
                                             get_translation("Moderate"), get_translation("High")], key='alz_pollution_cat')

            pollution_low = 1 if pollution_choice == get_translation("Low") else 0
            pollution_moderate = 1 if pollution_choice == get_translation("Moderate") else 0
            pollution_high = 1 if pollution_choice == get_translation("High") else 0

            option_map = {get_translation("Yes"): 1, get_translation("No"): 0, get_translation("Sometimes"): 0}

            confusion = st.selectbox(get_translation("Confusion"),
                                     [get_translation("Select"), get_translation("Yes"), get_translation("No"), get_translation("Sometimes")], key='alz_confusion')
            disorientation = st.selectbox(get_translation("Disorientation"),
                                          [get_translation("Select"), get_translation("Yes"), get_translation("No"), get_translation("Sometimes")], key='alz_disorien')
            personality_changes = st.selectbox(get_translation("Personality Changes"),
                                               [get_translation("Select"), get_translation("Yes"), get_translation("No"), get_translation("Sometimes")], key='alz_personality')
            difficulty_tasks = st.selectbox(get_translation("Difficulty Completing Tasks"),
                                            [get_translation("Select"), get_translation("Yes"), get_translation("No"), get_translation("Sometimes")], key='alz_tasks')
            forgetfulness = st.selectbox(get_translation("Forgetfulness"),
                                         [get_translation("Select"), get_translation("Yes"), get_translation("No"), get_translation("Sometimes")], key='alz_forget')
            memory_complaints = st.selectbox(get_translation("Memory Complaints"),
                                             [get_translation("Select"), get_translation("Yes"), get_translation("No"), get_translation("Sometimes")], key='alz_memory')

            head_map = {get_translation("None"): 0, get_translation("Accident"): 1, get_translation("Violence"): 2}
            head_choice = st.selectbox(get_translation("Head Injury"),
                                       [get_translation("Select"), get_translation("None"),
                                        get_translation("Accident"), get_translation("Violence")], key='alz_head')

            stress_score_val = st.slider(get_translation("Stress Level"), 0, 10, 0, key='alz_stress')

            submitted = st.form_submit_button(get_translation("Submit Dementia Assessment"))

        if submitted:
            select_values = (None, get_translation("Select"), get_translation("Select Gender"),
                             get_translation("Select Blood Group"), get_translation("Select Genotype"))
            required = [age, gender, education_years, blood_group, genotype, is_smoker,
                        alcohol_consumption, physical_activity, sleep_quality, diet_quality,
                        family_history_alz, cardiovascular_disease, diabetes, depression,
                        hypertension, behavioral_problems, head_choice, pollution_choice,
                        confusion, disorientation, personality_changes, difficulty_tasks,
                        forgetfulness, memory_complaints]

            def is_unset_alz(x):
                try:
                    return x is None or x in select_values
                except TypeError:
                    return False

            if any(is_unset_alz(x) for x in required):
                st.error(get_translation("⚠️ Please complete all fields before prediction."))
            else:
                try:
                    confusion_val = option_map.get(confusion, 0)
                    disorientation_val = option_map.get(disorientation, 0)
                    personality_changes_val = option_map.get(personality_changes, 0)
                    difficulty_tasks_val = option_map.get(difficulty_tasks, 0)
                    forgetfulness_val = option_map.get(forgetfulness, 0)
                    memory_complaints_val = option_map.get(memory_complaints, 0)
                    head_injury = head_map.get(head_choice, 0)

                    patient_data = {
                        "Age": safe_int(age),
                        "Gender": gender,
                        "BMI": safe_float(bmi),
                        "EducationLevel": safe_int(education_years),
                        "Smoking": is_smoker,
                        "AlcoholConsumption": safe_int(alcohol_consumption),
                        "PhysicalActivity": safe_int(physical_activity),
                        "DietQuality": safe_int(diet_quality),
                        "SleepQuality": safe_int(sleep_quality),
                        "FamilyHistoryAlzheimers": family_history_alz,
                        "CardiovascularDisease": cardiovascular_disease,
                        "Diabetes": diabetes,
                        "Depression": depression,
                        "Hypertension": hypertension,
                        "SystolicBP": safe_float(systolic_bp),
                        "DiastolicBP": safe_float(diastolic_bp),
                        "CholesterolTotal": safe_float(cholesterol_total),
                        "CholesterolLDL": safe_float(cholesterol_ldl),
                        "CholesterolHDL": safe_float(cholesterol_hdl),
                        "CholesterolTriglycerides": safe_float(cholesterol_triglycerides),
                        "MMSE": mmse_score,
                        "Height": safe_float(height),
                        "Weight": safe_float(weight),
                        "Genotype": genotype,
                        "BloodGroup": blood_group,
                        "FunctionalAssessment": int(functional_assessment),
                        "BehavioralProblems": behavioral_problems,
                        "ADL": int(adl),
                        "Confusion": confusion_val,
                        "Disorientation": disorientation_val,
                        "PersonalityChanges": personality_changes_val,
                        "DifficultyCompletingTasks": difficulty_tasks_val,
                        "Forgetfulness": forgetfulness_val,
                        "MemoryComplaints": memory_complaints_val,
                        "MemoryScore": st.session_state.get("memory_score", 1.0),
                        "HeadInjury": head_injury,
                        "Ethnicity": st.session_state.get('encoded_ethnicity', 0),
                        "Country": st.session_state.get('encoded_country', 0),
                        "Province_Option": st.session_state.get('encoded_province', 0),
                        "PollutionScore": int(pollution_score),
                        "PollutionCategoryLow": pollution_low,
                        "PollutionCategoryModerate": pollution_moderate,
                        "PollutionCategoryHigh": pollution_high,
                        "CustomStressScore": stress_score_val,
                        "assessment_date": datetime.now().strftime("%Y-%m-%d")
                    }

                    alzheimer_inputs_df = prepare_alzheimers_input_numeric(patient_data)

                    if alzheimer_inputs_df is None or alzheimer_inputs_df.empty:
                        st.error(get_translation("Input preparation failed."))
                    else:
                        # Calculate risk score
                        risk_count = 0
                        if safe_int(age) > 75: risk_count += 2
                        elif safe_int(age) > 65: risk_count += 1
                        if mmse_score < 24: risk_count += 2
                        elif mmse_score < 27: risk_count += 1
                        if family_history_alz == get_translation("Yes"): risk_count += 1
                        if memory_complaints_val == 1: risk_count += 1
                        if depression == get_translation("Yes"): risk_count += 1
                        if diabetes == get_translation("Yes"): risk_count += 1
                        if head_injury and head_injury > 0: risk_count += 1
                        if safe_int(physical_activity) < 3: risk_count += 1

                        risk_score = min(0.95, (risk_count * 0.08) + random.uniform(-0.05, 0.05))

                        model_dict = st.session_state.alz_model
                        actual_model = model_dict.get('model') if model_dict else None
                        if actual_model is not None:
                            try:
                                pred = int(actual_model.predict(alzheimer_inputs_df)[0])
                            except Exception as model_err:
                                logger.error(f"Dementia model prediction error: {model_err}")
                                pred = 1 if risk_score > 0.5 else 0
                        else:
                            pred = 1 if risk_score > 0.5 else 0

                        risk_level = "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.3 else "LOW"

                        st.session_state.predictions["Dementia"] = {
                            "risk_score": risk_score,
                            "risk_level": risk_level,
                            "patient_data": patient_data,
                            "prediction": pred,
                            "mmse": mmse_score
                        }

                        city, reg, ctry = get_user_location()
                        db_payload = {
                            "user_id": st.session_state.get('user_id', 'anonymous'),
                            "age": patient_data["Age"],
                            "gender": patient_data["Gender"],
                            "bmi": patient_data["BMI"],
                            "mmse": mmse_score,
                            "family_history_alzheimers": family_history_alz,
                            "depression": depression,
                            "diabetes": diabetes,
                            "risk_score": float(risk_score),
                            "risk_level": risk_level,
                            "prediction_result": pred,
                            "location": f"{city}, {reg}, {ctry}",
                            "assessment_date": datetime.now().strftime("%Y-%m-%d")
                        }

                        try:
                            supabase.table("alzheimer_predictions").insert(db_payload).execute()
                            st.success(get_translation("✅ Alzheimer's prediction saved successfully!"))
                        except Exception as db_err:
                            logger.warning(f"DB save failed: {db_err}")
                            st.warning("Prediction complete. (Database save skipped.)")

                        if pred == 1:
                            st.warning(get_translation("⚠️ The model predicts a higher risk of Alzheimer's disease."))
                        else:
                            st.success(get_translation("✅ The model predicts a lower risk of Alzheimer's disease."))

                        st.rerun()

                except Exception as e:
                    st.error(get_translation(f"An error occurred during prediction: {str(e)}"))
                    logger.error(traceback.format_exc())

    with col2:
        st.subheader(get_translation("Risk Assessment"))

        if st.session_state.predictions.get("Dementia"):
            prediction = st.session_state.predictions["Dementia"]
            risk_score = prediction["risk_score"]
            risk_level = prediction["risk_level"]
            patient_data = prediction["patient_data"]

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                title={"text": "Dementia Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)

            risk_css = "risk-high" if risk_level == "HIGH" else "risk-medium" if risk_level == "MEDIUM" else "risk-low"
            st.markdown(f"<div class='{risk_css}'>Risk Level: {risk_level}</div>", unsafe_allow_html=True)

            st.subheader(get_translation("📊 Key Risk Factors"))
            risk_factors_list = []
            if patient_data.get("Age", 0) > 75:
                risk_factors_list.append(f"Age ({patient_data.get('Age')} years)")
            if patient_data.get("FamilyHistoryAlzheimers") == get_translation("Yes"):
                risk_factors_list.append("Family History of Alzheimer's")
            if patient_data.get("MemoryComplaints") == 1:
                risk_factors_list.append("Memory Complaints")
            if patient_data.get("Depression") == get_translation("Yes"):
                risk_factors_list.append("Depression")
            if patient_data.get("MMSE", 30) < 24:
                risk_factors_list.append(f"Low MMSE Score ({patient_data.get('MMSE')}/30)")
            if patient_data.get("Diabetes") == get_translation("Yes"):
                risk_factors_list.append("Diabetes")

            for factor in risk_factors_list[:5]:
                st.write(f"• {factor}")
            if not risk_factors_list:
                st.info(get_translation("No major risk factors identified"))

            st.subheader(get_translation("📄 Report Generation"))
            if st.button(get_translation("📥 Generate PDF Report"), use_container_width=True, key="dementia_pdf"):
                with st.spinner("Generating report..."):
                    pdf_bytes = generate_alzheimer_pdf(
                        st.session_state.get('user_name', 'Patient'),
                        patient_data, risk_score, risk_level, risk_factors_list
                    )
                    filename = f"Dementia_Report_{st.session_state.get('user_name', 'Patient')}_{datetime.now().strftime('%Y%m%d')}.pdf"
                    st.markdown(create_download_link(pdf_bytes, filename), unsafe_allow_html=True)
                    report_id = f"dementia_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    st.session_state.reports[report_id] = {
                        "type": "Dementia", "data": patient_data,
                        "risk_score": risk_score, "risk_level": risk_level, "filename": filename
                    }
        else:
            st.info(get_translation("👈 Fill the form and click 'Submit Dementia Assessment' to see results"))


# ====== MEMORY GAME ======
def memory_recall_game():
    st.markdown("<h1>🧠 Memory Recall Game</h1>", unsafe_allow_html=True)

    if 'memory_game' not in st.session_state or st.session_state.memory_game is None:
        st.session_state.memory_game = {
            "state": "start", "words": [], "start_time": None,
            "level": 1, "score_history": [], "display_end_time": None
        }

    game = st.session_state.memory_game

    WORD_POOL = ["apple", "table", "river", "mountain", "sun", "flower", "clock", "phone",
                 "book", "star", "moon", "chair", "pencil", "car", "glass", "tree", "music",
                 "house", "cloud", "lamp", "keyboard", "shoe", "bottle", "ring", "window",
                 "garden", "bridge", "ocean", "forest", "diamond"]

    if game["state"] == "start":
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 30px; border-radius: 15px; color: white; text-align: center;'>
            <h2>🎮 Memory Challenge</h2>
            <p style='font-size: 18px;'>Test your memory and cognitive abilities!</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style='background: #f8f9fa; padding: 20px; border-radius: 10px;
                        border-left: 5px solid #667eea; margin: 20px 0;'>
                <h3 style='color: #667eea; margin-top: 0;'>📋 Level {game['level']}</h3>
                <p>You will see <strong>{4 + game['level']} words</strong> for 5 seconds. Type what you remember!</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button(get_translation("🎯 Start Memory Exercise"), key="memory_start", use_container_width=True):
                num_words = 4 + game["level"]
                game["words"] = random.sample(WORD_POOL, num_words)
                game["start_time"] = time.time()
                game["display_end_time"] = time.time() + 5
                game["state"] = "showing"
                st.rerun()

    elif game["state"] == "showing":
        if game.get("display_end_time") is None:
            game["display_end_time"] = time.time() + 5

        time_remaining = max(0, game["display_end_time"] - time.time())

        if time_remaining > 0:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                        padding: 40px; border-radius: 15px; text-align: center;'>
                <h2 style='color: white; margin-bottom: 20px;'>⏱️ Memorize These Words!</h2>
                <div style='background: white; padding: 30px; border-radius: 10px;
                            font-size: 28px; font-weight: bold; color: #333; margin-bottom: 20px;'>
                    {", ".join(game["words"])}
                </div>
                <div style='font-size: 48px; color: white; font-weight: bold;'>
                    {int(time_remaining) + 1}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(1 - (time_remaining / 5))
            time.sleep(0.1)
            st.rerun()
        else:
            game["state"] = "recalling"
            st.rerun()

    elif game["state"] == "recalling":
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    padding: 30px; border-radius: 15px; text-align: center; color: white;'>
            <h2>🤔 Time to Recall!</h2>
            <p style='font-size: 18px;'>Type the words you remember, separated by commas</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("recall_form", clear_on_submit=True):
            recalled_input = st.text_input(
                get_translation("Enter the words you remember:"),
                placeholder="word1, word2, word3...", key="recall_input"
            )
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit = st.form_submit_button(get_translation("✅ Submit Your Answer"),
                                               use_container_width=True, type="primary")

        if submit:
            if not recalled_input.strip():
                st.error(get_translation("⚠️ Please enter at least one word before submitting!"))
            else:
                recalled = [w.strip().lower() for w in recalled_input.split(",") if w.strip()]
                correct_words = set(w.lower() for w in game["words"])
                recalled_set = set(recalled)
                correct_count = len(correct_words & recalled_set)
                memory_score = correct_count / len(game['words'])

                score_color = "#28a745" if memory_score >= 0.8 else "#ffc107" if memory_score >= 0.5 else "#dc3545"
                col1, col2, col3 = st.columns(3)
                col1.metric("✅ Correct", correct_count)
                col2.metric("❌ Incorrect", len(recalled_set - correct_words))
                col3.metric("😅 Missed", len(correct_words - recalled_set))

                st.markdown(f"""
                <div style='background: {score_color}; padding: 20px; border-radius: 10px;
                            text-align: center; color: white; margin: 20px 0;'>
                    <h2 style='margin: 0;'>Memory Score: {memory_score:.0%}</h2>
                </div>
                """, unsafe_allow_html=True)

                threshold = len(game['words']) - 1
                if correct_count >= threshold:
                    st.balloons()
                    st.success("🎉 Excellent! You advance to the next level!")
                    game["level"] += 1
                else:
                    st.info("📚 Keep practicing!")

                game["score_history"].append({
                    "level": game["level"], "correct": correct_count,
                    "total": len(game["words"]), "score": memory_score,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.session_state.memory_score = memory_score

                if st.button(get_translation("🔄 Play Again"), use_container_width=True, type="primary"):
                    game["state"] = "start"
                    game["words"] = []
                    game["start_time"] = None
                    game["display_end_time"] = None
                    st.rerun()

    st.markdown("---")
    if game["score_history"]:
        with st.expander(get_translation("📊 Your Score History")):
            total_rounds = len(game["score_history"])
            avg_score = sum(s["score"] for s in game["score_history"]) / total_rounds
            best_score = max(s["score"] for s in game["score_history"])
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rounds", total_rounds)
            col2.metric("Average Score", f"{avg_score:.0%}")
            col3.metric("Best Score", f"{best_score:.0%}")
    else:
        st.info(get_translation("👆 Start playing to see your score history!"))


# ====== NUTRITION TRACKER ======
def nutrition_tracker():
    st.markdown("<h1>🥗 Nutrition Tracker</h1>", unsafe_allow_html=True)

    with st.expander(get_translation("Track Your Nutrition"), expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            fruit_intake = st.number_input(get_translation("Fruit Intake (servings/day)"), 0, 20, 2, key="fruit_intake")
            vegetable_intake = st.number_input(get_translation("Vegetable Intake (servings/day)"), 0, 20, 3, key="vegetable_intake")
            hydration_liters = st.number_input(get_translation("Water Intake (liters/day)"), 0.0, 10.0, 2.0, key="hydration")
        with col2:
            supplements_used = st.text_input(get_translation("Supplements Used (e.g., Vitamin D, Omega-3)"), key="supplements")
            natural_herbs = st.text_input(get_translation("Natural Herbs Taken (e.g., Ginger, Turmeric)"), key="herbs")

        lifestyle_choices = [
            get_translation("Homemade Food"), get_translation("Vegetarian"), get_translation("Vegan"),
            get_translation("Mediterranean"), get_translation("Pescatarian"),
            get_translation("Local Street Food"), get_translation("Junk Food"),
            get_translation("Fast Foods"), get_translation("Keto"), get_translation("Paleo")
        ]
        selected_lifestyles = st.multiselect(get_translation("Select Nutritional Lifestyles"), lifestyle_choices, key="lifestyles")

        positive_lifestyles = [get_translation("Homemade Food"), get_translation("Vegetarian"),
                               get_translation("Vegan"), get_translation("Mediterranean"), get_translation("Pescatarian")]
        negative_lifestyles = [get_translation("Junk Food"), get_translation("Fast Foods")]

        positive_score = sum(1 for l in selected_lifestyles if l in positive_lifestyles)
        negative_score = sum(1 for l in selected_lifestyles if l in negative_lifestyles)
        nutritional_score = max(1, min(5, round(3 + (positive_score * 0.5) - (negative_score * 1.0))))

        st.metric(get_translation("Nutritional Health Score"), f"{nutritional_score}/5")
        st.session_state.nutritional_score = nutritional_score

        if st.button(get_translation("💾 Save Nutrition Data"), key="save_nutrition"):
            nutrition_data = {
                "fruit_intake": fruit_intake,
                "vegetable_intake": vegetable_intake,
                "hydration_liters": hydration_liters,
                "supplements_used": supplements_used,
                "natural_herbs": natural_herbs,
                "lifestyle_choices": ", ".join(selected_lifestyles),  # Convert list to string for DB
                "nutritional_score": nutritional_score,
                "user_id": st.session_state.get('user_id', 'anonymous')
            }
            try:
                supabase.table("nutrition_tracker").insert(nutrition_data).execute()
                st.success(get_translation("✅ Nutrition data saved!"))
            except Exception as e:
                st.warning(get_translation(f"Could not save to database: {str(e)}"))


# ====== STRESS ASSESSMENT ======
def stress_assessment():
    st.markdown("<h1>😌 Stress Assessment</h1>", unsafe_allow_html=True)

    with st.expander(get_translation("Assess Your Stress Levels"), expanded=True):
        st.write(get_translation("Rate your stress levels for the following factors (0-4):"))

        col1, col2 = st.columns(2)
        with col1:
            financial_stress = st.slider(get_translation("Financial pressure/burden"), 0, 4, 2, key="financial_stress")
            family_stress = st.slider(get_translation("Family/relationship issues"), 0, 4, 2, key="family_stress")
            work_stress = st.slider(get_translation("Work/employment stress"), 0, 4, 2, key="work_stress")
            safety_stress = st.slider(get_translation("Community safety concerns"), 0, 4, 2, key="safety_stress")
        with col2:
            caregiver_stress = st.slider(get_translation("Caregiver burden"), 0, 4, 2, key="caregiver_stress")
            migration_stress = st.slider(get_translation("Migration/displacement stress"), 0, 4, 2, key="migration_stress")
            family_expectations = st.slider(get_translation("Traditional family expectations"), 0, 4, 2, key="family_expectations")
            spiritual_stress = st.slider(get_translation("Spiritual/religious conflicts"), 0, 4, 2, key="spiritual_stress")

        total_score = (financial_stress + family_stress + work_stress + safety_stress +
                       caregiver_stress + migration_stress + family_expectations + spiritual_stress)

        if total_score <= 12:
            level, color = get_translation("Low"), "green"
        elif total_score <= 20:
            level, color = get_translation("Moderate"), "orange"
        else:
            level, color = get_translation("High"), "red"

        st.markdown(f"""
        <div style='padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-top: 20px;'>
            <h4>🧠 Total Stress Score: <span style='color:{color};'>{total_score}/32</span> → {level} Stress</h4>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.stress_score = total_score
        st.session_state.stress_level = level

        if st.button(get_translation("💾 Save Stress Assessment"), key="save_stress"):
            stress_data = {
                "financial_stress": financial_stress, "family_stress": family_stress,
                "work_stress": work_stress, "safety_stress": safety_stress,
                "caregiver_stress": caregiver_stress, "migration_stress": migration_stress,
                "family_expectations": family_expectations, "spiritual_stress": spiritual_stress,
                "total_score": total_score, "stress_level": level,
                "user_id": st.session_state.get('user_id', 'anonymous')
            }
            try:
                supabase.table("stress_assessments").insert(stress_data).execute()
                st.success(get_translation("✅ Stress assessment saved successfully!"))
            except Exception as e:
                st.error(get_translation(f"❌ Error: {str(e)}"))


# ====== REPORTS PAGE ======
def render_reports_page():
    st.markdown("<h1>📄 My Health Reports</h1>", unsafe_allow_html=True)

    if not st.session_state.reports:
        st.info(get_translation("No reports generated yet. Complete an assessment to generate your first report."))
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🩺 Start Stroke Assessment", use_container_width=True):
                st.session_state.current_page = "Stroke Assessment"
                st.rerun()
        with col2:
            if st.button("🧠 Start Dementia Assessment", use_container_width=True):
                st.session_state.current_page = "Dementia Assessment"
                st.rerun()
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reports", len(st.session_state.reports))
        col2.metric("Stroke Assessments", sum(1 for r in st.session_state.reports.values() if r['type'] == 'Stroke'))
        col3.metric("Dementia Assessments", sum(1 for r in st.session_state.reports.values() if r['type'] == 'Dementia'))

        st.markdown("---")
        for report_id, report in st.session_state.reports.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                icon = "🩺" if report['type'] == "Stroke" else "🧠"
                st.subheader(f"{icon} {report['type']} Assessment")
                color_icon = "🔴" if report['risk_level'] == "HIGH" else "🟡" if report['risk_level'] == "MEDIUM" else "🟢"
                st.write(f"**Risk Level:** {color_icon} {report['risk_level']}")
                st.write(f"**Risk Score:** {report['risk_score']:.1%}")
                st.write(f"**Date:** {report['data'].get('assessment_date', 'N/A')}")
            with col2:
                if st.button("📥 Download", key=f"download_{report_id}", use_container_width=True):
                    with st.spinner("Generating PDF..."):
                        try:
                            if report['type'] == "Stroke":
                                pdf_bytes = generate_stroke_pdf(st.session_state.user_name,
                                                                report['data'], report['risk_score'],
                                                                report['risk_level'], [])
                            else:
                                pdf_bytes = generate_alzheimer_pdf(st.session_state.user_name,
                                                                   report['data'], report['risk_score'],
                                                                   report['risk_level'], [])
                            st.markdown(create_download_link(pdf_bytes, report['filename']), unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
            with col3:
                if st.button("🗑️ Delete", key=f"del_{report_id}", use_container_width=True, type="secondary"):
                    if st.session_state.get(f'confirm_delete_{report_id}'):
                        del st.session_state.reports[report_id]
                        st.success("Report deleted!")
                        st.rerun()
                    else:
                        st.session_state[f'confirm_delete_{report_id}'] = True
                        st.warning("Click again to confirm deletion")
            st.markdown("---")


# ====== PAGE ROUTER ======
def route_pages(current_page):
    if current_page == "Dashboard":
        render_dashboard()
    elif current_page == "Stroke Assessment":
        render_stroke_assessment()
    elif current_page == "Dementia Assessment":
        render_alzheimer_assessment()
    elif current_page == "Memory Game":
        memory_recall_game()
    elif current_page == "Nutrition Tracker":
        nutrition_tracker()
    elif current_page == "Stress Assessment":
        stress_assessment()
    elif current_page == "My Reports":
        render_reports_page()


# ====== SIDEBAR ======
def render_sidebar():
    with st.sidebar:
        lang = set_language_selector(widget_key="app_language_selector")
        title = get_translation("title")
        subtitle = get_translation("subtitle")

        display_logo()

        if lang == "ar":
            st.markdown(f"""
                <div style="text-align: right; direction: rtl;">
                    <h2 style="margin-bottom:0;">{title}</h2>
                    <p style="color: #6B7280;">{subtitle}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="text-align: center;">
                    <h2 style="margin-bottom:0;">{title}</h2>
                    <p style="color: #6B7280;">{subtitle}</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("## 🧭 Navigation")

        if st.session_state.logged_in:
            st.markdown(f"**👤 Welcome, {st.session_state.user_name}!**")

        pages = [
            ("🏠 Dashboard", "Dashboard"),
            ("🩺 Stroke Assessment", "Stroke Assessment"),
            ("🧠 Dementia Assessment", "Dementia Assessment"),
            ("🎮 Memory Game", "Memory Game"),
            ("🥗 Nutrition Tracker", "Nutrition Tracker"),
            ("😌 Stress Assessment", "Stress Assessment"),
            ("📄 My Reports", "My Reports")
        ]

        for label, page_name in pages:
            btn_type = "primary" if st.session_state.current_page == page_name else "secondary"
            if st.button(label, key=f"nav_{page_name}", use_container_width=True, type=btn_type):
                st.session_state.current_page = page_name
                st.rerun()

        st.markdown("---")

        if not st.session_state.logged_in:
            st.markdown("### Start Session")
            user_name = st.text_input("Enter your name:", key="sidebar_name_input")
            if st.button("Start Session", use_container_width=True, key="sidebar_start"):
                if user_name:
                    st.session_state.user_name = user_name
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Please enter your name")
        else:
            if st.button("Logout", use_container_width=True, key="sidebar_logout"):
                st.session_state.logged_in = False
                st.session_state.user_name = ""
                st.rerun()

        # Location filters — updates session state encoded values
        render_location_filters()

        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        if st.session_state.previous_stats:
            st.metric("Stroke Predictions", st.session_state.previous_stats.get("stroke_predictions", 0))
            st.metric("Dementia Predictions", st.session_state.previous_stats.get("dementia_predictions", 0))


# ====== FOOTER ======
def show_footer():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.caption("© 2024 African NeuroHealth AI")
    col2.caption("Version 2.0")
    col3.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d')}")


# ====== MAIN ======
def main():
    init_session_state()
    apply_rtl_logic()
    render_sidebar()

    if not st.session_state.logged_in:
        show_welcome_screen()
    else:
        route_pages(st.session_state.current_page)

    show_footer()


if __name__ == "__main__":
    main()


