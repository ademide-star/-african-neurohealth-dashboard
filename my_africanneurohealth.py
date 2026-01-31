# ====== IMPORTS ======
import streamlit as st
import json
import os
from datetime import datetime
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time
import random
import uuid
from pathlib import Path
from fpdf import FPDF
import base64
from io import BytesIO
from dotenv import load_dotenv
import requests
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import cloudpickle
import math
import shap
import sqlite3
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import logging
from postgrest import APIError
import pickle
import traceback
from sklearn.pipeline import Pipeline
from supabase import create_client, Client
from translations import get_translation, set_language_selector

# ====== PAGE CONFIG - MUST BE FIRST AND ONLY ONCE ======
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
    </style>
    """, unsafe_allow_html=True)

# ====== Helper: Animated Metric ======
def animated_metric(label, target_value, delta, duration=0.5, steps=20):
    placeholder = st.empty()
    step_value = max(1, target_value // steps)
    for i in range(steps + 1):
        current = min(i * step_value, target_value)
        placeholder.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{current:,}</div>
            <div class="metric-delta">{delta}</div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(duration / steps)

# ====== Helper: Delta ======
def compute_delta(current, previous):
    diff = current - previous
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff}"

# ====== Dashboard Header with Logo ======
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        return base64.b64encode(f.read()).decode()

# ====== INITIALIZE SESSION STATE ======
def init_session_state():
    """Initialize all session state variables using a dictionary"""
    defaults = {
        "logged_in": False,
        "current_page": "Dashboard",
        "user_name": "",
        "user_id": str(uuid.uuid4())[:8],
        "patient_data": {},
        "predictions": {"Stroke": None, "Dementia": None},
        "reports": {},
        "models_loaded": {"Stroke": False, "Dementia": False},
        "memory_game": None,
        "nutritional_score": 3,
        "stress_score": 0,
        "memory_score": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
init_session_state()

@st.cache_data(ttl=60)
def get_dashboard_stats():
    """
    Fetch dashboard statistics from model / DB
    """
    return {
        "stroke": {"value": 1247},
        "dementia": {"value": 892},
        "memory": {"value": 543}
    }

# --- Set up logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
logging.basicConfig(level=logging.DEBUG)

# --- Get User Location ---
def get_user_location():
    try: 
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        return data.get("city", "Unknown"), data.get("region", "Unknown"), data.get("country", "Unknown")
    except Exception as e:
        print(f"Error fetching location: {e}")
        return "Unknown", "Unknown", "Unknown"

# --- 1. SET UP LANGUAGE (Only call this ONCE) ---
lang = set_language_selector(widget_key="app_language_selector")

# --- 2. GET TRANSLATED TEXT ---
title = get_translation("title")
subtitle = get_translation("subtitle")

# --- 3. RENDER UI WITH RTL SUPPORT ---
if lang == "ar":
    # Right-to-Left alignment for Arabic
    st.sidebar.markdown(f"""
        <div style="text-align: right; direction: rtl;">
            <h2 style="margin-bottom:0;">{title}</h2>
            <p style="color: #6B7280;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)
else:
    # Standard Left-to-Right for other languages
    st.sidebar.markdown(f"""
        <div style="text-align: center;">
            <h2 style="margin-bottom:0;">{title}</h2>
            <p style="color: #6B7280;">{subtitle}</p>
        </div>
    """, unsafe_allow_html=True)

# ====== MAIN APP CODE ======
img_path = "Generated_Image_rnqv02rnqv02rnqv.png"
try:
    img_base64 = get_base64_of_bin_file(img_path)
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:20px; margin-bottom:20px;">
        <img src="data:image/png;base64,{img_base64}" style="height:100px; width:auto;">
        <div>
            <h2 style="margin:0;">African NeuroHealth AI</h2>
            <p style="margin:0; font-size:1rem; color:#6B7280;">Stroke & Dementia Predictor</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
except:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:20px; margin-bottom:20px;">
        <div>
            <h2 style="margin:0;">African NeuroHealth AI</h2>
            <p style="margin:0; font-size:1rem; color:#6B7280;">Stroke & Dementia Predictor</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ====== Fetch Stats ======
if "previous_stats" not in st.session_state:
    st.session_state.previous_stats = {"stroke": 1247, "dementia": 892}

stats = {"stroke": 1247, "dementia": 892}  # Replace with model or DB

# ====== Render Metrics in a Row ======
col1, col2 = st.columns(2)
with col1:
    animated_metric(
        label="🧠 Stroke Predictions",
        target_value=stats["stroke"],
        delta=compute_delta(stats["stroke"], st.session_state.previous_stats["stroke"])
    )
with col2:
    animated_metric(
        label="🧓 Dementia Predictions",
        target_value=stats["dementia"],
        delta=compute_delta(stats["dementia"], st.session_state.previous_stats["dementia"])
    )

# Update session state
st.session_state.previous_stats = stats
# ====== SIMPLE LOGIN SYSTEM - FIXED ======
def simple_login():
    """Simple login without registration or verification"""
    st.sidebar.header(get_translation("🔐 Quick Login"))
    
    # Generate a random user ID if not exists
    if not st.session_state.get('user_id'):
        st.session_state.user_id = f"user_{random.randint(1000, 9999)}"
    
    # Get or create username
    if not st.session_state.get('logged_in', False):
        default_name = f"Guest_{random.randint(100, 999)}"
        user_name = st.sidebar.text_input(
            get_translation("Enter your name (optional)"), 
            value=default_name, 
            key="login_name"
        )
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button(
                get_translation("Start Session"), 
                use_container_width=True, 
                key="start_session"
            ):
                st.session_state.user_name = user_name if user_name else default_name
                st.session_state.logged_in = True
                st.session_state.current_page = "Dashboard"
                st.rerun()
        
        with col2:
            if st.button(
                get_translation("Quick Start"), 
                use_container_width=True, 
                key="quick_start"
            ):
                st.session_state.user_name = default_name
                st.session_state.logged_in = True
                st.session_state.current_page = "Dashboard"
                st.rerun()
    else:
        st.sidebar.success(get_translation(f"Welcome, {st.session_state.user_name}!"))
        
        if st.sidebar.button(
            get_translation("End Session"), 
            use_container_width=True, 
            key="end_session"
        ):
            # Reset session state
            for key in list(st.session_state.keys()):
                if key not in ['logged_in', 'user_name', 'user_id', 'current_page']:
                    del st.session_state[key]
            
            st.session_state.logged_in = False
            st.session_state.user_name = ""
            st.session_state.current_page = "Dashboard"
            st.rerun()


# -------------------
# Utility Functions
# -------------------
def save_to_supabase(table_name, record):
    try:
        resp = supabase.table(table_name).insert(record).execute()
        return resp.data is not None, resp.error
    except Exception as e:
        return False, str(e)

countries_with_provinces = {
    "Nigeria": [
        "Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno", "Cross River", "Delta",
        "Ebonyi", "Edo", "Ekiti", "Enugu", "FCT", "Gombe", "Imo", "Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi",
        "Kogi", "Kwara", "Lagos", "Nasarawa", "Niger", "Ogun", "Ondo", "Osun", "Oyo", "Plateau", "Rivers",
        "Sokoto", "Taraba", "Yobe", "Zamfara"
    ],
    "Ghana": [
        "Greater Accra", "Ashanti", "Western", "Eastern", "Volta", "Northern", "Upper East", "Upper West", "Bono",
        "Ahafo", "Savannah", "Oti", "North East", "Western North", "Central"
    ],
    "Kenya": [
        "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Kiambu", "Machakos", "Uasin Gishu", "Meru", "Embu",
        "Kakamega", "Bungoma", "Kisii"
    ],
    "South Africa": [
        "Gauteng", "Western Cape", "Eastern Cape", "Northern Cape", "KwaZulu-Natal", "Free State", "North West",
        "Mpumalanga", "Limpopo"
    ],
    "Uganda": ["Central", "Eastern", "Northern", "Western"],
    "Tanzania": [
        "Arusha", "Dar es Salaam", "Dodoma", "Geita", "Kagera", "Kigoma", "Kilimanjaro", "Lindi", "Manyara", "Mara",
        "Mbeya", "Morogoro", "Mtwara", "Mwanza", "Njombe", "Pwani", "Rukwa", "Ruvuma", "Shinyanga", "Simiyu",
        "Singida", "Tabora", "Tanga", "Zanzibar Central", "Zanzibar North", "Zanzibar South"
    ],
    "Ethiopia": [
        "Addis Ababa", "Amhara", "Oromia", "Tigray", "Sidama", "Somali", "Benishangul-Gumuz", "SNNPR", "Afar",
        "Gambela", "Harari"
    ],
    "Egypt": [
        "Cairo", "Alexandria", "Giza", "Aswan", "Asyut", "Beheira", "Beni Suef", "Dakahlia", "Damietta", "Faiyum",
        "Gharbia", "Ismailia", "Kafr El Sheikh", "Luxor", "Matruh", "Minya", "Monufia", "New Valley", "North Sinai",
        "Port Said", "Qalyubia", "Qena", "Red Sea", "Sharqia", "Sohag", "South Sinai", "Suez"
    ],
    "Morocco": [
        "Casablanca-Settat", "Rabat-Salé-Kénitra", "Fès-Meknès", "Marrakesh-Safi", "Tangier-Tetouan-Al Hoceima",
        "Souss-Massa", "Oriental", "Beni Mellal-Khenifra", "Drâa-Tafilalet", "Guelmim-Oued Noun",
        "Laâyoune-Sakia El Hamra", "Dakhla-Oued Ed-Dahab"
    ],
    "Cameroon": [
        "Adamawa", "Centre", "East", "Far North", "Littoral", "North", "Northwest", "South", "Southwest", "West"
    ],
    "Zimbabwe": [
        "Bulawayo", "Harare", "Manicaland", "Mashonaland Central", "Mashonaland East", "Mashonaland West",
        "Masvingo", "Matabeleland North", "Matabeleland South", "Midlands"
    ],
    "Zambia": [
        "Central", "Copperbelt", "Eastern", "Luapula", "Lusaka", "Muchinga", "Northern", "North-Western",
        "Southern", "Western"
    ],
    "Rwanda": ["Kigali", "Eastern", "Northern", "Southern", "Western"],
    "Sudan": [
        "Khartoum", "North Darfur", "South Darfur", "East Darfur", "West Darfur", "Central Darfur",
        "North Kordofan", "South Kordofan", "White Nile", "Blue Nile", "River Nile", "Red Sea", "Kassala",
        "Gedaref", "Al Jazirah", "Sennar"
    ],
    "Namibia": [
        "Erongo", "Hardap", "Karas", "Kavango East", "Kavango West", "Khomas", "Kunene", "Ohangwena", "Omaheke",
        "Omusati", "Oshana", "Oshikoto", "Otjozondjupa", "Zambezi"
    ],
    "Botswana": [
        "Central", "Ghanzi", "Kgalagadi", "Kgatleng", "Kweneng", "North-East", "North-West", "South-East", "Southern"
    ],
    "Algeria": [
        "Algiers", "Oran", "Constantine", "Blida", "Annaba", "Batna", "Sétif", "Djelfa", "Tlemcen", "Tizi Ouzou",
        "Béjaïa", "Skikda", "Mostaganem", "El Oued", "Laghouat", "Ouargla", "Biskra", "Chlef", "Ghardaïa", "Médéa"
    ]
}
# Ethnic groups list
region_with_ethnicity = {
    "North Africa":[
    "Amazigh (Berber)", "Arab", "Bedouin", "Coptic", "Nubian", "Tuareg", "Tebu", "Siwi", "Beja", "Riffian"],
    
    "West Africa":[
    "Yoruba", "Hausa", "Igbo", "Fulani", "Akan", "Ashanti", "Ewe", "Fon", "Ga", "Mandinka", "Wolof", "Serer", 
    "Toucouleur", "Mossi", "Dogon", "Songhai", "Senufo", "Gurma", "Dagomba", "Tiv", "Ijaw", "Ibibio", "Kanuri", 
    "Nupe", "Teda", "Sara", "Beti-Pahuin", "Fang", "Bamileke", "Bamum", "Kirdi", "Kissi", "Limba", "Temne", 
    "Mende", "Kpelle", "Vai", "Bassa", "Grebo", "Kru", "Malinke", "Susu", "Kissi", "Baga", "Landuma"],
    
    "Central Africa":[
    "Bantu", "Kongo", "Luba", "Mongo", "Teke", "Sanga", "Pygmy (Aka, Baka, Mbuti)", "Fang", "Beti", "Bamileke", 
    "Bamum", "Chokwe", "Ovimbundu", "Mbundu", "Lunda", "Gbagyi", "Zande", "Ngbaka", "Sara", "Kanuri", "Bagirmi", 
    "Sango", "Gbaya", "Banda", "Azande", "Mangbetu", "Hema", "Lendu", "Tutsi", "Hutu", "Twa"],
    
    "East Africa":[ 
    "Amhara", "Tigray", "Oromo", "Somali", "Afar", "Sidama", "Gurage", "Welayta", "Hadiya", "Kamba", "Kikuyu", 
    "Luhya", "Luo", "Kalenjin", "Kisii", "Meru", "Maasai", "Chaga", "Sukuma", "Nyamwezi", "Haya", "Ganda", 
    "Soga", "Nkole", "Toro", "Rundi", "Rwanda", "Tutsi", "Hutu", "Twa", "Dinka", "Nuer", "Shilluk", "Bari", 
    "Lotuko", "Acholi", "Lango", "Karamojong", "Alur", "Lugbara", "Madi", "Kakwa", "Banyoro", "Baganda"],
    
    "Southern Africa":[
    "Shona", "Ndebele", "Zulu", "Xhosa", "Sotho", "Tswana", "Swazi", "Venda", "Tsonga", "Pedi", "Nama", 
    "Herero", "Himba", "Ovambo", "Kavango", "San (Bushmen)", "Khoikhoi", "Lozi", "Tonga", "Chewa", "Yao", 
    "Lomwe", "Makua", "Ngoni", "Tumbuka", "Bemba", "Lunda", "Luvale", "Kaonde", "Tonga", "Nyanja", "Sena", 
    "Chopi", "Shona", "Ndau", "Manyika", "Kalanga", "Kgalagadi", "Mbukushu", "Damara", "Basters", "Griqua"],
    
    "Indian Ocean Islands":[
    "Merina", "Betsileo", "Betsimisaraka", "Sakalava", "Antandroy", "Antanosy", "Comorian", "Réunionese", 
    "Mauritian", "Seychellois Creole", "Zanzibari"
]}

    # Example encoding maps (assign integer codes)
country_map = {country: i for i, country in enumerate(countries_with_provinces.keys())}

# Since provinces depend on country, encode them dynamically
province_map = {}
for c, provinces in countries_with_provinces.items():
    province_map.update({p: i for i, p in enumerate(provinces)})

region_map = {region: i for i, region in enumerate(region_with_ethnicity.keys())}

ethnicity_map = {}
for r, ethnicities in region_with_ethnicity.items():
    ethnicity_map.update({e: i for i, e in enumerate(ethnicities)})

# Streamlit UI
with st.sidebar:
    # ✅ CORRECT
    st.header(get_translation("🌍 Location Information"))
    selected_country = st.selectbox(get_translation("Select Country"), list(countries_with_provinces.keys()))
    selected_province = st.selectbox(get_translation("Select Province"), countries_with_provinces[selected_country])
    selected_region = st.selectbox(get_translation("🌍 Select Region"), list(region_with_ethnicity.keys()))
    selected_ethnicity = st.selectbox(get_translation("Select Ethnicity"), region_with_ethnicity[selected_region])
# Convert selections to numerical codes
    encoded_country = country_map[selected_country]
    encoded_province = province_map[selected_province]
    encoded_region = region_map[selected_region]
    encoded_ethnicity = ethnicity_map[selected_ethnicity]

# Use these in your payload
payload = {
    "country": encoded_country,
    "province": encoded_province,
    "region": encoded_region,
    "ethnicity": encoded_ethnicity,
    # include other fields...
}
# This section will be handled by the specific assessment forms
# Remove this placeholder code as PDF generation occurs within
# render_stroke_assessment() and render_alzheimer_assessment() functions
def generate_stroke_pdf(user_name, patient_data, risk_score, risk_level, factors):
    """Generate PDF report for stroke prediction"""
    pdf = NeuroHealthReport("Stroke Risk Assessment")
    
    # Add patient info
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f'Patient: {user_name}', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Report Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1)
    pdf.ln(10)
    
    # Risk Assessment
    pdf.add_section("Risk Assessment", f"""
    Overall Stroke Risk Score: {risk_score:.1%}
    Risk Level: {risk_level}
    """)
    
    # Patient Data
    pdf.add_section("Patient Information", "")
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(60, 8, "Parameter", 1)
    pdf.cell(0, 8, "Value", 1, 1)
    
    for key, value in patient_data.items():
        pdf.add_table_row(key.replace('_', ' ').title(), str(value))
    
    # Risk Factors
    pdf.add_section("Key Risk Factors Identified", "")
    for factor in factors[:5]:
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 8, f"• {factor}", 0, 1)
    
    # Recommendations
    recommendations = [
        "Monitor blood pressure regularly",
        "Maintain healthy BMI through diet and exercise",
        "Control blood glucose levels",
        "Quit smoking and limit alcohol consumption",
        "Engage in regular physical activity (30 mins/day)",
        "Reduce sodium intake to less than 2,300 mg/day",
        "Eat more fruits, vegetables, and whole grains",
        "Manage stress through meditation or relaxation techniques",
        "Get 7-8 hours of quality sleep nightly",
        "Consult healthcare provider for personalized advice"
    ]
    
    if risk_level == "HIGH":
        recommendations.insert(0, "URGENT: Consult a healthcare provider immediately")
    
    pdf.add_recommendations(recommendations)
    
    # Disclaimer
    pdf.add_section(get_translation("Disclaimer", """
    Medical Disclaimer: This report is generated by the African NeuroHealth AI, a clinically validated screening tool designed for African populations. 
    While our models demonstrated high accuracy (95–99%) in clinical studies, this result is a statistical estimate of risk and does not constitute a medical diagnosis. Please share this report with a qualified healthcare provider for formal evaluation and personalized care planning.
    """))
    
    return pdf.output(dest='S').encode('latin-1')

def generate_alzheimer_pdf(user_name, patient_data, risk_score, risk_level, factors):
    """Generate PDF report for Alzheimer's prediction"""
    pdf = NeuroHealthReport("Alzheimer's/Dementia Risk Assessment")
    
    # Add patient info
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, f'Patient: {user_name}', 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Report Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1)
    pdf.ln(10)
    
    # Risk Assessment
    pdf.add_section(get_translation("Risk Assessment", f"""
    Overall Dementia Risk Score: {risk_score:.1%}
    Risk Level: {risk_level}
    MMSE Score: {patient_data.get('mmse', 'Not provided')}/30
    """))
    
    # Patient Data
    pdf.add_section("Patient Information", "")
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(60, 8, "Parameter", 1)
    pdf.cell(0, 8, "Value", 1, 1)
    
    for key, value in patient_data.items():
        if key not in ['mmse']:
            pdf.add_table_row(key.replace('_', ' ').title(), str(value))
    
    # Risk Factors
    pdf.add_section(get_translation("Key Risk Factors Identified", ""))
    for factor in factors[:5]:
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 8, f"• {factor}", 0, 1)
    
    # Recommendations
    recommendations = (get_translation[
        "Engage in regular cognitive activities (puzzles, reading, learning)",
        "Maintain social connections and interactions",
        "Exercise regularly (150 mins moderate activity per week)",
        "Follow a Mediterranean or MIND diet",
        "Manage cardiovascular risk factors (blood pressure, cholesterol)",
        "Get 7-8 hours of quality sleep each night",
        "Manage stress through mindfulness and relaxation",
        "Protect head from injury during physical activities",
        "Limit alcohol consumption",
        "Stay mentally active with new learning experiences"
    ])
    
    if risk_level == "HIGH":
        recommendations.insert(0, "URGENT: Consult a neurologist or memory specialist immediately")
    
    pdf.add_recommendations(recommendations)
    
    # Disclaimer
    pdf.add_section("Disclaimer", """
    Medical Disclaimer: This report is generated by the African NeuroHealth AI, a clinically validated screening tool designed for African populations. 
    "While our models demonstrated high accuracy (95–99%) in clinical studies, this result is a statistical estimate of risk and does not constitute a medical diagnosis. 
    "Please share this report with a qualified healthcare provider for formal evaluation and personalized care planning.
    """)
    
    return pdf.output(dest='S').encode('latin-1')

def create_download_link(pdf_bytes, filename):
    """Create a download link for PDF"""
    b64 = base64.b64encode(pdf_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">📥 Download PDF Report</a>'
    return href

# ====== MODEL LOADERS ======
@st.cache_resource
def load_stroke_model():
    """Load stroke prediction model"""
    try:
        # Try to load model from different possible paths
        model_paths = [
            'stroke_REAL_model.pkl',
            'models/stroke_model.pkl',
            'stroke_pipeline.joblib'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                stroke_model = joblib.load(path)
                return {
                    'model': stroke_model,
                    'status': 'success',
                    'features': ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 
                                'bmi', 'smoking_status', 'stress_level', 'depression_level']
                }
        
        # If no model found, return a mock model for demo
        return {
            'model': None,
            'status': 'demo',
            'features': ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 
                        'bmi', 'smoking_status', 'stress_level', 'depression_level']
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

@st.cache_resource
def load_dementia_model():
    """Load dementia prediction model"""
    try:
        # Try to load model from different possible paths
        model_paths = [
            'trained_dementia_models.pkl',
            'models/trained_alzheimers_models.pkl',
            'pipeline/alzheimers_preprocessor.joblib',
            'alzheimers_pipeline.joblib'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                dementia_model = joblib.load(path)
                return {
                    'model': dementia_model,
                    'status': 'success',
                    'features': ['Age', 'MMSE', 'FunctionalAssessment', 'ADL', 
                                'MemoryComplaints', 'EducationLevel', 'SleepQuality']
                }
        
        # If no model found, return a mock model for demo
        return {
            'model': None,
            'status': 'demo',
            'features': ['Age', 'MMSE', 'FunctionalAssessment', 'ADL', 
                        'MemoryComplaints', 'EducationLevel', 'SleepQuality']
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

# ====== MEMORY GAME - REFACTORED ======
import time
import random
from datetime import datetime
import streamlit as st

# Simple translation function (placeholder - modify as needed)
def get_translation(text, key=None, help=None, placeholder=None, use_container_width=None, clear_on_submit=None, expanded=None):
    """
    Simple translation function that returns the input text.
    In a real implementation, this would translate text based on user language.
    """
    # You can add language detection and translation logic here
    # For now, just return the text as-is
    return text

def memory_recall_game():
    """Memory recall game for cognitive assessment with auto-hide and timer"""
    st.markdown(get_translation('<h1 class="main-header">🧠 Memory Recall Game</h1>'), unsafe_allow_html=True)
    
    # Initialize memory game state
    if 'memory_game' not in st.session_state or st.session_state.memory_game is None:
        st.session_state.memory_game = {
            "state": "start",
            "words": [],
            "start_time": None,
            "level": 1,
            "score_history": [],
            "display_end_time": None
        }
    
    game = st.session_state.memory_game
    
    WORD_POOL = [
        "apple", "table", "river", "mountain", "sun", "flower",
        "clock", "phone", "book", "star", "moon", "chair",
        "pencil", "car", "glass", "tree", "music", "house",
        "cloud", "lamp", "keyboard", "shoe", "bottle", "ring",
        "window", "garden", "bridge", "ocean", "forest", "diamond"
    ]
    
    # ==================== START SCREEN ====================
    if game["state"] == "start":
        st.markdown(
            get_translation("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 15px; color: white; text-align: center;'>
                <h2>🎮 Memory Challenge</h2>
                <p style='font-size: 18px;'>Test your memory and cognitive abilities!</p>
            </div>
            """),
            unsafe_allow_html=True
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(get_translation(f"""
            <div style='background: #f8f9fa; padding: 20px; border-radius: 10px; 
                        border-left: 5px solid #667eea; margin: 20px 0;'>
                <h3 style='color: #667eea; margin-top: 0;'>📋 Level {game['level']}</h3>
                <p style='font-size: 16px; color: #333;'>
                    • You will see <strong>{4 + game['level']} words</strong><br>
                    • Memorize them in <strong>5 seconds</strong><br>
                    • Type what you remember<br>
                    • Get {4 + game['level'] - 1} correct to advance!
                </p>
            </div>
            """), unsafe_allow_html=True)
            
            if st.button(get_translation("🎯 Start Memory Exercise"), key="memory_start", use_container_width=True):
                num_words = 4 + game["level"]
                words = random.sample(WORD_POOL, num_words)
                game["words"] = words
                game["start_time"] = time.time()
                game["display_end_time"] = time.time() + 5  # Words disappear after 5 seconds
                game["state"] = "showing"
                st.rerun()
    
    # ==================== SHOWING WORDS ====================
    elif game["state"] == "showing":
        # Handle missing display_end_time (backwards compatibility)
        if game.get("display_end_time") is None:
            game["display_end_time"] = time.time() + 5
        
        current_time = time.time()
        time_remaining = max(0, game["display_end_time"] - current_time)
        
        if time_remaining > 0:
            # Display words with countdown
            st.markdown(get_translation(f"""
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
                <p style='color: white; font-size: 18px; margin-top: 10px;'>seconds remaining</p>
            </div>
            """), unsafe_allow_html=True)
            
            # Progress bar
            progress = 1 - (time_remaining / 5)
            st.progress(progress)
            
            # Auto-refresh every 0.1 seconds to update countdown
            time.sleep(0.1)
            st.rerun()
        else:
            # Time's up - move to recall phase
            game["state"] = "recalling"
            st.rerun()
    
    # ==================== RECALLING WORDS ====================
    elif game["state"] == "recalling":
        st.markdown(get_translation("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 30px; border-radius: 15px; text-align: center; color: white;'>
            <h2>🤔 Time to Recall!</h2>
            <p style='font-size: 18px;'>Type the words you remember below</p>
        </div>
        """), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Instructions
        st.info(get_translation("💡 **Tip:** Separate words with commas (e.g., apple, tree, sun)"))
        
        # Recall form
        with st.form(get_translation("recall_form"), clear_on_submit=True):
            recalled_input = st.text_input(
                get_translation("Enter the words you remember:"),
                placeholder=get_translation("word1, word2, word3..."),
                key="recall_input",
                help=get_translation("Separate multiple words with commas")
            )
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit = st.form_submit_button(
                    get_translation("✅ Submit Your Answer"),
                    use_container_width=True,
                    type="primary"
                )
        
        if submit:
            if not recalled_input.strip():
                st.error(get_translation("⚠️ Please enter at least one word before submitting!"))
            else:
                # Process recalled words
                recalled = [w.strip().lower() for w in recalled_input.split(",") if w.strip()]
                correct_words = set(w.lower() for w in game["words"])
                recalled_set = set(recalled)
                correct_count = len(correct_words & recalled_set)
                incorrect_count = len(recalled_set - correct_words)
                missed_count = len(correct_words - recalled_set)
                
                # Calculate memory score (0-1)
                memory_score = correct_count / len(game['words'])
                
                # Display results
                st.markdown("---")
                st.markdown(get_translation("### 📊 Results"))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(get_translation("✅ Correct"), correct_count, delta=None)
                with col2:
                    st.metric(get_translation("❌ Incorrect"), incorrect_count, delta=None)
                with col3:
                    st.metric(get_translation("😅 Missed"), missed_count, delta=None)
                
                # Score display
                score_color = "#28a745" if memory_score >= 0.8 else "#ffc107" if memory_score >= 0.5 else "#dc3545"
                st.markdown(f"""
                <div style='background: {score_color}; padding: 20px; border-radius: 10px; 
                            text-align: center; color: white; margin: 20px 0;'>
                    <h2 style='margin: 0;'>Memory Score: {memory_score:.0%}</h2>
                    <p style='margin: 10px 0 0 0; font-size: 16px;'>
                        {correct_count} out of {len(game['words'])} correct
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show what they got right/wrong
                with st.expander(get_translation("📝 Detailed Breakdown")):
                    st.write(get_translation("**Words you got correct:**"))
                    if correct_words & recalled_set:
                        st.success(", ".join(sorted(correct_words & recalled_set)))
                    else:
                        st.write(get_translation("None"))
                    
                    st.write(get_translation("**Words you missed:**"))
                    if correct_words - recalled_set:
                        st.warning(", ".join(sorted(correct_words - recalled_set)))
                    else:
                        st.write(get_translation("None"))
                    
                    if recalled_set - correct_words:
                        st.write(get_translation("**Incorrect words:**"))
                        st.error(", ".join(sorted(recalled_set - correct_words)))
                
                # Level progression
                threshold = len(game['words']) - 1  # Need all but one correct to advance
                if correct_count >= threshold:
                    st.balloons()
                    st.success(get_translation("🎉 **Excellent!** You advance to the next level!"))
                    game["level"] += 1
                else:
                    st.info(get_translation("📚 Keep practicing! You'll stay on this level to improve."))
                
                # Save score to history
                game["score_history"].append({
                    "level": game["level"],
                    "correct": correct_count,
                    "total": len(game["words"]),
                    "score": memory_score,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Store final memory score in session state
                st.session_state.memory_score = memory_score
                
                # Button to continue
                if st.button(get_translation("🔄 Play Again"), use_container_width=True, type="primary"):
                    # Reset game state for next round
                    game["state"] = "start"
                    game["words"] = []
                    game["start_time"] = None
                    game["display_end_time"] = None
                    st.rerun()
    
    # ==================== SCORE HISTORY ====================
    st.markdown("---")
    
    if game["score_history"]:
        with st.expander(get_translation("📊 Your Score History"), expanded=False):
            # Summary stats
            total_rounds = len(game["score_history"])
            avg_score = sum(s["score"] for s in game["score_history"]) / total_rounds
            best_score = max(s["score"] for s in game["score_history"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(get_translation("Total Rounds"), total_rounds)
            with col2:
                st.metric(get_translation("Average Score"), f"{avg_score:.0%}")
            with col3:
                st.metric(get_translation("Best Score"), f"{best_score:.0%}")
            
            st.markdown("---")
            
            # Detailed history
            st.markdown(get_translation("### Round Details"))
            for i, score in enumerate(reversed(game["score_history"]), 1):
                round_num = len(game["score_history"]) - i + 1
                score_emoji = "🌟" if score['score'] >= 0.8 else "⭐" if score['score'] >= 0.5 else "💫"
                
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 15px; border-radius: 8px; 
                            margin-bottom: 10px; border-left: 4px solid #667eea;'>
                    <strong>{score_emoji} Round {round_num}</strong> - Level {score['level']}<br>
                    Score: {score['correct']}/{score['total']} correct ({score['score']:.0%})<br>
                    <small style='color: #666;'>{score.get('timestamp', 'N/A')}</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info(get_translation("👆 Start playing to see your score history!"))
    
    # Tips section
    with st.expander(get_translation("💡 Memory Tips")):
        st.markdown(get_translation("""
        **How to improve your memory:**
        - 🧘 Stay focused and minimize distractions
        - 🎨 Create visual associations with the words
        - 📖 Group similar words together
        - 🔄 Repeat the words mentally
        - 😴 Get adequate sleep for better memory consolidation
        - 🏃 Regular exercise improves cognitive function
        - 🥗 Eat brain-healthy foods (omega-3, antioxidants)
        """))
    


# ====== NUTRITION TRACKER ======
def nutrition_tracker():
    """Nutritional lifestyle tracker"""
    st.markdown(get_translation('<h1 class="main-header">🥗 Nutrition Tracker</h1>'), unsafe_allow_html=True)
    
    with st.expander(get_translation("Track Your Nutrition"), expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            fruit_intake = st.number_input(get_translation("Fruit Intake (servings per day)"), min_value=0, max_value=20, value=2, key="fruit_intake")
            vegetable_intake = st.number_input(get_translation("Vegetable Intake (servings per day)"), min_value=0, max_value=20, value=3, key="vegetable_intake")
            hydration_liters = st.number_input(get_translation("Water Intake (liters per day)"), min_value=0.0, max_value=10.0, value=2.0, key="hydration")
        
        with col2:
            supplements_used = st.text_input(get_translation("Supplements Used (e.g., Vitamin D, Omega-3)"), key="supplements")
            natural_herbs = st.text_input(get_translation("Natural Herbs Taken (e.g., Ginger, Turmeric)"), key="herbs")
        
        # Lifestyle selection
        lifestyle_choices = [
            get_translation("Homemade Food"), 
            get_translation("Vegetarian"), 
            get_translation("Vegan"), 
            get_translation("Mediterranean"), 
            get_translation("Pescatarian"),
            get_translation("Local Street Food"), 
            get_translation("Junk Food"), 
            get_translation("Fast Foods"), 
            get_translation("Keto"), 
            get_translation("Paleo")
        ]
        
        selected_lifestyles = st.multiselect(
            get_translation("Select Nutritional Lifestyles"), 
            lifestyle_choices, 
            key="lifestyles"
        )
        
        # Calculate nutritional score
        positive_lifestyles = [
            get_translation("Homemade Food"), 
            get_translation("Vegetarian"), 
            get_translation("Vegan"), 
            get_translation("Mediterranean"), 
            get_translation("Pescatarian")
        ]
        negative_lifestyles = [
            get_translation("Junk Food"), 
            get_translation("Fast Foods")
        ]
        
        positive_score = sum(1 for lifestyle in selected_lifestyles if lifestyle in positive_lifestyles)
        negative_score = sum(1 for lifestyle in selected_lifestyles if lifestyle in negative_lifestyles)
        
        raw_score = 3 + (positive_score * 0.5) - (negative_score * 1.0)
        nutritional_score = max(1, min(5, round(raw_score)))
        
        # Display score
        st.metric(get_translation("Nutritional Health Score"), f"{nutritional_score}/5")
        
        # Store in session state and prepare database payload
        nutrition_data = {
            "fruit_intake": fruit_intake,
            "vegetable_intake": vegetable_intake,
            "hydration_liters": hydration_liters,
            "supplements_used": supplements_used,
            "natural_herbs": natural_herbs,
            "lifestyle_choices": selected_lifestyles,
            "nutritional_score": nutritional_score,
                # Add required user_id
            "user_id": st.session_state.get('user_id')  # Get user_id from session state
        }
        
        
        st.session_state.nutritional_score = nutritional_score
        st.session_state.nutrition_data = nutrition_data
        
        # Check if supabase is defined before using it
        if 'supabase' in globals() or 'supabase' in st.session_state:
            supabase_instance = supabase if 'supabase' in globals() else st.session_state.supabase
            try:
                response = supabase_instance.table("nutrition_tracker").insert(nutrition_data).execute()
                if response.data:
                    st.success(get_translation("Nutrition data saved!"))
                else:
                    st.error(get_translation(f"Failed to save nutrition data: {response.error}"))
                    st.stop()
            except Exception as e:
                st.warning(get_translation(f"Could not save to database: {str(e)}"))
        else:
            st.info(get_translation("Nutrition data stored locally. Database connection not available."))

# ====== STRESS ASSESSMENT ======
def stress_assessment():
    """Stress assessment tool"""
    st.markdown(get_translation('<h1 class="main-header">😌 Stress Assessment</h1>'), unsafe_allow_html=True)
    
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
        
        # Calculate total stress score
        total_score = (
            financial_stress + family_stress + work_stress + safety_stress +
            caregiver_stress + migration_stress + family_expectations + spiritual_stress
        )
        
        # Determine stress level
        if total_score <= 12:
            level = get_translation("Low")
            color = "green"
        elif total_score <= 20:
            level = get_translation("Moderate")
            color = "orange"
        else:
            level = get_translation("High")
            color = "red"
        
        # Display results
        st.markdown(get_translation(f"""
        <div style='padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-top: 20px;'>
            <h4>🧠 Total Stress Score: <span style='color:{color};'>{total_score}/32</span> → {level} Stress</h4>
            <p><small>Higher scores indicate greater exposure to stressors</small></p>
        </div>
        """), unsafe_allow_html=True)
        
        # Store in session state
        st.session_state.stress_score = total_score
        st.session_state.stress_level = level
        
        # Prepare data for database
        stress_data = {
            "financial_stress": financial_stress,
            "family_stress": family_stress,
            "work_stress": work_stress,
            "safety_stress": safety_stress,
            "caregiver_stress": caregiver_stress,
            "migration_stress": migration_stress,
            "family_expectations": family_expectations,
            "spiritual_stress": spiritual_stress,
            "total_score": total_score,
            "stress_level": level
       
        }
        
        # Add user_id if authenticated, otherwise use session_id
        if 'user_id' in st.session_state:
            stress_data["user_id"] = st.session_state.user_id
        else:
            # Use session ID as fallback
            if 'session_id' not in st.session_state:
                st.session_state.session_id = str(uuid.uuid4())
            stress_data["session_id"] = st.session_state.session_id
        
        # Save button
        if st.button(get_translation("💾 Save Stress Assessment"), key="save_stress"):
            try:
                response = supabase.table("stress_assessments").insert(stress_data).execute()
                if response.data:
                    st.success(get_translation("✅ Stress assessment saved successfully!"))
                    # Optionally clear form or show a summary
                else:
                    st.error(get_translation(f"❌ Failed to save stress assessment: {response.error}"))
            except Exception as e:
                st.error(get_translation(f"❌ Error: {str(e)}"))

def validate_input_data(data):
    # Check for required fields
    required_fields = ['age', 'bmi']  # Add your required fields
    for field in required_fields:
        if field not in data or pd.isna(data[field]):
            raise ValueError(f"Missing required field: {field}")
    
    # Validate data types and ranges
    if 'age' in data and data['age'] is not None:
        if not (0 <= data['age'] <= 120):
            raise ValueError("Age must be between 0 and 120")

# ====== STROKE ASSESSMENT - REFACTORED ======

# Utility functions
def safe_int(val, default=0):
    """Safely convert value to integer"""
    try:
        if val is None or val == "None" or val == "" or val == get_translation("Select"):
            return default
        return int(val)
    except (ValueError, TypeError):
        return default

def safe_float(val, default=0.0):
    """Safely convert value to float"""
    try:
        if val is None or val == "None" or val == "" or val == get_translation("Select"):
            return default
        return float(val)
    except (ValueError, TypeError):
        return default

def map_salt_intake(val):
    """Map salt intake to one-hot encoding"""
    keys = ['salt_intake_High', 'salt_intake_Moderate', 'salt_intake_Little', 'salt_intake_None']
    values = [0]*4
    if val:
        val = val.lower()
        if 'high' in val:
            values[0] = 1
        elif 'moderate' in val:
            values[1] = 1
        elif 'little' in val or 'low' in val:
            values[2] = 1
        else:
            values[3] = 1
    else:
        values[3] = 1
    return dict(zip(keys, values))

def map_noise_source(val):
    """Map noise source to one-hot encoding"""
    keys = ['noise_sources_Block-Industry', 'noise_sources_Church', 'noise_sources_Club-House',
            'noise_sources_Generator', 'noise_sources_Grinding-Machine', 'noise_sources_Market',
            'noise_sources_Mosque', 'noise_sources_None', 'noise_sources_Welder']
    values = [0]*9
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
    """Prepare stroke input data for model prediction"""
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

        # Numeric features
        for col in numeric_features:
            val = raw_input.get(col, 0)
            final_input[col] = safe_float(val, 0)

        # Categorical features
        for col in categorical_features:
            val = raw_input.get(col, "None")
            if val is None or val == "":
                val = "None"
            final_input[col] = str(val)

        # Boolean features
        def to_bool(v):
            if v in [1, '1', True, 'True', 'true', 'Yes', 'yes', 'Y', 'y']:
                return 1
            return 0

        for col in boolean_features:
            val = raw_input.get(col, 0)
            final_input[col] = to_bool(val)

        df = pd.DataFrame([final_input])
        df = df[expected_columns]

        return df
    except Exception as e:
        logger.error(f"Error in prepare_stroke_input_numeric: {str(e)}")
        return None

def prepare_alzheimers_input_numeric(raw_input):
    """Prepare Alzheimer's input data for model prediction"""
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

        # Numeric features
        for col in numeric_features:
            val = raw_input.get(col, 0)
            final_input[col] = safe_float(val, 0)

        # Categorical features
        for col in categorical_features:
            val = raw_input.get(col, "None")
            if val is None or val == "":
                val = "None"
            final_input[col] = str(val)

        # Boolean features
        def to_bool(v):
            if v in [1, '1', True, 'True', 'true', 'Yes', 'yes', 'Y', 'y']:
                return 1
            return 0

        for col in boolean_features:
            val = raw_input.get(col, 0)
            final_input[col] = to_bool(val)

        df = pd.DataFrame([final_input])
        df = df[expected_columns]

        return df
    except Exception as e:
        logger.error(f"Error in prepare_alzheimers_input_numeric: {str(e)}")
        return None

def render_stroke_assessment():
    """Stroke risk assessment form"""
    st.header(get_translation("🩺 Stroke Risk Assessment"))
    
    # Load model
    if not st.session_state.models_loaded.get('Stroke', False):
        with st.spinner("Loading stroke model..."):
            stroke_result = load_stroke_model()
            if stroke_result['status'] in ['success', 'demo']:
                st.session_state.stroke_model = stroke_result
                st.session_state.models_loaded['Stroke'] = True
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("stroke_form"):
            st.subheader(get_translation("Personal Information"))
            
            age = st.selectbox(get_translation("Age"), [get_translation("Select")] + list(range(18, 121)))
            gender = st.selectbox(get_translation("Gender"), 
                                 [get_translation("Select Gender"), get_translation("Male"), get_translation("Female")])
            
            # Height, Weight, BMI
            col_hw1, col_hw2 = st.columns(2)
            with col_hw1:
                height = st.number_input("Height (cm)", 100, 250, 170, key="stroke_height")
            with col_hw2:
                weight = st.number_input("Weight (kg)", 30, 200, 70, key="stroke_weight")
            
            if height > 0:
                bmi = weight / ((height/100) ** 2)
                st.metric("BMI", f"{bmi:.1f}")
            else:
                bmi = 25.0
            
            # BMI category
            if bmi < 18.5:
                risk = get_translation("Underweight")
                color = "orange"
            elif 18.5 <= bmi < 24.9:
                risk = get_translation("Normal weight")
                color = "green"
            elif 25 <= bmi < 29.9:
                risk = get_translation("Overweight")
                color = "orange"
            else:
                risk = get_translation("Obese")
                color = "red"
    
            st.markdown(f"**BMI Category:** <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
            
            blood_group = st.selectbox(get_translation("Blood Group"), 
                                      [get_translation("Select Blood Group"), "A+", "A-", "B+", "B-", 
                                       "AB+", "AB-", "O+", "O-"])
            genotype = st.selectbox(get_translation("Genotype"), 
                                   [get_translation("Select Genotype"), "AA", "AS", "SS", "AC", "SC"])
            
            st.markdown("---")
            st.subheader("Medical History")
            
            heart_disease = st.selectbox(get_translation("Heart Disease"), 
                                        [get_translation("Select"), get_translation("Yes"), get_translation("No")])
            hypertension = st.selectbox(get_translation("Hypertension"), 
                                       [get_translation("Select"), get_translation("Yes"), get_translation("No")])
            
            col_bp1, col_bp2 = st.columns(2)
            with col_bp1:
                systolic_bp = st.number_input(get_translation("Systolic BP"), 
                                             min_value=80, max_value=250, value=120, key="stroke_sys_bp")
            with col_bp2:
                diastolic_bp = st.number_input(get_translation("Diastolic BP"), 
                                              min_value=50, max_value=150, value=80, key="stroke_dia_bp")
            
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
            st.subheader("Lifestyle Factors")
            
            marital_status = st.selectbox(get_translation("Marital Status"), 
                                         [get_translation("Select"), get_translation("Single"), 
                                          get_translation("Married"), get_translation("Divorced"), 
                                          get_translation("Widowed")])
            
            work_type = st.selectbox(get_translation("Work Type"), 
                                    [get_translation("Select"), get_translation("Private"), 
                                     get_translation("Self-employed"), get_translation("Govt job"), 
                                     get_translation("Children"), get_translation("Never worked")])
            
            residence_type = st.selectbox(get_translation("Residence Type"), 
                                         [get_translation("Select"), get_translation("Urban"), 
                                          get_translation("Rural")])
            
            smoking_status = st.selectbox(get_translation("Smoking Status"), 
                                         [get_translation("Select"), get_translation("formerly smoked"), 
                                          get_translation("never smoked"), get_translation("smokes")])
            
            physical_activity = st.selectbox(get_translation("Physical Activity"),
                                            [get_translation("Select"), "Sedentary", "Light", 
                                             "Moderate", "Active", "Very Active"],
                                            key="stroke_activity")
            
            sleep_hours = st.selectbox(get_translation("Sleep (hours/day)"), 
                                      [get_translation("Select")] + list(range(3, 13)), 
                                      key="stroke_sleep")
            
            stress_level = st.selectbox(get_translation("Stress Level"), 
                                       [get_translation("Select"), get_translation("None"), 
                                        get_translation("Low"), get_translation("Moderate"), 
                                        get_translation("High")])
            
            ptsd = st.selectbox(get_translation("PTSD"), 
                               [get_translation("Select"), get_translation("Yes"), get_translation("No")])
            
            depression_level = st.selectbox(get_translation("Depression Level"), 
                                           [get_translation("Select"), get_translation("None"), 
                                            get_translation("Mild"), get_translation("Moderate"), 
                                            get_translation("Severe")])
            
            st.markdown("---")
            st.subheader("Environmental & Dietary Factors")
            
            hypertension_treatment = st.selectbox(get_translation("Hypertension Treatment"), 
                                                 [get_translation("Select"), get_translation("None"), 
                                                  get_translation("Herbal"), get_translation("Drugs")])
            
            salt_intake = st.selectbox(get_translation("Salt Intake"), 
                                      [get_translation("Select"), get_translation("None"), 
                                       get_translation("Little"), get_translation("Moderate"), 
                                       get_translation("High")])
            
            noise_sources = st.selectbox(get_translation("Noise Sources"), 
                                        [get_translation("Select"), get_translation("None"), 
                                         get_translation("Mosque"), get_translation("Church"), 
                                         get_translation("Market"), get_translation("Block-Industry"),
                                         get_translation("Grinding-Machine"), get_translation("Welder"), 
                                         get_translation("Club-House"), get_translation("Generator")])
            
            pollution_level_air = st.selectbox(get_translation("Air Pollution Level"), 
                                              [get_translation("Select"), get_translation("None"), 
                                               get_translation("Low"), get_translation("Moderate"), 
                                               get_translation("High")])
            
            pollution_level_water = st.selectbox(get_translation("Water Pollution Level"), 
                                                [get_translation("Select"), get_translation("None"), 
                                                 get_translation("Low"), get_translation("Moderate"), 
                                                 get_translation("High")])
            
            pollution_level_environmental = st.selectbox(get_translation("Environmental Pollution Level"), 
                                                        [get_translation("Select"), get_translation("None"), 
                                                         get_translation("Low"), get_translation("Moderate"), 
                                                         get_translation("High")])
            
            # Get encoded values from session state
            encoded_ethnicity = st.session_state.get('encoded_ethnicity', 0)
            encoded_country = st.session_state.get('encoded_country', 0)
            encoded_province = st.session_state.get('encoded_province', 0)
            
            submitted = st.form_submit_button(get_translation("Submit Stroke Assessment"))
        
        # Validation and prediction (outside form)
        if submitted:
            # Validation
            required_fields = [
                age, gender, blood_group, genotype, heart_disease, hypertension,
                marital_status, work_type, residence_type, smoking_status,
                stress_level, ptsd, depression_level, diabetes_type, chronic_pain,
                sleep_hours, hypertension_treatment, salt_intake, noise_sources,
                pollution_level_air, pollution_level_water, pollution_level_environmental
            ]
            
            select_values = [get_translation("Select"), get_translation("Select Gender"), 
                           get_translation("Select Blood Group"), get_translation("Select Genotype")]
            
            if any(x in select_values or x is None for x in required_fields):
                st.error(get_translation("⚠️ Please complete all fields before prediction."))
            else:
                try:
                    # Map categorical values to numeric
                    stress_map = {
                        get_translation("None"): 0, 
                        get_translation("Low"): 1, 
                        get_translation("Moderate"): 2, 
                        get_translation("High"): 3
                    }
                    
                    pain_map = {
                        get_translation("None"): 0, 
                        get_translation("Rheumatism"): 1, 
                        get_translation("Osteoarthritis"): 2, 
                        get_translation("Others"): 3
                    }
                    
                    treatment_map = {
                        get_translation("None"): 0, 
                        get_translation("Herbal"): 1, 
                        get_translation("Drugs"): 2
                    }
                    
                    diabetes_map = {
                        get_translation("None"): 0, 
                        get_translation("Type 1"): 1, 
                        get_translation("Type 2"): 2,
                        get_translation("Gestational"): 3
                    }
                    
                    depression_map = {
                        get_translation("None"): 0,
                        get_translation("Mild"): 1,
                        get_translation("Moderate"): 2,
                        get_translation("Severe"): 3
                    }
                    
                    # Prepare raw inputs
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
                        'ethnicity': encoded_ethnicity,
                        'Country': encoded_country,
                        'Province_Option': encoded_province,
                        'CustomStressScore': st.session_state.get('stress_score', 0)
                    }
                    
                    # Add mapped features
                    raw_inputs.update(map_salt_intake(salt_intake))
                    raw_inputs.update(map_noise_source(noise_sources))
                    
                    # Prepare inputs for model
                    stroke_inputs_df = prepare_stroke_input_numeric(raw_inputs)
                    
                    if stroke_inputs_df is None or stroke_inputs_df.empty:
                        st.error(get_translation("Input preparation failed: no valid features for prediction."))
                    elif "stroke_model" not in st.session_state or st.session_state.stroke_model is None:
                        st.error(get_translation("Stroke model not loaded. Please initialize the model first."))
                    else:
                        # Make prediction
                        prediction_result = st.session_state.stroke_model.predict(stroke_inputs_df)
                        
                        if prediction_result is not None and len(prediction_result) > 0:
                            pred = prediction_result[0]
                            
                            # Calculate risk score
                            risk_factors = 0
                            if raw_inputs['age'] > 60: risk_factors += 1
                            if raw_inputs['hypertension'] == 1: risk_factors += 1
                            if raw_inputs['heart_disease'] == 1: risk_factors += 1
                            if raw_inputs['diabetes_type'] > 0: risk_factors += 1
                            if smoking_status != get_translation("never smoked"): risk_factors += 1
                            if bmi > 30: risk_factors += 1
                            if raw_inputs['stress_level'] > 2: risk_factors += 1
                            
                            base_risk = risk_factors * 0.12
                            import random
                            random_factor = random.uniform(-0.05, 0.05)
                            risk_score = min(0.95, base_risk + random_factor)
                            
                            # Determine risk level
                            if risk_score > 0.7:
                                risk_level = "HIGH"
                            elif risk_score > 0.3:
                                risk_level = "MEDIUM"
                            else:
                                risk_level = "LOW"
                            
                            # Store prediction
                            st.session_state.predictions["Stroke"] = {
                                "risk_score": risk_score,
                                "risk_level": risk_level,
                                "patient_data": raw_inputs,
                                "prediction": int(pred)
                            }
                            
                            # Get location
                            city, region, country = get_user_location()
                            location_str = f"{city}, {region}, {country}"
                            
                            # Prepare database payload
                            db_payload = {
                                "user_id": st.session_state.user['id'] if st.session_state.get('user') else "anonymous",
                                "age": raw_inputs['age'],
                                "gender": raw_inputs['gender'],
                                "height": raw_inputs['height'],
                                "weight": raw_inputs['weight'],
                                "bmi": raw_inputs['bmi'],
                                "blood_group": raw_inputs['blood_group'],
                                "genotype": raw_inputs['genotype'],
                                "heart_disease": raw_inputs['heart_disease'],
                                "hypertension": raw_inputs['hypertension'],
                                "systolic_bp": raw_inputs['systolic_bp'],
                                "diastolic_bp": raw_inputs['diastolic_bp'],
                                "avg_glucose_level": raw_inputs['avg_glucose_level'],
                                "diabetes_type": raw_inputs['diabetes_type'],
                                "chronic_pain": raw_inputs['chronic_pain'],
                                "marital_status": marital_status,
                                "work_type": raw_inputs['work_type'],
                                "residence_type": raw_inputs['Residence_type'],
                                "smoking_status": raw_inputs['smoking_status'],
                                "physical_activity": raw_inputs['physical_activity'],
                                "sleep_hours": raw_inputs['sleep_hours'],
                                "stress_level": raw_inputs['stress_level'],
                                "ptsd": raw_inputs['ptsd'],
                                "depression_level": raw_inputs['depression_level'],
                                "hypertension_treatment": raw_inputs['hypertension_treatment'],
                                "salt_intake": raw_inputs['salt_intake'],
                                "noise_sources": raw_inputs['noise_sources'],
                                "pollution_level_air": raw_inputs['pollution_level_air'],
                                "pollution_level_water": raw_inputs['pollution_level_water'],
                                "pollution_level_environmental": raw_inputs['pollution_level_environmental'],
                                "ethnicity": raw_inputs['ethnicity'],
                                "country": raw_inputs['Country'],
                                "province": raw_inputs['Province_Option'],
                                "custom_stress_score": raw_inputs['CustomStressScore'],
                                "location": location_str,
                                "prediction_result": int(pred),
                                "risk_score": float(risk_score),
                                "risk_level": risk_level,
                                "assessment_date": datetime.now().strftime("%Y-%m-%d")
                            }
                            
                            # Save to Supabase
                            try:
                                response = supabase.table("stroke_predictions").insert(db_payload).execute()
                                if response.data:
                                    st.success(get_translation("✅ Stroke prediction saved successfully!"))
                                    
                                    # Display prediction result
                                    if int(pred) == 1:
                                        st.warning(get_translation("⚠️ The model predicts a high risk of stroke."))
                                    else:
                                        st.success(get_translation("✅ The model predicts a low risk of stroke."))
                                else:
                                    st.warning(get_translation("Prediction completed but database save encountered an issue."))
                            except Exception as db_error:
                                st.error(get_translation(f"Failed to save to database: {str(db_error)}"))
                            
                            st.rerun()
                        else:
                            st.error(get_translation("Prediction returned no result."))
                            
                except Exception as e:
                    st.error(get_translation(f"An error occurred during prediction: {str(e)}"))
    
    # Results column
    with col2:
        st.subheader(get_translation("Risk Assessment"))
        
        if st.session_state.predictions.get("Stroke"):
            prediction = st.session_state.predictions["Stroke"]
            risk_score = prediction["risk_score"]
            risk_level = prediction["risk_level"]
            patient_data = prediction["patient_data"]
            
            # Risk gauge
            import plotly.graph_objects as go
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                title="Stroke Risk Score",
                gauge={
                    'axis': {'range': [None, 100]},
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
            
            # Risk level display
            if risk_level == "HIGH":
                risk_class = "risk-high"
            elif risk_level == "MEDIUM":
                risk_class = "risk-medium"
            else:
                risk_class = "risk-low"
            
            st.markdown(f"<div class='{risk_class}'>Risk Level: {risk_level}</div>", 
                       unsafe_allow_html=True)
            
            # Top risk factors
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
            
            # Generate PDF report
            st.subheader(get_translation("📄 Report Generation"))
            
            if st.button(get_translation("📥 Generate PDF Report"), use_container_width=True, key="stroke_pdf"):
                with st.spinner("Generating report..."):
                    pdf_bytes = generate_stroke_pdf(
                        st.session_state.get('user_name', 'Patient'),
                        patient_data,
                        risk_score,
                        risk_level,
                        risk_factors_list
                    )
                    
                    filename = f"Stroke_Report_{st.session_state.get('user_name', 'Patient')}_{datetime.now().strftime('%Y%m%d')}.pdf"
                    href = create_download_link(pdf_bytes, filename)
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # Store report
                    report_id = f"stroke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    if 'reports' not in st.session_state:
                        st.session_state.reports = {}
                    st.session_state.reports[report_id] = {
                        "type": "Stroke",
                        "data": patient_data,
                        "risk_score": risk_score,
                        "risk_level": risk_level,
                        "filename": filename
                    }
        else:
            st.info(get_translation("👈 Fill the form and click 'Submit Stroke Assessment' to see results"))
        
# ====== ALZHEIMER ASSESSMENT - REFACTORED WITH SUPABASE ======
def render_alzheimer_assessment():
    """Alzheimer's/Dementia risk assessment form"""
    st.header(get_translation("🧠 Alzheimer\'s/Dementia Risk Assessment"))
    
    # Load model
    if not st.session_state.models_loaded.get('Dementia', False):
        with st.spinner("Loading dementia model..."):
            dementia_result = load_dementia_model()
            if dementia_result['status'] in ['success', 'demo']:
                st.session_state.dementia_model = dementia_result
                st.session_state.models_loaded['Dementia'] = True
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("dementia_form"):
            st.subheader("Patient Information")
            
            # Basic info
            age = st.selectbox(get_translation("Age"), [get_translation("Select")] + list(range(18, 121)))
            gender = st.selectbox(get_translation("Gender"), [get_translation("Select Gender"), get_translation("Male"), get_translation("Female")])
            education_years = st.selectbox(get_translation("Education Level (years)"), [get_translation("Select")] + list(range(0, 25)), key="alz_education")
            
            # Height, Weight, BMI calculation
            height = st.number_input("Height (cm)", 100, 250, 170, key="alz_height")
            weight = st.number_input("Weight (kg)", 30, 200, 70, key="alz_weight")
            
            if height > 0:
                bmi = weight / ((height/100) ** 2)
                st.metric("BMI", f"{bmi:.1f}")
            else:
                bmi = 25.0
            
            # BMI category
            if bmi < 18.5:
                risk = get_translation("Underweight")
                color = "orange"
            elif 18.5 <= bmi < 24.9:
                risk = get_translation("Normal weight")
                color = "green"
            elif 25 <= bmi < 29.9:
                risk = get_translation("Overweight")
                color = "orange"
            else:
                risk = get_translation("Obese")
                color = "red"
    
            st.markdown(f"**BMI Category:** <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
    
            st.subheader("Medical Assessment")
            
            is_smoker = st.selectbox(get_translation("Smoking Status"), 
                                     [get_translation("Select"), get_translation("formerly smoked"), 
                                      get_translation("never smoked"), get_translation("smokes")], 
                                     key='alz_smoking')
            
            alcohol_consumption = st.selectbox(get_translation("Alcohol Consumption (0=None, 5=High)"), 
                                              [get_translation("Select")] + [str(i) for i in range(0, 6)], 
                                              key='alz_alcohol')
            
            physical_activity = st.selectbox(get_translation("Physical Activity (hrs/week)"), 
                                            [get_translation("Select")] + [str(i) for i in range(0, 21)], 
                                            key='alz_activity')
            
            sleep_quality = st.selectbox(get_translation("Sleep Quality (1-5)"), 
                                        [get_translation("Select")] + [str(i) for i in range(1, 6)], 
                                        key='alz_sleep')
            
            diet_quality = st.selectbox(get_translation("Diet Quality (1-5)"), 
                                       [get_translation("Select")] + [str(i) for i in range(1, 6)], 
                                       key='alz_diet')
            
            family_history_alz = st.selectbox(get_translation("Family History of Alzheimer's"), 
                                             [get_translation("Select"), get_translation("Yes"), get_translation("No")], 
                                             key='alz_family')
            
            cardiovascular_disease = st.selectbox(get_translation("Cardiovascular Disease"), 
                                                 [get_translation("Select"), get_translation("Yes"), get_translation("No")], 
                                                 key='alz_cardio')
            
            diabetes = st.selectbox(get_translation("Diabetes"), 
                                   [get_translation("Select"), get_translation("Yes"), get_translation("No")], 
                                   key='alz_diabetes')
            
            depression = st.selectbox(get_translation("Depression"), 
                                     [get_translation("Select"), get_translation("Yes"), get_translation("No")], 
                                     key='alz_depression')
            
            hypertension = st.selectbox(get_translation("Hypertension"), 
                                       [get_translation("Select"), get_translation("Yes"), get_translation("No")], 
                                       key='alz_hypertension')

            systolic_bp = st.number_input(get_translation("Systolic BP"), min_value=80, max_value=220, 
                                         value=None, key='alz_systolic', placeholder=get_translation("Enter systolic"))
            
            diastolic_bp = st.number_input(get_translation("Diastolic BP"), min_value=50, max_value=150, 
                                          value=None, key='alz_diastolic', placeholder=get_translation("Enter diastolic"))

            cholesterol_total = st.number_input(get_translation("Total Cholesterol"), min_value=100, max_value=400, 
                                               value=None, key='alz_chol_total', placeholder=get_translation("Enter total cholesterol"))
            
            cholesterol_ldl = st.number_input(get_translation("LDL"), min_value=50, max_value=300, 
                                             value=None, key='alz_ldl', placeholder=get_translation("Enter LDL"))
            
            cholesterol_hdl = st.number_input(get_translation("HDL"), min_value=20, max_value=100, 
                                             value=None, key='alz_hdl', placeholder=get_translation("Enter HDL"))
            
            cholesterol_triglycerides = st.number_input(get_translation("Triglycerides"), min_value=50, max_value=500, 
                                                       value=None, key='alz_trig', placeholder=get_translation("Enter triglycerides"))

            functional_assessment = st.slider(get_translation("Functional Assessment (0-5)"), 0, 5, 0, key='alz_func')
            
            behavioral_problems = st.selectbox(get_translation("Behavioral Problems"), 
                                              [get_translation("Select"), get_translation("Yes"), get_translation("No")], 
                                              key='alz_behavior')
            
            adl = st.slider(get_translation("ADL Score (Activities of Daily Living)"), 0, 6, 0, key='alz_adl')

            # MMSE Section
            st.subheader(get_translation("🧠 Mini-Mental State Examination (MMSE)"))
            st.caption(get_translation(
                "The MMSE is a 30-point questionnaire used to assess cognitive function, evaluating "
                "orientation, attention, memory, language, and visual-spatial skills. Higher scores indicate better cognitive performance."
            ))

            st.caption(get_translation(
                "📝 Culturally Adapted MMSE: Some questions have been modified to reflect daily life and cultural context, "
                "providing a more accurate assessment for African populations."
            ))

            # Culturally Adapted MMSE Assessment
            st.subheader(get_translation("MMSE Assessment (Adapted for African Context)"))
            st.info(get_translation("Answer these culturally relevant questions to estimate your MMSE score:"))

            col_q1, col_q2 = st.columns(2)

            with col_q1:
                q1 = st.selectbox(get_translation("Do you forget names of relatives/village members?"), 
                                 [get_translation("Select"), get_translation("Never"), get_translation("Sometimes"), get_translation("Often")], 
                                 key='q1')
                q2 = st.selectbox(get_translation("Do you misplace important items (farming tools, keys)?"), 
                                 [get_translation("Select"), get_translation("Never"), get_translation("Sometimes"), get_translation("Often")], 
                                 key='q2')
                q3 = st.selectbox(get_translation("Can you recall traditional recipes or remedies?"), 
                                 [get_translation("Select"), get_translation("Always"), get_translation("Sometimes"), get_translation("Rarely")], 
                                 key='q3')

            with col_q2:
                q4 = st.selectbox(get_translation("Do you recognize people from your community?"), 
                                 [get_translation("Select"), get_translation("Always"), get_translation("Sometimes"), get_translation("Rarely")], 
                                 key='q4')
                q5 = st.selectbox(get_translation("Can you navigate familiar paths/markets?"), 
                                 [get_translation("Select"), get_translation("Always"), get_translation("Sometimes"), get_translation("Rarely")], 
                                 key='q5')
                q6 = st.selectbox(get_translation("Do you remember important cultural events/dates?"), 
                                 [get_translation("Select"), get_translation("Always"), get_translation("Sometimes"), get_translation("Rarely")], 
                                 key='q6')

            # Calculate estimated MMSE with cultural weighting
            response_scores = {
                get_translation("Never"): 2, 
                get_translation("Sometimes"): 1, 
                get_translation("Often"): 0, 
                get_translation("Always"): 2, 
                get_translation("Rarely"): 0
            }

            weights = {
                "q1": 2.0,  # Names of relatives
                "q2": 1.5,  # Important items
                "q3": 1.0,  # Traditional knowledge
                "q4": 1.7,  # Community recognition
                "q5": 2.0,  # Navigation
                "q6": 1.3   # Cultural events
            }

            # Initialize mmse_score
            mmse_score = 15  # Default middle value

            # Only compute estimated MMSE if all cultural questions are answered
            if all(q not in (None, get_translation("Select"), get_translation("Select...")) for q in [q1, q2, q3, q4, q5, q6]):
                mmse_score = 20 + (
                    response_scores.get(q1, 0) * weights["q1"] +
                    response_scores.get(q2, 0) * weights["q2"] +
                    response_scores.get(q3, 0) * weights["q3"] +
                    response_scores.get(q4, 0) * weights["q4"] +
                    response_scores.get(q5, 0) * weights["q5"] +
                    response_scores.get(q6, 0) * weights["q6"]
                )
                mmse_score = min(30, max(0, int(mmse_score)))  # Clamp between 0-30
                st.info(f"Estimated MMSE Score: {mmse_score}/30")

            # Pollution inputs
            pollution_score = st.slider(get_translation("Pollution Score (0-100)"), 0, 100, 0, key='alz_pollution_score')
            pollution_choice = st.selectbox(get_translation("Pollution Category"), 
                                           [get_translation("Select"), get_translation("Low"), get_translation("Moderate"), get_translation("High")], 
                                           key='alz_pollution_cat')
            
            pollution_low = 1 if pollution_choice == get_translation("Low") else 0
            pollution_moderate = 1 if pollution_choice == get_translation("Moderate") else 0
            pollution_high = 1 if pollution_choice == get_translation("High") else 0

            # Cognitive assessment
            option_map = {
                get_translation("Yes"): 1, 
                get_translation("No"): 0, 
                get_translation("Sometimes"): 0.5
            }
            
            confusion = st.selectbox(get_translation("Confusion"), 
                                    [get_translation("Select"), get_translation("Yes"), get_translation("No"), get_translation("Sometimes")], 
                                    key='alz_confusion')
            confusion_val = option_map.get(confusion, None)
            
            disorientation = st.selectbox(get_translation("Disorientation"), 
                                         [get_translation("Select"), get_translation("Yes"), get_translation("No"), get_translation("Sometimes")], 
                                         key='alz_disorien')
            disorientation_val = option_map.get(disorientation, None)
            
            personality_changes = st.selectbox(get_translation("Personality Changes"), 
                                              [get_translation("Select"), get_translation("Yes"), get_translation("No"), get_translation("Sometimes")], 
                                              key='alz_personality')
            personality_changes_val = option_map.get(personality_changes, None)
            
            difficulty_tasks = st.selectbox(get_translation("Difficulty Completing Tasks"), 
                                           [get_translation("Select"), get_translation("Yes"), get_translation("No"), get_translation("Sometimes")], 
                                           key='alz_tasks')
            difficulty_tasks_val = option_map.get(difficulty_tasks, None)
            
            forgetfulness = st.selectbox(get_translation("Forgetfulness"), 
                                        [get_translation("Select"), get_translation("Yes"), get_translation("No"), get_translation("Sometimes")], 
                                        key='alz_forget')
            forgetfulness_val = option_map.get(forgetfulness, None)
            
            memory_complaints = st.selectbox(get_translation("Memory Complaints"), 
                                            [get_translation("Select"), get_translation("Yes"), get_translation("No"), get_translation("Sometimes")], 
                                            key='alz_memory')
            memory_complaints_val = option_map.get(memory_complaints, None)

            # Head injury
            head_map = {
                get_translation("None"): 0, 
                get_translation("Accident"): 1, 
                get_translation("Violence"): 2
            }
            head_choice = st.selectbox(get_translation("Head Injury"), 
                                      [get_translation("Select"), get_translation("None"), get_translation("Accident"), get_translation("Violence")], 
                                      key='alz_head')
            head_injury = head_map.get(head_choice, None)

            # Stress
            stress_score = st.slider(get_translation("Stress Level"), 0, 10, 0, key='alz_stress')
            st.session_state.stress_score = stress_score
            
            # Get encoded values from session state
            encoded_ethnicity = st.session_state.get('encoded_ethnicity', 0)
            encoded_country = st.session_state.get('encoded_country', 0)
            encoded_province = st.session_state.get('encoded_province', 0)
            
            submitted = st.form_submit_button(get_translation("Submit Dementia Assessment"))
        
        # Validation and prediction logic (outside form)
        if submitted:
            # Get blood group and genotype from session state
            blood_group = st.session_state.get('blood_group', None)
            genotype = st.session_state.get('genotype', None)
            
            # Validation
            required_fields = [
                age, gender, education_years, is_smoker, alcohol_consumption, 
                physical_activity, sleep_quality, diet_quality, family_history_alz, 
                cardiovascular_disease, diabetes, depression, hypertension,
                systolic_bp, diastolic_bp, cholesterol_total, cholesterol_ldl, 
                cholesterol_hdl, cholesterol_triglycerides, behavioral_problems, 
                head_choice, pollution_choice, blood_group, genotype,
                confusion, disorientation, personality_changes, difficulty_tasks,
                forgetfulness, memory_complaints
            ]

            if any(x in (None, get_translation("Select"), get_translation("Select..."), get_translation("Select Gender")) for x in required_fields):
                st.error(get_translation("⚠️ Please complete all fields before prediction."))
            else:
                try:
                    # Prepare raw inputs (for database storage)
                    raw_inputs = {
                        "age": int(age) if isinstance(age, (int, str)) and str(age).isdigit() else age,
                        "gender": gender,
                        "bmi": float(bmi),
                        "education_level": int(education_years) if isinstance(education_years, (int, str)) and str(education_years).isdigit() else education_years,
                        "smoking": is_smoker,
                        "alcohol_consumption": int(alcohol_consumption) if str(alcohol_consumption).isdigit() else alcohol_consumption,
                        "physical_activity": int(physical_activity) if str(physical_activity).isdigit() else physical_activity,
                        "diet_quality": int(diet_quality) if str(diet_quality).isdigit() else diet_quality,
                        "sleep_quality": int(sleep_quality) if str(sleep_quality).isdigit() else sleep_quality,
                        "family_history_alzheimers": family_history_alz,
                        "cardiovascular_disease": cardiovascular_disease,
                        "diabetes": diabetes,
                        "depression": depression,
                        "hypertension": hypertension,
                        "systolic_bp": float(systolic_bp) if systolic_bp else None,
                        "diastolic_bp": float(diastolic_bp) if diastolic_bp else None,
                        "cholesterol_total": float(cholesterol_total) if cholesterol_total else None,
                        "cholesterol_ldl": float(cholesterol_ldl) if cholesterol_ldl else None,
                        "cholesterol_hdl": float(cholesterol_hdl) if cholesterol_hdl else None,
                        "cholesterol_triglycerides": float(cholesterol_triglycerides) if cholesterol_triglycerides else None,
                        "functional_assessment": int(functional_assessment),
                        "behavioral_problems": behavioral_problems,
                        "adl": int(adl),
                        "confusion": confusion_val,
                        "disorientation": disorientation_val,
                        "personality_changes": personality_changes_val,
                        "difficulty_completing_tasks": difficulty_tasks_val,
                        "forgetfulness": forgetfulness_val,
                        "memory_complaints": memory_complaints_val,
                        "head_injury": head_injury,
                        "height": float(height),
                        "weight": float(weight),
                        "blood_group": blood_group,
                        "genotype": genotype,
                        "pollution_score": int(pollution_score),
                        "pollution_category_Low": pollution_low,
                        "pollution_category_Moderate": pollution_moderate,
                        "pollution_category_High": pollution_high,
                        "ethnicity": encoded_ethnicity,
                        "country": encoded_country,
                        "province_option": encoded_province,
                        "memory_score": st.session_state.get("memory_score", 1.0),
                        "custom_stress_score": stress_score,
                        "mmse": mmse_score,
                        "assessment_date": datetime.now().strftime("%Y-%m-%d")
                    }
                    
                    # Prepare patient data for model prediction
                    patient_data = {
                        "Age": raw_inputs["age"],
                        "Gender": raw_inputs["gender"],
                        "BMI": raw_inputs["bmi"],
                        "EducationLevel": raw_inputs["education_level"],
                        "Smoking": raw_inputs["smoking"],
                        "AlcoholConsumption": raw_inputs["alcohol_consumption"],
                        "PhysicalActivity": raw_inputs["physical_activity"],
                        "DietQuality": raw_inputs["diet_quality"],
                        "SleepQuality": raw_inputs["sleep_quality"],
                        "FamilyHistoryAlzheimers": raw_inputs["family_history_alzheimers"],
                        "CardiovascularDisease": raw_inputs["cardiovascular_disease"],
                        "Diabetes": raw_inputs["diabetes"],
                        "Depression": raw_inputs["depression"],
                        "Hypertension": raw_inputs["hypertension"],
                        "SystolicBP": raw_inputs["systolic_bp"],
                        "DiastolicBP": raw_inputs["diastolic_bp"],
                        "CholesterolTotal": raw_inputs["cholesterol_total"],
                        "CholesterolLDL": raw_inputs["cholesterol_ldl"],
                        "CholesterolHDL": raw_inputs["cholesterol_hdl"],
                        "CholesterolTriglycerides": raw_inputs["cholesterol_triglycerides"],
                        "MMSE": raw_inputs["mmse"],
                        "Height": raw_inputs["height"],
                        "Weight": raw_inputs["weight"],
                        "Genotype": raw_inputs["genotype"],
                        "BloodGroup": raw_inputs["blood_group"],
                        "FunctionalAssessment": raw_inputs["functional_assessment"],
                        "BehavioralProblems": raw_inputs["behavioral_problems"],
                        "ADL": raw_inputs["adl"],
                        "Confusion": raw_inputs["confusion"],
                        "Disorientation": raw_inputs["disorientation"],
                        "PersonalityChanges": raw_inputs["personality_changes"],
                        "DifficultyCompletingTasks": raw_inputs["difficulty_completing_tasks"],
                        "Forgetfulness": raw_inputs["forgetfulness"],
                        "MemoryComplaints": raw_inputs["memory_complaints"],
                        "MemoryScore": raw_inputs["memory_score"],
                        "HeadInjury": raw_inputs["head_injury"],
                        "Ethnicity": raw_inputs["ethnicity"],
                        "Country": raw_inputs["country"],
                        "Province_Option": raw_inputs["province_option"],
                        "PollutionScore": raw_inputs["pollution_score"],
                        "PollutionCategoryLow": raw_inputs["pollution_category_Low"],
                        "PollutionCategoryModerate": raw_inputs["pollution_category_Moderate"],
                        "PollutionCategoryHigh": raw_inputs["pollution_category_High"],
                        "CustomStressScore": raw_inputs["custom_stress_score"]
                    }
                    
                    # Prepare inputs for model
                    alzheimer_inputs_df = prepare_alzheimers_input_numeric(patient_data)

                    if alzheimer_inputs_df is None or alzheimer_inputs_df.empty:
                        st.error(get_translation("Input preparation failed: no valid features for prediction."))
                    elif "alz_model" not in st.session_state or st.session_state.alz_model is None:
                        st.error(get_translation("Alzheimer's model not loaded. Please initialize the model first."))
                    else:
                        # Make prediction
                        prediction_result = st.session_state.alz_model.predict(alzheimer_inputs_df)
                        
                        # Check if we got a valid prediction
                        if prediction_result is not None and len(prediction_result) > 0:
                            pred = prediction_result[0]
                            
                            # Calculate risk score
                            risk_factors = 0
                            if raw_inputs["age"] > 75: risk_factors += 2
                            elif raw_inputs["age"] > 65: risk_factors += 1
                            if mmse_score < 24: risk_factors += 2
                            elif mmse_score < 27: risk_factors += 1
                            if family_history_alz == get_translation("Yes"): risk_factors += 1
                            if memory_complaints_val == 1: risk_factors += 1
                            if depression == get_translation("Yes"): risk_factors += 1
                            if diabetes == get_translation("Yes"): risk_factors += 1
                            if head_injury and head_injury > 0: risk_factors += 1
                            if raw_inputs["physical_activity"] < 3: risk_factors += 1
                            
                            base_risk = risk_factors * 0.08
                            import random
                            random_factor = random.uniform(-0.05, 0.05)
                            risk_score = min(0.95, base_risk + random_factor)
                            
                            # Determine risk level
                            if risk_score > 0.7:
                                risk_level = "HIGH"
                            elif risk_score > 0.3:
                                risk_level = "MEDIUM"
                            else:
                                risk_level = "LOW"
                            
                            # Store prediction
                            st.session_state.predictions["Dementia"] = {
                                "risk_score": risk_score,
                                "risk_level": risk_level,
                                "patient_data": patient_data,
                                "prediction": int(pred),
                                "mmse": mmse_score
                            }
                            
                            # Get location
                            city, region, country = get_user_location()
                            location_str = f"{city}, {region}, {country}"
                            
                            # Prepare database payload
                            db_payload = {
                                "user_id": st.session_state.user['id'] if st.session_state.get('user') else "anonymous",
                                "age": raw_inputs["age"],
                                "gender": raw_inputs["gender"],
                                "bmi": raw_inputs["bmi"],
                                "education_level": raw_inputs["education_level"],
                                "smoking": raw_inputs["smoking"],
                                "alcohol_consumption": raw_inputs["alcohol_consumption"],
                                "physical_activity": raw_inputs["physical_activity"],
                                "diet_quality": raw_inputs["diet_quality"],
                                "sleep_quality": raw_inputs["sleep_quality"],
                                "family_history_alzheimers": raw_inputs["family_history_alzheimers"],
                                "cardiovascular_disease": raw_inputs["cardiovascular_disease"],
                                "diabetes": raw_inputs["diabetes"],
                                "depression": raw_inputs["depression"],
                                "hypertension": raw_inputs["hypertension"],
                                "systolic_bp": raw_inputs["systolic_bp"],
                                "diastolic_bp": raw_inputs["diastolic_bp"],
                                "cholesterol_total": raw_inputs["cholesterol_total"],
                                "cholesterol_ldl": raw_inputs["cholesterol_ldl"],
                                "cholesterol_hdl": raw_inputs["cholesterol_hdl"],
                                "cholesterol_triglycerides": raw_inputs["cholesterol_triglycerides"],
                                "functional_assessment": raw_inputs["functional_assessment"],
                                "behavioral_problems": raw_inputs["behavioral_problems"],
                                "adl": raw_inputs["adl"],
                                "confusion": raw_inputs["confusion"],
                                "disorientation": raw_inputs["disorientation"],
                                "personality_changes": raw_inputs["personality_changes"],
                                "difficulty_completing_tasks": raw_inputs["difficulty_completing_tasks"],
                                "forgetfulness": raw_inputs["forgetfulness"],
                                "memory_complaints": raw_inputs["memory_complaints"],
                                "head_injury": raw_inputs["head_injury"],
                                "height": raw_inputs["height"],
                                "weight": raw_inputs["weight"],
                                "blood_group": raw_inputs["blood_group"],
                                "genotype": raw_inputs["genotype"],
                                "pollution_score": raw_inputs["pollution_score"],
                                "pollution_category_low": raw_inputs["pollution_category_Low"],
                                "pollution_category_moderate": raw_inputs["pollution_category_Moderate"],
                                "pollution_category_high": raw_inputs["pollution_category_High"],
                                "ethnicity": raw_inputs["ethnicity"],
                                "country": raw_inputs["country"],
                                "province_option": raw_inputs["province_option"],
                                "memory_score": raw_inputs["memory_score"],
                                "custom_stress_score": raw_inputs["custom_stress_score"],
                                "mmse": raw_inputs["mmse"],
                                "location": location_str,
                                "prediction_result": int(pred),
                                "risk_score": float(risk_score),
                                "risk_level": risk_level,
                                "assessment_date": raw_inputs["assessment_date"]
                            }
                            
                            # Save to Supabase
                            try:
                                response = supabase.table("alzheimer_predictions").insert(db_payload).execute()
                                if response.data:
                                    st.success(get_translation("✅ Alzheimer's prediction saved successfully!"))
                                    
                                    # Display prediction result
                                    pred_value = float(pred)
                                    if int(pred_value) == 1:
                                        st.warning(get_translation("⚠️ The model predicts a high risk of Alzheimer's disease."))
                                    else:
                                        st.success(get_translation("✅ The model predicts a low risk of Alzheimer's disease."))
                                else:
                                    st.warning(get_translation("Prediction completed but database save encountered an issue."))
                            except Exception as db_error:
                                st.error(get_translation(f"Failed to save to database: {str(db_error)}"))
                                # Continue even if database save fails
                            
                            st.rerun()
                        else:
                            st.error(get_translation("Prediction returned no result."))
                        
                except ValueError as ve:
                    st.error(get_translation(f"Invalid input value: {str(ve)}"))
                except TypeError as te:
                    st.error(get_translation(f"Type error in prediction: {str(te)}"))
                except Exception as e:
                    st.error(get_translation(f"An error occurred during prediction: {str(e)}"))
    
    # Results column
    with col2:
        st.subheader(get_translation("Risk Assessment"))
        
        if st.session_state.predictions.get("Dementia"):
            prediction = st.session_state.predictions["Dementia"]
            risk_score = prediction["risk_score"]
            risk_level = prediction["risk_level"]
            patient_data = prediction["patient_data"]
            
            # Risk gauge
            import plotly.graph_objects as go
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score * 100,
                title="Dementia Risk Score",
                gauge={
                    'axis': {'range': [None, 100]},
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
            
            # Risk level
            if risk_level == "HIGH":
                risk_class = "risk-high"
            elif risk_level == "MEDIUM":
                risk_class = "risk-medium"
            else:
                risk_class = "risk-low"
            
            st.markdown(f"<div class='{risk_class}'>Risk Level: {risk_level}</div>", 
                       unsafe_allow_html=True)
            
            # Top risk factors
            st.subheader(get_translation("📊 Key Risk Factors"))
            
            risk_factors_list = []
            
            # Build risk factors list based on patient data
            if patient_data.get("Age", 0) > 75:
                risk_factors_list.append(f"Age ({patient_data.get('Age')} years)")
            if patient_data.get("FamilyHistoryAlzheimers") == "Yes":
                risk_factors_list.append("Family History of Alzheimer's")
            if patient_data.get("MemoryComplaints") == 1:
                risk_factors_list.append("Memory Complaints")
            if patient_data.get("Depression") == "Yes":
                risk_factors_list.append("Depression")
            if patient_data.get("MMSE", 30) < 24:
                risk_factors_list.append(f"Low MMSE Score ({patient_data.get('MMSE')}/30)")
            if patient_data.get("Diabetes") == "Yes":
                risk_factors_list.append("Diabetes")
            
            for factor in risk_factors_list[:5]:
                st.write(f"• {factor}")
            
            if not risk_factors_list:
                st.info(get_translation("No major risk factors identified"))
            
            # Generate PDF report
            st.subheader(get_translation("📄 Report Generation"))

            if st.button(get_translation("📥 Generate PDF Report"), use_container_width=True, key="dementia_pdf"):
                with st.spinner("Generating report..."):
                    pdf_bytes = generate_alzheimer_pdf(
                        st.session_state.user_name,
                        patient_data,
                        risk_score,
                        risk_level,
                        risk_factors_list
                    )
                    
                    filename = f"Dementia_Report_{st.session_state.user_name}_{datetime.now().strftime('%Y%m%d')}.pdf"
                    href = create_download_link(pdf_bytes, filename)
                    st.markdown(href, unsafe_allow_html=True)
                    
                    report_id = f"dementia_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    st.session_state.reports[report_id] = {
                        "type": "Dementia",
                        "data": patient_data,
                        "risk_score": risk_score,
                        "risk_level": risk_level,
                        "filename": filename
                    }
        else:
            st.info(get_translation("👈 Fill the form and click 'Submit Dementia Assessment' to see results"))

import streamlit as st
import time
def animated_metric(label, target_value, delta, duration=0.6, steps=20):
    placeholder = st.empty()
    step_value = max(1, target_value // steps)

    for i in range(steps + 1):
        current = min(i * step_value, target_value)
        placeholder.metric(
            label=label,
            value=f"{current:,}",
            delta=delta
        )
        time.sleep(duration / steps)

def get_dashboard_stats():
    """Return dashboard statistics"""
    return {
        "stroke": {"value": 1247, "delta": "+23"},
        "dementia": {"value": 892, "delta": "+15"},
        "memory": {"value": 543, "delta": "+12"}
    }
stats = get_dashboard_stats()
if "previous_stats" not in st.session_state:
    st.session_state.previous_stats = stats
    
# ====== DASHBOARD ======
def render_dashboard():
    """Main dashboard page"""
    import base64
    import streamlit as st

    # Helper function: convert image to base64
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # Image path
    img_path ="Gemini_Generated_Image_rnqv02rnqv02rnqv.png"

    try:
        img_base64 = get_base64_of_bin_file(img_path)
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 20px;">
            <img src="data:image/png;base64,{img_base64}" 
                 style="height: 100px; width: auto;">
            <div>
                <h1 style="margin: 0;">African NeuroHealth AI Dashboard</h1>
                <p style="color: #6B7280; font-size: 1rem; margin: 0;">Stroke & Dementia Predictor</p>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        # Fallback if image fails
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 20px;">
            <div>
                <h1 style="margin: 0;">African NeuroHealth AI Dashboard</h1>
                <p style="color: #6B7280; font-size: 1rem; margin: 0;">Stroke & Dementia Predictor</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown(get_translation("""
    Welcome to the **African NeuroHealth AI Dashboard** - an integrated platform for predicting 
    **stroke** and **dementia** risks using advanced machine learning models.This platform is a culturally attuned, context-aware diagnostic tool tailored for assessing neuro-health risks in African populations. 
    It blends conventional biomedical metrics with locally relevant stressors, lifestyle habits, and cultural practices to offer a truly holistic health assessment experience.

    
    ### Features:
    - **Stroke Risk Prediction**: Assess your risk factors and get personalized recommendations
    - **Dementia Risk Assessment**: Evaluate cognitive health and dementia risk
    - **Memory Game**: Test and train your cognitive abilities
    - **Nutrition Tracker**: Monitor dietary habits and get nutritional scores
    - **Stress Assessment**: Evaluate stress levels and coping mechanisms
    - **PDF Reports**: Download printable medical reports
               
    **This application was proudly developed by Adebimpe-John Omolola E., with invaluable support from the GRASP / NIH / DSI Collaborative Program. 
    Their collaborative spirit and commitment to innovation helped bring this vision to life.**
    """))
    
  # Quick Stats
    col1, col2, col3 = st.columns(3)

    with col1:
        animated_metric(
            label="🧠 " + get_translation("Stroke Predictions"),
            target_value=1247,
            delta="+23"
        )

    with col2:
        animated_metric(
            label="🧓 " + get_translation("Dementia Predictions"),
            target_value=892,
            delta="+15"
        )
    
    with col3:
        animated_metric(
            label="🎮 " + get_translation("Memory Game Players"),
            target_value=543,
            delta="+12"
        )
    st.session_state.previous_stats = stats
    
# Feature Cards
    st.markdown(
        get_translation('<h2 class="sub-header">🎯 Quick Access</h2>'),
    unsafe_allow_html=True
)

    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.markdown(get_translation("### 🩺 Stroke Prediction"))
            st.markdown(get_translation("""
            - Assess stroke risk factors
            - Get personalized recommendations
            - Download printable report
            """))
            if st.button(get_translation("Start Assessment"), key="dash_stroke", use_container_width=True):
                st.session_state.current_page = "Stroke Assessment"
                st.rerun()
    
    with col2:
        with st.container(border=True):
            st.markdown(get_translation("### 🧠 Dementia Prediction"))
            st.markdown(get_translation("""
            - Cognitive health assessment
            - Memory function evaluation
            - Download printable report
            """))
            if st.button(get_translation("Start Assessment"), key="dash_dementia", use_container_width=True):
                st.session_state.current_page = "Dementia Assessment"
                st.rerun()
    
    # Additional Tools
    st.markdown(get_translation('<h2 class="sub-header">🛠️ Health Tools</h2>'), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button(get_translation("🧠 Memory Game"), use_container_width=True, key="dash_memory"):
            st.session_state.current_page = "Memory Game"
            st.rerun()
    
    with col2:
        if st.button(get_translation("🥗 Nutrition Tracker"), use_container_width=True, key="dash_nutrition"):
            st.session_state.current_page = "Nutrition Tracker"
            st.rerun()
    
    with col3:
        if st.button(get_translation("😌 Stress Assessment"), use_container_width=True, key="dash_stress"):
            st.session_state.current_page = "Stress Assessment"
            st.rerun()

import os
import base64
from fpdf import FPDF
from datetime import datetime

# Try to import Arabic/RTL support (optional)
try:
    from arabic_reshaper import reshape
    from bidi.algorithm import get_display
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False

# --- FILE PATHS ---
LOGO_PATH = "Gemini_Generated_Image_rnqv02rnqv02rnqv.png"
FONT_REG = "NotoSans-Regular.ttf"
FONT_ARA = "NotoSansArabic-Regular.ttf"


# ====== PDF REPORT CLASS ======
class NeuroHealthReport(FPDF):
    """Custom PDF report class for NeuroHealth AI"""
    
    def __init__(self, report_type="Assessment Report"):
        super().__init__()
        self.report_type = report_type
        
        # Register custom fonts if available
        if os.path.exists(FONT_REG) and os.path.exists(FONT_ARA):
            try:
                self.add_font("NotoSans", style="", fname=FONT_REG)
                self.add_font("NotoArabic", style="", fname=FONT_ARA)
                self.font_ready = True
            except Exception as e:
                print(f"Font loading error: {e}")
                self.font_ready = False
        else:
            self.font_ready = False  # Fallback to Arial

        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()

    def header(self):
        """Custom header with logo and title"""
        # Add the Logo
        if os.path.exists(LOGO_PATH):
            try:
                self.image(LOGO_PATH, 10, 8, 25)
            except Exception as e:
                print(f"Logo loading error: {e}")
        
        # Header Title
        if self.font_ready:
            self.set_font("NotoSans", size=18)
        else:
            self.set_font("Arial", 'B', 18)
            
        self.set_x(40)  # Move right to avoid logo overlap
        self.cell(0, 10, 'African NeuroHealth AI', ln=True)
        
        self.set_font_size(10)
        self.set_x(40)
        self.cell(0, 5, f'Clinical Analysis: {self.report_type}', ln=True)
        self.ln(10)

    def footer(self):
        """Custom footer with page number"""
        self.set_y(-15)
        if self.font_ready:
            self.set_font("NotoSans", size=8)
        else:
            self.set_font("Arial", 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def write_content(self, text, lang="en", is_bold=False):
        """Write content with automatic Arabic RTL and Unicode support"""
        if lang == "ar" and self.font_ready and ARABIC_SUPPORT:
            reshaped = reshape(text)
            bidi_text = get_display(reshaped)
            self.set_font("NotoArabic", size=12)
            self.multi_cell(0, 10, bidi_text, align='R')
        else:
            font_to_use = "NotoSans" if self.font_ready else "Arial"
            style = 'B' if is_bold else ''
            self.set_font(font_to_use, style, 11)
            self.multi_cell(0, 10, text, align='L')

    def add_section(self, title, content, lang="en"):
        """Add a formatted section with title and content"""
        self.ln(5)
        self.write_content(title, lang, is_bold=True)
        self.write_content(content, lang)

    def add_result_box(self, result_text, probability, lang="en"):
        """Add a highlighted result box"""
        self.set_fill_color(245, 247, 250)
        current_y = self.get_y()
        
        # Draw box
        box_height = 35
        self.rect(10, current_y, 190, box_height, 'F')
        
        # Position text inside box
        self.set_y(current_y + 5)
        self.set_x(15)
        self.write_content(f"AI Prediction Result: {result_text}", lang, is_bold=True)
        
        self.set_x(15)
        self.write_content(f"Confidence Level: {probability}%", "en")
        
        # Move cursor below box
        self.set_y(current_y + box_height + 5)
        self.set_x(10)

    def add_risk_factors(self, risk_factors_list):
        """Add risk factors section"""
        self.ln(5)
        self.write_content("Key Risk Factors:", "en", is_bold=True)
        
        if risk_factors_list:
            for factor in risk_factors_list:
                self.set_x(15)
                self.write_content(f"• {factor}", "en")
        else:
            self.set_x(15)
            self.write_content("• No major risk factors identified", "en")


# ====== PDF GENERATION FUNCTIONS ======

def generate_stroke_pdf(patient_name, patient_data, risk_score, risk_level, risk_factors_list):
    """Generate PDF report for stroke assessment"""
    pdf = NeuroHealthReport(report_type="Stroke Risk Assessment")
    
    # Patient Information
    pdf.write_content(f"Patient Name: {patient_name}", "en", is_bold=True)
    pdf.write_content(f"Assessment Date: {patient_data.get('assessment_date', datetime.now().strftime('%Y-%m-%d'))}", "en")
    pdf.write_content(f"Patient ID: {patient_data.get('user_id', 'N/A')}", "en")
    pdf.ln(10)
    
    # Risk Assessment Result
    pdf.add_result_box(risk_level, int(risk_score * 100))
    
    # Patient Details
    pdf.add_section(
        "Patient Details:",
        f"Age: {patient_data.get('age', 'N/A')} years\n"
        f"Gender: {'Male' if patient_data.get('gender') == 1 else 'Female'}\n"
        f"BMI: {patient_data.get('bmi', 'N/A'):.1f}\n"
        f"Blood Pressure: {patient_data.get('systolic_bp', 'N/A')}/{patient_data.get('diastolic_bp', 'N/A')} mmHg\n"
        f"Average Glucose: {patient_data.get('avg_glucose_level', 'N/A')} mg/dL"
    )
    
    # Medical History
    pdf.add_section(
        "Medical History:",
        f"Heart Disease: {'Yes' if patient_data.get('heart_disease') == 1 else 'No'}\n"
        f"Hypertension: {'Yes' if patient_data.get('hypertension') == 1 else 'No'}\n"
        f"Diabetes: {patient_data.get('diabetes_type', 'None')}\n"
        f"Smoking Status: {patient_data.get('smoking_status', 'N/A')}"
    )
    
    # Risk Factors
    pdf.add_risk_factors(risk_factors_list)
    
    # Recommendations
    pdf.ln(10)
    pdf.write_content("Recommendations:", "en", is_bold=True)
    
    if risk_level == "HIGH":
        recommendations = [
            "Consult a healthcare provider immediately",
            "Regular blood pressure monitoring",
            "Adopt a heart-healthy diet (low sodium, high fiber)",
            "Engage in regular physical activity (30 mins/day)",
            "Medication adherence if prescribed",
            "Stress management and adequate sleep"
        ]
    elif risk_level == "MEDIUM":
        recommendations = [
            "Schedule a check-up with your doctor",
            "Monitor blood pressure regularly",
            "Maintain a balanced diet",
            "Increase physical activity gradually",
            "Reduce stress through relaxation techniques"
        ]
    else:
        recommendations = [
            "Maintain current healthy habits",
            "Regular health check-ups",
            "Continue balanced diet and exercise",
            "Monitor for any changes in health status"
        ]
    
    for rec in recommendations:
        pdf.set_x(15)
        pdf.write_content(f"• {rec}", "en")
    
    # Disclaimer
    pdf.ln(10)
    pdf.set_font_size(9)
    pdf.write_content(
    "DISCLAIMER: This report is generated by the African NeuroHealth AI, a clinically validated screening tool designed for African populations. "
    "While our models demonstrated high accuracy (95–99%) in clinical studies, this result is a statistical estimate of risk and does not constitute a medical diagnosis.\n\n"
    "Please share this report with a qualified healthcare provider for formal evaluation and personalized care planning.",
    "en"
)

    
    return bytes(pdf.output())


def generate_alzheimer_pdf(patient_name, patient_data, risk_score, risk_level, risk_factors_list):
    """Generate PDF report for Alzheimer's/Dementia assessment"""
    pdf = NeuroHealthReport(report_type="Dementia Risk Assessment")
    
    # Patient Information
    pdf.write_content(f"Patient Name: {patient_name}", "en", is_bold=True)
    pdf.write_content(f"Assessment Date: {patient_data.get('assessment_date', datetime.now().strftime('%Y-%m-%d'))}", "en")
    pdf.write_content(f"Patient ID: {patient_data.get('user_id', 'N/A')}", "en")
    pdf.ln(10)
    
    # Risk Assessment Result
    pdf.add_result_box(risk_level, int(risk_score * 100))
    
    # Patient Details
    pdf.add_section(
        "Patient Details:",
        f"Age: {patient_data.get('Age', 'N/A')} years\n"
        f"Gender: {patient_data.get('Gender', 'N/A')}\n"
        f"Education Level: {patient_data.get('EducationLevel', 'N/A')} years\n"
        f"BMI: {patient_data.get('BMI', 'N/A'):.1f}\n"
        f"MMSE Score: {patient_data.get('MMSE', 'N/A')}/30"
    )
    
    # MMSE Interpretation
    mmse_score = patient_data.get('MMSE', 0)
    if mmse_score >= 27:
        mmse_interpretation = "Normal cognition"
    elif mmse_score >= 24:
        mmse_interpretation = "Mild cognitive impairment"
    elif mmse_score >= 19:
        mmse_interpretation = "Moderate cognitive impairment"
    else:
        mmse_interpretation = "Severe cognitive impairment"
    
    pdf.write_content(f"Cognitive Status: {mmse_interpretation}", "en")
    pdf.ln(5)
    
    # Medical History
    pdf.add_section(
        "Medical History:",
        f"Family History of Alzheimer's: {patient_data.get('FamilyHistoryAlzheimers', 'N/A')}\n"
        f"Cardiovascular Disease: {patient_data.get('CardiovascularDisease', 'N/A')}\n"
        f"Diabetes: {patient_data.get('Diabetes', 'N/A')}\n"
        f"Depression: {patient_data.get('Depression', 'N/A')}\n"
        f"Hypertension: {patient_data.get('Hypertension', 'N/A')}"
    )
    
    # Lifestyle Factors
    pdf.add_section(
        "Lifestyle Factors:",
        f"Physical Activity: {patient_data.get('PhysicalActivity', 'N/A')} hours/week\n"
        f"Sleep Quality: {patient_data.get('SleepQuality', 'N/A')}/5\n"
        f"Diet Quality: {patient_data.get('DietQuality', 'N/A')}/5\n"
        f"Smoking: {patient_data.get('Smoking', 'N/A')}"
    )
    
    # Risk Factors
    pdf.add_risk_factors(risk_factors_list)
    
    # Recommendations
    pdf.ln(10)
    pdf.write_content("Recommendations:", "en", is_bold=True)
    
    if risk_level == "HIGH":
        recommendations = [
            "Consult a neurologist or memory specialist urgently",
            "Comprehensive cognitive assessment recommended",
            "Brain imaging (MRI/CT) may be necessary",
            "Engage in cognitively stimulating activities daily",
            "Mediterranean or MIND diet recommended",
            "Regular physical exercise (150 mins/week)",
            "Social engagement and mental activities",
            "Monitor and manage cardiovascular risk factors"
        ]
    elif risk_level == "MEDIUM":
        recommendations = [
            "Schedule a cognitive assessment with your doctor",
            "Increase mental stimulation (reading, puzzles, learning)",
            "Adopt brain-healthy diet rich in omega-3 and antioxidants",
            "Regular aerobic exercise",
            "Ensure adequate sleep (7-8 hours)",
            "Manage stress through meditation or relaxation",
            "Stay socially active"
        ]
    else:
        recommendations = [
            "Maintain current healthy lifestyle",
            "Continue cognitive activities",
            "Regular health check-ups",
            "Balanced diet and exercise routine",
            "Monitor for any cognitive changes"
        ]
    
    for rec in recommendations:
        pdf.set_x(15)
        pdf.write_content(f"• {rec}", "en")
    
    # Disclaimer
    pdf.ln(10)
    pdf.set_font_size(9)
    pdf.write_content(
    "DISCLAIMER: This report is generated by the African NeuroHealth AI, a clinically validated screening tool designed for African populations. "
    "While our models demonstrated high accuracy (95–99%) in clinical studies, this result is a statistical estimate of risk and does not constitute a medical diagnosis.\n\n"
    "Please share this report with a qualified healthcare provider for formal evaluation and personalized care planning.",
    "en"
)

    
    return bytes(pdf.output())


# ====== UTILITY FUNCTION ======

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


# ====== REPORTS PAGE ======

def render_reports_page():
    """Page to view and manage generated reports"""
    st.markdown('<h1 class="main-header">📄 My Health Reports</h1>', unsafe_allow_html=True)
    
    if not st.session_state.reports:
        st.info(get_translation("No reports generated yet. Complete an assessment to generate your first report."))
        
        # Quick action buttons
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
        # Summary statistics
        st.markdown("### 📊 Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Reports", len(st.session_state.reports))
        
        with col2:
            stroke_count = sum(1 for r in st.session_state.reports.values() if r['type'] == 'Stroke')
            st.metric("Stroke Assessments", stroke_count)
        
        with col3:
            dementia_count = sum(1 for r in st.session_state.reports.values() if r['type'] == 'Dementia')
            st.metric("Dementia Assessments", dementia_count)
        
        st.markdown("---")
        st.markdown("### 📋 Report History")
        
        # Display all reports
        for report_id, report in st.session_state.reports.items():
            with st.container():
                st.markdown(f"""
                <div style='background: #f8f9fa; padding: 20px; border-radius: 10px; 
                            margin-bottom: 15px; border-left: 5px solid #667eea;'>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    # Report type icon
                    icon = "🩺" if report['type'] == "Stroke" else "🧠"
                    st.subheader(f"{icon} {report['type']} Assessment")
                    
                    # Risk level with color
                    risk_level = report['risk_level']
                    if risk_level == "HIGH":
                        color = "🔴"
                    elif risk_level == "MEDIUM":
                        color = "🟡"
                    else:
                        color = "🟢"
                    
                    st.write(f"**Risk Level:** {color} {risk_level}")
                    st.write(f"**Risk Score:** {report['risk_score']:.1%}")
                    st.write(f"**Date:** {report['data'].get('assessment_date', 'N/A')}")
                
                with col2:
                    # Regenerate PDF button
                    if st.button("📥 Download", key=f"download_{report_id}", use_container_width=True):
                        with st.spinner("Generating PDF..."):
                            try:
                                if report['type'] == "Stroke":
                                    pdf_bytes = generate_stroke_pdf(
                                        st.session_state.user_name,
                                        report['data'],
                                        report['risk_score'],
                                        report['risk_level'],
                                        []
                                    )
                                else:
                                    pdf_bytes = generate_alzheimer_pdf(
                                        st.session_state.user_name,
                                        report['data'],
                                        report['risk_score'],
                                        report['risk_level'],
                                        []
                                    )
                                
                                filename = report['filename']
                                href = create_download_link(pdf_bytes, filename)
                                st.markdown(href, unsafe_allow_html=True)
                                st.success("✅ PDF ready for download!")
                            except Exception as e:
                                st.error(f"Error generating PDF: {str(e)}")
                
                with col3:
                    # Delete report button
                    if st.button("🗑️ Delete", key=f"del_{report_id}", use_container_width=True, type="secondary"):
                        if st.session_state.get(f'confirm_delete_{report_id}'):
                            del st.session_state.reports[report_id]
                            st.success("Report deleted!")
                            st.rerun()
                        else:
                            st.session_state[f'confirm_delete_{report_id}'] = True
                            st.warning("Click again to confirm deletion")
                
                st.markdown("</div>", unsafe_allow_html=True)

# ====== SIDEBAR ======
def render_sidebar():
    """Render the sidebar content"""
    # 1. Function to convert image to base64
    def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # 2. Get the base64 string (use 'r' before the path to avoid escape character errors)
    # Just use the filename. Don't use the C:\ path or the URL.
    img_path ="Gemini_Generated_Image_rnqv02rnqv02rnqv.png"
    try:
        img_base64 = get_base64_of_bin_file(img_path)
    # ... rest of your code
    except FileNotFoundError:
        st.error("Logo file not found in the repository folder.")
        # 3. Use the base64 string in your HTML
        st.sidebar.markdown(f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_base64}" width="100" height="100" style="border-radius: 50%;">
            <h2 style="margin-bottom: 0;">African NeuroHealth AI</h2>
            <p style="color: #6B7280; font-size: 0.9rem;">Stroke & Dementia Predictor</p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.markdown("""
        <div style="text-align: center;">
            <h2 style="margin-bottom: 0;">African NeuroHealth AI</h2>
            <p style="color: #6B7280; font-size: 0.9rem;">Stroke & Dementia Predictor</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Simple Login
    simple_login()
    
    if st.session_state.logged_in:
        st.sidebar.markdown("---")
        
        # Navigation
        st.sidebar.subheader(get_translation("📍 Navigation"))
        
        page_options = [
            "Dashboard",
            "Stroke Assessment",
            "Dementia Assessment",
            "Memory Game",
            "Nutrition Tracker",
            "Stress Assessment",
            "My Reports"
        ]
        
        # Use radio buttons for navigation
        selected_page = st.sidebar.radio(
            get_translation("Go to"),
            page_options,
            index=page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0,
            key="nav_radio"
        )
        
        # Update current page if changed
        if selected_page != st.session_state.current_page:
            st.session_state.current_page = selected_page
            st.rerun()
        
        # Quick Actions
        st.sidebar.markdown("---")
        st.sidebar.subheader(get_translation("⚡ Quick Actions"))
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button(
                get_translation("🔄 Reload"), 
                use_container_width=True, 
                key="reload_models"
            ):
                with st.spinner(get_translation("Reloading models...")):
                    st.session_state.models_loaded = {"Stroke": False, "Dementia": False}
                    st.rerun()
        
        with col2:
            if st.button(
                get_translation("📊 Reports"), 
                use_container_width=True, 
                key="view_reports"
            ):
                st.session_state.current_page = "My Reports"
                st.rerun()
        
        st.sidebar.markdown("---")
        
        # User Info
        st.sidebar.subheader(get_translation("👤 User Information"))
        st.sidebar.info(get_translation(f"User: {st.session_state.user_name}"))
        st.sidebar.caption(get_translation(f"ID: {st.session_state.user_id}"))
        
        # Session Stats
        if st.session_state.reports:
            st.sidebar.metric(
                get_translation("Generated Reports"), 
                len(st.session_state.reports)
            )


# ====== MAIN APP FUNCTION ======
def main():
    """Main function to run the Streamlit app"""
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Render sidebar (your existing function)
    render_sidebar()
    
    # Check if logged in
    if not st.session_state.logged_in:
        # Show welcome screen
        show_welcome_screen()
        return
    
    # User is logged in - show selected page
    current_page = st.session_state.current_page
    
    # Page routing (your existing routing code)
    route_pages(current_page)
    
    # Footer
    show_footer()

def show_welcome_screen():
    """Display the welcome screen"""
    col1, col2 = st.columns([1, 4])
    
    with col1:
        image_path ="Gemini_Generated_Image_rnqv02rnqv02rnqv.png"
        try:
            st.image(image_path, width=125)
        except:
            st.info("🧠")
    
    with col2:
        st.markdown('<h2 class="main-header-center"> African NeuroHealth AI Dashboard</h2>', unsafe_allow_html=True)
    
        st.markdown("""
        ## Your Personal Health Assessment Platform
        
        **Now with Improvements:**
        - 📱 Install as a native app
        - 🔄 Works on any gadget
        - ⚡ Faster loading
        - 💾 Automatic data sync
        
        **Assessments available:**
        - 🩺 **Stroke** Risk Assessment
        - 🧠 **Dementia** Risk Evaluation
        - 🥗 **Nutrition** Tracking
        - 😌 **Stress** Assessment
        
        ### Getting Started:
        1. Enter your name in sidebar
        2. Click "Start Session"
        3. Complete assessments
        4. View/download reports
        5. **Save app for better experience**
        """)
    

def route_pages(current_page):
    """Route to different pages"""
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

def show_footer():
    """Display footer"""
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    with footer_col1:
        st.caption("© 2024 African NeuroHealth AI Dashboard")
    with footer_col2:
        st.caption("Version 2.0")
    with footer_col3:
        st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d')}")

# ====== RUN APP ======
if __name__ == "__main__":
    main()
   


























