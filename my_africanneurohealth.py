import streamlit as st

# Must be the very first Streamlit command
import streamlit as st

# --- PAGE CONFIG (safe for mobile + desktop) ---
st.set_page_config(
    page_title="AFRICAN NEUROHEALTH",
    page_icon="📊",
    layout="centered",   # ✅ works on both desktop & mobile
)

# --- CUSTOM RESPONSIVE STYLING ---
st.markdown(
    """
    <style>
    /* Desktop: mimic wide layout */
    @media (min-width: 900px) {
        .block-container {
            max-width: 95% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
    }

    /* Mobile: make sure expanders are fully visible */
    @media (max-width: 899px) {
        .block-container {
            max-width: 100% !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        .streamlit-expanderHeader {
            font-size: 1.1rem !important;  /* Bigger expander text on mobile */
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)
permanent_sidebar = """
    <style>
    section[data-testid="stSidebar"] {
        min-width: 250px !important;  /* force width */
        max-width: 250px !important;
    }
    button[kind="header"] {
        display: none !important;  /* hide expander toggle */
    }
    </style>
"""
st.markdown(permanent_sidebar, unsafe_allow_html=True)


import pandas as pd
import numpy as np
import joblib
import os
import random
import time
from dotenv import load_dotenv
import requests
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cloudpickle
import math
from uuid import UUID
import json
import jsonschema
import shap
import sqlite3
import logging
from postgrest import APIError
import pickle
from datetime import datetime
import traceback
from sklearn.pipeline import Pipeline
from supabase import create_client, Client

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

# ----------------------------
# LOGIN FUNCTION
# ----------------------------
def login():
    st.subheader("Login with Email & Password")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login", key="login_btn"):
        try:
            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
            if response.user:
                st.session_state.user = {"id": response.user.id, "email": response.user.email}
                st.success(f"Logged in as {st.session_state.user['email']}")
            else:
                st.error("Invalid login credentials")
        except Exception as e:
            st.error(f"Login error: {e}")


# ----------------------------
# Handle OAuth callback
# ----------------------------
query_params = st.query_params
if "access_token" in query_params:
    try:
        user_session = supabase.auth.get_user()
        if user_session.user:
            st.session_state.user = {"id": user_session.user.id, "email": user_session.user.email}
            st.success(f"Welcome, {st.session_state.user['email']}!")
    except Exception as e:
        st.error(f"OAuth login error: {e}")

# ----------------------------
# LOGOUT FUNCTION
# ----------------------------
def logout():
    try:
        supabase.auth.sign_out()
        # Always reset to dict, never None
        st.session_state.user = {"id": None, "email": None}
        st.success("Logged out successfully.")
        st.experimental_rerun()
    except Exception as e:
        st.error(f"Logout error: {e}")

# ----------------------------
# REGISTER FUNCTION
# ----------------------------
def register():
    st.subheader("Register")
    email = st.text_input("New Email", key="register_email")
    password = st.text_input("New Password", type="password", key="register_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")

    if st.button("Register", key="register_btn"):
        if password != confirm_password:
            st.error("Passwords do not match")
        else:
            try:
                response = supabase.auth.sign_up({"email": email, "password": password})
                if response.user:
                    st.success("Registration successful! Please check your email to confirm your account.")
                else:
                    st.error("Registration failed.")
            except Exception as e:
                st.error(f"Registration error: {e}")
# SESSION MANAGEMENT
# ----------------------------
if "user" not in st.session_state:
    st.session_state.user = {"id": None, "email": None}
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = None

# ----------------------------
# RESTORE SESSION IF TOKENS EXIST
# ----------------------------
if st.session_state.access_token and st.session_state.refresh_token:
    try:
        supabase.auth.set_session({
            "access_token": st.session_state.access_token,
            "refresh_token": st.session_state.refresh_token,
        })
        user = supabase.auth.get_user()
        if user.user:
            st.session_state.user = {"id": user.user.id, "email": user.user.email}
    except Exception:
        st.session_state.user = {"id": None, "email": None}
        st.session_state.access_token = None
        st.session_state.refresh_token = None
        
st.success("✅ Welcome Take a Moment to Know About The African Neurohealth Dashboard")

if "user" in st.session_state and st.session_state.user:
    logged_in = True
else:
    logged_in = False

st.markdown("### User Access")
if st.session_state.user.get("id"):
    st.write(f"✅ Logged in as: **{st.session_state.user['email']}**")
    if st.button("Logout"):
        logout()
else:
    tab1, tab2 = st.tabs(["🔑 Login", "🆕 Register"])
    with tab1:
        login()
    with tab2:
        register()
def custom_stress_score(prefix="", use_container=False):
    """Calculate stress score with option to avoid nested expanders"""
    title = f"🧮 {prefix} Stress Estimator Based on Cultural & Contextual Stress Factors" 
    # Create either a container or expander based on context
    if use_container:
        container = st.container()
        container.header(title)
    else:
        container = st.expander(title)
    
    with container:
        q1 = st.slider("Financial pressure/burden", 0, 4, 2, 
                      help="Struggling with basic needs, debts, or unemployment")
        q2 = st.slider("Family/relationship issues", 0, 4, 2,
                      help="Marital conflicts, caring for extended family, generational conflicts")
        q3 = st.slider("Work/employment stress", 0, 4, 2,
                      help="Job insecurity, long commutes, workplace discrimination")
        q4 = st.slider("Community safety concerns", 0, 4, 2,
                      help="Crime, political instability, or ethnic tensions in your area")
        q5 = st.slider("Caregiver burden", 0, 4, 2,
                      help="Caring for children/elderly with limited support")
        q6 = st.slider("Migration/displacement stress", 0, 4, 2,
                      help="Relocation challenges, missing homeland, adapting to new culture")
        q7 = st.slider("Traditional family expectations", 0, 4, 2,
                      help="Pressure to uphold cultural traditions, marriage expectations")
        q8 = st.slider("Spiritual/religious conflicts", 0, 4, 2,
                      help="Tension between traditional beliefs and modern life")
        
        total_score = q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8
        
        if total_score <= 12:
            level = 0
            label = "Low"
            color = "green"
        elif total_score <= 20:
            level = 1
            label = "Moderate"
            color = "orange"
        else:
            level = 2
            label = "High"
            color = "red"
 
        st.markdown(f"""
        <div style='padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-top: 20px;'>
            <h4>🧠 Total Stress Score: <span style='color:{color};'>{total_score}/32</span> → {label} Stress</h4>
            <p><small>Higher scores indicate greater exposure to Africa-specific stressors</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        return level, label, total_score

def smart_load_model(path):
    """
    Tries to load a model using joblib first, then falls back to cloudpickle.
    Works for both .joblib and .pkl files.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        # Try joblib first (common for scikit-learn)
        return joblib.load(path)
    except (AttributeError, EOFError, ImportError, pickle.UnpicklingError):
        # If joblib fails, try cloudpickle
        with open(path, "rb") as f:
            return cloudpickle.load(f)

def smart_load_model(path):
    """
    Tries to load a model using joblib first, then falls back to cloudpickle.
    Works for both .joblib and .pkl files.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        return joblib.load(path)
    except Exception:
        # If joblib fails (version mismatch, missing class, etc.), try cloudpickle
        with open(path, "rb") as f:
            return cloudpickle.load(f)

# Get the current directory
current_dir = Path(__file__).resolve().parent

# Define model paths using relative paths
ALZ_MODEL_PATH = current_dir / "alzheimers_pipeline.joblib"
STROKE_MODEL_PATH = current_dir / "stroke_pipeline.joblib"
ALZ_PREPROCESSOR_PATH = current_dir / "alzheimers_preprocessor.joblib"

# Function to load models with error handling
@st.cache_resource
def load_models():
    try:
        # Check if files exist
        if not ALZ_MODEL_PATH.exists():
            st.error(f"Alzheimer's model file not found at {ALZ_MODEL_PATH}")
            return None, None, None
            
        if not STROKE_MODEL_PATH.exists():
            st.error(f"Stroke model file not found at {STROKE_MODEL_PATH}")
            return None, None, None
            
        if not ALZ_PREPROCESSOR_PATH.exists():
            st.error(f"Preprocessor file not found at {ALZ_PREPROCESSOR_PATH}")
            return None, None, None
            
        # Load the models
        alz_model = joblib.load(ALZ_MODEL_PATH)
        stroke_model = joblib.load(STROKE_MODEL_PATH)
        preprocessor = joblib.load(ALZ_PREPROCESSOR_PATH)
        return alz_model, stroke_model, preprocessor
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

    # Load models into session state once
if "stroke_model" not in st.session_state or "alz_model" not in st.session_state:
    alz_model, stroke_model, preprocessor = load_models()
    
    if alz_model and stroke_model and preprocessor:
        st.session_state.alz_model = alz_model
        st.session_state.stroke_model = stroke_model
        st.session_state.preprocessor = preprocessor
    else:
        st.stop()  # Prevent app from running without models



def save_prediction_to_supabase(features, prediction, probability, memory_score=None):
    try:
        # Prepare data for insertion
        data = {
            "age": int(features['Age'].iloc[0]) if not pd.isna(features['Age'].iloc[0]) else None,
            "gender": int(features['Gender'].iloc[0]) if not pd.isna(features['Gender'].iloc[0]) else None,
            "education_level": int(features['EducationLevel'].iloc[0]) if not pd.isna(features['EducationLevel'].iloc[0]) else None,
            # Add all other features...
            "prediction": int(prediction[0]),
            "probability": float(probability[0][1]),
        }
        
        # Only add memory_score if it's provided
        if memory_score is not None:
            data["memory_score"] = int(memory_score)
        
        # Insert into Supabase
        response = supabase.table("alzheimers_predictions").insert(data).execute()
        
        logger.info("Prediction saved successfully to Supabase!")
        return response
        
    except Exception as e:  # This defines 'e' in the except block
        logger.error(f"Error saving to Supabase: {e}")
        logger.error(traceback.format_exc())  # This will show the full traceback
        
        # If the error is about a missing column, try without the memory_score
        if "memory_score" in str(e):
            logger.warning("memory_score column might not exist, retrying without it...")
            if "memory_score" in data:
                del data["memory_score"]
                try:
                    response = supabase.table("alzheimers_predictions").insert(data).execute()
                    logger.info("Prediction saved successfully without memory_score!")
                    return response
                except Exception as inner_e:
                    logger.error(f"Error saving without memory_score: {inner_e}")
                    raise inner_e
        
        # Re-raise the original exception if we can't handle it
        raise e

# Function to make predictions and handle errors properly
def make_prediction(input_data):
    try:
                # Load the pipeline
        pipeline = alz_model  # Assuming alz_model is already loaded as a Pipeline
        
        # Make prediction
        prediction = pipeline.predict(input_data)
        probability = pipeline.predict_proba(input_data)
        
        # Save to Supabase
        result = save_prediction_to_supabase(input_data, prediction, probability)
        
        return prediction, probability, result
        
    except Exception as e:  # This defines 'e' in the function scope
        logger.error(f"Error during prediction: {e}")
        logger.error(traceback.format_exc())
        
        # Re-raise or handle as needed
        raise



# ======================
# Build stroke input dictionary from Streamlit widgets
# ======================
def build_stroke_input_from_ui(ui_input: dict) -> dict:
    """
    Converts Streamlit inputs into a numeric dictionary ready for prediction.
    Handles categorical mappings and defaults.
    """
    input_dict = {}
    
    # Map numeric fields
    for key, default in NUMERIC_DEFAULTS.items():
        input_dict[key] = ui_input.get(key, default)
    
    # Map categorical fields
    for key, mapping in CATEGORY_MAPS.items():
        value = ui_input.get(key, "None")
        # Handle unseen categories
        input_dict[key] = mapping.get(value, mapping.get("None", 0))
    
    return input_dict

# Validate input
def prepare_stroke_input(raw_input) -> pd.DataFrame:
    if isinstance(raw_input, pd.DataFrame):
        if len(raw_input) != 1:
            raise ValueError("Expected a single-row DataFrame.")
        return raw_input.reset_index(drop=True)
    elif isinstance(raw_input, dict):
        return pd.DataFrame([raw_input])
    elif isinstance(raw_input, list) and len(raw_input) == 1 and isinstance(raw_input[0], dict):
        return pd.DataFrame(raw_input)
    else:
        raise TypeError("Input must be a dict, list of dicts, or single-row DataFrame.")

# Predict stroke risk
def predict_stroke(raw: dict) -> int:
    """
    Returns 0 (no stroke) or 1 (stroke risk)
    """
    # 1. Build full input
    full_input_df = build_full_input(raw)

    # 2. Validate
    stroke_data_df = prepare_stroke_input(full_input_df)

    # 3. Predict using pipeline (handles categorical encoding)
    pred = stroke_model.predict(stroke_data_df)[0]

    return int(pred)


# ----------------------------
# ABOUT FUNCTION
# ----------------------------
def about():
    st.header("ℹ️ About This App")
    st.title("🧠 African NeuroHealth Dashboard")
    st.markdown("""
This platform is a culturally attuned, context-aware diagnostic tool tailored for assessing neuro-health risks in African populations. 
It blends conventional biomedical metrics with locally relevant stressors, lifestyle habits, and cultural practices to offer a truly holistic health assessment experience.

**Key Features:**
- Environmental exposures (e.g., noise, air pollution)
- Dietary patterns (including traditional nutrition)
- Sleep quality and hydration
- Use of herbal or traditional remedies
- Psychosocial stressors unique to African settings
- Ethnocultural identity tracking for precision health insights

**This application was proudly developed by Adebimpe-John Omolola E., 
with invaluable support from the GRASP / NIH / DSI Collaborative Program. 
Their collaborative spirit and commitment to innovation helped bring this vision to life.**



    
# -------------------
# Initialize session state
# -------------------
session_keys = [
    "user", "stroke_data", "alz_data", "nutritional_data",
    "default_lifestyles", "memory_game", "auth_mode"
]
for key in session_keys:
    if key not in st.session_state:
        st.session_state[key] = {} if key.endswith("_data") else None

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "login"  # login or signup

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
    st.header("🌍 Location Information")
    selected_country = st.selectbox("Select Country", list(countries_with_provinces.keys()))
    selected_province = st.selectbox("Select Province", countries_with_provinces[selected_country])
    selected_region = st.selectbox("🌍 Select Region", list(region_with_ethnicity.keys()))
    selected_ethnicity = st.selectbox("Select Ethnicity", region_with_ethnicity[selected_region])
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

def nutrition_tracker_app():
    st.header("Nutrition Tracker")
    st.title("🥗 Nutrition Tracker")
# --- Nutritional Lifestyle Tracker ---
def calculate_weekly_servings(freq, servings):
    if freq == "Daily":
        return servings * 7
    elif freq == "Weekly":
        return servings
    elif freq == "Monthly":
        return servings / 4
    return 0

def compute_nutritional_score():
    if not st.session_state.nutritional_data:
        return 3
    
    positive = ["Homemade Food", "Vegetarian", "Vegan", "Mediterranean", "Pescatarian"]
    negative = ["Junk Food", "Fast Foods"]
    
    positive_score = sum(
        data["weekly_servings"] * 0.5 
        for lifestyle, data in st.session_state.nutritional_data.items() 
        if lifestyle in positive
    )
    
    negative_score = sum(
        data["weekly_servings"] * 1.0 
        for lifestyle, data in st.session_state.nutritional_data.items() 
        if lifestyle in negative
    )
    
    raw_score = 3 + (positive_score / 10) - (negative_score / 5)
    return max(1, min(5, round(raw_score)))

st.sidebar.header("🍽️ Nutritional Lifestyle Tracker")
st.sidebar.header("Additional Nutrition Details")

fruit_intake = st.sidebar.number_input(
    "Fruit Intake (servings per day)", min_value=0, max_value=20, value=2, key="fruit_intake"
)

vegetable_intake = st.sidebar.number_input(
    "Vegetable Intake (servings per day)", min_value=0, max_value=20, value=3, key="vegetable_intake"
)

hydration_liters = st.sidebar.number_input(
    "Water Intake (liters per day)", min_value=0.0, max_value=10.0, value=2.0, key="hydration_liters"
)

supplements_used = st.sidebar.text_input(
    "Supplements Used (e.g., Vitamin D, Omega-3)", key="supplements_used"
)

natural_herbs = st.sidebar.text_input(
    "Natural Herbs Taken (e.g., Ginger, Turmeric)", key="natural_herbs"
)

# Available options
all_options = [
    "Local Bukka/Street Food", "Homemade Food", "Junk Food", 
    "Fast Foods", "Vegetarian", "Vegan", "Pescatarian", 
    "Mediterranean", "Keto", "Paleo"
]

# Lifestyle selection
selected_lifestyles = st.sidebar.multiselect(
    "Select Nutritional Lifestyles",
    all_options,
    default=st.session_state.default_lifestyles,
    key="nutritional_lifestyle"
)
st.session_state.default_lifestyles = selected_lifestyles

# Process each selected lifestyle
if selected_lifestyles:
    with st.sidebar.expander("Track Consumption", expanded=True):
        for lifestyle in selected_lifestyles:
            st.subheader(lifestyle)
            col1, col2 = st.columns(2)
            
            with col1:
                freq = st.selectbox(
                    "Frequency",
                    ["Daily", "Weekly", "Monthly"],
                    key=f"freq_{lifestyle}"
                )
            
            with col2:
                freq_label = "day" if freq == "Daily" else "week" if freq == "Weekly" else "month"
                servings = st.number_input(
                    f"Servings per {freq_label}:",
                    min_value=1,
                    max_value=100,
                    value=1,
                    key=f"servings_{lifestyle}"
                )
            
            # Update session state
            weekly = calculate_weekly_servings(freq, servings)
            st.session_state.nutritional_data[lifestyle] = {
                "frequency": freq,
                "servings": servings,
                "weekly_servings": weekly
            }

# Display score after processing inputs
if st.session_state.nutritional_data:
    nutritional_score = compute_nutritional_score()
    st.sidebar.info(f"🍎 Nutritional Health Score: **{nutritional_score}/5**")         

# --- Save Functionality ---
if st.sidebar.button("Save Nutritional Data"):
    if st.session_state.user is None:
        st.sidebar.warning("Please log in to save nutritional data")
    elif not st.session_state.nutritional_data:
        st.sidebar.warning("No nutritional data to save")
    else:
        try:
            nutrition_data = {
                "user_id": st.session_state.user['id'] if st.session_state.get('user') else "anonymous",
                "fruit_intake": fruit_intake,
                "vegetable_intake": vegetable_intake,
                "hydration_liters": hydration_liters,
                "supplements_used": supplements_used,
                "natural_herbs": natural_herbs,
                "lifestyle_choices": ", ".join(st.session_state.nutritional_data.keys()),  # or JSON if preferred
                "nutritional_score": compute_nutritional_score()
            }

            response = supabase.table("nutrition_tracker").insert(nutrition_data).execute()
            if response.data:
                st.success("Nutrition data saved!")
            else:
                st.error(f"Failed to save nutrition data: {response.error}")

        except Exception as e:
            st.error(f"Error saving nutrition data: {e}")
            st.stop()

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
        
# -------------------
# TAB 1: Stroke Prediction
# -------------------  

def map_salt_intake(val):
    keys = ['salt_intake_High', 'salt_intake_Moderate', 'salt_intake_Little', 'salt_intake_None']
    values = [0]*4
    val = val.lower()
    if 'high' in val:
        values[0] = 1
    elif 'moderate' in val:
        values[1] = 1
    elif 'little' in val or 'low' in val:
        values[2] = 1
    else:
        values[3] = 1
    return dict(zip(keys, values))

def map_noise_source(val):
    keys = ['noise_sources_Block-Industry', 'noise_sources_Church', 'noise_sources_Club-House',
            'noise_sources_Generator', 'noise_sources_Grinding-Machine', 'noise_sources_Market',
            'noise_sources_Mosque', 'noise_sources_None', 'noise_sources_Welder']
    values = [0]*9
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
    return dict(zip(keys, values))

# Add other mapping functions similarly if needed (nutritional_lifestyle, hypertension_treatment, chronic_pain)...

def prepare_stroke_input_numeric(raw_input):
    numeric_features = ['age', 'avg_glucose_level', 'bmi', 'stress_level',
                        'ptsd', 'depression_level', 'diabetes_type', 'sleep_hours', 'height', 'weight']

    categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type',
                            'smoking_status', 'chronic_pain_None', 'blood_group', 'genotype', 'chronic_pain_Osteoarthritis',
                            'chronic_pain_Others']
    
    boolean_features = ['chronic_pain_Rheumatism', 'salt_intake_High', 'salt_intake_Little',
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
        try:
            final_input[col] = float(val)
        except:
            final_input[col] = 0

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

def build_full_input(raw):
    full_input = {}

    # Direct numeric and categorical
    direct_cols = ['age', 'avg_glucose_level', 'bmi', 'stress_level', 'ptsd', 'depression_level',
                   'diabetes_type', 'sleep_hours', 'gender', 'ever_married', 'work_type',
                   'residence_type', 'smoking_status', 'ethnicity', 'Country', 'Province_Option']
    for col in direct_cols:
        full_input[col] = raw.get(col, 0 if col in ['age','avg_glucose_level','bmi','stress_level',
                                                    'ptsd','depression_level','diabetes_type','sleep_hours','systolic_bp', 'diastolic_bp'] else 'None')

    # Map salt intake & noise sources
    full_input.update(map_salt_intake(raw.get('salt_intake', 'None')))
    full_input.update(map_noise_source(raw.get('noise_sources', 'None')))
# Gender encoding
    full_input['gender'] = 1 if raw_inputs.get('gender') == "Male" else 0

    stress_map = {"None": 0, "Low": 1, "Moderate": 2, "High": 3}
    stress_raw = raw_inputs.get("stress_level", "None")
    full_input["stress_level"] = stress_map.get(stress_raw, 0)


              #chronic pain#
    pain_map = {"None": 0, "Rheumatism": 1, "Osteoarthritis": 2, "Others": 3}
    pain_raw = raw_inputs.get("chronic_pain", "None")
    full_input["chronic_pain"] = pain_map.get(pain_raw, 0)

    # Hypertension treatment
    treatment_map = {"None": 0, "Herbal": 1, "Drugs": 2}
    treatment_raw = raw_inputs.get("hypertension_treatment", "None")
    full_input["hypertension_treatment"] = treatment_map.get(treatment_raw, 0)       

       # Convert categorical columns to string
    numeric_cols = [
        'age', 'avg_glucose_level', 'bmi', 'stress_level',
        'ptsd', 'depression_level', 'sleep_hours', 'systolic_bp', 'diastolic_bp', 'height', 'weight'
    ]
    for col in numeric_cols:
        try:
            full_input[col] = float(raw.get(col, 0))
        except:
            full_input[col] = 0.0

    # ---- Mapped/encoded categorical values ----

    diabetes_map = {"None": 0, "Type 1": 1, "Type 2": 2}
    full_input["diabetes_type"] = diabetes_map.get(raw.get("diabetes_type", "None"), 0)

    # ---- Categorical values (raw for OneHotEncoder) ----
    categorical_cols = [
        "marital_status", "work_type", "Residence_type", "smoking_status",'blood_group', 'genotype'
        "salt_intake", "nutritional_lifestyle", "ethnicity", "Country", "Province_Option"
    ]
    for col in categorical_cols:
        full_input[col] = str(raw.get(col, "None"))

    # ---- Binary/bool checkboxes ----
    boolean_feats = [
        'chronic_pain_Rheumatism', 'hypertension_treatment_Drugs',
        'hypertension_treatment_Herbal', 'hypertension_treatment_None',
        'nutritional_lifestyle_Fast Foods', 'nutritional_lifestyle_Homemade Food',
        'nutritional_lifestyle_Junk Food', 'nutritional_lifestyle_Local Bukka/Street Food',
        'noise_sources_Block-Industry', 'noise_sources_Church',
        'noise_sources_Club-House', 'noise_sources_Generator',
        'noise_sources_Grinding-Machine', 'noise_sources_Market',
        'noise_sources_Mosque', 'noise_sources_None', 'noise_sources_Welder'
    ]
    for bf in boolean_feats:
        full_input[bf] = int(raw.get(bf, 0))

    # ---- Custom stress score (optional) ----
    full_input["CustomStressScore"] = float(raw.get("CustomStressScore", 0))
    
       # 4️⃣ Create DataFrame
    stroke_inputs_df = pd.DataFrame([full_input])

    # 5️⃣ Ensure all expected columns exist
    for col in expected_columns:
        if col not in stroke_inputs_df.columns:
            stroke_inputs_df[col] = 0

    stroke_inputs_df = stroke_inputs_df[expected_columns]

    return stroke_inputs_df

          # Collect raw inputs
    # Utility functions to safely convert inputs
def safe_int(val, default=None):
    try:
        if val is None or val == "None" or val == "":
            return default
        return int(val)
    except (ValueError, TypeError):
        return default

def safe_float(val, default=None):
    try:
        if val is None or val == "None" or val == "":
            return default
        return float(val)
    except (ValueError, TypeError):
        return default

def parameter_select(label, min_val, max_val, step=1, key=None):
    """
    Creates a selectbox input for a parameter.
    
    Args:
        label (str): The label to display.
        min_val (int): Minimum value.
        max_val (int): Maximum value.
        step (int): Step size.
        key (str): Optional key for Streamlit session state.
    
    Returns:
        int: Selected value as integer.
    """
    options = list(range(min_val, max_val + 1, step))
    return st.selectbox(label, options, key=key)

# =======================
# TAB 1: STROKE PREDICTION
# =======================
# ---------- REQUIRED INPUT HELPERS ----------
def required_text_input(label, key):
    value = st.text_input(label, key=key)
    return value if value.strip() else None

def required_number_input(label, key, min_value=None, max_value=None):
    value = st.text_input(label, key=key)
    if value.strip().replace(".", "", 1).isdigit():
        num = float(value)
        if min_value is not None and num < min_value: 
            return None
        if max_value is not None and num > max_value: 
            return None
        return num
    return None

def required_selectbox(label, options, key):
    value = st.selectbox(label, ["-- Select --"] + options, key=key)
    return None if value == "-- Select --" else value

def required_slider(label, min_value, max_value, default, key):
    return st.slider(label, min_value, max_value, default, key=key)

def stroke_prediction_app():
    st.title("🫀 Stroke Risk Predictor")
    st.warning("Complete all fields for accurate assessment")
# Get nutritional score
    nutritional_score = compute_nutritional_score()
    st.info(f"🍎 Nutritional Health Score: **{nutritional_score}/5**")
    with st.form("stroke_form"):
    # ✅ Selectbox with placeholder
        age = st.selectbox("Age", ["Select"] + [i for i in range(18, 221)])
        gender = st.selectbox("Gender", ["Select Gender", "Male", "Female"])
        blood_group = st.selectbox("Blood Group", ["Select Blood Group", "A+","A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
        genotype = st.selectbox("Genotype", ["Select Genotype", "AA", "AS", "SS", "AC", "SC"])
        heart_disease = st.selectbox("Heart Disease", ["Select", "Yes", "No"])
        hypertension = st.selectbox("Hypertension", ["Select", "Yes", "No"])
        systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=220, value=None)
        diastolic_bp = st.number_input("Diastolic BP", min_value=50, max_value=150, value=None)
       

        st.subheader("📏 BMI Assessment (Body Mass Index)")
        st.caption("BMI is calculated as weight (kg) ÷ height (m)². It estimates body fat and overall health risk.")

# Weight input
        weight = st.number_input(
        "Enter your weight (kg)", 
        min_value=20.0, 
        max_value=200.0, 
        value=20.0,
        step=0.1,
        key='weight'
)
        st.caption("⚠️ Weight must be in kilograms for accurate BMI calculation (e.g., 70 kg).")

# Height input
        height = st.number_input(
        "Enter your height (m)", 
        min_value=1.0, 
        max_value=2.5, 
        value=1.00,
        step=0.01,
        key='height'
)
        st.caption("⚠️ Height must be in meters for accurate BMI calculation (e.g., 1.75 m).")

# Calculate BMI and display immediately
        if weight > 0 and height > 0:
            bmi = round(weight / (height ** 2), 2)
            st.success(f"Calculated BMI: **{bmi}**")
        else:
            bmi = None
        # Determine BMI category
        if bmi < 18.5:
            risk = "Underweight"
            color = "warning"
        elif 18.5 <= bmi < 24.9:
            risk = "Normal weight"
            color = "success"
        elif 25 <= bmi < 29.9:
            risk = "Overweight"
            color = "warning"
        else:
            risk = "Obese"
            color = "error"
    
        st.markdown(f"**BMI Category:** <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
        # Collecting additional user information
        marital_status = st.selectbox("Marital Status", ["Select", "Single", "Married", "Divorced", "Widowed"])
        work_type = st.selectbox("Work Type", ["Select", "Private", "Self-employed", "Govt job", "Children", "Never worked"])
        residence_type = st.selectbox("Residence Type", ["Select", "Urban", "Rural"])
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=None, format="%.2f")
        smoking_status = st.selectbox("Smoking Status", ["Select", "formerly smoked", "never smoked", "smokes"])

        stress_level = st.selectbox("Stress Level", ["Select", "None", "Low", "Moderate", "High"])
        ptsd = st.selectbox("PTSD", ["Select", "Yes", "No"])
        depression_level = st.selectbox("Depression Level", ["Select", "None", "Mild", "Moderate", "Severe"])
        diabetes_type = st.selectbox("Diabetes Type", ["Select", "None", "Type 1", "Type 2", "Gestational"])
        chronic_pain = st.selectbox("Chronic Pain", ["Select", "None", "Rheumatism", "Osteoarthritis", "Others"])

        sleep_hours = st.slider("Sleep Hours", 3.0, 10.0, value=None)
        hypertension_treatment = st.selectbox("Hypertension Treatment", ["Select", "None", "Herbal", "Drugs"])
        salt_intake = st.selectbox("Salt Intake", ["Select", "None", "Little", "Moderate", "High"])
        noise_sources = st.selectbox("Noise Sources", ["Select", "None", "Mosque", "Church", "Market", "Block-Industry",
                                                  "Grinding-Machine", "Welder", "Club-House", "Generator"])

        pollution_level_air = st.selectbox("Air Pollution Level", ["Select", "None", "Low", "Moderate", "High"])
        pollution_level_water = st.selectbox("Water Pollution Level", ["Select", "None", "Low", "Moderate", "High"])
        pollution_level_environmental = st.selectbox("Environmental Pollution Level", ["Select", "None", "Low", "Moderate", "High"])
        CustomStressScore = st.number_input("Custom Stress Score", min_value=0, max_value=10, value=None)
        
        st.subheader("🧠 Additional Stress Assessment")
        _, _, stress_score = custom_stress_score(use_container=True)
        st.session_state.stress_score = stress_score

        submit_stroke_inputs = st.form_submit_button("Predict Stroke Risk")
        def all_fields():
                age, gender, heart_disease, hypertension, systolic_bp, diastolic_bp, bmi,
                marital_status, work_type, residence_type, avg_glucose_level, smoking_status,
                stress_level, ptsd, depression_level, diabetes_type, chronic_pain,
                sleep_hours, hypertension_treatment, salt_intake, noise_sources,
                pollution_level_air, pollution_level_water, pollution_level_environmental,
                CustomStressScore, stress_score, height, weight,
                blood_group, genotype
        all_fields = [age, gender, heart_disease, hypertension, systolic_bp, diastolic_bp, bmi,
                      marital_status, work_type, residence_type, avg_glucose_level, smoking_status,
                      stress_level, ptsd, depression_level, diabetes_type, chronic_pain,
                      sleep_hours, hypertension_treatment, salt_intake, noise_sources,
                      pollution_level_air, pollution_level_water, pollution_level_environmental,
                      CustomStressScore, stress_score, height, weight, blood_group, genotype
        ]
        if all(v is not None for v in all_fields):
            submit_stroke_inputs = st.form_submit_button("Predict Stroke Risk")
        else:
        # Fake disabled button when not all fields filled
            st.markdown(
            "<button disabled style='background-color:grey; color:white; padding:0.5em 1em; border-radius:8px; border:none;'>Predict Stroke Risk</button>",
            unsafe_allow_html=True
        )
        submit_stroke_inputs = False
        st.warning("Complete all fields for accurate assessment")
    if submit_stroke_inputs:
        try:
        # 1️⃣ Collect raw inputs safely
            raw_inputs = {
            'age': safe_int(age),
            'avg_glucose_level': safe_float(avg_glucose_level),
            'bmi': safe_float(bmi),
            'ptsd': 1 if ptsd == "Yes" else 0,
            'depression_level': safe_int(depression_level),
            'sleep_hours': safe_int(sleep_hours),
            'gender': 1 if gender == "Male" else 0,
            'ethnicity': encoded_ethnicity,  # required
            'country': encoded_country,      # required
            'province': encoded_province,    # required
            'ever_married': 1 if marital_status=="Yes" else 0,
            'work_type': safe_int(work_type),
            'residence_type': safe_int(residence_type),
            'smoking_status': safe_int(smoking_status),
            'salt_intake': safe_int(salt_intake),
            'noise_sources': safe_int(noise_sources),
            'stress_level': safe_int(stress_level),
            'height': safe_float(height),
            'weight': safe_float(weight),
            'pollution_level_air': safe_int(pollution_level_air),
            'pollution_level_water': safe_int(pollution_level_water),
            'pollution_level_environmental': safe_int(pollution_level_environmental),
            'chronic_pain': safe_int(chronic_pain),
            'hypertension_treatment': safe_int(hypertension_treatment),
            'diabetes_type': safe_int(diabetes_type),
            'blood_group': blood_group,
            'genotype': genotype,
            'hypertension': 1 if hypertension else 0,
            'heart_disease': 1 if heart_disease else 0,
            'CustomStressScore': safe_float(CustomStressScore),
            'location_str': raw_inputs.get('location_str', None) if 'raw_inputs' in locals() else None
        }

        # 2️⃣ Create DataFrame from raw_inputs
            stroke_inputs_df = prepare_stroke_input_numeric(raw_inputs)


        # 3️⃣ Make prediction
            if "stroke_model" in st.session_state:
                pred = st.session_state.stroke_model.predict(stroke_inputs_df)[0]
            else:
                st.error("Model not loaded. Please initialize the model first.")
                st.stop()


        # 4️⃣ Build inputs dict AFTER pred is defined
            db_payload = {
            "user_id": st.session_state.user['id'] if st.session_state.get('user') else "anonymous",
            "age": age,
            "gender": gender,
            "heart_disease": heart_disease,
            "hypertension": hypertension,
            "systolicbp": systolic_bp,
            "diastolicbp": diastolic_bp,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "marital_status": marital_status,
            "work_type": work_type,
            "residence_type": residence_type,
            "smoking_status": smoking_status,
            "stress_level": stress_level,
            "ptsd": ptsd,
            "depression_level": depression_level,
            "diabetes_type": diabetes_type,
            "chronic_pain": chronic_pain,
            "sleep_hours": sleep_hours,
            "hypertension_treatment": hypertension_treatment,
            "salt_intake": salt_intake,
            "blood_group": blood_group,
            "genotype": genotype,
            "noise_sources": noise_sources,
            "pollution_level_air": pollution_level_air,
            "pollution_level_water": pollution_level_water,
            "pollution_level_environmental": pollution_level_environmental,
            "custom_stress_score": CustomStressScore,
            "ethnicity": encoded_ethnicity,
            "country": encoded_country,
            "province": encoded_province,
            "prediction_result": float(pred)
        }

        # 4️⃣ Location
            city, region, country = get_user_location()
            location_str = f"{city}, {region}, {province_map}, {country}, Africa"

        # 5️⃣ Display prediction result
           if pred == 1:
    st.error("⚠️ HIGH STROKE RISK DETECTED")
    st.markdown("""
    ## 🚨 Immediate Action Recommended:
    - **Consult a healthcare provider immediately**
    - **Add Saigon Cinnamon and Alligator Pepper to diet**
    - Monitor blood pressure daily
    - Avoid strenuous activities
    - Reduce salt intake
    - Increase consumption of leafy greens
    - Limit fried foods and processed meals
    - Eat more fruits and vegetables (rich in potassium and fiber)
    - Maintain regular sleep schedule
    - Maintain a healthy weight (avoid obesity)
    """)
else:
    st.success("✅ LOW STROKE RISK DETECTED")




        # Lifestyle suggestions expander
            with st.expander("🛠️ Lifestyle Suggestions for Stroke Prevention"):
                st.markdown("""
            ### 🍽️ Dietary Recommendations:
            - Reduce salt intake
            - Increase consumption of leafy greens
            - Limit fried foods and processed meals
            - Eat more fruits and vegetables (rich in potassium and fiber)
            ### 🏃 Physical Activity:
            - Brisk walking 30 minutes/day
            - Light stretching morning and evening
            - Avoid prolonged sitting
            ### 😌 Stress Management:
            - Practice deep breathing exercises (e.g., prayer, yoga, deep breathing)
            - Maintain regular sleep schedule
            - Engage in community activities
            ### 🩺 Medical Follow-up:
            - BP check every 2 weeks
            - Annual glucose screening
            - Medication adherence if prescribed
            - Prioritize 7-8 hours of sleep per night  
            - **Take cinnamon (e.g., Saigon cinnamon) regularly**  
            - *Reduces blood sugar, inflammation, and oxidative stress*  
            - *Supports brain and heart health naturally*  
            - Stop smoking and reduce alcohol intake  
            - Maintain a healthy weight (avoid obesity)
            """)
      
            pred = st.session_state.stroke_model.predict(stroke_inputs_df)[0]

        # 4️⃣ Build payload for Supabase
        # Only include columns that exist in the table
            stroke_table_columns = [
            "user_id","gender","ethnicity","country","prediction_result","age","BMI","avg_glucose_level","SystolicBP",
            "DiastolicBP","sleep_hours","salt_intake","stress_level","custom_stress_score",
            "depression_level","ptsd","chronic_pain","diabetes_type","hypertension",
            "hypertension_treatment","heart_disease","smoking_status","work_type","height","weight",
            "residence_type","noise_sources","pollution_level_air","pollution_level_water",
            "pollution_level_environmental","marital_status","blood_group","genotype","location_str","raw_inputs"
        ]

            db_payload = {"user_id": st.session_state.user['id'] if st.session_state.get('user') else "anonymous",
                      "prediction_result": float(pred)}

            for col in stroke_table_columns:
                if col in raw_inputs:
                    db_payload[col] = raw_inputs[col]

        # Optional: include raw_inputs jsonb if the column exists
            if 'raw_inputs' in stroke_table_columns:
                db_payload['raw_inputs'] = raw_inputs
# Optional: include location_str if column exists
            if 'location_str' in stroke_table_columns:
                db_payload['location_str'] = raw_inputs.get('location_str')
        # 5️⃣ Save to Supabase
            response = supabase.table("stroke_predictions").insert(db_payload).execute()

            if response.data:
                st.success("Stroke prediction saved successfully!")
            elif response.error:
                st.error(f"Failed to save stroke prediction: {response.error}")

        except Exception as e:
            st.error(f"Error during stroke prediction or saving: {e}")


def build_full_input(raw):
    # Map gender
    gender = 1 if raw.get("gender") == "Male" else 0

    # Head injury mapping
    head_map = {"None": 0, "Accident": 1, "Violence": 2}
    head_injury = head_map.get(raw.get("HeadInjury", "None"), 0)

    # Cognitive symptoms mapping
    option_map = {"Yes": 1, "No": 0, "Sometimes": 0.5}

    # Map option_map and head_map
    option_map = {"Yes": 1, "No": 0, "Sometimes": 0.5}
    head_map = {"None": 0, "Accident": 1, "Violence": 2}

    # Default MMSE if not in raw
    mmse = raw.get("MMSE", 30)

    # -----------------------------
    # Numeric-only full input
    full_input = {
        
    }

def prepare_alzheimers_input_numeric(raw_inputs):
    # Ensure raw_inputs is always a dictionary, even if None is passed
    if raw_inputs is None:
        raw_inputs = {}
    
    # Define all expected columns for the Alzheimer's model
    expected_columns = [
        'FunctionalAssessment', 'PhysicalActivity', 'Smoking', 'AlcoholConsumption', 
        'Gender', 'PersonalityChanges', 'EducationLevel', 'FamilyHistoryAlzheimers', 
        'Confusion', 'Hypertension', 'Disorientation', 'Forgetfulness', 'ADL', 
        'CholesterolHDL', 'BehavioralProblems', 'DiastolicBP', 'CardiovascularDisease', 
        'BMI', 'Depression', 'DietQuality', 'SystolicBP', 'Diabetes', 'CholesterolTotal', 
        'MMSE', 'MemoryComplaints', 'Age', 'CholesterolTriglycerides', 'SleepQuality', 
        'HeadInjury', 'CholesterolLDL', 'DifficultyCompletingTasks', 'Height', 'Weight', 'Genotype', 'BloodGroup'
    ]
    
    # Create a dictionary with default values for all expected columns
    default_values = {
        'FunctionalAssessment': 0, 'PhysicalActivity': 0, 'Smoking': 0, 
        'AlcoholConsumption': 0, 'Gender': 0, 'PersonalityChanges': 0, 
        'EducationLevel': 0, 'FamilyHistoryAlzheimers': 0, 'Confusion': 0, 
        'Hypertension': 0, 'Disorientation': 0, 'Forgetfulness': 0, 'ADL': 0, 
        'CholesterolHDL': 0.0, 'BehavioralProblems': 0, 'DiastolicBP': 0, 
        'CardiovascularDisease': 0, 'BMI': 0.0, 'Depression': 0, 'DietQuality': 0, 
        'SystolicBP': 0, 'Diabetes': 0, 'CholesterolTotal': 0.0, 'MMSE': 0, 
        'MemoryComplaints': 0, 'Age': 0, 'CholesterolTriglycerides': 0.0, 
        'SleepQuality': 0, 'HeadInjury': 0, 'CholesterolLDL': 0.0, 
        'DifficultyCompletingTasks': 0, 'Height': 0, 'Weight': 0, 'Genotype': 0, 'BloodGroup': 0
    }
    
    # Initialize the full input with default values
    full_input = default_values.copy()
    
    # Update with actual values from raw_inputs where available
    for col in expected_columns:
        if col in raw_inputs:
            # Handle different data types appropriately
            if col in ['Age', 'BMI', 'CholesterolHDL', 'CholesterolTotal', 
                      'CholesterolTriglycerides', 'CholesterolLDL']:
                try:
                    full_input[col] = float(raw_inputs[col])
                except (ValueError, TypeError):
                    full_input[col] = default_values[col]
            elif col in ['MMSE', 'EducationLevel', 'SystolicBP', 'DiastolicBP']:
                try:
                    full_input[col] = int(raw_inputs[col])
                except (ValueError, TypeError):
                    full_input[col] = default_values[col]
            elif col == 'Gender':
                # Convert gender to numeric (0 for Female, 1 for Male)
                gender_val = raw_inputs.get('Gender', 'Female')
                full_input['Gender'] = 1 if str(gender_val).lower() == 'male' else 0
            else:
                # For binary/categorical features
                val = raw_inputs[col]
                if str(val).lower() in ['1', 'true', 'yes', 'y']:
                    full_input[col] = 1
                else:
                    try:
                        full_input[col] = int(val)
                    except (ValueError, TypeError):
                        full_input[col] = default_values[col]
    
    # Create DataFrame
    alzheimer_inputs_df = pd.DataFrame([full_input])
    
    # Ensure all expected columns are present
    alzheimer_inputs_df = alzheimer_inputs_df[expected_columns]
    
    return alzheimer_inputs_df


def raw_inputs_collection():
        # Collect all required inputs
        raw_inputs = {}
        # Numeric inputs
        raw_inputs['Age'] = st.number_input('Age', min_value=0, max_value=120, value=50)
        raw_inputs['BMI'] = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0)
        raw_inputs['MMSE'] = st.number_input('MMSE Score', min_value=0, max_value=30, value=25)
        raw_inputs['EducationLevel'] = st.number_input('Education Level', min_value=0, max_value=20, value=12)
        raw_inputs['SystolicBP'] = st.number_input('Systolic Blood Pressure', min_value=80, max_value=200, value=120)
        raw_inputs['DiastolicBP'] = st.number_input('Diastolic Blood Pressure', min_value=50, max_value=120, value=80)
        raw_inputs['CholesterolHDL'] = st.number_input('HDL Cholesterol', min_value=20.0, max_value=100.0, value=50.0)
        raw_inputs['CholesterolLDL'] = st.number_input('LDL Cholesterol', min_value=50.0, max_value=200.0, value=100.0)
        raw_inputs['CholesterolTotal'] = st.number_input('Total Cholesterol', min_value=100.0, max_value=300.0, value=200.0)
        raw_inputs['CholesterolTriglycerides'] = st.number_input('Triglycerides', min_value=50.0, max_value=500.0, value=150.0)
        
        # Binary/categorical inputs
        raw_inputs['Gender'] = st.selectbox('Gender', ['Male', 'Female'])
        raw_inputs['Hypertension'] = st.selectbox('Hypertension', ['No', 'Yes'])
        raw_inputs['Diabetes'] = st.selectbox('Diabetes', ['No', 'Yes'])
        raw_inputs['CardiovascularDisease'] = st.selectbox('Cardiovascular Disease', ['No', 'Yes'])
        raw_inputs['FamilyHistoryAlzheimers'] = st.selectbox('Family History of Alzheimer\'s', ['No', 'Yes'])
        raw_inputs['HeadInjury'] = st.selectbox('History of Head Injury', ['No', 'Yes'])
        
        # Additional binary inputs
        binary_options = ['No', 'Yes']
        raw_inputs['Smoking'] = st.selectbox('Smoking', binary_options)
        raw_inputs['AlcoholConsumption'] = st.selectbox('Alcohol Consumption', binary_options)
        raw_inputs['PhysicalActivity'] = st.selectbox('Physical Activity', binary_options)
        raw_inputs['Depression'] = st.selectbox('Depression', binary_options)
        raw_inputs['Forgetfulness'] = st.selectbox('Forgetfulness', binary_options)
        raw_inputs['Confusion'] = st.selectbox('Confusion', binary_options)
        raw_inputs['Disorientation'] = st.selectbox('Disorientation', binary_options)
        raw_inputs['PersonalityChanges'] = st.selectbox('Personality Changes', binary_options)
        raw_inputs['BehavioralProblems'] = st.selectbox('Behavioral Problems', binary_options)
        raw_inputs['DifficultyCompletingTasks'] = st.selectbox('Difficulty Completing Tasks', binary_options)
        raw_inputs['MemoryComplaints'] = st.selectbox('Memory Complaints', binary_options)
        # Height and Weight for BMI calculation
        raw_inputs['Height'] = st.number_input('Height (m)', min_value=0.0, max_value=3.0, value=1.75)
        raw_inputs['Weight'] = st.number_input('Weight (kg)', min_value=0, max_value=300, value=70)
        raw_inputs['Genotype'] = st.selectbox('Genotype', ['AA', 'AS', 'SS', 'AC', 'SC'])
        raw_inputs['BloodGroup'] = st.selectbox('Blood Group', ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-'])
        # Additional inputs with different scales
        raw_inputs['SleepQuality'] = st.slider('Sleep Quality', 0, 10, 5)
        raw_inputs['DietQuality'] = st.slider('Diet Quality', 0, 10, 5)
        raw_inputs['FunctionalAssessment'] = st.slider('Functional Assessment', 0, 10, 5)
        raw_inputs['ADL'] = st.slider('Activities of Daily Living', 0, 10, 5)
        
        
import streamlit as st

def select_input(label, options, key=None):
    """Universal select input with a 'Select...' placeholder."""
    return st.selectbox(label, ["Select..."] + options, index=0, key=key)

def alzheimers_prediction_app():
    pred = None
    alzheimer_inputs_df = None

    st.title("🧠 Alzheimer's Predictor")
    st.warning("Complete all fields for accurate assessment")

    # Nutritional score (computed separately)
    nutritional_score = compute_nutritional_score()
    st.info(f"🍎 Nutritional Health Score: **{nutritional_score}/5**")

    with st.form("alz_form"):
        age = st.number_input("Age", min_value=0, max_value=100, value=None, key='alz_age', placeholder="Enter Age")
        gender = select_input("Gender", ["Male", "Female"], key='alz_gender')
        blood_group = select_input("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], key='alz_bloodgroup')
        genotype = select_input("Genotype", ["AA", "AS", "SS", "AC", "SC"], key='alz_genotype')

        education_years = select_input("Education Level (Years)", [str(i) for i in range(0, 21)], key='alz_eduyears')
        # ........................BMI input with Weight and Height....................
        st.subheader("📏 BMI Assessment (Body Mass Index)")
        st.caption("BMI is calculated as weight (kg) ÷ height (m)². It estimates body fat and overall health risk.")
        # Weight input
        weight = st.number_input("Enter your weight (kg)", key='alz_weight', min_value=20, max_value=200)

        # Height input
        height = st.number_input("Enter your height (m)", key='alz_height', min_value=1.0, max_value=2.5)

# Calculate BMI if both values are provided
        bmi = None
        if weight is not None and height is not None:
            try:
                bmi = round(weight / (height ** 2), 2)
                st.info(f"Calculated BMI: **{bmi}**")
            except ZeroDivisionError:
                st.error("Height cannot be zero.")
        # Determine BMI category
        if bmi < 18.5:
            risk = "Underweight"
            color = "warning"
        elif 18.5 <= bmi < 24.9:
            risk = "Normal weight"
            color = "success"
        elif 25 <= bmi < 29.9:
            risk = "Overweight"
            color = "warning"
        else:
            risk = "Obese"
            color = "error"
    
        st.markdown(f"**BMI Category:** <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
        is_smoker = select_input("Smoking Status", ["Select", "formerly smoked", "never smoked", "smokes"], key='alz_smoking')
        alcohol_consumption = select_input("Alcohol Consumption (0=None, 5=High)", [str(i) for i in range(0, 6)], key='alz_alcohol')
        physical_activity = select_input("Physical Activity (hrs/week)", [str(i) for i in range(0, 21)], key='alz_activity')
        sleep_quality = select_input("Sleep Quality (1-5)", [str(i) for i in range(1, 6)], key='alz_sleep')
        family_history_alz = select_input("Family History of Alzheimer's", ["Yes", "No"], key='alz_family')
        cardiovascular_disease = select_input("Cardiovascular Disease", ["Yes", "No"], key='alz_cardio')
        diabetes = select_input("Diabetes", ["Yes", "No"], key='alz_diabetes')
        depression = select_input("Depression", ["Yes", "No"], key='alz_depression')
        hypertension = select_input("Hypertension", ["Yes", "No"], key='alz_hypertension')

        systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=220, value=None, key='alz_systolic', placeholder="Enter systolic")
        diastolic_bp = st.number_input("Diastolic BP", min_value=50, max_value=150, value=None, key='alz_diastolic', placeholder="Enter diastolic")

        cholesterol_total = st.number_input("Total Cholesterol", min_value=100, max_value=400, value=None, key='alz_chol_total', placeholder="Enter total cholesterol")
        cholesterol_ldl = st.number_input("LDL", min_value=50, max_value=300, value=None, key='alz_ldl', placeholder="Enter LDL")
        cholesterol_hdl = st.number_input("HDL", min_value=20, max_value=100, value=None, key='alz_hdl', placeholder="Enter HDL")
        cholesterol_triglycerides = st.number_input("Triglycerides", min_value=50, max_value=500, value=None, key='alz_trig', placeholder="Enter triglycerides")

        functional_assessment = st.slider("Functional Assessment (0-5)", 0, 5, 0, key='alz_func')
        behavioral_problems = select_input("Behavioral Problems", ["Yes", "No"], key='alz_behavior')
        adl = st.slider("ADL Score (Activities of Daily Living)", 0, 6, 0, key='alz_adl')

       # MMSE (direct)
        st.subheader("🧠 Mini-Mental State Examination (MMSE)")
        st.caption(
    "The MMSE is a 30-point questionnaire used to assess cognitive function, evaluating "
    "orientation, attention, memory, language, and visual-spatial skills. Higher scores indicate better cognitive performance."
)

# Optional: culturally adapted version note
        st.caption(
    "📝 Culturally Adapted MMSE: Some questions have been modified to reflect daily life and cultural context, "
    "providing a more accurate assessment for African populations."
)

        mmse = st.slider("MMSE Score (0-30)", 0, 30, 0, key='alz_mmse')


# ===== CULTURALLY ADAPTED MMSE ASSESSMENT ===== #
        st.subheader("MMSE Assessment (Adapted for African Context)")
        mmse_option = st.radio("Do you know your MMSE score?", ["Estimate using cultural questions"], key='alz_mmse_radio')


        st.info("Answer these culturally relevant questions to estimate your MMSE score:")


        col1, col2 = st.columns(2)


        with col1:
            q1 = select_input("Do you forget names of relatives/village members?", ["Never", "Sometimes", "Often"], key='q1')
            q2 = select_input("Do you misplace important items (farming tools, keys)?", ["Never", "Sometimes", "Often"], key='q2')
            q3 = select_input("Can you recall traditional recipes or remedies?", ["Always", "Sometimes", "Rarely"], key='q3')


        with col2:
            q4 = select_input("Do you recognize people from your community?", ["Always", "Sometimes", "Rarely"], key='q4')
            q5 = select_input("Can you navigate familiar paths/markets?", ["Always", "Sometimes", "Rarely"], key='q5')
            q6 = select_input("Do you remember important cultural events/dates?", ["Always", "Sometimes", "Rarely"], key='q6')


# Calculate estimated MMSE with cultural weighting#
        response_scores = {
            "Never": 2, "Sometimes": 1, "Often": 0, "Always": 2, "Rarely": 0
            }


        weights = {
            "q1": 2.0, # Names of relatives
            "q2": 1.5, # Important items
            "q3": 1.0, # Traditional knowledge
            "q4": 1.7, # Community recognition
            "q5": 2.0, # Navigation
            "q6": 1.3  # Cultural events
}


# Only compute estimated MMSE if all cultural questions are answered (not "Select...")
        if all(q not in (None, "Select...") for q in [q1, q2, q3, q4, q5, q6]):
            mmse_score = 20 + (
                response_scores[q1] * weights["q1"] +
                response_scores[q2] * weights["q2"] +
                response_scores[q3] * weights["q3"] +
                response_scores[q4] * weights["q4"] +
                response_scores[q5] * weights["q5"] +
                response_scores[q6] * weights["q6"]
            )
        # Pollution inputs
        pollution_score = st.slider("Pollution Score (0-100)", 0, 100, 0, key='alz_pollution_score')
        pollution_choice = select_input("Pollution Category", ["Low", "Moderate", "High"], key='alz_pollution_cat')

        # Cognitive assessment
        option_map = {"Yes": 1, "No": 0, "Sometimes": 0.5}
        confusion = option_map.get(select_input("Confusion", ["Yes", "No", "Sometimes"], key='alz_confusion'), None)
        disorientation = option_map.get(select_input("Disorientation", ["Yes", "No", "Sometimes"], key='alz_disorien'), None)
        personality_changes = option_map.get(select_input("Personality Changes", ["Yes", "No", "Sometimes"], key='alz_personality'), None)
        difficulty_tasks = option_map.get(select_input("Difficulty Completing Tasks", ["Yes", "No", "Sometimes"], key='alz_tasks'), None)
        forgetfulness = option_map.get(select_input("Forgetfulness", ["Yes", "No", "Sometimes"], key='alz_forget'), None)
        memory_complaints = option_map.get(select_input("Memory Complaints", ["Yes", "No", "Sometimes"], key='alz_memory'), None)

        # Head injury
        head_map = {"None": 0, "Accident": 1, "Violence": 2}
        head_choice = select_input("Head Injury", ["None", "Accident", "Violence"], key='alz_head')
        head_injury = head_map.get(head_choice, None)

        # Stress
        stress_score = st.slider("Stress Level", 0, 10, 0, key='alz_stress')

        submit_alz = st.form_submit_button("🔍 Predict Alzheimer Risk")

    # Validation step
    required_fields = [age, gender, education_years, bmi, is_smoker, alcohol_consumption, physical_activity,
                       sleep_quality, family_history_alz, cardiovascular_disease, diabetes, depression, hypertension,
                       systolic_bp, diastolic_bp, cholesterol_total, cholesterol_ldl, cholesterol_hdl,
                       cholesterol_triglycerides, behavioral_problems, head_choice, pollution_choice,
                       blood_group, genotype, height, weight]

    if submit_alz:
        if any(x in (None, "Select...") for x in required_fields):
            st.error("⚠️ Please complete all fields before prediction.")
            st.stop()

        # 🔹 From here, continue your raw_inputs dict and prediction logic as in your original code 🔹
        st.success("All inputs validated! Now run prediction pipeline...")

        
    if submit_alz:
            try:
                # Prepare data for prediction
                raw_inputs = {
                    "Age": age,
                    "Gender": gender,
                    "BMI": bmi,
                    "EducationLevel": education_years,
                    "Smoking": is_smoker,
                    "AlcoholConsumption": alcohol_consumption,
                    "PhysicalActivity": physical_activity,
                    "DietQuality": diet_quality,
                    "SleepQuality": sleep_quality,
                    "FamilyHistoryAlzheimers": family_history_alz,
                    "CardiovascularDisease": cardiovascular_disease,
                    "Diabetes": diabetes,
                    "Depression": depression,
                    "Hypertension": hypertension,
                    "SystolicBP": systolic_bp,
                    "DiastolicBP": diastolic_bp,
                    "CholesterolTotal": cholesterol_total,
                    "CholesterolLDL": cholesterol_ldl,
                    "CholesterolHDL": cholesterol_hdl,
                    "CholesterolTriglycerides": cholesterol_triglycerides,
                    "MMSE": mmse,
                    "Height": height,
                    "Weight": weight,
                    "Genotype": genotype,
                    "BloodGroup": blood_group,
                    "FunctionalAssessment": functional_assessment,
                    "BehavioralProblems": behavioral_problems,
                    "ADL": adl,
                    "Confusion": confusion,
                    "Disorientation": disorientation,
                    "PersonalityChanges": personality_changes,
                    "DifficultyCompletingTasks": difficulty_tasks,
                    "Forgetfulness": forgetfulness,
                    "MemoryComplaints": memory_complaints,
                    "MemoryScore": st.session_state.get("memory_score", 1.0),
                    "HeadInjury": head_injury,
                    "Ethnicity": encoded_ethnicity,
                    "Country": encoded_country,
                    "Province_Option": encoded_province,
                    "PollutionScore": pollution_score,
                    "PollutionCategoryLow": pollution_low,
                    "PollutionCategoryModerate": pollution_moderate,
                    "PollutionCategoryHigh": pollution_high,
                    "CustomStressScore": st.session_state.stress_score
                }
                # Convert raw inputs into proper DataFrame for prediction
                # 1️⃣ Prepare inputs
                alzheimer_inputs_df = prepare_alzheimers_input_numeric(raw_inputs)

                if alzheimer_inputs_df is None or alzheimer_inputs_df.empty:
                    st.error("Input preparation failed: no valid features for prediction.")
                    st.stop()

# 2️⃣ Ensure model exists
                if "alz_model" not in st.session_state or st.session_state.alz_model is None:
                    st.error("Alzheimer's model not loaded. Please initialize the model first.")
                    st.stop()

# 3️⃣ Make prediction
                pred = st.session_state.alz_model.predict(alzheimer_inputs_df)[0]

# 4️⃣ Display results
                if pred == 1:
                    st.error("⚠️ HIGH ALZHEIMER RISK DETECTED")
                    st.markdown("""## 🚨 Immediate Action Recommended:
- **Consult a healthcare provider immediately**
- Begin cognitive training exercises
- Review family medical history
- Do mental exercises (e.g., puzzles, memory games)
- Stay physically active (exercise increases brain health)
- Reduce stress practice mindfulness or prayer
- Stay socially engaged talk to friends, join a group
- Eat brain-healthy foods (nuts, omega-3s, leafy greens)
- **Use cinnamon regularly** may protect memory and reduce inflammation
- Avoid smoking and limit alcohol
- Prioritize sleep and manage depression
""")
                else:
                    st.success("✅ LOW ALZHEIMER'S RISK DETECTED")

# 5️⃣ Expandable lifestyle suggestions
                with st.expander("🛠️ Lifestyle Suggestions for Alzheimer's Prevention"):
                    st.markdown("""
### 🍽️ Dietary Recommendations:
- Increase omega-3 fatty acids (fish, flax seeds)
- Consume antioxidant-rich foods (berries, dark chocolate)
- Eat leafy green vegetables daily
- Reduce processed sugars and refined carbs

### 🏃 Physical Activity:
- Aerobic exercise 3–5 times/week
- Strength training 2–3 times/week
- Balance and coordination exercises

### 😌 Mental Wellness:
- Practice meditation or mindfulness
- Maintain social connections
- Learn new skills or languages

### 🩺 Medical Follow-up:
- Annual cognitive screening after age 60
- Manage vascular risk factors (blood pressure, cholesterol)
- Medication review with doctor
""")

                # Get user location
                city, region, country = get_user_location()
                location_str = f"{city}, {region}, {country}"

                # Prepare data dict for database save
                alz_data = {
                    "user_id": st.session_state.user['id'] if st.session_state.get('user') else "anonymous",
                    "age": age,
                    "gender": gender,  # 1 = Male, 0 = Female
                    "bmi": bmi,
                    "education_level": education_years,
                    "smoking": is_smoker,
                    "alcohol_consumption": alcohol_consumption,
                    "physical_activity": physical_activity,
                    "diet_quality": diet_quality,
                    "sleep_quality": sleep_quality,
                    "family_history_alzheimers": family_history_alz,
                    "cardiovascular_disease": cardiovascular_disease,
                    "diabetes": diabetes,
                    "depression": depression,
                    "hypertension": hypertension,
                    "systolic_bp": systolic_bp,
                    "diastolic_bp": diastolic_bp,
                    "cholesterol_total": cholesterol_total,
                    "cholesterol_ldl": cholesterol_ldl,
                    "cholesterol_hdl": cholesterol_hdl,
                    "cholesterol_triglycerides": cholesterol_triglycerides,
                    "functional_assessment": functional_assessment,
                    "behavioral_problems": behavioral_problems,
                    "adl": adl,
                    "confusion": confusion,
                    "disorientation": disorientation,
                    "personality_changes": personality_changes,
                    "difficulty_completing_tasks": difficulty_tasks,
                    "forgetfulness": forgetfulness,
                    "memory_complaints": memory_complaints,
                    "head_injury": head_injury,
                    "height": height,
                    "weight": weight,
                    "blood_group": blood_group,
                    "genotype": genotype,
                    "pollution_score": pollution_score,
                    "pollution_category_Low": pollution_low,
                    "pollution_category_Moderate": pollution_moderate,
                    "pollution_category_High": pollution_high,
                    "ethnicity": encoded_ethnicity,
                    "country": encoded_country,
                    "province_option": encoded_province,
                    "memory_score": st.session_state.get("memory_score", 1.0),  # <-- Save memory game score
                    "custom_stress_score": st.session_state.get("stress_score", None),
                    "location": location_str,
                    "prediction_result": int(pred)
                }
                if not raw_inputs:
                    st.error("Please provide all required inputs.")
                    return
            
        # Prepare the input data
                alzheimer_inputs_df = prepare_alzheimers_input_numeric(raw_inputs)
        
        # Check if DataFrame was created successfully
                if alzheimer_inputs_df is None or alzheimer_inputs_df.empty:
                    st.error("Failed to prepare input data for prediction.")
                    return
            
        # Make prediction using your model
                if 'alz_model' in st.session_state:
            # Get prediction from model
                    prediction_result = st.session_state.alz_model.predict(alzheimer_inputs_df)

            # Check if we got a valid prediction
                if prediction_result is not None and len(prediction_result) > 0:
                    pred = prediction_result[0]
                
                # Only proceed if we have a valid prediction
                if pred is not None:
                    # Prepare database dictionary
                    db_payload = {
                        "user_id": st.session_state.user['id'] if st.session_state.get('user') else "anonymous",
                        "raw_inputs": raw_inputs,
                        "prediction_result": float(pred)
                    }
                    
                    # Save to database
                    response = supabase.table("alzheimer_predictions").insert(db_payload).execute()
                    if response.data:
                        st.success("Alzheimer's prediction saved successfully!")
                    elif response.error:
                        st.error(f"Failed to save Alzheimer's prediction: {response.error}")
                
                else:
                    st.error("Prediction returned no result.")
                    return
            except Exception as e:
        # Check if the error occurred during DataFrame creation or prediction
                if alzheimer_inputs_df is None:
                    st.error(f"Error during input preparation for Alzheimer's prediction: {e}")
            else:
                st.error(f"Error during Alzheimer's prediction: {e}")
                return
    
    # Now safely check the prediction result
    if pred is not None:
        try:
            # Convert to float first, then to int for comparison
            pred_value = float(pred)
            if int(pred_value) == 1:
                st.warning("The model predicts a high risk of Alzheimer's disease.")
                # Display additional information or recommendations
            else:
                st.success("The model predicts a low risk of Alzheimer's disease.")
                # Display additional information or recommendations
        except (ValueError, TypeError):
            st.error(f"Invalid prediction value: {pred}")


def memory_recall_game():
    st.subheader("🧩 Memory Recall Game")

    # --- Initialize memory game state ---
    if "memory_game" not in st.session_state or st.session_state.memory_game is None:
        st.session_state.memory_game = {
            "state": "start",
            "words": [],
            "start_time": None,
            "level": 1,
            "score_history": []
        }

    game = st.session_state.memory_game
    WORD_POOL = [
        "apple", "table", "river", "mountain", "sun", "flower",
        "clock", "phone", "book", "star", "moon", "chair",
        "pencil", "car", "glass", "tree", "music", "house",
        "cloud", "lamp", "keyboard", "shoe", "bottle", "ring"
    ]

    # --- Start screen ---
    if game["state"] == "start":
        st.markdown(f"**Level {game['level']}** - You will see {4 + game['level']} words.")
        if st.button("Start Memory Exercise"):
            num_words = 4 + game["level"]
            words = random.sample(WORD_POOL, num_words)
            game["words"] = words
            game["start_time"] = time.time()
            game["state"] = "showing"
            st.rerun()

    # --- Showing words ---
    elif game["state"] == "showing":
        st.write("Memorize these words (5 seconds):")
        st.info(", ".join(game["words"]))
        if time.time() - game["start_time"] > 5:
            game["state"] = "recalling"
            st.rerun()

    # --- Recalling words ---
    elif game["state"] == "recalling":
        with st.form("recall_form"):
            recalled_input = st.text_input("Type the words you remember, separated by commas:")
            submit = st.form_submit_button("Submit Recall")

        recalled = []  # Ensure variable exists
        correct_count = 0

        if submit:
            recalled = [w.strip().lower() for w in recalled_input.split(",") if w.strip()]
            correct_words = set(w.lower() for w in game["words"])
            recalled_set = set(recalled)
            correct_count = len(correct_words & recalled_set)

            st.success(f"You recalled {correct_count} out of {len(game['words'])} correctly.")

            # Level progression
            if correct_count >= len(game['words']) - 1:
                st.balloons()
                st.info("🎉 Great job! You advance to the next level.")
                game["level"] += 1
            else:
                st.warning("You'll stay on the same level to improve.")

            # Save score in history
            game["score_history"].append({
                "level": game["level"],
                "correct": correct_count,
                "total": len(game["words"]),
                "words": game["words"],
                "recalled": recalled
            })

            # Reset game state
            game["state"] = "start"
            game["words"] = []
            game["start_time"] = None
            st.rerun()

    # --- Display score history ---
    if game["score_history"]:
        with st.expander("📊 Score History"):
            for i, score in enumerate(reversed(game["score_history"])):
                st.write(
                    f"**Round {len(game['score_history']) - i}**: "
                    f"Level {score['level']} - {score['correct']}/{score['total']} correct"
                )

# --- Initialize session state ---
if "user" not in st.session_state:
    st.session_state.user = None
if "nutritional_data" not in st.session_state:
    st.session_state.nutritional_data = {}
if "default_lifestyles" not in st.session_state:
    st.session_state.default_lifestyles = []
if "stress_score" not in st.session_state:
    st.session_state.stress_score = 0
if "location_str" not in st.session_state:
    st.session_state.location_str = {}

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a feature:",
    ["About", "Stroke Prediction", "Alzheimer's Prediction", "Memory Recall Game"]
)

# --- Display pages ---
if page == "About":
    about()  # Always visible, even if no user is logged in
elif page == "Stroke Prediction":
    if st.session_state.user is not None:
        stroke_prediction_app()
    else:
        st.warning("⚠️ Please log in to access Stroke Prediction.")
elif page == "Alzheimer's Prediction":
    if st.session_state.user is not None:
        alzheimers_prediction_app()
    else:
        st.warning("⚠️ Please log in to access Alzheimer's Prediction.")
elif page == "Memory Recall Game":
    if st.session_state.user is not None:
        memory_recall_game()
    else:
        st.warning("⚠️ Please log in to access Memory Recall Game.")

if st.session_state.user is None:
    st.sidebar.header("🔐 User Authentication")
    auth_option = st.sidebar.radio("Select option:", ["Login", "Register"], key="auth_option")
    if auth_option == "Login":
        login()
    else:
        register()
   








































