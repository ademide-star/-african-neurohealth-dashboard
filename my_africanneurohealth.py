import streamlit as st
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
import logging
from sklearn import __version__ as sklearn_version
import cloudpickle
import math
from uuid import UUID
import json
import jsonschema
import sys
import shap
import sqlite3
import logging
from postgrest import APIError
import pickle
from datetime import datetime
import traceback
from sklearn.pipeline import Pipeline
from supabase import create_client, Client

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


# MODEL PATHS
# ======================
ALZ_MODEL_PATH = r"C:\Users\sibs2\african-neurohealth-dashboard\alz_model_pipeline_v1.7.pkl"
STROKE_MODEL_PATH = r"C:\Users\sibs2\african-neurohealth-dashboard\stroke_model_pipeline_v1.7.pkl"

# ======================
# CHECK SKLEARN VERSION
# ======================
REQUIRED_SKLEARN_VERSION = "1.3.2"  # set to the version you trained your models with
if sklearn_version != REQUIRED_SKLEARN_VERSION:
    st.warning(f"⚠️ Your scikit-learn version is {sklearn_version}. "
               f"Models were trained with {REQUIRED_SKLEARN_VERSION}. "
               "Version mismatch may cause errors.")

# ======================
# LOAD FUNCTION
# ======================
def load_model(path):
    if not os.path.exists(path):
        st.error(f"❌ Model file not found: {path}. Please retrain and save using pickle.")
        st.stop()
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading model {os.path.basename(path)}: {str(e)}")
        st.stop()

# ======================
# LOAD MODELS
# ======================
stroke_model = load_model(STROKE_MODEL_PATH)
alz_model = load_model(ALZ_MODEL_PATH)

st.success("✅ Models loaded successfully!")

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

# --- Auth Functions ---
def login():
    st.subheader("Login")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login", key="login_btn"):
        try:
            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
            if response.user:
                st.session_state.user = response.user
                st.success(f"Logged in as {email}")
                st.experimental_rerun()
            else:
                st.error("Invalid login credentials")
        except Exception as e:
            st.error(f"Login error: {e}")

    st.markdown("---")
    st.subheader("Resend Magic Link")
    resend_email = st.text_input("Enter your email to resend the magic link", key="resend_email")
    if st.button("Resend Magic Link", key="resend_btn"):
        if resend_email:
            try:
                response = supabase.auth.sign_in_with_password({"email": resend_email})
                if response.get("error"):
                    st.error(f"Error: {response['error']['message']}")
                else:
                    st.success("Magic link sent! Please check your email.")
            except Exception as e:
                st.error(f"Failed to send magic link: {e}")
        else:
            st.warning("Please enter your email.")

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

def logout():
    supabase.auth.sign_out()
    st.session_state.user = None
    st.experimental_rerun()

# --- App Feature Functions ---
def stroke_prediction_app():
    st.header("Stroke Risk Prediction")
    st.title("🫀 Stroke Risk Predictor")
    st.write("Stroke prediction UI and logic here...")

def alzheimers_prediction_app():
    st.header("Alzheimer's Prediction")
    st.title("🧠 Alzheimer’s Predictor")
    st.write("Alzheimer's prediction UI and logic here...")

def nutrition_tracker_app():
    st.header("Nutrition Tracker")
    st.title("🥗 Nutrition Tracker")
    st.write("Nutrition tracker UI and logic here...")

# --- Main App Flow ---
if st.session_state.user is None:
    with st.sidebar:
        st.header("🔐 User Authentication")
        auth_option = st.radio("Select option:", ["Login", "Register"], key="auth_option")
        if auth_option == "Login":
            login()
        else:
            register()

    with st.expander("ℹ️ About This App 🧠 African NeuroHealth Dashboard"):
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

**By:** Adebimpe-John Omolola E  
**Supervisor:** Prof. Bamidele Owoyele Victor  
**Institution:** University of Ilorin  
**Principal Investigator:** Prof Mayowa Owolabi  
**GRASP / NIH / DSI Collaborative Program**
""")

else:
    # Authenticated users see app selection + tools
    with st.sidebar:
        st.write(f"👋 Welcome, {st.session_state.user.email}!")
        app_choice = st.radio(
            "Choose an App:",
            ["Stroke Prediction", "Alzheimer's Prediction", "Nutrition Tracker"],
            key="app_choice"
        )
        if st.button("Logout", key="logout_button"):
            logout()

    if app_choice == "Stroke Prediction":
        stroke_prediction_app()
    elif app_choice == "Alzheimer's Prediction":
        alzheimers_prediction_app()
    elif app_choice == "Nutrition Tracker":
        nutrition_tracker_app()


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
with st.sidebar:
    st.header("🌍 Location Information")
    selected_country = st.selectbox("Select Country", list(countries_with_provinces.keys()))
    selected_province = st.selectbox("Select Province", countries_with_provinces[selected_country])
    selected_region = st.selectbox("🌍 Select Region", list(region_with_ethnicity.keys()))
    selected_ethnicity = st.selectbox("Select Ethnicity", region_with_ethnicity[selected_region])


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

def prepare_stroke_input_robust(raw_input):
    numeric_features = ['age', 'avg_glucose_level', 'bmi', 'stress_level',
                        'ptsd', 'depression_level', 'diabetes_type', 'sleep_hours']
    
    categorical_features = ['gender', 'ever_married', 'work_type', 'Residence_type',
                            'smoking_status', 'chronic_pain_None', 'chronic_pain_Osteoarthritis',
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
                   'Residence_type', 'smoking_status', 'ethnicity', 'Country', 'Province_Option']
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
        'ptsd', 'depression_level', 'sleep_hours', 'systolic_bp', 'diastolic_bp'
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
        "marital_status", "work_type", "Residence_type", "smoking_status",
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

    # ---- Convert to DataFrame ----
    input_df = pd.DataFrame([full_input])

    # ---- Reorder and reindex to match model input if needed ----
    # Optional: if you have a saved list of expected columns (from training), you can do:
    # input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    return input_df

# --- Streamlit app ---
# --- Stroke Predictor ---

st.title("🧠 African NeuroHealth Dashboard")
app_mode = st.selectbox("Choose Section", ["Stroke Risk Prediction", "Alzheimer Risk Prediction"])


# =======================
# TAB 1: STROKE PREDICTION
# =======================
if app_mode == "Stroke Risk Prediction":
    st.title("🫀 Stroke Risk Predictor")
    st.warning("Complete all fields for accurate assessment")
# Get nutritional score
    nutritional_score = compute_nutritional_score()
    st.info(f"🍎 Nutritional Health Score: **{nutritional_score}/5**")
    with st.form("stroke_form"):
            age = st.slider("Age", 18, 100, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
            heart_disease = st.selectbox("Heart Disease",[0, 1], format_func=lambda x: ["Yes", "No"][x])
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: ["Yes", "No"][x])
            systolic_bp = st.number_input("Systolic BP", 80, 220, 120, key='stroke_systolic')
            diastolic_bp = st.number_input("Diastolic BP", 50, 150, 80, key='stroke_diastolic')
            avg_glucose = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=110.0)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
            work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt job", "Children", "Never worked"])
            residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
            avg_glucose_level = st.number_input("Average Glucose Level", 50.0, 300.0, 100.0, format="%.2f") # Ensure float
            smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes"])
       
        
        
            stress_level = st.selectbox("Stress Level", [0, 1, 2, 3], format_func=lambda x: ["None", "Low", "Moderate", "High"][x])
            ptsd = st.selectbox("PTSD", [0, 1], format_func=lambda x: ["Yes", "No"][x])
            depression_level = st.selectbox("Depression Level", [0, 1, 2], format_func=lambda x: ["Mild", "Moderate", "Severe"][x])
            diabetes_type = st.selectbox("Diabetes Type", [0, 1, 2, 3], format_func=lambda x: ["None", "Type 1", "Type 2", "Gestational"][x])
            chronic_pain = st.selectbox("Chronic Pain", ["None", "Rheumatism", "Osteoarthritis", "Others"])
            sleep_hours = st.slider("Sleep Hours", 3.0, 10.0, 7.0)
            hypertension_treatment = st.selectbox("Hypertension Treatment", ["None", "Herbal", "Drugs"])
            salt_intake = st.selectbox("Salt Intake", ["None", "Little", "Moderate", "High"])
            noise_sources = st.selectbox("Noise Sources", ["Mosque", "Church", "Market", "Block-Industry", "Grinding-Machine", "Welder", "Club-House",    
                                                           "Generator", "None"])
            pollution_level_air = st.selectbox("Air Pollution Level", ["None", "Low", "Moderate", "High"])
            pollution_level_water = st.selectbox("Water Pollution Level", ["None", "Low", "Moderate", "High"])
            pollution_level_environmental = st.selectbox("Environmental Pollution Level", ["None", "Low", "Moderate", "High"])
            CustomStressScore = st.number_input("Custom Stress Score", min_value=0, max_value=100, value=5)
        
            st.subheader("🧠 Additional Stress Assessment")
            _, _, stress_score = custom_stress_score(use_container=True)
            st.session_state.stress_score = stress_score

            submit_stroke_inputs = st.form_submit_button("Predict Stroke Risk")

            # Collect raw inputs
    if submit_stroke_inputs:
            raw_inputs = {
                'age': age,
                'avg_glucose_level': avg_glucose_level,
                'bmi': bmi,
                'stress_level': stress_level,
                'ptsd': 1 if ptsd == "Yes" else 0,
                'depression_level': depression_level,
                'diabetes_type': 0 if diabetes_type == "None" else 1,
                'sleep_hours': sleep_hours,
                'gender': gender,
                'work_type': work_type,
                'Residence_type': residence_type,
                'smoking_status': smoking_status,
                "SystolicBP": systolic_bp,
                "DiastolicBP": diastolic_bp,
                'salt_intake': salt_intake,
                'noise_sources': noise_sources,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "marital_status": marital_status,
                "chronic_pain": chronic_pain,
                "hypertension_treatment": hypertension_treatment,
                "pollution_level_air": pollution_level_air,
                "pollution_level_water": pollution_level_water,
                "pollution_level_environmental": pollution_level_environmental,
                'ethnicity': selected_ethnicity,
                'Country': selected_country,
                'Province_Option': selected_province,
                'CustomStressScore': CustomStressScore
            }

            # You may need to update the above dictionary to include all required fields

    if submit_stroke_inputs:
    # Prepare raw inputs dict#
        try: 
            stroke_inputs_df = prepare_stroke_input_robust(raw_inputs)  # Adjust to your function name
            pred = stroke_model.predict(stroke_inputs_df)[0]   
            inputs = {
                "user_id": st.session_state.user['id'] if st.session_state.get('user') else "anonymous",
                "age": age,
                "gender": gender,
                "heart_disease": heart_disease,
                "hypertension": hypertension,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "avg_glucose": avg_glucose,
                "bmi": bmi,
                "marital_status": marital_status,
                "work_type": work_type,
                "residence_type": residence_type,
                "avg_glucose_level": avg_glucose_level,
                "smoking_status": smoking_status,
                "stress_level": stress_level,
                "ptsd": ptsd,
                "depression_level": depression_level,
                "diabetes_type": diabetes_type,
                "chronic_pain": chronic_pain,
                "sleep_hours": sleep_hours,
                "hypertension_treatment": hypertension_treatment,
                "salt_intake": salt_intake,
                "noise_sources": noise_sources,
                "pollution_level_air": pollution_level_air,
                "pollution_level_water": pollution_level_water,
                "pollution_level_environmental": pollution_level_environmental,
                "custom_stress_score": CustomStressScore,
                "ethnicity": selected_ethnicity,
                "country": selected_country,
                "province_option": selected_province,
                "prediction_result": float(pred)
        }

    # Convert raw inputs into DataFrame for prediction #
            stroke_df = prepare_stroke_input_robust(inputs)

    # Predict#
            pred = stroke_model.predict(stroke_inputs_df)[0]

    # Location#
            city, region, country = get_user_location()
            location_str = f"{city}, {region}, {country}"


     # Display prediction result with detailed advice#
            if pred == 1:
                st.error("⚠️ HIGH STROKE RISK DETECTED")
                st.markdown("""
            ## 🚨 Immediate Action Recommended:
            - **Consult a healthcare provider immediately**
            - **Add Saigon Cinnamon and Alligator Pepper to diet**
            - Monitor blood pressure daily
            - Avoid strenuous activities
            - Reduce salt intake to <5g/day
            - Increase consumption of leafy greens
            - Limit fried foods and processed meals
            - Eat more fruits and vegetables (rich in potassium and fiber)
            - Maintain regular sleep schedule
            - ⚖️ Maintain a healthy weight (avoid obesity)
            """)
            else:
                st.success("✅ LOW STROKE RISK DETECTED")

        # Lifestyle suggestions expander
            with st.expander("🛠️ Lifestyle Suggestions for Stroke Prevention"):
                st.markdown("""
            ### 🍽️ Dietary Recommendations:
            - Reduce salt intake to <5g/day
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
            🧘 Prioritize 7–8 hours of sleep per night  
            🌿 **Take cinnamon (e.g., Saigon cinnamon) regularly**  
            🧪 *Reduces blood sugar, inflammation, and oxidative stress*  
            🩺 *Supports brain and heart health naturally*  
            🚭 Stop smoking and reduce alcohol intake  
            ⚖️ Maintain a healthy weight (avoid obesity)
            """)


    # Prepare database dictionary (separate from stroke_df)
            db_payload = {
        "user_id": st.session_state.user['id'] if st.session_state.get('user') else "anonymous",
        "raw_inputs": inputs,
        "location": location_str,
        "prediction_result": float(pred)
    }

    # Save to Supabase#
            response = supabase.table("stroke_predictions").insert(inputs).execute()
            if response.data:
                st.success("Stroke prediction saved to database!")
            else:
                st.error(f"Failed to save Stroke prediction: {response.error}")

        except Exception as e:
                st.error(f"Error during Stroke prediction or saving: {e}")

             

def build_full_input(raw):
    # Map gender
    gender = 1 if raw.get("gender") == "Male" else 0

    # Head injury mapping
    head_map = {"None": 0, "Accident": 1, "Violence": 2}
    head_injury = head_map.get(raw.get("HeadInjury", "None"), 0)

    # Cognitive symptoms mapping
    option_map = {"Yes": 1, "No": 0, "Sometimes": 0.5}

    full_input = {
        "Age": raw.get("age", 65),
        "Gender": gender,
        "BMI": raw.get("bmi", 25.0),
        "EducationLevel": raw.get("EducationLevel", 12),
        "Smoking": int(raw.get("Smoking", 0)),
        "AlcoholConsumption": raw.get("AlcoholConsumption", 2),
        "PhysicalActivity": raw.get("PhysicalActivity", 3),
        "DietQuality": raw.get("DietQuality", 3),
        "SleepQuality": raw.get("SleepQuality", 3),
        "FamilyHistoryAlzheimers": int(raw.get("FamilyHistoryAlzheimers", 0)),
        "CardiovascularDisease": int(raw.get("CardiovascularDisease", 0)),
        "Diabetes": int(raw.get("Diabetes", 0)),
        "Depression": int(raw.get("Depression", 0)),
        "Hypertension": int(raw.get("Hypertension", 0)),
        "SystolicBP": raw.get("SystolicBP", 120),
        "DiastolicBP": raw.get("DiastolicBP", 80),
        "CholesterolTotal": raw.get("CholesterolTotal", 200),
        "CholesterolLDL": raw.get("CholesterolLDL", 130),
        "CholesterolHDL": raw.get("CholesterolHDL", 50),
        "CholesterolTriglycerides": raw.get("CholesterolTriglycerides", 150),
        "FunctionalAssessment": raw.get("FunctionalAssessment", 0),
        "BehavioralProblems": int(raw.get("BehavioralProblems", 0)),
        "ADL": raw.get("ADL", 6),
        "Confusion": option_map.get(raw.get("Confusion", "No"), 0),
        "Disorientation": option_map.get(raw.get("Disorientation", "No"), 0),
        "PersonalityChanges": option_map.get(raw.get("PersonalityChanges", "No"), 0),
        "DifficultyCompletingTasks": option_map.get(raw.get("DifficultyCompletingTasks", "No"), 0),
        "Forgetfulness": option_map.get(raw.get("Forgetfulness", "No"), 0),
        "MemoryComplaints": option_map.get(raw.get("MemoryComplaints", "No"), 0),
        "HeadInjury": head_injury,
        "ethnicity": raw.get("ethnicity", "Other"),
        "Country": raw.get("Country", "Nigeria"),
        "Province_Option": raw.get("Province_Option", "Lagos"),
        "CustomStressScore": raw.get("CustomStressScore", 50)
    }
 # ---- Convert to DataFrame ----
    alz_data_df = pd.DataFrame([full_input])
    return alz_data_df


def prepare_alz_data_robust(full_input):
    """
    Ensure Alzheimer's input is returned as a single-row DataFrame matching the model's expected format.
    """

    if isinstance(full_input, pd.DataFrame):
        if len(full_input) == 1:
            return full_input.reset_index(drop=True)
        else:
            raise ValueError("Expected a single-row DataFrame, got multiple rows.")

    elif isinstance(full_input, dict):
        return pd.DataFrame([full_input])

    elif isinstance(full_input, list):
        if len(full_input) == 1 and isinstance(full_input[0], dict):
            return pd.DataFrame(full_input)
        else:
            raise ValueError("List input must contain exactly one dictionary.")

    else:
        raise TypeError("Input must be a dict, list of dicts, or single-row DataFrame.")


                            # --- Alzheimer Predictor ---#
if app_mode == "Alzheimer Risk Prediction":
    st.title("🧠 Alzheimer Risk Predictor")

# =======================#
# TAB 1: ALZHEIMER FORM #
# =======================#
    
    st.warning("Complete all fields for accurate assessment")
    
    # Get nutritional score #
    nutritional_score = compute_nutritional_score()
    st.info(f"🍎 Nutritional Health Score: **{nutritional_score}/5**")
    
    user_id = st.session_state.user['id'] if st.session_state.user else "anonymous"
    with st.form("alz_form"):
        age = st.number_input("Age", 0, 100, 65, key='alz_age')
        gender = 1 if st.selectbox("Gender", ["Male", "Female"], key='alz_gender') == "Male" else 0
        education_years = st.slider("Education Level (Years)", 0, 20, 12, key='alz_eduyears')
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0, key='alz_bmi')
        is_smoker = st.selectbox("Smoking", [0, 1], key='alz_smoking')
        alcohol_consumption = st.slider("Alcohol Consumption (0 = None, 5 = High)", 0, 5, 2, key='alz_alcohol')
        physical_activity = st.slider("Physical Activity (hrs/week)", 0, 20, 3, key='alz_activity')
        diet_quality = nutritional_score
        sleep_quality = st.slider("Sleep Quality (1-5)", 1, 5, 3, key='alz_sleep')
        family_history_alz = st.selectbox("Family History of Alzheimer's", [0, 1], format_func=lambda x: ["Yes", "No"][x], key='alz_family')
        cardiovascular_disease = st.selectbox("Cardiovascular Disease", [0, 1], format_func=lambda x: ["Yes", "No"][x], key='alz_cardio')


        diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: ["Yes", "No"][x], key='alz_diabetes')
        depression = st.selectbox("Depression", [0, 1], format_func=lambda x: ["Yes", "No"][x], key='alz_depression')
        hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: ["Yes", "No"][x], key='alz_hypertension')
        systolic_bp = st.number_input("Systolic BP", 80, 220, 120, key='alz_systolic')
        diastolic_bp = st.number_input("Diastolic BP", 50, 150, 80, key='alz_diastolic')
        cholesterol_total = st.number_input("Total Cholesterol", 100, 400, 200, key='alz_chol_total')
        cholesterol_ldl = st.number_input("LDL", 50, 300, 130, key='alz_ldl')
        cholesterol_hdl = st.number_input("HDL", 20, 100, 50, key='alz_hdl')
        cholesterol_triglycerides = st.number_input("Triglycerides", 50, 500, 150, key='alz_trig')
        functional_assessment = st.slider("Functional Assessment (0-5)", 0, 5, 0, key='alz_func')
        behavioral_problems = st.selectbox("Behavioral Problems", [0, 1], format_func=lambda x: ["Yes", "No"][x], key='alz_behavior')
        adl = st.slider("ADL Score (Activities of Daily Living)", 0, 6, 6, key='alz_adl')
    
        st.subheader("Cognitive Assessment")
        cognitive_col, ethnic_col = st.columns(2)
    
        # Cognitive symptoms mapping #
        option_map = {"Yes": 1, "No": 0, "Sometimes": 0.5}
        confusion = option_map[st.selectbox("Confusion", ["Yes", "No", "Sometimes"], key='alz_confusion')]
        disorientation = option_map[st.selectbox("Disorientation", ["Yes", "No", "Sometimes"], key='alz_disorien')]
        personality_changes = option_map[st.selectbox("Personality Changes", ["Yes", "No", "Sometimes"],   
                                                      key='alz_personality')]
        difficulty_tasks = option_map[st.selectbox("Difficulty Completing Tasks", ["Yes", "No", "Sometimes"], 
                                                   key='alz_tasks')]
        forgetfulness = option_map[st.selectbox("Forgetfulness", ["Yes", "No", "Sometimes"], key='alz_forget')]
        memory_complaints = option_map[st.selectbox("Memory Complaints", ["Yes", "No", "Sometimes"], key='alz_memory')]
        
        # Head injury mapping #
        head_map = {"None": 0, "Accident": 1, "Violence": 2}
        head_injury = head_map[st.selectbox("Head Injury", ["None", "Accident", "Violence"], key='alz_head')]
        
    
        # ===== CULTURALLY ADAPTED MMSE ASSESSMENT ===== #
        st.subheader("MMSE Assessment (Adapted for African Context)")
        mmse_option = st.radio("Do you know your MMSE score?", ["Estimate using cultural questions"], key='alz_mmse_radio')

        st.info("Answer these culturally relevant questions to estimate your MMSE score:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            q1 = st.selectbox("Do you forget names of relatives/village members?", 
                         ["Never", "Sometimes", "Often"], key='q1')
            q2 = st.selectbox("Do you misplace important items (farming tools, keys)?", 
                         ["Never", "Sometimes", "Often"], key='q2')
            q3 = st.selectbox("Can you recall traditional recipes or remedies?", 
                         ["Always", "Sometimes", "Rarely"], key='q3')
            
        with col2:
            q4 = st.selectbox("Do you recognize people from your community?", 
                         ["Always", "Sometimes", "Rarely"], key='q4')
            q5 = st.selectbox("Can you navigate familiar paths/markets?", 
                         ["Always", "Sometimes", "Rarely"], key='q5')
            q6 = st.selectbox("Do you remember important cultural events/dates?", 
                         ["Always", "Sometimes", "Rarely"], key='q6')
        
        # Calculate estimated MMSE with cultural weighting#
        response_scores = {
            "Never": 2, "Sometimes": 1, "Often": 0,"Always": 2, "Rarely": 0
        }
        
        weights = {
            "q1": 2.0,   # Names of relatives
            "q2": 1.5,   # Important items
            "q3": 1.0,   # Traditional knowledge
            "q4": 1.7,   # Community recognition
            "q5": 2.0,   # Navigation
            "q6": 1.3    # Cultural events
        }
        
        mmse_score = 20 + ( 
            response_scores[q1] * weights["q1"] +
            response_scores[q2] * weights["q2"] +
            response_scores[q3] * weights["q3"] +
            response_scores[q4] * weights["q4"] +
            response_scores[q5] * weights["q5"] +
            response_scores[q6] * weights["q6"]
        )
        # ================================#

        if age > 70:
            mmse_score *= 0.95
        elif age > 60:
            mmse_score *= 0.97

        mmse = max(0, min(30, round(mmse_score)))
        st.info(f"Estimated MMSE Score: **{mmse}/30**")

        st.caption("*Note: This assessment emphasizes culturally significant cognitive functions.*")

        stress_score = st.slider("Stress Level", 0, 10, 5)
        st.session_state.stress_score = stress_score

        submit_alz = st.form_submit_button("🔍 Predict Alzheimer Risk")

    if submit_alz:
        pred = 1 if mmse < 24 or stress_score > 6 else 0

        if pred == 1:
            st.error("⚠️ HIGH ALZHEIMER RISK DETECTED")
            st.markdown("""## 🚨 Immediate Action Recommended:
- **Consult a healthcare provider immediately**
- Begin cognitive training exercises
- Review family medical history
- 🧩 Do mental exercises (e.g., puzzles, memory games)
- 🏃 Stay physically active (exercise increases brain health)
- 🧘 Reduce stress — practice mindfulness or prayer
- 👥 Stay socially engaged — talk to friends, join a group
- 🥦 Eat brain-healthy foods (nuts, omega-3s, leafy greens)
- 🌿 **Use cinnamon regularly** – may protect memory and reduce inflammation
- 🚭 Avoid smoking and limit alcohol
- 💤 Prioritize sleep and manage depression
""")
        else:
            st.success("✅ LOW ALZHEIMER'S RISK DETECTED")

        # ===========================
        # 🛠️ Cognitive Health Advice
        # ===========================
        with st.expander("🛠️ Cognitive Health Suggestions"):
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

# ============================#
# TAB 2: MEMORY RECALL GAME#
# ============================#
    st.subheader("🎮 Memory Recall Game")

    WORD_POOL = [
        "apple", "table", "river", "mountain", "sun", "flower",
        "clock", "phone", "book", "star", "moon", "chair",
        "pencil", "car", "glass", "tree", "music", "house",
        "cloud", "lamp", "keyboard", "shoe", "bottle", "ring"
    ]

    if 'memory_game' not in st.session_state:
        st.session_state.memory_game = {
            "state": "start",
            "words": [],
            "start_time": None,
            "level": 1,
            "score_history": []
        }

    game = st.session_state.memory_game

    if game["state"] == "start":
        st.markdown(f"**Level {game['level']}** - You will see {4 + game['level']} words.")
        if st.button("Start Memory Exercise"):
            num_words = 4 + game["level"]
            words = random.sample(WORD_POOL, num_words)
            game["words"] = words
            game["start_time"] = time.time()
            game["state"] = "showing"
            st.rerun()

    elif game["state"] == "showing":
        st.write("Memorize these words (5 seconds):")
        st.info(", ".join(game["words"]))

        if time.time() - game["start_time"] > 5:
            game["state"] = "recalling"
            st.rerun()

    elif game["state"] == "recalling":
        with st.form("recall_form"):
            recalled_input = st.text_input("Type the words you remember, separated by commas:")
            submit = st.form_submit_button("Submit Recall")

            if submit:
                recalled = [w.strip().lower() for w in recalled_input.split(",") if w.strip()]
                correct_words = set(w.lower() for w in game["words"])
                recalled_set = set(recalled)

                correct_count = len(correct_words & recalled_set)

                st.success(f"You recalled {correct_count} out of {len(game['words'])} correctly.")

                if correct_count >= len(game['words']) - 1:
                    st.balloons()
                    st.info("🎉 Great job! You advance to the next level.")
                    game["level"] += 1
                else:
                    st.warning("You'll stay on the same level to improve.")

                game["score_history"].append({
                    "level": game["level"],
                    "correct": correct_count,
                    "total": len(game["words"]),
                    "words": game["words"],
                    "recalled": recalled
                })

                game["state"] = "start"
                game["words"] = []
                game["start_time"] = None
                st.rerun()

        if game["score_history"]:
            with st.expander("📊 Score History"):
                for i, score in enumerate(reversed(game["score_history"])):
                    st.write(f"**Round {len(game['score_history']) - i}**: Level {score['level']} - {score['correct']}/{score['total']} correct")

# ================
# 🧮 Prediction Logic
# ================
    if submit_alz:
            # Prepare data for prediction
        alz_inputs = {
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
                "FunctionalAssessment": functional_assessment,
                "BehavioralProblems": behavioral_problems,
                "ADL": adl,
                "Confusion": confusion,
                "Disorientation": disorientation,
                "PersonalityChanges": personality_changes,
                "DifficultyCompletingTasks": difficulty_tasks,
                "Forgetfulness": forgetfulness,
                "MemoryComplaints": memory_complaints,
                "HeadInjury": head_injury,
                "ethnicity": selected_ethnicity,
                "Country": selected_country,
                "Province_Option": selected_province,
                "CustomStressScore": st.session_state.stress_score
            }
    if submit_alz:
        try:
            # Convert raw inputs into proper DataFrame for prediction
            alz_inputs_df = prepare_alz_data_robust(alz_inputs)

            # Predict Alzheimer’s risk
            pred = int(alz_model.predict(alz_inputs_df)[0])

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
                "ethnicity": selected_ethnicity,
                "country": selected_country,
                "province_option": selected_province,
                "custom_stress_score": st.session_state.get("stress_score", None),
                "location": location_str,
                "prediction_result": int(pred)
        }

            response = supabase.table("alzheimers_predictions").insert(alz_data).execute()
            if response.data:
                st.success("alzheimers prediction saved to database!")
            else:
                st.error(f"Failed to save alzheimers prediction: {response.error}")

        except Exception as e:
                st.error(f"Error during alzheimers prediction or saving: {e}")

