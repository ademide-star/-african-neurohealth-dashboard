import streamlit as st

# Must be the very first Streamlit command
st.set_page_config(
    page_title="AFRICAN NEUROHEALTH",
    page_icon="📊",
    layout="centered",
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
            font-size: 1.1rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

permanent_sidebar = """
    <style>
    section[data-testid="stSidebar"] {
        min-width: 250px !important;
        max-width: 250px !important;
    }
    button[kind="header"] {
        display: none !important;
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

# --- Translation Dictionary ---
TRANSLATIONS = {
    "en": {
        # Authentication
        "login": "Login",
        "register": "Register",
        "email": "Email",
        "password": "Password",
        "new_email": "New Email",
        "new_password": "New Password",
        "confirm_password": "Confirm Password",
        "logged_in_as": "Logged in as",
        "invalid_credentials": "Invalid login credentials",
        "registration_success": "Registration successful! Please check your email to confirm your account.",
        "logout_success": "Logged out successfully.",
        "logout_error": "Logout error",
        "passwords_not_match": "Passwords do not match",
        "login_with_email": "Login with Email & Password",
        
        # Navigation
        "navigation": "Navigation",
        "go_to": "Go to",
        "about": "About",
        "stroke_prediction": "Stroke Prediction",
        "alzheimers_prediction": "Alzheimer's Prediction",
        "memory_recall_game": "Memory Recall Game",
        "nutrition_tracker": "Nutrition Tracker",
        
        # Common UI
        "select_language": "Select Language",
        "complete_all_fields": "Complete all fields for accurate assessment",
        "submit": "Submit",
        "save": "Save",
        "predict": "Predict",
        "high_risk": "HIGH RISK DETECTED",
        "low_risk": "LOW RISK DETECTED",
        "welcome": "Welcome to African NeuroHealth Dashboard",
        
        # Form Fields
        "age": "Age",
        "gender": "Gender",
        "male": "Male",
        "female": "Female",
        "weight": "Weight",
        "height": "Height",
        "blood_group": "Blood Group",
        "genotype": "Genotype",
        "select": "Select",
        "select_gender": "Select Gender",
        "select_blood_group": "Select Blood Group",
        "select_genotype": "Select Genotype",
        
        # Stroke Specific
        "predict_stroke_risk": "Predict Stroke Risk",
        "stroke_risk_predictor": "Stroke Risk Predictor",
        "heart_disease": "Heart Disease",
        "hypertension": "Hypertension",
        "systolic_bp": "Systolic BP",
        "diastolic_bp": "Diastolic BP",
        "marital_status": "Marital Status",
        "work_type": "Work Type",
        "residence_type": "Residence Type",
        "avg_glucose_level": "Average Glucose Level",
        "smoking_status": "Smoking Status",
        "stress_level": "Stress Level",
        "ptsd": "PTSD",
        "depression_level": "Depression Level",
        "diabetes_type": "Diabetes Type",
        "chronic_pain": "Chronic Pain",
        "sleep_hours": "Sleep Hours",
        "hypertension_treatment": "Hypertension Treatment",
        "salt_intake": "Salt Intake",
        "noise_sources": "Noise Sources",
        "air_pollution": "Air Pollution Level",
        "water_pollution": "Water Pollution Level",
        "environmental_pollution": "Environmental Pollution Level",
        "custom_stress_score": "Custom Stress Score",
        
        # Alzheimer's Specific
        "alzheimers_predictor": "Alzheimer's Predictor",
        "predict_alzheimers_risk": "Predict Alzheimer Risk",
        "education_level": "Education Level",
        "physical_activity": "Physical Activity",
        "alcohol_consumption": "Alcohol Consumption",
        "sleep_quality": "Sleep Quality",
        "family_history": "Family History of Alzheimer's",
        "cardiovascular_disease": "Cardiovascular Disease",
        "depression": "Depression",
        "cholesterol_total": "Total Cholesterol",
        "cholesterol_ldl": "LDL",
        "cholesterol_hdl": "HDL",
        "triglycerides": "Triglycerides",
        "functional_assessment": "Functional Assessment",
        "behavioral_problems": "Behavioral Problems",
        "adl": "ADL Score",
        "confusion": "Confusion",
        "disorientation": "Disorientation",
        "personality_changes": "Personality Changes",
        "difficulty_tasks": "Difficulty Completing Tasks",
        "forgetfulness": "Forgetfulness",
        "memory_complaints": "Memory Complaints",
        "head_injury": "Head Injury",
        
        # Memory Game
        "memory_recall_game": "Memory Recall Game",
        "start_memory_exercise": "Start Memory Exercise",
        "level": "Level",
        "you_will_see_words": "You will see",
        "words": "words",
        "memorize_words": "Memorize these words (5 seconds):",
        "type_words_remembered": "Type the words you remember, separated by commas:",
        "submit_recall": "Submit Recall",
        
        # Nutrition Tracker
        "nutrition_tracker": "Nutrition Tracker",
        "fruit_intake": "Fruit Intake (servings per day)",
        "vegetable_intake": "Vegetable Intake (servings per day)",
        "water_intake": "Water Intake (liters per day)",
        "supplements_used": "Supplements Used",
        "natural_herbs": "Natural Herbs Taken",
        "select_lifestyles": "Select Nutritional Lifestyles",
        "track_consumption": "Track Consumption",
        "frequency": "Frequency",
        "servings": "Servings",
        "save_nutrition_data": "Save Nutritional Data",
        "nutritional_health_score": "Nutritional Health Score",
        
        # Location
        "location_info": "Location Information",
        "select_country": "Select Country",
        "select_province": "Select Province",
        "select_region": "Select Region",
        "select_ethnicity": "Select Ethnicity",
        
        # About Page
        "about_this_app": "About This App",
        "african_neurohealth": "African NeuroHealth Dashboard",
        "about_description": """This platform is a culturally attuned, context-aware diagnostic tool tailored for assessing neuro-health risks in African populations. 
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
Their collaborative spirit and commitment to innovation helped bring this vision to life.**""",
        
        # Stress Assessment
        "stress_estimator": "Stress Estimator Based on Cultural & Contextual Stress Factors",
        "financial_pressure": "Financial pressure/burden",
        "family_issues": "Family/relationship issues",
        "work_stress": "Work/employment stress",
        "safety_concerns": "Community safety concerns",
        "caregiver_burden": "Caregiver burden",
        "migration_stress": "Migration/displacement stress",
        "family_expectations": "Traditional family expectations",
        "religious_conflicts": "Spiritual/religious conflicts",
        "total_stress_score": "Total Stress Score",
        "low_stress": "Low",
        "moderate_stress": "Moderate",
        "high_stress": "High",
        "additional_stress_assessment": "Additional Stress Assessment",
        
        # BMI
        "bmi_assessment": "BMI Assessment (Body Mass Index)",
        "enter_weight": "Enter your weight (kg)",
        "enter_height": "Enter your height (m)",
        "calculated_bmi": "Calculated BMI",
        "bmi_category": "BMI Category",
        "underweight": "Underweight",
        "normal_weight": "Normal weight",
        "overweight": "Overweight",
        "obese": "Obese",
        
        # Messages
        "please_login": "Please log in to access this feature",
        "logged_in_success": "✅ Logged in as",
        "registration_failed": "Registration failed.",
        "login_error": "Login error",
        "registration_error": "Registration error",
        
        # Missing translations
        "additional_nutrition_details": "Additional Nutrition Details",
        "stress_level": "Stress Level",
        "track_consumption": "Track Consumption",
    },
    "ar": {
        "login": "تسجيل الدخول",
        "register": "تسجيل",
        "email": "البريد الإلكتروني",
        "password": "كلمة المرور",
        "new_email": "بريد إلكتروني جديد",
        "new_password": "كلمة مرور جديدة",
        "confirm_password": "تأكيد كلمة المرور",
        "logged_in_as": "تم تسجيل الدخول باسم",
        "invalid_credentials": "بيانات تسجيل دخول غير صالحة",
        "registration_success": "تم التسجيل بنجاح! يرجى التحقق من بريدك الإلكتروني لتأكيد حسابك.",
        "logout_success": "تم تسجيل الخروج بنجاح.",
        "logout_error": "خطأ في تسجيل الخروج",
        "passwords_not_match": "كلمات المرور غير متطابقة",
        "login_with_email": "تسجيل الدخول بالبريد الإلكتروني وكلمة المرور",
    },
    "sw": {
        "login": "Ingia",
        "register": "Jisajili",
        "email": "Barua Pepe",
        "password": "Nenosiri",
        "new_email": "Barua Pepe Mpya",
        "new_password": "Nenosiri Jipya",
        "confirm_password": "Thibitisha Nenosiri",
        "logged_in_as": "Umeingia kama",
        "invalid_credentials": "Maelezo ya kuingia sio sahihi",
        "registration_success": "Usajili umefanikiwa! Tafadhali angalia barua pepe yako kuthibitisha akaunti yako.",
        "logout_success": "Umetoka kwa mafanikio.",
        "logout_error": "Hitilafu ya kutoka",
        "passwords_not_match": "Nenosiri halifanani",
        "login_with_email": "Ingia na Barua Pepe na Nenosiri",
    },
    "fr": {
        "login": "Connexion",
        "register": "S'inscrire",
        "email": "E-mail",
        "password": "Mot de passe",
        "new_email": "Nouvel e-mail",
        "new_password": "Nouveau mot de passe",
        "confirm_password": "Confirmer le mot de passe",
        "logged_in_as": "Connecté en tant que",
        "invalid_credentials": "Identifiants de connexion invalides",
        "registration_success": "Inscription réussie ! Veuillez vérifier votre e-mail pour confirmer votre compte.",
        "logout_success": "Déconnexion réussie.",
        "logout_error": "Erreur de déconnexion",
        "passwords_not_match": "Les mots de passe ne correspondent pas",
        "login_with_email": "Se connecter avec e-mail et mot de passe",
    }
}

# Helper function to get translated text
def t(key):
    lang = st.session_state.get('current_lang', 'en')
    return TRANSLATIONS.get(lang, {}).get(key, TRANSLATIONS['en'].get(key, key))

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
# SESSION MANAGEMENT
# ----------------------------
if "user" not in st.session_state:
    st.session_state.user = {"id": None, "email": None}
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "refresh_token" not in st.session_state:
    st.session_state.refresh_token = None
if "current_lang" not in st.session_state:
    st.session_state.current_lang = "en"

# ----------------------------
# LOGIN FUNCTION
# ----------------------------
def login():
    st.subheader(t("login_with_email"))
    email = st.text_input(t("email"), key="login_email")
    password = st.text_input(t("password"), type="password", key="login_password")

    if st.button(t("login"), key="login_btn"):
        try:
            response = supabase.auth.sign_in_with_password({"email": email, "password": password})
            if response.user:
                st.session_state.user = {"id": response.user.id, "email": response.user.email}
                st.success(f"✅ {t('logged_in_as')} {st.session_state.user['email']}")
            else:
                st.error(t("invalid_credentials"))
        except Exception as e:
            st.error(f"{t('login_error')}: {e}")

# ----------------------------
# LOGOUT FUNCTION
# ----------------------------
def logout():
    try:
        supabase.auth.sign_out()
        st.session_state.user = {"id": None, "email": None}
        st.success(t("logout_success"))
        st.rerun()
    except Exception as e:
        st.error(f"{t('logout_error')}: {e}")

# ----------------------------
# REGISTER FUNCTION
# ----------------------------
def register():
    st.subheader(t("register"))
    email = st.text_input(t("new_email"), key="register_email")
    password = st.text_input(t("new_password"), type="password", key="register_password")
    confirm_password = st.text_input(t("confirm_password"), type="password", key="register_confirm_password")

    if st.button(t("register"), key="register_btn"):
        if password != confirm_password:
            st.error(t("passwords_not_match"))
        else:
            try:
                response = supabase.auth.sign_up({"email": email, "password": password})
                if response.user:
                    st.success(t("registration_success"))
                else:
                    st.error(t("registration_failed"))
            except Exception as e:
                st.error(f"{t('registration_error')}: {e}")

# ----------------------------
# ABOUT FUNCTION
# ----------------------------
def about():
    st.header(f"ℹ️ {t('about_this_app')}")
    st.title(f"🧠 {t('african_neurohealth')}")
    st.markdown(t("about_description"))

# ----------------------------
# STRESS SCORE FUNCTION
# ----------------------------
def custom_stress_score(prefix="", use_container=False):
    title = f"🧮 {t('stress_estimator')}"
    
    if use_container:
        container = st.container()
        container.header(title)
    else:
        container = st.expander(title)
    
    with container:
        q1 = st.slider(t("financial_pressure"), 0, 4, 2)
        q2 = st.slider(t("family_issues"), 0, 4, 2)
        q3 = st.slider(t("work_stress"), 0, 4, 2)
        q4 = st.slider(t("safety_concerns"), 0, 4, 2)
        q5 = st.slider(t("caregiver_burden"), 0, 4, 2)
        q6 = st.slider(t("migration_stress"), 0, 4, 2)
        q7 = st.slider(t("family_expectations"), 0, 4, 2)
        q8 = st.slider(t("religious_conflicts"), 0, 4, 2)
        
        total_score = q1 + q2 + q3 + q4 + q5 + q6 + q7 + q8
        
        if total_score <= 12:
            level = 0
            label = t("low_stress")
            color = "green"
        elif total_score <= 20:
            level = 1
            label = t("moderate_stress")
            color = "orange"
        else:
            level = 2
            label = t("high_stress")
            color = "red"
 
        st.markdown(f"""
        <div style='padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-top: 20px;'>
            <h4>🧠 {t('total_stress_score')}: <span style='color:{color};'>{total_score}/32</span> → {label}</h4>
            <p><small>Higher scores indicate greater exposure to Africa-specific stressors</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        return level, label, total_score

# ----------------------------
# MODEL LOADING
# ----------------------------
def smart_load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return cloudpickle.load(f)

# Get the current directory
current_dir = Path(__file__).resolve().parent

# Define model paths
ALZ_MODEL_PATH = current_dir / "alz_model_v17.joblib"
STROKE_MODEL_PATH = current_dir / "stroke_model_v17.joblib"

@st.cache_resource
def load_models():
    try:
        if not ALZ_MODEL_PATH.exists():
            st.error(f"Alzheimer's model file not found at {ALZ_MODEL_PATH}")
            return None, None
            
        if not STROKE_MODEL_PATH.exists():
            st.error(f"Stroke model file not found at {STROKE_MODEL_PATH}")
            return None, None
            
        alz_model = joblib.load(ALZ_MODEL_PATH)
        stroke_model = joblib.load(STROKE_MODEL_PATH)
        return alz_model, stroke_model
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Load models into session state
if "stroke_model" not in st.session_state or "alz_model" not in st.session_state:
    alz_model, stroke_model = load_models()
    
    if alz_model and stroke_model:
        st.session_state.alz_model = alz_model
        st.session_state.stroke_model = stroke_model
    else:
        st.error("Failed to load models. Please check model files.")
        st.stop()

# ----------------------------
# COUNTRIES AND ETHNICITIES DATA
# ----------------------------
countries_with_provinces = {
    "Nigeria": ["Abia", "Adamawa", "Akwa Ibom", "Anambra", "Bauchi", "Bayelsa", "Benue", "Borno", "Cross River", "Delta"],
    "Ghana": ["Greater Accra", "Ashanti", "Western", "Eastern", "Volta", "Northern", "Upper East", "Upper West", "Bono"],
    "Kenya": ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Kiambu", "Machakos", "Uasin Gishu", "Meru", "Embu"],
    "South Africa": ["Gauteng", "Western Cape", "Eastern Cape", "Northern Cape", "KwaZulu-Natal", "Free State", "North West"],
}

region_with_ethnicity = {
    "North Africa": ["Amazigh (Berber)", "Arab", "Bedouin", "Coptic", "Nubian", "Tuareg"],
    "West Africa": ["Yoruba", "Hausa", "Igbo", "Fulani", "Akan", "Ashanti", "Ewe", "Fon", "Ga"],
    "Central Africa": ["Bantu", "Kongo", "Luba", "Mongo", "Teke", "Sanga", "Pygmy"],
    "East Africa": ["Amhara", "Tigray", "Oromo", "Somali", "Afar", "Sidama", "Gurage"],
    "Southern Africa": ["Shona", "Ndebele", "Zulu", "Xhosa", "Sotho", "Tswana", "Swazi"],
}

# Create encoding maps
country_map = {country: i for i, country in enumerate(countries_with_provinces.keys())}
province_map = {}
for c, provinces in countries_with_provinces.items():
    province_map.update({p: i for i, p in enumerate(provinces)})
region_map = {region: i for i, region in enumerate(region_with_ethnicity.keys())}
ethnicity_map = {}
for r, ethnicities in region_with_ethnicity.items():
    ethnicity_map.update({e: i for i, e in enumerate(ethnicities)})

# ----------------------------
# NUTRITION TRACKER
# ----------------------------
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

def nutrition_tracker_app():
    st.header(t("nutrition_tracker"))
    st.title(f"🥗 {t('nutrition_tracker')}")
    
    # Nutrition inputs
    fruit_intake = st.number_input(t("fruit_intake"), min_value=0, max_value=20, value=2)
    vegetable_intake = st.number_input(t("vegetable_intake"), min_value=0, max_value=20, value=3)
    hydration_liters = st.number_input(t("water_intake"), min_value=0.0, max_value=10.0, value=2.0)
    supplements_used = st.text_input(t("supplements_used"))
    natural_herbs = st.text_input(t("natural_herbs"))
    
    # Lifestyle selection
    all_options = ["Local Bukka/Street Food", "Homemade Food", "Junk Food", "Fast Foods", "Vegetarian", "Vegan"]
    selected_lifestyles = st.multiselect(t("select_lifestyles"), all_options)
    
    nutritional_data = {}
    if selected_lifestyles:
        for lifestyle in selected_lifestyles:
            col1, col2 = st.columns(2)
            with col1:
                freq = st.selectbox(f"Frequency for {lifestyle}", ["Daily", "Weekly", "Monthly"])
            with col2:
                servings = st.number_input(f"Servings for {lifestyle}", min_value=1, max_value=100, value=1)
            
            weekly = calculate_weekly_servings(freq, servings)
            nutritional_data[lifestyle] = {
                "frequency": freq,
                "servings": servings,
                "weekly_servings": weekly
            }
    
    # Display score
    if nutritional_data:
        nutritional_score = compute_nutritional_score()
        st.info(f"🍎 {t('nutritional_health_score')}: **{nutritional_score}/5**")
    
    # Save functionality
    if st.button(t("save_nutrition_data")):
        if not st.session_state.user or not st.session_state.user.get("id"):
            st.warning(t("please_login"))
        elif not nutritional_data:
            st.warning("No nutritional data to save")
        else:
            try:
                nutrition_data = {
                    "user_id": st.session_state.user['id'],
                    "fruit_intake": fruit_intake,
                    "vegetable_intake": vegetable_intake,
                    "hydration_liters": hydration_liters,
                    "supplements_used": supplements_used,
                    "natural_herbs": natural_herbs,
                    "nutritional_score": compute_nutritional_score()
                }
                response = supabase.table("nutrition_tracker").insert(nutrition_data).execute()
                if response.data:
                    st.success("Nutrition data saved!")
                else:
                    st.error(f"Failed to save nutrition data: {response.error}")
            except Exception as e:
                st.error(f"Error saving nutrition data: {e}")

# ----------------------------
# STROKE PREDICTION
# ----------------------------
def prepare_stroke_input_numeric(raw_input):
    # Simplified input preparation for stroke prediction
    numeric_features = ['age', 'avg_glucose_level', 'bmi', 'stress_level', 'sleep_hours', 'height', 'weight']
    final_input = {}
    
    for col in numeric_features:
        val = raw_input.get(col, 0)
        try:
            final_input[col] = float(val)
        except:
            final_input[col] = 0
    
    # Add categorical features
    final_input['gender'] = 1 if raw_input.get('gender') == "Male" else 0
    final_input['hypertension'] = 1 if raw_input.get('hypertension') == "Yes" else 0
    final_input['heart_disease'] = 1 if raw_input.get('heart_disease') == "Yes" else 0
    
    df = pd.DataFrame([final_input])
    return df

def stroke_prediction_app():
    st.title(f"🫀 {t('stroke_risk_predictor')}")
    st.warning(t("complete_all_fields"))
    
    with st.form("stroke_form"):
        # Basic information
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input(t("age"), min_value=18, max_value=120, value=50)
            gender = st.selectbox(t("gender"), [t("select_gender"), t("male"), t("female")])
            blood_group = st.selectbox(t("blood_group"), [t("select_blood_group"), "A+", "A-", "B+", "B-", "O+", "O-"])
        
        with col2:
            weight = st.number_input(t("weight"), min_value=20, max_value=200, value=70)
            height = st.number_input(t("height"), min_value=1.0, max_value=2.5, value=1.7)
            genotype = st.selectbox(t("genotype"), [t("select_genotype"), "AA", "AS", "SS"])
        
        # Health information
        col1, col2 = st.columns(2)
        with col1:
            heart_disease = st.selectbox(t("heart_disease"), [t("select"), "Yes", "No"])
            hypertension = st.selectbox(t("hypertension"), [t("select"), "Yes", "No"])
            systolic_bp = st.number_input(t("systolic_bp"), min_value=80, max_value=220, value=120)
        
        with col2:
            diastolic_bp = st.number_input(t("diastolic_bp"), min_value=50, max_value=150, value=80)
            avg_glucose_level = st.number_input(t("avg_glucose_level"), min_value=50.0, max_value=300.0, value=100.0)
            smoking_status = st.selectbox(t("smoking_status"), [t("select"), "never smoked", "formerly smoked", "smokes"])
        
        # BMI calculation
        if weight > 0 and height > 0:
            bmi = round(weight / (height ** 2), 2)
            st.info(f"📏 {t('calculated_bmi')}: **{bmi}**")
            
            if bmi < 18.5:
                risk = t("underweight")
                color = "orange"
            elif 18.5 <= bmi < 25:
                risk = t("normal_weight")
                color = "green"
            elif 25 <= bmi < 30:
                risk = t("overweight")
                color = "yellow"
            else:
                risk = t("obese")
                color = "red"
            
            st.markdown(f"**{t('bmi_category')}:** <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)
        
        # Stress assessment
        st.subheader(f"🧠 {t('additional_stress_assessment')}")
        _, _, stress_score = custom_stress_score(use_container=True)
        
        submit_stroke = st.form_submit_button(t("predict_stroke_risk"))
        
        if submit_stroke:
            if not all([age, gender != t("select_gender"), blood_group != t("select_blood_group")]):
                st.error("Please complete all required fields")
                return
                
            try:
                # Prepare input data
                raw_inputs = {
                    'age': age,
                    'gender': gender,
                    'weight': weight,
                    'height': height,
                    'bmi': bmi,
                    'hypertension': hypertension,
                    'heart_disease': heart_disease,
                    'systolic_bp': systolic_bp,
                    'diastolic_bp': diastolic_bp,
                    'avg_glucose_level': avg_glucose_level,
                    'stress_level': stress_score,
                    'ethnicity': ethnicity_map.get(selected_ethnicity, 0),
                    'country': country_map.get(selected_country, 0),
                    'province': province_map.get(selected_province, 0)
                }
                
                # Prepare input for model
                stroke_inputs_df = prepare_stroke_input_numeric(raw_inputs)
                
                # Make prediction
                if "stroke_model" in st.session_state:
                    pred = st.session_state.stroke_model.predict(stroke_inputs_df)[0]
                    proba = st.session_state.stroke_model.predict_proba(stroke_inputs_df)[0]
                    
                    if pred == 1:
                        st.error(f"⚠️ {t('high_risk')}")
                        st.markdown("""
                        ## 🚨 Immediate Action Recommended:
                        - **Consult a healthcare provider immediately**
                        - Monitor blood pressure daily
                        - Reduce salt intake
                        - Maintain healthy weight
                        - Exercise regularly
                        """)
                    else:
                        st.success(f"✅ {t('low_risk')}")
                        
                    # Save to database
                    try:
                        db_payload = {
                            "user_id": st.session_state.user['id'] if st.session_state.get('user') else "anonymous",
                            "prediction_result": float(pred),
                            "probability": float(proba[1]),
                            "age": age,
                            "gender": gender,
                            "bmi": bmi
                        }
                        response = supabase.table("stroke_predictions").insert(db_payload).execute()
                        if response.data:
                            st.success("Prediction saved successfully!")
                    except Exception as e:
                        st.error(f"Error saving prediction: {e}")
                        
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# ----------------------------
# ALZHEIMER'S PREDICTION
# ----------------------------
def prepare_alzheimers_input_numeric(raw_inputs):
    expected_columns = ['Age', 'Gender', 'BMI', 'EducationLevel', 'MMSE', 'SleepQuality']
    default_values = {'Age': 0, 'Gender': 0, 'BMI': 0, 'EducationLevel': 0, 'MMSE': 0, 'SleepQuality': 0}
    
    full_input = default_values.copy()
    for col in expected_columns:
        if col in raw_inputs:
            try:
                if col == 'Gender':
                    full_input[col] = 1 if str(raw_inputs[col]).lower() == 'male' else 0
                else:
                    full_input[col] = float(raw_inputs[col])
            except (ValueError, TypeError):
                full_input[col] = default_values[col]
    
    alzheimer_inputs_df = pd.DataFrame([full_input])
    return alzheimer_inputs_df

def alzheimers_prediction_app():
    st.title(f"🧠 {t('alzheimers_predictor')}")
    st.warning(t("complete_all_fields"))

    with st.form("alz_form"):
        # Basic information
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input(t("age"), min_value=0, max_value=100, value=50, key='alz_age')
            gender = st.selectbox(t("gender"), [t("select_gender"), t("male"), t("female")], key='alz_gender')
            education_years = st.number_input(t("education_level"), min_value=0, max_value=20, value=12, key='alz_education')
        
        with col2:
            weight = st.number_input(t("weight"), min_value=20, max_value=200, value=70, key='alz_weight')
            height = st.number_input(t("height"), min_value=1.0, max_value=2.5, value=1.7, key='alz_height')
            mmse = st.slider("MMSE Score (0-30)", 0, 30, 25, key='alz_mmse')
        
        # BMI calculation
        if weight > 0 and height > 0:
            bmi = round(weight / (height ** 2), 2)
            st.info(f"📏 {t('calculated_bmi')}: **{bmi}**")
        
        # Health factors
        col1, col2 = st.columns(2)
        with col1:
            family_history = st.selectbox(t("family_history"), ["No", "Yes"], key='alz_family')
            cardiovascular = st.selectbox(t("cardiovascular_disease"), ["No", "Yes"], key='alz_cardio')
            diabetes = st.selectbox(t("diabetes"), ["No", "Yes"], key='alz_diabetes')
        
        with col2:
            hypertension = st.selectbox(t("hypertension"), ["No", "Yes"], key='alz_hypertension')
            depression = st.selectbox(t("depression"), ["No", "Yes"], key='alz_depression')
            sleep_quality = st.slider(t("sleep_quality"), 1, 5, 3, key='alz_sleep')
        
        submit_alz = st.form_submit_button(t("predict_alzheimers_risk"))
        
        if submit_alz:
            if not all([age, gender != t("select_gender"), education_years]):
                st.error("Please complete all required fields")
                return
                
            try:
                raw_inputs = {
                    "Age": age,
                    "Gender": gender,
                    "BMI": bmi,
                    "EducationLevel": education_years,
                    "MMSE": mmse,
                    "SleepQuality": sleep_quality,
                    "FamilyHistoryAlzheimers": 1 if family_history == "Yes" else 0,
                    "CardiovascularDisease": 1 if cardiovascular == "Yes" else 0,
                    "Diabetes": 1 if diabetes == "Yes" else 0,
                    "Hypertension": 1 if hypertension == "Yes" else 0,
                    "Depression": 1 if depression == "Yes" else 0
                }
                
                alzheimer_inputs_df = prepare_alzheimers_input_numeric(raw_inputs)
                
                if "alz_model" in st.session_state:
                    pred = st.session_state.alz_model.predict(alzheimer_inputs_df)[0]
                    proba = st.session_state.alz_model.predict_proba(alzheimer_inputs_df)[0]
                    
                    if pred == 1:
                        st.error(f"⚠️ {t('high_risk')}")
                        st.markdown("""
                        ## 🚨 Immediate Action Recommended:
                        - **Consult a healthcare provider**
                        - Cognitive exercises
                        - Healthy diet
                        - Regular exercise
                        - Social engagement
                        """)
                    else:
                        st.success(f"✅ {t('low_risk')}")
                        
            except Exception as e:
                st.error(f"Error during prediction: {e}")

# ----------------------------
# MEMORY GAME
# ----------------------------
def memory_recall_game():
    st.subheader(f"🧩 {t('memory_recall_game')}")

    if "memory_game" not in st.session_state:
        st.session_state.memory_game = {
            "state": "start",
            "words": [],
            "start_time": None,
            "level": 1,
            "score_history": []
        }

    game = st.session_state.memory_game
    WORD_POOL = ["apple", "table", "river", "mountain", "sun", "flower", "clock", "phone", "book", "star"]

    if game["state"] == "start":
        st.markdown(f"**{t('level')} {game['level']}** - {t('you_will_see_words')} {4 + game['level']} {t('words')}.")
        if st.button(t("start_memory_exercise")):
            num_words = 4 + game["level"]
            words = random.sample(WORD_POOL, num_words)
            game["words"] = words
            game["start_time"] = time.time()
            game["state"] = "showing"
            st.rerun()

    elif game["state"] == "showing":
        st.write(t("memorize_words"))
        st.info(", ".join(game["words"]))
        if time.time() - game["start_time"] > 5:
            game["state"] = "recalling"
            st.rerun()

    elif game["state"] == "recalling":
        with st.form("recall_form"):
            recalled_input = st.text_input(t("type_words_remembered"))
            submit = st.form_submit_button(t("submit_recall"))

        if submit:
            recalled = [w.strip().lower() for w in recalled_input.split(",") if w.strip()]
            correct_words = set(w.lower() for w in game["words"])
            recalled_set = set(recalled)
            correct_count = len(correct_words & recalled_set)

            st.success(f"You recalled {correct_count} out of {len(game['words'])} correctly.")

            game["score_history"].append({
                "level": game["level"],
                "correct": correct_count,
                "total": len(game["words"])
            })

            game["state"] = "start"
            game["words"] = []
            game["start_time"] = None
            st.rerun()

# ----------------------------
# MAIN APP LAYOUT
# ----------------------------

# Initialize session state
if "user" not in st.session_state:
    st.session_state.user = None
if "nutritional_data" not in st.session_state:
    st.session_state.nutritional_data = {}
if "stress_score" not in st.session_state:
    st.session_state.stress_score = 0

# Sidebar configuration
with st.sidebar:
    st.header(f"🌍 {t('location_info')}")
    selected_country = st.selectbox(t("select_country"), list(countries_with_provinces.keys()))
    selected_province = st.selectbox(t("select_province"), countries_with_provinces[selected_country])
    selected_region = st.selectbox(f"🌍 {t('select_region')}", list(region_with_ethnicity.keys()))
    selected_ethnicity = st.selectbox(t("select_ethnicity"), region_with_ethnicity[selected_region])
    
    # Convert to numerical codes
    encoded_country = country_map.get(selected_country, 0)
    encoded_province = province_map.get(selected_province, 0)
    encoded_region = region_map.get(selected_region, 0)
    encoded_ethnicity = ethnicity_map.get(selected_ethnicity, 0)

# Language Selection
st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Language")
language_options = {
    "English": "en",
    "Arabic": "ar", 
    "Swahili": "sw",
    "French": "fr"
}
selected_language = st.sidebar.selectbox(
    t("select_language"),
    list(language_options.keys())
)
st.session_state.current_lang = language_options[selected_language]

st.sidebar.title(t("navigation"))
st.success(f"✅ {t('welcome')}")

# Helper function to check login status
def is_logged_in():
    return st.session_state.user and st.session_state.user.get("id") is not None

# Authentication or Page Navigation
if not is_logged_in():
    st.sidebar.header("🔐 User Authentication")
    auth_option = st.sidebar.radio("Select option:", [t("login"), t("register")], key="auth_option")
    if auth_option == t("login"):
        login()
    else:
        register()
    page_options = [t("about")]
else:
    page_options = [t("about"), t("stroke_prediction"), t("alzheimers_prediction"), t("memory_recall_game"), t("nutrition_tracker")]
    st.sidebar.markdown(f"{t('logged_in_as')}: **{st.session_state.user['email']}**")
    if st.sidebar.button(t("logout")):
        logout()

# Page Selection and Display
page = st.sidebar.selectbox(t("go_to"), page_options)

if page == t("about"):
    about()
elif page == t("stroke_prediction"):
    if is_logged_in():
        stroke_prediction_app()
    else:
        st.warning(f"⚠️ {t('please_login')}")
elif page == t("alzheimers_prediction"):
    if is_logged_in():
        alzheimers_prediction_app()
    else:
        st.warning(f"⚠️ {t('please_login')}")
elif page == t("memory_recall_game"):
    if is_logged_in():
        memory_recall_game()
    else:
        st.warning(f"⚠️ {t('please_login')}")
elif page == t("nutrition_tracker"):
    if is_logged_in():
        nutrition_tracker_app()
    else:
        st.warning(f"⚠️ {t('please_login')}")
