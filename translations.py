import streamlit as st
from deep_translator import GoogleTranslator
from functools import lru_cache

# ====== LANGUAGE CONFIGURATION ======

LANGUAGES = {
    "en": "🇬🇧 English",
    "fr": "🇫🇷 Français",
    "ar": "🇸🇦 العربية (Arabic)",
    "sw": "🇰🇪 Kiswahili",
    "ha": "🇳🇬 Hausa",
    "pt": "🇵🇹 Português",
    "yo": "🇳🇬 Yorùbá"
}

# ====== TRANSLATION DICTIONARY ======
# Static translations for core medical/UI terms (High Accuracy)

TRANSLATIONS = {
    "en": {
        # App Header
        "title": "African NeuroHealth AI",
        "subtitle": "Stroke & Dementia Predictor",
        
        # Common UI
        "age": "Age",
        "gender": "Gender",
        "male": "Male",
        "female": "Female",
        "select": "Select",
        "select_gender": "Select Gender",
        "predict": "Run Prediction",
        "submit": "Submit",
        "result": "Assessment Result",
        "download": "Download Report",
        "generate_report": "Generate Report",
        
        # Navigation
        "navigation": "📍 Navigation",
        "go_to": "Go to",
        "dashboard": "Dashboard",
        "stroke_assessment": "Stroke Assessment",
        "dementia_assessment": "Dementia Assessment",
        "memory_game": "Memory Game",
        "nutrition_tracker": "Nutrition Tracker",
        "stress_assessment": "Stress Assessment",
        "my_reports": "My Reports",
        
        # Login
        "quick_login": "🔐 Quick Login",
        "enter_name": "Enter your name (optional)",
        "start_session": "Start Session",
        "quick_start": "Quick Start",
        "end_session": "End Session",
        "welcome": "Welcome",
        
        # Quick Actions
        "quick_actions": "⚡ Quick Actions",
        "reload": "🔄 Reload",
        "reports": "📊 Reports",
        "reloading_models": "Reloading models...",
        
        # User Info
        "user_information": "👤 User Information",
        "user": "User",
        "id": "ID",
        "generated_reports": "Generated Reports",
        
        # Medical Terms
        "blood_group": "Blood Group",
        "genotype": "Genotype",
        "select_blood_group": "Select Blood Group",
        "select_genotype": "Select Genotype",
        "bmi": "BMI",
        "height": "Height (cm)",
        "weight": "Weight (kg)",
        "systolic_bp": "Systolic BP",
        "diastolic_bp": "Diastolic BP",
        "hypertension": "Hypertension",
        "diabetes": "Diabetes",
        "heart_disease": "Heart Disease",
        "smoking_status": "Smoking Status",
        "yes": "Yes",
        "no": "No",
        
        # BMI Categories
        "underweight": "Underweight",
        "normal_weight": "Normal weight",
        "overweight": "Overweight",
        "obese": "Obese",
        
        # Risk Levels
        "risk_level": "Risk Level",
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
        
        # Messages
        "complete_fields": "⚠️ Please complete all fields before prediction.",
        "no_reports": "No reports generated yet. Complete an assessment to generate your first report.",
        "all_validated": "All inputs validated! Now run prediction pipeline...",
        
        # Errors
        "error_occurred": "An error occurred during prediction:",
        "model_not_loaded": "Model not loaded. Please initialize the model first.",
        "input_failed": "Input preparation failed: no valid features for prediction."
    },
    
    "fr": {
        "title": "IA Santé Neuro Africaine",
        "subtitle": "Prédicteur d'AVC et de Démence",
        "age": "Âge",
        "gender": "Sexe",
        "male": "Homme",
        "female": "Femme",
        "select": "Sélectionner",
        "select_gender": "Sélectionner le sexe",
        "predict": "Exécuter la prédiction",
        "submit": "Soumettre",
        "result": "Résultat de l'évaluation",
        "download": "Télécharger le rapport",
        "generate_report": "Générer le rapport",
        "quick_login": "🔐 Connexion rapide",
        "enter_name": "Entrez votre nom (optionnel)",
        "start_session": "Démarrer la session",
        "quick_start": "Démarrage rapide",
        "end_session": "Terminer la session",
        "welcome": "Bienvenue",
        "blood_group": "Groupe sanguin",
        "genotype": "Génotype",
        "yes": "Oui",
        "no": "Non"
    },
    
    "ar": {
        "title": "الذكاء الاصطناعي للصحة العصبية الأفريقية",
        "subtitle": "متنبئ السكتة الدماغية والخرف",
        "age": "العمر",
        "gender": "الجنس",
        "male": "ذكر",
        "female": "أنثى",
        "select": "اختر",
        "select_gender": "اختر الجنس",
        "predict": "تشغيل التنبؤ",
        "submit": "إرسال",
        "result": "نتيجة التقييم",
        "download": "تحميل التقرير",
        "generate_report": "إنشاء التقرير",
        "quick_login": "🔐 تسجيل دخول سريع",
        "enter_name": "أدخل اسمك (اختياري)",
        "start_session": "بدء الجلسة",
        "quick_start": "بداية سريعة",
        "end_session": "إنهاء الجلسة",
        "welcome": "مرحباً",
        "blood_group": "فصيلة الدم",
        "genotype": "النمط الجيني",
        "yes": "نعم",
        "no": "لا"
    },
    
    "sw": {
        "title": "Akili Mnemba ya Afya ya Mishipa ya Afrika",
        "subtitle": "Kizabuni cha Kiharusi na Demensia",
        "age": "Umri",
        "gender": "Jinsia",
        "male": "Mwanaume",
        "female": "Mwanamke",
        "select": "Chagua",
        "select_gender": "Chagua Jinsia",
        "predict": "Fanya Utabiri",
        "submit": "Wasilisha",
        "result": "Matokeo ya Tathmini",
        "download": "Pakua Ripoti",
        "generate_report": "Tengeneza Ripoti",
        "yes": "Ndiyo",
        "no": "Hapana"
    },
    
    "ha": {
        "title": "Ilimin Na'ura na Lafiyar Jijiya a Afirka",
        "subtitle": "Mai Hasashen Shanyewar Jiki da Hauka",
        "age": "Shekaru",
        "gender": "Jinsi",
        "male": "Namiji",
        "female": "Mace",
        "select": "Zaɓi",
        "select_gender": "Zaɓi Jinsi",
        "predict": "Gudanar da Hasashe",
        "submit": "Tura",
        "result": "Sakamakon Bincike",
        "download": "Zazzage Rahoto",
        "generate_report": "Samar da Rahoto",
        "yes": "I",
        "no": "A'a"
    },
    
    "pt": {
        "title": "IA Africana de Neurosaúde",
        "subtitle": "Preditor de AVC e Demência",
        "age": "Idade",
        "gender": "Gênero",
        "male": "Masculino",
        "female": "Feminino",
        "select": "Selecionar",
        "select_gender": "Selecionar Gênero",
        "predict": "Executar Predição",
        "submit": "Enviar",
        "result": "Resultado da Avaliação",
        "download": "Baixar Relatório",
        "generate_report": "Gerar Relatório",
        "yes": "Sim",
        "no": "Não"
    },
    
    "yo": {
        "title": "Ìmọ̀-ẹ̀rọ Ìlera Ọpọlọ Áfíríkà",
        "subtitle": "Olùsọtẹ́lẹ̀ Àrùn Ọpọlọ àti Rọpárọsẹ̀",
        "age": "Ọjọ́ orí",
        "gender": "Ọmọbìnrin tàbí Ọmọkùnrin",
        "male": "Ọkùnrin",
        "female": "Obìnrin",
        "select": "Yan",
        "select_gender": "Yan Ọmọbìnrin tàbí Ọmọkùnrin",
        "predict": "Ṣe Àyẹ̀wò",
        "submit": "Fi ránṣẹ́",
        "result": "Èsì Àyẹ̀wò",
        "download": "Gba Ràpọ̀tì",
        "generate_report": "Ṣẹ̀dá Ràpọ̀tì",
        "yes": "Bẹ́ẹ̀ni",
        "no": "Bẹ́ẹ̀kọ́"
    }
}


# ====== LANGUAGE SELECTOR ======

def set_language_selector(widget_key="main_lang_selector"):
    """Displays the language dropdown in the sidebar"""
    selected_lang = st.sidebar.selectbox(
        "Select Language / Sélectionner la langue",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x],
        key=widget_key
    )
    
    # Store in session state
    st.session_state.current_language = selected_lang
    return selected_lang


# ====== TRANSLATION FUNCTION ======

def get_translation(text_key):
    """
    Get translation for a given text key in the current language.
    
    Args:
        text_key (str): The key to translate (or plain text for dynamic translation)
    
    Returns:
        str: Translated text
    """
    # Get current language from session state, default to English
    current_lang = st.session_state.get('current_language', 'en')
    
    # Normalize the key (lowercase, replace spaces with underscores)
    normalized_key = text_key.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    
    # Try to get from static dictionary
    lang_dict = TRANSLATIONS.get(current_lang, TRANSLATIONS['en'])
    
    if normalized_key in lang_dict:
        return lang_dict[normalized_key]
    
    # If not in dictionary and not English, try dynamic translation
    if current_lang != 'en':
        return cached_translate(text_key, current_lang)
    
    # Return original text if all else fails
    return text_key


@lru_cache(maxsize=256)
def cached_translate(text, target_lang):
    """
    Dynamic translation using Google Translate for phrases NOT in dictionary.
    Cached to avoid repeated API calls.
    """
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text


# ====== HELPER FUNCTIONS ======

def translate_list(items):
    """Translate a list of items"""
    return [get_translation(item) for item in items]


def get_current_language():
    """Get the current language code"""
    return st.session_state.get('current_language', 'en')


def is_rtl_language():
    """Check if current language is RTL (Right-to-Left)"""
    return get_current_language() == 'ar'


    
    

