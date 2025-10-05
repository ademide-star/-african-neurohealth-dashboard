# translations.py
import streamlit as st
from deep_translator import GoogleTranslator

# Supported languages
LANGUAGES = {
    "English": "en",
    "Arabic": "ar",
    "Swahili": "sw",
    "French": "fr"
}

def set_language_selector():
    """Display language selector in sidebar and store choice in session_state."""
    selected_lang = st.sidebar.selectbox("🌍 Select Language", options=list(LANGUAGES.keys()))
    st.session_state["lang"] = LANGUAGES[selected_lang]
    return selected_lang

@st.cache_data(show_spinner=False)
def cached_translate(text, target_lang):
    """Translate text with caching for better performance."""
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        return text

def get_translation(text):
    """Translate text, list, or dictionary according to selected language."""
    lang = st.session_state.get("lang", "en")
    if lang == "en":
        return text

    try:
        if isinstance(text, str):
            return cached_translate(text, lang)
        elif isinstance(text, list):
            return [cached_translate(t, lang) for t in text]
        elif isinstance(text, dict):
            return {k: cached_translate(v, lang) for k, v in text.items()}
        elif hasattr(text, "tolist"):
            return [cached_translate(t, lang) for t in text.tolist()]
        return text
    except Exception:
        return text
