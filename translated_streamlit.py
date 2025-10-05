# translated_streamlit.py
import streamlit as _st
from translations import get_translation, set_language_selector
from deep_translator import GoogleTranslator

def _translate_item(item, lang=None):
    """Translate strings (or lists/tuples of strings). Leave non-strings unchanged."""
    if isinstance(item, str):
        return get_translation(item, lang)
    if isinstance(item, (list, tuple)):
        return type(item)(_translate_item(x, lang) for x in item)
    if isinstance(item, dict):
        return {k: _translate_item(v, lang) for k, v in item.items()}
    return item

class _SidebarProxy:
    def __init__(self, sidebar):
        self._sidebar = sidebar

    def selectbox(self, label, options, *args, **kwargs):
        return self._sidebar.selectbox(
            _translate_item(label),
            _translate_item(options),
            *args, **kwargs
        )

    def button(self, label, *args, **kwargs):
        return self._sidebar.button(_translate_item(label), *args, **kwargs)

    def write(self, *args, **kwargs):
        args = tuple(_translate_item(a) for a in args)
        return self._sidebar.write(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._sidebar, name)


class TranslatedSt:
    def __init__(self):
        self._st = _st
        # expose session_state & other attrs directly
        self.session_state = _st.session_state
        self.sidebar = _SidebarProxy(_st.sidebar)

    # Common UI functions — translate string args
    def title(self, text, *args, **kwargs):
        return self._st.title(get_translation(text))

    def header(self, text, *args, **kwargs):
        return self._st.header(get_translation(text))

    def subheader(self, text, *args, **kwargs):
        return self._st.subheader(get_translation(text))

    def markdown(self, text, *args, **kwargs):
        return self._st.markdown(get_translation(text), *args, **kwargs)

    def write(self, *args, **kwargs):
        new_args = tuple(_translate_item(a) for a in args)
        return self._st.write(*new_args, **kwargs)

    def text(self, text, *args, **kwargs):
        return self._st.text(get_translation(text), *args, **kwargs)

    def button(self, label, *args, **kwargs):
        return self._st.button(get_translation(label), *args, **kwargs)

    def checkbox(self, label, *args, **kwargs):
        return self._st.checkbox(get_translation(label), *args, **kwargs)

    def radio(self, label, options, *args, **kwargs):
        return self._st.radio(get_translation(label), _translate_item(options), *args, **kwargs)

    def selectbox(self, label, options, *args, **kwargs):
        return self._st.selectbox(get_translation(label), _translate_item(options), *args, **kwargs)

    def multiselect(self, label, options, *args, **kwargs):
        return self._st.multiselect(get_translation(label), _translate_item(options), *args, **kwargs)

    def text_input(self, label, *args, **kwargs):
        return self._st.text_input(get_translation(label), *args, **kwargs)

    def text_area(self, label, *args, **kwargs):
        return self._st.text_area(get_translation(label), *args, **kwargs)

    def number_input(self, label, *args, **kwargs):
        return self._st.number_input(get_translation(label), *args, **kwargs)

    def slider(self, label, *args, **kwargs):
        return self._st.slider(get_translation(label), *args, **kwargs)

    def metric(self, label, value, *args, **kwargs):
        return self._st.metric(get_translation(label), value, *args, **kwargs)

    def dataframe(self, *args, **kwargs):
        return self._st.dataframe(*args, **kwargs)

    # Fallback: any attribute not defined proxies to actual streamlit
    def __getattr__(self, name):
        return getattr(self._st, name)
    
    # Cache translation for speed
@st.cache_data(show_spinner=False)
def cached_translate(text, target_lang):
    """Translate text and cache the result."""
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        return text


def get_translation(text):
    """Translate text, lists, or dicts based on current language."""
    lang = st.session_state.get("lang", "en")
    if lang == "en":
        return text

    try:
        # Single string
        if isinstance(text, str):
            return cached_translate(text, lang)

        # List of strings
        elif isinstance(text, list):
            return [cached_translate(t, lang) for t in text]

        # Dictionary
        elif isinstance(text, dict):
            return {k: cached_translate(v, lang) for k, v in text.items()}

        # DataFrame columns (if passed as Index)
        elif hasattr(text, "tolist"):
            return [cached_translate(t, lang) for t in text.tolist()]

        return text
    except Exception:
        return text
# create instance 'st' to import in your app
st = TranslatedSt()
