# test_pwa.py
import streamlit as st
import streamlit as st
import json
import os
from pathlib import Path

st.set_page_config(page_title="PWA Test", layout="wide")

st.markdown("""
<div id="pwa-test" style="
    position: fixed;
    top: 10px;
    right: 10px;
    background: #667eea;
    color: white;
    padding: 10px;
    border-radius: 5px;
    z-index: 9999;
">
    PWA Test: <span id="status">Checking...</span>
</div>

<script>
const statusEl = document.getElementById('status');
statusEl.textContent = navigator.onLine ? '🟢 Online' : '🔴 Offline';
statusEl.style.color = navigator.onLine ? 'lightgreen' : 'lightcoral';

window.addEventListener('online', () => {
    statusEl.textContent = '🟢 Online';
    statusEl.style.color = 'lightgreen';
});

window.addEventListener('offline', () => {
    statusEl.textContent = '🔴 Offline';
    statusEl.style.color = 'lightcoral';
});
</script>
""", unsafe_allow_html=True)

st.title("PWA Test Page")
st.write("If you see 'Online' in the top right, it works!")


# create_icons.py
from PIL import Image, ImageDraw
import os

os.makedirs("static", exist_ok=True)

# Create simple icons
size = 192
img = Image.new('RGBA', (size, size), (102, 126, 234, 255))  # #667eea color
draw = ImageDraw.Draw(img)

# Add a simple brain shape (or use your logo)
draw.ellipse((40, 40, 152, 152), fill=(255, 255, 255, 255))

img.save("static/icon-192.png")
img.resize((512, 512)).save("static/icon-512.png")
print("✅ Icons created at static/icon-192.png and static/icon-512.png")