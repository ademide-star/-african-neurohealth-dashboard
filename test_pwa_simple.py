
import streamlit as st
import os

st.set_page_config(page_title="PWA Test", page_icon="🧠", layout="wide")

# Create static folder and basic files
if not os.path.exists("static"):
    os.makedirs("static")

# Create a simple visible indicator
st.markdown("""
<div id="pwa-indicator" style="
    position: fixed;
    top: 10px;
    right: 10px;
    background: #667eea;
    color: white;
    padding: 8px 12px;
    border-radius: 5px;
    z-index: 9999;
    font-weight: bold;
">
    Status: <span id="status-text">Checking...</span>
</div>

<script>
console.log('PWA Test script running...');

// Update status immediately
const statusEl = document.getElementById('status-text');
if (navigator.onLine) {
    statusEl.textContent = '🟢 Online';
    statusEl.style.color = 'lightgreen';
} else {
    statusEl.textContent = '🔴 Offline';
    statusEl.style.color = 'lightcoral';
}

// Update on network changes
window.addEventListener('online', () => {
    statusEl.textContent = '🟢 Online';
    statusEl.style.color = 'lightgreen';
});

window.addEventListener('offline', () => {
    statusEl.textContent = '🔴 Offline';
    statusEl.style.color = 'lightcoral';
});

// Check if manifest is loaded
const manifestLink = document.createElement('link');
manifestLink.rel = 'manifest';
manifestLink.href = './static/manifest.json';
document.head.appendChild(manifestLink);
console.log('Manifest link added');

// Simple icon test
const iconTest = document.createElement('link');
iconTest.rel = 'icon';
iconTest.href = './static/test-icon.png';
iconTest.type = 'image/png';
document.head.appendChild(iconTest);
console.log('Icon link added');
</script>
""", unsafe_allow_html=True)

st.title("PWA Test Page")
st.write("If you see 'Online' in the top right, PWA JavaScript is working.")

# Check if static files exist
st.subheader("Static Files Check")
if os.path.exists("static"):
    st.success("✅ static/ folder exists")
    files = os.listdir("static")
    if files:
        st.write("Files in static/:")
        for file in files:
            st.write(f"- {file}")
    else:
        st.error("❌ No files in static/ folder")
else:
    st.error("❌ static/ folder does not exist")

# Create a test icon button
if st.button("Create Test Icon"):
    try:
        from PIL import Image, ImageDraw
        
        # Create a simple colored icon
        img = Image.new('RGBA', (192, 192), (102, 126, 234, 255))  # Blue
        draw = ImageDraw.Draw(img)
        
        # Add a white circle
        draw.ellipse((20, 20, 172, 172), fill=(255, 255, 255, 255))
        
        img.save("static/test-icon.png")
        st.success("✅ Test icon created at static/test-icon.png")
        
        # Try to display it
        st.image("static/test-icon.png", caption="Test Icon", width=100)
    except Exception as e:
        st.error(f"❌ Failed to create icon: {e}")