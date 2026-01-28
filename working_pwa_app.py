
import streamlit as st
import os

# MUST BE FIRST
st.set_page_config(
    page_title="NeuroHealth AI",
    page_icon="🧠",
    layout="wide"
)

# Create PWA header with visible status
st.markdown("""
<style>
/* Fixed header at the very top */
#pwa-header {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 8px 20px;
    z-index: 9999;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: Arial, sans-serif;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2);
}

/* Add padding to main content so it's not hidden behind fixed header */
.stApp > header {
    margin-top: 50px;
}

/* Status indicator styles */
#pwa-status {
    background: #28a745;
    padding: 4px 12px;
    border-radius: 15px;
    font-size: 14px;
    font-weight: bold;
}

/* Install button */
#pwa-install-btn {
    background: white;
    color: #667eea;
    border: none;
    padding: 6px 12px;
    border-radius: 20px;
    font-weight: bold;
    cursor: pointer;
    margin-left: 10px;
}
</style>

<div id="pwa-header">
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 20px;">🧠</span>
        <span style="font-weight: bold;">NeuroHealth AI</span>
    </div>
    <div style="display: flex; align-items: center;">
        <span id="pwa-status">🟢 Online</span>
        <button id="pwa-install-btn" onclick="installPWA()">📱 Install</button>
    </div>
</div>

<script>
// 1. Online/Offline Status
function updateStatus() {
    const statusEl = document.getElementById('pwa-status');
    const isOnline = navigator.onLine;
    
    if (isOnline) {
        statusEl.innerHTML = '🟢 Online';
        statusEl.style.background = '#28a745';
    } else {
        statusEl.innerHTML = '🔴 Offline';
        statusEl.style.background = '#dc3545';
    }
}

// Initial update
updateStatus();

// Listen for changes
window.addEventListener('online', updateStatus);
window.addEventListener('offline', updateStatus);

// 2. PWA Install
let deferredPrompt;

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    console.log('PWA install prompt available');
    
    // Show install button
    document.getElementById('pwa-install-btn').style.display = 'block';
});

function installPWA() {
    if (deferredPrompt) {
        deferredPrompt.prompt();
        deferredPrompt.userChoice.then((choiceResult) => {
            if (choiceResult.outcome === 'accepted') {
                console.log('User installed the PWA');
                document.getElementById('pwa-install-btn').style.display = 'none';
            }
            deferredPrompt = null;
        });
    }
}

// 3. Service Worker
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('./static/service-worker.js')
            .then(reg => console.log('Service Worker registered:', reg.scope))
            .catch(err => console.log('Service Worker failed:', err));
    });
}

// 4. Add manifest dynamically
const link = document.createElement('link');
link.rel = 'manifest';
link.href = './static/manifest.json';
document.head.appendChild(link);

console.log('PWA setup complete - you should see the blue header at the top!');
</script>
""", unsafe_allow_html=True)

# Your app content below
st.title("African NeuroHealth AI Dashboard")
st.write("This is a test - you should see a blue header at the top of the page.")

# Check if files exist
st.sidebar.title("PWA Status")
if os.path.exists("static/manifest.json"):
    st.sidebar.success("✅ manifest.json exists")
else:
    st.sidebar.error("❌ manifest.json missing")

if os.path.exists("static/icon-192.png"):
    st.sidebar.success("✅ icon-192.png exists")
else:
    st.sidebar.error("❌ icon-192.png missing")

# Quick instructions
st.sidebar.markdown("---")
st.sidebar.markdown("**To test offline:**")
st.sidebar.markdown("1. Open DevTools (F12)")
st.sidebar.markdown("2. Go to Network tab")
st.sidebar.markdown("3. Check 'Offline'")
st.sidebar.markdown("4. Refresh page")

# Create files button
if st.sidebar.button("Create PWA Files"):
    os.makedirs("static", exist_ok=True)
    
    # Simple manifest
    import json
    manifest = {"name": "NeuroHealth", "theme_color": "#667eea"}
    with open("static/manifest.json", "w") as f:
        json.dump(manifest, f)
    
    # Simple service worker
    with open("static/service-worker.js", "w") as f:
        f.write("console.log('Service Worker active');")
    
    st.sidebar.success("Files created!")
    st.rerun()