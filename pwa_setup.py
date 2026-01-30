# ====== COMPLETE WORKING PWA STREAMLIT APP ======
import streamlit as st
import json
import os
from datetime import datetime

# ====== PWA SETUP ======
def setup_pwa():
    """Complete PWA setup that actually works"""
    
    # Create static folder
    os.makedirs("static", exist_ok=True)
    
    # Create essential files
    create_pwa_files()
    
    # Inject PWA HTML/JS
    st.markdown("""
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="manifest" href="/static/manifest.json" crossorigin="use-credentials">
        <meta name="theme-color" content="#667eea">
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        
        <style>
        /* PWA Header */
        #pwa-header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 10px 20px;
            z-index: 10000;
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 50px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        
        /* Fix for Streamlit content */
        .main > div {
            margin-top: 70px !important;
        }
        
        .status-badge {
            padding: 5px 12px;
            border-radius: 15px;
            font-weight: bold;
            font-size: 13px;
        }
        
        .online {
            background: #28a745;
        }
        
        .offline {
            background: #dc3545;
        }
        
        #install-btn {
            background: white;
            color: #667eea;
            border: none;
            padding: 6px 15px;
            border-radius: 20px;
            font-weight: bold;
            cursor: pointer;
            margin-left: 10px;
            display: none;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        
        #install-btn:hover {
            transform: scale(1.05);
            background: #f8f9fa;
        }
        </style>
    </head>
    <body>
        <div id="pwa-header">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 24px;">🧠</span>
                <span style="font-weight: bold; font-size: 16px;">NeuroHealth AI</span>
            </div>
            <div style="display: flex; align-items: center;">
                <span id="status-indicator" class="status-badge online">🟢 Online</span>
                <button id="install-btn">📱 Install App</button>
                <div id="install-log" style="margin-left: 10px; font-size: 12px;"></div>
            </div>
        </div>
        
        <script>
        // PWA Script - Runs immediately
        console.log("🧠 PWA Initializing...");
        
        // DOM Elements
        const statusIndicator = document.getElementById('status-indicator');
        const installBtn = document.getElementById('install-btn');
        const installLog = document.getElementById('install-log');
        
        // 1. Update connection status
        function updateStatus() {
            const isOnline = navigator.onLine;
            
            if (isOnline) {
                statusIndicator.textContent = '🟢 Online';
                statusIndicator.className = 'status-badge online';
                console.log("Status: Online");
            } else {
                statusIndicator.textContent = '🔴 Offline';
                statusIndicator.className = 'status-badge offline';
                console.log("Status: Offline");
                installBtn.style.display = 'none';
                installLog.innerHTML = '<span style="color:#dc3545">⚠️ Offline - cannot install</span>';
            }
        }
        
        // Set initial status
        updateStatus();
        
        // Listen for network changes
        window.addEventListener('online', updateStatus);
        window.addEventListener('offline', updateStatus);
        
        // 2. Install PWA functionality
        let deferredPrompt = null;
        let isAppInstalled = false;
        
        // Check if app is already installed
        if (window.matchMedia('(display-mode: standalone)').matches || 
            window.navigator.standalone ||
            document.referrer.includes('android-app://')) {
            console.log("✅ App already installed");
            isAppInstalled = true;
            installBtn.style.display = 'none';
            installLog.innerHTML = '<span style="color:#28a745">✅ App Installed</span>';
        }
        
        // Listen for install prompt
        window.addEventListener('beforeinstallprompt', (e) => {
            console.log("📱 beforeinstallprompt event fired");
            e.preventDefault();
            deferredPrompt = e;
            
            // Show install button if app isn't already installed
            if (!isAppInstalled) {
                installBtn.style.display = 'block';
                installLog.innerHTML = '<span style="color:#007bff">📱 Install available</span>';
                
                // Auto-show button with animation
                setTimeout(() => {
                    installBtn.style.opacity = '1';
                    installBtn.style.transform = 'translateY(0)';
                }, 100);
            }
            
            // Log event details
            console.log("Install prompt triggered by:", e.userGesture ? "user action" : "automatic");
            console.log("Platform:", navigator.platform);
            console.log("User Agent:", navigator.userAgent);
        });
        
        // Install button click handler
        installBtn.addEventListener('click', async () => {
            console.log("Install button clicked");
            
            if (!deferredPrompt) {
                console.log("❌ No install prompt available");
                installLog.innerHTML = '<span style="color:#dc3545">❌ Install not available</span>';
                installBtn.style.display = 'none';
                return;
            }
            
            if (!navigator.onLine) {
                console.log("❌ Cannot install while offline");
                installLog.innerHTML = '<span style="color:#dc3545">❌ Need internet to install</span>';
                return;
            }
            
            try {
                console.log("Prompting user to install...");
                deferredPrompt.prompt();
                
                const choiceResult = await deferredPrompt.userChoice;
                console.log("User choice result:", choiceResult);
                
                if (choiceResult.outcome === 'accepted') {
                    console.log('✅ User accepted the install prompt');
                    installBtn.style.display = 'none';
                    isAppInstalled = true;
                    installLog.innerHTML = '<span style="color:#28a745">✅ Installing app...</span>';
                    
                    // Wait a moment then update
                    setTimeout(() => {
                        installLog.innerHTML = '<span style="color:#28a745">✅ App Installed!</span>';
                    }, 2000);
                } else {
                    console.log('❌ User dismissed the install prompt');
                    installLog.innerHTML = '<span style="color:#ffc107">⚠️ Installation cancelled</span>';
                }
                
                deferredPrompt = null;
                
            } catch (error) {
                console.error('❌ Install error:', error);
                installLog.innerHTML = '<span style="color:#dc3545">❌ Install failed: ' + error.message + '</span>';
            }
        });
        
        // 3. Service Worker Registration with better offline support
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', () => {
                navigator.serviceWorker.register('/static/service-worker.js')
                    .then(reg => {
                        console.log('✅ Service Worker registered:', reg.scope);
                        console.log('Service Worker state:', reg.active ? 'active' : 'installing');
                        
                        // Check if service worker is controlling the page
                        if (navigator.serviceWorker.controller) {
                            console.log('Service Worker is controlling the page');
                        }
                    })
                    .catch(err => {
                        console.log('❌ Service Worker failed:', err);
                        installLog.innerHTML = '<span style="color:#dc3545">⚠️ Offline mode limited</span>';
                    });
            });
        } else {
            console.log('❌ Service Worker not supported');
            installLog.innerHTML = '<span style="color:#dc3545">⚠️ Browser does not support PWA</span>';
        }
        
        // 4. Listen for app installed event
        window.addEventListener('appinstalled', () => {
            console.log('✅ PWA was successfully installed');
            isAppInstalled = true;
            installBtn.style.display = 'none';
            installLog.innerHTML = '<span style="color:#28a745">✅ App Installed Successfully!</span>';
            
            // Optional: Track installation
            if (window.gtag) {
                gtag('event', 'pwa_installed');
            }
        });
        
        // 5. Debug info
        console.log("✅ PWA setup complete");
        console.log("Install button visible:", installBtn.style.display);
        console.log("Deferred prompt available:", deferredPrompt !== null);
        console.log("App installed status:", isAppInstalled);
        
        </script>
    </body>
    </html>
    """, unsafe_allow_html=True)

def create_pwa_files():
    """Create all PWA files with proper caching for offline"""
    
    # Create manifest with proper paths
    manifest = {
        "name": "African NeuroHealth AI",
        "short_name": "NeuroHealth",
        "description": "Stroke and dementia risk assessment",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#667eea",
        "scope": "/",
        "orientation": "portrait-primary",
        "icons": [
            {
                "src": "/static/icon-192.png",
                "sizes": "192x192",
                "type": "image/png",
                "purpose": "any maskable"
            },
            {
                "src": "/static/icon-512.png",
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "any maskable"
            }
        ],
        "categories": ["health", "medical"],
        "screenshots": [
            {
                "src": "/static/screenshot.png",
                "sizes": "1280x720",
                "type": "image/png"
            }
        ]
    }
    
    with open("static/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Create service worker with proper offline caching
    sw_code = """// Service Worker for NeuroHealth AI - Enhanced for offline
const CACHE_NAME = 'neurohealth-v2.0';
const OFFLINE_URL = '/static/offline.html';

// Files to cache immediately
const STATIC_CACHE_FILES = [
  OFFLINE_URL,
  '/static/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png',
  '/'
];

self.addEventListener('install', event => {
    console.log('Service Worker installing...');
    
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('Caching app shell');
                return cache.addAll(STATIC_CACHE_FILES);
            })
            .then(() => {
                console.log('Skip waiting to activate');
                return self.skipWaiting();
            })
            .catch(err => {
                console.log('Cache error:', err);
            })
    );
});

self.addEventListener('activate', event => {
    console.log('Service Worker activated');
    
    // Clean up old caches
    event.waitUntil(
        caches.keys().then(cacheNames => {
            return Promise.all(
                cacheNames.map(cacheName => {
                    if (cacheName !== CACHE_NAME) {
                        console.log('Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        }).then(() => {
            console.log('Claiming clients');
            return self.clients.claim();
        })
    );
});

self.addEventListener('fetch', event => {
    // Skip non-GET requests
    if (event.request.method !== 'GET') return;
    
    // Handle navigation requests
    if (event.request.mode === 'navigate') {
        event.respondWith(
            fetch(event.request)
                .catch(() => {
                    // If offline, show offline page
                    return caches.match(OFFLINE_URL);
                })
        );
        return;
    }
    
    // For other requests: try network first, then cache
    event.respondWith(
        fetch(event.request)
            .then(response => {
                // If valid response, cache it
                if (response.status === 200) {
                    const responseClone = response.clone();
                    caches.open(CACHE_NAME).then(cache => {
                        cache.put(event.request, responseClone);
                    });
                }
                return response;
            })
            .catch(() => {
                // If network fails, try cache
                return caches.match(event.request)
                    .then(cachedResponse => {
                        return cachedResponse || new Response('Offline', {
                            status: 503,
                            statusText: 'Service Unavailable'
                        });
                    });
            })
    );
});

self.addEventListener('message', event => {
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
});
"""
    
    with open("static/service-worker.js", "w") as f:
        f.write(sw_code)
    
    # Create better offline page
    offline_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Offline - NeuroHealth AI</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            padding: 20px;
        }
        .container {
            max-width: 500px;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 20px;
        }
        p {
            font-size: 1.2em;
            line-height: 1.6;
            margin-bottom: 30px;
        }
        button {
            background: white;
            color: #667eea;
            border: none;
            padding: 12px 30px;
            border-radius: 30px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Offline Mode</h1>
        <p>NeuroHealth AI needs an internet connection to assess stroke and dementia risks.</p>
        <p>Please check your connection and try again.</p>
        <button onclick="location.reload()">↻ Retry Connection</button>
    </div>
</body>
</html>"""
    
    with open("static/offline.html", "w") as f:
        f.write(offline_html)
    
    # Create icons if they don't exist
    create_icons()

def create_icons():
    """Create proper PWA icons"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Create 192x192 icon
        img = Image.new('RGBA', (192, 192), (102, 126, 234, 255))
        draw = ImageDraw.Draw(img)
        
        # Draw brain icon
        draw.ellipse((40, 40, 152, 152), fill=(255, 255, 255, 255))
        draw.ellipse((56, 56, 88, 88), fill=(102, 126, 234, 255))
        draw.ellipse((104, 56, 136, 88), fill=(102, 126, 234, 255))
        draw.arc((56, 80, 136, 152), 0, 180, fill=(102, 126, 234, 255), width=8)
        
        img.save("static/icon-192.png")
        
        # Create 512x512 icon
        img_large = Image.new('RGBA', (512, 512), (102, 126, 234, 255))
        draw_large = ImageDraw.Draw(img_large)
        
        draw_large.ellipse((100, 100, 412, 412), fill=(255, 255, 255, 255))
        draw_large.ellipse((150, 150, 250, 250), fill=(102, 126, 234, 255))
        draw_large.ellipse((262, 150, 362, 250), fill=(102, 126, 234, 255))
        draw_large.arc((150, 220, 362, 400), 0, 180, fill=(102, 126, 234, 255), width=20)
        
        img_large.save("static/icon-512.png")
        
        print("✅ PWA icons created successfully")
    except ImportError:
        print("⚠️ PIL not available - using default icons")
        # Create simple placeholder icons
        import base64
        
        # Simple 1px placeholder icons (will be replaced with proper icons)
        icon_192 = "iVBORw0KGgoAAAANSUhEUgAAAMAAAADACAMAAABlApw1AAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////AAAAVcLTfgAAAAF0Uk5TAEDm2GYAAAAtSURBVHja7MGBAAAAAMOg+VPf4ARVAQAAAHwTYACAAQAGABgAYACAAQAGABgAYACAAQAYAJwCBBgAMOZjVAAAAABJRU5ErkJggg=="
        icon_512 = "iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAMAAADDpiTIAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAAAZQTFRF////AAAAVcLTfgAAAAF0Uk5TAEDm2GYAAAAtSURBVHja7MGBAAAAAMOg+VPf4ARVAQAAAHwTYACAAQAGABgAYACAAQAGABgAYACAAQAYAJwCBBgAMOZjVAAAAABJRU5ErkJggg=="
        
        with open("static/icon-192.png", "wb") as f:
            f.write(base64.b64decode(icon_192))
        with open("static/icon-512.png", "wb") as f:
            f.write(base64.b64decode(icon_512))

# ====== DEBUG FUNCTION ======
def check_pwa_status():
    """Debug function to check PWA status"""
    st.sidebar.subheader("🔧 PWA Debug Info")
    
    if st.sidebar.button("Check PWA Status"):
        st.sidebar.info("""
        **PWA Requirements Checklist:**
        - ✅ HTTPS: Required (Streamlit Cloud provides this)
        - ✅ Manifest: /static/manifest.json
        - ✅ Service Worker: /static/service-worker.js
        - ✅ Icons: /static/icon-192.png, /static/icon-512.png
        - ✅ Offline Page: /static/offline.html
        """)
        
        # Check files exist
        files = [
            "static/manifest.json",
            "static/service-worker.js", 
            "static/offline.html",
            "static/icon-192.png",
            "static/icon-512.png"
        ]
        
        for file in files:
            if os.path.exists(file):
                st.sidebar.success(f"✅ {file}")
            else:
                st.sidebar.error(f"❌ {file} missing!")
        
        st.sidebar.warning("""
        **Install Button Issues:**
        1. App must be served over HTTPS
        2. User must interact with page first (click somewhere)
        3. App must meet PWA criteria (checklist above)
        4. Browser must support PWA (Chrome, Edge, Safari iOS)
        """)

# ====== MAIN APP ======
def main():
    st.set_page_config(
        page_title="NeuroHealth AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Setup PWA
    setup_pwa()
    
    # Debug panel
    check_pwa_status()
    
    # Main app content
    st.title("🧠 African NeuroHealth AI")
    st.subheader("Stroke & Dementia Risk Assessment")
    
    # Your app content here...
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Patient Information")
        age = st.slider("Age", 18, 100, 50)
        hypertension = st.checkbox("Hypertension")
        diabetes = st.checkbox("Diabetes")
        
    with col2:
        st.markdown("### Assessment")
        if st.button("Assess Risk"):
            score = age + (10 if hypertension else 0) + (10 if diabetes else 0)
            st.success(f"Risk Score: {score}/120")
            
            if score > 70:
                st.error("High Risk - Consult specialist")
            elif score > 40:
                st.warning("Medium Risk - Monitor regularly")
            else:
                st.info("Low Risk - Maintain healthy lifestyle")

# ====== RUN APP ======
if __name__ == "__main__":
    main()
