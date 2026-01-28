# ====================================================================
# PROGRESSIVE WEB APP (PWA) SETUP FOR STREAMLIT
# ====================================================================
# This enables offline functionality for your NeuroHealth AI dashboard
# ====================================================================

import streamlit as st
import json
import os
from pathlib import Path

# ====================================================================
# 1. CREATE MANIFEST.JSON FILE
# ====================================================================

def create_manifest():
    """Create PWA manifest file"""
    manifest = {
        "name": "African NeuroHealth AI",
        "short_name": "NeuroHealth",
        "description": "AI-powered stroke and dementia risk assessment for African populations",
        "start_url": "./",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#667eea",
        "orientation": "portrait-primary",
        "scope": "./",
        "icons": [
            {
                "src": "./static/icon-192x192.png",
                "sizes": "192x192",
                "type": "image/png",
                "purpose": "any maskable"
            },
            {
                "src": "./static/icon-512x512.png",
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "any maskable"
            }
        ],
        "categories": ["health", "medical", "productivity"],
        "lang": "en",
        "dir": "ltr"
    }
    
    # Save manifest.json in static folder
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    
    with open(static_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    
    return manifest


# ====================================================================
# 2. CREATE SERVICE WORKER (service-worker.js)
# ====================================================================

SERVICE_WORKER_JS = """
// Service Worker for African NeuroHealth AI PWA
const CACHE_NAME = 'neurohealth-v1.0.0';
const OFFLINE_URL = './static/offline.html';

// Files to cache for offline use
const CACHE_URLS = [
  './',
  './static/offline.html',
  './static/manifest.json',
  './static/icon-192x192.png',
  './static/icon-512x512.png'
];

// Install event - cache essential files
self.addEventListener('install', (event) => {
  console.log('[ServiceWorker] Installing...');
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[ServiceWorker] Caching app shell');
      return cache.addAll(CACHE_URLS);
    }).then(() => {
      return self.skipWaiting();
    })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[ServiceWorker] Activating...');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('[ServiceWorker] Removing old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      return self.clients.claim();
    })
  );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
  // Skip non-GET requests and Streamlit websocket
  if (event.request.method !== 'GET' || 
      event.request.url.includes('_stcore/') ||
      event.request.url.includes('stream')) {
    return;
  }

  // Skip cross-origin requests
  if (!event.request.url.startsWith(self.location.origin)) {
    return;
  }

  event.respondWith(
    caches.match(event.request).then((response) => {
      // Cache hit - return response
      if (response) {
        return response;
      }

      return fetch(event.request).then((response) => {
        // Check if valid response
        if (!response || response.status !== 200 || response.type !== 'basic') {
          return response;
        }

        // Clone the response
        const responseToCache = response.clone();

        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, responseToCache);
        });

        return response;
      }).catch(() => {
        // Network failed, return offline page for HTML requests
        if (event.request.headers.get('accept').includes('text/html')) {
          return caches.match(OFFLINE_URL);
        }
        return new Response('Offline', {
          status: 408,
          headers: { 'Content-Type': 'text/plain' }
        });
      });
    })
  );
});

// Background sync for offline data submission
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-assessments') {
    event.waitUntil(syncAssessments());
  }
});

async function syncAssessments() {
  try {
    const cache = await caches.open('offline-assessments');
    const requests = await cache.keys();
    
    for (const request of requests) {
      const response = await cache.match(request);
      const data = await response.json();
      
      // Send to server
      await fetch('/api/submit-assessment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      
      // Remove from cache after successful sync
      await cache.delete(request);
    }
  } catch (error) {
    console.error('[ServiceWorker] Sync failed:', error);
  }
}
"""

def create_service_worker():
    """Create service worker JavaScript file"""
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    
    with open(static_dir / "service-worker.js", "w", encoding="utf-8") as f:
        f.write(SERVICE_WORKER_JS)


# ====================================================================
# 3. CREATE OFFLINE.HTML PAGE
# ====================================================================

OFFLINE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Offline - NeuroHealth AI</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            padding: 20px;
        }
        .container {
            max-width: 500px;
        }
        .icon {
            font-size: 80px;
            margin-bottom: 20px;
        }
        h1 {
            font-size: 32px;
            margin-bottom: 10px;
        }
        p {
            font-size: 18px;
            line-height: 1.6;
            margin-bottom: 30px;
        }
        .button {
            background: white;
            color: #667eea;
            border: none;
            padding: 15px 30px;
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .button:hover {
            transform: scale(1.05);
        }
        .features {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            margin-top: 30px;
            text-align: left;
        }
        .features h2 {
            margin-top: 0;
            font-size: 20px;
        }
        .features ul {
            list-style: none;
            padding: 0;
        }
        .features li {
            padding: 8px 0;
            padding-left: 25px;
            position: relative;
        }
        .features li:before {
            content: "✓";
            position: absolute;
            left: 0;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">🧠</div>
        <h1>You're Offline</h1>
        <p>NeuroHealth AI is currently unavailable. Your data has been saved locally and will sync when you're back online.</p>
        <button class="button" onclick="location.reload()">Try Again</button>
        
        <div class="features">
            <h2>Available Offline:</h2>
            <ul>
                <li>View cached risk assessments</li>
                <li>Complete new assessments (saved locally)</li>
                <li>Access educational resources</li>
                <li>View saved PDF reports</li>
            </ul>
        </div>
    </div>
    
    <script>
        // Check for connection every 5 seconds
        setInterval(() => {
            if (navigator.onLine) {
                location.reload();
            }
        }, 5000);
    </script>
</body>
</html>
"""

def create_offline_page():
    """Create offline fallback HTML page"""
    static_dir = Path("static")
    static_dir.mkdir(exist_ok=True)
    
    with open(static_dir / "offline.html", "w", encoding="utf-8") as f:
        f.write(OFFLINE_HTML)


# ====================================================================
# 4. STREAMLIT PWA INTEGRATION
# ====================================================================

def inject_pwa_code():
    """Inject PWA meta tags and service worker registration into Streamlit"""
    
    pwa_html = """
    <head>
        <!-- PWA Meta Tags -->
        <meta name="mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <meta name="apple-mobile-web-app-title" content="NeuroHealth AI">
        <meta name="theme-color" content="#667eea">
        
        <!-- Manifest -->
        <link rel="manifest" href="./static/manifest.json">
        
        <!-- Icons -->
        <link rel="icon" type="image/png" sizes="192x192" href="./static/icon-192x192.png">
        <link rel="apple-touch-icon" href="./static/icon-512x512.png">
        
        <!-- Service Worker Registration -->
        <script>
            if ('serviceWorker' in navigator) {
                window.addEventListener('load', () => {
                    navigator.serviceWorker.register('./static/service-worker.js', { scope: './' })
                        .then((registration) => {
                            console.log('✅ ServiceWorker registered:', registration.scope);
                            
                            // Check for updates periodically
                            setInterval(() => {
                                registration.update();
                            }, 60000); // Check every minute
                        })
                        .catch((error) => {
                            console.error('❌ ServiceWorker registration failed:', error);
                        });
                });
            } else {
                console.warn('⚠️ Service Workers not supported in this browser');
            }
            
            // Detect online/offline status
            window.addEventListener('online', () => {
                console.log('🟢 Back online');
                document.body.style.borderTop = '3px solid green';
                setTimeout(() => {
                    document.body.style.borderTop = 'none';
                }, 3000);
                
                // Show notification
                const notification = document.createElement('div');
                notification.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #28a745;
                    color: white;
                    padding: 15px 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
                    z-index: 10000;
                    font-weight: bold;
                `;
                notification.textContent = '🟢 Back Online - Syncing data...';
                document.body.appendChild(notification);
                setTimeout(() => notification.remove(), 3000);
                
                // Trigger background sync
                if ('serviceWorker' in navigator && 'sync' in navigator.serviceWorker) {
                    navigator.serviceWorker.ready.then((registration) => {
                        return registration.sync.register('sync-assessments');
                    }).catch((err) => {
                        console.log('Background sync registration failed:', err);
                    });
                }
            });
            
            window.addEventListener('offline', () => {
                console.log('🔴 Gone offline');
                document.body.style.borderTop = '3px solid red';
                
                // Show notification
                const notification = document.createElement('div');
                notification.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #dc3545;
                    color: white;
                    padding: 15px 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
                    z-index: 10000;
                    font-weight: bold;
                `;
                notification.textContent = '🔴 Offline Mode - Data saved locally';
                document.body.appendChild(notification);
            });
            
            // Install prompt
            let deferredPrompt;
            window.addEventListener('beforeinstallprompt', (e) => {
                e.preventDefault();
                deferredPrompt = e;
                
                // Show install button (you can customize this)
                const installDiv = document.createElement('div');
                installDiv.id = 'install-prompt';
                installDiv.innerHTML = `
                    <div style="position: fixed; bottom: 20px; right: 20px; 
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                color: white; padding: 15px 20px; border-radius: 10px;
                                box-shadow: 0 4px 6px rgba(0,0,0,0.1); z-index: 9999;
                                cursor: pointer; transition: transform 0.2s;"
                                onmouseover="this.style.transform='scale(1.05)'"
                                onmouseout="this.style.transform='scale(1)'"
                                onclick="installPWA()">
                        <strong>📱 Install App</strong><br>
                        <small>Use offline, save to home screen</small>
                    </div>
                `;
                document.body.appendChild(installDiv);
                console.log('📱 Install prompt ready');
            });
            
            function installPWA() {
                if (deferredPrompt) {
                    deferredPrompt.prompt();
                    deferredPrompt.userChoice.then((choiceResult) => {
                        if (choiceResult.outcome === 'accepted') {
                            console.log('✅ PWA installed');
                            document.getElementById('install-prompt').remove();
                        } else {
                            console.log('❌ PWA installation declined');
                        }
                        deferredPrompt = null;
                    });
                }
            }
            
            // Make installPWA globally accessible
            window.installPWA = installPWA;
            
            // Cache API for offline storage
            async function saveOfflineData(assessmentData) {
                try {
                    const cache = await caches.open('offline-assessments');
                    const request = new Request('/offline-assessment-' + Date.now());
                    const response = new Response(JSON.stringify(assessmentData), {
                        headers: { 'Content-Type': 'application/json' }
                    });
                    await cache.put(request, response);
                    console.log('💾 Assessment saved offline');
                    return true;
                } catch (error) {
                    console.error('❌ Failed to save offline:', error);
                    return false;
                }
            }
            
            // Make saveOfflineData globally accessible
            window.saveOfflineData = saveOfflineData;
            
            // Log PWA status on load
            console.log('🧠 NeuroHealth AI PWA initialized');
            console.log('📱 Online status:', navigator.onLine ? 'Online' : 'Offline');
        </script>
    </head>
    """
    
    st.markdown(pwa_html, unsafe_allow_html=True)


# ====================================================================
# 5. OFFLINE DATA STORAGE HELPER
# ====================================================================

def create_offline_storage_helper():
    """Create helper for IndexedDB/localStorage fallback"""
    
    storage_js = """
    <script>
    // Enhanced offline storage for Streamlit
    class OfflineStorage {
        constructor() {
            this.dbName = 'NeuroHealthDB';
            this.storeName = 'assessments';
            this.init();
        }
        
        init() {
            // Create offline badge on init
            this.createOfflineBadge();
            this.updatePendingCount();
        }
        
        async saveAssessment(data) {
            try {
                // Try to save to server
                const response = await fetch('/api/save-assessment', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                if (response.ok) {
                    console.log('✅ Assessment saved to server');
                    return true;
                }
                throw new Error('Server error');
            } catch (error) {
                // Fallback to local storage
                console.log('💾 Saving to local storage');
                return this.saveToLocalStorage(data);
            }
        }
        
        saveToLocalStorage(data) {
            try {
                const key = `assessment_${Date.now()}`;
                const assessments = JSON.parse(localStorage.getItem('pendingAssessments') || '[]');
                assessments.push({
                    id: key,
                    data: data,
                    timestamp: new Date().toISOString()
                });
                localStorage.setItem('pendingAssessments', JSON.stringify(assessments));
                
                // Also save to cache for service worker
                if (window.saveOfflineData) {
                    window.saveOfflineData(data);
                }
                
                // Update UI
                this.updatePendingCount();
                return true;
            } catch (e) {
                console.error('Failed to save locally:', e);
                return false;
            }
        }
        
        updatePendingCount() {
            const assessments = JSON.parse(localStorage.getItem('pendingAssessments') || '[]');
            const badge = document.getElementById('offline-badge') || this.createOfflineBadge();
            badge.textContent = `📱 ${assessments.length} pending`;
            badge.style.display = assessments.length > 0 ? 'block' : 'none';
        }
        
        createOfflineBadge() {
            const badge = document.createElement('div');
            badge.id = 'offline-badge';
            badge.style.cssText = `
                position: fixed;
                bottom: 70px;
                right: 20px;
                background: #ff6b6b;
                color: white;
                padding: 8px 12px;
                border-radius: 15px;
                font-size: 12px;
                font-weight: bold;
                z-index: 10000;
                display: none;
                cursor: pointer;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            `;
            badge.onclick = () => this.showPendingList();
            document.body.appendChild(badge);
            return badge;
        }
        
        showPendingList() {
            const assessments = JSON.parse(localStorage.getItem('pendingAssessments') || '[]');
            if (assessments.length === 0) return;
            
            // Create modal
            const modal = document.createElement('div');
            modal.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.5);
                z-index: 10001;
                display: flex;
                justify-content: center;
                align-items: center;
            `;
            
            modal.innerHTML = `
                <div style="
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    max-width: 500px;
                    width: 90%;
                    max-height: 80vh;
                    overflow-y: auto;
                ">
                    <h3 style="margin-top: 0;">Pending Assessments (${assessments.length})</h3>
                    <div id="pending-list"></div>
                    <button onclick="this.parentElement.parentElement.remove()" 
                            style="margin-top: 15px; padding: 8px 15px; background: #667eea; color: white; border: none; border-radius: 5px; cursor: pointer;">
                        Close
                    </button>
                </div>
            `;
            
            document.body.appendChild(modal);
        }
    }
    
    // Initialize offline storage
    window.offlineStorage = new OfflineStorage();
    </script>
    """
    
    return storage_js


# ====================================================================
# 6. PERFORMANCE OPTIMIZATIONS
# ====================================================================

def optimize_pwa_performance():
    """Add performance optimizations for PWA"""
    
    perf_js = """
    <script>
    // Performance optimizations
    document.addEventListener('DOMContentLoaded', () => {
        // Lazy load images
        const images = document.querySelectorAll('img[data-src]');
        const imageObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    imageObserver.unobserve(img);
                }
            });
        });
        images.forEach(img => imageObserver.observe(img));
        
        // Preload critical resources
        const preloadLinks = [
            { href: './static/icon-512x512.png', as: 'image' },
            { href: './static/manifest.json', as: 'manifest' }
        ];
        
        preloadLinks.forEach(link => {
            const el = document.createElement('link');
            el.rel = 'preload';
            el.href = link.href;
            el.as = link.as;
            document.head.appendChild(el);
        });
    });
    
    // Cache management
    function clearOldCache() {
        if ('caches' in window) {
            caches.keys().then(cacheNames => {
                cacheNames.forEach(cacheName => {
                    // Keep only current version
                    if (!cacheName.includes('neurohealth-v')) {
                        caches.delete(cacheName);
                    }
                });
            });
        }
    }
    
    // Run cleanup on load
    clearOldCache();
    </script>
    """
    
    return perf_js


# ====================================================================
# 7. STREAMLIT NAVIGATION FIX
# ====================================================================

STREAMLIT_NAV_FIX = """
<script>
// Fix for Streamlit navigation in PWA
if (window.matchMedia('(display-mode: standalone)').matches) {
    // PWA is running in standalone mode
    console.log('PWA in standalone mode');
    
    // Prevent Streamlit from breaking on back button
    window.history.pushState(null, null, window.location.href);
    window.onpopstate = function () {
        window.history.go(1);
    };
}
</script>
"""


# ====================================================================
# 8. ONLINE/OFFLINE INDICATOR
# ====================================================================

def handle_offline_mode():
    """Display offline mode indicator and handle data storage"""
    
    online_check = """
    <script>
        const isOnline = navigator.onLine;
        const statusDiv = document.createElement('div');
        statusDiv.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 8px 12px;
            border-radius: 20px;
            background: ${isOnline ? '#28a745' : '#dc3545'};
            color: white;
            font-size: 12px;
            font-weight: bold;
            z-index: 9999;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            display: flex;
            align-items: center;
            gap: 5px;
        `;
        statusDiv.innerHTML = isOnline ? '🟢 Online' : '🔴 Offline';
        document.body.appendChild(statusDiv);
    </script>
    """
    
    st.markdown(online_check, unsafe_allow_html=True)


# ====================================================================
# 9. STREAMLIT CLOUD PATH FIX
# ====================================================================

def fix_paths_for_streamlit_cloud():
    """Fix file paths for Streamlit Cloud deployment"""
    
    # Check if running on Streamlit Cloud
    is_streamlit_cloud = os.environ.get('STREAMLIT_SHARING', '').lower() == 'true'
    
    if is_streamlit_cloud:
        print("⚙️ Detected Streamlit Cloud - fixing paths...")
        
        # Update manifest.json paths
        manifest_path = Path("static/manifest.json")
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Update paths to be relative
            manifest['icons'] = [
                {**icon, 'src': icon['src'].replace('/app/static/', './static/')}
                for icon in manifest['icons']
            ]
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
        
        # Update service worker paths
        sw_path = Path("static/service-worker.js")
        if sw_path.exists():
            with open(sw_path, 'r') as f:
                content = f.read()
            
            # Replace absolute paths with relative
            content = content.replace("/app/static/", "./static/")
            content = content.replace("'/offline.html'", "'./static/offline.html'")
            
            with open(sw_path, 'w') as f:
                f.write(content)
        
        print("✅ Paths fixed for Streamlit Cloud")


# ====================================================================
# 10. MAIN INITIALIZATION FUNCTION
# ====================================================================

def initialize_pwa():
    """
    Initialize PWA for Streamlit app.
    Call this at the very start of your main() function.
    """
    # MUST be the first Streamlit command
    st.set_page_config(
        page_title="African NeuroHealth AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/yourusername/neurohealth-ai',
            'Report a bug': 'https://github.com/yourusername/neurohealth-ai/issues',
            'About': "NeuroHealth AI - Stroke and Dementia Risk Assessment for African Populations"
        }
    )
    
    # Create all necessary files
    create_manifest()
    create_service_worker()
    create_offline_page()
    
    # Fix paths for Streamlit Cloud
    fix_paths_for_streamlit_cloud()
    
    # Inject PWA code into page
    inject_pwa_code()
    
    # Add navigation fix
    st.markdown(STREAMLIT_NAV_FIX, unsafe_allow_html=True)
    
    # Add offline storage helper
    offline_storage_js = create_offline_storage_helper()
    st.markdown(offline_storage_js, unsafe_allow_html=True)
    
    # Add performance optimizations
    perf_js = optimize_pwa_performance()
    st.markdown(perf_js, unsafe_allow_html=True)
    
    # Handle offline mode
    handle_offline_mode()
    
    # Display install instructions
    with st.sidebar.expander("📱 Install as App", expanded=False):
        st.markdown("""
        ### Use Offline
        
        **On Mobile:**
        1. Tap the share button (⎋ or ↑)
        2. Select "Add to Home Screen"
        3. Open the app from your home screen
        
        **On Desktop:**
        1. Click the install icon in address bar (⊕)
        2. Click "Install"
        3. Launch from desktop or start menu
        
        **Offline Features:**
        - ✅ Complete assessments offline
        - ✅ View previous reports
        - ✅ Educational resources
        - ✅ Automatic sync when online
        
        **Storage:** ~10MB required
        """)


# ====================================================================
# 11. CREATE APP ICONS
# ====================================================================

def create_app_icons():
    """
    Helper function to create app icons from your logo.
    Requires Pillow library.
    """
    try:
        from PIL import Image
        
        # Load your logo (update this path)
        logo_path = "logo.png"  # Change to your logo path
        img = Image.open(logo_path)
        
        static_dir = Path("static")
        static_dir.mkdir(exist_ok=True)
        
        # Create 192x192 icon
        img_192 = img.resize((192, 192), Image.Resampling.LANCZOS)
        img_192.save(static_dir / "icon-192x192.png")
        
        # Create 512x512 icon
        img_512 = img.resize((512, 512), Image.Resampling.LANCZOS)
        img_512.save(static_dir / "icon-512x512.png")
        
        print("✅ Icons created successfully!")
        return True
        
    except Exception as e:
        print(f"⚠️ Warning: Could not create icons: {e}")
        print("Please create icons manually:")
        print("1. Create 192x192 PNG icon as static/icon-192x192.png")
        print("2. Create 512x512 PNG icon as static/icon-512x512.png")
        return False

def create_icons():
    """Create simple icons if they don't exist"""
    import os
    if not os.path.exists("static/icon-192.png"):
        try:
            from PIL import Image, ImageDraw
            # Create simple icon
            img = Image.new('RGBA', (192, 192), (102, 126, 234, 255))
            draw = ImageDraw.Draw(img)
            # Add brain emoji or your logo
            # Save
            img.save("static/icon-192.png")
            img.resize((512, 512)).save("static/icon-512.png")
            print("✅ Icons created")
        except:
            print("⚠️ Could not create icons - please add manually")
# ====================================================================
# 12. TEST FUNCTION
# ====================================================================

def test_pwa_functionality():
    """Test PWA functionality in the app"""
    
    st.sidebar.subheader("PWA Test")
    
    if st.sidebar.button("Test Offline Storage"):
        st.write("""
        <script>
        // Test saving offline data
        const testData = {
            test: "offline",
            timestamp: new Date().toISOString(),
            score: Math.random()
        };
        
        if (window.offlineStorage) {
            window.offlineStorage.saveAssessment(testData);
            alert('Test data saved! Check pending badge at bottom right.');
        } else {
            alert('Offline storage not initialized');
        }
        </script>
        """, unsafe_allow_html=True)
    
    if st.sidebar.button("Check Service Worker"):
        st.write("""
        <script>
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.getRegistration().then(reg => {
                if (reg) {
                    alert('✅ Service Worker Registered:\\n' + reg.scope);
                } else {
                    alert('❌ No Service Worker found');
                }
            });
        } else {
            alert('❌ Service Workers not supported');
        }
        </script>
        """, unsafe_allow_html=True)
    
    if st.sidebar.button("Clear Offline Data"):
        st.write("""
        <script>
        localStorage.removeItem('pendingAssessments');
        if ('caches' in window) {
            caches.delete('offline-assessments');
        }
        alert('Offline data cleared!');
        location.reload();
        </script>
        """, unsafe_allow_html=True)


# ====================================================================
# 13. USAGE EXAMPLE
# ====================================================================

def main_example():
    """Example main function showing how to use the PWA"""
    
    # Initialize PWA (MUST be first)
    initialize_pwa()
    
    # Add PWA test functionality
    test_pwa_functionality()
    
    # Your app content
    st.title("🧠 African NeuroHealth AI")
    st.markdown("### AI-powered stroke and dementia risk assessment")
    
    # Example section showing how to use offline storage
    st.subheader("Patient Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Patient Name")
        age = st.number_input("Age", 18, 100, 30)
    
    with col2:
        systolic_bp = st.number_input("Systolic BP", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol", 100, 400, 200)
    
    if st.button("Save Assessment"):
        assessment_data = {
            "name": name,
            "age": age,
            "systolic_bp": systolic_bp,
            "cholesterol": cholesterol,
            "timestamp": str(st.session_state.get("timestamp", "now"))
        }
        
        st.success("Assessment saved!")
        
        # Save to offline storage
        st.write(f"""
        <script>
        const assessmentData = {json.dumps(assessment_data)};
        if (window.offlineStorage) {{
            window.offlineStorage.saveAssessment(assessmentData);
        }}
        </script>
        """, unsafe_allow_html=True)
    
    # Show pending assessments
    st.sidebar.subheader("System Status")
    st.sidebar.info("PWA is active. Install for offline use.")
    
    # Create icons if needed (run this once)
    if st.sidebar.button("Generate App Icons"):
        if create_app_icons():
            st.sidebar.success("Icons created! Refresh the page.")
        else:
            st.sidebar.error("Failed to create icons. Check logs.")


# ====================================================================
# 14. DEPLOYMENT CONFIGURATION
# ====================================================================

def get_deployment_config():
    """
    Returns configuration for Streamlit Cloud deployment
    Copy this to your .streamlit/config.toml file
    """
    return """
# .streamlit/config.toml
[server]
maxUploadSize = 200
enableCORS = true
enableXsrfProtection = true
maxMessageSize = 200

[browser]
gatherUsageStats = false
serverAddress = "localhost"

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[client]
showErrorDetails = true
"""


# ====================================================================
# 15. MAIN EXECUTION
# ====================================================================

if __name__ == "__main__":
    # This runs the example when you execute the file directly
    main_example()