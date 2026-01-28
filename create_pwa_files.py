# create_pwa_files.py
import os
import json

def create_pwa_files():
    """Create all PWA files in one go"""
    
    # Create static folder
    os.makedirs("static", exist_ok=True)
    
    # 1. Create manifest.json
    manifest = {
        "name": "NeuroHealth AI",
        "short_name": "NeuroHealth",
        "description": "Health Risk Assessment",
        "start_url": ".",
        "display": "standalone",
        "background_color": "#ffffff",
        "theme_color": "#667eea",
        "icons": [
            {
                "src": "icon-192.png",
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "icon-512.png",
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    }
    
    with open("static/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("✅ Created manifest.json")
    
    # 2. Create service-worker.js
    service_worker = """
    console.log('Service Worker: Hello from NeuroHealth AI');
    
    self.addEventListener('install', (event) => {
        console.log('Service Worker installing...');
        self.skipWaiting();
    });
    
    self.addEventListener('activate', (event) => {
        console.log('Service Worker activated');
    });
    """
    
    with open("static/service-worker.js", "w") as f:
        f.write(service_worker)
    print("✅ Created service-worker.js")
    
    # 3. Create icons using matplotlib (no PIL required)
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Create 192x192 icon
        fig, ax = plt.subplots(figsize=(2, 2), dpi=96)
        circle = patches.Circle((0.5, 0.5), 0.4, color='#667eea')
        ax.add_patch(circle)
        ax.text(0.5, 0.5, '🧠', fontsize=60, ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig('static/icon-192.png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        
        # Create 512x512 icon
        fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
        circle = patches.Circle((0.5, 0.5), 0.4, color='#667eea')
        ax.add_patch(circle)
        ax.text(0.5, 0.5, '🧠', fontsize=200, ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig('static/icon-512.png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        
        print("✅ Created icons")
        
    except Exception as e:
        print(f"⚠️ Could not create icons with matplotlib: {e}")
        print("Please install matplotlib: pip install matplotlib")
    
    # 4. Create offline.html
    offline = """<!DOCTYPE html>
    <html><head><title>Offline</title></head>
    <body><h1>Offline Mode</h1><p>Check your connection.</p></body>
    </html>"""
    
    with open("static/offline.html", "w") as f:
        f.write(offline)
    print("✅ Created offline.html")
    
    print("\n🎉 All PWA files created in static/ folder!")
    print("Files created:")
    for file in os.listdir("static"):
        print(f"  - static/{file}")

if __name__ == "__main__":
    create_pwa_files()