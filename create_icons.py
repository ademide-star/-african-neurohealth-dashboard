# create_icons.py
import os
from PIL import Image, ImageDraw, ImageFont

os.makedirs("static", exist_ok=True)

# Create 192x192 icon
img = Image.new('RGBA', (192, 192), (102, 126, 234, 255))  # #667eea
draw = ImageDraw.Draw(img)

# Add a simple design
draw.ellipse((20, 20, 172, 172), fill=(255, 255, 255, 255))

img.save("static/icon-192.png")

# Create 512x512 icon
img_large = img.resize((512, 512))
img_large.save("static/icon-512.png")

print("✅ Icons created: static/icon-192.png and static/icon-512.png")