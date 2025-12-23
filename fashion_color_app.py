import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from sklearn.cluster import KMeans
from skimage import color
import matplotlib.pyplot as plt
import pandas as pd
import webcolors

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="PaletteIQ | AI Fashion Color Intelligence",
    page_icon="🎨",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    font-weight: 800;
    letter-spacing: -0.03em;
}
.card {
    background: white;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
}
.badge {
    display:inline-block;
    padding:6px 14px;
    border-radius:20px;
    background:#6C63FF;
    color:white;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
<h1 style="text-align:center;">✨ PaletteIQ</h1>
<p style="text-align:center;font-size:1.2rem;">
AI Fashion Color Intelligence Platform
</p>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Analysis Controls")
n_colors = st.sidebar.slider("Colors to Extract", 3, 8, 6)

features = {
    "Skin Tone Analysis": st.sidebar.checkbox("Skin Tone Analysis", True),
    "Season Analysis": st.sidebar.checkbox("Season Analysis", True),
    "Palette Extraction": st.sidebar.checkbox("Palette Extraction", True)
}

# ------------------ FACE + SKIN EXTRACTION ------------------
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

def extract_skin_pixels(image):
    img = np.array(image)
    h, w, _ = img.shape
    results = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if not results.multi_face_landmarks:
        return None

    mask = np.zeros((h, w), dtype=np.uint8)

    for lm in results.multi_face_landmarks:
        for i in [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]:
            x = int(lm.landmark[i].x * w)
            y = int(lm.landmark[i].y * h)
            cv2.circle(mask, (x,y), 20, 255, -1)

    skin = cv2.bitwise_and(img, img, mask=mask)
    pixels = skin.reshape(-1,3)
    pixels = pixels[np.any(pixels != [0,0,0], axis=1)]
    return pixels

# ------------------ COLOR NAMING ------------------
def closest_color(rgb):
    min_dist = float("inf")
    name = "Unknown"
    for hex_code, cname in webcolors.CSS3_HEX_TO_NAMES.items():
        r,g,b = webcolors.hex_to_rgb(hex_code)
        dist = np.linalg.norm(np.array(rgb) - np.array([r,g,b]))
        if dist < min_dist:
            min_dist = dist
            name = cname
    return name.title()

# ------------------ COLOR THEORY ------------------
def analyze_hvc(rgb):
    lab = color.rgb2lab(np.array([[rgb]])/255)[0][0]
    hsv = color.rgb2hsv(np.array([[rgb]])/255)[0][0]

    hue = hsv[0]*360
    value = hsv[2]
    chroma = hsv[1]

    return hue, value, chroma

def classify_season(h, v, c):
    if h < 60 or h > 300:
        hue = "Warm"
    elif 180 < h < 300:
        hue = "Cool"
    else:
        hue = "Neutral"

    if v > 0.65:
        value = "Light"
    else:
        value = "Dark"

    if c > 0.45:
        chroma = "Bright"
    else:
        chroma = "Muted"

    # 12-season logic (PDF-based)
    if hue=="Warm" and value=="Light" and chroma=="Bright":
        return "Light Spring"
    if hue=="Warm" and chroma=="Muted":
        return "Autumn"
    if hue=="Cool" and value=="Light":
        return "Summer"
    if hue=="Cool" and chroma=="Bright":
        return "Winter"

    return "Neutral Season"

# ------------------ PALETTE EXTRACTION ------------------
def extract_palette(pixels, k):
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_.astype(int)

# ------------------ MAIN ------------------
uploaded = st.file_uploader("Upload a fashion image", ["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True)

    skin_pixels = extract_skin_pixels(image)

    if skin_pixels is None:
        st.error("No face detected.")
        st.stop()

    palette = extract_palette(skin_pixels, n_colors)

    st.markdown("## 🎨 Extracted Skin-Based Palette")

    cols = st.columns(len(palette))
    analysis = []

    for col, color_rgb in zip(cols, palette):
        h,v,c = analyze_hvc(color_rgb)
        season = classify_season(h,v,c)
        name = closest_color(color_rgb)

        with col:
            st.markdown(f"""
            <div class="card" style="background:rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]});color:white;">
            <strong>{name}</strong><br>
            RGB {tuple(color_rgb)}<br>
            <span class="badge">{season}</span>
            </div>
            """, unsafe_allow_html=True)

        analysis.append([name, season, round(h,1), round(v,2), round(c,2)])

    df = pd.DataFrame(
        analysis,
        columns=["Color Name","Season","Hue","Value","Chroma"]
    )

    st.markdown("## 📊 Color Science Breakdown")
    st.dataframe(df, use_container_width=True)

else:
    st.markdown("""
    <div class="card" style="text-align:center;">
    <h2>🚀 How It Works</h2>
    <p>Upload → Face Detection → Skin Extraction → Color Science → Personal Season</p>
    <p><strong>Powered by Machine Learning & Professional Color Theory</strong></p>
    </div>
    """, unsafe_allow_html=True)
