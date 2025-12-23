import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import pandas as pd
import colorsys

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="PaletteIQ | AI Fashion Color Intelligence",
    page_icon="🎨",
    layout="wide"
)

# ------------------ THEME COLORS ------------------
PRIMARY = "#051F45"
ACCENT = "#F2C4CD"

# ------------------ CUSTOM CSS ------------------
st.markdown(f"""
<style>
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, {PRIMARY}, #0A2A5E);
}}
h1, h2, h3 {{
    font-weight: 800;
    letter-spacing: -0.03em;
    color: white;
}}
.subtitle {{
    color: {ACCENT};
    font-size: 1.2rem;
}}
.card {{
    background: white;
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.15);
}}
.badge {{
    display:inline-block;
    padding:6px 14px;
    border-radius:20px;
    background:{ACCENT};
    color:{PRIMARY};
    font-weight:700;
}}
.swatch {{
    border-radius:14px;
    padding:20px;
    color:white;
    font-weight:700;
    text-align:center;
}}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
<h1 style="text-align:center;">✨ PaletteIQ</h1>
<p class="subtitle" style="text-align:center;">
AI Fashion Color Intelligence Platform
</p>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Advanced Settings")
n_colors = st.sidebar.slider("Colors to Extract", 3, 8, 6)

show_skin = st.sidebar.checkbox("Skin Tone Analysis", True)
show_season = st.sidebar.checkbox("Season Analysis", True)
show_palette = st.sidebar.checkbox("Color Palette", True)

# ------------------ FACE DETECTION ------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w]

# ------------------ SKIN PIXEL EXTRACTION ------------------
def extract_skin_pixels(face_img):
    lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
    lower = np.array([20, 135, 135])
    upper = np.array([255, 180, 180])
    mask = cv2.inRange(lab, lower, upper)
    skin = cv2.bitwise_and(face_img, face_img, mask=mask)
    pixels = skin.reshape(-1, 3)
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]
    return pixels

# ------------------ COLOR SCIENCE ------------------
def analyze_hvc(rgb):
    r, g, b = rgb / 255
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, v, s

def classify_hvc(h, v, c):
    if h < 30 or h > 330:
        undertone = "Warm"
    elif 30 <= h <= 210:
        undertone = "Cool"
    else:
        undertone = "Neutral"

    if v > 0.75:
        value = "Light"
    elif v < 0.35:
        value = "Deep"
    else:
        value = "Medium"

    chroma = "Bright" if c > 0.4 else "Muted"
    return undertone, value, chroma

SEASON_MAP = {
    ("Warm", "Light", "Bright"): "Light Spring",
    ("Warm", "Medium", "Bright"): "Warm Spring",
    ("Warm", "Medium", "Muted"): "Soft Autumn",
    ("Warm", "Deep", "Muted"): "Dark Autumn",
    ("Cool", "Light", "Muted"): "Light Summer",
    ("Cool", "Medium", "Muted"): "Cool Summer",
    ("Cool", "Deep", "Bright"): "Dark Winter",
    ("Cool", "Bright", "Bright"): "Bright Winter",
}

def get_season(u, v, c):
    return SEASON_MAP.get((u, v, c), "Neutral / Transitional")

# ------------------ PALETTE EXTRACTION ------------------
def extract_palette(pixels, k):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_.astype(int)

# ------------------ MAIN ------------------
uploaded = st.file_uploader("Upload a fashion image", ["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True)

    face = detect_face(image)
    if face is None:
        st.error("Face not detected. Please upload a clear front-facing image.")
        st.stop()

    skin_pixels = extract_skin_pixels(face)
    if len(skin_pixels) < 300:
        st.error("Insufficient skin pixels detected.")
        st.stop()

    palette = extract_palette(skin_pixels, n_colors)

    analysis_rows = []

    st.markdown("## 🎨 Skin-Based Color Palette")

    cols = st.columns(len(palette))
    for col, rgb in zip(cols, palette):
        h, v, c = analyze_hvc(rgb)
        undertone, value, chroma = classify_hvc(h, v, c)
        season = get_season(undertone, value, chroma)
        hex_code = "#{:02x}{:02x}{:02x}".format(*rgb)

        with col:
            st.markdown(f"""
            <div class="swatch" style="background:{hex_code};">
            {hex_code}<br>
            <span class="badge">{season}</span>
            </div>
            """, unsafe_allow_html=True)

        analysis_rows.append([
            hex_code, undertone, value, chroma, season
        ])

    df = pd.DataFrame(
        analysis_rows,
        columns=["Color", "Undertone", "Value", "Chroma", "Season"]
    )

    st.markdown("## 📊 Color Theory Analysis")
    st.dataframe(df, use_container_width=True)

else:
    st.markdown("""
    <div class="card" style="text-align:center;">
    <h2 style="color:#051F45;">✨ How It Works</h2>
    <p>1️⃣ Upload a fashion image</p>
    <p>2️⃣ Face detection isolates skin</p>
    <p>3️⃣ HSV + LAB color analysis</p>
    <p>4️⃣ Hue–Value–Chroma theory applied</p>
    <p>5️⃣ Personal 12-season result</p>
    <br>
    <strong>Powered by Machine Learning & Professional Color Theory</strong>
    </div>
    """, unsafe_allow_html=True)
