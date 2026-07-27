import streamlit as st
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import pandas as pd
import colorsys
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="PaletteIQ | AI Fashion Color Intelligence",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ THEME COLORS ------------------
PRIMARY = "#1a1a2e"
SECONDARY = "#16213e"
ACCENT = "#0f3460"
DARK = "#0a192f"
TEXT_LIGHT = "#e6e6e6"
TEXT_ACCENT = "#64ffda"

# ------------------ CUSTOM CSS ------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.main {{
    background: linear-gradient(135deg, {DARK} 0%, {PRIMARY} 100%);
}}

[data-testid="stSidebar"] {{
    background: {SECONDARY};
    padding: 2rem 1rem;
}}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: {TEXT_LIGHT};
    font-weight: 700;
}}

[data-testid="stSidebar"] label {{
    color: {TEXT_LIGHT} !important;
}}

.header-container {{
    text-align: center;
    padding: 2rem 0 3rem 0;
}}

.main-title {{
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    color: {TEXT_LIGHT};
    text-shadow: 0 2px 10px rgba(0,0,0,0.5);
}}

.subtitle {{
    font-size: 1.1rem;
    color: {TEXT_ACCENT};
    font-weight: 500;
}}

.results-container {{
    background: {SECONDARY};
    border-radius: 24px;
    padding: 2.5rem;
    margin: 2rem 0;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    border: 2px solid {ACCENT};
}}

.section-title {{
    font-size: 1.8rem;
    font-weight: 800;
    color: {TEXT_LIGHT};
    margin-bottom: 1.5rem;
}}

.swatch {{
    border-radius: 16px;
    padding: 2rem 1rem;
    text-align: center;
    font-weight: 700;
    color: white;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3);
    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    transition: transform 0.2s ease;
}}

.swatch:hover {{
    transform: scale(1.05);
}}

.swatch-hex {{
    font-size: 0.9rem;
    margin-top: 0.5rem;
    opacity: 0.95;
}}

.metric-card {{
    background: linear-gradient(135deg, {ACCENT}, {PRIMARY});
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}}

.metric-label {{
    font-size: 0.85rem;
    font-weight: 600;
    color: {TEXT_ACCENT};
    margin-bottom: 0.5rem;
}}

.metric-value {{
    font-size: 1.5rem;
    font-weight: 800;
    color: {TEXT_LIGHT};
}}

.feature-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}}

.feature-card {{
    background: linear-gradient(135deg, {SECONDARY}, {ACCENT});
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid {ACCENT};
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}}

.feature-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(100, 255, 218, 0.3);
}}

.feature-icon {{
    font-size: 2.5rem;
    margin-bottom: 1rem;
}}

.feature-title {{
    font-size: 1.5rem;
    font-weight: 700;
    color: {TEXT_LIGHT};
    margin-bottom: 0.5rem;
}}

.feature-desc {{
    color: {TEXT_ACCENT};
    font-size: 0.95rem;
    line-height: 1.6;
}}

[data-testid="stFileUploadDropzone"] {{
    background: {PRIMARY};
    border: 2px dashed {TEXT_ACCENT};
    border-radius: 16px;
}}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
<div class="header-container">
    <div class="main-title">✨ AI Fashion Color Intelligence Platform</div>
    <div class="subtitle">Upload Your Photo • Get Your Perfect Color Palette</div>
</div>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.markdown(f'<div style="font-size:1.2rem; font-weight:700; color:{TEXT_LIGHT}; margin-bottom:1rem;">⚙️ Settings</div>', unsafe_allow_html=True)
n_colors = st.sidebar.slider("Palette Size", 3, 10, 6)

st.sidebar.markdown(f'<div style="font-size:1.2rem; font-weight:700; color:{TEXT_LIGHT}; margin:2rem 0 1rem 0;">📊 Features</div>', unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div style="color:{TEXT_LIGHT};">
• 🔍 Face Detection<br>
• 💅 Skin Tone Analysis<br>
• 🎭 Season Classification<br>
• 🎨 Personalized Palette<br>
• 👗 Outfit Recommendations
</div>
""", unsafe_allow_html=True)

# ------------------ FACE DETECTION (ROBUST LOADING) ------------------
@st.cache_resource(show_spinner=False)
def load_face_cascade():
    """
    Load the Haar cascade robustly. On Streamlit Cloud, cv2.data can be
    missing its 'data' attribute if opencv-python (GUI build) is installed
    instead of opencv-python-headless, or if both are installed together.
    This checks multiple fallback locations instead of crashing at import time.
    """
    candidate_paths = []

    # Preferred: cv2's own bundled data folder, if the attribute exists
    try:
        candidate_paths.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    except AttributeError:
        pass

    # Fallback: locate it relative to the installed cv2 package folder directly
    try:
        cv2_pkg_dir = os.path.dirname(cv2.__file__)
        candidate_paths.append(os.path.join(cv2_pkg_dir, "data", "haarcascade_frontalface_default.xml"))
    except Exception:
        pass

    # Fallback: a copy bundled alongside this script (recommended - see note below)
    candidate_paths.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml"))

    for path in candidate_paths:
        if path and os.path.exists(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                return cascade

    return None

face_cascade = load_face_cascade()

if face_cascade is None:
    st.error(
        "⚠️ **Face detection model failed to load.** This usually means the wrong "
        "OpenCV package is installed. In `requirements.txt`, use only "
        "`opencv-python-headless` (remove `opencv-python` if it's also listed), "
        "then redeploy. As a permanent fix, you can also download "
        "`haarcascade_frontalface_default.xml` from the OpenCV GitHub repo and place "
        "it in the same folder as this script."
    )
    st.stop()

def detect_face(image):
    """Detect face in image and return face region"""
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Try multiple detection parameters for better results
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=4,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        # Try with more relaxed parameters
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(40, 40)
        )

    if len(faces) == 0:
        return None, None

    # Get the largest face
    if len(faces) > 1:
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

    x, y, w, h = faces[0]
    face_region = img[y:y+h, x:x+w]

    return face_region, (x, y, w, h)

def extract_skin_pixels(face_img):
    """Extract skin pixels from face using LAB color space"""
    lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)

    lower = np.array([20, 125, 125])
    upper = np.array([255, 190, 190])

    mask = cv2.inRange(lab, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    skin = cv2.bitwise_and(face_img, face_img, mask=mask)
    pixels = skin.reshape(-1, 3)
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

    return pixels

# ------------------ COLOR SCIENCE ------------------
# Per classic color theory (Hue / Value / Intensity model): Hue is the pure
# color position on the wheel, Value is lightness vs darkness, and Intensity
# (chroma/saturation) is how bright vs muted a color reads. Skin tones sit in
# a narrow hue band between yellow-orange and red on the wheel - warmer
# undertones read closer to yellow-orange, cooler undertones read closer to
# red-purple. That narrow band means small hue shifts matter a lot more for
# skin than for general color classification, so thresholds below are tuned
# tighter around that band rather than the full 360° wheel.

def analyze_hvc(rgb):
    """Analyze Hue, Value, Intensity (chroma/saturation)"""
    r, g, b = rgb / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return h * 360, v, s

def classify_hvc(h, v, c):
    """Classify undertone, value, and intensity for season analysis"""
    # Skin hues live roughly between ~10° (red) and ~50° (yellow-orange).
    # Lower hue (toward red/pink/purple) = cooler undertone.
    # Higher hue (toward yellow-orange) = warmer undertone.
    if h <= 25:
        undertone = "Cool"
    elif h >= 38:
        undertone = "Warm"
    else:
        undertone = "Neutral"

    # Value (lightness)
    if v > 0.75:
        value = "Light"
    elif v < 0.35:
        value = "Deep"
    else:
        value = "Medium"

    # Intensity (chroma/saturation)
    chroma = "Bright" if c > 0.4 else "Muted"

    return undertone, value, chroma

# ------------------ SEASON MAPPING ------------------
SEASON_MAP = {
    ("Warm", "Light", "Bright"): "Light Spring",
    ("Warm", "Light", "Muted"): "Light Spring",
    ("Warm", "Medium", "Bright"): "Warm Spring",
    ("Warm", "Medium", "Muted"): "Soft Autumn",
    ("Warm", "Deep", "Bright"): "Warm Autumn",
    ("Warm", "Deep", "Muted"): "Dark Autumn",
    ("Cool", "Light", "Bright"): "Light Summer",
    ("Cool", "Light", "Muted"): "Light Summer",
    ("Cool", "Medium", "Bright"): "Cool Summer",
    ("Cool", "Medium", "Muted"): "Cool Summer",
    ("Cool", "Deep", "Bright"): "Dark Winter",
    ("Cool", "Deep", "Muted"): "Dark Winter",
    ("Neutral", "Light", "Bright"): "Bright Spring",
    ("Neutral", "Deep", "Bright"): "Bright Winter",
}

def get_season(u, v, c):
    """Determine color season based on characteristics"""
    season = SEASON_MAP.get((u, v, c), "Neutral / Transitional")
    return season

# ------------------ COLOR PALETTES ------------------
SEASON_PALETTES = {
    "Light Spring": ["#FFB6C1", "#FFDAB9", "#87CEEB", "#FFE4B5", "#F0E68C", "#FFFACD", "#FFD700", "#98FB98"],
    "Warm Spring": ["#FF6347", "#DAA520", "#FF8C00", "#FF4500", "#F4A460", "#D2691E", "#FFD700", "#FFA500"],
    "Soft Autumn": ["#BC8F8F", "#D2B48C", "#8FBC8F", "#5F9EA0", "#DDA0DD", "#C19A6B", "#CD853F", "#B8860B"],
    "Warm Autumn": ["#A0522D", "#8B4513", "#D2691E", "#CD853F", "#B8860B", "#DAA520", "#FF8C00", "#FF6347"],
    "Dark Autumn": ["#3E2723", "#556B2F", "#8B4513", "#A0522D", "#B22222", "#2F4F4F", "#8B0000", "#654321"],
    "Light Summer": ["#B0E0E6", "#E6E6FA", "#F0FFF0", "#FFE4E1", "#F5DEB3", "#D8BFD8", "#AFEEEE", "#DDA0DD"],
    "Cool Summer": ["#708090", "#6A5ACD", "#9370DB", "#4682B4", "#8B7D7B", "#B0C4DE", "#778899", "#BC8F8F"],
    "Dark Winter": ["#000000", "#800020", "#191970", "#4B0082", "#8B008B", "#2F4F4F", "#483D8B", "#00008B"],
    "Bright Winter": ["#0000FF", "#FF1493", "#DC143C", "#00CED1", "#8B00FF", "#FF00FF", "#1E90FF", "#FF0000"],
    "Bright Spring": ["#FFD700", "#FF69B4", "#00CED1", "#7FFF00", "#FF6347", "#1E90FF", "#FF1493", "#00FF7F"],
    "Neutral / Transitional": ["#D2B48C", "#BC8F8F", "#B0C4DE", "#DEB887", "#8FBC8F", "#DDA0DD", "#F5DEB3", "#C0C0C0"]
}

# Colors to avoid are chosen as the "complementary" side of the wheel from
# each season's palette - i.e. hues that would visually cancel/clash against
# the undertone rather than harmonize with it, the same logic used in
# makeup color-correction (complementary colors neutralize each other).
SEASON_AVOID = {
    "Light Spring": ["#000000", "#708090", "#8B7D7B", "#800020", "#2F4F4F"],
    "Warm Spring": ["#4169E1", "#E6E6FA", "#B0E0E6", "#D8BFD8", "#708090"],
    "Soft Autumn": ["#000000", "#FF0000", "#E6E6FA", "#F0FFF0", "#00FFFF"],
    "Warm Autumn": ["#B0E0E6", "#E6E6FA", "#D3D3D3", "#F0F8FF", "#FFFAFA"],
    "Dark Autumn": ["#FFB6C1", "#B0E0E6", "#D3D3D3", "#FFFACD", "#F0FFF0"],
    "Light Summer": ["#FF8C00", "#DAA520", "#B22222", "#FF1493", "#FF4500"],
    "Cool Summer": ["#A0522D", "#808000", "#D2691E", "#DAA520", "#FF8C00"],
    "Dark Winter": ["#F5DEB3", "#D2B48C", "#FFE4E1", "#A0522D", "#DEB887"],
    "Bright Winter": ["#8B7D7B", "#808000", "#BC8F8F", "#FFFACD", "#D2B48C"],
    "Bright Spring": ["#2F4F4F", "#556B2F", "#8B7D7B", "#696969", "#A9A9A9"],
    "Neutral / Transitional": []
}

SEASON_DESCRIPTIONS = {
    "Light Spring": "Your complexion has warm, peachy undertones with light value. Light, warm, and clear colors harmonize beautifully with your delicate coloring.",
    "Warm Spring": "You have golden warmth with vibrant energy. Warm, bright colors in yellow, orange, and coral families enhance your natural glow.",
    "Soft Autumn": "Your skin has muted warmth with medium depth. Earthy, softened tones complement your natural harmony without overwhelming.",
    "Warm Autumn": "You have rich, warm coloring. Deep, warm earth tones and golden hues match your natural richness.",
    "Dark Autumn": "Your coloring is deep and warm with rich intensity. Deep, warm, earthy colors match your dramatic depth.",
    "Light Summer": "Your complexion features cool, rosy undertones with light value. Soft, cool pastels enhance your delicate coloring.",
    "Cool Summer": "You have cool undertones with medium depth and soft contrast. Blended, cool colors harmonize beautifully.",
    "Dark Winter": "Your coloring has high contrast with cool undertones. Bold, cool, saturated colors match your dramatic presence.",
    "Bright Winter": "You have high contrast with cool undertones and vivid clarity. Clear, bright, cool colors complement your striking coloring.",
    "Bright Spring": "Your coloring is warm and clear with high contrast. Bright, warm colors with clarity enhance your vibrant appearance.",
    "Neutral / Transitional": "Your coloring has balanced characteristics, giving you flexibility with both warm and cool tones."
}

def rgb_to_hex(rgb):
    """Convert RGB to hex color code"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

# ------------------ MAIN APP ------------------
uploaded = st.file_uploader("📤 Upload Your Photo (Clear Face Photo for Best Results)", ["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📸 Uploaded Image</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with st.spinner("🔍 Detecting face..."):
        face, coords = detect_face(image)

    if face is None:
        st.markdown('</div>', unsafe_allow_html=True)
        st.error("❌ **No face detected!** Please upload a clear photo with your face visible. Tips: Good lighting, face clearly visible, front-facing photo works best.")
        st.stop()

    with col2:
        st.image(face, caption="Detected Face Region", use_container_width=True)

    st.success("✅ Face detected successfully!")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("💅 Analyzing skin tone..."):
        skin_pixels = extract_skin_pixels(face)

    if len(skin_pixels) < 100:
        st.error("❌ **Insufficient skin pixels detected.** Please upload a photo with better lighting and clearer view of your face.")
        st.stop()

    avg_skin = np.mean(skin_pixels, axis=0).astype(int)
    skin_hex = rgb_to_hex(avg_skin)
    h, v, c = analyze_hvc(avg_skin)
    undertone, value, chroma = classify_hvc(h, v, c)
    detected_season = get_season(undertone, value, chroma)

    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Your Skin Analysis</div>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Skin Tone</div>
            <div style="width:60px; height:60px; background:{skin_hex}; border-radius:50%; margin:0.5rem auto; border:3px solid white;"></div>
            <div style="font-size:0.8rem; color:{TEXT_ACCENT};">{skin_hex.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Undertone</div>
            <div class="metric-value">{undertone}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Value</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Intensity</div>
            <div class="metric-value">{chroma}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Your Season</div>
            <div class="metric-value">{detected_season}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    recommended_colors = SEASON_PALETTES.get(detected_season, SEASON_PALETTES["Neutral / Transitional"])

    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">🎨 Your Personalized Color Palette</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{TEXT_LIGHT}; font-size:1.1rem; margin-bottom:1.5rem;">{SEASON_DESCRIPTIONS[detected_season]}</p>', unsafe_allow_html=True)

    cols = st.columns(len(recommended_colors[:n_colors]))

    for col, hex_code in zip(cols, recommended_colors[:n_colors]):
        rgb_vals = [int(hex_code[i:i+2], 16) for i in (1, 3, 5)]
        brightness = sum(rgb_vals) / 3
        text_color = "white" if brightness < 128 else "black"

        with col:
            st.markdown(f"""
            <div class="swatch" style="background:{hex_code}; color:{text_color};">
                <div style="font-size:1.5rem;">●</div>
                <div class="swatch-hex">{hex_code.upper()}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    avoid_colors = SEASON_AVOID.get(detected_season, [])

    if avoid_colors:
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">❌ Colors to Avoid</div>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:{TEXT_LIGHT}; font-size:1.1rem; margin-bottom:1.5rem;">These shades sit on the complementary side of the color wheel from your <strong>{detected_season}</strong> undertone, so they tend to fight it rather than blend</p>', unsafe_allow_html=True)

        cols = st.columns(len(avoid_colors))
        for col, hex_code in zip(cols, avoid_colors):
            rgb_vals = [int(hex_code[i:i+2], 16) for i in (1, 3, 5)]
            brightness = sum(rgb_vals) / 3
            text_color = "white" if brightness < 128 else "black"

            with col:
                st.markdown(f"""
                <div class="swatch" style="background:{hex_code}; color:{text_color}; opacity:0.7;">
                    <div style="font-size:1.5rem;">✕</div>
                    <div class="swatch-hex">{hex_code.upper()}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Face Detection</div>
            <div class="feature-desc">Advanced AI detects your face automatically</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💅</div>
            <div class="feature-title">Skin Analysis</div>
            <div class="feature-desc">Analyzes thousands of skin pixels for accuracy</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎭</div>
            <div class="feature-title">Season Classification</div>
            <div class="feature-desc">Determines if you're Spring, Summer, Autumn, or Winter</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎨</div>
            <div class="feature-title">Custom Palette</div>
            <div class="feature-desc">Get colors scientifically matched to your skin tone</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="results-container">
        <div class="section-title">✨ How It Works</div>
        <div style="color:{TEXT_LIGHT}; line-height:2; font-size:1.05rem;">
            <p><strong>Step 1:</strong> Upload a clear photo of yourself (good lighting, face visible)</p>
            <p><strong>Step 2:</strong> Our AI detects your face and extracts skin tone data</p>
            <p><strong>Step 3:</strong> Advanced color analysis determines your undertone, value, and intensity</p>
            <p><strong>Step 4:</strong> Get your personalized color palette and season classification</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
