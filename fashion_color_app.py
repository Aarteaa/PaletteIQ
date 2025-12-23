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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ THEME COLORS ------------------
PRIMARY = "#5B6FD8"
SECONDARY = "#7B8FE8"
ACCENT = "#F2C4CD"
DARK = "#2D3748"
LIGHT = "#F7FAFC"

# ------------------ CUSTOM CSS ------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.main {{
    background: linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%);
}}

/* Sidebar Styling */
[data-testid="stSidebar"] {{
    background: white;
    padding: 2rem 1rem;
}}

[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: {DARK};
    font-weight: 700;
}}

/* Header */
.header-container {{
    text-align: center;
    padding: 2rem 0 3rem 0;
    color: white;
}}

.main-title {{
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    color: white;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}}

.subtitle {{
    font-size: 1.1rem;
    color: rgba(255,255,255,0.9);
    font-weight: 500;
}}

/* Upload Card */
.upload-card {{
    background: white;
    border-radius: 24px;
    padding: 3rem;
    text-align: center;
    box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    margin: 2rem auto;
    max-width: 900px;
}}

/* Feature Cards */
.feature-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}}

.feature-card {{
    background: linear-gradient(135deg, rgba(255,255,255,0.15), rgba(255,255,255,0.05));
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid rgba(255,255,255,0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}}

.feature-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}}

.feature-icon {{
    font-size: 2.5rem;
    margin-bottom: 1rem;
}}

.feature-title {{
    font-size: 1.5rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.5rem;
}}

.feature-desc {{
    color: rgba(255,255,255,0.85);
    font-size: 0.95rem;
    line-height: 1.6;
}}

/* Results Container */
.results-container {{
    background: white;
    border-radius: 24px;
    padding: 2.5rem;
    margin: 2rem 0;
    box-shadow: 0 10px 40px rgba(0,0,0,0.15);
}}

.section-title {{
    font-size: 1.8rem;
    font-weight: 800;
    color: {DARK};
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

/* Color Swatches */
.palette-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}}

.swatch {{
    border-radius: 16px;
    padding: 2rem 1rem;
    text-align: center;
    font-weight: 700;
    color: white;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
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

/* Metric Cards */
.metric-card {{
    background: linear-gradient(135deg, {PRIMARY}, {SECONDARY});
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}}

.metric-label {{
    font-size: 0.85rem;
    font-weight: 600;
    opacity: 0.9;
    margin-bottom: 0.5rem;
}}

.metric-value {{
    font-size: 1.5rem;
    font-weight: 800;
}}

/* How It Works */
.how-it-works {{
    background: white;
    border-radius: 24px;
    padding: 3rem;
    margin: 2rem 0;
    box-shadow: 0 10px 40px rgba(0,0,0,0.1);
}}

.step {{
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1rem 0;
    font-size: 1.1rem;
    color: {DARK};
}}

.step-number {{
    background: {PRIMARY};
    color: white;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    flex-shrink: 0;
}}

/* Sidebar styling */
.sidebar-header {{
    font-size: 1.2rem;
    font-weight: 700;
    color: {DARK};
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

/* Table styling */
.dataframe {{
    border-radius: 12px;
    overflow: hidden;
}}

/* Buttons */
.stButton > button {{
    background: {PRIMARY};
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
}}

.stButton > button:hover {{
    background: {SECONDARY};
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
}}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("""
<div class="header-container">
    <div class="main-title">✨ AI Fashion Color Intelligence Platform</div>
    <div class="subtitle">Advanced Color Analysis • Personal Styling • Trend Insights</div>
</div>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.markdown('<div class="sidebar-header">⚙️ Advanced Settings</div>', unsafe_allow_html=True)
n_colors = st.sidebar.slider("Colors to Extract", 3, 10, 6)

st.sidebar.markdown('<div class="sidebar-header" style="margin-top:2rem;">🎯 Analysis Features</div>', unsafe_allow_html=True)
show_skin = st.sidebar.checkbox("✅ Skin Tone Analysis", True)
show_season = st.sidebar.checkbox("✅ Season Analysis", True)
show_brand = st.sidebar.checkbox("✅ Brand Matching", True)
show_outfit = st.sidebar.checkbox("✅ Outfit Generator", True)
show_trend = st.sidebar.checkbox("✅ Trend Analysis", True)

st.sidebar.markdown('<div class="sidebar-header" style="margin-top:2rem;">📊 Features</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
- 🎨 AI Color Extraction
- 💅 Skin Tone Detection
- 🎭 Personal Season Analysis
- 👔 Outfit Generator
""")

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

# Recommended color palettes for each season (hex codes)
SEASON_PALETTES = {
    "Light Spring": ["#FFE5B4", "#FFDAB9", "#FFB6C1", "#87CEEB", "#98FB98", "#FFA07A"],
    "Warm Spring": ["#FF6347", "#FFD700", "#FF8C00", "#ADFF2F", "#00CED1", "#FF69B4"],
    "Soft Autumn": ["#D2B48C", "#BC8F8F", "#CD853F", "#8FBC8F", "#B0C4DE", "#DEB887"],
    "Dark Autumn": ["#8B4513", "#A0522D", "#800000", "#556B2F", "#2F4F4F", "#8B0000"],
    "Light Summer": ["#E6E6FA", "#F0E68C", "#B0E0E6", "#FFB6C1", "#DDA0DD", "#F5DEB3"],
    "Cool Summer": ["#708090", "#778899", "#B0C4DE", "#6A5ACD", "#9370DB", "#8FBC8F"],
    "Dark Winter": ["#000080", "#8B0000", "#2F4F4F", "#4B0082", "#800080", "#191970"],
    "Bright Winter": ["#FF1493", "#00CED1", "#FF4500", "#0000FF", "#FF00FF", "#00FF00"],
    "Neutral / Transitional": ["#D2B48C", "#BC8F8F", "#B0C4DE", "#DEB887", "#8FBC8F", "#DDA0DD"]
}

def get_season(u, v, c):
    return SEASON_MAP.get((u, v, c), "Neutral / Transitional")

def get_recommended_palette(season):
    return SEASON_PALETTES.get(season, SEASON_PALETTES["Neutral / Transitional"])

# ------------------ MAIN ------------------
uploaded = st.file_uploader("📤 Upload a fashion image", ["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    
    # Display uploaded image
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📸 Uploaded Image</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    face = detect_face(image)
    if face is None:
        st.error("❌ Face not detected. Please upload a clear front-facing image.")
        st.stop()

    skin_pixels = extract_skin_pixels(face)
    if len(skin_pixels) < 300:
        st.error("❌ Insufficient skin pixels detected.")
        st.stop()

    # Analyze skin tone to determine season
    avg_skin = np.mean(skin_pixels, axis=0).astype(int)
    h, v, c = analyze_hvc(avg_skin)
    undertone, value, chroma = classify_hvc(h, v, c)
    detected_season = get_season(undertone, value, chroma)
    
    # Display skin analysis
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 Your Skin Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Undertone</div>
            <div class="metric-value">{undertone}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Value</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Chroma</div>
            <div class="metric-value">{chroma}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Season</div>
            <div class="metric-value">{detected_season}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Get recommended colors for this season
    recommended_colors = get_recommended_palette(detected_season)
    
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">🎨 Your Recommended Color Palette</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{DARK}; font-size:1.1rem; margin-bottom:1.5rem;">Colors that will complement your <strong>{detected_season}</strong> complexion</p>', unsafe_allow_html=True)

    # Display color swatches
    cols = st.columns(len(recommended_colors))
    analysis_rows = []
    
    for col, hex_code in zip(cols, recommended_colors):
        # Convert hex to RGB for analysis
        rgb = np.array([int(hex_code[i:i+2], 16) for i in (1, 3, 5)])
        h_col, v_col, c_col = analyze_hvc(rgb)
        color_undertone, color_value, color_chroma = classify_hvc(h_col, v_col, c_col)

        with col:
            st.markdown(f"""
            <div class="swatch" style="background:{hex_code};">
                <div style="font-size:1.2rem;">●</div>
                <div class="swatch-hex">{hex_code.upper()}</div>
            </div>
            """, unsafe_allow_html=True)

        analysis_rows.append([
            hex_code.upper(), color_undertone, color_value, color_chroma
        ])

    # Display analysis table
    st.markdown('<div style="margin-top:2rem;">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">📊 Detailed Color Analysis</div>', unsafe_allow_html=True)
    df = pd.DataFrame(
        analysis_rows,
        columns=["Color", "Undertone", "Value", "Chroma"]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # Feature cards
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">💅</div>
            <div class="feature-title">Skin Tone Detection</div>
            <div class="feature-desc">Discover your undertone and get personalized color recommendations</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎭</div>
            <div class="feature-title">Season Analysis</div>
            <div class="feature-desc">Find out if you're a Spring, Summer, Autumn, or Winter</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">👔</div>
            <div class="feature-title">Outfit Generator</div>
            <div class="feature-desc">Get AI-powered outfit suggestions for any occasion</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🏷️</div>
            <div class="feature-title">Brand Matching</div>
            <div class="feature-desc">See which luxury brands match your color palette</div>
        </div>
        <div class="feature-card">
            <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Trend Analysis</div>
            <div class="feature-desc">See how your style aligns with 2024-2025 trends</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎨</div>
            <div class="feature-title">Color Intelligence</div>
            <div class="feature-desc">Extract dominant colors with psychology insights</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class="how-it-works">
        <div class="section-title">✨ How It Works</div>
        <div class="step">
            <div class="step-number">1</div>
            <div>Upload any fashion image</div>
        </div>
        <div class="step">
            <div class="step-number">2</div>
            <div>AI extracts and analyzes colors</div>
        </div>
        <div class="step">
            <div class="step-number">3</div>
            <div>Get personalized recommendations</div>
        </div>
        <div class="step">
            <div class="step-number">4</div>
            <div>Download your complete analysis</div>
        </div>
        <br>
        <div style="text-align:center; color:#718096; margin-top:2rem;">
            <strong>🧠 Powered by Machine Learning & Color Theory</strong><br>
            Built with Python • Streamlit • scikit-learn
        </div>
    </div>
    """, unsafe_allow_html=True)
