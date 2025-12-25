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
PRIMARY = "#1a1a2e"
SECONDARY = "#16213e"
ACCENT = "#0f3460"
DARK = "#0a192f"
LIGHT = "#eaeaea"
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

[data-testid="stSidebar"] p {{
    color: {TEXT_LIGHT} !important;
}}

.header-container {{
    text-align: center;
    padding: 2rem 0 3rem 0;
    color: white;
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
    display: flex;
    align-items: center;
    gap: 0.5rem;
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

.how-it-works {{
    background: {SECONDARY};
    border-radius: 24px;
    padding: 3rem;
    margin: 2rem 0;
    box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    border: 2px solid {ACCENT};
}}

.step {{
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 1rem 0;
    font-size: 1.1rem;
    color: {TEXT_LIGHT};
}}

.step-number {{
    background: {TEXT_ACCENT};
    color: {DARK};
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    flex-shrink: 0;
}}

.sidebar-header {{
    font-size: 1.2rem;
    font-weight: 700;
    color: {TEXT_LIGHT};
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

.stButton > button {{
    background: {TEXT_ACCENT};
    color: {DARK};
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    transition: all 0.3s ease;
}}

.stButton > button:hover {{
    background: {TEXT_LIGHT};
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(100, 255, 218, 0.4);
}}

[data-testid="stFileUploadDropzone"] {{
    background: {PRIMARY};
    border: 2px dashed {TEXT_ACCENT};
    border-radius: 16px;
}}

[data-testid="stFileUploadDropzone"] label {{
    color: {TEXT_LIGHT} !important;
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
show_outfit = st.sidebar.checkbox("✅ Outfit Generator", True)

st.sidebar.markdown('<div class="sidebar-header" style="margin-top:2rem;">📊 Features</div>', unsafe_allow_html=True)
st.sidebar.markdown(f"""
<div style="color:{TEXT_LIGHT};">
• 🎨 AI Color Extraction<br>
• 💅 Skin Tone Detection<br>
• 🎭 Personal Season Analysis<br>
• 👗 Outfit Generator
</div>
""", unsafe_allow_html=True)

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

def extract_skin_pixels(face_img):
    lab = cv2.cvtColor(face_img, cv2.COLOR_RGB2LAB)
    lower = np.array([20, 135, 135])
    upper = np.array([255, 180, 180])
    mask = cv2.inRange(lab, lower, upper)
    skin = cv2.bitwise_and(face_img, face_img, mask=mask)
    pixels = skin.reshape(-1, 3)
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]
    return pixels

# ------------------ EXTRACT COLORS FROM IMAGE ------------------
def extract_image_colors(image, n_colors=6):
    """Extract dominant colors from the entire fashion image"""
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (150, 150))
    pixels = img_resized.reshape(-1, 3)
    
    # Remove very dark and very bright pixels
    mask = np.all((pixels > 20) & (pixels < 245), axis=1)
    pixels = pixels[mask]
    
    if len(pixels) < 100:
        pixels = img_resized.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    percentages = (counts / counts.sum()) * 100
    
    sorted_indices = np.argsort(percentages)[::-1]
    colors = colors[sorted_indices]
    percentages = percentages[sorted_indices]
    
    return colors, percentages

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

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

SEASON_PALETTES = {
    "Light Spring": ["#FFB6C1", "#FFDAB9", "#87CEEB", "#FFE4B5", "#F0E68C", "#FFFACD"],
    "Warm Spring": ["#FF6347", "#DAA520", "#808000", "#FF4500", "#F4A460", "#D2691E"],
    "Soft Autumn": ["#BC8F8F", "#D2B48C", "#8FBC8F", "#5F9EA0", "#DDA0DD", "#C19A6B"],
    "Dark Autumn": ["#3E2723", "#556B2F", "#8B4513", "#A0522D", "#B22222", "#2F4F4F"],
    "Light Summer": ["#B0E0E6", "#E6E6FA", "#F0FFF0", "#FFE4E1", "#F5DEB3", "#D8BFD8"],
    "Cool Summer": ["#708090", "#6A5ACD", "#9370DB", "#4682B4", "#8B7D7B", "#B0C4DE"],
    "Dark Winter": ["#000000", "#800020", "#191970", "#4B0082", "#8B008B", "#2F4F4F"],
    "Bright Winter": ["#0000FF", "#FF1493", "#DC143C", "#00CED1", "#8B00FF", "#FF00FF"],
    "Neutral / Transitional": ["#D2B48C", "#BC8F8F", "#B0C4DE", "#DEB887", "#8FBC8F", "#DDA0DD"]
}

SEASON_AVOID = {
    "Light Spring": ["#000000", "#708090", "#8B7D7B", "#800020"],
    "Warm Spring": ["#4169E1", "#E6E6FA", "#B0E0E6", "#D8BFD8"],
    "Soft Autumn": ["#000000", "#FF0000", "#E6E6FA", "#F0FFF0"],
    "Dark Autumn": ["#FFB6C1", "#B0E0E6", "#D3D3D3", "#FFFACD"],
    "Light Summer": ["#FF8C00", "#DAA520", "#B22222", "#FF1493"],
    "Cool Summer": ["#A0522D", "#808000", "#D2691E", "#DAA520"],
    "Dark Winter": ["#F5DEB3", "#D2B48C", "#FFE4E1", "#A0522D"],
    "Bright Winter": ["#8B7D7B", "#808000", "#BC8F8F", "#FFFACD"],
    "Neutral / Transitional": []
}

OUTFIT_SUGGESTIONS = {
    "Light Spring": [
        {"name": "Pastel Floral Dress", "url": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400", "desc": "Light peach or mint dress"},
        {"name": "Cream Blazer Set", "url": "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=400", "desc": "Warm ivory suit"},
        {"name": "Coral Summer Outfit", "url": "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=400", "desc": "Bright coral top with white"},
    ],
}

def get_season(u, v, c):
    return SEASON_MAP.get((u, v, c), "Neutral / Transitional")

def get_recommended_palette(season):
    return SEASON_PALETTES.get(season, SEASON_PALETTES["Neutral / Transitional"])

def get_avoid_colors(season):
    return SEASON_AVOID.get(season, [])

def get_outfit_suggestions(season):
    return OUTFIT_SUGGESTIONS.get(season, OUTFIT_SUGGESTIONS.get("Light Spring", []))

# ------------------ MAIN ------------------
uploaded = st.file_uploader("📤 Upload a fashion image", ["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📸 Uploaded Image</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # EXTRACT COLORS FROM IMAGE
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎨 Extracted Colors from Your Image</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{TEXT_LIGHT}; font-size:1.1rem; margin-bottom:1.5rem;">Dominant colors detected in the uploaded fashion image</p>', unsafe_allow_html=True)
    
    colors, percentages = extract_image_colors(image, n_colors)
    
    cols = st.columns(len(colors))
    image_analysis_rows = []
    
    for col, (color, pct) in zip(cols, zip(colors, percentages)):
        hex_code = rgb_to_hex(color)
        h, v, c = analyze_hvc(color)
        u, val, chr = classify_hvc(h, v, c)
        
        with col:
            st.markdown(f"""
            <div class="swatch" style="background:{hex_code};">
                <div style="font-size:1.2rem;">●</div>
                <div class="swatch-hex">{hex_code.upper()}</div>
                <div style="font-size:0.85rem; margin-top:0.5rem;">{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        image_analysis_rows.append([hex_code.upper(), f"{pct:.1f}%", u, val, chr])
    
    st.markdown('<div style="margin-top:2rem;">', unsafe_allow_html=True)
    df_image = pd.DataFrame(
        image_analysis_rows,
        columns=["Color", "Percentage", "Undertone", "Value", "Chroma"]
    )
    st.dataframe(df_image, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # FACE DETECTION AND PERSONAL ANALYSIS
    face = detect_face(image)
    
    if face is not None and show_skin:
        skin_pixels = extract_skin_pixels(face)
        
        if len(skin_pixels) >= 300:
            avg_skin = np.mean(skin_pixels, axis=0).astype(int)
            h, v, c = analyze_hvc(avg_skin)
            undertone, value, chroma = classify_hvc(h, v, c)
            detected_season = get_season(undertone, value, chroma)
            
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🔍 Your Skin Analysis</div>', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Undertone</div><div class="metric-value">{undertone}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Value</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Chroma</div><div class="metric-value">{chroma}</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Season</div><div class="metric-value">{detected_season}</div></div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            recommended_colors = get_recommended_palette(detected_season)
            
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">🎨 Your Recommended Palette</div>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:{TEXT_LIGHT}; font-size:1.1rem; margin-bottom:1.5rem;">Colors for <strong>{detected_season}</strong></p>', unsafe_allow_html=True)

            cols = st.columns(len(recommended_colors))
            
            for col, hex_code in zip(cols, recommended_colors):
                with col:
                    st.markdown(f'<div class="swatch" style="background:{hex_code};"><div style="font-size:1.2rem;">●</div><div class="swatch-hex">{hex_code.upper()}</div></div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        if show_skin:
            st.info("💡 No face detected. Upload an image with a visible face for personalized skin analysis.")

else:
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">💅</div>
            <div class="feature-title">Skin Tone Detection</div>
            <div class="feature-desc">Discover your undertone and get personalized recommendations</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎭</div>
            <div class="feature-title">Season Analysis</div>
            <div class="feature-desc">Find out if you're a Spring, Summer, Autumn, or Winter</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">👗</div>
            <div class="feature-title">Outfit Generator</div>
            <div class="feature-desc">Get AI-powered outfit suggestions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="how-it-works">
        <div class="section-title">✨ How It Works</div>
        <div class="step"><div class="step-number">1</div><div>Upload any fashion image</div></div>
        <div class="step"><div class="step-number">2</div><div>AI extracts and analyzes colors</div></div>
        <div class="step"><div class="step-number">3</div><div>Get personalized recommendations</div></div>
    </div>
    """, unsafe_allow_html=True)
