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

.dataframe {{
    border-radius: 12px;
    overflow: hidden;
    background: {PRIMARY};
    color: {TEXT_LIGHT};
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

# ------------------ EXTRACT COLORS FROM IMAGE ------------------
def extract_image_colors(image, n_colors=6):
    """Extract dominant colors from the entire fashion image"""
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (150, 150))
    pixels = img_resized.reshape(-1, 3)
    
    # Remove very dark and very bright pixels (likely background/noise)
    mask = np.all((pixels > 20) & (pixels < 245), axis=1)
    pixels = pixels[mask]
    
    if len(pixels) < 100:
        pixels = img_resized.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_.astype(int)
    counts = np.bincount(kmeans.labels_)
    percentages = (counts / counts.sum()) * 100
    
    # Sort by percentage
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

def get_season(u, v, c):
    return SEASON_MAP.get((u, v, c), "Neutral / Transitional")

def get_recommended_palette(season):
    return SEASON_PALETTES.get(season, SEASON_PALETTES["Neutral / Transitional"])

def get_avoid_colors(season):
    return SEASON_AVOID.get(season, [])

OUTFIT_SUGGESTIONS = {
    "Light Spring": [
        {"name": "Pastel Floral Dress", "url": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400", "desc": "Light peach or mint dress"},
        {"name": "Cream Blazer Set", "url": "https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=400", "desc": "Warm ivory suit"},
        {"name": "Coral Summer Outfit", "url": "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=400", "desc": "Bright coral top with white"},
    ],
    "Warm Spring": [
        {"name": "Golden Yellow Dress", "url": "https://images.unsplash.com/photo-1496747611176-843222e1e57c?w=400", "desc": "Vibrant yellow sundress"},
        {"name": "Coral & Turquoise", "url": "https://images.unsplash.com/photo-1539008835657-9e8e9680c956?w=400", "desc": "Warm coral with turquoise accents"},
        {"name": "Peach Blazer Look", "url": "https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?w=400", "desc": "Peach blazer with cream pants"},
    ],
    "Soft Autumn": [
        {"name": "Olive Green Ensemble", "url": "https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=400", "desc": "Muted olive dress"},
        {"name": "Camel Coat Look", "url": "https://images.unsplash.com/photo-1591369822096-ffd140ec948f?w=400", "desc": "Warm camel tones"},
        {"name": "Terracotta Outfit", "url": "https://images.unsplash.com/photo-1558769132-cb1aea628c53?w=400", "desc": "Earthy terracotta"},
    ],
    "Dark Autumn": [
        {"name": "Burgundy Evening", "url": "https://images.unsplash.com/photo-1572804013309-59a88b7e92f1?w=400", "desc": "Deep burgundy dress"},
        {"name": "Forest Green", "url": "https://images.unsplash.com/photo-1551488831-00ddcb6c6bd3?w=400", "desc": "Rich forest green"},
        {"name": "Chocolate Brown", "url": "https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=400", "desc": "Warm brown tones"},
    ],
    "Light Summer": [
        {"name": "Lavender Dress", "url": "https://images.unsplash.com/photo-1566174053879-31528523f8ae?w=400", "desc": "Soft lavender"},
        {"name": "Powder Blue", "url": "https://images.unsplash.com/photo-1617019114583-affb34d1b3cd?w=400", "desc": "Light powder blue"},
        {"name": "Rose Pink", "url": "https://images.unsplash.com/photo-1583496661160-fb5886a0aaaa?w=400", "desc": "Soft dusty rose"},
    ],
    "Cool Summer": [
        {"name": "Slate Blue Suit", "url": "https://images.unsplash.com/photo-1594633312681-425c7b97ccd1?w=400", "desc": "Cool slate blue"},
        {"name": "Mauve Elegance", "url": "https://images.unsplash.com/photo-1602810319428-019690571b5b?w=400", "desc": "Soft mauve tones"},
        {"name": "Soft Gray", "url": "https://images.unsplash.com/photo-1581044777550-4cfa60707c03?w=400", "desc": "Cool gray outfit"},
    ],
    "Dark Winter": [
        {"name": "Navy Power Suit", "url": "https://images.unsplash.com/photo-1598522325074-042db73aa4e6?w=400", "desc": "Deep navy blue"},
        {"name": "Emerald Evening", "url": "https://images.unsplash.com/photo-1585487000160-6ebcfceb0d03?w=400", "desc": "Rich emerald green"},
        {"name": "Royal Purple", "url": "https://images.unsplash.com/photo-1618932260643-eee4a2f652a6?w=400", "desc": "Deep royal purple"},
    ],
    "Bright Winter": [
        {"name": "Electric Blue", "url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400", "desc": "Bright cobalt blue"},
        {"name": "Hot Pink", "url": "https://images.unsplash.com/photo-1562137369-1a1a0bc66744?w=400", "desc": "Vibrant fuchsia"},
        {"name": "Pure White", "url": "https://images.unsplash.com/photo-1617137968427-85924c800a22?w=400", "desc": "Crisp white ensemble"},
    ],
}

def get_outfit_suggestions(season):
    return OUTFIT_SUGGESTIONS.get(season, OUTFIT_SUGGESTIONS.get("Light Spring", []))

# ------------------ MAIN ------------------
uploaded = st.file_uploader("📤 Upload a fashion image", ["jpg", "png", "jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    
    # Display uploaded image
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📸 Uploaded Image</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # EXTRACT COLORS FROM THE FASHION IMAGE ITSELF
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎨 Extracted Colors from Your Image</div>', unsafe_allow_html=True)
    st.markdown(f'<p style="color:{TEXT_LIGHT}; font-size:1.1rem; margin-bottom:1.5rem;">These are the dominant colors detected in the uploaded fashion image</p>', unsafe_allow_html=True)
    
    colors, percentages = extract_image_colors(image, n_colors)
    
    # Display extracted colors
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
    
    # Display image color analysis table
    st.markdown('<div style="margin-top:2rem;">', unsafe_allow_html=True)
    df_image = pd.DataFrame(
        image_analysis_rows,
        columns=["Color", "Percentage", "Undertone", "Value", "Chroma"]
    )
    st.dataframe(df_image, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # NOW DO FACE DETECTION AND PERSONAL ANALYSIS
    face = detect_face(image)
    
    if face is not None and show_skin:
        skin_pixels = extract_skin_pixels(face)
        
        if len(skin_pixels) >= 300:
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
            avoid_colors = get_avoid_colors(detected_season)
            
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">🎨 Your Recommended Color Palette</div>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:{TEXT_LIGHT}; font-size:1.1rem; margin-bottom:1.5rem;">Colors that will complement your <strong>{detected_season}</strong> complexion</p>', unsafe_allow_html=True)

            # Display color swatches
            cols = st.columns(len(recommended_colors))
            analysis_rows = []
            
            for col, hex_code in zip(cols, recommended_colors):
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

                analysis_rows.append([hex_code.upper(), color_undertone, color_value, color_chroma])

            st.markdown('<div style="margin-top:2rem;">', unsafe_allow_html=True)
            st.markdown(f'<div class="section-title">📊 Detailed Color Analysis</div>', unsafe_allow_html=True)
            df = pd.DataFrame(
                analysis_rows,
                columns=["Color", "Undertone", "Value", "Chroma"]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Colors to Avoid Section
            if avoid_colors:
                st.markdown('<div class="results-container">', unsafe_allow_html=True)
                st.markdown(f'<div class="section-title">❌ Colors to Avoid</div>', unsafe_allow_html=True)
                st.markdown(f'<p style="color:{TEXT_LIGHT}; font-size:1.1rem; margin-bottom:1.5rem;">These shades may clash with your <strong>{detected_season}</strong> complexion</p>', unsafe_allow_html=True)
                
                cols = st.columns(len(avoid_colors))
                for col, hex_code in zip(cols, avoid_colors):
                    with col:
                        st.markdown(f"""
                        <div class="swatch" style="background:{hex_code}; opacity:0.7;">
                            <div style="font-size:1.2rem;">✕</div>
                            <div class="swatch-hex">{hex_code.upper()}</div>
                        </div>
                        """, unsafe_allow
