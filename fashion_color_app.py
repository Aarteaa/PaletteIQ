import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import cv2
import time
import os

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

st.set_page_config(page_title="PaletteIQ", layout="centered")

# =============================
# 🌸 PROPER FASHION CSS
# =============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500&display=swap');

/* FULL BACKGROUND */
html, body, .stApp {
    background-color: #F4C2C2;
    font-family: 'Inter', sans-serif;
}

/* HEADINGS */
h1, h2, h3 {
    font-family: 'Playfair Display', serif;
    color: #CD5E77;
}

/* MAIN CONTENT BLOCK */
.block-container {
    background-color: #F4C2C2;
}

/* CARDS */
.card {
    background-color: #EBA7AC;
    padding: 16px;
    border-radius: 18px;
    margin-bottom: 14px;
}

/* BUTTON */
.stButton>button {
    background-color: #CD5E77;
    color: white;
    border-radius: 14px;
    border: none;
    padding: 10px 22px;
    font-weight: 500;
}

.stButton>button:hover {
    background-color: #E17F93;
}
</style>
""", unsafe_allow_html=True)

# =============================
# 🧠 CORE FUNCTIONS
# =============================
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((300, 300))
    return np.array(image)

def extract_dominant_colors(image, k=6):
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_.astype(int)

def color_psychology(color):
    r, g, b = color
    if r > 200 and g < 120:
        return "Bold, confident, statement-making"
    if b > 150:
        return "Elegant, calm, luxurious"
    if g > 150:
        return "Fresh, balanced, natural"
    return "Soft, romantic, versatile"

def style_suggestion(color):
    r, g, b = color
    if r > 200:
        return "Perfect for evening wear, parties, or statement pieces."
    if b > 150:
        return "Ideal for formal looks, office chic, and minimal styling."
    if g > 150:
        return "Great for daytime outfits, brunch looks, and casual elegance."
    return "Works beautifully for neutral layering and soft aesthetics."

def celebrity_match(color):
    r, g, b = color
    if r > 200:
        return "Zendaya • Deepika Padukone • Rihanna"
    if b > 150:
        return "Victoria Beckham • Kendall Jenner"
    if g > 150:
        return "Emma Watson • Alia Bhatt"
    return "Hailey Bieber • Rosé (BLACKPINK)"

def show_palette(colors):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("off")
    for i, color in enumerate(colors):
        ax.add_patch(
            plt.Rectangle((i, 0), 1, 1, color=np.array(color) / 255)
        )
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# =============================
# 🎀 UI
# =============================
st.title("PaletteIQ")
st.subheader("AI Fashion Color Palette Generator")

uploaded_file = st.file_uploader(
    "Upload an outfit or fashion image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    start = time.time()
    image = preprocess_image(uploaded_file)
    st.image(image, caption="Uploaded Outfit", use_column_width=True)

    colors = extract_dominant_colors(image)

    st.subheader("🎨 Dominant Colors & Insights")

    for color in colors:
        st.markdown(f"""
        <div class="card">
        <b>RGB {tuple(color)}</b><br>
        ✨ <b>Psychology:</b> {color_psychology(color)}<br>
        👗 <b>Styling Tip:</b> {style_suggestion(color)}<br>
        🌟 <b>Celebrity Vibe:</b> {celebrity_match(color)}
        </div>
        """, unsafe_allow_html=True)

    st.subheader("🎨 Generated Palette")
    show_palette(colors)

    st.success(f"Processed in {round(time.time() - start, 2)} seconds")
