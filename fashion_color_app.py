import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import cv2
import time
import os

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="PaletteIQ", layout="centered")

# -----------------------------
# CUSTOM CSS (COLORS + FONT)
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F4C2C2;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
    color: #CD5E77;
}

.stButton>button {
    background-color: #CD5E77;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 20px;
}

.stButton>button:hover {
    background-color: #E17F93;
}

.upload-box {
    background-color: #EBA7AC;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
}

.color-card {
    background-color: #EE959E;
    padding: 10px;
    border-radius: 12px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# FUNCTIONS
# -----------------------------
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((300, 300))
    return np.array(image)

def extract_dominant_colors(image, k=6):
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_.astype(int)

def complementary_color(color):
    return [255 - c for c in color]

def analogous_colors(color):
    hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
    h, s, v = hsv
    return [
        cv2.cvtColor(np.uint8([[[ (h + 20) % 180, s, v ]]]), cv2.COLOR_HSV2RGB)[0][0],
        cv2.cvtColor(np.uint8([[[ (h - 20) % 180, s, v ]]]), cv2.COLOR_HSV2RGB)[0][0]
    ]

def color_psychology(color):
    r, g, b = color
    if r > 200 and g < 120:
        return "Confidence & bold femininity"
    if b > 150:
        return "Calm elegance & trust"
    if g > 150:
        return "Fresh, balanced energy"
    return "Soft, minimal & versatile"

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

# -----------------------------
# UI
# -----------------------------
st.title("PaletteIQ")
st.subheader("AI Fashion Color Palette Generator")

st.markdown("""
<div class="upload-box">
Upload a fashion or outfit image to instantly discover
harmonious color palettes and styling psychology.
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    start = time.time()
    image = preprocess_image(uploaded_file)

    st.image(image, caption="Uploaded Outfit", use_column_width=True)

    colors = extract_dominant_colors(image)

    st.subheader("Dominant Colors & Styling Psychology")
    for color in colors:
        st.markdown(
            f"""
            <div class="color-card">
                <b>RGB {tuple(color)}</b> — {color_psychology(color)}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.subheader("Generated Palette")
    show_palette(colors)

    st.subheader("Harmony Suggestions")
    base = colors[0]
    st.write("**Base Color:**", tuple(base))
    st.write("**Complementary:**", tuple(complementary_color(base)))
    st.write("**Analogous:**", [tuple(c) for c in analogous_colors(base)])

    st.success(f"Processed in {round(time.time() - start, 2)} seconds")
