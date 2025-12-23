import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import cv2
import time
import os

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="PaletteIQ", layout="centered")

# ---------------- FUN + CHIC CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Poppins:wght@300;400;500&display=swap');

html, body, .stApp {
    background-color: #F4C2C2;
    font-family: 'Poppins', sans-serif;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
    color: #CD5E77;
}

.fun-card {
    background: linear-gradient(135deg, #EBA7AC, #EE959E);
    padding: 18px;
    border-radius: 22px;
    margin-bottom: 16px;
    box-shadow: 0 8px 20px rgba(205,94,119,0.25);
}

.palette-box {
    border-radius: 16px;
    overflow: hidden;
    margin-top: 10px;
}

.highlight {
    color: #CD5E77;
    font-weight: 600;
}

.stButton>button {
    background-color: #CD5E77;
    color: white;
    border-radius: 20px;
    border: none;
    padding: 10px 26px;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #E17F93;
}
</style>
""", unsafe_allow_html=True)

# ---------------- IMAGE PROCESSING ----------------
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((300, 300))
    return np.array(image)

def extract_dominant_colors(image, k=6):
    pixels = image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_.astype(int)

# ---------------- SKIN TONE (PDF LOGIC) ----------------
def detect_skin_tone(color):
    r, g, b = color
    if r > 200 and g > 170:
        return "Warm"
    if b > r and b > g:
        return "Cool"
    return "Neutral"

# ---------------- COLOR THEORY ----------------
def color_psychology(color):
    r, g, b = color
    if r > 200:
        return "Bold, confident, feminine power"
    if b > 150:
        return "Calm, elegant, luxury energy"
    if g > 150:
        return "Fresh, youthful, balanced"
    return "Soft, romantic, effortless chic"

def style_suggestion(color, skin_tone):
    if skin_tone == "Warm":
        return "Gold accents, earthy tones, peachy pinks work beautifully."
    if skin_tone == "Cool":
        return "Silver jewelry, jewel tones, icy pastels elevate this look."
    return "You can experiment freely — both warm and cool shades suit you."

def celebrity_match(color, skin_tone):
    if skin_tone == "Warm":
        return "Deepika Padukone • Zendaya • Beyoncé"
    if skin_tone == "Cool":
        return "Taylor Swift • Anne Hathaway • Kendall Jenner"
    return "Hailey Bieber • Alia Bhatt • Rosé (BLACKPINK)"

# ---------------- PALETTE VIS ----------------
def show_palette(colors):
    fig, ax = plt.subplots(figsize=(7, 1.8))
    ax.axis("off")
    for i, color in enumerate(colors):
        ax.add_patch(
            plt.Rectangle((i, 0), 1, 1, color=np.array(color)/255)
        )
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    st.pyplot(fig)

# ---------------- UI ----------------
st.title("PaletteIQ ✨")
st.subheader("AI Fashion Color Palette Generator")

st.markdown(
    "<div class='fun-card'>Upload an outfit image and get color harmony, "
    "skin tone insights, celebrity vibes & styling magic 💖</div>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload your outfit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    start = time.time()

    image = preprocess_image(uploaded_file)
    st.image(image, caption="Your Outfit", use_column_width=True)

    colors = extract_dominant_colors(image)
    primary_color = colors[0]
    primary_color = [int(c) for c in primary_color]  # FIX np.int64

    skin_tone = detect_skin_tone(primary_color)

    st.markdown("<h3>💄 Your Color Story</h3>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="fun-card">
    🎨 <b>Primary Color:</b> RGB {tuple(primary_color)}<br>
    🌸 <b>Skin Tone Match:</b> {skin_tone}<br>
    🧠 <b>Color Psychology:</b> {color_psychology(primary_color)}<br>
    👗 <b>Styling Tip:</b> {style_suggestion(primary_color, skin_tone)}<br>
    🌟 <b>Celebrity Vibe:</b> {celebrity_match(primary_color, skin_tone)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3>🎨 Your Palette</h3>", unsafe_allow_html=True)
    show_palette(colors)

    st.success(f"✨ Done in {round(time.time() - start, 2)} seconds")
