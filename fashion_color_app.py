import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import time

# -----------------------------
# 1. LOAD & PREPROCESS IMAGE
# -----------------------------
def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((300, 300))
    image = np.array(image)
    return image


# -----------------------------
# 2. RGB → HSV CONVERSION
# -----------------------------
def rgb_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


# -----------------------------
# 3. K-MEANS COLOR EXTRACTION
# -----------------------------
def extract_dominant_colors(image, k=6):
    pixels = image.reshape((-1, 3))

    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    return colors, labels


# -----------------------------
# 4. COLOR HARMONY RULES
# -----------------------------
def complementary_color(color):
    return [255 - color[0], 255 - color[1], 255 - color[2]]

def analogous_colors(color):
    hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0][0]
    h = hsv[0]

    return [
        cv2.cvtColor(np.uint8([[[ (h+20)%180, hsv[1], hsv[2] ]]]), cv2.COLOR_HSV2RGB)[0][0],
        cv2.cvtColor(np.uint8([[[ (h-20)%180, hsv[1], hsv[2] ]]]), cv2.COLOR_HSV2RGB)[0][0]
    ]


# -----------------------------
# 5. COLOR PSYCHOLOGY
# -----------------------------
def color_psychology(color):
    r, g, b = color
    if r > 200 and g < 100:
        return "Confidence, power, boldness"
    if b > 150:
        return "Calm, trust, elegance"
    if g > 150:
        return "Freshness, growth, balance"
    return "Neutral, minimal, versatile"


# -----------------------------
# 6. VISUALIZE PALETTE
# -----------------------------
def plot_palette(colors):
    plt.figure(figsize=(8, 2))
    plt.axis('off')

    for i, color in enumerate(colors):
        plt.fill_between([i, i+1], 0, 1, color=np.array(color)/255)

    plt.show()


# -----------------------------
# 7. MAIN PIPELINE
# -----------------------------
def run_palette_generator(image_path):
    start_time = time.time()

    image = load_image(image_path)
    hsv_image = rgb_to_hsv(image)

    colors, labels = extract_dominant_colors(image, k=6)

    print("\n🎨 Dominant Colors (RGB):")
    for idx, color in enumerate(colors):
        print(f"{idx+1}. {color} → {color_psychology(color)}")

    print("\n✨ Harmony Suggestions:")
    for color in colors[:2]:
        print(f"Base: {color}")
        print("Complementary:", complementary_color(color))
        print("Analogous:", analogous_colors(color))
        print()

    plot_palette(colors)

    print(f"⏱️ Processing Time: {round(time.time() - start_time, 2)} seconds")


# -----------------------------
# 8. RUN
# -----------------------------
if __name__ == "__main__":
    run_palette_generator("sample_images/outfit.jpg")
