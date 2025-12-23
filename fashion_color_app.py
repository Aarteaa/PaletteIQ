import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io
from sklearn.cluster import KMeans
import colorsys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import random

# Page configuration
st.set_page_config(
    page_title="AI Fashion Color Intelligence",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3em;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    h2, h3 {
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    .color-box {
        padding: 20px;
        border-radius: 15px;
        background: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 10px 0;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>✨ AI Fashion Color Intelligence Platform</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 1.2em;'>Advanced Color Analysis • Personal Styling • Trend Insights</p>", unsafe_allow_html=True)

# Enhanced Color Psychology Database
COLOR_PSYCHOLOGY = {
    'red': {
        'emotion': 'Passionate, Bold, Energetic',
        'occasions': 'Parties, Date nights, Power meetings',
        'season': 'Fall/Winter',
        'skin_tone': 'Warm undertones (best), Cool undertones (use burgundy)',
        'brands': ['Valentino', 'Christian Louboutin', 'Ferrari']
    },
    'blue': {
        'emotion': 'Calm, Trustworthy, Professional',
        'occasions': 'Business meetings, Interviews, Casual wear',
        'season': 'Spring/Summer',
        'skin_tone': 'All skin tones (universal)',
        'brands': ['Ralph Lauren', 'Brooks Brothers', 'Tiffany & Co.']
    },
    'green': {
        'emotion': 'Fresh, Natural, Balanced',
        'occasions': 'Outdoor events, Casual wear, Eco-friendly events',
        'season': 'Spring/Summer',
        'skin_tone': 'Warm undertones (olive greens), Cool undertones (emerald)',
        'brands': ['Lacoste', 'Burberry', 'Bottega Veneta']
    },
    'yellow': {
        'emotion': 'Happy, Optimistic, Cheerful',
        'occasions': 'Summer events, Casual outings, Beach wear',
        'season': 'Summer',
        'skin_tone': 'Warm undertones (golden yellow), Cool undertones (lemon)',
        'brands': ['Forever 21', 'Versace', 'Fendi']
    },
    'purple': {
        'emotion': 'Luxurious, Creative, Mysterious',
        'occasions': 'Evening events, Art galleries, Formal occasions',
        'season': 'Fall/Winter',
        'skin_tone': 'Cool undertones (best), Warm undertones (plum shades)',
        'brands': ['Prada', 'Cadbury', 'Yahoo (vintage)']
    },
    'orange': {
        'emotion': 'Vibrant, Friendly, Energetic',
        'occasions': 'Casual events, Sports, Social gatherings',
        'season': 'Fall',
        'skin_tone': 'Warm undertones (perfect match)',
        'brands': ['Hermès', 'Nike', 'Fanta']
    },
    'pink': {
        'emotion': 'Romantic, Feminine, Playful',
        'occasions': 'Dates, Garden parties, Spring events',
        'season': 'Spring',
        'skin_tone': 'Cool undertones (hot pink), Warm undertones (coral pink)',
        'brands': ['Victoria\'s Secret', 'Barbie', 'T-Mobile']
    },
    'brown': {
        'emotion': 'Earthy, Stable, Comfortable',
        'occasions': 'Casual wear, Office, Autumn events',
        'season': 'Fall',
        'skin_tone': 'Warm undertones (chocolate brown), All tones (camel)',
        'brands': ['Louis Vuitton', 'UGG', 'Burberry']
    },
    'black': {
        'emotion': 'Elegant, Sophisticated, Powerful',
        'occasions': 'Formal events, Evening wear, Business',
        'season': 'All seasons',
        'skin_tone': 'All skin tones (universal)',
        'brands': ['Chanel', 'Gucci', 'Armani']
    },
    'white': {
        'emotion': 'Pure, Clean, Minimalist',
        'occasions': 'Summer events, Formal occasions, Beach wear',
        'season': 'Spring/Summer',
        'skin_tone': 'All skin tones (universal)',
        'brands': ['Calvin Klein', 'Gap', 'Uniqlo']
    },
    'gray': {
        'emotion': 'Neutral, Balanced, Professional',
        'occasions': 'Business, Casual wear, Versatile',
        'season': 'All seasons',
        'skin_tone': 'Cool undertones (charcoal), Warm undertones (taupe)',
        'brands': ['Hugo Boss', 'Theory', 'Zara']
    }
}

# Season Analysis Database
SEASONAL_PALETTES = {
    'Spring': {
        'colors': ['coral', 'peach', 'warm pink', 'light turquoise', 'golden yellow'],
        'characteristics': 'Warm, clear, light colors with golden undertones',
        'celebrities': 'Emma Watson, Scarlett Johansson',
        'avoid': 'Black, pure white, dark colors'
    },
    'Summer': {
        'colors': ['soft blue', 'lavender', 'rose pink', 'soft white', 'powder blue'],
        'characteristics': 'Cool, soft, muted colors with blue undertones',
        'celebrities': 'Emily Blunt, Naomi Watts',
        'avoid': 'Orange, bright warm colors'
    },
    'Autumn': {
        'colors': ['rust', 'olive green', 'mustard', 'camel', 'deep orange'],
        'characteristics': 'Warm, rich, earthy colors with golden undertones',
        'celebrities': 'Julia Roberts, Julianne Moore',
        'avoid': 'Icy colors, bright cool tones'
    },
    'Winter': {
        'colors': ['navy', 'pure white', 'emerald', 'ruby red', 'royal purple'],
        'characteristics': 'Cool, clear, vivid colors with blue undertones',
        'celebrities': 'Megan Fox, Anne Hathaway',
        'avoid': 'Muted colors, warm beiges'
    }
}

# Fashion Trends 2024-2025
CURRENT_TRENDS = {
    'Color Trends': [
        'Digital Lavender - Tech-inspired purple',
        'Viva Magenta - Bold and empowering',
        'Buttercream Yellow - Soft and optimistic',
        'Forest Green - Sustainability focus',
        'Electric Blue - Futuristic vibes'
    ],
    'Style Trends': [
        'Dopamine Dressing - Bright, mood-boosting colors',
        'Quiet Luxury - Neutral, muted tones',
        'Maximalist Prints - Bold pattern mixing',
        'Earth Tones - Natural, organic palettes',
        'Metallic Accents - Gold and silver highlights'
    ]
}

def rgb_to_hex(rgb):
    """Convert RGB to HEX color code"""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def get_color_name(rgb):
    """Get approximate color name from RGB"""
    r, g, b = rgb[0]/255, rgb[1]/255, rgb[2]/255
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    if s < 0.1:
        if v > 0.8:
            return 'white'
        elif v < 0.2:
            return 'black'
        else:
            return 'gray'
    
    h_deg = h * 360
    
    if h_deg < 15 or h_deg >= 345:
        return 'red'
    elif h_deg < 45:
        return 'orange'
    elif h_deg < 75:
        return 'yellow'
    elif h_deg < 165:
        return 'green'
    elif h_deg < 255:
        return 'blue'
    elif h_deg < 315:
        return 'purple'
    else:
        return 'pink'

def analyze_skin_tone(colors):
    """Analyze if colors suggest warm or cool undertones"""
    warm_count = 0
    cool_count = 0
    
    for color in colors:
        r, g, b = color[0]/255, color[1]/255, color[2]/255
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h_deg = h * 360
        
        # Warm hues: reds, oranges, yellows
        if (h_deg < 60 or h_deg > 300) and s > 0.2:
            warm_count += 1
        # Cool hues: blues, purples, greens
        elif 120 < h_deg < 300 and s > 0.2:
            cool_count += 1
    
    if warm_count > cool_count:
        return "Warm", "Spring or Autumn"
    elif cool_count > warm_count:
        return "Cool", "Summer or Winter"
    else:
        return "Neutral", "Can wear most seasons"

def determine_season(colors):
    """Determine color season based on palette"""
    color_names = [get_color_name(c) for c in colors]
    
    # Calculate brightness and saturation
    avg_brightness = np.mean([colorsys.rgb_to_hsv(c[0]/255, c[1]/255, c[2]/255)[2] for c in colors])
    avg_saturation = np.mean([colorsys.rgb_to_hsv(c[0]/255, c[1]/255, c[2]/255)[1] for c in colors])
    
    undertone, _ = analyze_skin_tone(colors)
    
    # Determine season
    if undertone == "Warm":
        if avg_brightness > 0.6 and avg_saturation > 0.4:
            return "Spring"
        else:
            return "Autumn"
    else:
        if avg_brightness > 0.6:
            return "Summer"
        else:
            return "Winter"

def extract_colors(image, n_colors=6):
    """Extract dominant colors using K-means clustering"""
    img = image.resize((150, 150))
    img_array = np.array(img)
    pixels = img_array.reshape(-1, 3)
    
    mask = (pixels.sum(axis=1) > 30) & (pixels.sum(axis=1) < 735)
    pixels = pixels[mask]
    
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    counts = Counter(labels)
    
    sorted_colors = [colors[i] for i in sorted(counts, key=counts.get, reverse=True)]
    percentages = [counts[i] / len(labels) * 100 for i in sorted(counts, key=counts.get, reverse=True)]
    
    return sorted_colors, percentages

def get_complementary_color(rgb):
    """Get complementary color"""
    r, g, b = rgb[0]/255, rgb[1]/255, rgb[2]/255
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h_comp = (h + 0.5) % 1.0
    r_comp, g_comp, b_comp = colorsys.hsv_to_rgb(h_comp, s, v)
    return np.array([r_comp * 255, g_comp * 255, b_comp * 255])

def generate_outfit_suggestions(dominant_color, season):
    """Generate AI outfit suggestions based on color and season"""
    color_name = get_color_name(dominant_color)
    
    outfits = {
        'Casual': [],
        'Business': [],
        'Evening': []
    }
    
    # Casual outfits
    if season in ['Spring', 'Summer']:
        outfits['Casual'] = [
            f"{color_name.title()} sundress with white sneakers",
            f"{color_name.title()} t-shirt with denim shorts",
            f"Light {color_name} linen shirt with khaki pants"
        ]
    else:
        outfits['Casual'] = [
            f"{color_name.title()} sweater with dark jeans",
            f"{color_name.title()} flannel with black pants",
            f"Cozy {color_name} cardigan with leggings"
        ]
    
    # Business outfits
    outfits['Business'] = [
        f"{color_name.title()} blazer with gray trousers",
        f"{color_name.title()} blouse with black pencil skirt",
        f"Neutral suit with {color_name} accent tie/scarf"
    ]
    
    # Evening outfits
    outfits['Evening'] = [
        f"Elegant {color_name} cocktail dress",
        f"{color_name.title()} satin blouse with black trousers",
        f"Little black dress with {color_name} statement accessories"
    ]
    
    return outfits

def match_to_brands(colors):
    """Match colors to famous fashion brands"""
    color_names = [get_color_name(c) for c in colors]
    matched_brands = []
    
    for color_name in set(color_names):
        if color_name in COLOR_PSYCHOLOGY and 'brands' in COLOR_PSYCHOLOGY[color_name]:
            matched_brands.extend(COLOR_PSYCHOLOGY[color_name]['brands'])
    
    return list(set(matched_brands))[:5]  # Return top 5 unique brands

def analyze_trend_alignment(colors):
    """Analyze how the palette aligns with current trends"""
    color_names = [get_color_name(c) for c in colors]
    
    trend_score = 0
    matched_trends = []
    
    # Check alignment with current color trends
    if 'purple' in color_names or 'pink' in color_names:
        trend_score += 20
        matched_trends.append("Digital Lavender / Viva Magenta trend")
    
    if 'yellow' in color_names:
        trend_score += 15
        matched_trends.append("Buttercream Yellow trend")
    
    if 'green' in color_names:
        trend_score += 15
        matched_trends.append("Forest Green / Sustainability trend")
    
    if 'blue' in color_names:
        trend_score += 10
        matched_trends.append("Electric Blue trend")
    
    # Check saturation for dopamine dressing
    avg_saturation = np.mean([colorsys.rgb_to_hsv(c[0]/255, c[1]/255, c[2]/255)[1] for c in colors])
    if avg_saturation > 0.5:
        trend_score += 20
        matched_trends.append("Dopamine Dressing (bright colors)")
    elif avg_saturation < 0.3:
        trend_score += 15
        matched_trends.append("Quiet Luxury (muted tones)")
    
    return min(trend_score, 100), matched_trends

def plot_color_palette(colors, percentages, title="Color Palette"):
    """Create a visual color palette"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 2))
    
    x_start = 0
    for i, (color, percentage) in enumerate(zip(colors, percentages)):
        width = percentage / 100
        rect = patches.Rectangle((x_start, 0), width, 1, 
                                 facecolor=np.array(color)/255, 
                                 edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        
        if percentage > 5:
            ax.text(x_start + width/2, 0.5, f'{percentage:.1f}%',
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   color='white' if sum(color) < 400 else 'black')
        
        x_start += width
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    return fig

def create_color_swatch(colors, title=""):
    """Create individual color swatches"""
    fig, axes = plt.subplots(1, len(colors), figsize=(len(colors)*2, 2))
    
    if len(colors) == 1:
        axes = [axes]
    
    for ax, color in zip(axes, colors):
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, 
                                       facecolor=np.array(color)/255,
                                       edgecolor='gray', linewidth=2))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.text(0.5, -0.15, rgb_to_hex(color), 
               ha='center', fontsize=10, fontweight='bold')
    
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    return fig

# Sidebar
with st.sidebar:
    st.header("⚙️ Advanced Settings")
    n_colors = st.slider("Colors to Extract", 3, 10, 6)
    
    st.markdown("---")
    st.subheader("🎯 Analysis Features")
    show_skin_tone = st.checkbox("Skin Tone Analysis", value=True)
    show_season = st.checkbox("Season Analysis", value=True)
    show_brands = st.checkbox("Brand Matching", value=True)
    show_outfits = st.checkbox("Outfit Generator", value=True)
    show_trends = st.checkbox("Trend Analysis", value=True)
    
    st.markdown("---")
    st.markdown("### 📚 Features")
    st.markdown("""
    - 🎨 AI Color Extraction
    - 🌡️ Skin Tone Detection
    - 🍂 Personal Season Analysis
    - 👗 Outfit Generator
    - 🏷️ Brand Color Matching
    - 📈 Trend Alignment
    - 💡 Color Psychology
    """)

# Main content
uploaded_file = st.file_uploader("Upload a fashion image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Main Analysis
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='color-box'>", unsafe_allow_html=True)
        st.subheader("📸 Your Image")
        st.image(image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='color-box'>", unsafe_allow_html=True)
        st.subheader("🤖 AI Quick Analysis")
        
        with st.spinner("Analyzing with AI..."):
            colors, percentages = extract_colors(image, n_colors)
            dominant_color_name = get_color_name(colors[0])
            
            st.success("✅ Analysis Complete!")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Dominant Color", dominant_color_name.title())
                st.metric("Color Diversity", f"{len(set([get_color_name(c) for c in colors]))}")
            
            with col_b:
                undertone, season_hint = analyze_skin_tone(colors)
                st.metric("Undertone", undertone)
                st.metric("Suggested Season", season_hint.split(' or ')[0])
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Color Palette
    st.markdown("---")
    st.markdown("<div class='color-box'>", unsafe_allow_html=True)
    st.subheader("🎨 Extracted Color Palette")
    fig = plot_color_palette(colors, percentages)
    st.pyplot(fig)
    
    cols = st.columns(min(len(colors), 3))
    for i, (color, percentage) in enumerate(zip(colors, percentages)):
        with cols[i % 3]:
            hex_code = rgb_to_hex(color)
            st.markdown(f"""
            <div style='background-color: {hex_code}; padding: 20px; border-radius: 10px; margin: 5px;'>
                <p style='color: {"white" if sum(color) < 400 else "black"}; font-weight: bold; text-align: center; margin: 0;'>
                    {get_color_name(color).title()}<br>{hex_code}<br>{percentage:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Skin Tone Analysis
    if show_skin_tone:
        st.markdown("---")
        st.markdown("<div class='color-box'>", unsafe_allow_html=True)
        st.subheader("🌡️ Skin Tone & Personal Color Analysis")
        
        undertone, season_range = analyze_skin_tone(colors)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🎨 Color Undertone**")
            if undertone == "Warm":
                st.success(f"**{undertone}** - Golden, peachy, yellow undertones")
            elif undertone == "Cool":
                st.info(f"**{undertone}** - Pink, red, bluish undertones")
            else:
                st.warning(f"**{undertone}** - Balanced undertones")
        
        with col2:
            st.markdown("**🍂 Season Category**")
            st.info(f"**{season_range}**")
        
        with col3:
            st.markdown("**💡 Best Colors**")
            if undertone == "Warm":
                st.write("Oranges, yellows, warm reds, olive greens")
            elif undertone == "Cool":
                st.write("Blues, purples, cool reds, emerald greens")
            else:
                st.write("Wide range - experiment!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Season Analysis
    if show_season:
        st.markdown("---")
        st.markdown("<div class='color-box'>", unsafe_allow_html=True)
        st.subheader("🍂 Personal Season Analysis")
        
        detected_season = determine_season(colors)
        season_info = SEASONAL_PALETTES[detected_season]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### You are a **{detected_season}**!")
            st.write(f"**Characteristics:** {season_info['characteristics']}")
            st.write(f"**Best Colors:** {', '.join(season_info['colors'])}")
            st.write(f"**Celebrity Examples:** {season_info['celebrities']}")
            st.warning(f"**Colors to Avoid:** {season_info['avoid']}")
        
        with col2:
            # Season icon
            season_emoji = {'Spring': '🌸', 'Summer': '☀️', 'Autumn': '🍂', 'Winter': '❄️'}
            st.markdown(f"<h1 style='text-align: center; font-size: 5em;'>{season_emoji[detected_season]}</h1>", 
                       unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Brand Matching
    if show_brands:
        st.markdown("---")
        st.markdown("<div class='color-box'>", unsafe_allow_html=True)
        st.subheader("🏷️ Fashion Brand Color Matching")
        
        matched_brands = match_to_brands(colors)
        
        if matched_brands:
            st.write("Your color palette matches these luxury brands:")
            
            cols = st.columns(min(len(matched_brands), 5))
            for i, brand in enumerate(matched_brands):
                with cols[i]:
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 15px; border-radius: 10px; text-align: center; color: white;'>
                        <strong>{brand}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.info("💡 **Shopping Tip**: These brands often feature your color palette in their collections!")
        else:
            st.info("No specific brand matches found for this unique palette - you're a trendsetter!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Outfit Generator
    if show_outfits:
        st.markdown("---")
        st.markdown("<div class='color-box'>", unsafe_allow_html=True)
        st.subheader("👗 AI Outfit Generator")
        
        season = determine_season(colors)
        outfit_suggestions = generate_outfit_suggestions(colors[0], season)
        
        st.markdown(f"**Personalized for {season} Season**")
        
        tab1, tab2, tab3 = st.tabs(["👔 Casual", "💼 Business", "🌟 Evening"])
        
        with tab1:
            for outfit in outfit_suggestions['Casual']:
                st.markdown(f"• {outfit}")
        
        with tab2:
            for outfit in outfit_suggestions['Business']:
                st.markdown(f"• {outfit}")
        
        with tab3:
            for outfit in outfit_suggestions['Evening']:
                st.markdown(f"• {outfit}")
        
        st.info("💡 **Pro Tip**: Mix and match these suggestions with accessories from your matched brands!")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Trend Analysis
    if show_trends:
        st.markdown("---")
        st.markdown("<div class='color-box'>", unsafe_allow_html=True)
        st.subheader("📈 Fashion Trend Analysis 2024-2025")
        
        trend_score, matched_trends = analyze_trend_alignment(colors)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Trend score gauge
            st.markdown("### Trend Score")
            st.markdown(f"<h1 style='text-align: center; color: #667eea; font-size: 4em;'>{trend_score}%</h1>", 
                       unsafe_allow_html=True)
            
            if trend_score >= 70:
                st.success("🔥 Ultra Trendy!")
            elif trend_score >= 40:
                st.info("✨ Moderately Trendy")
            else:
                st.warning("🎨 Timeless Classic")
        
        with col2:
            st.markdown("### Your Trend Alignment")
            if matched_trends:
                for trend in matched_trends:
                    st.markdown(f"✅ **{trend}**")
            else:
                st.write("Your palette is timeless and classic!")
            
            st.markdown("---")
            st.markdown("**2024-2025 Trending Colors:**")
            for trend in CURRENT_TRENDS['Color Trends'][:3]:
                st.markdown(f"• {trend}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Color Psychology
    st.markdown("---")
    st.markdown("<div class='color-box'>", unsafe_allow_html=True)
    st.subheader("🧠 Color Psychology Insights")
    
    dominant = get_color_name(colors[0])
    if dominant in COLOR_PSYCHOLOGY:
        psych = COLOR_PSYCHOLOGY[dominant]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🎭 Emotion**")
            st.info(psych['emotion'])
        
        with col2:
            st.markdown("**👔 Occasions**")
            st.info(psych['occasions'])
        
        with col3:
            st.markdown("**🍂 Season**")
            st.info(psych['season'])
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Download Section
    # Download Section
    st.markdown("---")
    st.markdown("<div class='color-box'>", unsafe_allow_html=True)
    st.subheader("💾 Export Complete Analysis")

    # Create comprehensive report
    report_data = {
        'Analysis Type': ['Dominant Color', 'Undertone', 'Season', 'Trend Score'],
        'Result': [
            dominant_color_name.title(),
            undertone if show_skin_tone else 'N/A',
            detected_season if show_season else 'N/A',
            f"{trend_score}%" if show_trends else 'N/A'
        ]
    }

    color_data = []
    for i, (color, percentage) in enumerate(zip(colors, percentages)):
        color_data.append({
            'Color': get_color_name(color).title(),
            'Hex': rgb_to_hex(color),
            'RGB': f"({int(color[0])}, {int(color[1])}, {int(color[2])})",
            'Percentage': f"{percentage:.1f}%"
        })

    df_report = pd.DataFrame(report_data)
    df_colors = pd.DataFrame(color_data)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Analysis Summary**")
        st.dataframe(df_report, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Color Details**")
        st.dataframe(df_colors, use_container_width=True, hide_index=True)

    # Download buttons
    col1, col2 = st.columns(2)

    with col1:
        csv_colors = df_colors.to_csv(index=False)
        st.download_button(
            label="📥 Download Colors (CSV)",
            data=csv_colors,
            file_name="fashion_color_analysis.csv",
            mime="text/csv"
        )

    with col2:
        full_report = f"""
FASHION COLOR INTELLIGENCE REPORT

DOMINANT COLOR: {dominant_color_name.title()}
UNDERTONE: {undertone if show_skin_tone else 'N/A'}
SEASON: {detected_season if show_season else 'N/A'}
TREND SCORE: {trend_score if show_trends else 'N/A'}%

COLOR PALETTE:
{df_colors.to_string()}

MATCHED BRANDS:
{', '.join(matched_brands) if show_brands and matched_brands else 'N/A'}

OUTFIT SUGGESTIONS:
{chr(10).join([f"- {outfit}" for outfit in outfit_suggestions['Casual']]) if show_outfits else 'N/A'}

Generated by AI Fashion Color Intelligence Platform
"""
        st.download_button(
            label="📄 Download Full Report",
            data=full_report,
            file_name="fashion_analysis_report.txt",
            mime="text/plain"
        )

    st.markdown("</div>", unsafe_allow_html=True)

else:
    # Landing page
    st.markdown("---")
    st.markdown("""
    <div style='background: white; padding: 40px; border-radius: 20px; text-align: center;'>
        <h2 style='color: #667eea;'>🚀 Welcome to AI Fashion Color Intelligence!</h2>
        <p style='font-size: 1.2em; color: #666;'>
            The most advanced fashion color analysis platform powered by AI
        </p>
    </div>
    """, unsafe_allow_html=True)


# Feature showcase
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='feature-card'>
        <h3>🌡️ Skin Tone Detection</h3>
        <p>Discover your undertone and get personalized color recommendations</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='feature-card'>
        <h3>🏷️ Brand Matching</h3>
        <p>See which luxury brands match your color palette</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='feature-card'>
        <h3>🍂 Season Analysis</h3>
        <p>Find out if you're a Spring, Summer, Autumn, or Winter</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='feature-card'>
        <h3>📈 Trend Analysis</h3>
        <p>See how your style aligns with 2024-2025 trends</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='feature-card'>
        <h3>👗 Outfit Generator</h3>
        <p>Get AI-powered outfit suggestions for any occasion</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='feature-card'>
        <h3>🎨 Color Intelligence</h3>
        <p>Extract dominant colors with psychology insights</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style='background: white; padding: 30px; border-radius: 15px; text-align: center;'>
    <h3 style='color: #667eea;'>✨ How It Works</h3>
    <p style='font-size: 1.1em;'>
        1️⃣ Upload any fashion image<br>
        2️⃣ AI extracts and analyzes colors<br>
        3️⃣ Get personalized recommendations<br>
        4️⃣ Download your complete analysis
    </p>
</div>
""", unsafe_allow_html=True)
#Footer
st.markdown("---")
st.markdown("""
<div style='background: white; padding: 20px; border-radius: 15px; text-align: center;'>
    <p style='color: #666;'>
        🎨 Powered by Machine Learning & Color Theory<br>
        Built with Python • Streamlit • scikit-learn<br>
        <strong>Portfolio Project by [Your Name]</strong>
    </p>
</div>
""", unsafe_allow_html=True)</parameter>
