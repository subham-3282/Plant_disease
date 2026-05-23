# =============================================================================
# app.py — Plant Disease Detection System (Single-Process Streamlit App)
# =============================================================================
# Run with:  streamlit run app.py
# The TensorFlow/Keras model is loaded directly in-process — no separate
# backend server required.
# =============================================================================

import io
import os
import requests
from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf

# ---------------------------------------------------------------------------
# PAGE CONFIG  (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title = "🌿 Plant Disease Detection",
    page_icon  = "🌿",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ---------------------------------------------------------------------------
# GLOBAL CONSTANTS & MODEL LOADING
# ---------------------------------------------------------------------------

# Raw class names matching the model's output classes (used for get_advice)
RAW_CLASS_NAMES = {
    0:  "Apple___Apple_scab",
    1:  "Apple___Black_rot",
    2:  "Apple___Cedar_apple_rust",
    3:  "Apple___healthy",
    4:  "Blueberry___healthy",
    5:  "Cherry_(including_sour)___Powdery_mildew",
    6:  "Cherry_(including_sour)___healthy",
    7:  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    8:  "Corn_(maize)___Common_rust_",
    9:  "Corn_(maize)___Northern_Leaf_Blight",
    10: "Corn_(maize)___healthy",
    11: "Grape___Black_rot",
    12: "Grape___Esca_(Black_Measles)",
    13: "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    14: "Grape___healthy",
    15: "Orange___Haunglongbing_(Citrus_greening)",
    16: "Peach___Bacterial_spot",
    17: "Peach___healthy",
    18: "Pepper,_bell___Bacterial_spot",
    19: "Pepper,_bell___healthy",
    20: "Potato___Early_blight",
    21: "Potato___Late_blight",
    22: "Potato___healthy",
    23: "Raspberry___healthy",
    24: "Soybean___healthy",
    25: "Squash___Powdery_mildew",
    26: "Strawberry___Leaf_scorch",
    27: "Strawberry___healthy",
    28: "Tomato___Bacterial_spot",
    29: "Tomato___Early_blight",
    30: "Tomato___Late_blight",
    31: "Tomato___Leaf_Mold",
    32: "Tomato___Septoria_leaf_spot",
    33: "Tomato___Spider_mites Two-spotted_spider_mite",
    34: "Tomato___Target_Spot",
    35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    36: "Tomato___Tomato_mosaic_virus",
    37: "Tomato___healthy",
}

@st.cache_resource
def load_prediction_model():
    """Loads and caches the TensorFlow/Keras model."""
    import os
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
    KERAS_PATH = os.path.join(BASE_DIR, "Plant_disease_model.keras")
    H5_PATH    = os.path.join(BASE_DIR, "Plant_disease_model.h5")

    paths_to_try = []
    if os.path.exists(H5_PATH):
        paths_to_try.append(H5_PATH)
    if os.path.exists(KERAS_PATH):
        paths_to_try.append(KERAS_PATH)

    if not paths_to_try:
        return None, "No model file found. Place 'Plant_disease_model.keras' or 'Plant_disease_model.h5' in the same directory as app.py."

    errors = []
    for path in paths_to_try:
        try:
            model = tf.keras.models.load_model(path)
            return model, None
        except Exception as exc:
            errors.append(f"Failed to load from {os.path.basename(path)}: {str(exc)}")
    return None, " | ".join(errors)

# All 38 class names (mirrors the backend dict — used for the About page)
CLASS_NAMES = [
    "Apple — Apple Scab",                               # 0
    "Apple — Black Rot",                                # 1
    "Apple — Cedar Apple Rust",                         # 2
    "Apple — Healthy",                                  # 3
    "Blueberry — Healthy",                              # 4
    "Cherry — Powdery Mildew",                          # 5
    "Cherry — Healthy",                                 # 6
    "Corn (Maize) — Cercospora / Gray Leaf Spot",       # 7
    "Corn (Maize) — Common Rust",                       # 8
    "Corn (Maize) — Northern Leaf Blight",              # 9
    "Corn (Maize) — Healthy",                           # 10
    "Grape — Black Rot",                                # 11
    "Grape — Esca (Black Measles)",                     # 12
    "Grape — Leaf Blight (Isariopsis Leaf Spot)",       # 13
    "Grape — Healthy",                                  # 14
    "Orange — Huanglongbing (Citrus Greening)",         # 15
    "Peach — Bacterial Spot",                           # 16
    "Peach — Healthy",                                  # 17
    "Pepper (Bell) — Bacterial Spot",                   # 18
    "Pepper (Bell) — Healthy",                          # 19
    "Potato — Early Blight",                            # 20
    "Potato — Late Blight",                             # 21
    "Potato — Healthy",                                 # 22
    "Raspberry — Healthy",                              # 23
    "Soybean — Healthy",                                # 24
    "Squash — Powdery Mildew",                          # 25
    "Strawberry — Leaf Scorch",                         # 26
    "Strawberry — Healthy",                             # 27
    "Tomato — Bacterial Spot",                          # 28
    "Tomato — Early Blight",                            # 29
    "Tomato — Late Blight",                             # 30
    "Tomato — Leaf Mold",                               # 31
    "Tomato — Septoria Leaf Spot",                      # 32
    "Tomato — Spider Mites / Two-Spotted Spider Mite",  # 33
    "Tomato — Target Spot",                             # 34
    "Tomato — Yellow Leaf Curl Virus",                  # 35
    "Tomato — Mosaic Virus",                            # 36
    "Tomato — Healthy",                                 # 37
]

# ---------------------------------------------------------------------------
# CUSTOM CSS — dark theme + forest-green palette
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        /* ── Google Font ── */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* ── Root palette ── */
        :root {
            --green-dark   : #1B5E20;
            --green-primary: #2E7D32;
            --green-mid    : #388E3C;
            --green-light  : #66BB6A;
            --green-pale   : #A5D6A7;
            --bg-dark      : #0D1117;
            --bg-card      : #161B22;
            --bg-card2     : #1C2130;
            --text-primary : #E6EDF3;
            --text-muted   : #8B949E;
            --border       : #30363D;
            --accent-gold  : #F0B429;
        }

        /* ── Global resets ── */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif !important;
            background-color: var(--bg-dark) !important;
            color: var(--text-primary) !important;
        }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0D1B0E 0%, #0D1117 100%) !important;
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] .stRadio label {
            color: var(--text-primary) !important;
            font-weight: 500;
        }

        /* ── Main content area ── */
        .main .block-container {
            padding-top: 1.5rem;
            max-width: 1100px;
        }

        /* ── Hero banner ── */
        .hero-banner {
            background: linear-gradient(135deg, #1B5E20 0%, #2E7D32 50%, #1A237E 100%);
            border-radius: 16px;
            padding: 3rem 2.5rem;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 32px rgba(46,125,50,0.35);
        }
        .hero-banner h1 {
            font-size: 2.6rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 0.5rem;
        }
        .hero-banner p {
            font-size: 1.1rem;
            color: var(--green-pale);
            max-width: 640px;
            margin: 0 auto;
            line-height: 1.7;
        }

        /* ── Section headings ── */
        .section-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--green-light);
            border-left: 4px solid var(--green-primary);
            padding-left: 0.75rem;
            margin: 2rem 0 1rem;
        }

        /* ── Feature cards ── */
        .feature-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
            height: 100%;
        }
        .feature-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 24px rgba(46,125,50,0.25);
        }
        .feature-card .icon {
            font-size: 2.4rem;
            margin-bottom: 0.75rem;
        }
        .feature-card h3 {
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--green-light);
            margin-bottom: 0.5rem;
        }
        .feature-card p {
            font-size: 0.88rem;
            color: var(--text-muted);
            line-height: 1.6;
        }

        /* ── Image preview card ── */
        .image-card {
            background: var(--bg-card);
            border: 1px solid var(--green-primary);
            border-radius: 14px;
            padding: 1rem;
            box-shadow: 0 4px 20px rgba(46,125,50,0.2);
        }

        /* ── Result banner ── */
        .result-healthy {
            background: linear-gradient(135deg, #1B5E20, #2E7D32);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            border-left: 5px solid #69F0AE;
            margin-top: 1rem;
        }
        .result-disease {
            background: linear-gradient(135deg, #7B1818, #C62828);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            border-left: 5px solid #FF6B6B;
            margin-top: 1rem;
        }
        .result-label {
            font-size: 1.05rem;
            font-weight: 700;
            color: #FFFFFF;
        }
        .result-sub {
            font-size: 0.9rem;
            color: rgba(255,255,255,0.75);
            margin-top: 0.25rem;
        }

        /* ── Confidence badge ── */
        .confidence-badge {
            display: inline-block;
            background: rgba(255,255,255,0.15);
            border-radius: 20px;
            padding: 0.2rem 0.75rem;
            font-size: 0.82rem;
            font-weight: 600;
            color: #FFFFFF;
            margin-top: 0.5rem;
        }

        /* ── Advice box ── */
        .advice-box {
            background: var(--bg-card2);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            margin-top: 1rem;
        }
        .advice-box h4 {
            color: var(--green-light);
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.6rem;
        }
        .advice-box ul {
            margin: 0;
            padding-left: 1.2rem;
            color: var(--text-primary);
            font-size: 0.9rem;
            line-height: 1.8;
        }

        /* ── Stat cards (About page) ── */
        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
        }
        .stat-card .stat-number {
            font-size: 2rem;
            font-weight: 700;
            color: var(--green-light);
        }
        .stat-card .stat-label {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }

        /* ── Streamlit button override ── */
        .stButton > button {
            background-color: var(--green-primary) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            padding: 0.55rem 1.8rem !important;
            transition: background 0.2s, transform 0.15s !important;
        }
        .stButton > button:hover {
            background-color: var(--green-mid) !important;
            transform: translateY(-2px) !important;
        }

        /* ── Divider ── */
        hr { border-color: var(--border) !important; }

        /* ── File uploader & text input ── */
        [data-testid="stFileUploader"],
        .stTextInput > div > div > input {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
        }

        /* ── Radio options ── */
        .stRadio > div { gap: 0.5rem; }

        /* ── Expander ── */
        .streamlit-expanderHeader {
            background: var(--bg-card) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg-dark); }
        ::-webkit-scrollbar-thumb { background: var(--green-dark); border-radius: 3px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# HELPER — fetch image bytes from a public URL
# ---------------------------------------------------------------------------
def fetch_image_from_url(url: str):
    """Downloads an image from a public URL and returns raw bytes + PIL Image."""
    try:
        # Handle Google Drive share links → convert to direct download
        if "drive.google.com" in url and "/file/d/" in url:
            file_id = url.split("/file/d/")[1].split("/")[0]
            url = f"https://drive.google.com/uc?export=download&id={file_id}"

        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        image_bytes = resp.content
        pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image_bytes, pil_image
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# HELPER — run model prediction locally
# ---------------------------------------------------------------------------
def call_predict_api(image_bytes: bytes, filename: str = "image.jpg"):
    """Predicts the disease class directly in the Streamlit app using the loaded model."""
    try:
        model, err = load_prediction_model()
        if err:
            return None, f"❌ Model Loading Error: {err}"
        if model is None:
            return None, "❌ Model is not loaded. Place the model file in the directory."

        # Preprocess: resize to 128×128 and normalise to match the training pipeline
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)   # shape: (128,128,3)
        input_arr = np.array([input_arr])                               # shape: (1,128,128,3)

        # Inference
        prediction = model.predict(input_arr)
        result_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        class_name = RAW_CLASS_NAMES.get(result_index, "Unknown")

        return {
            "result_index": result_index,
            "class_name": class_name,
            "confidence": f"{confidence:.2f}%",
        }, None
    except Exception as exc:
        return None, f"❌ Unexpected error during local inference: {exc}"


# ---------------------------------------------------------------------------
# HELPER — generate advice based on prediction
# ---------------------------------------------------------------------------
def get_advice(class_name: str) -> dict:
    """
    Returns a dict with 'status', 'title', and 'tips' list of recommendations
    tailored to the specific plant disease or healthy status.
    """
    name_lower = class_name.lower()
    is_healthy = "healthy" in name_lower

    if is_healthy:
        # Custom healthy crop tips
        crop = "plant"
        for c in ["apple", "blueberry", "cherry", "corn", "grape", "orange", "peach", "pepper", "potato", "raspberry", "soybean", "squash", "strawberry", "tomato"]:
            if c in name_lower:
                crop = c.capitalize()
                break

        return {
            "status": "healthy",
            "title" : f"✅ Excellent! Your {crop} appears to be healthy!",
            "tips"  : [
                "💧 Maintain consistent soil moisture — avoid waterlogged soils as well as severe drying out.",
                "☀️ Ensure optimal sunlight and spacing so the canopy gets ample daylight and fresh airflow.",
                "🌱 Feed the plant with appropriate balanced organic fertilizer/compost based on its growth stage.",
                "🔍 Do a weekly visual inspection under leaves and near stems to spot early signs of insects or spots.",
                "🛡️ As a preventative shield, you can apply organic neem oil spray once every 2–3 weeks.",
            ],
        }

    # ── 1. APPLE DISEASES ──
    if "apple___apple_scab" in name_lower:
        return {
            "status": "disease",
            "title" : "🍎 Apple Scab (Venturia inaequalis) Detected!",
            "tips"  : [
                "💧 Switch to drip irrigation to keep the foliage completely dry.",
                "🍂 Rake and destroy all fallen leaves in autumn to prevent overwintering fungal spores.",
                "✂️ Prune the tree canopy during winter to maximize air circulation and sunlight penetration.",
                "🧪 Apply preventative sulfur-based or copper fungicides at green tip, tight cluster, and petal fall stages."
            ]
        }
    elif "apple___black_rot" in name_lower:
        return {
            "status": "disease",
            "title" : "🍎 Apple Black Rot (Botryosphaeria obtusa) Detected!",
            "tips"  : [
                "✂️ Prune out dead wood, mummified fruit, and cankers during winter dormancy.",
                "🧼 Sterilize pruning tools between cuts using 70% isopropyl alcohol to avoid spreading.",
                "🗑️ Remove and burn or bury all infected debris from the orchard floor.",
                "🧪 Apply protective fungicides starting at early bud break and continue through bloom."
            ]
        }
    elif "apple___cedar_apple_rust" in name_lower:
        return {
            "status": "disease",
            "title" : "🍎 Cedar Apple Rust (Gymnosporangium) Detected!",
            "tips"  : [
                "🌲 Keep apple trees isolated from ornamental junipers or red cedars (which host the fungus).",
                "🍂 Clear fallen apple leaves to maintain excellent orchard hygiene.",
                "🧪 Apply preventative fungicides like Myclobutanil early in the spring at bud break.",
                "🌱 Plant rust-resistant apple cultivars like 'Liberty', 'Freedom', or 'Enterprise' in future plantings."
            ]
        }

    # ── 2. CHERRY DISEASES ──
    elif "cherry" in name_lower and "powdery_mildew" in name_lower:
        return {
            "status": "disease",
            "title" : "🍒 Cherry Powdery Mildew Detected!",
            "tips"  : [
                "☀️ Maximize sunlight exposure and spacing between cherry trees to keep humidity low.",
                "🌬️ Prune branches to ensure good air movement inside the leaf canopy.",
                "🧪 Apply sulfur-based or potassium bicarbonate fungicides as soon as white spots appear.",
                "💧 Avoid overhead sprinklers; irrigate at the ground level to keep leaves dry."
            ]
        }

    # ── 3. CORN DISEASES ──
    elif "corn" in name_lower and "cercospora" in name_lower:
        return {
            "status": "disease",
            "title" : "🌽 Corn Gray Leaf Spot (Cercospora zeae-maydis) Detected!",
            "tips"  : [
                "🔄 Rotate crops with non-host plants (like soybeans) for 1–2 years to break the pathogen cycle.",
                "🚜 Practice clean tillage to bury old infected corn stalks and debris.",
                "🌱 Choose hybrid corn varieties that show strong resistance or tolerance to Cercospora.",
                "🧪 Apply a foliar fungicide if the disease pressure is high before the tasseling stage."
            ]
        }
    elif "corn" in name_lower and "rust" in name_lower:
        return {
            "status": "disease",
            "title" : "🌽 Corn Common Rust (Puccinia sorghi) Detected!",
            "tips"  : [
                "🌱 Plant rust-resistant hybrids; this is by far the most effective control method.",
                "☀️ Ensure proper spacing to speed up canopy drying after rains or heavy dew.",
                "🚜 Till corn residue under the soil after harvest to decompose overwintering spores.",
                "🧪 Foliar fungicides are rarely economical for field corn, but recommended for sweet corn."
            ]
        }
    elif "corn" in name_lower and "blight" in name_lower:
        return {
            "status": "disease",
            "title" : "🌽 Northern Corn Leaf Blight Detected!",
            "tips"  : [
                "🔄 Rotate crops to significantly reduce fungal spore survival in the soil.",
                "🚜 Plow or shred crop residues after harvest to accelerate decomposition.",
                "🌱 Always select resistant corn hybrids with high disease-tolerance ratings.",
                "🧪 Apply protective fungicides early if lesions are spotted on lower leaves."
            ]
        }

    # ── 4. GRAPE DISEASES ──
    elif "grape___black_rot" in name_lower:
        return {
            "status": "disease",
            "title" : "🍇 Grape Black Rot (Guignardia bidwellii) Detected!",
            "tips"  : [
                "🍇 Remove and burn all 'mummies' (shriveled infected grapes) from the vines and ground.",
                "✂️ Keep grapevines well-pruned and trellised to maximize air flow and sun exposure.",
                "🧪 Apply protective fungicides starting at early bloom and continue until fruit sizing.",
                "💧 Irrigate in the early morning so leaves do not remain wet overnight."
            ]
        }
    elif "grape___esca" in name_lower:
        return {
            "status": "disease",
            "title" : "🍇 Esca / Black Measles (Trunk Disease) Detected!",
            "tips"  : [
                "✂️ Do not prune grapevines during wet, humid weather when fungal spores are highly active.",
                "🩹 Protect pruning wounds immediately by painting them with wound sealants or protective paste.",
                "🧪 Spray biological protection agents containing *Trichoderma* on fresh pruning cuts.",
                "🪓 Uproot and burn severely infected vines to save the rest of your vineyard."
            ]
        }
    elif "grape___leaf_blight" in name_lower:
        return {
            "status": "disease",
            "title" : "🍇 Grape Leaf Blight (Isariopsis) Detected!",
            "tips"  : [
                "🍂 Collect and destroy fallen leaves at the end of the season to remove spore sources.",
                "✂️ Prune lower leaves to improve aeration around the base of the vine.",
                "🧪 Apply copper-based fungicides or Bordeaux mixture post-harvest to protect the vines.",
                "🌱 Maintain balanced soil nutrition to support the grapevine's immune health."
            ]
        }

    # ── 5. ORANGE / CITRUS ──
    elif "orange___haunglongbing" in name_lower or "greening" in name_lower:
        return {
            "status": "disease",
            "title" : "🍊 Citrus Greening / Huanglongbing (HLB) Detected!",
            "tips"  : [
                "🚨 Highly Destructive: Control the Asian Citrus Psyllid vector using systemic insecticides.",
                "🪓 Immediately remove and burn infected trees to stop the spread to nearby healthy citrus.",
                "🌱 Source only certified, pathogen-free nursery stock for all replacement plantings.",
                "⚡ Provide foliar nutrients (zinc, manganese, iron) to prolong the productivity of mildly affected trees."
            ]
        }

    # ── 6. PEACH DISEASES ──
    elif "peach___bacterial_spot" in name_lower:
        return {
            "status": "disease",
            "title" : "🍑 Peach Bacterial Spot (Xanthomonas) Detected!",
            "tips"  : [
                "🧪 Apply copper-based bactericides during fall leaf drop and winter dormancy.",
                "💧 Avoid overhead watering; bacteria spread instantly through splashing water.",
                "🌱 Plant bacterial-spot resistant peach cultivars suitable for humid climates.",
                "🌬️ Avoid excessive nitrogen fertilization, which produces weak, highly susceptible leaves."
            ]
        }

    # ── 7. PEPPER DISEASES ──
    elif "pepper" in name_lower and "bacterial_spot" in name_lower:
        return {
            "status": "disease",
            "title" : "🫑 Pepper Bacterial Spot Detected!",
            "tips"  : [
                "🌱 Use only certified disease-free seeds and robust nursery transplants.",
                "🔄 Rotate pepper crops with non-solanaceous plants for a minimum of 2–3 years.",
                "🧪 Apply copper-based bactericides at the first sign of symptoms on foliage.",
                "🧤 Avoid working in the pepper beds when leaves are wet to prevent manual transmission."
            ]
        }

    # ── 8. POTATO DISEASES ──
    elif "potato___early_blight" in name_lower:
        return {
            "status": "disease",
            "title" : "🥔 Potato Early Blight (Alternaria solani) Detected!",
            "tips"  : [
                "🌱 Maintain high soil fertility (nitrogen & potassium) to prevent plant stress.",
                "🧪 Apply preventative protectant fungicides (like Chlorothalonil) before canopy closure.",
                "🔄 Rotate crops for at least 3 years away from potatoes, tomatoes, and eggplants.",
                "💧 Drip irrigate or water early in the morning to allow rapid leaf drying."
            ]
        }
    elif "potato___late_blight" in name_lower:
        return {
            "status": "disease",
            "title" : "🥔 Potato Late Blight (Phytophthora infestans) Detected!",
            "tips"  : [
                "🚨 High Threat! Remove and bag infected potato plants immediately. Do not compost.",
                "🌱 Plant only certified, disease-free seed tubers.",
                "🧪 Apply systemic fungicides under local agricultural extension guidelines.",
                "🍂 Wait at least 10–14 days after vines die before harvesting tubers to avoid contamination."
            ]
        }

    # ── 9. SQUASH DISEASES ──
    elif "squash" in name_lower and "powdery_mildew" in name_lower:
        return {
            "status": "disease",
            "title" : "🎃 Squash Powdery Mildew Detected!",
            "tips"  : [
                "☀️ Plant squash in full sun and space plants widely to reduce stagnant humidity.",
                "🧪 Spray preventative sulfur, potassium bicarbonate, or neem oil at the first sign of mildew.",
                "💧 Water at the soil level; dry foliage prevents powdery mildew spores from germinating.",
                "🌱 Select powdery mildew-tolerant squash varieties for next season."
            ]
        }

    # ── 10. STRAWBERRY DISEASES ──
    elif "strawberry" in name_lower and "scorch" in name_lower:
        return {
            "status": "disease",
            "title" : "🍓 Strawberry Leaf Scorch Detected!",
            "tips"  : [
                "✂️ Mince or remove old strawberry leaves after the final harvest during spring renovation.",
                "🌬️ Keep the strawberry beds thinned to maintain clean airflow and fast drying.",
                "💧 Always irrigate using drip lines to avoid wetting leaves.",
                "🧪 Use copper-based fungicides in the early spring if leaf scorch is a persistent problem."
            ]
        }

    # ── 11. TOMATO DISEASES ──
    elif "tomato___bacterial_spot" in name_lower:
        return {
            "status": "disease",
            "title" : "🍅 Tomato Bacterial Spot Detected!",
            "tips"  : [
                "🔄 Rotate tomato plantings with non-solanaceous plants for a full 2-year cycle.",
                "🧪 Apply a mix of copper and Mancozeb fungicides to control bacterial populations.",
                "💧 Always water at the base of the plant; splashing water transmits bacteria rapidly.",
                "🧼 Wash hands and sterilize all gardening tools after handling infected tomato plants."
            ]
        }
    elif "tomato___early_blight" in name_lower:
        return {
            "status": "disease",
            "title" : "🍅 Tomato Early Blight (Alternaria solani) Detected!",
            "tips"  : [
                "✂️ Prune off the lower 12 inches of leaves to prevent soil-borne spores from splashing up.",
                "🍂 Apply a thick layer of organic mulch around tomato plants to create a soil barrier.",
                "🧪 Spray preventative copper fungicides or bio-fungicides (*Bacillus subtilis*) regularly.",
                "🔄 Rotate crops away from tomatoes, peppers, and potatoes for 2–3 seasons."
            ]
        }
    elif "tomato___late_blight" in name_lower:
        return {
            "status": "disease",
            "title" : "🍅 Tomato Late Blight (Phytophthora infestans) Detected!",
            "tips"  : [
                "🚨 Highly Infectious! Pull, bag, and discard infected plants immediately to prevent airborne spread.",
                "🧪 Apply protectant fungicides (Chlorothalonil or copper) during cool, wet, humid weather.",
                "💨 Ensure wide spacing between tomato plants for optimal air circulation.",
                "🔍 Monitor tomato leaves daily during wet summer periods for water-soaked lesions."
            ]
        }
    elif "tomato___leaf_mold" in name_lower:
        return {
            "status": "disease",
            "title" : "🍅 Tomato Leaf Mold (Passalora fulva) Detected!",
            "tips"  : [
                "🌬️ Maximize greenhouse ventilation and keep relative humidity strictly below 85%.",
                "✂️ Prune lower suckers and foliage to facilitate continuous air movement.",
                "💧 Drip irrigate and avoid watering tomatoes in the evening.",
                "🌱 Plant leaf mold-resistant tomato hybrids in enclosed high tunnels or greenhouses."
            ]
        }
    elif "tomato___septoria_leaf_spot" in name_lower:
        return {
            "status": "disease",
            "title" : "🍅 Tomato Septoria Leaf Spot Detected!",
            "tips"  : [
                "🍂 Lay down mulch beneath tomatoes to keep fungal spores in the soil from splashing up.",
                "✂️ Prune lower branches once the tomato plant is fully established.",
                "💧 Water only at the base via drip lines; never wet the leaves.",
                "🧪 Spray copper-based fungicides when symptoms first appear on the lowest leaves."
            ]
        }
    elif "tomato___spider_mites" in name_lower:
        return {
            "status": "disease",
            "title" : "🍅 Tomato Spider Mites Infestation Detected!",
            "tips"  : [
                "💧 Spray the undersides of leaves with a strong water stream to physically dislodge mites.",
                "🐞 Introduce natural predatory mites (*Phytoseiulus persimilis*) as biological control.",
                "🧪 Apply insecticidal soap, neem oil, or horticultural oils under leaves to suffocate mites.",
                "🌬️ Mist plants to increase local humidity (spider mites thrive in dusty, hot, dry conditions)."
            ]
        }
    elif "tomato___target_spot" in name_lower:
        return {
            "status": "disease",
            "title" : "🍅 Tomato Target Spot (Corynespora cassiicola) Detected!",
            "tips"  : [
                "✂️ Thin and prune leaves to ensure adequate light and air pass through the canopy.",
                "🧪 Apply chlorothalonil or azoxystrobin fungicides at the first sign of concentric spots.",
                "🍂 Clean up and burn or dispose of all plant residues immediately after the final harvest.",
                "🔄 Practice crop rotation with non-host species for a full season."
            ]
        }
    elif "tomato___tomato_yellow_leaf_curl_virus" in name_lower:
        return {
            "status": "disease",
            "title" : "🍅 Tomato Yellow Leaf Curl Virus (TYLCV) Detected!",
            "tips"  : [
                "🐞 Whitefly Vector: Install yellow sticky traps to catch and monitor whitefly populations.",
                "🧪 Spray neem oil, horticultural oil, or systemic insecticidal soap to control whiteflies.",
                "🪓 Immediately uproot and destroy infected plants (there is no cure for the viral infection).",
                "🌱 Apply reflective silver or metallic mulches to repel whiteflies from landing on plants."
            ]
        }
    elif "tomato___tomato_mosaic_virus" in name_lower:
        return {
            "status": "disease",
            "title" : "🍅 Tomato Mosaic Virus (ToMV) Detected!",
            "tips"  : [
                "🚨 Super Infectious: Immediately pull up and discard infected tomato plants. Do not compost.",
                "🧤 Sanitise hands and wash garden tools in milk or disinfectant before touching other tomatoes.",
                "🌱 Plant only certified disease-free, ToMV-resistant tomato seeds or transplants.",
                "🚭 Do not smoke or handle tobacco products near tomato plants (the virus can spread from tobacco)."
            ]
        }

    # ── 12. FALLBACK FOR OTHER DISEASES ──
    else:
        return {
            "status": "disease",
            "title" : f"⚠️ Plant Disease Detected ({class_name})!",
            "tips"  : [
                "🚫 Isolate the affected plant immediately to prevent spreading the disease.",
                "✂️ Prune and safely discard heavily spotted/wilted foliage (do NOT compost them).",
                "💧 Avoid overhead watering; switch to ground watering to keep foliage dry.",
                "🧤 Sterilize pruning tools with 70% isopropyl alcohol after working with infected plants.",
                "🧪 Consult with a local agricultural extension specialist for a targeted treatment plan."
            ]
        }



# ---------------------------------------------------------------------------
# PAGE 1 — HOME
# ---------------------------------------------------------------------------
def page_home():
    # ── Hero banner ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-banner">
            <h1>🌿 AI Plant Pathologist</h1>
            <p>
                Harness the power of deep learning to instantly detect diseases across
                <strong>14 crop species</strong> and <strong>38 disease categories</strong>.
                Upload a photo of any leaf and receive an accurate diagnosis in seconds.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── How it works ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="icon">📸</div>
                <h3>1 — Upload a Leaf Photo</h3>
                <p>
                    Take a clear, well-lit photo of the plant leaf.
                    Upload it directly from your device or paste a public image URL.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="icon">🧠</div>
                <h3>2 — AI Analysis</h3>
                <p>
                    Our CNN model — trained on 87,000+ plant images —
                    analyses visual patterns to classify the disease with high accuracy.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="icon">💊</div>
                <h3>3 — Get Actionable Advice</h3>
                <p>
                    Receive the disease name plus practical guidance: treatment steps,
                    isolation tips, and organic prevention strategies.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Why it matters ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Why Early Detection Matters</div>', unsafe_allow_html=True)

    st.markdown(
        """
        > 🌾 **~40% of global crop yield** is lost annually to plant diseases and pests
        > *(FAO, 2023)*. Early and accurate identification allows farmers to act before
        > an outbreak becomes catastrophic — protecting both livelihoods and food security.

        This tool brings **laboratory-grade diagnostic capability** to the fingertips of
        smallholder farmers, agronomists, and researchers worldwide — completely free of charge.
        """,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Supported crops quick-list ────────────────────────────────────────────
    st.markdown('<div class="section-title">Supported Crops</div>', unsafe_allow_html=True)

    crops = [
        ("🍎", "Apple"),   ("🫐", "Blueberry"), ("🍒", "Cherry"),
        ("🌽", "Corn"),    ("🍇", "Grape"),      ("🍊", "Orange"),
        ("🍑", "Peach"),   ("🫑", "Pepper"),     ("🥔", "Potato"),
        ("🍓", "Raspberry/Strawberry"), ("🫘", "Soybean"),  ("🎃", "Squash"),
        ("🍅", "Tomato"),  ("🍃", "14 species total", ),
    ]

    crop_cols = st.columns(7)
    for i, crop in enumerate(crops[:14]):
        with crop_cols[i % 7]:
            emoji = crop[0]
            name  = crop[1]
            st.markdown(
                f"""
                <div style="
                    background:#161B22;border:1px solid #30363D;
                    border-radius:10px;padding:0.6rem 0.4rem;
                    text-align:center;margin-bottom:0.5rem;">
                    <span style="font-size:1.6rem;">{emoji}</span><br>
                    <span style="font-size:0.78rem;color:#8B949E;">{name}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# PAGE 2 — DISEASE RECOGNITION
# ---------------------------------------------------------------------------
def page_recognition():
    st.markdown('<h2 style="color:#66BB6A;">🔍 Disease Recognition</h2>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8B949E;'>Choose how to provide your plant image, then click <strong>Analyse Image</strong>.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Source selector ───────────────────────────────────────────────────────
    source = st.radio(
        "📂 Image Source",
        options=["📁  Browse Local Files", "🌐  Google Drive / Web URL"],
        horizontal=True,
    )

    image_bytes  = None   # raw bytes to send to the API
    pil_image    = None   # PIL Image for display
    source_name  = "image.jpg"

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Branch 1: local file upload ───────────────────────────────────────────
    if source.startswith("📁"):
        uploaded = st.file_uploader(
            "Drop your leaf image here",
            type    = ["jpg", "jpeg", "png"],
            help    = "Supported formats: JPG, JPEG, PNG",
        )
        if uploaded is not None:
            image_bytes = uploaded.read()
            pil_image   = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            source_name = uploaded.name

    # ── Branch 2: URL input ───────────────────────────────────────────────────
    else:
        url = st.text_input(
            "🔗 Paste a public image URL or Google Drive share link",
            placeholder="https://example.com/leaf.jpg  or  https://drive.google.com/file/d/.../view",
        )
        if url.strip():
            with st.spinner("Fetching image from URL …"):
                image_bytes, result = fetch_image_from_url(url.strip())
            if image_bytes is None:
                st.error(f"Could not fetch image: {result}")
            else:
                pil_image = result  # fetch_image_from_url returns PIL on success

    # ── Image preview ─────────────────────────────────────────────────────────
    if pil_image is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        col_preview, col_info = st.columns([1, 1])

        with col_preview:
            st.markdown('<div class="image-card">', unsafe_allow_html=True)
            st.image(pil_image, caption="📷 Loaded Image", width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_info:
            w, h = pil_image.size
            st.markdown(
                f"""
                <div style="padding:1rem 0;">
                    <p style="color:#8B949E;font-size:0.9rem;">
                        <strong style="color:#66BB6A;">File:</strong> {source_name}<br>
                        <strong style="color:#66BB6A;">Dimensions:</strong> {w} × {h} px<br>
                        <strong style="color:#66BB6A;">Mode:</strong> {pil_image.mode}<br>
                        <strong style="color:#66BB6A;">Size:</strong> {len(image_bytes) / 1024:.1f} KB
                    </p>
                    <p style="color:#8B949E;font-size:0.85rem;margin-top:1rem;">
                        ℹ️ The image will be resized to <strong>128 × 128 px</strong>
                        before being sent to the AI model.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Analyse button ─────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            analyse = st.button("🧠 Analyse Image", width="stretch")

        # ── Run inference ──────────────────────────────────────────────────
        if analyse:
            with st.spinner("🔬 Sending image to AI model … please wait"):
                result, error = call_predict_api(image_bytes, source_name)

            if error:
                st.error(error)
            else:
                class_name   = result.get("class_name",   "Unknown")
                confidence   = result.get("confidence",   "N/A")
                result_index = result.get("result_index", -1)
                advice       = get_advice(class_name)

                st.markdown("---")
                st.markdown('<div class="section-title">📋 Diagnosis Result</div>', unsafe_allow_html=True)

                res_col, adv_col = st.columns([1, 1])

                with res_col:
                    # Result banner
                    css_class = "result-healthy" if advice["status"] == "healthy" else "result-disease"
                    st.markdown(
                        f"""
                        <div class="{css_class}">
                            <div class="result-label">{advice['title']}</div>
                            <div class="result-sub">
                                🌿 Detected: <strong>{class_name.replace('___', ' → ').replace('_', ' ')}</strong>
                            </div>
                            <span class="confidence-badge">🎯 Confidence: {confidence}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Raw class info
                    st.markdown(
                        f"""
                        <div style="background:#161B22;border:1px solid #30363D;
                            border-radius:10px;padding:0.9rem 1rem;font-size:0.85rem;color:#8B949E;">
                            🔖 <strong style="color:#E6EDF3;">Class Index:</strong> {result_index} / 37<br>
                            📛 <strong style="color:#E6EDF3;">Raw Label:</strong> {class_name}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with adv_col:
                    # Advice box
                    tips_html = "".join(f"<li>{t}</li>" for t in advice["tips"])
                    st.markdown(
                        f"""
                        <div class="advice-box">
                            <h4>💡 Recommended Actions</h4>
                            <ul>{tips_html}</ul>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    else:
        # Placeholder when no image is loaded yet
        st.markdown(
            """
            <div style="
                background:#161B22;border:2px dashed #30363D;
                border-radius:14px;padding:3rem;text-align:center;margin-top:1rem;">
                <span style="font-size:3rem;">🍃</span>
                <p style="color:#8B949E;margin-top:1rem;font-size:1rem;">
                    No image loaded yet.<br>
                    Use one of the input methods above to get started.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# PAGE 3 — ABOUT & ANALYTICS
# ---------------------------------------------------------------------------
def page_about():
    st.markdown('<h2 style="color:#66BB6A;">📊 About & Analytics</h2>', unsafe_allow_html=True)
    st.markdown("<p style='color:#8B949E;'>Dataset scope, model architecture, and full class catalogue.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Stat cards ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Dataset at a Glance</div>', unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)
    stats = [
        ("87,000+", "Training Images"),
        ("38",      "Disease Classes"),
        ("14",      "Crop Species"),
        ("128×128", "Input Resolution"),
    ]
    for col, (number, label) in zip([s1, s2, s3, s4], stats):
        with col:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-number">{number}</div>
                    <div class="stat-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model Architecture ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🧠 Model Architecture</div>', unsafe_allow_html=True)

    st.markdown(
        """
        | Property | Detail |
        |---|---|
        | **Framework** | TensorFlow / Keras |
        | **Architecture** | Convolutional Neural Network (CNN) |
        | **Input Shape** | (128, 128, 3) — RGB |
        | **Output** | Softmax over 38 classes |
        | **Training Dataset** | PlantVillage (augmented) |
        | **Deployment** | Single-process Streamlit app (in-process inference) |
        | **Inference Latency** | < 200 ms per image (CPU) |
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Class catalogue ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🌿 Full Class Catalogue (38 Categories)</div>', unsafe_allow_html=True)

    # Group by crop
    crop_groups = {
        "🍎 Apple"       : [0, 1, 2, 3],
        "🫐 Blueberry"   : [4],
        "🍒 Cherry"      : [5, 6],
        "🌽 Corn (Maize)": [7, 8, 9, 10],
        "🍇 Grape"       : [11, 12, 13, 14],
        "🍊 Orange"      : [15],
        "🍑 Peach"       : [16, 17],
        "🫑 Pepper (Bell)": [18, 19],
        "🥔 Potato"      : [20, 21, 22],
        "🍓 Raspberry"   : [23],
        "🫘 Soybean"     : [24],
        "🎃 Squash"      : [25],
        "🍓 Strawberry"  : [26, 27],
        "🍅 Tomato"      : [28, 29, 30, 31, 32, 33, 34, 35, 36, 37],
    }

    for crop, indices in crop_groups.items():
        with st.expander(f"{crop}  —  {len(indices)} class(es)", expanded=False):
            for idx in indices:
                healthy_tag = (
                    '<span style="background:#1B5E20;color:#A5D6A7;'
                    'border-radius:4px;padding:1px 7px;font-size:0.75rem;margin-left:8px;">Healthy</span>'
                    if "healthy" in CLASS_NAMES[idx].lower()
                    else '<span style="background:#7B1818;color:#FFCDD2;'
                    'border-radius:4px;padding:1px 7px;font-size:0.75rem;margin-left:8px;">Disease</span>'
                )
                st.markdown(
                    f"""
                    <div style="padding:0.35rem 0.1rem;border-bottom:1px solid #30363D;">
                        <code style="color:#8B949E;font-size:0.78rem;">#{idx:02d}</code>&nbsp;
                        <span style="color:#E6EDF3;font-size:0.9rem;">{CLASS_NAMES[idx]}</span>
                        {healthy_tag}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Real-world impact ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🌍 Real-World Impact</div>', unsafe_allow_html=True)

    st.markdown(
        """
        - 🌾 **Food Security** — Rapid diagnosis helps smallholder farmers in developing
          nations act before a disease outbreak wipes out an entire season's yield.
        - 💰 **Cost Reduction** — Early intervention with targeted treatments dramatically
          reduces the volume of pesticide/fungicide applied, cutting costs and environmental impact.
        - 📡 **Offline Potential** — The lightweight CNN model (< 100 MB) can be deployed
          on edge devices for use in areas with poor internet connectivity.
        - 🔬 **Research Accelerator** — Agronomists can batch-process experimental plots
          to identify resistant cultivars faster than manual inspection.
        - 🌱 **Sustainable Farming** — Reduces over-reliance on blanket chemical sprays,
          supporting eco-friendly and organic farming practices.
        """
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align:center;color:#30363D;font-size:0.8rem;padding:1rem 0;">
            🌿 Plant Disease Detection System &nbsp;·&nbsp; Powered by TensorFlow + Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
def sidebar_nav():
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:1rem 0 0.5rem;">
                <span style="font-size:2.5rem;">🌿</span><br>
                <span style="font-size:1.1rem;font-weight:700;color:#66BB6A;">Plant Pathologist</span><br>
                <span style="font-size:0.75rem;color:#8B949E;">AI-Powered Disease Detection</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        page = st.radio(
            "Navigate",
            options=["🌿  Home", "🔍  Disease Recognition", "📊  About & Analytics"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size:0.78rem;color:#8B949E;padding:0.5rem 0;">
                <strong style="color:#66BB6A;">Model Status</strong><br>
                Model runs directly inside the Streamlit app.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Model health check
        try:
            model, err = load_prediction_model()
            if err:
                st.markdown(
                    f'<div style="color:#FF6B6B;font-size:0.82rem;">🔴 Load Error: {err[:30]}...</div>',
                    unsafe_allow_html=True,
                )
            elif model is not None:
                st.markdown(
                    '<div style="color:#69F0AE;font-size:0.82rem;">🟢 Model Ready</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div style="color:#FF6B6B;font-size:0.82rem;">🔴 Model Not Found</div>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.markdown(
                f'<div style="color:#FF6B6B;font-size:0.82rem;">🔴 Error: {str(e)[:30]}...</div>',
                unsafe_allow_html=True,
            )

    return page


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    inject_css()
    page = sidebar_nav()

    if page.startswith("🌿"):
        page_home()
    elif page.startswith("🔍"):
        page_recognition()
    elif page.startswith("📊"):
        page_about()


if __name__ == "__main__":
    main()