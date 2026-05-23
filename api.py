# =============================================================================
# api.py — FastAPI Backend for Plant Disease Detection System
# =============================================================================
# Run with:  uvicorn api:app --host 127.0.0.1 --port 8000 --reload
# =============================================================================

import io
import os
import numpy as np
from PIL import Image

import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# 1.  Class Index → Disease Name Mapping  (38 classes)
# ---------------------------------------------------------------------------
CLASS_NAMES = {
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

# ---------------------------------------------------------------------------
# 2.  Locate and Load the saved model file safely
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
KERAS_PATH = os.path.join(BASE_DIR, "Plant_disease_model.keras")
H5_PATH    = os.path.join(BASE_DIR, "Plant_disease_model.h5")

model = None

# We try H5_PATH first because a legacy .h5 file renamed to .keras raises a ValueError
# in Keras 3 (which expects a zip file for .keras format).
paths_to_try = []
if os.path.exists(H5_PATH):
    paths_to_try.append(H5_PATH)
if os.path.exists(KERAS_PATH):
    paths_to_try.append(KERAS_PATH)

if not paths_to_try:
    raise FileNotFoundError(
        "No model file found. Place 'Plant_disease_model.keras' "
        "or 'Plant_disease_model.h5' in the same directory as api.py."
    )

for path in paths_to_try:
    try:
        print(f"[INFO] Attempting to load model from: {path}")
        model = tf.keras.models.load_model(path)
        print(f"[INFO] Model loaded successfully from: {path}")
        break
    except Exception as exc:
        print(f"[WARNING] Failed to load model from {path}: {exc}")

if model is None:
    raise RuntimeError("Failed to load the model from all available paths.")


# ---------------------------------------------------------------------------
# 4.  FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title       = "Plant Disease Detection API",
    description = "Accepts a plant leaf image and returns a predicted disease class.",
    version     = "1.0.0",
)

# Allow Streamlit (or any browser) to call the API without CORS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# ---------------------------------------------------------------------------
# 5.  Health-check endpoint
# ---------------------------------------------------------------------------
@app.get("/", summary="Health Check")
def root():
    """Returns a simple status message to confirm the API is running."""
    return {"status": "ok", "message": "Plant Disease Detection API is running."}


# ---------------------------------------------------------------------------
# 6.  Prediction endpoint
# ---------------------------------------------------------------------------
@app.post("/predict", summary="Predict Plant Disease")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an uploaded image (JPG / JPEG / PNG) and returns:
    - result_index : integer index of the predicted class (0-37)
    - class_name   : human-readable disease / healthy label
    - confidence   : prediction confidence as a percentage string
    - all_classes  : full index → class-name mapping (for reference)
    """

    # --- 6a. Validate content type ---
    if file.content_type not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(
            status_code = 415,
            detail      = "Unsupported file type. Please upload a JPG or PNG image.",
        )

    # --- 6b. Read uploaded bytes ---
    try:
        image_bytes = await file.read()
        image       = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code = 400,
            detail      = f"Could not open image: {exc}",
        )

    # --- 6c. Preprocess (matches the training pipeline exactly) ---
    image     = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)   # shape: (128,128,3)
    input_arr = np.array([input_arr])                               # shape: (1,128,128,3)

    # --- 6d. Run inference ---
    try:
        prediction   = model.predict(input_arr)                     # shape: (1, 38)
        result_index = int(np.argmax(prediction))
        confidence   = float(np.max(prediction)) * 100
    except Exception as exc:
        raise HTTPException(
            status_code = 500,
            detail      = f"Model inference failed: {exc}",
        )

    # --- 6e. Map to class name ---
    class_name = CLASS_NAMES.get(result_index, "Unknown")

    return JSONResponse(content={
        "result_index" : result_index,
        "class_name"   : class_name,
        "confidence"   : f"{confidence:.2f}%",
        "all_classes"  : CLASS_NAMES,
    })
