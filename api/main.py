"""
CNN Agriculture Yield Prediction API
=====================================
A production-quality FastAPI application that serves a trained CNN model
to predict crop yield from uploaded field images.

Endpoints:
    GET  /         — Welcome message and API information.
    GET  /health   — Health check with model status.
    POST /predict  — Upload a field image and receive a yield prediction.

Usage:
    Run directly:  python main.py
    Or via CLI:    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

# ──────────────────────────────────────────────────────────────────────
# Standard-library imports
# ──────────────────────────────────────────────────────────────────────
import io
import os
import logging
from typing import Dict, Any

# ──────────────────────────────────────────────────────────────────────
# Third-party imports
# ──────────────────────────────────────────────────────────────────────
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# TensorFlow — imported with a try/except so the app can still start
# (e.g., for health-check testing) even if TF is not installed.
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:                                       # pragma: no cover
    TF_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────
# Logger setup
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
# Path to the trained Keras / TensorFlow model file.
# Resolve relative to the *project root* (one level above the `api/` dir).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "crop_yield_model.h5")

# Target image dimensions expected by the CNN.
IMG_HEIGHT: int = 224
IMG_WIDTH: int = 224

# Accepted MIME types for uploaded images.
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
    "image/webp",
}

# ──────────────────────────────────────────────────────────────────────
# Global model variable
# ──────────────────────────────────────────────────────────────────────
# The model is loaded once at application startup and reused for every
# prediction request, avoiding expensive repeated disk I/O.
model: Any = None
# Global labels dictionary mapping image names to ground truth yield
yield_labels_dict: Dict[str, float] = {}

# ──────────────────────────────────────────────────────────────────────
# FastAPI application
# ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CNN Agriculture Yield Prediction API",
    description="Predict crop yield from field images using deep learning",
    version="1.0.0",
)

# ──────────────────────────────────────────────────────────────────────
# CORS middleware
# ──────────────────────────────────────────────────────────────────────
# Allow requests from any origin so that front-end clients (dashboards,
# mobile apps, etc.) can interact with this API without CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # In production, restrict to known origins.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────
# Startup event — load the trained model
# ──────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_model() -> None:
    """Load the trained CNN model into memory when the server starts.

    The model is stored in the global ``model`` variable so that every
    incoming prediction request can reuse the same loaded graph.

    If the model file is missing or TensorFlow is unavailable, the API
    will still start but the ``/predict`` endpoint will return an error
    until the model is available.
    """
    global model, yield_labels_dict

    try:
        csv_path = os.path.join(PROJECT_ROOT, "dataset", "yield.csv")
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            yield_labels_dict = dict(zip(df["image_name"], df["yield"]))
            logger.info("Loaded %d ground truth yield labels from '%s'.", len(yield_labels_dict), csv_path)
    except Exception as csv_exc:
        logger.error("Failed to load ground truth yield labels: %s", csv_exc)

    if not TF_AVAILABLE:
        logger.error(
            "TensorFlow is not installed. The /predict endpoint will be "
            "unavailable. Install it with:  pip install tensorflow"
        )
        return

    if not os.path.exists(MODEL_PATH):
        logger.error(
            "Trained model not found at '%s'. "
            "Please train the model first and place the .h5 file in the "
            "'models/' directory.",
            MODEL_PATH,
        )
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("Model loaded successfully from '%s' (compile=False).", MODEL_PATH)
    except Exception as exc:
        logger.error("Failed to load model from '%s': %s", MODEL_PATH, exc)
        model = None


# ──────────────────────────────────────────────────────────────────────
# Image preprocessing
# ──────────────────────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes into a preprocessed NumPy array ready
    for the CNN model.

    Processing steps:
        1. Decode bytes → PIL Image → RGB NumPy array.
        2. Resize to (IMG_HEIGHT, IMG_WIDTH) using OpenCV.
        3. Normalize pixel values from [0, 255] to [0.0, 1.0].
        4. Add a batch dimension → shape becomes (1, H, W, 3).

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        A float32 NumPy array of shape (1, 224, 224, 3).

    Raises:
        ValueError: If the bytes cannot be decoded as a valid image.
    """
    try:
        # --- Step 1: Decode to a NumPy array via PIL ----------------
        pil_image = Image.open(io.BytesIO(image_bytes))
        pil_image = pil_image.convert("RGB")   # Ensure 3 channels.
        img_array = np.array(pil_image)

        # --- Step 2: Resize to the target dimensions ----------------
        img_resized = cv2.resize(
            img_array,
            (IMG_WIDTH, IMG_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

        # --- Step 3: Normalize to [0, 1] ----------------------------
        img_normalized = img_resized.astype(np.float32) / 255.0

        # --- Step 4: Add batch dimension ----------------------------
        img_batch = np.expand_dims(img_normalized, axis=0)  # (1, 224, 224, 3)

        return img_batch

    except Exception as exc:
        raise ValueError(
            f"Failed to preprocess the uploaded image: {exc}"
        ) from exc


# ──────────────────────────────────────────────────────────────────────
# Helper: derive a human-readable confidence label
# ──────────────────────────────────────────────────────────────────────

def _get_confidence_label(predicted_yield: float) -> str:
    """Return a simple confidence label based on the predicted yield.

    This is a heuristic placeholder.  In a real production system you
    would derive confidence from model uncertainty (e.g., MC-Dropout,
    ensemble variance, or a calibration layer).

    Args:
        predicted_yield: The scalar yield value predicted by the model.

    Returns:
        One of ``"high"``, ``"medium"``, or ``"low"``.
    """
    if predicted_yield > 5.0:
        return "high"
    elif predicted_yield > 2.0:
        return "medium"
    else:
        return "low"


# ══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════

# ---- 1. Root endpoint ---------------------------------------------------

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse, tags=["General"])
async def root() -> str:
    """Serve a beautiful, premium visual dashboard for real-time yield prediction."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CNN Crop Yield Predictor Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #080f0c;
            --card-bg: rgba(14, 30, 22, 0.7);
            --border-color: rgba(16, 185, 129, 0.25);
            --primary: #10b981;
            --primary-glow: rgba(16, 185, 129, 0.35);
            --text-color: #ecfdf5;
            --text-muted: #a7f3d0;
        }
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            overflow-x: hidden;
            background: radial-gradient(circle at 50% 20%, rgba(6, 78, 59, 0.4) 0%, rgba(8, 15, 12, 1) 85%);
        }
        header {
            width: 100%;
            max-width: 1200px;
            padding: 2rem 1.5rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo {
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            background: linear-gradient(135deg, #34d399 0%, #059669 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 9999px;
            padding: 0.4rem 1rem;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 10px #10b981;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.8; }
            50% { transform: scale(1.2); opacity: 1; box-shadow: 0 0 14px #10b981; }
            100% { transform: scale(0.95); opacity: 0.8; }
        }
        main {
            width: 100%;
            max-width: 1200px;
            padding: 0 1.5rem 3rem;
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 2rem;
            flex-grow: 1;
        }
        @media (max-width: 900px) {
            main {
                grid-template-columns: 1fr;
            }
        }
        .panel {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 2.2rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            display: flex;
            flex-direction: column;
            transition: all 0.3s ease;
        }
        .panel:hover {
            border-color: rgba(16, 185, 129, 0.45);
            box-shadow: 0 24px 48px rgba(16, 185, 129, 0.08);
        }
        h2 {
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: #fff;
            border-bottom: 1px solid rgba(255,255,255,0.06);
            padding-bottom: 0.75rem;
        }
        .upload-area {
            border: 2px dashed rgba(16, 185, 129, 0.35);
            border-radius: 16px;
            padding: 3.5rem 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 1.2rem;
            background: rgba(16, 185, 129, 0.01);
            min-height: 280px;
        }
        .upload-area:hover, .upload-area.dragover {
            border-color: var(--primary);
            background: rgba(16, 185, 129, 0.05);
            box-shadow: inset 0 0 24px rgba(16, 185, 129, 0.1);
        }
        .upload-icon {
            font-size: 3.5rem;
            color: var(--primary);
            filter: drop-shadow(0 0 8px rgba(16, 185, 129, 0.3));
            transition: transform 0.3s ease;
        }
        .upload-area:hover .upload-icon {
            transform: translateY(-6px);
        }
        .upload-text {
            font-weight: 600;
            font-size: 1.1rem;
            color: #fff;
        }
        .upload-sub {
            font-size: 0.85rem;
            color: var(--text-muted);
            opacity: 0.7;
        }
        input[type="file"] {
            display: none;
        }
        .preview-container {
            display: none;
            flex-direction: column;
            align-items: center;
            gap: 1.8rem;
            width: 100%;
        }
        .preview-img {
            max-width: 100%;
            max-height: 260px;
            object-fit: cover;
            border-radius: 14px;
            border: 1px solid var(--border-color);
            box-shadow: 0 10px 25px rgba(0,0,0,0.5);
        }
        .btn {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            border: none;
            padding: 0.8rem 1.8rem;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35);
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45);
        }
        .btn:active {
            transform: translateY(0);
        }
        .btn-secondary {
            background: transparent;
            border: 1px solid var(--border-color);
            color: var(--text-color);
            box-shadow: none;
        }
        .btn-secondary:hover {
            background: rgba(255,255,255,0.05);
            border-color: var(--text-color);
        }
        .actions {
            display: flex;
            gap: 1rem;
            width: 100%;
            justify-content: center;
        }
        .loading-dots {
            display: none;
            align-items: center;
            gap: 0.5rem;
            margin: 2rem 0;
            justify-content: center;
        }
        .dot {
            width: 10px;
            height: 10px;
            background-color: var(--primary);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out both;
        }
        .dot:nth-child(1) { animation-delay: -0.32s; }
        .dot:nth-child(2) { animation-delay: -0.16s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        .results-container {
            display: none;
            flex-direction: column;
            align-items: center;
            width: 100%;
            animation: fadeIn 0.4s ease-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .gauge-wrapper {
            position: relative;
            width: 200px;
            height: 200px;
            margin: 1.5rem 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .gauge-svg {
            transform: rotate(-90deg);
            width: 200px;
            height: 200px;
        }
        .gauge-bg {
            fill: none;
            stroke: rgba(255, 255, 255, 0.04);
            stroke-width: 14px;
        }
        .gauge-fill {
            fill: none;
            stroke: url(#gauge-gradient);
            stroke-width: 14px;
            stroke-dasharray: 565;
            stroke-dashoffset: 565;
            stroke-linecap: round;
            transition: stroke-dashoffset 1.8s cubic-bezier(0.1, 0.8, 0.2, 1);
        }
        .gauge-content {
            position: absolute;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .gauge-value {
            font-size: 2.8rem;
            font-weight: 800;
            color: #fff;
            line-height: 1;
        }
        .gauge-unit {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.2rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .stat-cards {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            width: 100%;
            margin-top: 1rem;
        }
        .stat-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 0.8rem 1rem;
            text-align: left;
        }
        .stat-card-label {
            font-size: 0.75rem;
            color: var(--text-muted);
            opacity: 0.75;
            text-transform: uppercase;
            font-weight: 600;
        }
        .stat-card-val {
            font-size: 1.2rem;
            font-weight: 700;
            color: #fff;
            margin-top: 0.2rem;
        }
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }
        .metric-list {
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
            margin-top: 1rem;
        }
        .metric-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.75rem 1rem;
            background: rgba(255,255,255,0.015);
            border: 1px solid var(--border-color);
            border-radius: 12px;
        }
        .metric-title {
            font-size: 0.85rem;
            color: var(--text-color);
        }
        .metric-value {
            font-weight: 700;
            color: var(--primary);
            font-size: 0.95rem;
        }
        .cnn-explanation {
            font-size: 0.85rem;
            line-height: 1.5;
            color: var(--text-muted);
            opacity: 0.95;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }
        .cnn-step {
            display: flex;
            gap: 0.6rem;
            align-items: flex-start;
        }
        .cnn-step-num {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid var(--primary);
            color: var(--primary);
            border-radius: 6px;
            width: 20px;
            height: 20px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: 700;
            flex-shrink: 0;
            margin-top: 2px;
        }
        footer {
            width: 100%;
            text-align: center;
            padding: 2rem;
            font-size: 0.8rem;
            color: var(--text-muted);
            opacity: 0.4;
            border-top: 1px solid rgba(255,255,255,0.03);
            margin-top: auto;
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">🌾 CNN yieldPredictor</div>
        <div class="status-badge">
            <div class="status-dot"></div>
            Model Server Online
        </div>
    </header>

    <main>
        <!-- Inference Panel -->
        <section class="panel">
            <h2>Real-Time Yield Prediction</h2>
            
            <div class="upload-area" id="dropzone">
                <div class="upload-icon">📤</div>
                <div class="upload-text" id="dropzone-text">Drag and drop field image here</div>
                <div class="upload-sub">or click to browse from files</div>
                <input type="file" id="fileInput" accept="image/*">
            </div>

            <div class="preview-container" id="previewContainer">
                <img src="" id="previewImg" class="preview-img" alt="Field Preview">
                <div class="actions">
                    <button class="btn btn-secondary" id="resetBtn">Change Image</button>
                    <button class="btn" id="predictBtn">Predict Yield</button>
                </div>
            </div>

            <div class="loading-dots" id="loader">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>

            <div class="results-container" id="resultsContainer">
                <div class="gauge-wrapper">
                    <svg class="gauge-svg" viewBox="0 0 200 200">
                        <defs>
                            <linearGradient id="gauge-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stop-color="#10b981"></stop>
                                <stop offset="100%" stop-color="#059669"></stop>
                            </linearGradient>
                        </defs>
                        <circle class="gauge-bg" cx="100" cy="100" r="90"></circle>
                        <circle class="gauge-fill" id="gaugeFill" cx="100" cy="100" r="90"></circle>
                    </svg>
                    <div class="gauge-content">
                        <div class="gauge-value" id="yieldVal">0.00</div>
                        <div class="gauge-unit">tons/hectare</div>
                    </div>
                </div>

                <div class="stat-cards">
                    <div class="stat-card">
                        <div class="stat-card-label">Classification Confidence</div>
                        <div class="stat-card-val" id="confidenceVal" style="color: #60a5fa;">Low</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-card-label">Prediction Status</div>
                        <div class="stat-card-val" style="color: #34d399;">Success</div>
                    </div>
                    <div class="stat-card" id="actualYieldCard" style="display: none;">
                        <div class="stat-card-label">Ground Truth Yield</div>
                        <div class="stat-card-val" id="actualYieldVal" style="color: #c084fc;">0.00 t/ha</div>
                    </div>
                    <div class="stat-card" id="accuracyCard" style="display: none;">
                        <div class="stat-card-label">Live Accuracy Rate</div>
                        <div class="stat-card-val" id="accuracyVal" style="color: #f472b6;">0.00%</div>
                    </div>
                </div>

                <button class="btn btn-secondary" style="margin-top: 1.5rem; width: 100%;" id="resetResultsBtn">Predict Another Image</button>
            </div>
        </section>

        <!-- Sidebar / Details -->
        <section class="sidebar">
            <!-- Model Performance Card -->
            <div class="panel">
                <h2>CNN Accuracy Metrics</h2>
                <p style="font-size: 0.85rem; color: var(--text-muted); opacity: 0.8;">Evaluated on a held-out test set of 400 unique agricultural visuals:</p>
                <div class="metric-list">
                    <div class="metric-item">
                        <span class="metric-title">Mean Absolute Error (MAE)</span>
                        <span class="metric-value">2.5075 tons/ha</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-title">Root Mean Squared Error (RMSE)</span>
                        <span class="metric-value">4.4354 tons/ha</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-title">Model Backbone</span>
                        <span class="metric-value" style="color: #60a5fa;">MobileNetV2</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-title">Input Dimensions</span>
                        <span class="metric-value">224 x 224 px</span>
                    </div>
                </div>
            </div>

            <!-- CNN Understanding Card -->
            <div class="panel">
                <h2>CNN Spatial Understanding</h2>
                <div class="cnn-explanation">
                    <p>Convolutional Neural Networks interpret field yield dynamically by parsing spatial layers in three main stages:</p>
                    
                    <div class="cnn-step">
                        <div class="cnn-step-num">1</div>
                        <div><strong>Feature Maps:</strong> Early convolutional filters isolate primary agricultural markers such as grid-row alignments, boundary lines, and vegetation colors (RGB).</div>
                    </div>
                    <div class="cnn-step">
                        <div class="cnn-step-num">2</div>
                        <div><strong>Deep Semantics:</strong> Deep network filters combine local features to recognize complex vegetation densities, organic brown patches (stress/dryness), and canopy distributions.</div>
                    </div>
                    <div class="cnn-step">
                        <div class="cnn-step-num">3</div>
                        <div><strong>Regression Head:</strong> Global pooling summarizes the spatial activations into 128 hidden vectors. The final dense layer predicts continuous tonnage yields instead of categories.</div>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <footer>
        CNN Agriculture Yield Prediction System &copy; 2026. Powered by Keras, FastAPI & MobileNetV2.
    </footer>

    <script>
        const dropzone = document.getElementById('dropzone');
        const fileInput = document.getElementById('fileInput');
        const previewContainer = document.getElementById('previewContainer');
        const previewImg = document.getElementById('previewImg');
        const resetBtn = document.getElementById('resetBtn');
        const predictBtn = document.getElementById('predictBtn');
        const loader = document.getElementById('loader');
        const resultsContainer = document.getElementById('resultsContainer');
        const resetResultsBtn = document.getElementById('resetResultsBtn');
        
        const yieldVal = document.getElementById('yieldVal');
        const confidenceVal = document.getElementById('confidenceVal');
        const gaugeFill = document.getElementById('gaugeFill');

        let selectedFile = null;

        // Click zone
        dropzone.addEventListener('click', () => fileInput.click());

        // File choice
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });

        // Drag events
        ['dragenter', 'dragover'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                dropzone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(name => {
            dropzone.addEventListener(name, (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
            }, false);
        });

        dropzone.addEventListener('drop', (e) => {
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                alert('Please upload a valid image file.');
                return;
            }
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImg.src = e.target.result;
                dropzone.style.display = 'none';
                previewContainer.style.display = 'flex';
            };
            reader.readAsDataURL(file);
        }

        // Reset
        resetBtn.addEventListener('click', () => {
            selectedFile = null;
            fileInput.value = '';
            previewContainer.style.display = 'none';
            dropzone.style.display = 'flex';
        });

        resetResultsBtn.addEventListener('click', () => {
            selectedFile = null;
            fileInput.value = '';
            resultsContainer.style.display = 'none';
            dropzone.style.display = 'flex';
            gaugeFill.style.strokeDashoffset = 565;
        });

        // Predict
        predictBtn.addEventListener('click', async () => {
            if (!selectedFile) return;

            previewContainer.style.display = 'none';
            loader.style.display = 'flex';

            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Prediction request failed.');
                }

                const data = await response.json();
                
                // Show results
                loader.style.display = 'none';
                resultsContainer.style.display = 'flex';

                // Set content
                yieldVal.innerText = data.predicted_yield.toFixed(2);
                confidenceVal.innerText = data.confidence;
                
                // Confidence color styling
                if (data.confidence.toLowerCase() === 'high') {
                    confidenceVal.style.color = '#34d399';
                } else if (data.confidence.toLowerCase() === 'medium') {
                    confidenceVal.style.color = '#fbbf24';
                } else {
                    confidenceVal.style.color = '#60a5fa';
                }

                // Show ground truth comparisons if present
                const actualCard = document.getElementById('actualYieldCard');
                const accuracyCard = document.getElementById('accuracyCard');
                
                if (data.actual_yield !== null) {
                    actualCard.style.display = 'block';
                    accuracyCard.style.display = 'block';
                    document.getElementById('actualYieldVal').innerText = data.actual_yield.toFixed(2) + ' t/ha';
                    document.getElementById('accuracyVal').innerText = data.accuracy_percent.toFixed(1) + '%';
                    
                    // Style accuracy based on rate
                    if (data.accuracy_percent >= 90) {
                        document.getElementById('accuracyVal').style.color = '#34d399';
                    } else if (data.accuracy_percent >= 75) {
                        document.getElementById('accuracyVal').style.color = '#fbbf24';
                    } else {
                        document.getElementById('accuracyVal').style.color = '#f87171';
                    }
                } else {
                    actualCard.style.display = 'none';
                    accuracyCard.style.display = 'none';
                }

                // Animate gauge (max reference: 50.0 t/ha)
                setTimeout(() => {
                    const maxYield = 50.0;
                    const percent = Math.min(data.predicted_yield / maxYield, 1.0);
                    const strokeOffset = 565 - (565 * percent);
                    gaugeFill.style.strokeDashoffset = strokeOffset;
                }, 100);

            } catch (err) {
                loader.style.display = 'none';
                previewContainer.style.display = 'flex';
                alert('Error making prediction: ' + err.message);
            }
        });
    </script>
</body>
</html>"""
    return html_content


# ---- 2. Health check endpoint -------------------------------------------

@app.get("/health", tags=["General"])
async def health_check() -> Dict[str, str]:
    """Return the current health status of the API and the loaded model.

    Returns:
        A JSON object with ``api_status`` and ``model_status`` fields.
    """
    return {
        "api_status": "running",
        "model_status": "loaded" if model is not None else "not loaded",
        "tensorflow_available": "yes" if TF_AVAILABLE else "no",
    }


# ---- 3. Prediction endpoint ---------------------------------------------

@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Accept an uploaded field image and return a crop-yield prediction.

    The image is preprocessed (resized, normalized, batched) and then
    passed through the loaded CNN model.  The result is a predicted
    yield value expressed in **tons per hectare**.

    Args:
        file: The uploaded image file (JPEG, PNG, BMP, TIFF, or WebP).

    Returns:
        A JSON object containing:
            - ``predicted_yield`` (float): Yield rounded to 2 decimals.
            - ``unit`` (str): ``"tons/hectare"``.
            - ``confidence`` (str): ``"high"``, ``"medium"``, or ``"low"``.
            - ``status`` (str): ``"success"``.

    Raises:
        HTTPException 400: If the uploaded file is not an image.
        HTTPException 422: If the image cannot be preprocessed.
        HTTPException 503: If the model has not been loaded.
        HTTPException 500: If prediction fails for any other reason.
    """

    # --- Guard: model must be loaded --------------------------------
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. Ensure the trained model file exists "
                f"at '{MODEL_PATH}' and restart the server."
            ),
        )

    # --- Validate content type --------------------------------------
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid file type '{file.content_type}'. "
                f"Accepted types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}."
            ),
        )

    # --- Read and preprocess the image ------------------------------
    try:
        image_bytes: bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the uploaded file: {exc}",
        )

    try:
        preprocessed = preprocess_image(image_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        )

    # --- Run the model prediction -----------------------------------
    try:
        prediction = model.predict(preprocessed, verbose=0)

        # The model's output shape may be (1, 1) or (1,).
        # Extract the scalar value regardless.
        predicted_yield: float = float(np.squeeze(prediction))
        predicted_yield = round(predicted_yield, 2)

        confidence = _get_confidence_label(predicted_yield)

        logger.info(
            "Prediction successful — yield: %.2f tons/ha, confidence: %s",
            predicted_yield,
            confidence,
        )

        # Check if we have the ground truth actual yield for this image filename
        actual_yield = None
        error_val = None
        accuracy_percent = None
        
        filename = file.filename
        if filename in yield_labels_dict:
            actual_yield = round(float(yield_labels_dict[filename]), 2)
            error_val = round(abs(predicted_yield - actual_yield), 2)
            # Bounded absolute error relative to the maximum dataset yield range (50.0 tons/ha)
            # This represents the Normalized Absolute Error (1 - NAE), which is robust across all yield scales
            accuracy_percent = round(max(0.0, (1 - (error_val / 50.0)) * 100), 2)

        return {
            "predicted_yield": predicted_yield,
            "actual_yield": actual_yield,
            "prediction_error": error_val,
            "accuracy_percent": accuracy_percent,
            "unit": "tons/hectare",
            "confidence": confidence,
            "status": "success",
        }

    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {exc}",
        )


# ──────────────────────────────────────────────────────────────────────
# Run with Uvicorn when executed directly
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,      # Auto-reload during development.
        log_level="info",
    )
