"""
sample_predictions.py - Standalone Offline Prediction Utility & Sample Predictions Runner
=============================================================================

This script runs sample predictions on the 5 sample crop images located in
`dataset/images/` using the trained CNN model. It displays a beautiful offline
evaluation table comparing Actual Yields vs. Predicted Yields, including
absolute errors and live accuracy ratings.

Usage:
    python sample_predictions.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import warnings

# Suppress noisy TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[Error] TensorFlow is not installed. Run: pip install tensorflow")

# Configuration constants
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "crop_yield_model.h5")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "dataset", "images")
CSV_PATH = os.path.join(PROJECT_ROOT, "dataset", "yield.csv")

IMG_HEIGHT = 224
IMG_WIDTH = 224

def preprocess_image(img_path: str) -> np.ndarray:
    """Load, standardize, and normalize an image for model input."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
        
    # Read as BGR and convert to RGB
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {img_path}")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to target CNN dimensions (224x224)
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
    
    # Normalize pixel values to [0.0, 1.0]
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension (1, 224, 224, 3)
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch

def main():
    print("=" * 75)
    print("  CNN Agriculture Yield Prediction - Standalone Sample Predictions")
    print("=" * 75)

    if not TF_AVAILABLE:
        return

    # Check for model existence
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Trained model not found at '{MODEL_PATH}'.")
        print("Please train the model first by running: python train.py")
        return

    # Load model (compile=False bypasses custom metric serialization checks in Keras 3)
    print(f"[*] Loading model from '{MODEL_PATH}' ...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("[OK] Model loaded successfully!\n")
    except Exception as e:
        print(f"[Error] Failed to load model: {e}")
        return

    # Load ground truth labels mapping
    yield_labels = {}
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            yield_labels = dict(zip(df["image_name"], df["yield"]))
        except Exception as e:
            print(f"[Warning] Failed to parse yield.csv: {e}")

    # List of 5 pruned sample images
    sample_files = [
        "crop_00000.png",
        "crop_00010.png",
        "crop_00016.png",
        "crop_00628.png",
        "crop_00637.png"
    ]

    # Verify sample images directory
    if not os.path.exists(IMAGES_DIR):
        print(f"[Error] Sample images directory not found at '{IMAGES_DIR}'.")
        return

    results = []

    # Run predictions on each sample image
    for filename in sample_files:
        img_path = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(img_path):
            print(f"[Warning] Sample file '{filename}' is missing from '{IMAGES_DIR}'.")
            continue

        try:
            # 1. Preprocess
            img_batch = preprocess_image(img_path)
            
            # 2. Predict
            prediction = model.predict(img_batch, verbose=0)
            predicted_yield = float(np.squeeze(prediction))
            predicted_yield = round(predicted_yield, 2)
            
            # 3. Fetch Actual Ground Truth
            actual_yield = yield_labels.get(filename, None)
            
            # 4. Compute errors and metrics
            if actual_yield is not None:
                actual_yield = round(float(actual_yield), 2)
                abs_error = round(abs(predicted_yield - actual_yield), 2)
                # Bounded Normalized Absolute Error (NAE) relative to max range (50.0 t/ha)
                accuracy = round(max(0.0, (1 - (abs_error / 50.0)) * 100), 2)
            else:
                abs_error = "N/A"
                accuracy = "N/A"
                
            # Derive confidence and status like the FastAPI server does
            if predicted_yield > 5.0:
                confidence = "high"
            elif predicted_yield > 2.0:
                confidence = "medium"
            else:
                confidence = "low"
            status = "Success"
                
            results.append({
                "Image Filename": filename,
                "Actual Yield": f"{actual_yield:.2f} t/ha" if actual_yield is not None else "N/A",
                "Predicted Yield": f"{predicted_yield:.2f} t/ha",
                "Status": status,
                "Confidence": confidence,
                "Abs Error": f"{abs_error:.2f} t/ha" if isinstance(abs_error, float) else "N/A",
                "Accuracy": f"{accuracy:.2f}%" if isinstance(accuracy, float) else "N/A"
            })
            
        except Exception as e:
            print(f"[Error] Failed predicting on {filename}: {e}")

    # Output results in a beautiful ASCII table
    if not results:
        print("[Error] No sample predictions were completed successfully.")
        return

    print("+" + "-" * 96 + "+")
    print(f"| {'Image Filename':<16} | {'Actual Yield':<12} | {'Predicted Yield':<15} | {'Status':<8} | {'Confidence':<10} | {'Abs Error':<10} | {'Accuracy':<8} |")
    print("+" + "-" * 96 + "+")
    for r in results:
        print(f"| {r['Image Filename']:<16} | {r['Actual Yield']:<12} | {r['Predicted Yield']:<15} | {r['Status']:<8} | {r['Confidence']:<10} | {r['Abs Error']:<10} | {r['Accuracy']:<8} |")
    print("+" + "-" * 96 + "+")
    print("\n[SUCCESS] Sample predictions runner finished successfully!")
    print("To test live requests, start the API: python -m uvicorn api.main:app")
    print("=" * 75)

if __name__ == "__main__":
    main()
