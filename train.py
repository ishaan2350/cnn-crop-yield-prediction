"""
train.py - CNN-based Agriculture Yield Prediction Training Script
=================================================================

This script trains a Convolutional Neural Network (CNN) to predict crop yield
from aerial/satellite images of agricultural fields. It uses transfer learning
with MobileNetV2 (pretrained on ImageNet) as the feature extractor and adds
custom regression layers on top.

Pipeline Overview:
    1. Load images and corresponding yield labels from disk.
    2. Preprocess images (resize, normalize, augment).
    3. Build a MobileNetV2-based regression model.
    4. Train the model with early stopping and learning-rate reduction.
    5. Evaluate on a held-out test set (MAE, RMSE).
    6. Save the trained model and training-history plots.

If no real dataset is found, the script auto-generates 100 synthetic crop-field
images so you can run the full pipeline out of the box.

Usage:
    python train.py
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import os
import sys
import warnings

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split

# Suppress noisy TF/Keras info logs (keep warnings & errors)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================================
# 2. CONFIGURATION CONSTANTS
# ============================================================================
# -- Paths (relative to the project root) ------------------------------------
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
CSV_PATH = os.path.join(DATASET_DIR, "yield.csv")
MODELS_DIR = "models"

# -- Image settings ----------------------------------------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# -- Training hyper-parameters -----------------------------------------------
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 10
TEST_SIZE = 0.20  # 80/20 train/test split
RANDOM_STATE = 42

# -- Synthetic data settings -------------------------------------------------
NUM_SYNTHETIC_SAMPLES = 100


# ============================================================================
# 3. SYNTHETIC DATA GENERATION
# ============================================================================
def generate_synthetic_data(
    images_dir: str = IMAGES_DIR,
    csv_path: str = CSV_PATH,
    num_samples: int = NUM_SYNTHETIC_SAMPLES,
) -> None:
    """Generate synthetic crop-field images and a matching ``yield.csv`` file.

    Each synthetic image simulates an aerial view of a crop field by compositing
    multiple layers:
        * A base colour drawn from green / brown / yellow palettes.
        * Random rectangular "patches" (e.g. brown bare-soil areas).
        * Gaussian noise to mimic natural texture.
        * Gaussian blur for a more organic look.

    The synthetic yield value is loosely correlated with the amount of green in
    the image so the model has a learnable signal.

    Parameters
    ----------
    images_dir : str
        Directory where the PNG images will be saved.
    csv_path : str
        Path for the output CSV file (columns: ``image_name``, ``yield``).
    num_samples : int
        Number of synthetic images to generate.
    """
    print(f"\n{'='*60}")
    print("  Generating Synthetic Crop-Field Dataset")
    print(f"{'='*60}")

    os.makedirs(images_dir, exist_ok=True)

    records: list[dict] = []
    np.random.seed(RANDOM_STATE)

    for i in range(num_samples):
        # -- 1. Random base colour ------------------------------------------
        # Green-ish base dominates; we mix in brown/yellow occasionally.
        green_intensity = np.random.randint(100, 220)
        red_base = np.random.randint(30, 120)
        blue_base = np.random.randint(20, 80)

        img = np.full((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8,
                       fill_value=[red_base, green_intensity, blue_base])

        # -- 2. Random rectangular patches (bare soil / dry areas) ----------
        num_patches = np.random.randint(2, 8)
        brown_area_ratio = 0.0  # track how much brown we add

        for _ in range(num_patches):
            x1 = np.random.randint(0, IMG_WIDTH - 20)
            y1 = np.random.randint(0, IMG_HEIGHT - 20)
            x2 = min(x1 + np.random.randint(15, 80), IMG_WIDTH)
            y2 = min(y1 + np.random.randint(15, 80), IMG_HEIGHT)

            patch_w, patch_h = x2 - x1, y2 - y1
            brown_area_ratio += (patch_w * patch_h) / (IMG_WIDTH * IMG_HEIGHT)

            # Choose patch colour: brown, dark green, or yellow
            colour_choice = np.random.choice(["brown", "dark_green", "yellow"],
                                             p=[0.5, 0.3, 0.2])
            if colour_choice == "brown":
                patch_colour = [
                    np.random.randint(100, 170),
                    np.random.randint(70, 130),
                    np.random.randint(30, 70),
                ]
            elif colour_choice == "dark_green":
                patch_colour = [
                    np.random.randint(20, 60),
                    np.random.randint(80, 140),
                    np.random.randint(20, 50),
                ]
            else:  # yellow
                patch_colour = [
                    np.random.randint(180, 240),
                    np.random.randint(180, 230),
                    np.random.randint(30, 80),
                ]

            img[y1:y2, x1:x2] = patch_colour

        # -- 3. Add Gaussian noise for texture ------------------------------
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # -- 4. Gaussian blur for a softer, more natural look ---------------
        kernel_size = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        # -- 5. Compute a synthetic yield correlated with greenness ----------
        # Higher green channel mean  ⟹  higher yield (simple heuristic)
        green_mean = float(img[:, :, 1].mean())  # channel index 1 = G
        yield_value = round(
            0.05 * green_mean
            - 10.0 * brown_area_ratio
            + np.random.normal(0, 1.0),
            2,
        )
        yield_value = max(yield_value, 0.5)  # keep yield positive

        # -- 6. Save the image -----------------------------------------------
        filename = f"crop_{i:04d}.png"
        filepath = os.path.join(images_dir, filename)
        # OpenCV uses BGR; our array is RGB, so convert before saving
        cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        records.append({"image_name": filename, "yield": yield_value})

        # Progress feedback every 25 images
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [synthetic] Generated {i + 1}/{num_samples} images …")

    # -- 7. Write CSV --------------------------------------------------------
    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved {num_samples} images to  → {images_dir}/")
    print(f"  ✓ Saved yield labels to          → {csv_path}")


# ============================================================================
# 4. DATASET LOADING
# ============================================================================
def load_dataset(
    images_dir: str = IMAGES_DIR,
    csv_path: str = CSV_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """Load crop images and their yield labels from disk.

    Images are resized to ``(IMG_HEIGHT, IMG_WIDTH)`` and converted to
    ``float32`` arrays.  Missing files are skipped with a warning.

    Parameters
    ----------
    images_dir : str
        Directory containing the crop images.
    csv_path : str
        Path to the CSV file with columns ``image_name`` and ``yield``.

    Returns
    -------
    images : np.ndarray, shape (N, IMG_HEIGHT, IMG_WIDTH, 3)
        Normalised pixel values in [0, 1].
    yields : np.ndarray, shape (N,)
        Corresponding yield values.
    """
    print(f"\n{'='*60}")
    print("  Loading Dataset")
    print(f"{'='*60}")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Yield CSV not found at '{csv_path}'. "
            "Run generate_synthetic_data() first or provide a real dataset."
        )

    df = pd.read_csv(csv_path)
    print(f"  CSV loaded: {len(df)} entries")

    images: list[np.ndarray] = []
    yields: list[float] = []
    skipped = 0

    for idx, row in df.iterrows():
        img_path = os.path.join(images_dir, str(row["image_name"]))

        if not os.path.isfile(img_path):
            skipped += 1
            if skipped <= 5:  # limit console noise
                print(f"  [Warning]  Image not found, skipping: {img_path}")
            continue

        # Read as BGR, convert to RGB, resize
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            print(f"  [Warning]  Could not decode image, skipping: {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        images.append(img)
        yields.append(float(row["yield"]))

    if skipped:
        print(f"  [Warning]  Total skipped images: {skipped}")

    if not images:
        raise RuntimeError("No valid images were loaded. Check your dataset.")

    # Convert to NumPy arrays and normalise pixel values to [0, 1]
    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(yields, dtype=np.float32)

    print(f"  [*] Loaded {len(X)} images  |  shape: {X.shape}")
    print(f"  [*] Yield range: [{y.min():.2f}, {y.max():.2f}]  "
          f"mean={y.mean():.2f}")

    return X, y


# ============================================================================
# 5. DATA AUGMENTATION
# ============================================================================
def build_augmenter() -> ImageDataGenerator:
    """Create a Keras ``ImageDataGenerator`` with augmentation transforms.

    Augmentations applied:
        * Random rotation up to ±20°
        * Random zoom up to 20 %
        * Horizontal flip
        * Width / height shifts up to 10 %
        * Nearest-pixel fill for empty areas

    Returns
    -------
    ImageDataGenerator
        Configured augmenter instance.
    """
    augmenter = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode="nearest",
    )
    print("  [*] Data augmenter configured")
    return augmenter


# ============================================================================
# 6. MODEL ARCHITECTURE
# ============================================================================
def build_model() -> Model:
    """Build a MobileNetV2-based regression model for yield prediction.

    Architecture:
        MobileNetV2 (frozen, ImageNet weights)
        → GlobalAveragePooling2D
        → Dense(128, relu)
        → Dropout(0.3)
        → Dense(1)          ← regression head

    Returns
    -------
    keras.Model
        The compiled Keras model ready for training.
    """
    print(f"\n{'='*60}")
    print("  Building Model (MobileNetV2 + Regression Head)")
    print(f"{'='*60}")

    # -- Load the pre-trained MobileNetV2 base ------------------------------
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SHAPE,
    )
    # Freeze all layers in the base model so only the head is trained
    base_model.trainable = False
    print(f"  [*] MobileNetV2 base loaded ({len(base_model.layers)} layers, frozen)")

    # -- Build the regression head ------------------------------------------
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dense(128, activation="relu", name="fc_128")(x)
    x = Dropout(0.3, name="dropout_0.3")(x)
    output = Dense(1, name="yield_output")(x)

    model = Model(inputs=base_model.input, outputs=output, name="CropYieldCNN")

    # -- Compile -------------------------------------------------------------
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"],
    )

    # Quick summary
    total_params = model.count_params()
    trainable_params = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    print(f"  [*] Model compiled  |  Total params: {total_params:,}")
    print(f"    Trainable params: {trainable_params:,}  "
          f"(frozen base: {total_params - trainable_params:,})")

    return model


# ============================================================================
# 7. CALLBACKS
# ============================================================================
def get_callbacks() -> list:
    """Return a list of Keras callbacks for training.

    Callbacks:
        * **EarlyStopping** – stops training when ``val_loss`` has not improved
          for 5 consecutive epochs; restores the best weights.
        * **ReduceLROnPlateau** – halves the learning rate when ``val_loss``
          stalls for 3 epochs (minimum LR = 1e-6).

    Returns
    -------
    list[keras.callbacks.Callback]
    """
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    print("  [*] Callbacks: EarlyStopping (patience=5), "
          "ReduceLROnPlateau (factor=0.5, patience=3)")

    return [early_stop, reduce_lr]


# ============================================================================
# 8. TRAINING
# ============================================================================
def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    augmenter: ImageDataGenerator,
) -> dict:
    """Train the model using augmented data and return the history.

    Parameters
    ----------
    model : keras.Model
        Compiled Keras model.
    X_train, y_train : np.ndarray
        Training images and labels.
    X_test, y_test : np.ndarray
        Validation/test images and labels.
    augmenter : ImageDataGenerator
        Configured augmenter for on-the-fly training data augmentation.

    Returns
    -------
    dict
        The ``history.history`` dictionary with training/validation metrics.
    """
    print(f"\n{'='*60}")
    print("  Training")
    print(f"{'='*60}")
    print(f"  Train samples : {len(X_train)}")
    print(f"  Test  samples : {len(X_test)}")
    print(f"  Epochs        : {EPOCHS}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Learning rate : {LEARNING_RATE}")
    print()

    callbacks = get_callbacks()

    # Fit the augmenter on training data (computes internal statistics)
    augmenter.fit(X_train)

    # Train using the augmented data generator
    history = model.fit(
        augmenter.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    print("\n  [*] Training complete")
    return history.history


# ============================================================================
# 9. EVALUATION
# ============================================================================
def evaluate_model(
    model: Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """Evaluate the trained model on the test set.

    Parameters
    ----------
    model : keras.Model
        Trained model.
    X_test, y_test : np.ndarray
        Test images and ground-truth yields.

    Returns
    -------
    mae : float
        Mean Absolute Error on the test set.
    rmse : float
        Root Mean Squared Error on the test set.
    """
    print(f"\n{'='*60}")
    print("  Evaluation on Test Set")
    print(f"{'='*60}")

    predictions = model.predict(X_test, verbose=0).flatten()

    mae = float(np.mean(np.abs(predictions - y_test)))
    rmse = float(np.sqrt(np.mean((predictions - y_test) ** 2)))

    print(f"  +------------------------------+")
    print(f"  |  MAE  : {mae:>10.4f}           |")
    print(f"  |  RMSE : {rmse:>10.4f}           |")
    print(f"  +------------------------------+")

    return mae, rmse


# ============================================================================
# 10. SAVE MODEL
# ============================================================================
def save_model(model: Model, models_dir: str = MODELS_DIR) -> None:
    """Save the trained model in both legacy H5 and modern Keras formats.

    Creates the ``models/`` directory if it does not already exist.

    Parameters
    ----------
    model : keras.Model
        Trained Keras model to save.
    models_dir : str
        Target directory for saved model files.
    """
    print(f"\n{'='*60}")
    print("  Saving Model")
    print(f"{'='*60}")

    os.makedirs(models_dir, exist_ok=True)

    h5_path = os.path.join(models_dir, "crop_yield_model.h5")
    keras_path = os.path.join(models_dir, "crop_yield_model.keras")

    model.save(h5_path)
    print(f"  [*] Saved H5 model    -> {h5_path}")

    model.save(keras_path)
    print(f"  [*] Saved Keras model -> {keras_path}")


# ============================================================================
# 11. VISUALIZATION
# ============================================================================
def plot_training_history(
    history: dict,
    save_path: str | None = None,
) -> None:
    """Plot training vs. validation loss and MAE curves.

    Generates a side-by-side figure:
        * Left panel  — MSE loss curve
        * Right panel — MAE curve

    Parameters
    ----------
    history : dict
        The ``history.history`` dictionary returned by ``model.fit()``.
    save_path : str or None
        If provided, the figure is saved to this path (PNG).
    """
    print(f"\n{'='*60}")
    print("  Plotting Training History")
    print(f"{'='*60}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- Loss (MSE) ----------------------------------------------------------
    axes[0].plot(history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title("Training vs Validation Loss (MSE)", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # -- MAE -----------------------------------------------------------------
    axes[1].plot(history["mae"], label="Train MAE", linewidth=2)
    axes[1].plot(history["val_mae"], label="Val MAE", linewidth=2)
    axes[1].set_title("Training vs Validation MAE", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  [*] Training plots saved -> {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ============================================================================
# 12. MAIN ENTRY POINT
# ============================================================================
def main() -> None:
    """Orchestrate the full training pipeline.

    Steps
    -----
    1. Check for an existing dataset; generate synthetic data if absent.
    2. Load and preprocess images + labels.
    3. Split into train / test sets (80 / 20).
    4. Build the MobileNetV2 regression model.
    5. Train with data augmentation and callbacks.
    6. Evaluate on the test set (MAE, RMSE).
    7. Save the model (H5 + Keras).
    8. Plot and save training-history curves.
    """
    print("=" * 60)
    print("  CNN-based Agriculture Yield Prediction - Training")
    print("=" * 60)

    # ---- Step 1: Ensure dataset exists ------------------------------------
    dataset_exists = (
        os.path.isdir(IMAGES_DIR)
        and len(os.listdir(IMAGES_DIR)) > 0
        and os.path.isfile(CSV_PATH)
    )

    if not dataset_exists:
        print("\n  [Info]  No dataset found - generating synthetic data ...")
        generate_synthetic_data()
    else:
        print(f"\n  [*] Existing dataset detected at '{DATASET_DIR}/'")

    # ---- Step 2: Load dataset ---------------------------------------------
    X, y = load_dataset()

    # ---- Step 3: Train / test split ---------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"\n  [*] Train/Test split: {len(X_train)} / {len(X_test)}")

    # ---- Step 4: Data augmenter -------------------------------------------
    augmenter = build_augmenter()

    # ---- Step 5: Build model ----------------------------------------------
    model = build_model()

    # ---- Step 6: Train ----------------------------------------------------
    history = train_model(model, X_train, y_train, X_test, y_test, augmenter)

    # ---- Step 7: Evaluate -------------------------------------------------
    mae, rmse = evaluate_model(model, X_test, y_test)

    # ---- Step 8: Save model -----------------------------------------------
    save_model(model)

    # ---- Step 9: Visualise ------------------------------------------------
    plot_path = os.path.join(MODELS_DIR, "training_history.png")
    plot_training_history(history, save_path=plot_path)

    # ---- Done! ------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  [Success]  Pipeline finished successfully!")
    print(f"{'='*60}")
    print(f"  - Model saved to     : {MODELS_DIR}/")
    print(f"  - Test MAE           : {mae:.4f}")
    print(f"  - Test RMSE          : {rmse:.4f}")
    print(f"  - History plot       : {plot_path}")
    print()


if __name__ == "__main__":
    main()
