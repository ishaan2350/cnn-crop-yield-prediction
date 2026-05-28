"""
convert_csv_to_images.py — Convert FAO Crop CSV Data to Realistic Field Images
================================================================================

This script reads FAO crop production CSV files and generates realistic
synthetic aerial crop field images where the visual appearance (greenness,
crop density, health indicators) is correlated with actual yield values.

Each image simulates a satellite/drone view of a crop field:
    - High yield → lush green, dense crop rows, minimal bare soil
    - Low yield  → yellowish/brown, sparse coverage, dry patches
    - Medium     → mixed green with some stress patterns

Usage:
    python convert_csv_to_images.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import cv2

# ============================================================================
# CONFIGURATION
# ============================================================================
DOWNLOADS_DIR = os.path.expanduser("~/Downloads")
IMAGES_DIR = os.path.join("dataset", "images")
YIELD_CSV_PATH = os.path.join("dataset", "yield.csv")
IMG_SIZE = 224
MAX_IMAGES = 2000  # Set to None for all records


# ============================================================================
# 1. LOCATE AND LOAD CSV FILES
# ============================================================================
def find_csv_files() -> list[str]:
    """Find FAO Production_Crops CSV files in Downloads."""
    files = glob.glob(os.path.join(DOWNLOADS_DIR, "Production_Crops_E_*.csv"))
    if not files:
        print("  [Warning]  No CSV files found in Downloads!")
        print(f"     Expected: Production_Crops_E_*.csv in {DOWNLOADS_DIR}")
        sys.exit(1)
    print(f"  [*] Found {len(files)} CSV file(s):")
    for f in files:
        print(f"    - {os.path.basename(f)}")
    return files


def load_and_parse_csvs(csv_files: list[str]) -> pd.DataFrame:
    """Load FAO CSV files and combine them."""
    all_data = []
    for csv_file in csv_files:
        print(f"  Loading: {os.path.basename(csv_file)} ...")
        try:
            df = pd.read_csv(csv_file, encoding='latin-1', low_memory=False)
        except Exception:
            df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
        print(f"    Rows: {len(df)}")
        year_cols = [c for c in df.columns if c.startswith('Y') and c[1:].isdigit()]
        keep_cols = ['Area', 'Item', 'Element', 'Unit'] + year_cols
        available_cols = [c for c in keep_cols if c in df.columns]
        all_data.append(df[available_cols].copy())
    combined = pd.concat(all_data, ignore_index=True)
    print(f"  [*] Combined: {len(combined)} rows")
    return combined


# ============================================================================
# 2. EXTRACT YIELD RECORDS
# ============================================================================
def extract_yield_records(df: pd.DataFrame) -> list[dict]:
    """Extract crop-country yield records with the latest valid yield."""
    year_cols = sorted([c for c in df.columns if c.startswith('Y') and c[1:].isdigit()])
    yield_df = df[df['Element'] == 'Yield'].copy()
    print(f"  Yield rows found: {len(yield_df)}")

    records = []
    for _, row in yield_df.iterrows():
        area = str(row.get('Area', 'Unknown'))
        item = str(row.get('Item', 'Unknown'))

        # Extract yield time series
        values = []
        for y in year_cols:
            try:
                values.append(float(row[y]))
            except (ValueError, TypeError):
                values.append(np.nan)
        arr = np.array(values)
        valid = arr[~np.isnan(arr)]

        if len(valid) < 3:
            continue

        # Convert from hg/ha to tons/hectare (1 ton = 10,000 hg)
        target_yield = round(valid[-1] / 10000.0, 2)
        mean_yield = float(np.nanmean(valid)) / 10000.0

        if target_yield <= 0 or target_yield > 50:
            continue

        records.append({
            'area': area,
            'item': item,
            'target_yield': target_yield,
            'mean_yield': mean_yield,
            'trend': valid,  # raw hg/ha values for generating texture
        })

    print(f"  [*] Extracted {len(records)} valid records")
    return records


# ============================================================================
# 3. GENERATE REALISTIC CROP FIELD IMAGES
# ============================================================================
def generate_crop_field_image(
    yield_value: float,
    mean_yield: float,
    seed: int,
    size: int = IMG_SIZE,
) -> np.ndarray:
    """Generate a realistic aerial crop field image based on yield value.

    High yield  → lush green, dense crop rows, minimal bare soil
    Low yield   → dry yellow/brown, sparse coverage, bare patches
    Medium      → mixed green with some stress

    Args:
        yield_value: Target yield in tons/hectare (0.5 – 50).
        mean_yield:  Historical mean yield for this crop-country.
        seed:        Random seed for reproducibility.
        size:        Output image dimensions (square).

    Returns:
        BGR image as np.ndarray of shape (size, size, 3), dtype uint8.
    """
    rng = np.random.RandomState(seed)

    # Normalize yield to a 0–1 "health" score (clamped)
    health = np.clip(yield_value / 15.0, 0.0, 1.0)

    # ── 1. BASE COLOR (RGB) ─────────────────────────────────────────
    # Healthy: deep green (40, 140, 30)  →  Stressed: yellow-brown (160, 140, 50)
    base_r = int(40 + (1 - health) * 120 + rng.randint(-10, 10))
    base_g = int(80 + health * 100 + rng.randint(-10, 10))
    base_b = int(20 + health * 30 + rng.randint(-5, 5))

    img = np.full((size, size, 3), [base_r, base_g, base_b], dtype=np.uint8)

    # ── 2. CROP ROW PATTERN ─────────────────────────────────────────
    # Simulate parallel crop rows visible from aerial view
    row_spacing = rng.randint(6, 14)
    row_thickness = max(1, int(row_spacing * (0.3 + health * 0.4)))
    row_angle = rng.uniform(-15, 15)  # slight tilt

    # Create crop rows
    rows_layer = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(0, size + 50, row_spacing):
        # Row color (darker green for healthier crops)
        row_r = max(0, base_r - int(20 * health) + rng.randint(-5, 5))
        row_g = min(255, base_g + int(30 * health) + rng.randint(-5, 5))
        row_b = max(0, base_b - 5 + rng.randint(-3, 3))

        pt1 = (0, y)
        pt2 = (size, y + int(row_angle * size / 50))
        cv2.line(rows_layer, pt1, pt2, (row_r, row_g, row_b), row_thickness)

    # Blend rows with base
    alpha = 0.4 + health * 0.3
    mask = rows_layer.sum(axis=2) > 0
    img[mask] = cv2.addWeighted(img, 1 - alpha, rows_layer, alpha, 0)[mask]

    # ── 3. TEXTURE VARIATION (simulate leaf canopy) ──────────────────
    # Add small-scale noise for natural texture
    noise = rng.normal(0, 8 + (1 - health) * 8, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ── 4. BARE SOIL / DRY PATCHES (more for low yield) ─────────────
    num_patches = int((1 - health) * rng.randint(3, 10))
    for _ in range(num_patches):
        cx = rng.randint(20, size - 20)
        cy = rng.randint(20, size - 20)
        rx = rng.randint(10, 40 + int((1 - health) * 30))
        ry = rng.randint(10, 30 + int((1 - health) * 20))
        angle = rng.randint(0, 180)

        # Brown/tan soil color
        soil_r = rng.randint(130, 180)
        soil_g = rng.randint(100, 140)
        soil_b = rng.randint(50, 80)

        overlay = img.copy()
        cv2.ellipse(overlay, (cx, cy), (rx, ry), angle, 0, 360,
                    (soil_r, soil_g, soil_b), -1)
        patch_alpha = 0.4 + (1 - health) * 0.3
        img = cv2.addWeighted(overlay, patch_alpha, img, 1 - patch_alpha, 0)

    # ── 5. HEALTHY BRIGHT SPOTS (more for high yield) ────────────────
    num_bright = int(health * rng.randint(2, 8))
    for _ in range(num_bright):
        cx = rng.randint(15, size - 15)
        cy = rng.randint(15, size - 15)
        radius = rng.randint(8, 25)

        # Bright green highlight
        bright_r = rng.randint(30, 70)
        bright_g = rng.randint(160, 220)
        bright_b = rng.randint(30, 70)

        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), radius, (bright_r, bright_g, bright_b), -1)
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)

    # ── 6. FIELD BOUNDARY / EDGE LINES ───────────────────────────────
    # Occasional irrigation lines or field edges
    if rng.random() > 0.4:
        num_lines = rng.randint(1, 3)
        for _ in range(num_lines):
            if rng.random() > 0.5:
                # Horizontal path
                y_pos = rng.randint(40, size - 40)
                width = rng.randint(2, 5)
                path_color = (
                    rng.randint(100, 150),
                    rng.randint(90, 130),
                    rng.randint(60, 90),
                )
                cv2.line(img, (0, y_pos), (size, y_pos), path_color, width)
            else:
                # Vertical path
                x_pos = rng.randint(40, size - 40)
                width = rng.randint(2, 5)
                path_color = (
                    rng.randint(100, 150),
                    rng.randint(90, 130),
                    rng.randint(60, 90),
                )
                cv2.line(img, (x_pos, 0), (x_pos, size), path_color, width)

    # ── 7. WATER / IRRIGATION (occasional blue tint) ────────────────
    if rng.random() > 0.7 and health > 0.5:
        # Small water body or irrigation canal
        cx = rng.randint(30, size - 30)
        cy = rng.randint(30, size - 30)
        rx = rng.randint(10, 30)
        ry = rng.randint(5, 15)
        water_color = (rng.randint(80, 120), rng.randint(100, 140), rng.randint(140, 190))
        overlay = img.copy()
        cv2.ellipse(overlay, (cx, cy), (rx, ry), rng.randint(0, 180),
                    0, 360, water_color, -1)
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    # ── 8. FINAL SMOOTHING ──────────────────────────────────────────
    # Light Gaussian blur for a natural aerial look
    kernel = rng.choice([3, 5])
    img = cv2.GaussianBlur(img, (kernel, kernel), 0)

    # Convert RGB → BGR for OpenCV saving
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img_bgr


# ============================================================================
# 4. MAIN PIPELINE
# ============================================================================
def main():
    print("=" * 60)
    print("  CSV -> Crop Field Images Converter")
    print("=" * 60)

    # Step 1: Find CSVs
    print(f"\n{'='*60}")
    print("  Step 1: Locating CSV Files")
    print(f"{'='*60}")
    csv_files = find_csv_files()

    # Step 2: Load data
    print(f"\n{'='*60}")
    print("  Step 2: Loading CSV Data")
    print(f"{'='*60}")
    df = load_and_parse_csvs(csv_files)

    # Step 3: Extract yield records
    print(f"\n{'='*60}")
    print("  Step 3: Extracting Yield Records")
    print(f"{'='*60}")
    records = extract_yield_records(df)

    if not records:
        print("  [Error] No valid records found!")
        sys.exit(1)

    # Limit if needed
    if MAX_IMAGES and len(records) > MAX_IMAGES:
        np.random.seed(42)
        indices = np.random.choice(len(records), MAX_IMAGES, replace=False)
        records = [records[i] for i in sorted(indices)]
        print(f"  [Info] Sampled {MAX_IMAGES} records")

    # Step 4: Generate crop field images
    print(f"\n{'='*60}")
    print(f"  Step 4: Generating {len(records)} Crop Field Images")
    print(f"{'='*60}")

    # Clear old images
    os.makedirs(IMAGES_DIR, exist_ok=True)
    old_files = glob.glob(os.path.join(IMAGES_DIR, "crop_*.png"))
    for f in old_files:
        os.remove(f)
    if old_files:
        print(f"  [Info] Removed {len(old_files)} old images")

    csv_rows = []
    for i, rec in enumerate(records):
        img = generate_crop_field_image(
            yield_value=rec['target_yield'],
            mean_yield=rec['mean_yield'],
            seed=i * 7 + 13,  # deterministic but varied
        )

        filename = f"crop_{i:05d}.png"
        cv2.imwrite(os.path.join(IMAGES_DIR, filename), img)
        csv_rows.append({'image_name': filename, 'yield': rec['target_yield']})

        if (i + 1) % 200 == 0 or i == 0:
            print(f"    [{i+1:>5}/{len(records)}]  yield={rec['target_yield']:>6.2f} t/ha"
                  f"  | {rec['area']} - {rec['item']}")

    # Step 5: Save yield.csv
    print(f"\n{'='*60}")
    print("  Step 5: Saving yield.csv")
    print(f"{'='*60}")

    yield_df = pd.DataFrame(csv_rows)
    yield_df.to_csv(YIELD_CSV_PATH, index=False)

    print(f"  [*] Images: {len(csv_rows)} saved to {IMAGES_DIR}/")
    print(f"  [*] Labels: {YIELD_CSV_PATH}")
    print(f"  [*] Yield range: [{yield_df['yield'].min():.2f}, "
          f"{yield_df['yield'].max():.2f}] tons/hectare")
    print(f"  [*] Mean yield:  {yield_df['yield'].mean():.2f} tons/hectare")

    # Done
    print(f"\n{'='*60}")
    print("  [Success] Done! Your dataset is ready.")
    print(f"{'='*60}")
    print(f"  Next: Run 'python train.py' to train the model!")
    print()


if __name__ == "__main__":
    main()
