import os
import sys
import glob

# Paths
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
YIELD_CSV_PATH = os.path.join(DATASET_DIR, "yield.csv")

# 5 images to keep
KEEP_IMAGES = {
    "crop_00000.png",
    "crop_00010.png",
    "crop_00016.png",
    "crop_00628.png",
    "crop_00637.png"
}

def main():
    print("=== Cleaning Dataset Folder ===")
    
    # 1. Check images dir
    if not os.path.exists(IMAGES_DIR):
        print(f"Directory not found: {IMAGES_DIR}")
        return
        
    # 2. List all files
    all_images = glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    print(f"Found {len(all_images)} images total.")
    
    # 3. Delete non-keep images
    deleted_count = 0
    for img_path in all_images:
        filename = os.path.basename(img_path)
        if filename not in KEEP_IMAGES:
            try:
                os.remove(img_path)
                deleted_count += 1
            except Exception as e:
                print(f"Error removing {filename}: {e}")
                
    print(f"Successfully deleted {deleted_count} other images.")
    
    # 4. Overwrite yield.csv
    csv_content = """image_name,yield
crop_00000.png,2.05
crop_00010.png,26.04
crop_00016.png,1.10
crop_00628.png,4.50
crop_00637.png,25.18
"""
    try:
        with open(YIELD_CSV_PATH, "w", newline="") as f:
            f.write(csv_content)
        print("Successfully updated dataset/yield.csv!")
    except Exception as e:
        print(f"Error writing yield.csv: {e}")
        
    print("=== Clean Complete! ===")

if __name__ == "__main__":
    main()
