import json
import os
import shutil
from tqdm import tqdm

# --- Configuration ---
SOURCE = "squat\dataset_videos_all"

# Where we will put the clean data
DEST_ROOT = "squat/dataset_videos_processed"

SPLIT_FILES = {
    "train_keys.json": "train",
    "test_keys.json":  "test",
    "val_keys.json":   "val"
}

def load_keys(json_path):
    if not os.path.exists(json_path):
        print(f"Missing {json_path}")
        return []
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data if isinstance(data, list) else list(data.keys())

def organize():
    # 1. Organize Labeled Data (Strict Splits)
    print("--- Organizing Labeled Data ---")
    for json_file, folder in SPLIT_FILES.items():
        keys = load_keys(os.path.join(SPLITS_DIR, json_file))
        dest_dir = os.path.join(DEST_ROOT, "labeled", folder)
        os.makedirs(dest_dir, exist_ok=True)
        
        print(f"Processing {folder} ({len(keys)} videos)...")
        for vid_id in tqdm(keys):
            # Find the file (sometimes extensions vary or are missing in keys)
            filename = f"{vid_id}.mp4"
            src = os.path.join(LABELED_SOURCE, filename)
            
            if not os.path.exists(src):
                # Fallback search if exact match fails
                candidates = [f for f in os.listdir(LABELED_SOURCE) if f.startswith(vid_id)]
                if candidates: src = os.path.join(LABELED_SOURCE, candidates[0])
                else: continue
            
            shutil.copy2(src, os.path.join(dest_dir, os.path.basename(src)))

    # 2. Organize Unlabeled Data (Single Pool)
    print("\n--- Organizing Unlabeled Data ---")
    dest_dir = os.path.join(DEST_ROOT, "unlabeled")
    os.makedirs(dest_dir, exist_ok=True)
    
    # We only copy videos that actually exist
    if os.path.exists(UNLABELED_SOURCE):
        videos = [f for f in os.listdir(UNLABELED_SOURCE) if f.endswith(".mp4")]
        print(f"Copying {len(videos)} unlabeled videos...")
        for vid in tqdm(videos):
            shutil.copy2(os.path.join(UNLABELED_SOURCE, vid), os.path.join(dest_dir, vid))

if __name__ == "__main__":
    organize()