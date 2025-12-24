import os
import json
import shutil
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

# --- Configuration ---
UNLABELED_FEATURES_DIR = r".\squat\extracted_features" # Your 6000 files
UNLABELED_VIDEOS_DIR = r".\squat\dataset_videos_all"        # Your raw videos

# Output
SUGGESTED_DIR = r".\squat\suggested_training_data"

# Model Paths (V4)
MODEL_PKL = r".\squat\view_classifier\view_classifier_v4_ensemble.pkl"
ENCODER_PKL = r".\squat\view_classifier\label_encoder_v4.pkl"
SCALER_PKL = r".\squat\view_classifier\scaler_v4.pkl"

# Goal: How many NEW videos do you want to verify per class?
TARGET_PER_CLASS = 50 

# --- Feature Logic (MUST MATCH V4 EXACTLY) ---
def calculate_frame_features(lm_norm, lm_world):
    # Indices
    NOSE = 0
    L_SHOULDER, R_SHOULDER = 11, 12
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_HEEL, R_HEEL = 29, 30
    L_TOE, R_TOE = 31, 32

    # A. Normalization
    mid_sh_y = (lm_norm[L_SHOULDER][1] + lm_norm[R_SHOULDER][1]) / 2
    mid_hip_y = (lm_norm[L_HIP][1] + lm_norm[R_HIP][1]) / 2
    torso_h = abs(mid_sh_y - mid_hip_y)
    if torso_h < 0.01: torso_h = 1.0

    # B. Ratios
    sh_width = abs(lm_norm[L_SHOULDER][0] - lm_norm[R_SHOULDER][0]) / torso_h
    hip_width = abs(lm_norm[L_HIP][0] - lm_norm[R_HIP][0]) / torso_h
    
    # C. Aspect Ratio
    xs = [lm_norm[i][0] for i in [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE]]
    ys = [lm_norm[i][1] for i in [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE]]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0

    # D. Symmetry
    sh_sym = abs(lm_norm[L_SHOULDER][3] - lm_norm[R_SHOULDER][3])
    hip_sym = abs(lm_norm[L_HIP][3] - lm_norm[R_HIP][3])
    knee_sym = abs(lm_norm[L_KNEE][3] - lm_norm[R_KNEE][3])
    total_symmetry = sh_sym + hip_sym + knee_sym

    # E. 3D Depths
    sh_rot_z = abs(lm_world[L_SHOULDER][2] - lm_world[R_SHOULDER][2])
    
    avg_hip_z = (lm_world[L_HIP][2] + lm_world[R_HIP][2]) / 2
    nose_rel_z = lm_world[NOSE][2] - avg_hip_z
    
    avg_heel_z = (lm_world[L_HEEL][2] + lm_world[R_HEEL][2]) / 2
    avg_toe_z = (lm_world[L_TOE][2] + lm_world[R_TOE][2]) / 2
    heel_toe_diff = avg_heel_z - avg_toe_z

    # F. Visibility
    min_sh_vis = min(lm_norm[L_SHOULDER][3], lm_norm[R_SHOULDER][3])
    
    return [
        sh_width, hip_width, aspect_ratio, 
        total_symmetry, 
        sh_rot_z, nose_rel_z, heel_toe_diff,
        min_sh_vis
    ]

def get_features(json_path):
    try:
        with open(json_path, 'r') as f: data = json.load(f)
        kpts_img = data.get("keypoints_img", [])
        kpts_world = data.get("keypoints_world", [])
        if not kpts_img or not kpts_world: return None

        feats_time_series = []
        for i in range(min(len(kpts_img), 60)):
            if kpts_img[i][0][0] == 0.0: continue
            f_feats = calculate_frame_features(kpts_img[i], kpts_world[i])
            feats_time_series.append(f_feats)
        
        if not feats_time_series: return None
        
        # Temporal Stats (Mean + Std)
        arr = np.array(feats_time_series)
        return np.concatenate([np.mean(arr, axis=0), np.std(arr, axis=0)])

    except: return None

# --- Main Logic ---
def main():
    if os.path.exists(SUGGESTED_DIR):
        shutil.rmtree(SUGGESTED_DIR) # Clean previous run
    
    print("--- Loading V4 Ensemble ---")
    model = joblib.load(MODEL_PKL)
    le = joblib.load(ENCODER_PKL)
    scaler = joblib.load(SCALER_PKL)
    classes = le.classes_

    # Create folders
    for cls in classes:
        os.makedirs(os.path.join(SUGGESTED_DIR, cls), exist_ok=True)

    # DEBUG: Check directories
    print("\n--- DEBUG: Checking Paths ---")
    print(f"Features directory: {UNLABELED_FEATURES_DIR}")
    print(f"Directory exists: {os.path.exists(UNLABELED_FEATURES_DIR)}")
    
    if os.path.exists(UNLABELED_FEATURES_DIR):
        # List all files in the directory
        all_files = []
        for root, dirs, files in os.walk(UNLABELED_FEATURES_DIR):
            print(f"Scanning: {root}")
            print(f"  Subdirs: {dirs}")
            print(f"  Files count: {len(files)}")
            if files:
                print(f"  Sample files: {files[:3]}")
            all_files.extend([f for f in files if f.endswith('.json')])
        print(f"\nTotal JSON files found: {len(all_files)}")
    else:
        print("ERROR: Features directory does not exist!")
        return

    print("\n--- Scanning Unlabeled Data (Finding best matches) ---")
    candidates = {cls: [] for cls in classes}
    
    # Get file list
    json_files = []
    for root, dirs, files in os.walk(UNLABELED_FEATURES_DIR):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    
    print(f"Found {len(json_files)} JSON files to process")
    
    if len(json_files) == 0:
        print("ERROR: No JSON files found. Check that:")
        print("  1. You've run '2_extract_features.py' first")
        print("  2. OUTPUT_ROOT in that script matches UNLABELED_FEATURES_DIR here")
        print(f"  3. Files exist in: {os.path.abspath(UNLABELED_FEATURES_DIR)}")
        return
    
    # Limit to random 2500 files to keep it fast
    np.random.shuffle(json_files)
    sample_files = json_files[:2500]

    for json_path in tqdm(sample_files):
        feats = get_features(json_path)
        if feats is None: continue
        
        # SCALE features
        feats_scaled = scaler.transform([feats])
        
        # Predict Probabilities
        probs = model.predict_proba(feats_scaled)[0]
        max_prob = np.max(probs)
        pred_idx = np.argmax(probs)
        pred_label = le.inverse_transform([pred_idx])[0]

        # Only suggest High Confidence matches (> 80%)
        # This filters out the "weird" ones, giving you clean data to add
        if max_prob > 0.80:
            candidates[pred_label].append((max_prob, json_path))

    print("\n--- Moving Videos ---")
    for cls in classes:
        # Sort by confidence
        candidates[cls].sort(key=lambda x: x[0], reverse=True)
        top_picks = candidates[cls][:TARGET_PER_CLASS]
        
        print(f"Suggestion for '{cls}': {len(top_picks)} videos")
        
        for prob, json_path in top_picks:
            vid_name = os.path.basename(json_path).replace(".json", ".mp4")
            
            # Find video file (Helper search)
            found_path = None
            possible = os.path.join(UNLABELED_VIDEOS_DIR, vid_name)
            if os.path.exists(possible):
                found_path = possible
            else:
                for root, _, files in os.walk(UNLABELED_VIDEOS_DIR):
                    if vid_name in files:
                        found_path = os.path.join(root, vid_name)
                        break
            
            if found_path:
                dst = os.path.join(SUGGESTED_DIR, cls, vid_name)
                shutil.copy2(found_path, dst)

    print("\n------------------------------------------------")
    print(f"DONE! Go to folder: {SUGGESTED_DIR}")
    print("1. Open the folders (side, front, etc).")
    print("2. DELETE any video that is clearly wrong.")
    print("3. DRAG the rest into 'squat/training_data_views/'.")
    print("4. RUN '5_train_view_classifier_v4.py' again.")
    print("------------------------------------------------")

if __name__ == "__main__":
    main()