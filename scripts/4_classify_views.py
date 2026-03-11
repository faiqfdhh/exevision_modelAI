
import os
import json
import math
import shutil
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

# --- Configuration ---
# Read from quality-organized folders created by 2.5
FEATURES_EXCELLENT = "./squat/extracted_features_clean/excellent"
FEATURES_GOOD = "./squat/extracted_features_clean/good"
FEATURES_FAIR = "./squat/extracted_features_clean/fair"

FEATURES_DIRS = [FEATURES_EXCELLENT, FEATURES_GOOD, FEATURES_FAIR]

# BlazePose landmark indices
NOSE = 0
L_EYE = 2
R_EYE = 5
L_EAR = 7
R_EAR = 8
L_HIP = 23
R_HIP = 24
L_SHOULDER = 11
R_SHOULDER = 12

# Thresholds
VIS_THRESHOLD = 0.5          # visibility score cutoff (if available)
SIDE_WIDTH = 0.08            # shoulder x-spread below this → side
DIAGONAL_WIDTH = 0.18        # shoulder x-spread below this → diagonal

MAX_FRAMES = 60


def _is_visible(landmark, has_vis: bool) -> bool:
    """Check if a landmark is visible."""
    if has_vis:
        return landmark[3] > VIS_THRESHOLD
    return not (landmark[0] == 0.0 and landmark[1] == 0.0)


def _face_score(frame, has_vis: bool) -> float:
    """Sum of nose + eye confidences. High = facing camera."""
    if not has_vis:
        return 0.0
    return frame[NOSE][3] + frame[L_EYE][3] + frame[R_EYE][3]


def _facing_camera(frame) -> bool:
    """Use nose-vs-hip depth to determine whether the torso faces the camera."""
    nose_z = frame[NOSE][2]
    hip_z = (frame[L_HIP][2] + frame[R_HIP][2]) / 2.0
    return nose_z < hip_z


def _classify_frame(frame) -> str | None:
    """Classify a single frame. Returns a view label or None if unusable."""
    if not frame or len(frame) < 25:
        return None

    has_vis = len(frame[0]) > 3

    # --- Signal 1: Face visibility ---
    nose_vis = _is_visible(frame[NOSE], has_vis)
    l_eye_vis = _is_visible(frame[L_EYE], has_vis)
    r_eye_vis = _is_visible(frame[R_EYE], has_vis)
    l_ear_vis = _is_visible(frame[L_EAR], has_vis)
    r_ear_vis = _is_visible(frame[R_EAR], has_vis)

    any_face = nose_vis or l_eye_vis or r_eye_vis
    both_eyes = l_eye_vis and r_eye_vis

    # --- Signal 2: Shoulder width (x-axis spread) ---
    lsx = frame[L_SHOULDER][0]
    rsx = frame[R_SHOULDER][0]
    shoulder_width = abs(lsx - rsx)

    # --- Decision logic ---

    # Side: shoulders nearly collapsed in x
    if shoulder_width < SIDE_WIDTH:
        return "side"

    # Back or diagonal front: can't see any face landmarks
    if not any_face:
        if shoulder_width < DIAGONAL_WIDTH:
            return "front_side" if _facing_camera(frame) else "back_side"
        return "back"

    # Front: both eyes clearly visible + wide shoulders
    if both_eyes:
        if shoulder_width < DIAGONAL_WIDTH:
            return "front_side" if _facing_camera(frame) else "back_side"
        return "front"

    # Remaining non-side, non-pure-front frames are diagonal.
    return "front_side" if _facing_camera(frame) else "back_side"


def get_view_label(keypoints_img: list) -> str:
    """Classify camera view. Returns the view with the most frame votes."""
    label, _ = get_view_label_with_probs(keypoints_img)
    return label


def get_view_label_with_probs(keypoints_img: list) -> tuple[str, dict]:
    """Classify with full vote distribution."""
    votes = []

    for frame in (keypoints_img or []):
        if len(votes) >= MAX_FRAMES:
            break
        label = _classify_frame(frame)
        if label:
            votes.append(label)

    if not votes:
        return "unknown", {"front": 0.0, "back": 0.0, "side": 0.0, "front_side": 0.0, "back_side": 0.0}

    counts = Counter(votes)
    total = len(votes)
    probs = {v: round(counts.get(v, 0) / total, 4)
             for v in ["front", "back", "side", "front_side", "back_side"]}

    label = max(probs, key=probs.get)
    return label, probs

def process_video_classification(json_path: str) -> tuple:
    """
    Read JSON, classify view, update JSON, return result
    """
    video_id = os.path.splitext(os.path.basename(json_path))[0]
    
    try:
        # Read JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get keypoints
        keypoints_img = data.get('keypoints_img', [])
        
        # Classify view
        view = get_view_label(keypoints_img)
        
        # Update JSON with view classification
        if 'info' not in data:
            data['info'] = {}
        data['info']['view'] = view
        
        # Write back to same file
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return video_id, "Success", view
        
    except Exception as e:
        return video_id, "Error", str(e)


def run_classification(quality_filter=None):
    """
    Process all extracted features and classify by view.
    Updates JSON files in place with view field.
    
    Args:
        quality_filter: None (all), 'excellent', 'good', or 'fair'
    """
    # Determine which folders to process
    if quality_filter:
        folders_to_process = [f for f in FEATURES_DIRS 
                             if quality_filter.lower() in f.lower()]
    else:
        folders_to_process = FEATURES_DIRS
    
    # Collect all JSON files
    json_files = []
    folder_map = {}  # Track which folder each file came from
    
    for folder in folders_to_process:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(".json"):
                    full_path = os.path.join(folder, f)
                    json_files.append(full_path)
                    folder_map[full_path] = folder
    
    if not json_files:
        print(f"❌ No videos found in {folders_to_process}")
        return
    
    print(f"\n🎥 Classifying {len(json_files)} videos by view...")
    print(f"📁 Processing from: {', '.join([os.path.basename(f) for f in folders_to_process])}\n")
    
    # Process classifications
    stats = {
        "front": 0,
        "back": 0,
        "side": 0,
        "front_side": 0,
        "back_side": 0,
        "unknown": 0,
        "error": 0
    }
    
    results = []
    
    for json_path in tqdm(json_files, desc="Classifying videos"):
        video_id, status, view = process_video_classification(json_path)
        
        if status == "Success":
            stats[view] += 1
            results.append((video_id, view, folder_map[json_path]))
            if view == "unknown":
                print(f"\n   ⚠ {video_id} classified as: {view} (May be skipped by later stages)")
            else:
                print(f"\n   ✓ {video_id} classified as: {view}")
        else:
            stats["error"] += 1
            print(f"\n  ✗ {video_id} failed: {view}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("CLASSIFICATION SUMMARY")
    print('='*70)
    for view in ["front", "back", "side", "front_side", "back_side", "unknown"]:
        count = stats[view]
        if count > 0:
            print(f"  {view.ljust(12)}: {count}")
    if stats["error"] > 0:
        print(f"  {'errors'.ljust(12)}: {stats['error']}")
    
    print(f"\n✅ All JSON files updated with 'view' field")
    print(f"📁 Updated in: {', '.join([os.path.basename(f) for f in folders_to_process])}")
print('='*70)


if __name__ == "__main__":
    # Run classification on all videos
    run_classification(quality_filter=None)
    
    # Or run on specific quality:
    # run_classification(quality_filter='good')
