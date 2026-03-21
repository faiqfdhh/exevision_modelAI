
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
FEATURES_RAW_UNFILTERED = r"D:\squat\unlabeled_features\raw_unfiltered"

FEATURES_DIRS = [FEATURES_EXCELLENT, FEATURES_GOOD, FEATURES_FAIR, FEATURES_RAW_UNFILTERED]

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


def _facing_camera(frame, has_vis: bool) -> bool | None:
    """Use nose-vs-hip Z-depth to determine whether the torso faces the camera.

    Returns True (facing), False (back), or None when the nose is too unreliable
    to produce a meaningful depth comparison (zeroed by filter, low visibility, or
    the depth difference is within noise range).
    """
    # Reject zeroed nose (stability filter set all coords to 0.0)
    nx, ny = frame[NOSE][0], frame[NOSE][1]
    if abs(nx) < 1e-5 and abs(ny) < 1e-5:
        return None
    # Reject low-visibility nose
    if has_vis and frame[NOSE][3] < 0.25:
        return None
    nose_z = frame[NOSE][2]
    hip_z = (frame[L_HIP][2] + frame[R_HIP][2]) / 2.0
    # Require a meaningful depth gap (anything smaller is noise in image-coord z)
    if abs(nose_z - hip_z) < 0.002:
        return None
    return nose_z < hip_z  # nose closer to camera → person faces camera


def _classify_frame(frame, face_detected: bool = False) -> str | None:
    """Classify a single frame. Returns a view label or None if unusable."""
    if not frame or len(frame) < 25:
        return None

    has_vis = len(frame[0]) > 3

    # --- Signal 1: Face visibility ---
    # With Face Detector Integration, we prioritize the independent face detector signal.
    # The heuristic landmark checks are kept as fallback if face detector is disabled.
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

    # Depth signal: True=facing camera, False=back to camera, None=unreliable
    depth_forward = _facing_camera(frame, has_vis)

    if shoulder_width < DIAGONAL_WIDTH:
        # Priority 1: reliable nose depth
        if depth_forward is True:
            return "front_side"
        if depth_forward is False:
            return "back_side"
        # Priority 2 (depth ambiguous): external face detector
        if face_detected:
            return "front_side"
        # Priority 3: MediaPipe face landmark visibility (nose/eyes still visible
        # even when their position is unreliable)
        if any_face:
            return "front_side"
        # Default: no facing signal available
        return "back_side"

    # Pure front vs back (wide-shoulder zone)
    if depth_forward is True:
        return "front"
    if depth_forward is False:
        return "back"
    if face_detected:
        return "front"
    if any_face:
        return "front"
    return "back"


def get_view_label(keypoints_img: list, face_detected_list: list = None) -> str:
    """Classify camera view. Returns the view with the most frame votes."""
    label, _ = get_view_label_with_probs(keypoints_img, face_detected_list)
    return label


def get_view_label_with_probs(keypoints_img: list, face_detected_list: list = None) -> tuple[str, dict]:
    """Classify with full vote distribution."""
    votes = []

    for idx, frame in enumerate(keypoints_img or []):
        if len(votes) >= MAX_FRAMES:
            break
        
        is_face_detected = face_detected_list[idx] if (face_detected_list and idx < len(face_detected_list)) else False
        label = _classify_frame(frame, face_detected=is_face_detected)
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
        
        # Get keypoints and features
        keypoints_img = data.get('keypoints_img', [])
        face_detected_list = data.get('face_detected', [])
        
        # Classify view
        view = get_view_label(keypoints_img, face_detected_list)
        
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
