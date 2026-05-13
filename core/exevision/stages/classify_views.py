
import os
import json
import math
import shutil
import sys
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

# --- Configuration ---
# Read from quality-organized folders created by 2.5
FEATURES_EXCELLENT = "./squat/extracted_features_clean/excellent"
FEATURES_GOOD = "./squat/extracted_features_clean/good"
FEATURES_FAIR = "./squat/extracted_features_clean/fair"
FEATURES_RAW_UNFILTERED = "./squat/extracted_features_clean/raw_unfiltered"

FEATURES_DIRS = [FEATURES_EXCELLENT, FEATURES_GOOD, FEATURES_FAIR, FEATURES_RAW_UNFILTERED]


def _build_features_dirs(exercise: str):
    """Build feature directories for the given exercise."""
    tiers = ["excellent", "good", "fair", "raw_unfiltered"]
    return [f"./{exercise}/extracted_features_clean/{tier}" for tier in tiers]


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_NEURAL_MODEL_CACHE = {}


def _try_neural_classify(
    keypoints_img: list,
    face_detected_list: list,
    model_path: str,
    confidence_threshold: float = 0.7,
) -> tuple[str | None, float]:
    """Return (label, confidence) or (None, 0.0) on failure or low confidence."""
    if not model_path or not os.path.exists(model_path):
        return None, 0.0
    try:
        from core.exevision.neural.ohp.view_classifier import load_view_classifier, predict_video

        if model_path not in _NEURAL_MODEL_CACHE:
            _NEURAL_MODEL_CACHE[model_path] = load_view_classifier(model_path)
        model = _NEURAL_MODEL_CACHE[model_path]

        label, conf = predict_video(model, keypoints_img, face_detected_list)
        if label == "unknown" or conf < confidence_threshold:
            return None, conf
        return label, conf
    except Exception:
        return None, 0.0

# BlazePose landmark indices
NOSE = 0
L_EYE = 2
R_EYE = 5
L_EAR = 7
R_EAR = 8
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_HIP = 23
R_HIP = 24

L_ARM_INDICES = (L_ELBOW, L_WRIST)
R_ARM_INDICES = (R_ELBOW, R_WRIST)

# Thresholds
VIS_THRESHOLD = 0.5          # visibility score cutoff (if available)
SIDE_WIDTH = 0.08            # shoulder x-spread below this → side candidate
SIDE_HIP_WIDTH = 0.08        # hip x-spread below this → side candidate (AND-gate with shoulders)
SIDE_GRAY_WIDTH = 0.12       # shoulder x-spread below this + single-arm-visible → side rescue
DIAGONAL_WIDTH = 0.18        # shoulder x-spread below this → diagonal

# Arm visibility thresholds (exercise-agnostic occlusion physics)
ARM_VIS_PRESENT = 0.5        # avg(elbow+wrist) visibility above this → arm clearly visible
ARM_ASYM_PURE = 0.10         # L vs R arm vis diff below this → arms truly symmetric (pure-view rescue)
ARM_ASYM_DIAGONAL = 0.35     # L vs R arm vis diff above this → force diagonal classification
ARM_ASYM_SIDE = 0.40         # L vs R arm vis diff above this (in side gray zone) → side
ARM_ASYM_BACK_DEMOTE = 0.20  # in wide-back zone, asymmetry above this → demote back → back_side
FAR_ARM_VIS_OCCLUDED = 0.20  # far arm vis below this required for side rescue (deep occlusion only)
FACE_OVERRIDE_RATIO = 0.25       # BlazeFace fires on >= this ratio of frames → override back/back_side
FACE_BACK_DEMOTE_RATIO = 0.20    # BlazeFace fires on < this ratio + label is front/front_side → demote

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


def _arm_vis_avg(frame, has_vis: bool, indices) -> float:
    """Mean visibility of arm landmarks (elbow+wrist). Returns 0.0 if vis scores absent."""
    if not has_vis:
        return 0.0
    return sum(frame[i][3] for i in indices) / len(indices)


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

    # --- Signal 2: Shoulder + hip width (x-axis spread) ---
    lsx = frame[L_SHOULDER][0]
    rsx = frame[R_SHOULDER][0]
    shoulder_width = abs(lsx - rsx)

    lhx = frame[L_HIP][0]
    rhx = frame[R_HIP][0]
    hip_width = abs(lhx - rhx)

    # --- Signal 3: Arm visibility (elbow + wrist avg per side) ---
    # Pure back/front: both arms clearly visible (symmetric occlusion).
    # Diagonal (back_side/front_side): one arm partially occluded by torso (asymmetric).
    # Side: only one arm visible from camera perspective (far arm hidden behind torso).
    l_arm_vis = _arm_vis_avg(frame, has_vis, L_ARM_INDICES)
    r_arm_vis = _arm_vis_avg(frame, has_vis, R_ARM_INDICES)
    arm_asym = abs(l_arm_vis - r_arm_vis)
    both_arms_visible = l_arm_vis > ARM_VIS_PRESENT and r_arm_vis > ARM_VIS_PRESENT
    single_arm_visible = (l_arm_vis > ARM_VIS_PRESENT) ^ (r_arm_vis > ARM_VIS_PRESENT)

    # --- Decision logic ---

    # Strong side: BOTH shoulder + hip collapsed in x (torso = thin vertical slab)
    if shoulder_width < SIDE_WIDTH and hip_width < SIDE_HIP_WIDTH:
        return "side"

    # Shoulder-only collapse with wide hips → NOT side (one shoulder occluded/raised).
    # Fall through to diagonal/face/depth logic instead of false-positive side.
    # (No early return here — control flow continues to diagonal check below.)

    # Side rescue: gray-zone shoulder + hip narrow + single arm DEEPLY occluded + strong asymmetry
    # Extra gates prevent OHP diagonal lockout (far arm behind head) from being mis-classified as side:
    #   - face_detected=False: BlazeFace fires only on front-facing faces; side profile rarely detected
    #   - far_arm_vis < FAR_ARM_VIS_OCCLUDED: pure side fully occludes far arm; diagonals only drop vis temporarily
    far_arm_vis = min(l_arm_vis, r_arm_vis)
    if (has_vis
            and not face_detected
            and shoulder_width < SIDE_GRAY_WIDTH
            and hip_width < SIDE_GRAY_WIDTH
            and single_arm_visible
            and arm_asym > ARM_ASYM_SIDE
            and far_arm_vis < FAR_ARM_VIS_OCCLUDED):
        return "side"

    # Depth signal: True=facing camera, False=back to camera, None=unreliable
    depth_forward = _facing_camera(frame, has_vis)

    # Diagonal zone: shoulder narrow OR arms strongly asymmetric (forces diagonal even at wide shoulder)
    in_diagonal = shoulder_width < DIAGONAL_WIDTH or (has_vis and arm_asym > ARM_ASYM_DIAGONAL)

    if in_diagonal:
        # Pure-view rescue: arms truly symmetric + both visible + depth definitive → really pure view.
        # Diagonals always occlude one arm partially (asym > ARM_ASYM_PURE); symmetric arms means
        # subject just narrower than DIAGONAL_WIDTH threshold but body genuinely faces camera.
        if (has_vis and both_arms_visible and arm_asym < ARM_ASYM_PURE
                and depth_forward is not None):
            return "front" if depth_forward else "back"

        if depth_forward is True:
            return "front_side"
        if depth_forward is False:
            return "back_side"
        # depth ambiguous (None): fall back to face signals
        if face_detected:
            return "front_side"
        if any_face:
            return "front_side"
        return "back_side"

    # Pure front vs back (wide-shoulder + symmetric arms)
    if depth_forward is True:
        return "front"
    if depth_forward is False:
        # Demote pure back → back_side when one arm clearly more occluded than the other
        # (pure back view shows both arms; asymmetry indicates rotation off-axis)
        if has_vis and not both_arms_visible and arm_asym > ARM_ASYM_BACK_DEMOTE:
            return "back_side"
        return "back"
    if face_detected:
        return "front"
    if any_face:
        return "front"
    # No depth, no face: arms decide back vs back_side
    if has_vis and not both_arms_visible and arm_asym > ARM_ASYM_BACK_DEMOTE:
        return "back_side"
    return "back"


def get_view_label(keypoints_img: list, face_detected_list: list = None) -> str:
    """Classify camera view. Returns the view with the most frame votes."""
    label, _ = get_view_label_with_probs(keypoints_img, face_detected_list)
    return label


def get_view_label_with_probs(keypoints_img: list, face_detected_list: list = None) -> tuple[str, dict]:
    """Classify with full vote distribution."""
    votes = []
    forward_depth_count = 0   # frames with nose definitively in front of hips
    backward_depth_count = 0  # frames with nose definitively behind hips
    face_detected_count = 0   # frames where BlazeFace fired
    valid_face_frames = 0     # frames with usable BlazeFace signal

    for idx, frame in enumerate(keypoints_img or []):
        if len(votes) >= MAX_FRAMES:
            break

        is_face_detected = face_detected_list[idx] if (face_detected_list and idx < len(face_detected_list)) else False
        label = _classify_frame(frame, face_detected=is_face_detected)
        if label:
            votes.append(label)

        # Track depth signal per frame (count forward vs backward, not just any-forward).
        if frame and len(frame) > 24:
            has_vis = len(frame[0]) > 3
            depth = _facing_camera(frame, has_vis)
            if depth is True:
                forward_depth_count += 1
            elif depth is False:
                backward_depth_count += 1

        # Track BlazeFace fire rate across the video
        if face_detected_list and idx < len(face_detected_list):
            valid_face_frames += 1
            if face_detected_list[idx]:
                face_detected_count += 1

    if not votes:
        return "unknown", {"front": 0.0, "back": 0.0, "side": 0.0, "front_side": 0.0, "back_side": 0.0}

    counts = Counter(votes)
    total = len(votes)
    probs = {v: round(counts.get(v, 0) / total, 4)
             for v in ["front", "back", "side", "front_side", "back_side"]}

    label = max(probs, key=probs.get)

    # Video-level corrections.
    #
    # BACK → FRONT override (OHP head-tilt fix). Two independent triggers:
    #   1. Forward depth dominates backward depth (depth signal contradicts back classification).
    #   2. BlazeFace fires on >= FACE_OVERRIDE_RATIO of frames. Pure back never sees the face;
    #      strong face presence means subject faces camera even when head-tilt pushes nose
    #      behind hips during OHP lockout.
    #
    # FRONT → BACK demote (depth hallucination fix). MediaPipe sometimes places nose_z forward
    # of hips even from back angles (face landmark hallucination, forward torso lean). When this
    # happens, BlazeFace is the only reliable discriminator. If face barely fires (< 20% of
    # frames), the subject is almost certainly back-facing despite the depth signal.
    forward_dominates = forward_depth_count > backward_depth_count and forward_depth_count > 0
    face_ratio = (face_detected_count / valid_face_frames) if valid_face_frames > 0 else None
    face_dominates = face_ratio is not None and face_ratio >= FACE_OVERRIDE_RATIO
    face_absent = face_ratio is not None and face_ratio < FACE_BACK_DEMOTE_RATIO

    if (forward_dominates or face_dominates) and label == "back_side":
        label = "front_side"
    elif (forward_dominates or face_dominates) and label == "back":
        label = "front"
    elif face_absent and label == "front_side":
        label = "back_side"
    elif face_absent and label == "front":
        label = "back"

    return label, probs

def process_video_classification(
    json_path: str,
    neural_model_path: str = "",
    confidence_threshold: float = 0.7,
) -> tuple:
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
        
        # Classify view (neural first, fallback to heuristic)
        view = None
        neural_used = False
        if neural_model_path:
            neural_label, _conf = _try_neural_classify(
                keypoints_img,
                face_detected_list,
                neural_model_path,
                confidence_threshold,
            )
            if neural_label is not None:
                view = neural_label
                neural_used = True
        if view is None:
            view = get_view_label(keypoints_img, face_detected_list)
        
        # Update JSON with view classification
        if 'info' not in data:
            data['info'] = {}
        data['info']['view'] = view
        data['info']['view_reliable'] = bool(neural_used or view != "unknown")
        
        # Write back to same file
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return video_id, "Success", view
        
    except Exception as e:
        return video_id, "Error", str(e)


def run_classification(
    quality_filter=None,
    video_id_filter=None,
    exercise="squat",
    neural_model_path: str = "",
    confidence_threshold: float = 0.7,
):
    """
    Process all extracted features and classify by view.
    Updates JSON files in place with view field.

    Args:
        quality_filter: None (all), 'excellent', 'good', or 'fair'
        exercise: exercise name (default: 'squat'); controls which directory tree is scanned
    """
    features_dirs = _build_features_dirs(exercise)

    # Determine which folders to process
    if quality_filter:
        folders_to_process = [f for f in features_dirs
                             if quality_filter.lower() in f.lower()]
    else:
        folders_to_process = features_dirs
    
    # Collect all JSON files
    json_files = []
    folder_map = {}  # Track which folder each file came from
    
    for folder in folders_to_process:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(".json"):
                    video_id = os.path.splitext(f)[0]
                    if video_id_filter and video_id != video_id_filter:
                        continue
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
        video_id, status, view = process_video_classification(
            json_path,
            neural_model_path=neural_model_path,
            confidence_threshold=confidence_threshold,
        )
        
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
    import argparse

    parser = argparse.ArgumentParser(description="Classify views from extracted pose features.")
    parser.add_argument("--quality", choices=["excellent", "good", "fair", "raw_unfiltered"], help="Filter by quality")
    parser.add_argument("--video-id", help="Process only one video id (e.g., 25709_1)")
    parser.add_argument("--exercise", default="squat", help="Exercise type (default: squat)")
    parser.add_argument(
        "--neural",
        action="store_true",
        help="Use neural view classifier (fallback to heuristic if low confidence)",
    )
    parser.add_argument(
        "--neural-model",
        default="models/view_classifier_ohp.pt",
        help="Path to view_classifier_ohp.pt",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Neural confidence below this uses heuristic fallback",
    )
    args = parser.parse_args()

    run_classification(
        quality_filter=args.quality,
        video_id_filter=args.video_id,
        exercise=args.exercise,
        neural_model_path=args.neural_model if args.neural else "",
        confidence_threshold=args.confidence_threshold,
    )
