"""
squat_recognition_functions.py
Squat recognition and classification microprograms.

Functions for identifying squat characteristics:
- View classification (front/back/side)
- Squat depth detection
- Phase detection helpers
- Body scale factor calculation

Compatible with: 2.5_extract_selected_features.py output (MediaPipe 33 landmarks)
"""

import numpy as np
import math
from typing import Optional, List, Dict, Tuple
from enum import Enum

# MediaPipe landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32


class SquatPhase(Enum):
    """Squat movement phases"""
    IDLE = "idle"
    ECCENTRIC = "eccentric"      # Descending
    ISOMETRIC = "isometric"      # Holding at bottom
    CONCENTRIC = "concentric"    # Ascending
    UNKNOWN = "unknown"


class ViewType(Enum):
    """Camera view types"""
    FRONT = "front"
    BACK = "back"
    SIDE = "side"
    FRONT_SIDE = "front_side"
    BACK_SIDE = "back_side"
    UNKNOWN = "unknown"


# =============================================================================
# SCALE FACTOR FUNCTIONS (for normalization)
# =============================================================================

def get_scale_factor(squat_data: Dict) -> float:
    """
    Calculate body scale factor for normalizing measurements.
    Uses torso length (shoulder to hip) as reference.
    
    Similar to dive's get_scale_factor but adapted for squat data structure.
    
    Args:
        squat_data: Dictionary with 'keypoints_img' from 2.5 script output
        
    Returns:
        Median torso length for normalization
    """
    keypoints = squat_data.get('keypoints_img', [])
    
    if not keypoints:
        return 1.0
    
    torso_lengths = []
    
    for frame in keypoints:
        if frame is None or len(frame) < 25:
            continue
        
        # Get shoulder and hip landmarks
        l_shoulder = frame[L_SHOULDER]
        r_shoulder = frame[R_SHOULDER]
        l_hip = frame[L_HIP]
        r_hip = frame[R_HIP]
        
        # Check confidence
        if min(l_shoulder[3], r_shoulder[3], l_hip[3], r_hip[3]) < 0.4:
            continue
        
        # Calculate midpoints
        shoulder_mid = np.array([(l_shoulder[0] + r_shoulder[0]) / 2,
                                 (l_shoulder[1] + r_shoulder[1]) / 2])
        hip_mid = np.array([(l_hip[0] + r_hip[0]) / 2,
                            (l_hip[1] + r_hip[1]) / 2])
        
        # Torso length
        torso_len = np.linalg.norm(shoulder_mid - hip_mid)
        if torso_len > 0.01:  # Sanity check
            torso_lengths.append(torso_len)
    
    if not torso_lengths:
        return 1.0
    
    return float(np.median(torso_lengths))


def get_leg_length(squat_data: Dict) -> float:
    """
    Calculate leg length (hip to ankle) for depth normalization.
    
    Returns:
        Median leg length
    """
    keypoints = squat_data.get('keypoints_img', [])
    
    if not keypoints:
        return 1.0
    
    leg_lengths = []
    
    for frame in keypoints:
        if frame is None or len(frame) < 28:
            continue
        
        l_hip = frame[L_HIP]
        l_ankle = frame[L_ANKLE]
        r_hip = frame[R_HIP]
        r_ankle = frame[R_ANKLE]
        
        if min(l_hip[3], l_ankle[3]) >= 0.4:
            l_leg = math.dist([l_hip[0], l_hip[1]], [l_ankle[0], l_ankle[1]])
            leg_lengths.append(l_leg)
        
        if min(r_hip[3], r_ankle[3]) >= 0.4:
            r_leg = math.dist([r_hip[0], r_hip[1]], [r_ankle[0], r_ankle[1]])
            leg_lengths.append(r_leg)
    
    if not leg_lengths:
        return 1.0
    
    return float(np.median(leg_lengths))


# =============================================================================
# VIEW CLASSIFICATION (from 4_classify_views.py)
# =============================================================================

# Thresholds (from your existing 4_classify_views.py)
PURE_SIDE_WIDTH = 0.08
ROTATION_THRESHOLD = 0.15
NOSE_Z_THRESHOLD = 0.05


def classify_view(keypoints_img: List) -> Tuple[ViewType, Dict[str, float]]:
    """
    Classify camera view from keypoints.
    
    Replicates logic from 4_classify_views.py.
    
    Args:
        keypoints_img: List of frames, each with 33 landmarks
        
    Returns:
        Tuple of (ViewType, metrics_dict)
    """
    if not keypoints_img or len(keypoints_img) < 5:
        return ViewType.UNKNOWN, {}
    
    valid_frames = 0
    total_shoulder_width = 0
    total_shoulder_depth = 0
    total_nose_hip_diff = 0
    
    frames_to_check = min(len(keypoints_img), 60)
    
    for i in range(frames_to_check):
        frame = keypoints_img[i]
        
        if not frame or len(frame) < 25:
            continue
        
        if frame[0][0] == 0.0:  # Skip empty frames
            continue
        
        # 1. Width (X-axis) - shoulder separation
        l_shoulder_x = frame[L_SHOULDER][0]
        r_shoulder_x = frame[R_SHOULDER][0]
        width = abs(l_shoulder_x - r_shoulder_x)
        
        # 2. Rotation (Shoulder Z-axis diff)
        l_shoulder_z = frame[L_SHOULDER][2] if len(frame[L_SHOULDER]) > 2 else 0
        r_shoulder_z = frame[R_SHOULDER][2] if len(frame[R_SHOULDER]) > 2 else 0
        depth = abs(l_shoulder_z - r_shoulder_z)
        
        # 3. Front/Back Indicator (Nose Z relative to Hips)
        nose_z = frame[NOSE][2] if len(frame[NOSE]) > 2 else 0
        l_hip_z = frame[L_HIP][2] if len(frame[L_HIP]) > 2 else 0
        r_hip_z = frame[R_HIP][2] if len(frame[R_HIP]) > 2 else 0
        avg_hip_z = (l_hip_z + r_hip_z) / 2.0
        
        nose_rel_z = nose_z - avg_hip_z
        
        total_shoulder_width += width
        total_shoulder_depth += depth
        total_nose_hip_diff += nose_rel_z
        valid_frames += 1
    
    if valid_frames < 5:
        return ViewType.UNKNOWN, {}
    
    avg_width = total_shoulder_width / valid_frames
    avg_rot = total_shoulder_depth / valid_frames
    avg_nose_diff = total_nose_hip_diff / valid_frames
    
    metrics = {
        'avg_shoulder_width': avg_width,
        'avg_shoulder_rotation': avg_rot,
        'avg_nose_hip_diff': avg_nose_diff,
        'valid_frames': valid_frames,
    }
    
    # Classification logic (from 4_classify_views.py)
    if avg_width < PURE_SIDE_WIDTH:
        return ViewType.SIDE, metrics
    
    if avg_nose_diff > NOSE_Z_THRESHOLD:
        facing = "back"
    else:
        facing = "front"
    
    if avg_rot > ROTATION_THRESHOLD:
        view_str = f"{facing}_side"
    else:
        view_str = facing
    
    view_type = ViewType(view_str)
    return view_type, metrics


def get_view_label(keypoints_img: List) -> str:
    """
    Convenience function returning view as string.
    Compatible with existing code.
    """
    view_type, _ = classify_view(keypoints_img)
    return view_type.value


# =============================================================================
# SQUAT DEPTH FUNCTIONS
# =============================================================================

def get_hip_height(frame: List) -> Optional[float]:
    """Get normalized hip height (Y coordinate) from frame."""
    if frame is None or len(frame) < 25:
        return None
    
    l_hip = frame[L_HIP]
    r_hip = frame[R_HIP]
    
    if min(l_hip[3], r_hip[3]) < 0.4:
        return None
    
    return (l_hip[1] + r_hip[1]) / 2


def get_knee_angle(frame: List) -> Optional[float]:
    """
    Calculate average knee angle from frame.
    
    Returns:
        Knee angle in degrees (180 = straight, ~90 = deep squat)
    """
    if frame is None or len(frame) < 28:
        return None
    
    angles = []
    
    # Left knee
    l_hip = frame[L_HIP]
    l_knee = frame[L_KNEE]
    l_ankle = frame[L_ANKLE]
    
    if min(l_hip[3], l_knee[3], l_ankle[3]) >= 0.4:
        v1 = np.array([l_hip[0] - l_knee[0], l_hip[1] - l_knee[1]])
        v2 = np.array([l_ankle[0] - l_knee[0], l_ankle[1] - l_knee[1]])
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 1e-6 and norm2 > 1e-6:
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angles.append(math.degrees(np.arccos(cos_angle)))
    
    # Right knee
    r_hip = frame[R_HIP]
    r_knee = frame[R_KNEE]
    r_ankle = frame[R_ANKLE]
    
    if min(r_hip[3], r_knee[3], r_ankle[3]) >= 0.4:
        v1 = np.array([r_hip[0] - r_knee[0], r_hip[1] - r_knee[1]])
        v2 = np.array([r_ankle[0] - r_knee[0], r_ankle[1] - r_knee[1]])
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 > 1e-6 and norm2 > 1e-6:
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angles.append(math.degrees(np.arccos(cos_angle)))
    
    if not angles:
        return None
    
    return float(np.mean(angles))


def detect_squat_depth_category(min_knee_angle: float) -> str:
    """
    Categorize squat depth based on minimum knee angle achieved.
    
    Args:
        min_knee_angle: Minimum knee angle during squat (degrees)
        
    Returns:
        Category string: 'parallel', 'below_parallel', 'above_parallel', 'quarter'
    """
    if min_knee_angle <= 70:
        return "below_parallel"  # ATG (ass-to-grass)
    elif min_knee_angle <= 90:
        return "parallel"
    elif min_knee_angle <= 110:
        return "above_parallel"
    else:
        return "quarter"


def compute_hip_displacement_signal(keypoints_img: List) -> np.ndarray:
    """
    Compute normalized hip displacement for all frames.
    Primary signal for phase detection.
    
    Returns:
        Array of normalized hip heights (0 = standing, positive = squatting)
    """
    hip_heights = []
    
    for frame in keypoints_img:
        height = get_hip_height(frame)
        hip_heights.append(height if height is not None else np.nan)
    
    hip_heights = np.array(hip_heights)
    
    # Interpolate NaN values
    nans = np.isnan(hip_heights)
    if not nans.all() and nans.any():
        indices = np.arange(len(hip_heights))
        hip_heights[nans] = np.interp(indices[nans], indices[~nans], hip_heights[~nans])
    
    # Normalize: 0 = minimum (standing), positive = lower (squatting)
    if not np.isnan(hip_heights).all():
        standing_height = np.percentile(hip_heights[~np.isnan(hip_heights)], 10)
        hip_heights = hip_heights - standing_height
    
    return hip_heights


def compute_knee_angle_signal(keypoints_img: List) -> np.ndarray:
    """
    Compute knee angle for all frames.
    Secondary signal for phase detection.
    """
    angles = []
    
    for frame in keypoints_img:
        angle = get_knee_angle(frame)
        angles.append(angle if angle is not None else np.nan)
    
    angles = np.array(angles)
    
    # Interpolate NaN values
    nans = np.isnan(angles)
    if not nans.all() and nans.any():
        indices = np.arange(len(angles))
        angles[nans] = np.interp(indices[nans], indices[~nans], angles[~nans])
    
    return angles


# =============================================================================
# PHASE DETECTION MICROPROGRAMS
# =============================================================================

def is_standing(frame: List, standing_threshold: float = 150.0) -> bool:
    """
    Check if person is in standing position.
    
    Args:
        frame: Single keypoint frame
        standing_threshold: Knee angle threshold for standing (degrees)
        
    Returns:
        True if standing (knee angle > threshold)
    """
    knee_angle = get_knee_angle(frame)
    if knee_angle is None:
        return True  # Assume standing if can't determine
    
    return knee_angle > standing_threshold


def is_at_bottom(frame: List, bottom_threshold: float = 110.0) -> bool:
    """
    Check if person is at bottom of squat.
    
    Args:
        frame: Single keypoint frame
        bottom_threshold: Knee angle threshold for bottom position
        
    Returns:
        True if at squat bottom (knee angle < threshold)
    """
    knee_angle = get_knee_angle(frame)
    if knee_angle is None:
        return False
    
    return knee_angle < bottom_threshold


def detect_phase_simple(frame: List, prev_hip_height: Optional[float] = None,
                       velocity_threshold: float = 0.005) -> SquatPhase:
    """
    Simple per-frame phase detection.
    
    NOTE: For production use, prefer window-based FSM from 5_temporal_segmentation.py.
    This is a simplified version for quick analysis.
    
    Args:
        frame: Current keypoint frame
        prev_hip_height: Previous frame's hip height
        velocity_threshold: Movement threshold
        
    Returns:
        Detected phase
    """
    current_height = get_hip_height(frame)
    
    if current_height is None:
        return SquatPhase.UNKNOWN
    
    if prev_hip_height is None:
        return SquatPhase.IDLE
    
    velocity = current_height - prev_hip_height
    
    if abs(velocity) < velocity_threshold:
        # Check if at bottom (isometric) or standing (idle)
        knee_angle = get_knee_angle(frame)
        if knee_angle is not None and knee_angle < 110:
            return SquatPhase.ISOMETRIC
        return SquatPhase.IDLE
    elif velocity > 0:
        return SquatPhase.ECCENTRIC  # Moving down (Y increases)
    else:
        return SquatPhase.CONCENTRIC  # Moving up


# =============================================================================
# REP COUNTING HELPERS
# =============================================================================

def find_rep_peaks(hip_displacement: np.ndarray, 
                   min_prominence: float = 0.05,
                   min_distance: int = 20) -> List[int]:
    """
    Find frame indices of squat bottoms (peak hip displacement).
    
    Args:
        hip_displacement: Normalized hip displacement signal
        min_prominence: Minimum peak prominence
        min_distance: Minimum frames between peaks
        
    Returns:
        List of frame indices at squat bottoms
    """
    from scipy.signal import find_peaks
    
    # Find peaks (local maxima = squat bottoms)
    peaks, properties = find_peaks(
        hip_displacement,
        prominence=min_prominence,
        distance=min_distance
    )
    
    return peaks.tolist()


def count_reps(squat_data: Dict) -> int:
    """
    Count number of repetitions in squat data.
    
    Args:
        squat_data: Dictionary with 'keypoints_img' or 'hip_displacement'
        
    Returns:
        Number of detected repetitions
    """
    if 'hip_displacement' in squat_data:
        hip_disp = np.array(squat_data['hip_displacement'])
    else:
        keypoints = squat_data.get('keypoints_img', [])
        hip_disp = compute_hip_displacement_signal(keypoints)
    
    peaks = find_rep_peaks(hip_disp)
    return len(peaks)
