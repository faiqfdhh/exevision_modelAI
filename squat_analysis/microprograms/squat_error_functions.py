"""
squat_error_functions.py
Squat form error detection microprograms.

Error functions that analyze pose data to detect common squat form mistakes.
Each function returns a numeric score or boolean indicating error presence/severity.

Compatible with: 2.5_extract_selected_features.py output (MediaPipe 33 landmarks)
"""

import numpy as np
import math
from typing import Optional, Tuple, List, Dict

# MediaPipe landmark indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_angle(p1: List, p2: List, p3: List) -> float:
    """
    Calculate angle at p2 formed by points p1-p2-p3.
    Returns angle in degrees (0-180).
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return math.degrees(np.arccos(cos_angle))


def calculate_angle_3d(p1: List, p2: List, p3: List) -> float:
    """
    Calculate 3D angle at p2 formed by points p1-p2-p3.
    Uses x, y, z coordinates.
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    
    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return math.degrees(np.arccos(cos_angle))


def get_landmark(frame: List, idx: int) -> Optional[List]:
    """Safely get landmark from frame with bounds checking."""
    if frame is None or len(frame) <= idx:
        return None
    return frame[idx]


def get_confidence(frame: List, idx: int) -> float:
    """Get visibility/confidence score for landmark (4th element)."""
    lm = get_landmark(frame, idx)
    if lm is None or len(lm) < 4:
        return 0.0
    return lm[3]


def landmarks_valid(frame: List, indices: List[int], min_conf: float = 0.5) -> bool:
    """Check if all specified landmarks have sufficient confidence."""
    for idx in indices:
        if get_confidence(frame, idx) < min_conf:
            return False
    return True


# =============================================================================
# KNEE ERROR MICROPROGRAMS
# =============================================================================

def knee_valgus_error(frame: List, min_conf: float = 0.5) -> Optional[float]:
    """
    Detect knee valgus (knees caving inward).
    
    Method: Compare knee X-spread to ankle X-spread.
    If knees are closer together than ankles, it's valgus.
    
    Returns:
        Valgus ratio (knee_spread / ankle_spread). 
        < 1.0 = valgus (knees caving in)
        = 1.0 = neutral
        > 1.0 = varus (knees pushing out)
        None if landmarks invalid
    """
    required = [L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
    if not landmarks_valid(frame, required, min_conf):
        return None
    
    l_knee = get_landmark(frame, L_KNEE)
    r_knee = get_landmark(frame, R_KNEE)
    l_ankle = get_landmark(frame, L_ANKLE)
    r_ankle = get_landmark(frame, R_ANKLE)
    
    knee_spread = abs(l_knee[0] - r_knee[0])
    ankle_spread = abs(l_ankle[0] - r_ankle[0])
    
    if ankle_spread < 0.01:  # Avoid division by zero
        return None
    
    return knee_spread / ankle_spread


def knee_forward_travel_error(frame: List, min_conf: float = 0.5) -> Optional[float]:
    """
    Detect excessive forward knee travel (knees going past toes).
    
    Method: Compare knee X position to toe X position.
    Positive value = knees past toes (potential error).
    
    Returns:
        Average forward travel as normalized distance.
        Positive = knees past toes
        None if landmarks invalid
    """
    required = [L_KNEE, R_KNEE, L_TOE, R_TOE]
    if not landmarks_valid(frame, required, min_conf):
        return None
    
    l_knee = get_landmark(frame, L_KNEE)
    r_knee = get_landmark(frame, R_KNEE)
    l_toe = get_landmark(frame, L_TOE)
    r_toe = get_landmark(frame, R_TOE)
    
    # Use Z coordinate if available (depth), otherwise X
    if len(l_knee) > 2:
        l_forward = l_knee[2] - l_toe[2]  # Positive = knee in front
        r_forward = r_knee[2] - r_toe[2]
    else:
        l_forward = l_knee[0] - l_toe[0]
        r_forward = r_knee[0] - r_toe[0]
    
    return (l_forward + r_forward) / 2


def knee_angle_asymmetry(frame: List, min_conf: float = 0.5) -> Optional[float]:
    """
    Detect asymmetry between left and right knee angles.
    
    Returns:
        Absolute difference in degrees between left and right knee angles.
        0 = perfectly symmetric
        None if landmarks invalid
    """
    required = [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]
    if not landmarks_valid(frame, required, min_conf):
        return None
    
    l_hip = get_landmark(frame, L_HIP)
    l_knee = get_landmark(frame, L_KNEE)
    l_ankle = get_landmark(frame, L_ANKLE)
    
    r_hip = get_landmark(frame, R_HIP)
    r_knee = get_landmark(frame, R_KNEE)
    r_ankle = get_landmark(frame, R_ANKLE)
    
    l_angle = calculate_angle(l_hip, l_knee, l_ankle)
    r_angle = calculate_angle(r_hip, r_knee, r_ankle)
    
    return abs(l_angle - r_angle)


# =============================================================================
# HIP & TORSO ERROR MICROPROGRAMS
# =============================================================================

def hip_shift_error(frame: List, min_conf: float = 0.5) -> Optional[float]:
    """
    Detect lateral hip shift (weight shift to one side).
    
    Method: Compare hip midpoint X to ankle midpoint X.
    
    Returns:
        Shift ratio (normalized by stance width).
        0 = centered
        Positive/negative = shift direction
        None if landmarks invalid
    """
    required = [L_HIP, R_HIP, L_ANKLE, R_ANKLE]
    if not landmarks_valid(frame, required, min_conf):
        return None
    
    l_hip = get_landmark(frame, L_HIP)
    r_hip = get_landmark(frame, R_HIP)
    l_ankle = get_landmark(frame, L_ANKLE)
    r_ankle = get_landmark(frame, R_ANKLE)
    
    hip_mid_x = (l_hip[0] + r_hip[0]) / 2
    ankle_mid_x = (l_ankle[0] + r_ankle[0]) / 2
    ankle_spread = abs(l_ankle[0] - r_ankle[0])
    
    if ankle_spread < 0.01:
        return None
    
    return (hip_mid_x - ankle_mid_x) / ankle_spread


def forward_lean_error(frame: List, min_conf: float = 0.5) -> Optional[float]:
    """
    Detect excessive forward torso lean.
    
    Method: Calculate angle of shoulder-hip vector from vertical.
    
    Returns:
        Forward lean angle in degrees from vertical.
        0 = perfectly upright
        Positive = forward lean
        None if landmarks invalid
    """
    required = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    if not landmarks_valid(frame, required, min_conf):
        return None
    
    l_shoulder = get_landmark(frame, L_SHOULDER)
    r_shoulder = get_landmark(frame, R_SHOULDER)
    l_hip = get_landmark(frame, L_HIP)
    r_hip = get_landmark(frame, R_HIP)
    
    # Midpoints
    shoulder_mid = [(l_shoulder[0] + r_shoulder[0]) / 2,
                    (l_shoulder[1] + r_shoulder[1]) / 2]
    hip_mid = [(l_hip[0] + r_hip[0]) / 2,
               (l_hip[1] + r_hip[1]) / 2]
    
    # Vector from hip to shoulder
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]  # Positive Y = down in image coords
    
    # Angle from vertical (note: in image coords, vertical is negative Y)
    # arctan2 gives angle from positive X axis
    angle_from_vertical = math.degrees(math.atan2(dx, -dy))
    
    return angle_from_vertical


def hip_hinge_angle(frame: List, min_conf: float = 0.5) -> Optional[float]:
    """
    Calculate hip hinge angle (torso-to-thigh angle).
    
    Method: Angle at hip joint between torso and thigh.
    
    Returns:
        Hip angle in degrees (180 = standing straight)
        None if landmarks invalid
    """
    required = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE]
    if not landmarks_valid(frame, required, min_conf):
        return None
    
    l_shoulder = get_landmark(frame, L_SHOULDER)
    r_shoulder = get_landmark(frame, R_SHOULDER)
    l_hip = get_landmark(frame, L_HIP)
    r_hip = get_landmark(frame, R_HIP)
    l_knee = get_landmark(frame, L_KNEE)
    r_knee = get_landmark(frame, R_KNEE)
    
    # Use midpoints for more stable measurement
    shoulder_mid = [(l_shoulder[0] + r_shoulder[0]) / 2,
                    (l_shoulder[1] + r_shoulder[1]) / 2]
    hip_mid = [(l_hip[0] + r_hip[0]) / 2,
               (l_hip[1] + r_hip[1]) / 2]
    knee_mid = [(l_knee[0] + r_knee[0]) / 2,
                (l_knee[1] + r_knee[1]) / 2]
    
    return calculate_angle(shoulder_mid, hip_mid, knee_mid)


# =============================================================================
# FOOT & ANKLE ERROR MICROPROGRAMS
# =============================================================================

def heel_rise_error(frame: List, standing_heel_y: Optional[float] = None, 
                   min_conf: float = 0.5) -> Optional[float]:
    """
    Detect heel rising off ground during squat.
    
    Method: Compare heel Y position to standing position or ankle.
    
    Returns:
        Heel rise amount (positive = heels rising)
        None if landmarks invalid
    """
    required = [L_HEEL, R_HEEL, L_ANKLE, R_ANKLE]
    if not landmarks_valid(frame, required, min_conf):
        return None
    
    l_heel = get_landmark(frame, L_HEEL)
    r_heel = get_landmark(frame, R_HEEL)
    l_ankle = get_landmark(frame, L_ANKLE)
    r_ankle = get_landmark(frame, R_ANKLE)
    
    # Compare heel Y to ankle Y (heel should be below ankle)
    # In image coords, higher Y = lower position
    l_rise = l_ankle[1] - l_heel[1]  # Positive = heel below ankle (normal)
    r_rise = r_ankle[1] - r_heel[1]
    
    avg_rise = (l_rise + r_rise) / 2
    
    # Negative value means heels are higher than normal
    return avg_rise


def stance_width(frame: List, min_conf: float = 0.5) -> Optional[float]:
    """
    Calculate stance width normalized by hip width.
    
    Returns:
        Stance width ratio (ankle_spread / hip_spread)
        ~1.0 = hip-width stance
        < 1.0 = narrow stance
        > 1.0 = wide stance
        None if landmarks invalid
    """
    required = [L_HIP, R_HIP, L_ANKLE, R_ANKLE]
    if not landmarks_valid(frame, required, min_conf):
        return None
    
    l_hip = get_landmark(frame, L_HIP)
    r_hip = get_landmark(frame, R_HIP)
    l_ankle = get_landmark(frame, L_ANKLE)
    r_ankle = get_landmark(frame, R_ANKLE)
    
    hip_spread = abs(l_hip[0] - r_hip[0])
    ankle_spread = abs(l_ankle[0] - r_ankle[0])
    
    if hip_spread < 0.01:
        return None
    
    return ankle_spread / hip_spread


# =============================================================================
# AGGREGATE ERROR FUNCTIONS
# =============================================================================

def get_all_errors_for_frame(frame: List, min_conf: float = 0.5) -> Dict[str, Optional[float]]:
    """
    Calculate all error metrics for a single frame.
    
    Returns dictionary with all error values.
    """
    return {
        'knee_valgus': knee_valgus_error(frame, min_conf),
        'knee_forward_travel': knee_forward_travel_error(frame, min_conf),
        'knee_asymmetry': knee_angle_asymmetry(frame, min_conf),
        'hip_shift': hip_shift_error(frame, min_conf),
        'forward_lean': forward_lean_error(frame, min_conf),
        'hip_angle': hip_hinge_angle(frame, min_conf),
        'heel_rise': heel_rise_error(frame, min_conf=min_conf),
        'stance_width': stance_width(frame, min_conf),
    }


def aggregate_phase_errors(frames: List[List], phase_indices: List[int], 
                           min_conf: float = 0.5) -> Dict[str, Dict[str, float]]:
    """
    Aggregate error metrics across a phase (e.g., eccentric portion of rep).
    
    Args:
        frames: List of all keypoint frames
        phase_indices: Frame indices belonging to this phase
        min_conf: Minimum landmark confidence
    
    Returns:
        Dictionary with mean, min, max for each error type.
    """
    all_errors = {
        'knee_valgus': [],
        'knee_forward_travel': [],
        'knee_asymmetry': [],
        'hip_shift': [],
        'forward_lean': [],
        'hip_angle': [],
        'heel_rise': [],
        'stance_width': [],
    }
    
    for idx in phase_indices:
        if idx >= len(frames):
            continue
        frame = frames[idx]
        errors = get_all_errors_for_frame(frame, min_conf)
        
        for key, value in errors.items():
            if value is not None:
                all_errors[key].append(value)
    
    # Compute statistics
    result = {}
    for key, values in all_errors.items():
        if values:
            result[key] = {
                'mean': float(np.mean(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'std': float(np.std(values)),
            }
        else:
            result[key] = None
    
    return result
