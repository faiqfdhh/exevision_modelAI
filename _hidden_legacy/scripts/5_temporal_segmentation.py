"""
Temporal Segmentation Module (5.4)
Segments squat motion into idle/eccentric/concentric phases using biomechanically-sound methods.
Implements a simplified velocity-based state machine for robust segmentation.

Key Design Principles:
1. Window-based smoothing for velocity.
2. Angle and view-invariant control signals (normalized hip displacement).
3. Simplified State Logic: Downwards = Eccentric, Upwards = Concentric, Still = Idle/Isometric.
4. Body-proportion normalization (torso/leg length).
"""

import os
import json
import cv2
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from enum import Enum, auto


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}

# --- Configuration ---
FEATURES_EXCELLENT = "./squat/extracted_features_clean/excellent"
FEATURES_GOOD = "./squat/extracted_features_clean/good"
FEATURES_FAIR = "./squat/extracted_features_clean/fair"
FEATURES_RAW_UNFILTERED = "./squat/extracted_features_clean/raw_unfiltered"

FEATURES_DIRS = [FEATURES_EXCELLENT, FEATURES_GOOD, FEATURES_FAIR, FEATURES_RAW_UNFILTERED]
VIDEO_IDS_TO_PROCESS = ["*"] 
# Output directories organized by quality to mirror input structure
OUTPUT_DIR_EXCELLENT = "./squat/segmented_reps/excellent"
OUTPUT_DIR_GOOD = "./squat/segmented_reps/good"
OUTPUT_DIR_FAIR = "./squat/segmented_reps/fair"
OUTPUT_DIR_RAW_UNFILTERED = "./squat/segmented_reps/raw_unfiltered"

OUTPUT_DIRS = {
    "excellent": OUTPUT_DIR_EXCELLENT,
    "good": OUTPUT_DIR_GOOD,
    "fair": OUTPUT_DIR_FAIR,
    "raw_unfiltered": OUTPUT_DIR_RAW_UNFILTERED,
}

VISUALIZATION_DIR_EXCELLENT = "./squat/visualized_segmentation/excellent"
VISUALIZATION_DIR_GOOD = "./squat/visualized_segmentation/good"
VISUALIZATION_DIR_FAIR = "./squat/visualized_segmentation/fair"
VISUALIZATION_DIR_RAW_UNFILTERED = "./squat/visualized_segmentation/raw_unfiltered"

VISUALIZATION_DIRS = {
    "excellent": VISUALIZATION_DIR_EXCELLENT,
    "good": VISUALIZATION_DIR_GOOD,
    "fair": VISUALIZATION_DIR_FAIR,
    "raw_unfiltered": VISUALIZATION_DIR_RAW_UNFILTERED,
}

VIDEO_DIR = "./squat/dataset_videos_all"

# --- Phase Colors (BGR for OpenCV) ---
PHASE_COLORS = {
    "idle": (128, 128, 128),       # Gray
    "eccentric": (0, 0, 255),      # Red
    "isometric": (255, 255, 0),    # Cyan
    "concentric": (0, 255, 0),     # Green
    "unknown": (50, 50, 50),       # Dark gray
}

# --- Landmark Indices (MediaPipe) ---
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_SHOULDER, R_SHOULDER = 11, 12
L_HEEL, R_HEEL = 29, 30
NOSE = 0

# =============================================================================
# WINDOW-BASED ANALYSIS PARAMETERS
# =============================================================================
ANALYSIS_WINDOW_SIZE = 15             # Frames for trend analysis (500ms @ 30fps)
HALF_WINDOW = ANALYSIS_WINDOW_SIZE // 2

# =============================================================================
# PHASE DURATION & HYSTERESIS
# =============================================================================
MIN_PHASE_DURATION_FRAMES = 12        # Kept for reporting/backward compatibility
GLITCH_MERGE_FRAMES = 3               # Only merge ultra-short phase flickers
MIN_CONCENTRIC_DURATION = 20          # Concentric must last at least 20 frames before IDLE
HYSTERESIS_FRAMES = 8                 # Must exceed threshold for this many frames to transition
PHASE_LOCKOUT_FRAMES = 6              # Frames to wait before allowing another transition

# =============================================================================
# VELOCITY THRESHOLDS (normalized, per-window)
# =============================================================================
VELOCITY_IDLE_THRESHOLD = 0.005       # Reduced from 0.008 (was too strict)
VELOCITY_MOVING_THRESHOLD = 0.010     # Reduced from 0.015
VELOCITY_ISOMETRIC_BAND = 0.1       # Reduced from 0.012

# Motion confirmation for anti-jitter transitions.
DOWNWARD_VELOCITY_THRESHOLD = 0.0015
UPWARD_VELOCITY_THRESHOLD = -0.0015
POSITION_JITTER_THRESHOLD = 0.0003
MOTION_CONFIRM_FRAMES = 2
KNEE_BEND_DELTA_DEG = 2.0
KNEE_EXTENDED_TOLERANCE_DEG = 3.0
IDLE_HEIGHT_MARGIN = 0.02

# Require hips to return near start-of-video baseline before allowing IDLE.
IDLE_RETURN_MARGIN = 0.03
IDLE_KNEE_EXTENSION_THRESHOLD = 150.0

# =============================================================================
# MOVEMENT DETECTION THRESHOLDS
# =============================================================================
ECCENTRIC_VELOCITY_MIN = 0.006        # Reduced from 0.012 (too strict for fair)
CONCENTRIC_VELOCITY_MIN = 0.006       # Reduced from 0.012
ISOMETRIC_MIN_DURATION = 10
ISOMETRIC_MAX_DURATION = 45

# =============================================================================
# SQUAT VALIDATION (loosened for fair quality)
# =============================================================================
MIN_REP_FRAMES = 20                   # Reduced from 25
MIN_SQUAT_DEPTH_RATIO = 0.05          # Reduced from 0.08 (was too strict)
MIN_SQUAT_DEPTH_ANGLE = 10.0          # Reduced from 15.0
STANDING_KNEE_ANGLE_THRESHOLD = 135.0  # ⭐ CHANGED: Knee angle must be > 135° for IDLE (nearly straight)

# =============================================================================
# SIMPLE (PHASE-ONLY) REP COUNTING
# =============================================================================
ENABLE_PHASE_ONLY_REP_FALLBACK = True
MIN_REP_FRAMES_PHASE_ONLY = 6  # very small, just to avoid 1-2 frame glitches

# =============================================================================
# CONFIDENCE & CALIBRATION
# =============================================================================
MIN_LANDMARK_CONFIDENCE = 0.4         # Discard frames with low confidence joints
MIN_KEY_JOINT_CONFIDENCE = 0.5        # Key joints need higher confidence
CALIBRATION_FRAMES = 60               # Increased from 45 (need more idle frames)
MIN_VALID_CALIBRATION_FRAMES = 20     # Increased from 15

# =============================================================================
# VIEW VALIDATION
# =============================================================================
VALID_VIEWS = {'side', 'front_side', 'back_side', 'front', 'back', 'unknown'}
RELIABLE_VIEWS = {'side', 'front_side', 'back_side'}  # Views with good depth perception


class SquatPhase(Enum):
    """Enumeration of squat phases with explicit ordering"""
    IDLE = 0
    ECCENTRIC = 1      # Descending (knee flexion)
    ISOMETRIC = 2      # Holding at bottom (optional)
    CONCENTRIC = 3     # Ascending (knee extension)
    UNKNOWN = 4        # Invalid/unreliable data


# Valid FSM transitions (from -> [allowed destinations])
VALID_TRANSITIONS = {
    SquatPhase.IDLE: [SquatPhase.ECCENTRIC, SquatPhase.IDLE],
    SquatPhase.ECCENTRIC: [SquatPhase.ISOMETRIC, SquatPhase.CONCENTRIC, SquatPhase.ECCENTRIC],
    SquatPhase.ISOMETRIC: [SquatPhase.CONCENTRIC, SquatPhase.ISOMETRIC],
    SquatPhase.CONCENTRIC: [SquatPhase.IDLE, SquatPhase.CONCENTRIC],
    SquatPhase.UNKNOWN: [SquatPhase.IDLE, SquatPhase.UNKNOWN],
}


def find_video_file(video_id: str, quality: Optional[str] = None) -> Optional[str]:
    """
    Find video file by ID.
    Priority:
    1. Annotated features visualization (step 2.5) if quality is provided
    2. Original raw video
    """
    # 1. Try to find annotated video from step 2.5 first (to combine visualizations)
    if quality:
        # Map quality to directory name if needed, but usually it matches
        quality_lower = quality.lower()
        annotated_dir = f"./squat/visualized_poses_clean/{quality_lower}"
        annotated_path = os.path.join(annotated_dir, f"{video_id}_annotated.mp4")
        
        if os.path.exists(annotated_path):
            # print(f"   ℹ️  Using annotated video source: {annotated_path}")
            return annotated_path

    # 2. Fallback to original video search
    video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.flv')
    
    for root, dirs, files in os.walk(VIDEO_DIR):
        for file in files:
            if os.path.splitext(file)[0] == video_id and file.lower().endswith(video_extensions):
                return os.path.join(root, file)
    
    return None


def convert_to_serializable(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, Enum):
        return obj.name.lower()
    return obj


@dataclass
class RepPhase:
    """Single phase within a repetition"""
    phase_type: str          # 'idle', 'eccentric', 'isometric', 'concentric'
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_seconds: float
    transition_reason: str = ""  # Why this phase was entered
    
    def to_dict(self):
        return {
            "phase_type": self.phase_type,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "duration_seconds": round(self.duration_seconds, 2),
            "transition_reason": self.transition_reason
        }


@dataclass 
class Repetition:
    """Single squat repetition with phase breakdown"""
    rep_id: int
    start_frame: int
    end_frame: int
    phases: List[RepPhase]
    squat_depth_normalized: float   # Normalized depth (0-1 scale)
    squat_depth_angle: float        # Knee angle change (degrees)
    bottom_frame: int               # Frame at deepest squat position
    bottom_knee_angle: float        # Actual knee angle at bottom
    
    def to_dict(self):
        return {
            "rep_id": self.rep_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.end_frame - self.start_frame,
            "squat_depth_normalized": round(self.squat_depth_normalized, 3),
            "squat_depth_angle": round(self.squat_depth_angle, 2),
            "bottom_knee_angle": round(self.bottom_knee_angle, 2),
            "bottom_frame": self.bottom_frame,
            "phases": [p.to_dict() for p in self.phases]
        }


@dataclass
class WindowMetrics:
    """Metrics computed over a temporal window"""
    center_frame: int
    velocity_trend: float           # Average velocity over window (+ = descending)
    velocity_stability: float       # Variance of velocity (low = stable)
    position_trend: float           # Net position change over window
    depth_ratio: float              # Current depth as ratio of max observed
    confidence: float               # Average landmark confidence in window


class BiomechanicalAnalyzer:
    """
    Calculates angle-invariant and view-invariant biomechanical metrics.
    Uses body-proportion normalization for consistency across subjects.
    """
    
    def __init__(self, keypoints: List, view: str, fps: float = 30.0):
        self.keypoints = keypoints
        self.view = view
        self.fps = fps
        self.frame_count = len(keypoints)
        
        # Calibrated anthropometrics (for normalization)
        self.torso_length = None
        self.femur_length = None
        self.tibia_length = None
        self.standing_hip_height = None
        self.body_scale = None  # Combined normalization factor
        
        # Primary control signals (view-invariant)
        self.normalized_hip_displacement = None    # Main signal for phase detection
        self.knee_angles = None                    # Secondary validation signal
        self.hip_heights_raw = None
        
        # Derived signals
        self.velocity_signal = None                # Smoothed velocity of hip displacement
        self.window_velocities = None              # Window-averaged velocities
        self.landmark_confidence = None
        self.valid_frame_mask = None
        
        # View reliability
        self.view_reliable = view.lower() in RELIABLE_VIEWS if view else False
        
        # Use knee angle as primary signal for front/back views
        self.use_knee_angle_primary = view.lower() in {'front', 'back'}
    
    def calibrate_from_idle(self) -> bool:
        """
        Extract anthropometric measurements from idle frames.
        Now includes outlier removal to find true standing position.
        """
        valid_frames = []
        
        for i, frame in enumerate(self.keypoints[:CALIBRATION_FRAMES]):
            if frame is None or len(frame) < 33:
                continue
            
            # Check key joint confidence
            key_joints = [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_SHOULDER, R_SHOULDER]
            confidences = [frame[j][3] for j in key_joints if j < len(frame)]
            
            if len(confidences) >= 6 and np.mean(confidences) >= MIN_KEY_JOINT_CONFIDENCE:
                valid_frames.append(frame)
        
        if len(valid_frames) < MIN_VALID_CALIBRATION_FRAMES:
            # Fallback: use all frames with relaxed thresholds
            valid_frames = [f for f in self.keypoints[:CALIBRATION_FRAMES] 
                          if f is not None and len(f) >= 28]
        
        if len(valid_frames) < 5:
            # Critical failure - use defaults
            self.torso_length = 0.25
            self.femur_length = 0.2
            self.tibia_length = 0.2
            self.standing_hip_height = 0.5
            self.body_scale = 0.25
            return False
        
        torso_lengths = []
        femur_lengths = []
        tibia_lengths = []
        hip_heights = []
        
        for frame in valid_frames:
            # Get landmarks
            l_shoulder = np.array(frame[L_SHOULDER][:3])
            r_shoulder = np.array(frame[R_SHOULDER][:3])
            l_hip = np.array(frame[L_HIP][:3])
            r_hip = np.array(frame[R_HIP][:3])
            l_knee = np.array(frame[L_KNEE][:3])
            r_knee = np.array(frame[R_KNEE][:3])
            l_ankle = np.array(frame[L_ANKLE][:3])
            r_ankle = np.array(frame[R_ANKLE][:3])
            
            # Torso length: shoulder-to-hip midpoint distance
            shoulder_mid = (l_shoulder + r_shoulder) / 2
            hip_mid = (l_hip + r_hip) / 2
            torso_len = np.linalg.norm(shoulder_mid - hip_mid)
            torso_lengths.append(torso_len)
            
            # Femur length: hip-to-knee distance
            l_femur = np.linalg.norm(l_hip - l_knee)
            r_femur = np.linalg.norm(r_hip - r_knee)
            femur_lengths.append((l_femur + r_femur) / 2)
            
            # Tibia length: knee-to-ankle distance
            l_tibia = np.linalg.norm(l_knee - l_ankle)
            r_tibia = np.linalg.norm(r_knee - r_ankle)
            tibia_lengths.append((l_tibia + r_tibia) / 2)
            
            # Hip height (Y coordinate)
            hip_heights.append(hip_mid[1])
        
        # Use median to be robust to outliers
        self.torso_length = np.median(torso_lengths)
        self.femur_length = np.median(femur_lengths)
        self.tibia_length = np.median(tibia_lengths)
        
        # ⭐ KEY FIX: Use MIN of hip heights as standing position (smallest Y = highest position)
        # This handles cases where calibration frames include some movement
        hip_heights_sorted = np.sort(hip_heights)
        # Use the lowest 25th percentile as standing height
        standing_idx = int(len(hip_heights_sorted) * 0.25)
        self.standing_hip_height = hip_heights_sorted[standing_idx]
        
        # Body scale = average of torso and leg lengths (for normalization)
        self.body_scale = (self.torso_length + self.femur_length + self.tibia_length) / 3
        
        return True
    
    def compute_normalized_hip_displacement(self) -> np.ndarray:
        """
        Compute the primary control signal: normalized vertical hip displacement.
        Enhanced to handle compressed signals better.
        
        Returns: Array where 0 = standing position, positive = squat depth
        """
        displacements = []
        confidences = []
        
        for frame in self.keypoints:
            if frame is None or len(frame) < 24:
                displacements.append(np.nan)
                confidences.append(0.0)
                continue
            
            # Get hip midpoint
            l_hip = np.array(frame[L_HIP][:3])
            r_hip = np.array(frame[R_HIP][:3])
            hip_mid = (l_hip + r_hip) / 2
            
            # Get confidence
            hip_conf = (frame[L_HIP][3] + frame[R_HIP][3]) / 2
            confidences.append(hip_conf)
            
            if hip_conf < MIN_LANDMARK_CONFIDENCE:
                displacements.append(np.nan)
                continue
            
            # ⭐ KEY FIX: Displacement from standing (positive = lower = squatting)
            # Use absolute Y difference, clipped to prevent negative depths
            displacement = max(0, hip_mid[1] - self.standing_hip_height)
            
            # Normalize by body scale
            normalized_disp = displacement / self.body_scale if self.body_scale > 0 else displacement
            displacements.append(normalized_disp)
        
        self.hip_heights_raw = np.array([(frame[L_HIP][1] + frame[R_HIP][1]) / 2 
                                         if frame is not None and len(frame) > R_HIP 
                                         else np.nan for frame in self.keypoints])
        self.landmark_confidence = np.array(confidences)
        self.normalized_hip_displacement = np.array(displacements)
        
        # Interpolate missing values
        self._interpolate_array(self.normalized_hip_displacement)
        self._interpolate_array(self.hip_heights_raw)
        
        return self.normalized_hip_displacement
    
    def compute_knee_angles(self) -> np.ndarray:
        """Compute knee bending angle for all frames (secondary signal)"""
        angles = []
        
        for frame in self.keypoints:
            if frame is None or len(frame) < 28:
                angles.append(np.nan)
                continue
            
            # Extract 3D positions
            l_hip = np.array(frame[L_HIP][:3])
            l_knee = np.array(frame[L_KNEE][:3])
            l_ankle = np.array(frame[L_ANKLE][:3])
            
            r_hip = np.array(frame[R_HIP][:3])
            r_knee = np.array(frame[R_KNEE][:3])
            r_ankle = np.array(frame[R_ANKLE][:3])
            
            # Check confidence
            l_conf = min(frame[L_HIP][3], frame[L_KNEE][3], frame[L_ANKLE][3])
            r_conf = min(frame[R_HIP][3], frame[R_KNEE][3], frame[R_ANKLE][3])
            
            l_angle = self._calculate_angle(l_hip, l_knee, l_ankle) if l_conf >= MIN_LANDMARK_CONFIDENCE else np.nan
            r_angle = self._calculate_angle(r_hip, r_knee, r_ankle) if r_conf >= MIN_LANDMARK_CONFIDENCE else np.nan
            
            # Use average of valid angles
            valid_angles = [a for a in [l_angle, r_angle] if not np.isnan(a)]
            avg_angle = np.mean(valid_angles) if valid_angles else np.nan
            
            angles.append(avg_angle)
        
        self.knee_angles = np.array(angles)
        self._interpolate_array(self.knee_angles)
        return self.knee_angles
    
    def compute_velocity_signal(self) -> np.ndarray:
        """
        Compute smoothed velocity of hip displacement.
        Positive velocity = moving down (eccentric)
        Negative velocity = moving up (concentric)
        """
        if self.normalized_hip_displacement is None:
            self.compute_normalized_hip_displacement()
        
        # Compute raw velocity (first derivative)
        raw_velocity = np.gradient(self.normalized_hip_displacement)
        
        # Apply Savitzky-Golay filter for smooth velocity
        window_len = min(11, len(raw_velocity) // 2 * 2 + 1)
        if window_len >= 5:
            self.velocity_signal = savgol_filter(raw_velocity, window_len, 2)
        else:
            self.velocity_signal = raw_velocity
        
        return self.velocity_signal
    
    def compute_window_velocities(self) -> np.ndarray:
        """
        Compute velocity trends averaged over windows.
        This is the primary input for phase classification.
        """
        if self.velocity_signal is None:
            self.compute_velocity_signal()
        
        # Apply uniform filter (moving average) over window
        self.window_velocities = uniform_filter1d(self.velocity_signal, size=ANALYSIS_WINDOW_SIZE, mode='nearest')
        
        return self.window_velocities
    
    def compute_valid_frame_mask(self) -> np.ndarray:
        """Identify frames with sufficient landmark confidence"""
        if self.landmark_confidence is None:
            return np.ones(self.frame_count, dtype=bool)
        
        self.valid_frame_mask = self.landmark_confidence >= MIN_LANDMARK_CONFIDENCE
        return self.valid_frame_mask
    
    def get_window_metrics(self, center_frame: int) -> WindowMetrics:
        """Compute metrics for a temporal window centered at given frame"""
        start = max(0, center_frame - HALF_WINDOW)
        end = min(self.frame_count, center_frame + HALF_WINDOW + 1)
        
        # Get window data
        window_velocity = self.velocity_signal[start:end]
        window_position = self.normalized_hip_displacement[start:end]
        window_confidence = self.landmark_confidence[start:end]
        
        # Compute window metrics
        velocity_trend = np.mean(window_velocity)
        velocity_stability = np.std(window_velocity)
        position_trend = window_position[-1] - window_position[0] if len(window_position) > 1 else 0
        
        # Current depth as ratio of max observed depth
        current_depth = self.normalized_hip_displacement[center_frame]
        max_depth = np.nanmax(self.normalized_hip_displacement) if np.any(~np.isnan(self.normalized_hip_displacement)) else 1
        depth_ratio = current_depth / max_depth if max_depth > 0 else 0
        
        avg_confidence = np.mean(window_confidence)
        
        return WindowMetrics(
            center_frame=center_frame,
            velocity_trend=velocity_trend,
            velocity_stability=velocity_stability,
            position_trend=position_trend,
            depth_ratio=depth_ratio,
            confidence=avg_confidence
        )
    
    def _calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle at p2 formed by p1-p2-p3"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return np.nan
        
        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        angle = np.arccos(cos_angle)
        return np.degrees(angle)
    
    def _interpolate_array(self, arr: np.ndarray):
        """Fill NaN values with linear interpolation"""
        nans = np.isnan(arr)
        if nans.all():
            arr[:] = 0
            return
        if nans.any():
            indices = np.arange(len(arr))
            arr[nans] = np.interp(indices[nans], indices[~nans], arr[~nans])


class SquatStateMachine:
    """
    Simplified Machine for squat phase detection.
    
    Logic:
    1. Positive Velocity (Hips Downwards) -> ECCENTRIC
    2. Negative Velocity (Hips Upwards) -> CONCENTRIC
    3. Low Velocity -> ISOMETRIC (if currently in a rep) or IDLE
    """
    
    def __init__(self, analyzer: BiomechanicalAnalyzer, fps: float = 30.0):
        self.analyzer = analyzer
        self.fps = fps
        self.frame_count = analyzer.frame_count
        
        # Ensure signals are computed
        if analyzer.window_velocities is None:
            analyzer.compute_window_velocities()
        
        self.velocity = analyzer.window_velocities
        self.position = analyzer.normalized_hip_displacement
        self.raw_velocity = analyzer.velocity_signal
        self.knee_angles = analyzer.knee_angles

        finite_positions = self.position[np.isfinite(self.position)]
        if finite_positions.size > 0:
            # Use the first valid frame to match "starting position" semantics.
            self.start_hip_position = float(finite_positions[0])
        else:
            self.start_hip_position = 0.0
        
        # State tracking
        self.current_state = SquatPhase.IDLE
        self.phase_labels = np.full(self.frame_count, SquatPhase.IDLE)
        self.transition_log = []
        self.illegal_transition_repairs = 0

    def _can_return_to_idle(self, frame_idx: int, pos: float) -> bool:
        """Only allow IDLE once hips return near beginning-of-video standing position."""
        hip_returned = pos <= (self.start_hip_position + IDLE_RETURN_MARGIN)
        if not hip_returned:
            return False

        if self.knee_angles is None or frame_idx >= len(self.knee_angles):
            return True

        knee_angle = self.knee_angles[frame_idx]
        if np.isnan(knee_angle):
            return True

        return knee_angle >= IDLE_KNEE_EXTENSION_THRESHOLD

    def _knee_extended(self, frame_idx: int) -> bool:
        if self.knee_angles is None or frame_idx >= len(self.knee_angles):
            return False
        knee_angle = self.knee_angles[frame_idx]
        if np.isnan(knee_angle):
            return False
        return knee_angle >= (STANDING_KNEE_ANGLE_THRESHOLD - KNEE_EXTENDED_TOLERANCE_DEG)

    def _knee_bending(self, frame_idx: int) -> bool:
        if self.knee_angles is None or frame_idx >= len(self.knee_angles):
            return False
        knee_angle = self.knee_angles[frame_idx]
        if np.isnan(knee_angle):
            return False
        return knee_angle <= (STANDING_KNEE_ANGLE_THRESHOLD - KNEE_BEND_DELTA_DEG)

    def _at_top_height(self, pos: float) -> bool:
        return pos <= (self.start_hip_position + IDLE_HEIGHT_MARGIN)

    def _is_transition_allowed(self, from_phase: SquatPhase, to_phase: SquatPhase) -> bool:
        return to_phase in VALID_TRANSITIONS.get(from_phase, [from_phase])

    def _sanitize_phase_sequence(self):
        """
        Enforce strict legal adjacency globally.

        Allowed high-level cycle:
        IDLE -> ECCENTRIC -> [ISOMETRIC] -> CONCENTRIC -> IDLE

        This pass repairs any illegal adjacency introduced by raw detection noise
        or post-processing by replacing illegal targets with previous legal phase.
        """
        if self.frame_count <= 1:
            return

        # Safety: first frame must start from IDLE in strict sequencing.
        if self.phase_labels[0] != SquatPhase.IDLE:
            self.transition_log.append({
                "frame": 0,
                "from": self.phase_labels[0].name.lower(),
                "to": SquatPhase.IDLE.name.lower(),
                "reason": "Sanitizer repair: force first frame to idle for strict cycle"
            })
            self.phase_labels[0] = SquatPhase.IDLE
            self.illegal_transition_repairs += 1

        for i in range(1, self.frame_count):
            prev_phase = self.phase_labels[i - 1]
            curr_phase = self.phase_labels[i]

            if self._is_transition_allowed(prev_phase, curr_phase):
                continue

            self.transition_log.append({
                "frame": i,
                "from": curr_phase.name.lower(),
                "to": prev_phase.name.lower(),
                "reason": (
                    f"Sanitizer repair: illegal adjacency {prev_phase.name.lower()} -> "
                    f"{curr_phase.name.lower()} replaced with {prev_phase.name.lower()}"
                )
            })
            self.phase_labels[i] = prev_phase
            self.illegal_transition_repairs += 1
    
    def detect_phases(self) -> np.ndarray:
        """
        Simplified phase detection looping over all frames.
        Returns array of phase labels for each frame.
        """
        downward_count = 0
        upward_count = 0
        still_count = 0

        for i in range(self.frame_count):
            vel = self.velocity[i]
            pos = self.position[i]
            raw_vel = self.raw_velocity[i] if self.raw_velocity is not None else vel
            eff_vel = 0.6 * float(vel) + 0.4 * float(raw_vel)
            bent = self._knee_bending(i)
            extended = self._knee_extended(i)
            at_top = self._at_top_height(pos)
            
            suggested_phase = self.current_state

            # Motion counters to reject slight jitter.
            if eff_vel > DOWNWARD_VELOCITY_THRESHOLD and abs(pos - self.start_hip_position) > POSITION_JITTER_THRESHOLD:
                downward_count += 1
            else:
                downward_count = 0

            if eff_vel < UPWARD_VELOCITY_THRESHOLD:
                upward_count += 1
            else:
                upward_count = 0

            if abs(eff_vel) <= VELOCITY_IDLE_THRESHOLD:
                still_count += 1
            else:
                still_count = 0

            # 1) Any confirmed downward movement + at least slight bend => ECCENTRIC.
            if downward_count >= MOTION_CONFIRM_FRAMES and bent:
                suggested_phase = SquatPhase.ECCENTRIC

            # 2) Upward movement from ECCENTRIC/ISOMETRIC => CONCENTRIC.
            elif (
                upward_count >= MOTION_CONFIRM_FRAMES
                and self.current_state in [SquatPhase.ECCENTRIC, SquatPhase.ISOMETRIC]
            ):
                suggested_phase = SquatPhase.CONCENTRIC

            # 3) ISOMETRIC entry is only valid as ECCENTRIC -> ISOMETRIC after >1s hold.
            elif (
                self.current_state == SquatPhase.ECCENTRIC
                and bent
                and still_count >= int(self.fps)
            ):
                suggested_phase = SquatPhase.ISOMETRIC

            # 4) When ascending, return to IDLE only at top + extended knees.
            elif self.current_state == SquatPhase.CONCENTRIC:
                if at_top and extended and self._can_return_to_idle(i, pos):
                    suggested_phase = SquatPhase.IDLE
                else:
                    suggested_phase = SquatPhase.CONCENTRIC

            # 5) True IDLE posture detection only from IDLE context.
            elif self.current_state == SquatPhase.IDLE and at_top and extended:
                suggested_phase = SquatPhase.IDLE

            else:
                # Default behavior per state, without creating new isometric entries.
                if self.current_state == SquatPhase.ISOMETRIC:
                    # Stay in isometric until an upward transition promotes to concentric.
                    suggested_phase = SquatPhase.ISOMETRIC
                elif bent and self.current_state == SquatPhase.ECCENTRIC:
                    # Keep eccentric active unless explicit >1s hold promotes isometric.
                    suggested_phase = SquatPhase.ECCENTRIC
                elif bent and self.current_state == SquatPhase.IDLE:
                    # From idle, bent posture alone is not enough for isometric.
                    suggested_phase = SquatPhase.ECCENTRIC if downward_count > 0 else SquatPhase.IDLE
                elif self.current_state == SquatPhase.ECCENTRIC:
                    # Do not allow direct ECCENTRIC -> IDLE; wait for upward transition first.
                    suggested_phase = SquatPhase.ECCENTRIC
                else:
                    suggested_phase = SquatPhase.IDLE

            # Handle phase transitions for logging
            if suggested_phase != self.current_state:
                self.transition_log.append({
                    "frame": i,
                    "from": self.current_state.name.lower(),
                    "to": suggested_phase.name.lower(),
                    "reason": (
                        f"Rule-based logic: eff_vel={eff_vel:.4f}, raw_vel={raw_vel:.4f}, "
                        f"pos={pos:.4f}, bent={int(bent)}, extended={int(extended)}, "
                        f"down_cnt={downward_count}, up_cnt={upward_count}, still_cnt={still_count}"
                    )
                })
                self.current_state = suggested_phase
            
            # Assign current state to this frame
            self.phase_labels[i] = self.current_state

        # Strict pass before and after cleanup to remove any illegal adjacency.
        self._sanitize_phase_sequence()
        
        # Post-processing: ensure minimum phase durations to avoid 1-frame flickering
        self._enforce_minimum_durations()
        self._sanitize_phase_sequence()
        
        # Convert enum to int for compatibility
        return np.array([p.value for p in self.phase_labels])
    
    def _enforce_minimum_durations(self):
        """
        Post-processing pass to merge short phases into neighbors.
        Eliminates any remaining 1-2 frame jitters.
        """
        # Find phase boundaries
        changes = [0]
        for i in range(1, self.frame_count):
            if self.phase_labels[i] != self.phase_labels[i-1]:
                changes.append(i)
        changes.append(self.frame_count)
        
        # Check each segment
        for i in range(len(changes) - 1):
            start = changes[i]
            end = changes[i + 1]
            duration = end - start
            
            if duration < GLITCH_MERGE_FRAMES:
                # Merge only ultra-short flickers.
                # Using a small threshold preserves valid shallow-squat phases.
                if i > 0:
                    prev_phase = self.phase_labels[changes[i-1]]
                    self.phase_labels[start:end] = prev_phase
    
    def get_phase_name(self, phase_id: int) -> str:
        """Convert phase ID to name"""
        phase_names = {0: "idle", 1: "eccentric", 2: "isometric", 3: "concentric", 4: "unknown"}
        return phase_names.get(phase_id, "unknown")


class TemporalSegmenter:
    """
    Advanced temporal segmentation with biomechanical rigor.
    
    Pipeline:
    1. View validation - reject unreliable views
    2. Anthropometric calibration from idle frames
    3. Compute view-invariant control signals
    4. Window-based FSM phase detection
    5. Repetition detection and validation
    """
    
    def __init__(self, keypoints_data: dict, video_id: str):
        self.video_id = video_id
        self.keypoints = keypoints_data.get('keypoints_img', [])
        self.info = keypoints_data.get('info', {})
        self.view = self.info.get('view', 'unknown')
        self.fps = self.info.get('fps', 30.0)
        self.frame_count = len(self.keypoints)
        self.quality_rating = self.info.get('quality_rating', 'Unknown')
        
        # Validate view
        self.view_valid = self._validate_view()
        
        # Analysis components
        self.analyzer = BiomechanicalAnalyzer(self.keypoints, self.view, self.fps)
        self.state_machine = None
        self.phase_labels = None
    
    def _validate_view(self) -> bool:
        """
        Validate that the camera view is processable.
        Allow all views including unknown; just flag reliability.
        """
        view_lower = self.view.lower() if self.view else 'unknown'
        return view_lower in VALID_VIEWS
    
    def segment(self) -> dict:
        """Main segmentation pipeline with comprehensive error handling"""
        try:
            # Step 0: View validation
            if not self.view_valid:
                return {
                    "video_id": self.video_id,
                    "error": f"Unreliable view: {self.view}",
                    "info": {
                        "fps": float(self.fps),
                        "frame_count": int(self.frame_count),
                        "quality_rating": self.quality_rating,
                        "view": self.view,
                        "view_reliable": False
                    }
                }
            
            # Step 1: Calibrate anthropometrics
            calibration_success = self.analyzer.calibrate_from_idle()
            
            # Step 2: Compute primary control signal (normalized hip displacement)
            self.analyzer.compute_normalized_hip_displacement()
            
            # Step 3: Compute secondary signals
            self.analyzer.compute_knee_angles()
            self.analyzer.compute_velocity_signal()
            self.analyzer.compute_window_velocities()
            self.analyzer.compute_valid_frame_mask()
            
            # Step 4: Window-based FSM phase detection
            self.state_machine = SquatStateMachine(self.analyzer, self.fps)
            phase_ids = self.state_machine.detect_phases()
            self.phase_labels = phase_ids
            
            # Step 5: Detect and validate repetitions
            reps = self._detect_repetitions()

            # ⭐ FALLBACK: if strict detection finds nothing, count by phase cycles only
            if ENABLE_PHASE_ONLY_REP_FALLBACK and len(reps) == 0:
                reps = self._detect_repetitions_phase_only()

            # Convert phase labels to names
            phase_names_list = [self.state_machine.get_phase_name(int(p)) for p in self.phase_labels]
            
            result = {
                "video_id": self.video_id,
                "info": {
                    "fps": float(self.fps),
                    "frame_count": int(self.frame_count),
                    "quality_rating": self.quality_rating,
                    "view": self.view,
                    "view_reliable": self.view.lower() in RELIABLE_VIEWS,
                    "total_reps": len(reps),
                    "calibration": {
                        "torso_length": float(self.analyzer.torso_length),
                        "femur_length": float(self.analyzer.femur_length),
                        "tibia_length": float(self.analyzer.tibia_length),
                        "standing_hip_height": float(self.analyzer.standing_hip_height),
                        "body_scale": float(self.analyzer.body_scale),
                        "calibration_success": calibration_success
                    },
                    "analysis_params": {
                        "window_size": ANALYSIS_WINDOW_SIZE,
                        "min_phase_duration": MIN_PHASE_DURATION_FRAMES,
                        "hysteresis_frames": HYSTERESIS_FRAMES,
                        "velocity_idle_threshold": VELOCITY_IDLE_THRESHOLD,
                        "velocity_moving_threshold": VELOCITY_MOVING_THRESHOLD,
                        "strict_phase_sequence": "idle->eccentric->[isometric]->concentric->idle",
                        "illegal_transition_repairs": int(self.state_machine.illegal_transition_repairs)
                    }
                },
                "frame_phases": phase_names_list,
                "repetitions": [r.to_dict() for r in reps],
                "transition_log": self.state_machine.transition_log,
                "signals": {
                    "normalized_hip_displacement": self.analyzer.normalized_hip_displacement.tolist(),
                    "window_velocity": self.analyzer.window_velocities.tolist(),
                    "raw_velocity": self.analyzer.velocity_signal.tolist(),
                    "knee_angles": self.analyzer.knee_angles.tolist(),
                    "landmark_confidence": self.analyzer.landmark_confidence.tolist()
                }
            }
            
            return convert_to_serializable(result)

        except Exception as e:
            import traceback
            return {
                "video_id": self.video_id,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _detect_repetitions(self) -> List[Repetition]:
        """
        Detect repetitions from phase sequences.
        A rep = eccentric → [isometric] → concentric (cycle)
        NOW: Properly handles reps starting from IDLE state.
        """
        reps = []
        i = 0
        
        while i < self.frame_count:
            # Skip leading IDLE frames
            if self.phase_labels[i] == SquatPhase.IDLE.value:
                i += 1
                continue
            
            # ⭐ START REP: Found eccentric (marks beginning of downward movement)
            if self.phase_labels[i] == SquatPhase.ECCENTRIC.value:
                rep_start = i
                bottom_frame = i
                max_depth = self.analyzer.normalized_hip_displacement[i]
                
                # Track what phases we've seen in this rep
                has_eccentric = True
                has_isometric = False
                has_concentric = False
                
                j = i + 1
                
                # ⭐ SCAN: Look for complete eccentric → [isometric] → concentric cycle
                while j < self.frame_count:
                    phase = self.phase_labels[j]
                    depth = self.analyzer.normalized_hip_displacement[j]
                    
                    # Track maximum depth (bottom of squat)
                    if depth > max_depth:
                        max_depth = depth
                        bottom_frame = j
                    
                    # Track phase progression
                    if phase == SquatPhase.ECCENTRIC.value:
                        pass  # Still in eccentric
                    
                    elif phase == SquatPhase.ISOMETRIC.value:
                        has_isometric = True
                    
                    elif phase == SquatPhase.CONCENTRIC.value:
                        has_concentric = True
                    
                    elif phase == SquatPhase.IDLE.value:
                        # ⭐ CRITICAL: We've returned to standing = end of rep
                        if has_concentric:
                            # Complete rep found!
                            rep_end = j - 1  # Last frame before returning to idle
                            
                            # Validate rep
                            rep_duration = rep_end - rep_start + 1
                            
                            if rep_duration >= MIN_REP_FRAMES and max_depth >= MIN_SQUAT_DEPTH_RATIO:
                                # Calculate knee angle depth
                                try:
                                    standing_angle = np.nanmedian(self.analyzer.knee_angles[:CALIBRATION_FRAMES])
                                    bottom_angle = self.analyzer.knee_angles[bottom_frame]
                                    angle_depth = standing_angle - bottom_angle if not np.isnan(standing_angle) and not np.isnan(bottom_angle) else 0
                                except:
                                    angle_depth = 0
                                    bottom_angle = np.nan
                                
                                # Check minimum angle depth
                                if angle_depth >= MIN_SQUAT_DEPTH_ANGLE:
                                    # ⭐ REP IS VALID!
                                    phases = self._extract_rep_phases(rep_start, rep_end, bottom_frame)
                                    
                                    rep = Repetition(
                                        rep_id=len(reps) + 1,
                                        start_frame=rep_start,
                                        end_frame=rep_end,
                                        phases=phases,
                                        squat_depth_normalized=max_depth,
                                        squat_depth_angle=angle_depth,
                                        bottom_frame=bottom_frame,
                                        bottom_knee_angle=bottom_angle if not np.isnan(bottom_angle) else 0.0
                                    )
                                    reps.append(rep)
                        
                        # Move past this IDLE, continue searching
                        i = j
                        break
                    
                    j += 1
                
                # ⭐ Handle end-of-video case: if we reached EOF without returning to IDLE
                if j >= self.frame_count:
                    if has_concentric:
                        rep_end = self.frame_count - 1
                        rep_duration = rep_end - rep_start + 1
                        
                        if rep_duration >= MIN_REP_FRAMES and max_depth >= MIN_SQUAT_DEPTH_RATIO:
                            try:
                                standing_angle = np.nanmedian(self.analyzer.knee_angles[:CALIBRATION_FRAMES])
                                bottom_angle = self.analyzer.knee_angles[bottom_frame]
                                angle_depth = standing_angle - bottom_angle if not np.isnan(standing_angle) and not np.isnan(bottom_angle) else 0
                            except:
                                angle_depth = 0
                                bottom_angle = np.nan
                            
                            if angle_depth >= MIN_SQUAT_DEPTH_ANGLE:
                                phases = self._extract_rep_phases(rep_start, rep_end, bottom_frame)
                                
                                rep = Repetition(
                                    rep_id=len(reps) + 1,
                                    start_frame=rep_start,
                                    end_frame=rep_end,
                                    phases=phases,
                                    squat_depth_normalized=max_depth,
                                    squat_depth_angle=angle_depth,
                                    bottom_frame=bottom_frame,
                                    bottom_knee_angle=bottom_angle if not np.isnan(bottom_angle) else 0.0
                                )
                                reps.append(rep)
                    
                    # Ensure we terminate the outer loop
                    i = self.frame_count
                    break
            else:
                # ⭐ CRITICAL SAFETY: Advance if frame is not handled by branches above 
                # (e.g. starts with ISOMETRIC or CONCENTRIC due to glitch)
                i += 1
        
        return reps
    
    def _detect_repetitions_phase_only(self) -> List[Repetition]:
        """
        Simple rep counting from phase sequence ONLY.

        Counts 1 rep for:
          - eccentric → concentric
          - eccentric → isometric → concentric

        Rep ends when:
          - concentric → idle
          - concentric → eccentric (next rep begins immediately)
          - end of video while in concentric

        This ignores depth/angle thresholds (by design) to avoid missing reps.
        """
        reps: List[Repetition] = []

        if self.phase_labels is None or len(self.phase_labels) == 0:
            return reps

        phase_ids = self.phase_labels  # ints: 0..4
        n = self.frame_count

        WAIT_ECC = 0
        IN_ECC = 1
        IN_ISO = 2
        IN_CONC = 3

        state = WAIT_ECC
        rep_start: Optional[int] = None

        def finalize_rep(start_idx: int, end_idx: int):
            if start_idx is None or end_idx is None:
                return
            if end_idx < start_idx:
                return
            if (end_idx - start_idx + 1) < MIN_REP_FRAMES_PHASE_ONLY:
                return

            # Choose bottom by max hip displacement (if available)
            disp = self.analyzer.normalized_hip_displacement
            seg = disp[start_idx:end_idx + 1] if disp is not None else None

            if seg is not None and len(seg) > 0:
                seg_safe = np.where(np.isnan(seg), -np.inf, seg)
                bottom_off = int(np.argmax(seg_safe))
                bottom_frame = start_idx + bottom_off
                max_depth = float(seg_safe[bottom_off]) if np.isfinite(seg_safe[bottom_off]) else 0.0
            else:
                bottom_frame = int((start_idx + end_idx) // 2)
                max_depth = 0.0

            # Knee angle info (best-effort, no filtering)
            bottom_knee_angle = 0.0
            angle_depth = 0.0
            try:
                if self.analyzer.knee_angles is not None and len(self.analyzer.knee_angles) > 0:
                    standing_angle = float(np.nanmedian(self.analyzer.knee_angles[:min(CALIBRATION_FRAMES, len(self.analyzer.knee_angles))]))
                    bottom_angle = float(self.analyzer.knee_angles[bottom_frame])
                    if not np.isnan(standing_angle) and not np.isnan(bottom_angle):
                        bottom_knee_angle = bottom_angle
                        angle_depth = standing_angle - bottom_angle
            except Exception:
                pass

            phases = self._extract_rep_phases(start_idx, end_idx, bottom_frame)

            reps.append(Repetition(
                rep_id=len(reps) + 1,
                start_frame=int(start_idx),
                end_frame=int(end_idx),
                phases=phases,
                squat_depth_normalized=float(max_depth),
                squat_depth_angle=float(angle_depth),
                bottom_frame=int(bottom_frame),
                bottom_knee_angle=float(bottom_knee_angle),
            ))

        for i in range(n):
            p = int(phase_ids[i])

            if state == WAIT_ECC:
                if p == SquatPhase.ECCENTRIC.value:
                    rep_start = i
                    state = IN_ECC
                else:
                    continue

            elif state == IN_ECC:
                if p == SquatPhase.ECCENTRIC.value:
                    continue
                if p == SquatPhase.ISOMETRIC.value:
                    state = IN_ISO
                    continue
                if p == SquatPhase.CONCENTRIC.value:
                    state = IN_CONC
                    continue
                if p == SquatPhase.IDLE.value:
                    # aborted: eccentric went back to idle without concentric
                    rep_start = None
                    state = WAIT_ECC
                    continue

            elif state == IN_ISO:
                if p == SquatPhase.ISOMETRIC.value:
                    continue
                if p == SquatPhase.CONCENTRIC.value:
                    state = IN_CONC
                    continue
                if p == SquatPhase.ECCENTRIC.value:
                    # treat as back to eccentric (bounce)
                    state = IN_ECC
                    continue
                if p == SquatPhase.IDLE.value:
                    # aborted: never hit concentric
                    rep_start = None
                    state = WAIT_ECC
                    continue

            elif state == IN_CONC:
                if p == SquatPhase.CONCENTRIC.value:
                    continue

                if p == SquatPhase.ISOMETRIC.value:
                    # Strict sequence does not allow concentric -> isometric.
                    # Keep state as concentric; sanitizer should already prevent this upstream.
                    continue

                if p == SquatPhase.IDLE.value:
                    # rep ends at last non-idle frame
                    finalize_rep(rep_start, i - 1)
                    rep_start = None
                    state = WAIT_ECC
                    continue

                if p == SquatPhase.ECCENTRIC.value:
                    # fast reps: next rep starts immediately, no idle in between
                    finalize_rep(rep_start, i - 1)
                    rep_start = i
                    state = IN_ECC
                    continue

                # unknown: ignore and keep going
                continue

        # End-of-video: if we were in concentric, count it
        if state == IN_CONC and rep_start is not None:
            finalize_rep(rep_start, n - 1)

        return reps

    def _extract_rep_phases(self, start: int, end: int, bottom: int) -> List[RepPhase]:
        """Extract phase breakdown for a single rep with transition reasons"""
        phases = []
        current_phase = None
        phase_start = start
        
        # Find relevant transitions for this rep
        rep_transitions = [t for t in self.state_machine.transition_log 
                         if start <= t["frame"] <= end]
        
        for i in range(start, end + 1):
            phase_id = int(self.phase_labels[i])
            phase_name = self.state_machine.get_phase_name(phase_id)
            
            if phase_name != current_phase and phase_name != "idle":
                if current_phase and current_phase != "idle":
                    # Find transition reason
                    reason = ""
                    for t in rep_transitions:
                        if t["to"] == current_phase and abs(t["frame"] - phase_start) < 5:
                            reason = t["reason"]
                            break
                    
                    phases.append(RepPhase(
                        phase_type=current_phase,
                        start_frame=phase_start,
                        end_frame=i - 1,
                        duration_frames=i - phase_start,
                        duration_seconds=(i - phase_start) / self.fps,
                        transition_reason=reason
                    ))
                
                current_phase = phase_name
                phase_start = i
        
        # Add final phase
        if current_phase and current_phase != "idle":
            reason = ""
            for t in rep_transitions:
                if t["to"] == current_phase:
                    reason = t["reason"]
                    break
            
            phases.append(RepPhase(
                phase_type=current_phase,
                start_frame=phase_start,
                end_frame=end,
                duration_frames=end - phase_start + 1,
                duration_seconds=(end - phase_start + 1) / self.fps,
                transition_reason=reason
            ))
        
        return phases


def process_video(json_path: str) -> Tuple[str, str, Optional[dict], Optional[str]]:
    """Process a single video's keypoints. Returns (video_id, status, result, quality)"""
    video_id = os.path.splitext(os.path.basename(json_path))[0]
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        segmenter = TemporalSegmenter(data, video_id)
        result = segmenter.segment()
        
        quality = result.get("info", {}).get("quality_rating", "unknown").lower()
        
        if "error" in result:
            return video_id, "Error", result, quality
        
        return video_id, "Success", result, quality
        
    except Exception as e:
        print(f"❌ Error processing {video_id}: {e}")
        return video_id, "Error", None, None


def create_segmentation_visualization(video_id: str, seg_data: dict, quality: str) -> bool:
    """Create annotated video with phase overlay, rep markers, and signal graphs"""
    
    video_path = find_video_file(video_id, quality)
    output_dir = VISUALIZATION_DIRS.get(quality.lower(), VISUALIZATION_DIR_EXCELLENT)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_id}_phases.mp4")
    
    if not video_path:
        return False
    
    # Extract segmentation data
    frame_phases = seg_data["frame_phases"]
    reps = seg_data["repetitions"]
    signals = seg_data.get("signals", {})
    info = seg_data.get("info", {})
    transition_log = seg_data.get("transition_log", [])
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Try mp4v first, fallback to XVID
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        output_path = output_path.replace('.mp4', '.avi')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    if not out.isOpened():
        cap.release()
        return False
    
    frame_idx = 0
    hip_displacement = signals.get("normalized_hip_displacement", [])
    window_velocity = signals.get("window_velocity", [])
    knee_angles = signals.get("knee_angles", [])
    
    view_type = info.get("view", "unknown").replace("_", " ").title()
    quality_display = info.get("quality_rating", "Unknown").upper()

    # Create transition frame lookup
    transition_frames = {t["frame"]: t for t in transition_log}
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx < len(frame_phases):
                phase = frame_phases[frame_idx]
                color = PHASE_COLORS.get(phase, (255, 255, 255))
                
                # Draw phase indicator bar with border
                cv2.rectangle(frame, (10, 10), (300, 65), (0, 0, 0), -1)
                cv2.rectangle(frame, (12, 12), (298, 63), color, -1)
                cv2.putText(frame, f"Phase: {phase.upper()}", (20, 48), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
                # Show transition reason if this is a transition frame
                if frame_idx in transition_frames:
                    trans = transition_frames[frame_idx]
                    reason_text = f"→ {trans['reason'][:50]}"
                    cv2.putText(frame, reason_text, (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Find current rep
                current_rep = None
                for rep in reps:
                    if rep["start_frame"] <= frame_idx <= rep["end_frame"]:
                        current_rep = rep
                        break
                
                if current_rep:
                    # Rep info box
                    rep_text = f"Rep {current_rep['rep_id']}"
                    depth_text = f"Depth: {current_rep['squat_depth_normalized']:.2f}"
                    cv2.rectangle(frame, (w - 220, 10), (w - 10, 70), (0, 0, 0), -1)
                    cv2.putText(frame, rep_text, (w - 210, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                    cv2.putText(frame, depth_text, (w - 210, 58),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    # Mark bottom position
                    if frame_idx == current_rep["bottom_frame"]:
                        cv2.putText(frame, "★ BOTTOM ★", (w//2 - 100, h - 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                
                # Signal display panel (bottom left)
                panel_y = h - 100
                cv2.rectangle(frame, (10, panel_y), (320, h - 10), (0, 0, 0, 180), -1)
                
                if frame_idx < len(hip_displacement):
                    disp_text = f"Hip Disp: {hip_displacement[frame_idx]:.3f}"
                    cv2.putText(frame, disp_text, (20, panel_y + 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)
                
                if frame_idx < len(window_velocity):
                    vel = window_velocity[frame_idx]
                    vel_color = (150, 150, 255) if vel > 0 else (255, 150, 150)
                    vel_text = f"Velocity: {vel:+.4f}"
                    cv2.putText(frame, vel_text, (20, panel_y + 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, vel_color, 1)
                
                if frame_idx < len(knee_angles):
                    angle_text = f"Knee: {knee_angles[frame_idx]:.1f}°"
                    cv2.putText(frame, angle_text, (20, panel_y + 75),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Frame counter and Metadata
                frame_text = f"Frame: {frame_idx+1}/{len(frame_phases)}"
                meta_text = f"View: {view_type} | Quality: {quality_display}"
                
                cv2.putText(frame, frame_text, (w - 200, h - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                cv2.putText(frame, meta_text, (w - 250, h - 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 200), 1)
            
            out.write(frame)
            frame_idx += 1
        
        cap.release()
        out.release()
        return True
        
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        cap.release()
        out.release()
        return False


def run_segmentation(quality_filter=None, create_visualization=True):
    """
    Process all extracted features and segment into reps using biomechanical analysis.
    
    Features:
    - Window-based phase detection (no per-frame classification)
    - Hysteresis and minimum duration enforcement
    - View-invariant normalized signals
    - FSM with valid transitions only
    - Output organized by quality (excellent/good/fair) to mirror input structure
    """
    # Create quality-specific output directories
    for quality_dir in OUTPUT_DIRS.values():
        os.makedirs(quality_dir, exist_ok=True)
    
    if create_visualization:
        for viz_dir in VISUALIZATION_DIRS.values():
            os.makedirs(viz_dir, exist_ok=True)
    
    # Determine which folders to process
    if quality_filter:
        folders_to_process = [f for f in FEATURES_DIRS 
                             if quality_filter.lower() in f.lower()]
    else:
        folders_to_process = FEATURES_DIRS
    
    # Collect JSON files with quality mapping
    json_files = []
    quality_mapping = {}  # Maps json_path to quality folder
    
    for folder in folders_to_process:
        if os.path.exists(folder):
            quality = "excellent" if "excellent" in folder else \
                      "good" if "good" in folder else \
                      "raw_unfiltered" if "raw_unfiltered" in folder else "fair"
            
            for f in os.listdir(folder):
                if f.endswith(".json"):
                    # ⭐ NEW: Filter by VIDEO_IDS_TO_PROCESS
                    video_id = os.path.splitext(f)[0]
                    
                    if VIDEO_IDS_TO_PROCESS != ["*"] and video_id not in VIDEO_IDS_TO_PROCESS:
                        continue  # Skip this video
                    
                    json_path = os.path.join(folder, f)
                    json_files.append(json_path)
                    quality_mapping[json_path] = quality
    
    if not json_files:
        print(f"❌ No videos found in {folders_to_process}")
        return
    
    print(f"\n{'='*70}")
    print("WINDOW-BASED BIOMECHANICAL TEMPORAL SEGMENTATION (v5.4)")
    print('='*70)
    print(f"⏱️  Processing {len(json_files)} videos...")
    if create_visualization:
        print(f"📹 Creating phase visualizations...")
    print(f"📁 Quality filters: {quality_filter if quality_filter else 'all'}")
    print(f"\n📊 Analysis Parameters:")
    print(f"   • Window size: {ANALYSIS_WINDOW_SIZE} frames")
    print(f"   • Min phase duration: {MIN_PHASE_DURATION_FRAMES} frames")
    print(f"   • Hysteresis: {HYSTERESIS_FRAMES} frames")
    print(f"   • Velocity thresholds: idle={VELOCITY_IDLE_THRESHOLD:.4f}, moving={VELOCITY_MOVING_THRESHOLD:.4f}")
    print()
    
    stats = {
        "success": 0,
        "error": 0,
        "view_rejected": 0,
        "total_reps": 0,
        "visualized": 0,
        "viz_failed": 0,
        "by_quality": {"excellent": 0, "good": 0, "fair": 0, "raw_unfiltered": 0},
        "phase_transitions": 0,
        "avg_transitions_per_rep": []
    }
    
    results_by_quality = {"excellent": [], "good": [], "fair": [], "raw_unfiltered": []}
    
    for idx, json_path in enumerate(tqdm(json_files, desc="Segmenting videos"), start=1):
        video_id = os.path.splitext(os.path.basename(json_path))[0]
        print(f"\n▶ [{idx}/{len(json_files)}] Segmenting {video_id}...", flush=True)

        video_id, status, result, quality = process_video(json_path)
        source_quality = quality_mapping.get(json_path, quality)
        
        if status == "Success" and result and "error" not in result:
            stats["success"] += 1
            info = result.get("info", {})
            reps = info.get("total_reps", 0)
            stats["total_reps"] += reps
            
            # Count transitions
            transitions = len(result.get("transition_log", []))
            stats["phase_transitions"] += transitions
            if reps > 0:
                stats["avg_transitions_per_rep"].append(transitions / reps)
            
            # Determine quality (from metadata or folder)
            quality_from_metadata = info.get("quality_rating", "unknown").lower()
            if source_quality == "raw_unfiltered":
                quality_to_use = "raw_unfiltered"
            else:
                quality_to_use = quality_from_metadata if quality_from_metadata != "unknown" else source_quality
            
            if quality_to_use in stats["by_quality"]:
                stats["by_quality"][quality_to_use] += 1
                results_by_quality[quality_to_use].append({
                    "video_id": video_id,
                    "reps": reps,
                    "transitions": transitions
                })
            
            # Save JSON output to quality-specific directory
            output_dir = OUTPUT_DIRS.get(quality_to_use, OUTPUT_DIR_EXCELLENT)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{video_id}_segmented.json")
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"   ✓ Saved segmentation JSON: {output_path}", flush=True)
            
            # Create visualization
            if create_visualization:
                if create_segmentation_visualization(video_id, result, quality_to_use):
                    stats["visualized"] += 1
                    print(f"   ✓ Visualization created for {video_id}", flush=True)
                else:
                    stats["viz_failed"] += 1
                    print(f"   ⚠ Visualization failed for {video_id}", flush=True)
            else:
                print(f"   ℹ Visualization disabled for this run", flush=True)
        
        elif result and "error" in result:
            if "view" in str(result.get("error", "")).lower():
                stats["view_rejected"] += 1
                print(f"\n  ⚠ {video_id} skipped: {result['error']}")
            else:
                stats["error"] += 1
                print(f"\n  ✗ {video_id}: {result['error']}")
        else:
            stats["error"] += 1
            print(f"\n  ✗ {video_id}: Unknown error")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SEGMENTATION RESULTS")
    print('='*70)
    print(f"✓ Successfully segmented: {stats['success']}")
    print(f"✗ Failed: {stats['error']}")
    print(f"⚠ View rejected: {stats['view_rejected']}")
    print(f"\n📊 Repetition Statistics:")
    print(f"   • Total reps detected: {stats['total_reps']}")
    print(f"   • Total phase transitions: {stats['phase_transitions']}")
    if stats['avg_transitions_per_rep']:
        avg_trans = np.mean(stats['avg_transitions_per_rep'])
        print(f"   • Avg transitions per rep: {avg_trans:.1f}")
    
    if create_visualization:
        print(f"\n📹 Visualization Results:")
        print(f"   ✓ Created: {stats['visualized']}")
        print(f"   ✗ Failed: {stats['viz_failed']}")
    
    if stats['success'] > 0:
        print(f"\n📈 By Quality:")
        for quality in ["excellent", "good", "fair", "raw_unfiltered"]:
            count = stats["by_quality"][quality]
            if count > 0:
                quality_results = results_by_quality[quality]
                total_reps = sum(r["reps"] for r in quality_results)
                avg_reps = total_reps / count if count > 0 else 0
                print(f"   {quality.capitalize()}: {count} videos, {total_reps} reps (avg: {avg_reps:.1f}/video)")
    
    print(f"\n📁 Output Structure:")
    print(f"   • Excellent: {OUTPUT_DIR_EXCELLENT}/")
    print(f"   • Good:      {OUTPUT_DIR_GOOD}/")
    print(f"   • Fair:      {OUTPUT_DIR_FAIR}/")
    print(f"   • Raw:       {OUTPUT_DIR_RAW_UNFILTERED}/")
    if create_visualization:
        print(f"\n📹 Visualization Structure:")
        print(f"   • Excellent: {VISUALIZATION_DIR_EXCELLENT}/")
        print(f"   • Good:      {VISUALIZATION_DIR_GOOD}/")
        print(f"   • Fair:      {VISUALIZATION_DIR_FAIR}/")
        print(f"   • Raw:       {VISUALIZATION_DIR_RAW_UNFILTERED}/")
    print('='*70 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Segment squat videos into repetitions.")
    parser.add_argument("--no-viz", action="store_true", help="Disable video visualization")
    parser.add_argument("--quality", choices=["excellent", "good", "fair", "raw_unfiltered"], help="Filter by quality")
    parser.add_argument("--video-id", help="Process only one video id (e.g., 25709_1)")
    args = parser.parse_args()

    # Priority: CLI flag > Env var > Default True
    create_viz = not args.no_viz
    if not create_viz:
        print("ℹ️  Visualization disabled via CLI")

    if args.video_id:
        VIDEO_IDS_TO_PROCESS[:] = [args.video_id]
        print(f"ℹ️  Video filter enabled: {args.video_id}")

    run_segmentation(quality_filter=args.quality, create_visualization=create_viz)