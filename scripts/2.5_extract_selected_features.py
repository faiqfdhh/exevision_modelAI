import os
import json
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import signal
import sys

# --- Suppress MediaPipe CPU Warnings ---
os.environ['GLOG_minloglevel'] = '2'

# --- Configuration ---
DATASET_ROOT = "./squat/dataset_videos_all"
OUTPUT_ROOT = "./squat/extracted_features_clean"
VISUALIZATION_OUTPUT_ROOT = "./squat/visualized_poses_clean"
ANALYSIS_OUTPUT_ROOT = "./squat/analysis_reports"
MODEL_PATH = os.environ.get("EXEVISION_MODEL_PATH", os.path.join('models', 'pose_landmarker_heavy.task'))
FACE_MODEL_PATH = os.environ.get("EXEVISION_FACE_MODEL_PATH", os.path.join('models', 'blaze_face_short_range.tflite'))
CREATE_VISUALIZATION = True
CREATE_ANALYSIS_REPORT = True  # Generate detailed analysis report

# Quality-based output organization
QUALITY_FOLDERS = {
    'Excellent': os.path.join(OUTPUT_ROOT, 'excellent'),
    'Good': os.path.join(OUTPUT_ROOT, 'good'),
    'Fair': os.path.join(OUTPUT_ROOT, 'fair'),
    'Poor': None  # Poor videos are not saved
}

# List of video IDs to process (without extension)
# Use ["*"] to process ALL videos, or specific IDs like ["25713_3", "46315_6"]
VIDEO_IDS = ["*"]

# Processing mode: "filtered" (default, with full processing) or
# "unfiltered" (One Euro smoothing only, no stability filtering)
PROCESSING_MODE = "filtered"

# One Euro Filter settings for lightly smoothed unfiltered mode.
ONE_EURO_SETTINGS = {
    'min_cutoff': 1.0,
    'beta': 0.5,
    'd_cutoff': 1.0,
}

# MediaPipe 33 landmarks connections
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]

# Squat-specific key joints (MediaPipe indices)
SQUAT_KEY_JOINTS = {
    'left_hip': 23,
    'right_hip': 24,
    'left_knee': 25,
    'right_knee': 26,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_shoulder': 11,
    'right_shoulder': 12,
    'left_heel': 29,
    'right_heel': 30,
    'left_toe': 31,
    'right_toe': 32
}

ANKLE_INDICES = (27, 28)
ANKLE_INFERENCE_VISIBILITY_THRESHOLD = 0.20
PLANTED_FOOT_LOCK_VISIBILITY_THRESHOLD = 0.60
LONG_CHAIN_MIN_RATIO = 0.50

# Foot region consolidation (merge ankle+heel+toe into unified foot landmark)
LEFT_FOOT_INDICES = [27, 29, 31]    # ankle, heel, toe
RIGHT_FOOT_INDICES = [28, 30, 32]   # ankle, heel, toe
CONSOLIDATED_FOOT_INDICES = [27, 28]  # Where consolidated left/right feet are placed

# Lenient stability thresholds for consolidated foot region (feet have higher motion variability)
FOOT_STABILITY_THRESHOLDS = {
    'max_normalized_jerk': 2.5,      # Much more lenient than default 1.35 (feet have high jerk during squat)
    'max_erratic_rate': 0.25,        # Much more lenient than default 0.10 (feet move erratically)
}

# Joint-specific visibility thresholds (higher for critical joints)
JOINT_THRESHOLDS = {
    'hip': 0.4,           # Lower body core
    'knee': 0.6,          # Critical for squat analysis
    'ankle': ANKLE_INFERENCE_VISIBILITY_THRESHOLD,  # Trust inferred ankles when cropped/occluded
    'shoulder': 0.3,      # Less critical but still important
    'heel': 0.4,          # Foot placement
    'toe': 0.3            # Least critical
}

# Thresholds for overall quality assessment
OVERALL_THRESHOLDS = {
    'excellent': 0.8,     # Excellent tracking
    'good': 0.6,          # Good for analysis
    'fair': 0.4,          # Some issues but usable
    'poor': 0.2           # Major tracking issues
}

# Mandatory lower-body chains (non-negotiable visibility gate)
MANDATORY_CORE_CHAINS = {
    'right_core': [24, 26],  # right hip-knee
    'left_core': [23, 25],   # left hip-knee
}
REFERENCE_FULL_CHAINS = {
    'right_full': [24, 26, 28, 32, 30],
    'left_full': [23, 25, 27, 29, 31],
}
MANDATORY_VISIBILITY_THRESHOLD = 0.4
MANDATORY_CHAIN_MIN_RATIO = 0.90

# Landmark stability filtering settings
LANDMARK_FILTER_SETTINGS = {
    'visibility_threshold': 0.4,
    'ankle_visibility_threshold': ANKLE_INFERENCE_VISIBILITY_THRESHOLD,
    'min_presence_ratio': 0.6,
    'max_switch_rate': 0.18,
    'max_normalized_jerk': 1.35,
}

def get_color(visibility, joint_type=None):
    """Get color based on visibility and joint importance"""
    if joint_type and joint_type in JOINT_THRESHOLDS:
        threshold = JOINT_THRESHOLDS[joint_type]
        if visibility > threshold * 1.2:
            return (0, 255, 0)    # Green - excellent
        elif visibility > threshold:
            return (0, 255, 255)  # Yellow - acceptable
        else:
            return (0, 0, 255)    # Red - poor
    else:
        # Generic coloring
        if visibility > 0.7:
            return (0, 255, 0)
        elif visibility > 0.4:
            return (0, 255, 255)
        else:
            return (0, 0, 255)

def get_joint_type(index):
    """Map landmark index to joint type for threshold selection"""
    for joint_name, joint_idx in SQUAT_KEY_JOINTS.items():
        if joint_idx == index:
            if 'hip' in joint_name:
                return 'hip'
            elif 'knee' in joint_name:
                return 'knee'
            elif 'ankle' in joint_name:
                return 'ankle'
            elif 'shoulder' in joint_name:
                return 'shoulder'
            elif 'heel' in joint_name:
                return 'heel'
            elif 'toe' in joint_name:
                return 'toe'
    return None

def draw_landmarks_enhanced(frame, landmarks, h, w, frame_idx, visibility_data):
    """Draw pose landmarks with enhanced visualization"""
    frame_copy = frame.copy()
    
    # Draw connections
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start_lm = landmarks[start_idx]
        end_lm = landmarks[end_idx]
        
        # Only draw if both points have reasonable visibility
        if start_lm[3] > 0.3 and end_lm[3] > 0.3:
            start_point = (int(start_lm[0] * w), int(start_lm[1] * h))
            end_point = (int(end_lm[0] * w), int(end_lm[1] * h))
            
            # Color based on average visibility of the two points
            avg_visibility = (start_lm[3] + end_lm[3]) / 2
            if avg_visibility > 0.7:
                color = (0, 255, 0)
            elif avg_visibility > 0.4:
                color = (0, 200, 200)
            else:
                color = (0, 100, 255)
            
            cv2.line(frame_copy, start_point, end_point, color, 2)
    
    # Draw landmark points with joint-specific coloring
    for idx, lm in enumerate(landmarks):
        x, y, z, visibility = lm
        
        if visibility > 0.3:
            point = (int(x * w), int(y * h))
            joint_type = get_joint_type(idx)
            color = get_color(visibility, joint_type)
            
            # Size based on importance
            if idx in SQUAT_KEY_JOINTS.values():
                radius = 5
            else:
                radius = 3
            
            cv2.circle(frame_copy, point, radius, color, -1)
            cv2.circle(frame_copy, point, radius + 1, (255, 255, 255), 1)
    
    # Add frame info with detailed visibility metrics
    if 'frame_metrics' in visibility_data:
        metrics = visibility_data['frame_metrics'][frame_idx]
        info_texts = [
            f"Frame: {frame_idx+1}/{len(landmarks)}",
            f"Overall: {metrics['overall']:.2f}",
            f"Key Joints: {metrics['key_joints']:.2f}",
            f"Lowest: {metrics['worst_joint']} ({metrics['worst_visibility']:.2f})"
        ]
        
        y_offset = 30
        for text in info_texts:
            cv2.putText(frame_copy, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame_copy, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            y_offset += 25
    
    # Add legend
    legend_texts = [
        "Green: Good (>thresh)", 
        "Yellow: Acceptable", 
        "Red: Poor (<thresh)"
    ]
    
    y_offset = h - 80
    for text in legend_texts:
        cv2.putText(frame_copy, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_offset += 20
    
    return frame_copy

def analyze_visibility_trends(visibility_scores, key_joint_scores):
    """Analyze visibility patterns across frames"""
    analysis = {
        'quality_rating': 'Unknown',
        'problem_frames': [],
        'continuous_problems': [],
        'joint_performance': {},
        'recommendations': []
    }
    
    # Overall quality rating
    avg_overall = np.mean(visibility_scores)
    avg_key = np.mean(key_joint_scores)
    
    if avg_overall >= OVERALL_THRESHOLDS['excellent'] and avg_key >= 0.7:
        analysis['quality_rating'] = 'Excellent'
    elif avg_overall >= OVERALL_THRESHOLDS['good'] and avg_key >= 0.6:
        analysis['quality_rating'] = 'Good'
    elif avg_overall >= OVERALL_THRESHOLDS['fair'] and avg_key >= 0.4:
        analysis['quality_rating'] = 'Fair'
    else:
        analysis['quality_rating'] = 'Poor'
    
    # Identify problem frames (overall visibility < 0.4 or key joints < 0.3)
    for i, (overall, key) in enumerate(zip(visibility_scores, key_joint_scores)):
        if overall < 0.4 or key < 0.3:
            analysis['problem_frames'].append({
                'frame': i,
                'overall_visibility': float(overall),
                'key_joints_visibility': float(key),
                'issue': 'Low visibility' if overall < 0.4 else 'Key joints occluded'
            })
    
    # Identify continuous problem segments (3+ consecutive problem frames)
    problem_segments = []
    current_segment = []
    for prob in analysis['problem_frames']:
        if not current_segment or prob['frame'] == current_segment[-1]['frame'] + 1:
            current_segment.append(prob)
        else:
            if len(current_segment) >= 3:
                problem_segments.append(current_segment)
            current_segment = [prob]
    
    if len(current_segment) >= 3:
        problem_segments.append(current_segment)
    
    for segment in problem_segments:
        analysis['continuous_problems'].append({
            'start_frame': segment[0]['frame'],
            'end_frame': segment[-1]['frame'],
            'duration_frames': len(segment),
            'avg_visibility': np.mean([f['overall_visibility'] for f in segment])
        })
    
    # Generate recommendations
    if analysis['quality_rating'] in ['Fair', 'Poor']:
        if len(analysis['continuous_problems']) > 0:
            analysis['recommendations'].append(
                f"Major occlusion in frames {analysis['continuous_problems'][0]['start_frame']}-"
                f"{analysis['continuous_problems'][0]['end_frame']}. Consider adjusting camera angle."
            )
        if avg_key < 0.5:
            analysis['recommendations'].append(
                "Key joints (knees, hips) have poor visibility. Ensure full body is in frame."
            )
    
    if analysis['quality_rating'] in ['Excellent', 'Good']:
        analysis['recommendations'].append("Video quality is sufficient for squat analysis.")
    
    return analysis, avg_overall, avg_key

def apply_savgol_filter(pose_data, window_length=11, polyorder=3):
    """
    Applies Savitzky-Golay filter to smooth 3D pose data across frames.
    Leaves visibility scores untouched.
    """
    if not pose_data or len(pose_data) < polyorder + 2:
        return pose_data

    if window_length > len(pose_data):
        window_length = len(pose_data)
    if window_length % 2 == 0:
        window_length -= 1

    if window_length <= polyorder:
        return pose_data

    data_np = np.array(pose_data)
    smoothed_xyz = savgol_filter(data_np[:, :, :3], window_length, polyorder, axis=0)
    smoothed_data = np.concatenate((smoothed_xyz, data_np[:, :, 3:]), axis=2)

    return smoothed_data.tolist()


class OneEuroFilter:
    """
    One Euro Filter (Casiez et al., 2012) for adaptive low-pass smoothing.

    Slow-moving signals receive stronger smoothing to suppress jitter,
    while fast-moving signals receive lighter smoothing to preserve motion.
    """

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _smoothing_factor(self, t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, t, x):
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        t_e = t - self.t_prev
        if t_e <= 0:
            t_e = 1e-6

        a_d = self._smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        speed = np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed

        a = self._smoothing_factor(t_e, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


def apply_one_euro_filter(pose_data, fps, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
    """
    Apply One Euro smoothing to pose landmark XYZ coordinates.
    Visibility values are preserved unchanged.
    """
    if not pose_data or len(pose_data) < 2:
        return pose_data

    data_np = np.array(pose_data, dtype=np.float64)
    if data_np.ndim != 3 or data_np.shape[2] < 4:
        return pose_data

    frame_count, joint_count, _ = data_np.shape
    dt = 1.0 / fps if fps and fps > 0 else 1.0 / 30.0

    for joint_idx in range(joint_count):
        for coord_idx in range(3):
            filt = OneEuroFilter(
                min_cutoff=min_cutoff,
                beta=beta,
                d_cutoff=d_cutoff,
            )
            for frame_idx in range(frame_count):
                timestamp = frame_idx * dt
                value = data_np[frame_idx, joint_idx, coord_idx]
                data_np[frame_idx, joint_idx, coord_idx] = filt(timestamp, value)

    return data_np.tolist()

def recover_planted_foot_ankles(
    pose_img_data,
    pose_world_data,
    lock_visibility_threshold=PLANTED_FOOT_LOCK_VISIBILITY_THRESHOLD,
    missing_visibility_threshold=ANKLE_INFERENCE_VISIBILITY_THRESHOLD,
):
    """
    Planted-foot recovery:
    lock ankle coordinates from high-confidence frames and reuse them
    when ankle visibility drops (cropped/occluded).
    """
    if not pose_img_data:
        return pose_img_data, pose_world_data, {
            'recovered_left_ankle_frames': 0,
            'recovered_right_ankle_frames': 0,
        }

    img_np = np.array(pose_img_data, dtype=np.float32)
    world_np = np.array(pose_world_data, dtype=np.float32)

    if img_np.ndim != 3 or img_np.shape[1] != 33 or img_np.shape[2] < 4:
        return pose_img_data, pose_world_data, {
            'recovered_left_ankle_frames': 0,
            'recovered_right_ankle_frames': 0,
        }

    locked_img_xyz = {}
    locked_world_xyz = {}
    recovered_counts = {27: 0, 28: 0}

    for frame_idx in range(img_np.shape[0]):
        for ankle_idx in ANKLE_INDICES:
            visibility = float(img_np[frame_idx, ankle_idx, 3])

            if visibility >= lock_visibility_threshold:
                locked_img_xyz[ankle_idx] = img_np[frame_idx, ankle_idx, :3].copy()
                if world_np.ndim == 3 and world_np.shape[1] == 33 and world_np.shape[2] >= 3:
                    locked_world_xyz[ankle_idx] = world_np[frame_idx, ankle_idx, :3].copy()
                continue

            if visibility < missing_visibility_threshold and ankle_idx in locked_img_xyz:
                img_np[frame_idx, ankle_idx, :3] = locked_img_xyz[ankle_idx]
                img_np[frame_idx, ankle_idx, 3] = missing_visibility_threshold

                if ankle_idx in locked_world_xyz and world_np.ndim == 3 and world_np.shape[1] == 33 and world_np.shape[2] >= 3:
                    world_np[frame_idx, ankle_idx, :3] = locked_world_xyz[ankle_idx]
                    if world_np.shape[2] >= 4:
                        world_np[frame_idx, ankle_idx, 3] = max(
                            float(world_np[frame_idx, ankle_idx, 3]),
                            missing_visibility_threshold,
                        )

                recovered_counts[ankle_idx] += 1

    return img_np.tolist(), world_np.tolist(), {
        'recovered_left_ankle_frames': int(recovered_counts[27]),
        'recovered_right_ankle_frames': int(recovered_counts[28]),
    }

def build_mandatory_chain_flags(
    pose_img_data,
    visibility_threshold=MANDATORY_VISIBILITY_THRESHOLD,
    ankle_visibility_threshold=ANKLE_INFERENCE_VISIBILITY_THRESHOLD,
):
    """Build per-frame core chain visibility flags with lower ankle threshold."""
    if not pose_img_data:
        return []

    flags = []
    for frame in pose_img_data:
        if not frame or len(frame) < 29:
            flags.append((False, False))
            continue

        vis = [lm[3] if len(lm) > 3 else 0.0 for lm in frame]

        def chain_ok(indices):
            for idx in indices:
                threshold = ankle_visibility_threshold if idx in ANKLE_INDICES else visibility_threshold
                if vis[idx] < threshold:
                    return False
            return True

        flags.append((
            chain_ok(MANDATORY_CORE_CHAINS['right_core']),
            chain_ok(MANDATORY_CORE_CHAINS['left_core']),
        ))

    return flags

def consolidate_foot_region(pose_img_data, pose_world_data):
    """
    Consolidate foot region (ankle + heel + toe) into unified foot landmarks.
    Merges 3 foot indices into 1 per foot with weighted visibility.
    
    Right foot: 28 (ankle), 30 (heel), 32 (toe) -> consolidate into 28
    Left foot: 27 (ankle), 29 (heel), 31 (toe) -> consolidate into 27
    
    Consolidation strategy:
    - Average XYZ coordinates weighted by visibility
    - Use max visibility among the 3 foot points
    - Zero out heel (29, 30) and toe (31, 32) indices after consolidation
    """
    if not pose_img_data:
        return pose_img_data, pose_world_data, {
            'left_foot_consolidated': 0,
            'right_foot_consolidated': 0,
        }
    
    img_np = np.array(pose_img_data, dtype=np.float32)
    world_np = np.array(pose_world_data, dtype=np.float32)
    
    if img_np.ndim != 3 or img_np.shape[1] != 33 or img_np.shape[2] < 4:
        return pose_img_data, pose_world_data, {
            'left_foot_consolidated': 0,
            'right_foot_consolidated': 0,
        }
    
    consolidated_counts = {'left': 0, 'right': 0}
    
    for frame_idx in range(img_np.shape[0]):
        # === LEFT FOOT CONSOLIDATION (27, 29, 31) ===
        left_foot_lms = [
            img_np[frame_idx, 27, :],   # ankle (27)
            img_np[frame_idx, 29, :],   # heel (29)
            img_np[frame_idx, 31, :],   # toe (31)
        ]
        left_foot_vis = np.array([lm[3] for lm in left_foot_lms])
        
        if np.any(left_foot_vis > 0):
            # Weighted average: weight by visibility
            weight_sum = np.sum(left_foot_vis)
            if weight_sum > 0:
                # Normalize weights
                weights = left_foot_vis / weight_sum
                # Average XYZ
                consolidated_xyz = np.sum(
                    np.array([lm[:3] for lm in left_foot_lms]) * weights[:, np.newaxis],
                    axis=0
                )
                # Use max visibility (most confident point in this foot region)
                consolidated_vis = np.max(left_foot_vis)
                
                # Place consolidated foot at index 27 (ankle)
                img_np[frame_idx, 27, :3] = consolidated_xyz
                img_np[frame_idx, 27, 3] = consolidated_vis
                
                # Zero out heel and toe
                img_np[frame_idx, 29, :] = 0.0
                img_np[frame_idx, 31, :] = 0.0
                
                # Same for world space
                if world_np.ndim == 3 and world_np.shape[1] == 33 and world_np.shape[2] >= 4:
                    left_foot_world_lms = [
                        world_np[frame_idx, 27, :],
                        world_np[frame_idx, 29, :],
                        world_np[frame_idx, 31, :],
                    ]
                    left_foot_world_vis = np.array([lm[3] if lm.shape[0] > 3 else 0.0 for lm in left_foot_world_lms])
                    
                    if np.any(left_foot_world_vis > 0):
                        world_weight_sum = np.sum(left_foot_world_vis)
                        if world_weight_sum > 0:
                            world_weights = left_foot_world_vis / world_weight_sum
                            consolidated_world_xyz = np.sum(
                                np.array([lm[:3] for lm in left_foot_world_lms]) * world_weights[:, np.newaxis],
                                axis=0
                            )
                            consolidated_world_vis = np.max(left_foot_world_vis)
                            
                            world_np[frame_idx, 27, :3] = consolidated_world_xyz
                            if world_np.shape[2] > 3:
                                world_np[frame_idx, 27, 3] = consolidated_world_vis
                            
                            world_np[frame_idx, 29, :] = 0.0
                            world_np[frame_idx, 31, :] = 0.0
                
                consolidated_counts['left'] += 1
        else:
            # No visibility in any left foot point, zero all
            img_np[frame_idx, 27, :] = 0.0
            img_np[frame_idx, 29, :] = 0.0
            img_np[frame_idx, 31, :] = 0.0
            if world_np.ndim == 3:
                world_np[frame_idx, 27, :] = 0.0
                world_np[frame_idx, 29, :] = 0.0
                world_np[frame_idx, 31, :] = 0.0
        
        # === RIGHT FOOT CONSOLIDATION (28, 30, 32) ===
        right_foot_lms = [
            img_np[frame_idx, 28, :],   # ankle (28)
            img_np[frame_idx, 30, :],   # heel (30)
            img_np[frame_idx, 32, :],   # toe (32)
        ]
        right_foot_vis = np.array([lm[3] for lm in right_foot_lms])
        
        if np.any(right_foot_vis > 0):
            weight_sum = np.sum(right_foot_vis)
            if weight_sum > 0:
                weights = right_foot_vis / weight_sum
                consolidated_xyz = np.sum(
                    np.array([lm[:3] for lm in right_foot_lms]) * weights[:, np.newaxis],
                    axis=0
                )
                consolidated_vis = np.max(right_foot_vis)
                
                # Place consolidated foot at index 28 (ankle)
                img_np[frame_idx, 28, :3] = consolidated_xyz
                img_np[frame_idx, 28, 3] = consolidated_vis
                
                # Zero out heel and toe
                img_np[frame_idx, 30, :] = 0.0
                img_np[frame_idx, 32, :] = 0.0
                
                # Same for world space
                if world_np.ndim == 3 and world_np.shape[1] == 33 and world_np.shape[2] >= 4:
                    right_foot_world_lms = [
                        world_np[frame_idx, 28, :],
                        world_np[frame_idx, 30, :],
                        world_np[frame_idx, 32, :],
                    ]
                    right_foot_world_vis = np.array([lm[3] if lm.shape[0] > 3 else 0.0 for lm in right_foot_world_lms])
                    
                    if np.any(right_foot_world_vis > 0):
                        world_weight_sum = np.sum(right_foot_world_vis)
                        if world_weight_sum > 0:
                            world_weights = right_foot_world_vis / world_weight_sum
                            consolidated_world_xyz = np.sum(
                                np.array([lm[:3] for lm in right_foot_world_lms]) * world_weights[:, np.newaxis],
                                axis=0
                            )
                            consolidated_world_vis = np.max(right_foot_world_vis)
                            
                            world_np[frame_idx, 28, :3] = consolidated_world_xyz
                            if world_np.shape[2] > 3:
                                world_np[frame_idx, 28, 3] = consolidated_world_vis
                            
                            world_np[frame_idx, 30, :] = 0.0
                            world_np[frame_idx, 32, :] = 0.0
                
                consolidated_counts['right'] += 1
        else:
            # No visibility in any right foot point, zero all
            img_np[frame_idx, 28, :] = 0.0
            img_np[frame_idx, 30, :] = 0.0
            img_np[frame_idx, 32, :] = 0.0
            if world_np.ndim == 3:
                world_np[frame_idx, 28, :] = 0.0
                world_np[frame_idx, 30, :] = 0.0
                world_np[frame_idx, 32, :] = 0.0
    
    return img_np.tolist(), world_np.tolist(), {
        'left_foot_consolidated': int(consolidated_counts['left']),
        'right_foot_consolidated': int(consolidated_counts['right']),
    }

def longest_true_run(flags_1d):
    best = 0
    current = 0
    for value in flags_1d:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)

def filter_unstable_landmarks(
    pose_img_data,
    pose_world_data,
    visibility_threshold=0.4,
    ankle_visibility_threshold=ANKLE_INFERENCE_VISIBILITY_THRESHOLD,
    min_presence_ratio=0.5,
    max_switch_rate=0.10,
    max_normalized_jerk=5.00,
    max_erratic_rate=1.0,
):
    """
    Filters unstable landmarks across the full clip:
    1. First consolidates foot region (ankle+heel+toe -> unified foot landmark)
    2. Then applies per-landmark stability filtering with lenient thresholds for feet
    """
    if not pose_img_data:
        return pose_img_data, pose_world_data, {
            'stable_indices': [],
            'unstable_indices': [],
            'stable_count': 0,
            'unstable_count': 0,
            'foot_consolidation': {
                'left_foot_consolidated': 0,
                'right_foot_consolidated': 0,
            }
        }

    # Step 1: Consolidate foot region (merge ankle/heel/toe)
    pose_img_data, pose_world_data, consolidation_summary = consolidate_foot_region(
        pose_img_data, pose_world_data
    )
    
    # Step 2: Apply stability filtering with foot-specific lenient thresholds
    img_np = np.array(pose_img_data, dtype=np.float32)  # (frames, 33, 4)
    world_np = np.array(pose_world_data, dtype=np.float32)

    if img_np.ndim != 3 or img_np.shape[1] != 33 or img_np.shape[2] < 4:
        return pose_img_data, pose_world_data, {
            'stable_indices': list(range(33)),
            'unstable_indices': [],
            'stable_count': 33,
            'unstable_count': 0,
            'foot_consolidation': consolidation_summary,
        }

    visibility = img_np[:, :, 3]
    per_joint_threshold = np.full(33, visibility_threshold, dtype=np.float32)
    per_joint_threshold[list(ANKLE_INDICES)] = ankle_visibility_threshold
    present = visibility >= per_joint_threshold[np.newaxis, :]
    presence_ratio = np.mean(present, axis=0)

    if img_np.shape[0] > 1:
        switch_count = np.sum(np.abs(np.diff(present.astype(np.int16), axis=0)), axis=0)
        switch_rate = switch_count / (img_np.shape[0] - 1)
    else:
        switch_rate = np.zeros(33, dtype=np.float32)

    xyz = img_np[:, :, :3]
    if img_np.shape[0] > 2:
        velocity = np.diff(xyz, axis=0)
        accel = np.diff(velocity, axis=0)
        jerk_mag = np.linalg.norm(accel, axis=2)
        jerk_score = np.median(jerk_mag, axis=0)
        jerk_norm_base = np.median(jerk_score) + 1e-6
        normalized_jerk = jerk_score / jerk_norm_base

        global_jerk_base = np.median(jerk_mag) + 1e-6
        erratic_flags = jerk_mag > (3.0 * global_jerk_base)
        erratic_rate = np.mean(erratic_flags, axis=0)
    else:
        normalized_jerk = np.ones(33, dtype=np.float32)
        erratic_rate = np.zeros(33, dtype=np.float32)

    # Apply standard thresholds to all landmarks
    stable_mask = (
        (presence_ratio >= min_presence_ratio)
        & (switch_rate <= max_switch_rate)
        & (normalized_jerk <= max_normalized_jerk)
        & (erratic_rate <= max_erratic_rate)
    )
    
    # Apply LENIENT thresholds to consolidated foot region (indices 27, 28)
    # Feet have inherently high motion during squat, so we need much more lenient limits
    for foot_idx in CONSOLIDATED_FOOT_INDICES:
        foot_stable = (
            (presence_ratio[foot_idx] >= min_presence_ratio)
            & (switch_rate[foot_idx] <= max_switch_rate)
            & (normalized_jerk[foot_idx] <= FOOT_STABILITY_THRESHOLDS['max_normalized_jerk'])
            & (erratic_rate[foot_idx] <= FOOT_STABILITY_THRESHOLDS['max_erratic_rate'])
        )
        stable_mask[foot_idx] = foot_stable

    # Per-landmark filtering (no chain-level discard).
    # Each landmark passes/fails individually based on stability metrics.
    #
    # Face landmarks (0-10) are intentionally excluded from zeroing.
    # They are only used for camera-view classification (stage 4), and head movement
    # during a squat naturally triggers the motion instability filter — that should
    # not erase face visibility information which the classifier depends on.
    FACE_LANDMARK_INDICES = set(range(11))  # MediaPipe 0-10: nose, eyes, ears, mouth, neck

    filtered_img = img_np.copy()
    filtered_world = world_np.copy()
    unstable_idx = np.where(~stable_mask)[0]
    body_unstable_idx = np.array([i for i in unstable_idx if i not in FACE_LANDMARK_INDICES])
    if body_unstable_idx.size > 0:
        filtered_img[:, body_unstable_idx, :] = 0.0
        filtered_world[:, body_unstable_idx, :] = 0.0

    stable_indices = np.where(stable_mask)[0].tolist()
    unstable_indices = unstable_idx.tolist()

    summary = {
        'stable_indices': stable_indices,
        'unstable_indices': unstable_indices,
        'stable_count': len(stable_indices),
        'unstable_count': len(unstable_indices),
        'foot_consolidation': consolidation_summary,
    }

    return filtered_img.tolist(), filtered_world.tolist(), summary

def evaluate_mandatory_chain_gate(
    frame_chain_flags,
    stability_summary,
    min_ratio=0.90,
    long_chain_min_ratio=LONG_CHAIN_MIN_RATIO,
):
    """
    Evaluates non-negotiable chain continuity gate:
    both core chains (right 24-26-28 and left 23-25-27) must be visible >= min_ratio.
    """
    if not frame_chain_flags:
        return {
            'right_core_ratio': 0.0,
            'left_core_ratio': 0.0,
            'both_core_ratio': 0.0,
            'min_ratio_required': float(min_ratio),
            'long_chain_min_ratio': float(long_chain_min_ratio),
            'right_longest_chain': 0,
            'left_longest_chain': 0,
            'both_longest_chain': 0,
            'passes_gate': False,
        }

    flags_np = np.array(frame_chain_flags, dtype=bool)
    right_ratio = float(np.mean(flags_np[:, 0]))
    left_ratio = float(np.mean(flags_np[:, 1]))
    both_flags = np.logical_and(flags_np[:, 0], flags_np[:, 1])
    both_ratio = float(np.mean(both_flags))

    right_longest_chain = longest_true_run(flags_np[:, 0].tolist())
    left_longest_chain = longest_true_run(flags_np[:, 1].tolist())
    both_longest_chain = longest_true_run(both_flags.tolist())
    min_long_chain_frames = max(5, int(np.ceil(flags_np.shape[0] * long_chain_min_ratio)))

    ideal_pass = (
        right_ratio >= min_ratio
        and left_ratio >= min_ratio
        and both_ratio >= min_ratio
    )

    fallback_single_chain_pass = (
        (right_longest_chain >= min_long_chain_frames)
        or (left_longest_chain >= min_long_chain_frames)
    )

    passes = ideal_pass or fallback_single_chain_pass

    return {
        'right_core_ratio': right_ratio,
        'left_core_ratio': left_ratio,
        'both_core_ratio': both_ratio,
        'min_ratio_required': float(min_ratio),
        'long_chain_min_ratio': float(long_chain_min_ratio),
        'min_long_chain_frames': int(min_long_chain_frames),
        'right_longest_chain': int(right_longest_chain),
        'left_longest_chain': int(left_longest_chain),
        'both_longest_chain': int(both_longest_chain),
        'ideal_both_chains_pass': bool(ideal_pass),
        'fallback_single_chain_pass': bool(fallback_single_chain_pass),
        'passes_gate': bool(passes),
    }

def get_mediapipe_options():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    return vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )

def get_face_detector_options():
    base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    return vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5
    )

def find_video_path(video_id):
    for root, dirs, files in os.walk(DATASET_ROOT):
        for ext in (".mp4", ".mov", ".avi", ".mkv", ".flv"):
            filename = f"{video_id}{ext}"
            if filename in files:
                return os.path.join(root, filename)
    return None

def create_visualization_report(video_id, analysis, visibility_scores, key_joint_scores):
    """Create a comprehensive visualization report"""
    os.makedirs(ANALYSIS_OUTPUT_ROOT, exist_ok=True)
    
    # Create visibility trend plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Visibility Analysis - {video_id}', fontsize=16, fontweight='bold')
    
    # Plot 1: Overall visibility trend
    axes[0, 0].plot(visibility_scores, label='Overall', color='blue', alpha=0.7)
    axes[0, 0].plot(key_joint_scores, label='Key Joints', color='red', alpha=0.7)
    axes[0, 0].axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Good threshold')
    axes[0, 0].axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Fair threshold')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Visibility Score')
    axes[0, 0].set_title('Visibility Trend Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram of visibility scores
    axes[0, 1].hist(visibility_scores, bins=20, alpha=0.7, color='blue', label='Overall')
    axes[0, 1].hist(key_joint_scores, bins=20, alpha=0.7, color='red', label='Key Joints')
    axes[0, 1].set_xlabel('Visibility Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Visibility Scores')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Problem frames
    if analysis['problem_frames']:
        problem_frames = [p['frame'] for p in analysis['problem_frames']]
        problem_scores = [p['overall_visibility'] for p in analysis['problem_frames']]
        axes[1, 0].scatter(problem_frames, problem_scores, color='red', s=50, 
                          label='Problem Frames', zorder=5)
        axes[1, 0].plot(visibility_scores, color='blue', alpha=0.3, label='Overall Trend')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Visibility Score')
        axes[1, 0].set_title('Problem Frames Identified')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Quality metrics
    metrics = {
        'Overall Avg': np.mean(visibility_scores),
        'Key Joints Avg': np.mean(key_joint_scores),
        'Min Overall': np.min(visibility_scores),
        'Max Overall': np.max(visibility_scores),
        'Std Dev': np.std(visibility_scores)
    }
    
    bars = axes[1, 1].bar(range(len(metrics)), list(metrics.values()))
    axes[1, 1].set_xticks(range(len(metrics)))
    axes[1, 1].set_xticklabels(list(metrics.keys()), rotation=45, ha='right')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Quality Metrics')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics.values()):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(ANALYSIS_OUTPUT_ROOT, f"{video_id}_visibility_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create text report
    report_path = os.path.join(ANALYSIS_OUTPUT_ROOT, f"{video_id}_analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"=== SQUAT ANALYSIS REPORT ===\n")
        f.write(f"Video ID: {video_id}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n--- QUALITY ASSESSMENT ---\n")
        f.write(f"Overall Rating: {analysis['quality_rating']}\n")
        f.write(f"Average Overall Visibility: {np.mean(visibility_scores):.3f}\n")
        f.write(f"Average Key Joints Visibility: {np.mean(key_joint_scores):.3f}\n")
        f.write(f"Frames Analyzed: {len(visibility_scores)}\n")
        
        f.write(f"\n--- PROBLEM AREAS ---\n")
        if analysis['problem_frames']:
            f.write(f"Total Problem Frames: {len(analysis['problem_frames'])} ({len(analysis['problem_frames'])/len(visibility_scores)*100:.1f}%)\n")
            if analysis['continuous_problems']:
                f.write("\nContinuous Problem Segments:\n")
                for segment in analysis['continuous_problems']:
                    f.write(f"  Frames {segment['start_frame']}-{segment['end_frame']} "
                           f"({segment['duration_frames']} frames, avg visibility: {segment['avg_visibility']:.3f})\n")
        else:
            f.write("No significant problems detected.\n")
        
        f.write(f"\n--- RECOMMENDATIONS ---\n")
        for i, rec in enumerate(analysis['recommendations'], 1):
            f.write(f"{i}. {rec}\n")
        
        f.write(f"\n--- DETAILED METRICS ---\n")
        f.write(f"Visibility Range: {np.min(visibility_scores):.3f} - {np.max(visibility_scores):.3f}\n")
        f.write(f"Standard Deviation: {np.std(visibility_scores):.3f}\n")
        f.write(f"Frames with visibility < 0.4: {sum(v < 0.4 for v in visibility_scores)}\n")
        f.write(f"Frames with key joints visibility < 0.3: {sum(v < 0.3 for v in key_joint_scores)}\n")
    
    return plot_path, report_path

def process_single_video(vid_path, mode="filtered"):
    try:
        vid_id = os.path.splitext(os.path.basename(vid_path))[0]

        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            return vid_path, "Failed to open video", None, None, None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        data_img_space = []
        data_world_space = []
        visibility_scores = []
        key_joint_scores = []
        frames_for_viz = []
        frame_metrics = []
        mandatory_chain_flags = []
        face_detected_per_frame = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with vision.PoseLandmarker.create_from_options(get_mediapipe_options()) as landmarker, \
             vision.FaceDetector.create_from_options(get_face_detector_options()) as face_detector:
            for frame_idx in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                
                if CREATE_VISUALIZATION:
                    frames_for_viz.append(frame.copy())
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                
                # Face Detection
                face_detection_result = face_detector.detect(mp_image)
                face_detected_per_frame.append(len(face_detection_result.detections) > 0)
                
                # Pose Landmarking
                detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                
                if detection_result.pose_landmarks:
                    frame_img = [[lm.x, lm.y, lm.z, lm.visibility] 
                                for lm in detection_result.pose_landmarks[0]]
                    frame_world = [[lm.x, lm.y, lm.z, lm.visibility] 
                                  for lm in detection_result.pose_world_landmarks[0]]
                    
                    # Calculate detailed visibility metrics
                    visibilities = [lm.visibility for lm in detection_result.pose_landmarks[0]]
                    overall_visibility = np.mean(visibilities)
                    
                    # Key joints visibility (squat-specific)
                    key_indices = list(SQUAT_KEY_JOINTS.values())
                    key_visibilities = [visibilities[i] for i in key_indices]
                    key_visibility = np.mean(key_visibilities)
                    
                    # Find worst joint in this frame
                    worst_joint_idx = np.argmin(key_visibilities)
                    worst_joint_name = list(SQUAT_KEY_JOINTS.keys())[worst_joint_idx]
                    worst_visibility = key_visibilities[worst_joint_idx]
                    
                    visibility_scores.append(overall_visibility)
                    key_joint_scores.append(key_visibility)
                    
                    frame_metrics.append({
                        'overall': overall_visibility,
                        'key_joints': key_visibility,
                        'worst_joint': worst_joint_name,
                        'worst_visibility': worst_visibility
                    })

                    right_core_ok = all(
                        visibilities[idx] >= MANDATORY_VISIBILITY_THRESHOLD
                        for idx in MANDATORY_CORE_CHAINS['right_core']
                    )
                    left_core_ok = all(
                        visibilities[idx] >= MANDATORY_VISIBILITY_THRESHOLD
                        for idx in MANDATORY_CORE_CHAINS['left_core']
                    )
                    mandatory_chain_flags.append((right_core_ok, left_core_ok))
                else:
                    frame_img = [[0.0, 0.0, 0.0, 0.0] for _ in range(33)]
                    frame_world = [[0.0, 0.0, 0.0, 0.0] for _ in range(33)]
                    visibility_scores.append(0.0)
                    key_joint_scores.append(0.0)
                    frame_metrics.append({
                        'overall': 0.0,
                        'key_joints': 0.0,
                        'worst_joint': 'None',
                        'worst_visibility': 0.0
                    })
                    mandatory_chain_flags.append((False, False))
                
                data_img_space.append(frame_img)
                data_world_space.append(frame_world)

        cap.release()

        # In unfiltered mode, apply only One Euro smoothing and skip the
        # heavier recovery/filtering pipeline used by filtered mode.
        if mode == "unfiltered":
            data_img_space = apply_one_euro_filter(
                data_img_space,
                fps,
                min_cutoff=ONE_EURO_SETTINGS['min_cutoff'],
                beta=ONE_EURO_SETTINGS['beta'],
                d_cutoff=ONE_EURO_SETTINGS['d_cutoff'],
            )
            data_world_space = apply_one_euro_filter(
                data_world_space,
                fps,
                min_cutoff=ONE_EURO_SETTINGS['min_cutoff'],
                beta=ONE_EURO_SETTINGS['beta'],
                d_cutoff=ONE_EURO_SETTINGS['d_cutoff'],
            )
            print(f"[{vid_id}] Lightly smoothed mode: One Euro Filter applied (no stability filtering)")
            stability_summary = {}
            planted_foot_summary = {}
            mandatory_chain_summary = {}
            mandatory_chain_warning = None
            analysis = {
                'quality_rating': 'Raw',
                'problem_frames': [],
                'continuous_problems': [],
                'recommendations': ['One Euro smoothed MediaPipe output without stability filtering'],
            }
            avg_overall = np.mean(visibility_scores) if visibility_scores else 0.0
            avg_key = np.mean(key_joint_scores) if key_joint_scores else 0.0
        else:
            # --- Recover cropped/occluded ankles by planted-foot coordinate locking ---
            data_img_space, data_world_space, planted_foot_summary = recover_planted_foot_ankles(
                data_img_space,
                data_world_space,
                lock_visibility_threshold=PLANTED_FOOT_LOCK_VISIBILITY_THRESHOLD,
                missing_visibility_threshold=ANKLE_INFERENCE_VISIBILITY_THRESHOLD,
            )

            # Rebuild mandatory chain flags after ankle recovery.
            mandatory_chain_flags = build_mandatory_chain_flags(
                data_img_space,
                visibility_threshold=MANDATORY_VISIBILITY_THRESHOLD,
                ankle_visibility_threshold=ANKLE_INFERENCE_VISIBILITY_THRESHOLD,
            )

            # --- Apply temporal smoothing to pose coordinates (XYZ only) ---
            data_img_space = apply_savgol_filter(data_img_space, window_length=11, polyorder=3)
            data_world_space = apply_savgol_filter(data_world_space, window_length=11, polyorder=3)

            # --- Keep only stable landmarks and discard unstable/jittery ones ---
            data_img_space, data_world_space, stability_summary = filter_unstable_landmarks(
                data_img_space,
                data_world_space,
                visibility_threshold=LANDMARK_FILTER_SETTINGS['visibility_threshold'],
                ankle_visibility_threshold=LANDMARK_FILTER_SETTINGS['ankle_visibility_threshold'],
                min_presence_ratio=LANDMARK_FILTER_SETTINGS['min_presence_ratio'],
                max_switch_rate=LANDMARK_FILTER_SETTINGS['max_switch_rate'],
                max_normalized_jerk=LANDMARK_FILTER_SETTINGS['max_normalized_jerk'],
                max_erratic_rate=0.10,
            )
            stability_summary['planted_foot_recovery'] = planted_foot_summary
            print(
                f"[{vid_id}] Stable landmarks: {stability_summary['stable_count']}/33, "
                f"Discarded unstable: {stability_summary['unstable_count']}"
            )

            mandatory_chain_summary = evaluate_mandatory_chain_gate(
                mandatory_chain_flags,
                stability_summary,
                min_ratio=MANDATORY_CHAIN_MIN_RATIO,
            )

            mandatory_chain_warning = None

            if not mandatory_chain_summary['passes_gate']:
                reason = (
                    "Mandatory chain gate failed (ideal both-chains continuity not met). "
                    f"Right(24-26-28): {mandatory_chain_summary['right_core_ratio']:.1%}, "
                    f"Left(23-25-27): {mandatory_chain_summary['left_core_ratio']:.1%}, "
                    f"Both together: {mandatory_chain_summary['both_core_ratio']:.1%}, "
                    f"Required: >= {MANDATORY_CHAIN_MIN_RATIO:.0%}"
                )
                if mandatory_chain_summary.get('both_chains_discarded', False):
                    reason = (
                        "UNRELIABLE VIDEO: Both mandatory chains (24-26-28 and 23-25-27) were discarded. "
                        "This video is unreliable for analysis. "
                        + reason
                    )
                    print(f"⚠️  {vid_id}: {reason}")
                mandatory_chain_warning = reason
        
            # Analyze visibility trends
            analysis, avg_overall, avg_key = analyze_visibility_trends(visibility_scores, key_joint_scores)
            quality_rating = analysis['quality_rating']
        
        if mode == "unfiltered":
            quality_rating = "Raw"
        
        # Skip poor videos - don't save anything (unless in unfiltered mode)
        if quality_rating == 'Poor':
            cap.release()
            return vid_id, "Skipped", f"Poor quality: {avg_overall:.2f}", analysis, avg_overall
        
        # Determine output folder based on quality
        if mode == "unfiltered":
            quality_output_root = os.path.join(OUTPUT_ROOT, "raw_unfiltered")
        else:
            quality_output_root = QUALITY_FOLDERS[quality_rating]
        os.makedirs(quality_output_root, exist_ok=True)
        
        # Save JSON data with enhanced metrics to quality-specific folder
        save_path = os.path.join(quality_output_root, f"{vid_id}.json")
        
        with open(save_path, 'w') as f:
            json.dump({
                "info": {
                    "fps": fps,
                    "frame_count": len(data_img_space),
                    "resolution": f"{w}x{h}",
                    "model": "PoseLandmarker_Heavy",
                    "processed_on": "CPU",
                    "timestamp": datetime.now().isoformat(),
                    "quality_rating": analysis['quality_rating'],
                    "smoothing_method": "one_euro_filter" if mode == "unfiltered" else "savgol_plus_stability",
                    "one_euro_params": ONE_EURO_SETTINGS if mode == "unfiltered" else None,
                },
                "visibility_metrics": {
                    "overall_avg": float(avg_overall),
                    "key_joints_avg": float(avg_key),
                    "overall_min": float(np.min(visibility_scores)),
                    "overall_max": float(np.max(visibility_scores)),
                    "key_joints_min": float(np.min(key_joint_scores)),
                    "key_joints_max": float(np.max(key_joint_scores))
                },
                "mandatory_chain_gate": {
                    **mandatory_chain_summary,
                    "visibility_threshold": MANDATORY_VISIBILITY_THRESHOLD,
                    "core_chains": MANDATORY_CORE_CHAINS,
                    "reference_full_chains": REFERENCE_FULL_CHAINS,
                },
                "landmark_stability": stability_summary,
                "problem_frames": analysis['problem_frames'],
                "continuous_problems": analysis['continuous_problems'],
                "face_detected": face_detected_per_frame,
                "keypoints_img": data_img_space,
                "keypoints_world": data_world_space
            }, f, indent=2)
        
        # Create visualization in quality-specific folder
        if CREATE_VISUALIZATION and frames_for_viz:
            try:
                viz_quality_root = os.path.join(VISUALIZATION_OUTPUT_ROOT, quality_rating.lower())
                os.makedirs(viz_quality_root, exist_ok=True)
                viz_output_path = os.path.join(viz_quality_root, f"{vid_id}_annotated.mp4")
                
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(viz_output_path, fourcc, fps, (w, h))
                
                if not out.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(viz_output_path, fourcc, fps, (w, h))
                
                if out.isOpened():
                    visibility_data = {
                        'frame_metrics': frame_metrics,
                        'overall_avg': avg_overall,
                        'key_joints_avg': avg_key
                    }
                    
                    for frame_idx, frame in enumerate(frames_for_viz):
                        if frame_idx < len(data_img_space):
                            landmarks = data_img_space[frame_idx]
                            annotated_frame = draw_landmarks_enhanced(
                                frame, landmarks, h, w, frame_idx, visibility_data
                            )
                            out.write(annotated_frame)
                    
                    out.release()
                else:
                    return vid_id, "Error", "Could not initialize VideoWriter", analysis, None
                    
            except Exception as viz_error:
                print(f"  Warning: Visualization failed for {vid_id}: {viz_error}")
        
        # Create analysis report
        if CREATE_ANALYSIS_REPORT and visibility_scores:
            plot_path, report_path = create_visualization_report(
                vid_id, analysis, visibility_scores, key_joint_scores
            )
            analysis['plot_path'] = plot_path
            analysis['report_path'] = report_path
        
        warning = None
        warning_parts = []
        if analysis['quality_rating'] in ['Fair', 'Poor']:
            warning_parts.append(
                f"Quality: {analysis['quality_rating']}. Avg visibility: {avg_overall:.2f}, Key joints: {avg_key:.2f}"
            )
        if mandatory_chain_warning:
            warning_parts.append(mandatory_chain_warning)
        if warning_parts:
            warning = " | ".join(warning_parts)
        
        return vid_id, "Success", warning, analysis, avg_overall
        
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        return vid_path, "Error", error_details, None, None

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n⚠️  Received interrupt signal. Shutting down gracefully...")
    sys.exit(0)

def run_extraction(mode="filtered"):
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file '{MODEL_PATH}' not found.")
        return
    
    for directory in [OUTPUT_ROOT, VISUALIZATION_OUTPUT_ROOT, ANALYSIS_OUTPUT_ROOT]:
        os.makedirs(directory, exist_ok=True)
    
    # Handle wildcard - find ALL videos
    if VIDEO_IDS == ["*"]:
        video_paths = []
        video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.flv')
        
        for root, dirs, files in os.walk(DATASET_ROOT):
            for file in files:
                if file.lower().endswith(video_extensions):
                    video_paths.append(os.path.join(root, file))
        
        print(f"Found {len(video_paths)} videos to process")
    else:
        # Process specific video IDs
        video_paths = []
        for vid_id in VIDEO_IDS:
            path = find_video_path(vid_id)
            if path:
                video_paths.append(path)
                print(f"✓ Found: {vid_id}")
            else:
                print(f"⚠️  Video '{vid_id}' not found.")
    
    if not video_paths:
        print("No videos found to process.")
        return
    
    print(f"\nProcessing {len(video_paths)} videos in '{mode}' mode...")
    print(f"Using {cpu_count()} CPU cores...")
    print("Press Ctrl+C to stop gracefully\n")
    
    # Process videos
    try:
        if len(video_paths) == 1:
            results = [process_single_video(video_paths[0], mode=mode)]
        else:
            from functools import partial
            worker = partial(process_single_video, mode=mode)
            with Pool(min(cpu_count(), len(video_paths))) as pool:
                results = list(tqdm(
                    pool.imap(worker, video_paths),
                    total=len(video_paths),
                    desc="Processing videos"
                ))
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user.")
        sys.exit(1)
    
    # Process results
    successes = [r for r in results if r[1] == "Success"]
    errors = [r for r in results if r[1] == "Error"]
    skipped = [r for r in results if r[1] == "Skipped"]
    warnings = [r for r in successes if r[2] is not None]
    
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print('='*60)
    print(f"✓ Successfully processed: {len(successes)}")
    print(f"⊘ Skipped (Poor quality): {len(skipped)}")
    print(f"✗ Failed: {len(errors)}")
    
    if successes:
        print(f"\n{'='*60}")
        print("QUALITY ASSESSMENT")
        print('='*60)
        for vid, status, warning, analysis, avg_vis in successes:
            print(f"\n{vid}:")
            print(f"  - Quality Rating: {analysis['quality_rating'] if analysis else 'N/A'}")
            avg_vis_str = f"{avg_vis:.3f}" if avg_vis and avg_vis > 0 else 'N/A'
            print(f"  - Avg Visibility: {avg_vis_str}")
            print(f"  - Problem Frames: {len(analysis['problem_frames']) if analysis else 'N/A'}")
            if analysis and 'recommendations' in analysis:
                print(f"  - Recommendations: {len(analysis['recommendations'])}")
    
    if warnings:
        print(f"\n{'='*60}")
        print("WARNINGS")
        print('='*60)
        for vid, status, warning, analysis, avg_vis in warnings:
            print(f"  - {vid}: {warning}")
    
    if skipped:
        print(f"\n{'='*60}")
        print("SKIPPED VIDEOS")
        print('='*60)
        for vid, status, reason, analysis, avg_vis in skipped:
            print(f"  - {vid}: {reason}")
            if isinstance(reason, str) and "UNRELIABLE VIDEO" in reason:
                print("    ⚠️  Disclaimer: Video is unreliable because both mandatory chains were discarded.")
    
    if errors:
        print(f"\n{'='*60}")
        print("ERRORS")
        print('='*60)
        for vid, status, error, analysis, avg_vis in errors:
            vid_name = os.path.basename(vid) if isinstance(vid, str) else vid
            print(f"  - {vid_name}: {error[:100]}...")
    
    print(f"\n{'='*60}")
    print("OUTPUT FILES")
    print('='*60)
    print(f"1. Pose landmarks (by quality):")
    print(f"   - Excellent: {QUALITY_FOLDERS['Excellent']}/")
    print(f"   - Good: {QUALITY_FOLDERS['Good']}/")
    print(f"   - Fair: {QUALITY_FOLDERS['Fair']}/")
    print(f"   - Poor: (not saved)")
    print(f"2. Annotated videos (by quality): {VISUALIZATION_OUTPUT_ROOT}/{{excellent|good|fair}}/")
    print(f"3. Analysis reports: {ANALYSIS_OUTPUT_ROOT}/")
    print('='*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract features from squat videos.")
    parser.add_argument("mode", nargs="?", default="filtered", choices=["filtered", "unfiltered"], help="Processing mode")
    parser.add_argument("--no-viz", action="store_true", help="Disable video visualization")
    parser.add_argument("--no-report", action="store_true", help="Disable analysis reports (PNG plots + text files)")
    args = parser.parse_args()

    if args.no_viz:
        CREATE_VISUALIZATION = False
        print("ℹ️  Visualization disabled via CLI")

    if args.no_report:
        CREATE_ANALYSIS_REPORT = False
        print("ℹ️  Analysis reports disabled via CLI")

    run_extraction(mode=args.mode)