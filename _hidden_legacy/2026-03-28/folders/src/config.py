"""
Configuration and result dataclasses for the ExeVision Pipeline
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


@dataclass
class PipelineConfig:
    """Central configuration for the entire pipeline"""
    mp_model_path: str = "models/pose_landmarker_heavy.task"
    dataset_root: str = "./squat/dataset_videos_all"
    features_output_root: str = "./squat/extracted_features_clean"
    features_input_root: str = "./squat/extracted_features1/pose"
    visualization_output_root: str = "./squat/visualized_poses_clean"
    analysis_output_root: str = "./squat/analysis_reports"
    view_classifier_path: str = "./squat/view_classifier/view_classifier_v5_ensemble.pkl"
    view_encoder_path: str = "./squat/view_classifier/label_encoder_v5.pkl"
    view_scaler_path: str = "./squat/view_classifier/scaler_v5.pkl"
    segmentation_output_root: str = "./squat/segmented_reps"
    visualization_segmentation_root: str = "./squat/visualized_segmentation"
    pipeline_output_root: str = "./squat/pipeline_output"
    
    # Pose extraction quality thresholds
    create_visualization: bool = True
    create_analysis_report: bool = True
    joint_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'hip': 0.4,
        'knee': 0.6,
        'ankle': 0.5,
        'shoulder': 0.3,
        'heel': 0.4,
        'toe': 0.3
    })
    overall_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'excellent': 0.8,
        'good': 0.6,
        'fair': 0.4,
        'poor': 0.2
    })
    
    # Segmentation parameters (from script 5)
    analysis_window_size: int = 15
    min_phase_duration_frames: int = 12
    min_concentric_duration: int = 20
    hysteresis_frames: int = 8
    phase_lockout_frames: int = 6
    velocity_idle_threshold: float = 0.005
    velocity_moving_threshold: float = 0.010
    velocity_isometric_band: float = 0.1
    eccentric_velocity_min: float = 0.006
    concentric_velocity_min: float = 0.006
    isometric_min_duration: int = 10
    isometric_max_duration: int = 45
    min_rep_frames: int = 20
    min_squat_depth_ratio: float = 0.05
    min_squat_depth_angle: float = 10.0
    standing_knee_angle_threshold: float = 135.0
    min_landmark_confidence: float = 0.4
    min_key_joint_confidence: float = 0.5
    calibration_frames: int = 60
    min_valid_calibration_frames: int = 20
    
    # Legacy segmentation parameters (kept for compatibility)
    smoothing_window: int = 7
    smoothing_poly: int = 2
    idle_velocity_thresh: float = 0.002
    min_squat_depth: float = 0.03
    peak_prominence: float = 0.01
    
    # View classification (rule-based from script 4)
    valid_views: tuple = ("side", "front", "back", "front_side", "back_side")
    reliable_views: tuple = ("side", "front_side", "back_side")
    pure_side_width: float = 0.08
    rotation_threshold: float = 0.15
    nose_z_threshold: float = 0.05
    
    # Pose extraction
    max_extraction_frames: Optional[int] = None
    num_poses: int = 1
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class ExtractionResult:
    """Result from pose extraction stage"""
    success: bool
    video_id: str
    fps: float
    frame_count: int
    keypoints_img: Optional[List[List[float]]] = None
    keypoints_world: Optional[List[List[float]]] = None
    quality_rating: Optional[str] = None  # 'excellent', 'good', 'fair', 'poor'
    visibility_scores: Optional[List[float]] = None
    key_joint_scores: Optional[List[float]] = None
    problem_frames: Optional[List[int]] = None
    error: Optional[str] = None


@dataclass
class ViewResult:
    """Result from view classification stage"""
    success: bool
    predicted_view: str
    confidence: float
    all_probabilities: Dict[str, float] = field(default_factory=dict)
    is_valid_view: bool = True
    error: Optional[str] = None


@dataclass
class RepPhase:
    """Single phase within a repetition"""
    phase_type: str  # 'idle', 'eccentric', 'isometric', 'concentric', 'unknown'
    start_frame: int
    end_frame: int
    duration_frames: int
    transition_reason: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Repetition:
    """Single squat repetition with phase breakdown"""
    rep_id: int
    start_frame: int
    end_frame: int
    phases: List[RepPhase]
    squat_depth: float
    bottom_frame: int

    def to_dict(self):
        return {
            "rep_id": self.rep_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "squat_depth": round(self.squat_depth, 4),
            "bottom_frame": self.bottom_frame,
            "phases": [p.to_dict() for p in self.phases],
        }


@dataclass
class SegmentationResult:
    """Result from temporal segmentation stage"""
    success: bool
    total_reps: int
    repetitions: List[Dict[str, Any]] = field(default_factory=list)
    frame_phases: Optional[List[str]] = None
    signals: Optional[Dict[str, List[float]]] = None
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Complete pipeline result for a single video"""
    video_path: str
    video_id: str
    timestamp: str
    extraction: Optional[ExtractionResult] = None
    view: Optional[ViewResult] = None
    segmentation: Optional[SegmentationResult] = None
    form_errors: Optional[List[str]] = None
    form_score: Optional[float] = None
    feedback: Optional[List[str]] = None

    def to_dict(self) -> dict:
        """Convert result to serializable dictionary"""
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "timestamp": self.timestamp,
            "extraction": {
                "success": self.extraction.success,
                "fps": self.extraction.fps,
                "frame_count": self.extraction.frame_count,
                "error": self.extraction.error,
            } if self.extraction else None,
            "view_classification": {
                "success": self.view.success,
                "predicted_view": self.view.predicted_view,
                "confidence": round(self.view.confidence, 3),
                "is_valid": self.view.is_valid_view,
                "error": self.view.error,
            } if self.view else None,
            "segmentation": {
                "success": self.segmentation.success,
                "total_reps": self.segmentation.total_reps,
                "repetitions": self.segmentation.repetitions,
                "error": self.segmentation.error,
            } if self.segmentation else None,
            "form_analysis": {
                "form_score": self.form_score,
                "errors": self.form_errors,
                "feedback": self.feedback,
            } if (self.form_score or self.form_errors) else None,
        }
