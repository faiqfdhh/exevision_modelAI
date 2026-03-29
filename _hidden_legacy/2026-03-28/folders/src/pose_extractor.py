"""
Pose Extraction Module - Extracts MediaPipe pose landmarks from videos with quality assessment
Faithful repackaging of 2.5_extract_selected_features.py functionality
"""

import os
import json
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, Tuple, Dict, List

from config import PipelineConfig, ExtractionResult

# Suppress MediaPipe CPU Warnings
os.environ['GLOG_minloglevel'] = '2'

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


class PoseExtractor:
    """Extracts pose landmarks from exercise videos using MediaPipe with quality assessment"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._validate_model()

    def _validate_model(self):
        """Check that the MediaPipe model exists"""
        if not os.path.exists(self.config.mp_model_path):
            raise FileNotFoundError(
                f"Model file '{self.config.mp_model_path}' not found. "
                "Download from: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker"
            )

    def _get_mediapipe_options(self) -> vision.PoseLandmarkerOptions:
        """Create MediaPipe PoseLandmarker options"""
        base_options = python.BaseOptions(model_asset_path=self.config.mp_model_path)
        return vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=self.config.num_poses,
            min_pose_detection_confidence=self.config.min_pose_detection_confidence,
            min_pose_presence_confidence=self.config.min_pose_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            output_segmentation_masks=False,
        )

    def _get_joint_type(self, index: int) -> Optional[str]:
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

    def _get_color(self, visibility: float, joint_type: Optional[str] = None) -> Tuple[int, int, int]:
        """Get color based on visibility and joint importance (BGR for OpenCV)"""
        if joint_type and joint_type in self.config.joint_thresholds:
            threshold = self.config.joint_thresholds[joint_type]
            if visibility >= threshold:
                return (0, 255, 0)  # Green - good
            elif visibility >= threshold * 0.7:
                return (0, 255, 255)  # Yellow - acceptable
            else:
                return (0, 0, 255)  # Red - poor
        else:
            if visibility >= 0.5:
                return (0, 255, 0)
            else:
                return (0, 0, 255)

    def _analyze_visibility_trends(self, visibility_scores: List[float], 
                                   key_joint_scores: List[float]) -> Dict:
        """Analyze visibility patterns across frames"""
        analysis = {
            'quality_rating': 'Unknown',
            'problem_frames': [],
            'continuous_problems': [],
            'recommendations': []
        }
        
        # Overall quality rating
        avg_overall = np.mean(visibility_scores)
        avg_key = np.mean(key_joint_scores)
        
        thresholds = self.config.overall_thresholds
        if avg_overall >= thresholds['excellent'] and avg_key >= 0.7:
            analysis['quality_rating'] = 'Excellent'
        elif avg_overall >= thresholds['good'] and avg_key >= 0.6:
            analysis['quality_rating'] = 'Good'
        elif avg_overall >= thresholds['fair'] and avg_key >= 0.4:
            analysis['quality_rating'] = 'Fair'
        else:
            analysis['quality_rating'] = 'Poor'
        
        # Identify problem frames
        for i, (overall, key) in enumerate(zip(visibility_scores, key_joint_scores)):
            is_problem = overall < 0.4 or key < 0.3
            if is_problem:
                analysis['problem_frames'].append(i)
        
        # Identify continuous problem segments (3+ consecutive problem frames)
        problem_segments = []
        current_segment = []
        for prob in analysis['problem_frames']:
            if not current_segment or prob == current_segment[-1] + 1:
                current_segment.append(prob)
            else:
                if len(current_segment) >= 3:
                    problem_segments.append(current_segment)
                current_segment = [prob]
        
        if len(current_segment) >= 3:
            problem_segments.append(current_segment)
        
        for segment in problem_segments:
            analysis['continuous_problems'].append({
                'start_frame': segment[0],
                'end_frame': segment[-1],
                'duration': len(segment)
            })
        
        # Generate recommendations
        if analysis['quality_rating'] in ['Fair', 'Poor']:
            analysis['recommendations'].append("Consider retaking video with better camera angle")
            if len(analysis['continuous_problems']) > 0:
                analysis['recommendations'].append("Significant tracking issues detected")
        
        if analysis['quality_rating'] in ['Excellent', 'Good']:
            analysis['recommendations'].append("Video quality is suitable for analysis")
        
        return analysis

    def _create_analysis_report(self, video_id: str, analysis: Dict, 
                               visibility_scores: List[float], 
                               key_joint_scores: List[float]) -> str:
        """Create a comprehensive text and visualization report"""
        if not self.config.create_analysis_report:
            return None
        
        os.makedirs(self.config.analysis_output_root, exist_ok=True)
        
        # Create visualization plot
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
            prob_vis = [visibility_scores[i] for i in analysis['problem_frames']]
            axes[1, 0].scatter(analysis['problem_frames'], prob_vis, c='red', alpha=0.6)
            axes[1, 0].plot(visibility_scores, color='blue', alpha=0.3)
            axes[1, 0].set_xlabel('Frame')
            axes[1, 0].set_ylabel('Visibility')
            axes[1, 0].set_title(f'Problem Frames ({len(analysis["problem_frames"])} total)')
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
        
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config.analysis_output_root, 
                                f"{video_id}_visibility_analysis.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create text report
        report_path = os.path.join(self.config.analysis_output_root, 
                                  f"{video_id}_analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"POSE EXTRACTION ANALYSIS REPORT\n")
            f.write(f"Video ID: {video_id}\n")
            f.write(f"="*60 + "\n\n")
            
            f.write(f"QUALITY RATING: {analysis['quality_rating']}\n\n")
            
            f.write(f"VISIBILITY METRICS:\n")
            f.write(f"  Overall Average: {np.mean(visibility_scores):.3f}\n")
            f.write(f"  Key Joints Average: {np.mean(key_joint_scores):.3f}\n")
            f.write(f"  Min Overall: {np.min(visibility_scores):.3f}\n")
            f.write(f"  Max Overall: {np.max(visibility_scores):.3f}\n")
            f.write(f"  Std Deviation: {np.std(visibility_scores):.3f}\n\n")
            
            if analysis['problem_frames']:
                f.write(f"PROBLEM FRAMES: {len(analysis['problem_frames'])} frames\n")
                if analysis['continuous_problems']:
                    f.write(f"\nContinuous problem segments:\n")
                    for seg in analysis['continuous_problems']:
                        f.write(f"  Frames {seg['start_frame']}-{seg['end_frame']} "
                               f"(duration: {seg['duration']} frames)\n")
            
            f.write(f"\nRECOMMENDATIONS:\n")
            for rec in analysis['recommendations']:
                f.write(f"  • {rec}\n")
        
        return report_path

    def extract(self, video_path: str, save_output: bool = True, 
               output_quality_folder: Optional[str] = None) -> ExtractionResult:
        """
        Extract pose landmarks from a video with quality assessment.

        Args:
            video_path: Path to the video file
            save_output: Whether to save JSON and visualizations
            output_quality_folder: Optional quality folder ('excellent', 'good', 'fair')

        Returns:
            ExtractionResult with keypoints, quality metrics, and analysis
        """
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        # Validate video exists
        if not os.path.exists(video_path):
            return ExtractionResult(
                success=False,
                video_id=video_id,
                fps=0,
                frame_count=0,
                error=f"Video file not found: {video_path}",
            )

        try:
            return self._process_video(video_path, video_id, save_output, output_quality_folder)
        except Exception as e:
            return ExtractionResult(
                success=False,
                video_id=video_id,
                fps=0,
                frame_count=0,
                error=str(e),
            )

    def _process_video(self, video_path: str, video_id: str, 
                      save_output: bool = True,
                      output_quality_folder: Optional[str] = None) -> ExtractionResult:
        """Internal video processing logic with quality assessment"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return ExtractionResult(
                success=False,
                video_id=video_id,
                fps=0,
                frame_count=0,
                error="Failed to open video",
            )

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Limit frames if configured
        if self.config.max_extraction_frames:
            frame_count = min(frame_count, self.config.max_extraction_frames)

        data_img_space = []
        data_world_space = []
        visibility_scores = []
        key_joint_scores = []
        frames_for_vis = []

        try:
            with vision.PoseLandmarker.create_from_options(
                self._get_mediapipe_options()
            ) as landmarker:
                for frame_idx in range(frame_count):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Store frame for visualization
                    if self.config.create_visualization:
                        frames_for_vis.append(frame.copy())

                    # Convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=frame_rgb
                    )

                    # Get timestamp for video mode
                    frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

                    # Detect pose
                    detection_result = landmarker.detect_for_video(
                        mp_image, frame_timestamp_ms
                    )

                    # Extract landmarks and calculate visibility
                    if detection_result.pose_landmarks:
                        frame_img = [
                            [lm.x, lm.y, lm.z, lm.visibility]
                            for lm in detection_result.pose_landmarks[0]
                        ]
                        frame_world = [
                            [lm.x, lm.y, lm.z, lm.visibility]
                            for lm in detection_result.pose_world_landmarks[0]
                        ]
                        
                        # Calculate frame visibility
                        all_vis = [lm[3] for lm in frame_img]
                        overall_vis = np.mean(all_vis)
                        
                        # Calculate key joint visibility
                        key_vis = [frame_img[idx][3] for idx in SQUAT_KEY_JOINTS.values()]
                        key_vis_score = np.mean(key_vis)
                        
                    else:
                        # No pose detected - use zero landmarks
                        frame_img = [[0.0, 0.0, 0.0, 0.0] for _ in range(33)]
                        frame_world = [[0.0, 0.0, 0.0, 0.0] for _ in range(33)]
                        overall_vis = 0.0
                        key_vis_score = 0.0

                    data_img_space.append(frame_img)
                    data_world_space.append(frame_world)
                    visibility_scores.append(overall_vis)
                    key_joint_scores.append(key_vis_score)

        finally:
            cap.release()

        # Analyze visibility trends
        analysis = self._analyze_visibility_trends(visibility_scores, key_joint_scores)
        quality_rating = analysis['quality_rating']
        
        # Create analysis report if enabled
        if self.config.create_analysis_report:
            self._create_analysis_report(video_id, analysis, 
                                        visibility_scores, key_joint_scores)

        # Save output if requested
        if save_output:
            # Determine output folder
            if output_quality_folder:
                output_dir = os.path.join(self.config.features_output_root, 
                                        output_quality_folder)
            else:
                output_dir = os.path.join(self.config.features_output_root, 
                                        quality_rating.lower())
            
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, f"{video_id}.json")

            with open(output_path, 'w') as f:
                json.dump({
                    "info": {
                        "fps": fps,
                        "frame_count": len(data_img_space),
                        "model": "PoseLandmarker_FULL",
                        "processed_on": "CPU",
                        "quality_rating": quality_rating,
                        "avg_visibility": float(np.mean(visibility_scores)),
                        "avg_key_joint_visibility": float(np.mean(key_joint_scores)),
                    },
                    "keypoints_img": data_img_space,
                    "keypoints_world": data_world_space,
                }, f, indent=2)

        result = ExtractionResult(
            success=True,
            video_id=video_id,
            fps=fps,
            frame_count=len(data_img_space),
            keypoints_img=data_img_space,
            keypoints_world=data_world_space,
            quality_rating=quality_rating,
            visibility_scores=visibility_scores,
            key_joint_scores=key_joint_scores,
            problem_frames=analysis['problem_frames'],
        )

        return result

    def extract_and_save(self, video_path: str, output_dir: Optional[str] = None) -> ExtractionResult:
        """
        Extract pose landmarks and save to JSON file.

        Args:
            video_path: Path to video
            output_dir: Output directory (defaults to config.features_output_root)

        Returns:
            ExtractionResult
        """
        return self.extract(video_path, save_output=True, output_quality_folder=output_dir)
