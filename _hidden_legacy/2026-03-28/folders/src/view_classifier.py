"""
View Classification Module - Rule-based camera angle classification
Faithful repackaging of 4_classify_views.py functionality
"""

import numpy as np
from typing import Optional

from config import PipelineConfig, ExtractionResult, ViewResult


class ViewClassifier:
    """Classifies camera view using rule-based geometric analysis"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def classify(self, extraction: ExtractionResult) -> ViewResult:
        """
        Classify the camera view from extracted pose landmarks using rule-based logic.

        Args:
            extraction: ExtractionResult from pose extraction

        Returns:
            ViewResult with predicted view and confidence
        """
        if not extraction.success:
            return ViewResult(
                success=False,
                predicted_view="unknown",
                confidence=0.0,
                error="Invalid extraction result",
            )

        if extraction.keypoints_img is None:
            return ViewResult(
                success=False,
                predicted_view="unknown",
                confidence=0.0,
                error="Missing keypoints_img",
            )

        try:
            view = self._get_view_label(extraction.keypoints_img)
            
            # For rule-based, we don't have probabilities, so set high confidence
            # for valid views, lower for unknown
            if view == "unknown":
                confidence = 0.0
            else:
                confidence = 0.95
            
            is_valid = view in self.config.valid_views

            return ViewResult(
                success=True,
                predicted_view=view,
                confidence=confidence,
                all_probabilities={view: confidence},
                is_valid_view=is_valid,
            )

        except Exception as e:
            return ViewResult(
                success=False,
                predicted_view="unknown",
                confidence=0.0,
                error=str(e),
            )

    def _get_view_label(self, keypoints_img: list) -> str:
        """
        Classify camera view from keypoints using geometric rules.
        Returns: 'front', 'back', 'side', 'front_side', 'back_side', or 'unknown'
        
        Logic from 4_classify_views.py:
        1. Check shoulder width (X-axis) - very narrow = SIDE view
        2. Check nose Z relative to hips - determines FRONT vs BACK
        3. Check shoulder rotation (Z-axis diff) - large diff = DIAGONAL view
        """
        if not keypoints_img or len(keypoints_img) < 5:
            return "unknown"
        
        valid_frames = 0
        total_shoulder_width = 0
        total_shoulder_depth = 0
        total_nose_hip_diff = 0 

        # Check first 60 frames
        frames_to_check = min(len(keypoints_img), 60)

        for i in range(frames_to_check):
            frame = keypoints_img[i]
            
            if not frame or len(frame) < 25:
                continue
                
            if frame[0][0] == 0.0:  # Skip empty frames
                continue
            
            # 1. Width (X-axis) - shoulder separation
            l_shoulder_x = frame[11][0] if len(frame[11]) > 0 else 0
            r_shoulder_x = frame[12][0] if len(frame[12]) > 0 else 0
            width = abs(l_shoulder_x - r_shoulder_x)
            
            # 2. Rotation (Shoulder Z-axis diff) - depth difference
            l_shoulder_z = frame[11][2] if len(frame[11]) > 2 else 0
            r_shoulder_z = frame[12][2] if len(frame[12]) > 2 else 0
            depth = abs(l_shoulder_z - r_shoulder_z)
            
            # 3. Front/Back Indicator (Nose Z relative to Hips)
            nose_z = frame[0][2] if len(frame[0]) > 2 else 0
            l_hip_z = frame[23][2] if len(frame[23]) > 2 else 0
            r_hip_z = frame[24][2] if len(frame[24]) > 2 else 0
            avg_hip_z = (l_hip_z + r_hip_z) / 2.0
            
            nose_rel_z = nose_z - avg_hip_z
            
            total_shoulder_width += width
            total_shoulder_depth += depth
            total_nose_hip_diff += nose_rel_z
            valid_frames += 1

        if valid_frames < 5:
            return "unknown"

        avg_width = total_shoulder_width / valid_frames
        avg_rot = total_shoulder_depth / valid_frames
        avg_nose_diff = total_nose_hip_diff / valid_frames
        
        # --- Classification Logic ---
        
        # 1. Side Check - very narrow shoulder width
        if avg_width < self.config.pure_side_width:
            return "side"

        # 2. Front vs Back Check - nose Z relative to hips
        if avg_nose_diff > self.config.nose_z_threshold:
            facing = "back"
        else:
            facing = "front"

        # 3. Rotation Check - shoulder depth difference
        if avg_rot > self.config.rotation_threshold:
            return f"{facing}_side"
        else:
            return facing

    def classify_from_json(self, json_data: dict) -> ViewResult:
        """
        Classify view directly from JSON data (for updating existing JSON files).
        
        Args:
            json_data: Dictionary with 'keypoints_img' field
            
        Returns:
            ViewResult
        """
        try:
            keypoints_img = json_data.get('keypoints_img', [])
            view = self._get_view_label(keypoints_img)
            
            confidence = 0.95 if view != "unknown" else 0.0
            is_valid = view in self.config.valid_views
            
            return ViewResult(
                success=True,
                predicted_view=view,
                confidence=confidence,
                all_probabilities={view: confidence},
                is_valid_view=is_valid,
            )
        except Exception as e:
            return ViewResult(
                success=False,
                predicted_view="unknown",
                confidence=0.0,
                error=str(e),
            )
