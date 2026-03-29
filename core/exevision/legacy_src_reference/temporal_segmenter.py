"""
Temporal Segmentation Module - Biomechanically-sound squat phase detection
Faithful repackaging of 5_temporal_segmentation.py functionality
Uses window-based FSM with angle-invariant metrics
"""

import json
import numpy as np
from typing import List, Optional, Tuple, Dict
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import uniform_filter1d, median_filter
from dataclasses import dataclass
from enum import Enum, auto

from config import (
    PipelineConfig,
    ExtractionResult,
    SegmentationResult,
    RepPhase,
    Repetition,
)

# MediaPipe landmark indices
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_SHOULDER, R_SHOULDER = 11, 12
L_HEEL, R_HEEL = 29, 30
NOSE = 0


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
    
    def __init__(self, keypoints: List, view: str, config: PipelineConfig, fps: float = 30.0):
        self.keypoints = keypoints
        self.view = view
        self.config = config
        self.fps = fps
        self.frame_count = len(keypoints)
        
        # Calibrated anthropometrics (for normalization)
        self.torso_length = None
        self.femur_length = None
        self.tibia_length = None
        self.standing_hip_height = None
        self.body_scale = None
        
        # Primary control signals (view-invariant)
        self.normalized_hip_displacement = None
        self.knee_angles = None
        self.hip_heights_raw = None
        
        # Derived signals
        self.velocity_signal = None
        self.window_velocities = None
        self.landmark_confidence = None
        self.valid_frame_mask = None
        
        # View reliability
        reliable_views = getattr(config, 'reliable_views', ('side', 'front_side', 'back_side'))
        self.view_reliable = view.lower() in reliable_views if view else False
        self.use_knee_angle_primary = view.lower() in {'front', 'back'}
    
    def calibrate_from_idle(self) -> bool:
        """Extract anthropometric measurements from idle frames"""
        valid_frames = []
        
        for i, frame in enumerate(self.keypoints[:self.config.calibration_frames]):
            if frame is None or len(frame) < 33:
                continue
            
            # Check key joint confidence
            key_joints = [L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_SHOULDER, R_SHOULDER]
            confidences = [frame[j][3] for j in key_joints if j < len(frame)]
            
            if len(confidences) >= 6 and np.mean(confidences) >= self.config.min_key_joint_confidence:
                valid_frames.append(frame)
        
        if len(valid_frames) < self.config.min_valid_calibration_frames:
            valid_frames = [f for f in self.keypoints[:self.config.calibration_frames] 
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
        
        # Use MIN of hip heights as standing position (smallest Y = highest position)
        hip_heights_sorted = np.sort(hip_heights)
        standing_idx = int(len(hip_heights_sorted) * 0.25)
        self.standing_hip_height = hip_heights_sorted[standing_idx]
        
        # Body scale = average of torso and leg lengths
        self.body_scale = (self.torso_length + self.femur_length + self.tibia_length) / 3
        
        return True
    
    def compute_normalized_hip_displacement(self) -> np.ndarray:
        """Compute the primary control signal: normalized vertical hip displacement"""
        displacements = []
        confidences = []
        
        for frame in self.keypoints:
            if frame is None or len(frame) < 24:
                displacements.append(np.nan)
                confidences.append(0.0)
                continue
            
            l_hip = np.array(frame[L_HIP][:3])
            r_hip = np.array(frame[R_HIP][:3])
            hip_mid = (l_hip + r_hip) / 2
            
            hip_conf = (frame[L_HIP][3] + frame[R_HIP][3]) / 2
            confidences.append(hip_conf)
            
            if hip_conf < self.config.min_landmark_confidence:
                displacements.append(np.nan)
                continue
            
            # Displacement from standing (positive = lower = squatting)
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
            
            l_angle = self._calculate_angle(l_hip, l_knee, l_ankle) if l_conf >= self.config.min_landmark_confidence else np.nan
            r_angle = self._calculate_angle(r_hip, r_knee, r_ankle) if r_conf >= self.config.min_landmark_confidence else np.nan
            
            # Use average of valid angles
            valid_angles = [a for a in [l_angle, r_angle] if not np.isnan(a)]
            avg_angle = np.mean(valid_angles) if valid_angles else np.nan
            
            angles.append(avg_angle)
        
        self.knee_angles = np.array(angles)
        self._interpolate_array(self.knee_angles)
        return self.knee_angles
    
    def compute_velocity_signal(self) -> np.ndarray:
        """Compute smoothed velocity of hip displacement"""
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
        """Compute velocity trends averaged over windows"""
        if self.velocity_signal is None:
            self.compute_velocity_signal()
        
        window_size = self.config.analysis_window_size
        self.window_velocities = uniform_filter1d(self.velocity_signal, size=window_size, mode='nearest')
        
        return self.window_velocities
    
    def compute_valid_frame_mask(self) -> np.ndarray:
        """Identify frames with sufficient landmark confidence"""
        if self.landmark_confidence is None:
            return np.ones(self.frame_count, dtype=bool)
        
        self.valid_frame_mask = self.landmark_confidence >= self.config.min_landmark_confidence
        return self.valid_frame_mask
    
    def get_window_metrics(self, center_frame: int) -> WindowMetrics:
        """Compute metrics for a temporal window centered at given frame"""
        half_window = self.config.analysis_window_size // 2
        start = max(0, center_frame - half_window)
        end = min(self.frame_count, center_frame + half_window + 1)
        
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
    """Window-based Finite State Machine for squat phase detection"""
    
    def __init__(self, analyzer: BiomechanicalAnalyzer, config: PipelineConfig, fps: float = 30.0):
        self.analyzer = analyzer
        self.config = config
        self.fps = fps
        self.frame_count = analyzer.frame_count
        
        # State tracking
        self.current_state = SquatPhase.UNKNOWN
        self.phase_labels = np.full(self.frame_count, SquatPhase.UNKNOWN.value, dtype=int)
        self.state_start_frame = 0
        self.frames_in_state = 0
        
        # Hysteresis tracking
        self.transition_vote_count = 0
        self.pending_transition = None
        self.frames_since_last_transition = 0
        
        # History for debugging
        self.transition_history = []
    
    def detect_phases(self) -> np.ndarray:
        """Main FSM loop - classifies each frame"""
        # Initialize to IDLE (assuming video starts with person standing)
        self.current_state = SquatPhase.IDLE
        self.phase_labels[0] = SquatPhase.IDLE.value
        
        for frame in range(1, self.frame_count):
            metrics = self.analyzer.get_window_metrics(frame)
            
            # Evaluate what phase this frame suggests
            suggested_phase = self._evaluate_phase(frame, metrics)
            
            # Handle transition (with hysteresis and validation)
            self._handle_transition(frame, suggested_phase, metrics)
            
            # Record current state
            self.phase_labels[frame] = self.current_state.value
            self.frames_in_state += 1
            self.frames_since_last_transition += 1
        
        # Post-processing: enforce minimum durations
        self._enforce_minimum_durations()
        
        return self.phase_labels
    
    def _evaluate_phase(self, frame: int, metrics: WindowMetrics) -> SquatPhase:
        """Determine what phase the current window suggests"""
        vel = metrics.velocity_trend
        depth = metrics.depth_ratio
        knee_angle = self.analyzer.knee_angles[frame]
        
        # IDLE: Near standing position with minimal movement
        if knee_angle > self.config.standing_knee_angle_threshold and abs(vel) < self.config.velocity_idle_threshold:
            return SquatPhase.IDLE
        
        # ECCENTRIC: Moving downward
        if vel > self.config.eccentric_velocity_min:
            return SquatPhase.ECCENTRIC
        
        # CONCENTRIC: Moving upward
        if vel < -self.config.concentric_velocity_min:
            return SquatPhase.CONCENTRIC
        
        # ISOMETRIC: Low velocity at significant depth
        if abs(vel) < self.config.velocity_isometric_band and depth > 0.6:
            return SquatPhase.ISOMETRIC
        
        # Default: maintain current state
        return self.current_state
    
    def _handle_transition(self, frame: int, suggested: SquatPhase, metrics: WindowMetrics):
        """Handle state transitions with hysteresis and validation"""
        # Check if transition is allowed
        if suggested not in VALID_TRANSITIONS[self.current_state]:
            return
        
        # No transition needed
        if suggested == self.current_state:
            self.transition_vote_count = 0
            self.pending_transition = None
            return
        
        # Check lockout period
        if self.frames_since_last_transition < self.config.phase_lockout_frames:
            return
        
        # Check minimum duration in current phase
        if self.frames_in_state < self.config.min_phase_duration_frames:
            return
        
        # Hysteresis: need sustained evidence for transition
        if self.pending_transition == suggested:
            self.transition_vote_count += 1
        else:
            self.pending_transition = suggested
            self.transition_vote_count = 1
        
        # Execute transition if threshold met
        if self.transition_vote_count >= self.config.hysteresis_frames:
            reason = self._get_transition_reason(self.current_state, suggested, metrics)
            self.transition_history.append({
                'frame': frame,
                'from': self.current_state.name,
                'to': suggested.name,
                'reason': reason
            })
            
            self.current_state = suggested
            self.state_start_frame = frame
            self.frames_in_state = 0
            self.transition_vote_count = 0
            self.pending_transition = None
            self.frames_since_last_transition = 0
    
    def _get_transition_reason(self, from_state: SquatPhase, to_state: SquatPhase, 
                               metrics: WindowMetrics) -> str:
        """Generate human-readable transition reason"""
        if to_state == SquatPhase.ECCENTRIC:
            return f"Downward velocity: {metrics.velocity_trend:.4f}"
        elif to_state == SquatPhase.CONCENTRIC:
            return f"Upward velocity: {metrics.velocity_trend:.4f}"
        elif to_state == SquatPhase.ISOMETRIC:
            return f"Low velocity at depth: vel={metrics.velocity_trend:.4f}, depth={metrics.depth_ratio:.2f}"
        elif to_state == SquatPhase.IDLE:
            return f"Returned to standing"
        return "Unknown"
    
    def _enforce_minimum_durations(self):
        """Post-process to remove very short phases"""
        # Apply median filter to remove isolated misclassifications
        filtered = median_filter(self.phase_labels, size=5)
        self.phase_labels = filtered.astype(int)
    
    def get_phase_name(self, phase_id: int) -> str:
        """Convert phase ID to string name"""
        return SquatPhase(phase_id).name.lower()


class TemporalSegmenter:
    """Segments exercise motion into phases and repetitions using biomechanical FSM"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def segment(self, extraction: ExtractionResult, view_label: Optional[str] = None) -> SegmentationResult:
        """
        Main entry point: run full segmentation pipeline.

        Args:
            extraction: ExtractionResult from pose extraction
            view_label: Optional view classification ('side', 'front', etc.)

        Returns:
            SegmentationResult with phases and repetitions
        """
        if not extraction.success or extraction.keypoints_world is None:
            return SegmentationResult(
                success=False,
                total_reps=0,
                error="Invalid extraction result or missing world keypoints",
            )

        try:
            keypoints = extraction.keypoints_world
            fps = extraction.fps
            
            # Initialize biomechanical analyzer
            analyzer = BiomechanicalAnalyzer(keypoints, view_label or "unknown", self.config, fps)
            
            # Calibrate from idle frames
            calibration_success = analyzer.calibrate_from_idle()
            if not calibration_success:
                return SegmentationResult(
                    success=False,
                    total_reps=0,
                    error="Failed to calibrate from idle frames",
                )
            
            # Compute control signals
            analyzer.compute_normalized_hip_displacement()
            analyzer.compute_knee_angles()
            analyzer.compute_velocity_signal()
            analyzer.compute_window_velocities()
            analyzer.compute_valid_frame_mask()
            
            # Run FSM for phase detection
            fsm = SquatStateMachine(analyzer, self.config, fps)
            phase_labels = fsm.detect_phases()
            
            # Extract repetitions from phase labels
            reps = self._extract_repetitions(phase_labels, analyzer, fsm)
            
            # Build output
            phase_names = [fsm.get_phase_name(p) for p in phase_labels]
            
            return SegmentationResult(
                success=True,
                total_reps=len(reps),
                repetitions=[r.to_dict() for r in reps],
                frame_phases=phase_names,
                signals={
                    "hip_displacement": analyzer.normalized_hip_displacement.tolist(),
                    "velocity": analyzer.velocity_signal.tolist(),
                    "knee_angles": analyzer.knee_angles.tolist(),
                },
            )

        except Exception as e:
            return SegmentationResult(
                success=False,
                total_reps=0,
                error=str(e),
            )

    def _extract_repetitions(self, phase_labels: np.ndarray, 
                           analyzer: BiomechanicalAnalyzer,
                           fsm: SquatStateMachine) -> List[Repetition]:
        """Extract individual repetitions from phase labels"""
        reps = []
        
        # Find sequences: IDLE -> ECCENTRIC -> (ISOMETRIC) -> CONCENTRIC -> IDLE
        in_rep = False
        rep_start = None
        eccentric_start = None
        concentric_start = None
        isometric_start = None
        bottom_frame = None
        
        for i in range(len(phase_labels)):
            phase = SquatPhase(phase_labels[i])
            
            if phase == SquatPhase.ECCENTRIC and not in_rep:
                # Start of new rep
                in_rep = True
                rep_start = i
                eccentric_start = i
                
            elif phase == SquatPhase.ISOMETRIC and in_rep and eccentric_start is not None:
                # Transition to isometric
                if isometric_start is None:
                    isometric_start = i
                    bottom_frame = i
                    
            elif phase == SquatPhase.CONCENTRIC and in_rep:
                # Transition to concentric
                if concentric_start is None:
                    concentric_start = i
                    if bottom_frame is None:
                        bottom_frame = i
                        
            elif phase == SquatPhase.IDLE and in_rep and concentric_start is not None:
                # End of rep
                rep_end = i
                
                # Validate rep
                if rep_end - rep_start >= self.config.min_rep_frames:
                    rep = self._create_repetition(
                        len(reps) + 1,
                        rep_start,
                        rep_end,
                        eccentric_start,
                        concentric_start,
                        isometric_start,
                        bottom_frame,
                        analyzer
                    )
                    if rep:
                        reps.append(rep)
                
                # Reset for next rep
                in_rep = False
                rep_start = None
                eccentric_start = None
                concentric_start = None
                isometric_start = None
                bottom_frame = None
        
        return reps

    def _create_repetition(self, rep_id: int, start: int, end: int,
                          ecc_start: int, con_start: int, iso_start: Optional[int],
                          bottom: int, analyzer: BiomechanicalAnalyzer) -> Optional[Repetition]:
        """Create a Repetition object with phase breakdown"""
        phases = []
        
        # Eccentric phase
        if con_start > ecc_start:
            phases.append(RepPhase(
                phase_type="eccentric",
                start_frame=ecc_start,
                end_frame=con_start,
                duration_frames=con_start - ecc_start,
            ))
        
        # Optional isometric phase
        if iso_start is not None and con_start > iso_start:
            phases.append(RepPhase(
                phase_type="isometric",
                start_frame=iso_start,
                end_frame=con_start,
                duration_frames=con_start - iso_start,
            ))
        
        # Concentric phase
        if end > con_start:
            phases.append(RepPhase(
                phase_type="concentric",
                start_frame=con_start,
                end_frame=end,
                duration_frames=end - con_start,
            ))
        
        # Calculate squat depth metrics
        baseline_depth = analyzer.normalized_hip_displacement[start]
        bottom_depth = analyzer.normalized_hip_displacement[bottom]
        depth_normalized = bottom_depth - baseline_depth
        
        # Knee angle metrics
        start_angle = analyzer.knee_angles[start]
        bottom_angle = analyzer.knee_angles[bottom]
        depth_angle = start_angle - bottom_angle
        
        # Validate minimum depth
        if depth_normalized < self.config.min_squat_depth_ratio:
            return None
        if depth_angle < self.config.min_squat_depth_angle:
            return None
        
        return Repetition(
            rep_id=rep_id,
            start_frame=start,
            end_frame=end,
            phases=phases,
            squat_depth=depth_normalized,
            bottom_frame=bottom,
        )
