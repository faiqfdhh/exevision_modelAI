"""
Squat Analysis - Rule-Based AQA Framework

Automated Quality Assessment for squat form analysis.
Compatible with the existing pipeline (2.5, 4, 5 scripts).
"""

from .microprograms.squat_error_functions import (
    get_all_errors_for_frame,
    aggregate_phase_errors,
    knee_valgus_error,
    knee_forward_travel_error,
    knee_angle_asymmetry,
    hip_shift_error,
    forward_lean_error,
    hip_hinge_angle,
    heel_rise_error,
    stance_width,
)

from .microprograms.squat_recognition_functions import (
    classify_view,
    get_view_label,
    get_scale_factor,
    get_leg_length,
    get_knee_angle,
    get_hip_height,
    compute_hip_displacement_signal,
    compute_knee_angle_signal,
    find_rep_peaks,
    detect_squat_depth_category,
    ViewType,
    SquatPhase,
)

from .scoring_functions import (
    compute_rep_score,
    compute_set_score,
    score_knee_valgus,
    score_forward_lean,
    score_hip_shift,
    score_squat_depth,
)

__all__ = [
    # Error functions
    'get_all_errors_for_frame',
    'aggregate_phase_errors',
    'knee_valgus_error',
    'knee_forward_travel_error',
    'knee_angle_asymmetry',
    'hip_shift_error',
    'forward_lean_error',
    'hip_hinge_angle',
    'heel_rise_error',
    'stance_width',
    
    # Recognition functions
    'classify_view',
    'get_view_label',
    'get_scale_factor',
    'get_leg_length',
    'get_knee_angle',
    'get_hip_height',
    'compute_hip_displacement_signal',
    'compute_knee_angle_signal',
    'find_rep_peaks',
    'detect_squat_depth_category',
    'ViewType',
    'SquatPhase',
    
    # Scoring functions
    'compute_rep_score',
    'compute_set_score',
    'score_knee_valgus',
    'score_forward_lean',
    'score_hip_shift',
    'score_squat_depth',
]
