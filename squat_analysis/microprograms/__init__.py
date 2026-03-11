"""
Squat Analysis Microprograms

Low-level functions for error detection and squat recognition.
"""

from .squat_error_functions import (
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

from .squat_recognition_functions import (
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
