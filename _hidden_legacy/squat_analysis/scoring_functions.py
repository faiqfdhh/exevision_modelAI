"""
scoring_functions.py
Squat quality scoring functions.

Converts error measurements into normalized scores (0-100 scale).
Uses percentile-based scoring similar to the diving AQA approach.

Compatible with: aqa_metaProgram_squat.py output
"""

import numpy as np
import pickle
import os
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from microprograms.squat_recognition_functions import (
    get_scale_factor, get_leg_length, detect_squat_depth_category,
    SquatPhase
)


@dataclass
class ScoreThresholds:
    """Thresholds for converting measurements to scores"""
    
    # Knee valgus (ratio: knee_spread / ankle_spread)
    # 1.0 = neutral, < 0.8 = significant valgus
    KNEE_VALGUS_EXCELLENT = 0.95
    KNEE_VALGUS_GOOD = 0.85
    KNEE_VALGUS_POOR = 0.70
    
    # Forward lean (degrees from vertical)
    FORWARD_LEAN_EXCELLENT = 15.0
    FORWARD_LEAN_GOOD = 25.0
    FORWARD_LEAN_POOR = 40.0
    
    # Hip shift (normalized by stance width)
    HIP_SHIFT_EXCELLENT = 0.05
    HIP_SHIFT_GOOD = 0.10
    HIP_SHIFT_POOR = 0.20
    
    # Knee asymmetry (degrees difference)
    KNEE_ASYMMETRY_EXCELLENT = 5.0
    KNEE_ASYMMETRY_GOOD = 10.0
    KNEE_ASYMMETRY_POOR = 20.0
    
    # Depth categories
    DEPTH_BELOW_PARALLEL = 100
    DEPTH_PARALLEL = 90
    DEPTH_ABOVE_PARALLEL = 70
    DEPTH_QUARTER = 40


THRESHOLDS = ScoreThresholds()


# =============================================================================
# PERCENTILE-BASED SCORING
# =============================================================================

def load_distribution_data(filepath: str = "./squat/distribution_data.pkl") -> Optional[Dict]:
    """
    Load pre-computed distribution data for percentile scoring.
    
    Similar to diving's distribution_data.pkl concept.
    """
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None


def compute_percentile_score(value: float, distribution: List[float], 
                            higher_is_better: bool = True) -> float:
    """
    Convert a measurement to percentile score (0-100).
    
    Args:
        value: Measured value
        distribution: Reference distribution of values
        higher_is_better: If True, higher values get higher scores
        
    Returns:
        Percentile score (0-100)
    """
    if not distribution:
        return 50.0
    
    sorted_dist = np.sort(distribution)
    percentile = np.searchsorted(sorted_dist, value) / len(sorted_dist) * 100
    
    if not higher_is_better:
        percentile = 100 - percentile
    
    return float(percentile)


# =============================================================================
# INDIVIDUAL METRIC SCORES
# =============================================================================

def score_knee_valgus(valgus_ratio: Optional[float], 
                      distribution: Optional[List[float]] = None) -> Tuple[float, str]:
    """
    Score knee valgus measurement.
    
    Args:
        valgus_ratio: Knee spread / ankle spread ratio
        distribution: Optional reference distribution
        
    Returns:
        Tuple of (score 0-100, feedback string)
    """
    if valgus_ratio is None:
        return 0.0, "Could not measure knee valgus"
    
    if distribution:
        score = compute_percentile_score(valgus_ratio, distribution, higher_is_better=True)
    else:
        # Rule-based scoring
        if valgus_ratio >= THRESHOLDS.KNEE_VALGUS_EXCELLENT:
            score = 95.0
        elif valgus_ratio >= THRESHOLDS.KNEE_VALGUS_GOOD:
            # Linear interpolation
            score = 75.0 + (valgus_ratio - THRESHOLDS.KNEE_VALGUS_GOOD) / \
                    (THRESHOLDS.KNEE_VALGUS_EXCELLENT - THRESHOLDS.KNEE_VALGUS_GOOD) * 20
        elif valgus_ratio >= THRESHOLDS.KNEE_VALGUS_POOR:
            score = 50.0 + (valgus_ratio - THRESHOLDS.KNEE_VALGUS_POOR) / \
                    (THRESHOLDS.KNEE_VALGUS_GOOD - THRESHOLDS.KNEE_VALGUS_POOR) * 25
        else:
            score = max(0, 50.0 * (valgus_ratio / THRESHOLDS.KNEE_VALGUS_POOR))
    
    # Generate feedback
    if valgus_ratio >= THRESHOLDS.KNEE_VALGUS_EXCELLENT:
        feedback = "Excellent knee tracking"
    elif valgus_ratio >= THRESHOLDS.KNEE_VALGUS_GOOD:
        feedback = "Good knee tracking, minor valgus"
    elif valgus_ratio >= THRESHOLDS.KNEE_VALGUS_POOR:
        feedback = "Moderate knee valgus - focus on pushing knees out"
    else:
        feedback = "Significant knee valgus - consider reducing load and practicing form"
    
    return float(score), feedback


def score_forward_lean(lean_angle: Optional[float],
                       distribution: Optional[List[float]] = None) -> Tuple[float, str]:
    """
    Score forward torso lean.
    
    Args:
        lean_angle: Forward lean in degrees from vertical
        distribution: Optional reference distribution
        
    Returns:
        Tuple of (score 0-100, feedback string)
    """
    if lean_angle is None:
        return 0.0, "Could not measure forward lean"
    
    abs_lean = abs(lean_angle)
    
    if distribution:
        score = compute_percentile_score(abs_lean, distribution, higher_is_better=False)
    else:
        if abs_lean <= THRESHOLDS.FORWARD_LEAN_EXCELLENT:
            score = 95.0
        elif abs_lean <= THRESHOLDS.FORWARD_LEAN_GOOD:
            score = 75.0 + (THRESHOLDS.FORWARD_LEAN_GOOD - abs_lean) / \
                    (THRESHOLDS.FORWARD_LEAN_GOOD - THRESHOLDS.FORWARD_LEAN_EXCELLENT) * 20
        elif abs_lean <= THRESHOLDS.FORWARD_LEAN_POOR:
            score = 50.0 + (THRESHOLDS.FORWARD_LEAN_POOR - abs_lean) / \
                    (THRESHOLDS.FORWARD_LEAN_POOR - THRESHOLDS.FORWARD_LEAN_GOOD) * 25
        else:
            score = max(0, 50.0 * (1 - (abs_lean - THRESHOLDS.FORWARD_LEAN_POOR) / 30))
    
    if abs_lean <= THRESHOLDS.FORWARD_LEAN_EXCELLENT:
        feedback = "Excellent torso position"
    elif abs_lean <= THRESHOLDS.FORWARD_LEAN_GOOD:
        feedback = "Good torso position with slight forward lean"
    elif abs_lean <= THRESHOLDS.FORWARD_LEAN_POOR:
        feedback = "Moderate forward lean - focus on chest up, core braced"
    else:
        feedback = "Excessive forward lean - work on mobility and core strength"
    
    return float(score), feedback


def score_hip_shift(shift_ratio: Optional[float],
                    distribution: Optional[List[float]] = None) -> Tuple[float, str]:
    """
    Score lateral hip shift.
    
    Args:
        shift_ratio: Hip shift normalized by stance width
        distribution: Optional reference distribution
        
    Returns:
        Tuple of (score 0-100, feedback string)
    """
    if shift_ratio is None:
        return 0.0, "Could not measure hip shift"
    
    abs_shift = abs(shift_ratio)
    
    if distribution:
        score = compute_percentile_score(abs_shift, distribution, higher_is_better=False)
    else:
        if abs_shift <= THRESHOLDS.HIP_SHIFT_EXCELLENT:
            score = 95.0
        elif abs_shift <= THRESHOLDS.HIP_SHIFT_GOOD:
            score = 75.0 + (THRESHOLDS.HIP_SHIFT_GOOD - abs_shift) / \
                    (THRESHOLDS.HIP_SHIFT_GOOD - THRESHOLDS.HIP_SHIFT_EXCELLENT) * 20
        elif abs_shift <= THRESHOLDS.HIP_SHIFT_POOR:
            score = 50.0 + (THRESHOLDS.HIP_SHIFT_POOR - abs_shift) / \
                    (THRESHOLDS.HIP_SHIFT_POOR - THRESHOLDS.HIP_SHIFT_GOOD) * 25
        else:
            score = max(0, 50.0 * (1 - (abs_shift - THRESHOLDS.HIP_SHIFT_POOR) / 0.2))
    
    if abs_shift <= THRESHOLDS.HIP_SHIFT_EXCELLENT:
        feedback = "Excellent weight distribution"
    elif abs_shift <= THRESHOLDS.HIP_SHIFT_GOOD:
        feedback = "Minor lateral shift - focus on even weight distribution"
    elif abs_shift <= THRESHOLDS.HIP_SHIFT_POOR:
        feedback = "Moderate hip shift - check for imbalances or mobility issues"
    else:
        feedback = "Significant hip shift - consider unilateral work"
    
    return float(score), feedback


def score_knee_asymmetry(asymmetry: Optional[float],
                        distribution: Optional[List[float]] = None) -> Tuple[float, str]:
    """
    Score knee angle asymmetry between legs.
    
    Args:
        asymmetry: Difference in knee angles (degrees)
        distribution: Optional reference distribution
        
    Returns:
        Tuple of (score 0-100, feedback string)
    """
    if asymmetry is None:
        return 0.0, "Could not measure knee asymmetry"
    
    if distribution:
        score = compute_percentile_score(asymmetry, distribution, higher_is_better=False)
    else:
        if asymmetry <= THRESHOLDS.KNEE_ASYMMETRY_EXCELLENT:
            score = 95.0
        elif asymmetry <= THRESHOLDS.KNEE_ASYMMETRY_GOOD:
            score = 75.0 + (THRESHOLDS.KNEE_ASYMMETRY_GOOD - asymmetry) / \
                    (THRESHOLDS.KNEE_ASYMMETRY_GOOD - THRESHOLDS.KNEE_ASYMMETRY_EXCELLENT) * 20
        elif asymmetry <= THRESHOLDS.KNEE_ASYMMETRY_POOR:
            score = 50.0 + (THRESHOLDS.KNEE_ASYMMETRY_POOR - asymmetry) / \
                    (THRESHOLDS.KNEE_ASYMMETRY_POOR - THRESHOLDS.KNEE_ASYMMETRY_GOOD) * 25
        else:
            score = max(0, 50.0 * (1 - (asymmetry - THRESHOLDS.KNEE_ASYMMETRY_POOR) / 20))
    
    if asymmetry <= THRESHOLDS.KNEE_ASYMMETRY_EXCELLENT:
        feedback = "Symmetrical knee movement"
    elif asymmetry <= THRESHOLDS.KNEE_ASYMMETRY_GOOD:
        feedback = "Minor knee asymmetry - acceptable"
    elif asymmetry <= THRESHOLDS.KNEE_ASYMMETRY_POOR:
        feedback = "Moderate asymmetry - check for strength imbalances"
    else:
        feedback = "Significant asymmetry - consider mobility assessment"
    
    return float(score), feedback


def score_squat_depth(min_knee_angle: Optional[float],
                      target_depth: str = "parallel") -> Tuple[float, str]:
    """
    Score squat depth achievement.
    
    Args:
        min_knee_angle: Minimum knee angle achieved
        target_depth: Target depth ('parallel', 'below_parallel', etc.)
        
    Returns:
        Tuple of (score 0-100, feedback string)
    """
    if min_knee_angle is None:
        return 0.0, "Could not measure squat depth"
    
    category = detect_squat_depth_category(min_knee_angle)
    
    depth_scores = {
        'below_parallel': THRESHOLDS.DEPTH_BELOW_PARALLEL,
        'parallel': THRESHOLDS.DEPTH_PARALLEL,
        'above_parallel': THRESHOLDS.DEPTH_ABOVE_PARALLEL,
        'quarter': THRESHOLDS.DEPTH_QUARTER,
    }
    
    score = depth_scores.get(category, 50)
    
    feedback_map = {
        'below_parallel': f"Excellent depth (below parallel) - knee angle: {min_knee_angle:.0f}°",
        'parallel': f"Good depth (at parallel) - knee angle: {min_knee_angle:.0f}°",
        'above_parallel': f"Partial squat (above parallel) - knee angle: {min_knee_angle:.0f}°",
        'quarter': f"Quarter squat - increase depth for full benefit - knee angle: {min_knee_angle:.0f}°",
    }
    
    return float(score), feedback_map.get(category, "Unknown depth")


# =============================================================================
# AGGREGATE SCORING
# =============================================================================

def compute_rep_score(rep_errors: Dict, weights: Optional[Dict] = None) -> Dict:
    """
    Compute overall score for a single repetition.
    
    Args:
        rep_errors: Dictionary of error measurements for the rep
        weights: Optional custom weights for each metric
        
    Returns:
        Dictionary with scores and feedback
    """
    if weights is None:
        weights = {
            'knee_valgus': 0.25,
            'forward_lean': 0.20,
            'hip_shift': 0.15,
            'knee_asymmetry': 0.15,
            'depth': 0.25,
        }
    
    scores = {}
    feedback = {}
    
    # Score each metric
    if 'knee_valgus' in rep_errors and rep_errors['knee_valgus'] is not None:
        valgus = rep_errors['knee_valgus'].get('mean') if isinstance(rep_errors['knee_valgus'], dict) else rep_errors['knee_valgus']
        scores['knee_valgus'], feedback['knee_valgus'] = score_knee_valgus(valgus)
    
    if 'forward_lean' in rep_errors and rep_errors['forward_lean'] is not None:
        lean = rep_errors['forward_lean'].get('mean') if isinstance(rep_errors['forward_lean'], dict) else rep_errors['forward_lean']
        scores['forward_lean'], feedback['forward_lean'] = score_forward_lean(lean)
    
    if 'hip_shift' in rep_errors and rep_errors['hip_shift'] is not None:
        shift = rep_errors['hip_shift'].get('mean') if isinstance(rep_errors['hip_shift'], dict) else rep_errors['hip_shift']
        scores['hip_shift'], feedback['hip_shift'] = score_hip_shift(shift)
    
    if 'knee_asymmetry' in rep_errors and rep_errors['knee_asymmetry'] is not None:
        asymm = rep_errors['knee_asymmetry'].get('mean') if isinstance(rep_errors['knee_asymmetry'], dict) else rep_errors['knee_asymmetry']
        scores['knee_asymmetry'], feedback['knee_asymmetry'] = score_knee_asymmetry(asymm)
    
    if 'min_knee_angle' in rep_errors:
        scores['depth'], feedback['depth'] = score_squat_depth(rep_errors['min_knee_angle'])
    
    # Compute weighted average
    total_weight = 0
    weighted_sum = 0
    
    for metric, weight in weights.items():
        if metric in scores:
            weighted_sum += scores[metric] * weight
            total_weight += weight
    
    overall_score = weighted_sum / total_weight if total_weight > 0 else 0
    
    return {
        'overall_score': round(overall_score, 1),
        'individual_scores': scores,
        'feedback': feedback,
        'grade': _score_to_grade(overall_score),
    }


def compute_set_score(squat_data: Dict) -> Dict:
    """
    Compute overall score for an entire set of squats.
    
    Args:
        squat_data: Full squat analysis data from aqa_metaProgram
        
    Returns:
        Dictionary with set-level scores and rep-by-rep breakdown
    """
    repetitions = squat_data.get('repetitions', [])
    
    if not repetitions:
        return {
            'overall_score': 0,
            'raw_score': 0,
            'grade': 'F',
            'rep_count': 0,
            'rep_scores': [],
            'feedback': "No repetitions detected",
        }
    
    rep_scores = []
    for rep in repetitions:
        if 'errors' in rep:
            rep_result = compute_rep_score(rep['errors'])
            rep_result['rep_id'] = rep.get('rep_id', len(rep_scores) + 1)
            rep_scores.append(rep_result)
    
    if not rep_scores:
        return {
            'overall_score': 0,
            'raw_score': 0,
            'rep_count': len(repetitions),
            'rep_scores': [],
            'grade': 'F',
            'feedback': "Could not score repetitions",
        }
    
    # Overall set score (average of rep scores)
    overall_scores = [r['overall_score'] for r in rep_scores]
    overall_score = np.mean(overall_scores)
    
    # Consistency bonus/penalty
    score_std = np.std(overall_scores)
    consistency_modifier = max(-10, min(5, 10 - score_std))
    
    # Fatigue analysis (compare first vs last reps)
    if len(rep_scores) >= 3:
        first_half = np.mean(overall_scores[:len(overall_scores)//2])
        second_half = np.mean(overall_scores[len(overall_scores)//2:])
        fatigue_effect = second_half - first_half
    else:
        fatigue_effect = 0
    
    return {
        'overall_score': round(overall_score + consistency_modifier, 1),
        'raw_score': round(overall_score, 1),
        'consistency_modifier': round(consistency_modifier, 1),
        'rep_count': len(repetitions),
        'rep_scores': rep_scores,
        'score_std': round(score_std, 1),
        'fatigue_effect': round(fatigue_effect, 1),
        'grade': _score_to_grade(overall_score),
    }


def _score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'


# =============================================================================
# DISTRIBUTION BUILDING (for percentile scoring)
# =============================================================================

def build_distribution_data(squat_data_list: List[Dict], 
                           output_path: str = "./squat/distribution_data.pkl"):
    """
    Build distribution data from a collection of analyzed squats.
    
    Args:
        squat_data_list: List of squat analysis outputs
        output_path: Where to save the distribution pickle
    """
    distributions = {
        'knee_valgus': [],
        'forward_lean': [],
        'hip_shift': [],
        'knee_asymmetry': [],
        'min_knee_angle': [],
    }
    
    for squat_data in squat_data_list:
        for rep in squat_data.get('repetitions', []):
            errors = rep.get('errors', {})
            
            for key in distributions:
                if key in errors and errors[key] is not None:
                    value = errors[key].get('mean') if isinstance(errors[key], dict) else errors[key]
                    if value is not None:
                        distributions[key].append(value)
    
    # Save distribution data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(distributions, f)
    
    print(f"Saved distribution data to {output_path}")
    for key, values in distributions.items():
        print(f"  {key}: {len(values)} samples")
    
    return distributions
