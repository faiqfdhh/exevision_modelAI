"""
aqa_metaProgram_squat.py
Automated Quality Assessment MetaProgram for Squat Analysis

Main orchestration script that:
1. Loads extracted features from 2.5_extract_selected_features.py output
2. Classifies view using 4_classify_views.py logic
3. Segments temporal phases using 5_temporal_segmentation.py logic
4. Calculates form errors for each phase/rep
5. Generates quality scores and feedback

Usage:
    python squat_analysis/aqa_metaProgram_squat.py <video_id>
    python squat_analysis/aqa_metaProgram_squat.py --all
    python squat_analysis/aqa_metaProgram_squat.py --json path/to/features.json
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from squat_analysis.microprograms.squat_error_functions import (
    get_all_errors_for_frame,
    aggregate_phase_errors,
    knee_valgus_error,
    forward_lean_error,
    hip_shift_error,
    knee_angle_asymmetry,
)
from squat_analysis.microprograms.squat_recognition_functions import (
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
from squat_analysis.scoring_functions import (
    compute_rep_score,
    compute_set_score,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Input directories (from your existing scripts)
FEATURES_EXCELLENT = "./squat/extracted_features_clean/excellent"
FEATURES_GOOD = "./squat/extracted_features_clean/good"
FEATURES_FAIR = "./squat/extracted_features_clean/fair"

FEATURES_DIRS = [FEATURES_EXCELLENT, FEATURES_GOOD, FEATURES_FAIR]

# Segmented data (from 5_temporal_segmentation.py)
SEGMENTED_EXCELLENT = "./squat/segmented_reps/excellent"
SEGMENTED_GOOD = "./squat/segmented_reps/good"
SEGMENTED_FAIR = "./squat/segmented_reps/fair"

SEGMENTED_DIRS = [SEGMENTED_EXCELLENT, SEGMENTED_GOOD, SEGMENTED_FAIR]

# Output directory
OUTPUT_DIR = "./squat/aqa_analysis"

# Phase analysis parameters
MIN_LANDMARK_CONFIDENCE = 0.5


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def find_feature_json(video_id: str) -> Optional[str]:
    """Find extracted features JSON for a video ID."""
    for features_dir in FEATURES_DIRS:
        json_path = os.path.join(features_dir, f"{video_id}.json")
        if os.path.exists(json_path):
            return json_path
    return None


def find_segmented_json(video_id: str) -> Optional[str]:
    """Find segmented data JSON for a video ID."""
    for seg_dir in SEGMENTED_DIRS:
        json_path = os.path.join(seg_dir, f"{video_id}_segmented.json")
        if os.path.exists(json_path):
            return json_path
    return None


def load_feature_data(json_path: str) -> Dict:
    """Load extracted features from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_segmented_data(json_path: str) -> Dict:
    """Load segmented temporal data from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_quality_level(json_path: str) -> str:
    """Determine quality level from file path."""
    if 'excellent' in json_path.lower():
        return 'excellent'
    elif 'good' in json_path.lower():
        return 'good'
    elif 'fair' in json_path.lower():
        return 'fair'
    return 'unknown'


# =============================================================================
# TEMPORAL SEGMENTATION (simplified version compatible with 5_ output)
# =============================================================================

def segment_into_phases(keypoints: List, fps: float = 30.0) -> Dict:
    """
    Segment squat into phases using simplified algorithm.
    
    This is a lightweight version that can run standalone.
    For production, prefer using output from 5_temporal_segmentation.py.
    """
    hip_displacement = compute_hip_displacement_signal(keypoints)
    knee_angles = compute_knee_angle_signal(keypoints)
    
    # Compute velocity
    velocity = np.gradient(hip_displacement)
    
    # Smooth velocity
    from scipy.ndimage import uniform_filter1d
    window_size = 11
    smoothed_velocity = uniform_filter1d(velocity, size=window_size, mode='nearest')
    
    # Find rep peaks (squat bottoms)
    peaks = find_rep_peaks(hip_displacement, min_prominence=0.03, min_distance=15)
    
    # Simple phase labeling
    frame_count = len(keypoints)
    phase_labels = ['idle'] * frame_count
    
    velocity_threshold = 0.003
    
    for i, frame in enumerate(keypoints):
        vel = smoothed_velocity[i]
        
        if abs(vel) < velocity_threshold:
            # Check if at bottom
            if i in peaks or (len(peaks) > 0 and min(abs(i - p) for p in peaks) < 5):
                phase_labels[i] = 'isometric'
            else:
                phase_labels[i] = 'idle'
        elif vel > 0:
            phase_labels[i] = 'eccentric'
        else:
            phase_labels[i] = 'concentric'
    
    return {
        'phase_labels': phase_labels,
        'hip_displacement': hip_displacement.tolist(),
        'knee_angles': knee_angles.tolist(),
        'velocity': smoothed_velocity.tolist(),
        'rep_peaks': peaks,
    }


def extract_repetitions_from_phases(phase_labels: List[str], 
                                    hip_displacement: List[float],
                                    knee_angles: List[float],
                                    fps: float = 30.0) -> List[Dict]:
    """
    Extract individual repetitions from phase labels using state machine.
    
    Rep pattern: ECCENTRIC [-> ISOMETRIC] -> CONCENTRIC
    
    Simple state machine:
    1. IDLE: waiting for eccentric start
    2. ECCENTRIC: in downward phase
    3. ISOMETRIC (optional): holding at bottom
    4. CONCENTRIC: in upward phase
    5. Complete when we return to IDLE or start new ECCENTRIC
    """
    repetitions = []
    frame_count = len(phase_labels)
    
    state = 'idle'  # Current state
    rep_start = None
    ecc_start = None
    iso_start = None
    con_start = None
    bottom_frame = None
    max_depth_so_far = 0.0
    
    for i in range(frame_count):
        phase = phase_labels[i]
        
        if state == 'idle':
            if phase == 'eccentric':
                # Start of new rep
                state = 'eccentric'
                rep_start = i
                ecc_start = i
                iso_start = None
                con_start = None
                bottom_frame = None
                max_depth_so_far = 0.0
        
        elif state == 'eccentric':
            if phase == 'isometric':
                # Transition to hold at bottom
                state = 'isometric'
                iso_start = i
                bottom_frame = i
            elif phase == 'concentric':
                # Skip isometric, go straight to concentric
                state = 'concentric'
                con_start = i
                # Find bottom (deepest point reached)
                if ecc_start < i:
                    ecc_range = hip_displacement[ecc_start:i]
                    if ecc_range:
                        bottom_frame = ecc_start + int(np.argmax(ecc_range))
            elif phase == 'eccentric':
                # Continue in eccentric
                pass
            else:
                # Unexpected phase, reset
                state = 'idle'
        
        elif state == 'isometric':
            if phase == 'concentric':
                # Transition from hold to concentric
                state = 'concentric'
                con_start = i
            elif phase == 'isometric':
                # Continue holding at bottom
                if bottom_frame is None or hip_displacement[i] > hip_displacement[bottom_frame]:
                    bottom_frame = i
            else:
                # Unexpected phase, reset
                state = 'idle'
        
        elif state == 'concentric':
            if phase == 'idle' or phase == 'eccentric':
                # Rep complete!
                rep_end = i
                
                if ecc_start is not None and con_start is not None:
                    # Calculate depth metrics
                    if bottom_frame is not None and bottom_frame < len(knee_angles):
                        min_knee_angle = knee_angles[bottom_frame]
                        max_hip_disp = hip_displacement[bottom_frame]
                    else:
                        # Fallback: find min knee angle in range
                        search_range = knee_angles[ecc_start:rep_end]
                        if search_range:
                            valid_angles = [a for a in search_range if not np.isnan(a)]
                            min_knee_angle = min(valid_angles) if valid_angles else 180
                            bottom_frame = ecc_start + search_range.index(min_knee_angle)
                        else:
                            min_knee_angle = 180
                        
                        search_range = hip_displacement[ecc_start:rep_end]
                        max_hip_disp = max(search_range) if search_range else 0
                    
                    rep = {
                        'rep_id': len(repetitions) + 1,
                        'start_frame': rep_start,
                        'end_frame': rep_end,
                        'duration_frames': rep_end - rep_start,
                        'duration_seconds': (rep_end - rep_start) / fps,
                        'phases': {
                            'eccentric': {'start': ecc_start, 'end': iso_start if iso_start else con_start},
                            'isometric': {'start': iso_start, 'end': con_start} if iso_start else None,
                            'concentric': {'start': con_start, 'end': rep_end},
                        },
                        'bottom_frame': bottom_frame,
                        'min_knee_angle': float(min_knee_angle) if not np.isnan(min_knee_angle) else None,
                        'max_hip_displacement': float(max_hip_disp),
                        'depth_category': detect_squat_depth_category(min_knee_angle) if not np.isnan(min_knee_angle) else 'unknown',
                    }
                    repetitions.append(rep)
                
                # Reset for next rep
                if phase == 'eccentric':
                    # Start next rep immediately
                    state = 'eccentric'
                    rep_start = i
                    ecc_start = i
                    iso_start = None
                    con_start = None
                    bottom_frame = None
                else:
                    # Go back to idle
                    state = 'idle'
            
            elif phase == 'concentric':
                # Continue in concentric
                pass
            else:
                # Unexpected phase, reset
                state = 'idle'
    
    # Handle last rep if it ends at EOF
    if state == 'concentric' and ecc_start is not None and con_start is not None:
        rep_end = frame_count
        
        if bottom_frame is not None and bottom_frame < len(knee_angles):
            min_knee_angle = knee_angles[bottom_frame]
            max_hip_disp = hip_displacement[bottom_frame]
        else:
            search_range = knee_angles[ecc_start:rep_end]
            if search_range:
                valid_angles = [a for a in search_range if not np.isnan(a)]
                min_knee_angle = min(valid_angles) if valid_angles else 180
            else:
                min_knee_angle = 180
            max_hip_disp = max(hip_displacement[ecc_start:rep_end]) if ecc_start < rep_end else 0
        
        rep = {
            'rep_id': len(repetitions) + 1,
            'start_frame': rep_start,
            'end_frame': rep_end,
            'duration_frames': rep_end - rep_start,
            'duration_seconds': (rep_end - rep_start) / fps,
            'phases': {
                'eccentric': {'start': ecc_start, 'end': iso_start if iso_start else con_start},
                'isometric': {'start': iso_start, 'end': con_start} if iso_start else None,
                'concentric': {'start': con_start, 'end': rep_end},
            },
            'bottom_frame': bottom_frame,
            'min_knee_angle': float(min_knee_angle) if not np.isnan(min_knee_angle) else None,
            'max_hip_displacement': float(max_hip_disp),
            'depth_category': detect_squat_depth_category(min_knee_angle) if not np.isnan(min_knee_angle) else 'unknown',
        }
        repetitions.append(rep)
    
    return repetitions


# =============================================================================
# ERROR ANALYSIS
# =============================================================================

def analyze_rep_errors(keypoints: List, rep: Dict, 
                       min_conf: float = MIN_LANDMARK_CONFIDENCE) -> Dict:
    """
    Analyze form errors for a single repetition.
    
    Returns error metrics aggregated by phase.
    Handles both dict and list formats for phases (from different sources).
    """
    start = rep['start_frame']
    end = rep['end_frame']
    phases = rep.get('phases', {})
    
    errors = {
        'overall': {},
        'by_phase': {},
    }
    
    # Analyze entire rep
    rep_indices = list(range(start, end))
    errors['overall'] = aggregate_phase_errors(keypoints, rep_indices, min_conf)
    
    # Handle both dict and list formats for phases
    if isinstance(phases, dict):
        # Format from extract_repetitions_from_phases
        if phases.get('eccentric'):
            ecc = phases['eccentric']
            ecc_indices = list(range(ecc['start'], ecc['end']))
            errors['by_phase']['eccentric'] = aggregate_phase_errors(keypoints, ecc_indices, min_conf)
        
        if phases.get('concentric'):
            con = phases['concentric']
            con_indices = list(range(con['start'], con['end']))
            errors['by_phase']['concentric'] = aggregate_phase_errors(keypoints, con_indices, min_conf)
    
    elif isinstance(phases, list):
        # Format from 5_temporal_segmentation.py
        for phase in phases:
            if isinstance(phase, dict):
                phase_type = phase.get('phase_type')
                phase_start = phase.get('start_frame')
                phase_end = phase.get('end_frame')
                
                if phase_type and phase_start is not None and phase_end is not None:
                    phase_indices = list(range(phase_start, phase_end))
                    errors['by_phase'][phase_type] = aggregate_phase_errors(keypoints, phase_indices, min_conf)
    
    # Add min knee angle for depth scoring
    if rep.get('min_knee_angle') is not None:
        errors['overall']['min_knee_angle'] = rep['min_knee_angle']
    
    # Get bottom frame errors (most critical point)
    if rep.get('bottom_frame') is not None:
        bottom_idx = rep['bottom_frame']
        if bottom_idx < len(keypoints):
            errors['at_bottom'] = get_all_errors_for_frame(keypoints[bottom_idx], min_conf)
    
    return errors


# =============================================================================
# MAIN AQA METAPROGRAM
# =============================================================================

def aqa_metaprogram_squat(video_id: str, 
                          features_json_path: Optional[str] = None,
                          segmented_json_path: Optional[str] = None,
                          use_existing_segmentation: bool = True) -> Dict:
    """
    Main AQA metaprogram for squat analysis.
    
    Orchestrates:
    1. Feature loading
    2. View classification
    3. Temporal segmentation (or loading existing)
    4. Error detection per rep
    5. Scoring
    
    Args:
        video_id: Video identifier
        features_json_path: Optional path to features JSON
        segmented_json_path: Optional path to segmented JSON
        use_existing_segmentation: If True, try to load existing segmentation first
        
    Returns:
        Complete analysis dictionary
    """
    print(f"\n{'='*60}")
    print(f"AQA MetaProgram - Squat Analysis: {video_id}")
    print('='*60)
    
    # Step 1: Load feature data
    print("\n[1/5] Loading extracted features...")
    
    if features_json_path is None:
        features_json_path = find_feature_json(video_id)
    
    if features_json_path is None or not os.path.exists(features_json_path):
        return {
            'video_id': video_id,
            'success': False,
            'error': f"Features not found for {video_id}",
        }
    
    feature_data = load_feature_data(features_json_path)
    keypoints = feature_data.get('keypoints_img', [])
    quality_level = get_quality_level(features_json_path)
    
    print(f"  Loaded {len(keypoints)} frames")
    print(f"  Quality level: {quality_level}")
    
    # Get metadata
    fps = feature_data.get('info', {}).get('fps', 30.0)
    frame_count = len(keypoints)
    
    # Step 2: View classification
    print("\n[2/5] Classifying camera view...")
    
    # Check if view already classified
    existing_view = feature_data.get('info', {}).get('view')
    if existing_view:
        view_label = existing_view
        view_type = ViewType(existing_view) if existing_view in [v.value for v in ViewType] else ViewType.UNKNOWN
        print(f"  Using existing classification: {view_label}")
    else:
        view_type, view_metrics = classify_view(keypoints)
        view_label = view_type.value
        print(f"  Classified as: {view_label}")
    
    # Step 3: Temporal segmentation
    print("\n[3/5] Temporal segmentation...")
    
    segmentation_data = None
    repetitions = None
    
    # Try loading existing segmentation first
    if use_existing_segmentation:
        if segmented_json_path is None:
            segmented_json_path = find_segmented_json(video_id)
        
        if segmented_json_path and os.path.exists(segmented_json_path):
            print(f"  Loading existing segmentation...")
            seg_data = load_segmented_data(segmented_json_path)
            
            phase_labels = seg_data.get('frame_phases', [])
            repetitions = seg_data.get('repetitions', [])
            
            segmentation_data = {
                'source': 'existing',
                'phase_labels': phase_labels,
                'hip_displacement': seg_data.get('signals', {}).get('hip_y', []),
                'knee_angles': seg_data.get('signals', {}).get('knee_angle', []),
                'velocity': seg_data.get('signals', {}).get('velocity', []),
            }
            print(f"  Loaded {len(repetitions)} repetitions from existing data")
    
    # If no existing segmentation, compute it
    if segmentation_data is None:
        print(f"  Computing segmentation...")
        segmentation_data = segment_into_phases(keypoints, fps)
        segmentation_data['source'] = 'computed'
        
        # Extract repetitions
        repetitions = extract_repetitions_from_phases(
            segmentation_data['phase_labels'],
            segmentation_data['hip_displacement'],
            segmentation_data['knee_angles'],
            fps
        )
        print(f"  Detected {len(repetitions)} repetitions")
    
    # Step 4: Error analysis
    print("\n[4/5] Analyzing form errors...")
    
    for i, rep in enumerate(repetitions):
        print(f"  Analyzing rep {rep['rep_id']}...")
        rep['errors'] = analyze_rep_errors(keypoints, rep)
    
    # Step 5: Scoring
    print("\n[5/5] Computing scores...")
    
    squat_data = {
        'video_id': video_id,
        'repetitions': repetitions,
    }
    
    set_scores = compute_set_score(squat_data)
    
    print(f"  Overall score: {set_scores.get('overall_score', 0)}/100 ({set_scores.get('grade', '?')})")
    
    # Compile final output
    result = {
        'video_id': video_id,
        'success': True,
        'timestamp': datetime.now().isoformat(),
        'quality_level': quality_level,
        'view': view_label,
        'frame_count': frame_count,
        'fps': fps,
        'rep_count': len(repetitions),
        'repetitions': repetitions,
        'signals': {
            'hip_displacement': segmentation_data.get('hip_displacement', []),
            'knee_angles': segmentation_data.get('knee_angles', []),
            'velocity': segmentation_data.get('velocity', []),
        },
        'phase_labels': segmentation_data.get('phase_labels', []),
        'scores': set_scores,
        'scale_factors': {
            'torso_length': get_scale_factor(feature_data),
            'leg_length': get_leg_length(feature_data),
        },
    }
    
    return result


def save_analysis(result: Dict, output_dir: str = OUTPUT_DIR) -> str:
    """Save analysis result to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    video_id = result.get('video_id', 'unknown')
    output_path = os.path.join(output_dir, f"{video_id}_aqa.json")
    
    # Convert numpy types to native Python for JSON serialization
    def convert_to_serializable(obj):
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
        return obj
    
    result = convert_to_serializable(result)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return output_path


def print_summary(result: Dict):
    """Print analysis summary to console."""
    if not result.get('success'):
        print(f"\n❌ Analysis failed: {result.get('error')}")
        return
    
    print(f"\n{'='*60}")
    print("ANALYSIS SUMMARY")
    print('='*60)
    
    print(f"Video ID: {result['video_id']}")
    print(f"Quality: {result['quality_level']}")
    print(f"View: {result['view']}")
    print(f"Frames: {result['frame_count']} ({result['frame_count']/result['fps']:.1f}s)")
    print(f"Repetitions: {result['rep_count']}")
    
    scores = result.get('scores', {})
    print(f"\nOverall Score: {scores.get('overall_score', 0)}/100 ({scores.get('grade', '?')})")
    
    if scores.get('rep_scores'):
        print("\nPer-Rep Breakdown:")
        for rep_score in scores['rep_scores']:
            print(f"  Rep {rep_score.get('rep_id', '?')}: {rep_score.get('overall_score', 0)}/100")
            
            # Show individual scores
            ind_scores = rep_score.get('individual_scores', {})
            for metric, score in ind_scores.items():
                print(f"    - {metric}: {score:.0f}")
    
    # Show feedback for worst metric
    if scores.get('rep_scores'):
        all_feedback = []
        for rep_score in scores['rep_scores']:
            feedback = rep_score.get('feedback', {})
            ind_scores = rep_score.get('individual_scores', {})
            for metric, score in ind_scores.items():
                if score < 75 and metric in feedback:
                    all_feedback.append((metric, score, feedback[metric]))
        
        if all_feedback:
            # Sort by score (lowest first)
            all_feedback.sort(key=lambda x: x[1])
            print("\n⚠️  Areas for Improvement:")
            seen = set()
            for metric, score, text in all_feedback[:3]:
                if metric not in seen:
                    print(f"  • {text}")
                    seen.add(metric)
    
    print('='*60)


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_all_videos(quality_filter: Optional[str] = None) -> List[Dict]:
    """
    Process all available videos.
    
    Args:
        quality_filter: Optional filter ('excellent', 'good', 'fair')
        
    Returns:
        List of analysis results
    """
    # Find all feature JSONs
    json_files = []
    
    for features_dir in FEATURES_DIRS:
        if quality_filter and quality_filter not in features_dir.lower():
            continue
        
        if os.path.exists(features_dir):
            for f in os.listdir(features_dir):
                if f.endswith('.json'):
                    json_files.append(os.path.join(features_dir, f))
    
    print(f"Found {len(json_files)} videos to process")
    
    results = []
    for json_path in tqdm(json_files, desc="Processing videos"):
        video_id = os.path.splitext(os.path.basename(json_path))[0]
        
        try:
            result = aqa_metaprogram_squat(video_id, features_json_path=json_path)
            
            if result.get('success'):
                save_analysis(result)
            
            results.append(result)
        except Exception as e:
            results.append({
                'video_id': video_id,
                'success': False,
                'error': str(e),
            })
    
    # Summary
    successes = sum(1 for r in results if r.get('success'))
    print(f"\nProcessed {len(results)} videos: {successes} successful, {len(results) - successes} failed")
    
    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="AQA MetaProgram for Squat Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aqa_metaProgram_squat.py 25713_3          # Analyze specific video
  python aqa_metaProgram_squat.py --all            # Process all videos
  python aqa_metaProgram_squat.py --all --quality good  # Process only 'good' quality
  python aqa_metaProgram_squat.py --json path.json # Analyze from specific JSON
        """
    )
    
    parser.add_argument('video_id', nargs='?', help='Video ID to analyze')
    parser.add_argument('--all', action='store_true', help='Process all available videos')
    parser.add_argument('--json', type=str, help='Path to features JSON file')
    parser.add_argument('--quality', type=str, choices=['excellent', 'good', 'fair'],
                       help='Quality filter for batch processing')
    parser.add_argument('--no-save', action='store_true', help='Do not save output')
    
    args = parser.parse_args()
    
    if args.all:
        # Batch processing
        results = process_all_videos(quality_filter=args.quality)
        
    elif args.json:
        # Process specific JSON
        video_id = os.path.splitext(os.path.basename(args.json))[0]
        result = aqa_metaprogram_squat(video_id, features_json_path=args.json)
        print_summary(result)
        
        if not args.no_save and result.get('success'):
            output_path = save_analysis(result)
            print(f"\n📄 Saved to: {output_path}")
        
    elif args.video_id:
        # Process single video by ID
        result = aqa_metaprogram_squat(args.video_id)
        print_summary(result)
        
        if not args.no_save and result.get('success'):
            output_path = save_analysis(result)
            print(f"\n📄 Saved to: {output_path}")
    
    else:
        parser.print_help()
