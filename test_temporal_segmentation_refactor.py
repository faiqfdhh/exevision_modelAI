#!/usr/bin/env python3
"""
Quick verification script for temporal_segmentation.py refactoring.
Tests that:
1. Control signal extraction works for squat and OHP
2. Exercise parameter threading works correctly
3. Debug output generation works
"""

import sys
import os
import json
import numpy as np

# Add core to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "core"))

from exevision.stages.temporal_segmentation import (
    _hip_y_sequence,
    _wrist_y_sequence_ohp,
    _get_control_signal,
    _get_thresholds,
    BiomechanicalAnalyzer,
    TemporalSegmenter,
    _debug_enabled,
)

def create_mock_keypoints(n_frames=100, include_wrist=True):
    """Create mock MediaPipe keypoints for testing."""
    keypoints = []
    for frame_idx in range(n_frames):
        frame = []
        for landmark_idx in range(33):
            # Create realistic values that change over time
            if landmark_idx == 23:  # L_HIP
                y = 0.5 + 0.1 * np.sin(frame_idx / 20.0)
            elif landmark_idx == 24:  # R_HIP
                y = 0.5 + 0.1 * np.sin(frame_idx / 20.0)
            elif landmark_idx == 15:  # L_WRIST
                y = 0.3 - 0.15 * np.sin(frame_idx / 20.0)  # Inverse for wrist
            elif landmark_idx == 16:  # R_WRIST
                y = 0.3 - 0.15 * np.sin(frame_idx / 20.0)
            else:
                y = 0.5
            
            # [x, y, z, confidence]
            frame.append([
                0.5 + 0.05 * np.random.randn(),
                y + 0.02 * np.random.randn(),
                0.0 + 0.02 * np.random.randn(),
                0.85 + 0.1 * np.random.randn()  # confidence
            ])
        keypoints.append(frame)
    
    return keypoints

def test_control_signal_extraction():
    """Test that control signal extraction works for both exercises."""
    print("\n" + "="*70)
    print("TEST 1: Control Signal Extraction")
    print("="*70)
    
    keypoints = create_mock_keypoints(100)
    
    # Test squat signal (hip Y)
    squat_signal = _get_control_signal(keypoints, exercise="squat")
    print(f"✓ Squat signal extracted: shape={squat_signal.shape}, range=[{np.nanmin(squat_signal):.3f}, {np.nanmax(squat_signal):.3f}]")
    assert squat_signal.shape[0] == 100, "Signal length mismatch"
    assert not np.all(np.isnan(squat_signal)), "Signal is all NaN"
    
    # Test OHP signal (inverted wrist Y)
    ohp_signal = _get_control_signal(keypoints, exercise="overhead_press")
    print(f"✓ OHP signal extracted: shape={ohp_signal.shape}, range=[{np.nanmin(ohp_signal):.3f}, {np.nanmax(ohp_signal):.3f}]")
    assert ohp_signal.shape[0] == 100, "Signal length mismatch"
    assert not np.all(np.isnan(ohp_signal)), "Signal is all NaN"
    
    # Verify signals are different (OHP should be inverted relative to squat)
    assert not np.allclose(squat_signal[~np.isnan(squat_signal)], ohp_signal[~np.isnan(ohp_signal)]), \
        "Squat and OHP signals should differ"
    print("✓ Squat and OHP signals are appropriately different")

def test_thresholds():
    """Test that exercise-specific thresholds are returned correctly."""
    print("\n" + "="*70)
    print("TEST 2: Exercise-Specific Thresholds")
    print("="*70)
    
    squat_thresh = _get_thresholds("squat")
    ohp_thresh = _get_thresholds("overhead_press")
    seated_ohp_thresh = _get_thresholds("seated_overhead_press")
    
    print(f"✓ Squat thresholds loaded: {len(squat_thresh)} params")
    print(f"✓ OHP thresholds loaded: {len(ohp_thresh)} params")
    print(f"✓ Seated OHP thresholds loaded: {len(seated_ohp_thresh)} params")
    
    assert "MIN_REP_FRAMES" in squat_thresh, "Missing MIN_REP_FRAMES"
    assert "MIN_DEPTH_RATIO" in ohp_thresh, "Missing MIN_DEPTH_RATIO"
    print("✓ All expected threshold parameters present")

def test_biomechanical_analyzer():
    """Test that BiomechanicalAnalyzer accepts exercise parameter."""
    print("\n" + "="*70)
    print("TEST 3: BiomechanicalAnalyzer Exercise Parameter")
    print("="*70)
    
    keypoints = create_mock_keypoints(100)
    
    # Test squat analyzer
    analyzer_squat = BiomechanicalAnalyzer(keypoints, view="side", exercise="squat")
    assert analyzer_squat.exercise == "squat", "Exercise not set"
    assert analyzer_squat.control_signal is None, "Control signal should be None before compute"
    print("✓ Squat analyzer initialized correctly")
    
    # Test OHP analyzer
    analyzer_ohp = BiomechanicalAnalyzer(keypoints, view="side", exercise="overhead_press")
    assert analyzer_ohp.exercise == "overhead_press", "Exercise not set"
    print("✓ OHP analyzer initialized correctly")
    
    # Test control signal computation
    analyzer_squat.calibrate_from_idle()
    squat_signal = analyzer_squat.compute_normalized_hip_displacement()
    print(f"✓ Squat analyzer computed signal: shape={squat_signal.shape}")
    
    analyzer_ohp.calibrate_from_idle()
    ohp_signal = analyzer_ohp.compute_normalized_hip_displacement()
    print(f"✓ OHP analyzer computed signal: shape={ohp_signal.shape}")

def test_debug_mode():
    """Test that debug mode can be enabled."""
    print("\n" + "="*70)
    print("TEST 4: Debug Mode")
    print("="*70)
    
    # Check default (should be disabled)
    enabled = _debug_enabled()
    print(f"✓ Default debug mode: {enabled}")
    
    # Enable via environment variable
    os.environ["DEBUG_PHASES"] = "1"
    enabled = _debug_enabled()
    assert enabled, "Debug mode should be enabled"
    print("✓ Debug mode can be enabled via environment variable")
    
    # Disable via environment variable
    os.environ["DEBUG_PHASES"] = "0"
    enabled = _debug_enabled()
    assert not enabled, "Debug mode should be disabled"
    print("✓ Debug mode can be disabled via environment variable")

def test_temporal_segmenter_exercise_param():
    """Test that TemporalSegmenter accepts exercise parameter."""
    print("\n" + "="*70)
    print("TEST 5: TemporalSegmenter Exercise Parameter")
    print("="*70)
    
    keypoints = create_mock_keypoints(100)
    mock_data = {
        "keypoints_img": keypoints,
        "info": {
            "view": "side",
            "fps": 30.0,
            "quality_rating": "good"
        }
    }
    
    # Test squat segmenter
    segmenter_squat = TemporalSegmenter(mock_data, video_id="test_001", exercise="squat")
    assert segmenter_squat.exercise == "squat", "Exercise not set"
    print("✓ TemporalSegmenter accepts exercise='squat'")
    
    # Test OHP segmenter
    segmenter_ohp = TemporalSegmenter(mock_data, video_id="test_002", exercise="overhead_press")
    assert segmenter_ohp.exercise == "overhead_press", "Exercise not set"
    print("✓ TemporalSegmenter accepts exercise='overhead_press'")

def main():
    """Run all verification tests."""
    print("\n" + "="*70)
    print("TEMPORAL SEGMENTATION REFACTORING VERIFICATION")
    print("="*70)
    
    try:
        test_control_signal_extraction()
        test_thresholds()
        test_biomechanical_analyzer()
        test_debug_mode()
        test_temporal_segmenter_exercise_param()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nImplementation Summary:")
        print("  • Control signals: Squat (hip Y) and OHP (inverted wrist Y)")
        print("  • Exercise parameter threading: Complete")
        print("  • Thresholds: Exercise-specific")
        print("  • Debug mode: Functional")
        print("\nNext Steps:")
        print("  1. Run temporal_segmentation.py on sample squat video (regression test)")
        print("  2. Run temporal_segmentation.py on sample OHP video with --debug-phases")
        print("  3. Verify debug JSON output shows correct phase sequences")
        print("\n" + "="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
