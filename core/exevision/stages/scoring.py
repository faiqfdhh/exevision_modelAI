"""
8_squat_aqa_simple.py
 
Single-file, minimal rule-based squat AQA (scoring + errors) compatible with your pipeline outputs.

Inputs (preferred):
- 2.5 output JSON: ./squat/extracted_features_clean/{quality}/{video_id}.json
- 5 output JSON (optional but recommended): ./squat/segmented_reps/{quality}/{video_id}_segmented.json

What it does (simple):
1) Load keypoints from 2.5 JSON.
2) Load phases/reps from 5 JSON if present; otherwise do a basic phase estimate.
3) Rep counting from phase order only:
   - eccentric -> concentric  (1 rep)
   - eccentric -> isometric -> concentric (1 rep)
   - supports fast reps: ... concentric -> eccentric (next rep starts immediately)
4) Compute a few basic metrics per rep (mostly at bottom + rep-average).
5) Produce an overall 0-100 score and per-rep breakdown.

Usage:
  python scripts/8_squat_aqa_simple.py 25713_3
  python scripts/8_squat_aqa_simple.py --json ./squat/extracted_features_clean/good/25713_3.json
  python scripts/8_squat_aqa_simple.py --seg ./squat/segmented_reps/good/25713_3_segmented.json --json ./squat/extracted_features_clean/good/25713_3.json

Output:
  ./squat/aqa_analysis_simple/{video_id}_aqa_simple.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# --------------------------
# Paths / discovery
# --------------------------
FEATURES_ROOT = Path("./squat/extracted_features_clean")
SEGMENTED_ROOT = Path("./squat/segmented_reps")
OUTPUT_DIR = "./squat/aqa_analysis_simple"


def _build_scoring_paths(exercise: str):
    """Compute path constants for the given exercise."""
    return {
        "features_root": Path(f"./{exercise}/extracted_features_clean"),
        "segmented_root": Path(f"./{exercise}/segmented_reps"),
        "output_dir": f"./{exercise}/aqa_analysis_simple",
    }


# --------------------------
# MediaPipe indices
# --------------------------
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32


# --------------------------
# Helpers: landmarks + geometry
# --------------------------

def _lm(frame: Any, idx: int) -> Optional[List[float]]:
        if frame is None or not isinstance(frame, list) or idx >= len(frame):
                return None
        lm = frame[idx]
        if not isinstance(lm, list) or len(lm) < 2:
                return None
        return lm


def _conf(frame: Any, idx: int) -> float:
        lm = _lm(frame, idx)
        if lm is None or len(lm) < 4:
                return 0.0
        return float(lm[3])


def _valid(frame: Any, indices: List[int], min_conf: float = 0.4) -> bool:
        return all(_conf(frame, i) >= min_conf for i in indices)


def _angle_2d(a: List[float], b: List[float], c: List[float]) -> Optional[float]:
        # angle at b formed by a-b-c
        v1 = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
        v2 = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
                return None
        cosv = float(np.dot(v1, v2) / (n1 * n2))
        cosv = float(np.clip(cosv, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosv)))


def knee_angle(frame: Any, min_conf: float = 0.4) -> Optional[float]:
        # average left/right knee angle (2D)
        angles = []
        if _valid(frame, [L_HIP, L_KNEE, L_ANKLE], min_conf):
                a = _angle_2d(_lm(frame, L_HIP), _lm(frame, L_KNEE), _lm(frame, L_ANKLE))
                if a is not None:
                        angles.append(a)
        if _valid(frame, [R_HIP, R_KNEE, R_ANKLE], min_conf):
                a = _angle_2d(_lm(frame, R_HIP), _lm(frame, R_KNEE), _lm(frame, R_ANKLE))
                if a is not None:
                        angles.append(a)
        if not angles:
                return None
        return float(np.mean(angles))


def knee_valgus_ratio(frame: Any, min_conf: float = 0.4) -> Optional[float]:
        # knee spread / ankle spread (<1 tends to valgus)
        if not _valid(frame, [L_KNEE, R_KNEE, L_ANKLE, R_ANKLE], min_conf):
                return None
        lk = _lm(frame, L_KNEE)
        rk = _lm(frame, R_KNEE)
        la = _lm(frame, L_ANKLE)
        ra = _lm(frame, R_ANKLE)
        knee_spread = abs(lk[0] - rk[0])
        ankle_spread = abs(la[0] - ra[0])
        if ankle_spread < 1e-3:
                return None
        return float(knee_spread / ankle_spread)


def forward_lean_deg(frame: Any, min_conf: float = 0.4) -> Optional[float]:
        # angle of (hip->shoulder) from vertical. 0=upright.
        if not _valid(frame, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP], min_conf):
                return None
        ls = _lm(frame, L_SHOULDER)
        rs = _lm(frame, R_SHOULDER)
        lh = _lm(frame, L_HIP)
        rh = _lm(frame, R_HIP)
        shoulder_mid = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2], dtype=float)
        hip_mid = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2], dtype=float)
        dx = float(shoulder_mid[0] - hip_mid[0])
        dy = float(shoulder_mid[1] - hip_mid[1])
        # image coords: +y down. Vertical up is (0,-1). Use atan2(dx, -dy)
        return float(np.degrees(np.arctan2(dx, -dy)))


def hip_y(frame: Any, min_conf: float = 0.4) -> Optional[float]:
        if not _valid(frame, [L_HIP, R_HIP], min_conf):
                return None
        lh = _lm(frame, L_HIP)
        rh = _lm(frame, R_HIP)
        return float((lh[1] + rh[1]) / 2)


def calculate_vertical_depth(frame: Any, min_conf: float = 0.4) -> Optional[float]:
        """
        Calculates depth based on vertical hip vs knee position, per-leg.
        Returns the BEST (most positive) depth reading across both valid legs.
        Only one leg needs to show depth — avoids being penalised by a
        poorly-tracked contralateral knee (common in diagonal/front-side views).

        Returns: Positive value if hip is BELOW knee (Good), Negative if ABOVE (Bad).
        Normalized by femur length for scale invariance.
        """
        candidates = []

        # Left leg
        if _valid(frame, [L_HIP, L_KNEE], min_conf):
                lh = _lm(frame, L_HIP)
                lk = _lm(frame, L_KNEE)
                femur = abs(lh[1] - lk[1])
                if femur >= 1e-3:
                        candidates.append((lh[1] - lk[1]) / femur)

        # Right leg
        if _valid(frame, [R_HIP, R_KNEE], min_conf):
                rh = _lm(frame, R_HIP)
                rk = _lm(frame, R_KNEE)
                femur = abs(rh[1] - rk[1])
                if femur >= 1e-3:
                        candidates.append((rh[1] - rk[1]) / femur)

        if not candidates:
                return None

        # Use the best (deepest) reading: one good leg is sufficient
        return float(max(candidates))


def calculate_torso_tibia_offset(frame: Any, min_conf: float = 0.4) -> Optional[float]:
        """
        Calculates the absolute difference between torso angle and shin angle.
        Returns: Degrees of deviation (0 is perfect parallel).
        Lower values = better posture (torso parallel to shins).
        """
        if not _valid(frame, [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE], min_conf):
                return None
    
        ls = _lm(frame, L_SHOULDER)
        rs = _lm(frame, R_SHOULDER)
        lh = _lm(frame, L_HIP)
        rh = _lm(frame, R_HIP)
        lk = _lm(frame, L_KNEE)
        rk = _lm(frame, R_KNEE)
        la = _lm(frame, L_ANKLE)
        ra = _lm(frame, R_ANKLE)
    
        # Calculate midpoints
        shoulder = np.array([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2])
        hip = np.array([(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2])
        knee = np.array([(lk[0] + rk[0]) / 2, (lk[1] + rk[1]) / 2])
        ankle = np.array([(la[0] + ra[0]) / 2, (la[1] + ra[1]) / 2])
    
        # Torso vector (shoulder to hip)
        torso_vector = shoulder - hip
        torso_angle = np.degrees(np.arctan2(abs(torso_vector[0]), abs(torso_vector[1])))
    
        # Tibia/Shin vector (knee to ankle)
        tibia_vector = knee - ankle
        tibia_angle = np.degrees(np.arctan2(abs(tibia_vector[0]), abs(tibia_vector[1])))
    
        # Return absolute difference
        return float(abs(torso_angle - tibia_angle))


# --------------------------
# Phase + rep detection (simple)
# --------------------------

def _fallback_single_json(root: Path, suffix: str) -> Optional[str]:
    """Fallback for single-video API runs when id mapping drifts across stages."""
    candidates = sorted(root.rglob(f"*{suffix}"))
    if len(candidates) == 1:
        return str(candidates[0])
    return None


def find_feature_json(video_id: str) -> Optional[str]:
    if not FEATURES_ROOT.exists():
        return None

    exact_matches = sorted(FEATURES_ROOT.rglob(f"{video_id}.json"))
    if exact_matches:
        return str(exact_matches[0])

    return _fallback_single_json(FEATURES_ROOT, ".json")


def find_segmented_json(video_id: str) -> Optional[str]:
    if not SEGMENTED_ROOT.exists():
        return None

    exact_matches = sorted(SEGMENTED_ROOT.rglob(f"{video_id}_segmented.json"))
    if exact_matches:
        return str(exact_matches[0])

    return _fallback_single_json(SEGMENTED_ROOT, "_segmented.json")


def find_all_video_ids() -> List[Tuple[str, str]]:
    """Find all video IDs from the extracted feature tree, regardless of tier layout."""
    if not FEATURES_ROOT.exists():
        return []

    video_ids: List[Tuple[str, str]] = []
    for path in FEATURES_ROOT.rglob("*.json"):
        quality = path.parent.name.lower()
        video_ids.append((path.stem, quality))
    return video_ids


def get_output_path(video_id: str, overall_score: float, source_quality: str) -> str:
    """Get output path based on source quality and score tier.
    
    Args:
        video_id: Video identifier
        overall_score: Overall score (0-100)
        source_quality: Source folder (excellent, good, fair)
    
    Returns:
        Full path to output JSON file
    """
    # Determine score tier
    if overall_score >= 80:
        score_tier = "good"
    elif overall_score >= 60:
        score_tier = "acceptable"
    else:
        score_tier = "poor"
    
    # Create nested directory: OUTPUT_DIR/source_quality/score_tier/
    output_dir = os.path.join(OUTPUT_DIR, source_quality, score_tier)
    os.makedirs(output_dir, exist_ok=True)
    
    return os.path.join(output_dir, f"{video_id}_aqa_simple.json")


def _basic_phases_from_hip(keypoints: List[Any], fps: float) -> List[str]:
    # very simple: derive hip_y velocity; threshold to label
    ys = []
    for fr in keypoints:
        y = hip_y(fr)
        ys.append(y if y is not None else np.nan)
    ys = np.array(ys, dtype=float)
    if np.isnan(ys).all():
        return ["unknown"] * len(keypoints)
    # interpolate
    nans = np.isnan(ys)
    if nans.any():
        idx = np.arange(len(ys))
        ys[nans] = np.interp(idx[nans], idx[~nans], ys[~nans])
    # normalize relative to standing (low y) => standing, squat => higher y
    standing = float(np.percentile(ys, 10))
    disp = ys - standing
    v = np.gradient(disp)
    # smooth
    if len(v) >= 7:
        k = 7
        v = np.convolve(v, np.ones(k)/k, mode="same")
    idle_thr = 0.002
    phases = []
    for vi in v:
        if abs(vi) < idle_thr:
            phases.append("idle")
        elif vi > 0:
            phases.append("eccentric")
        else:
            phases.append("concentric")
    return phases


@dataclass
class Rep:
    rep_id: int
    start: int
    end: int
    bottom: int


def reps_from_phases(phases: List[str], min_rep_frames: int = 6) -> List[Rep]:
    """Count reps purely from phase order.

    Valid reps:
      - eccentric -> concentric
      - eccentric -> isometric -> concentric

    End conditions:
      - concentric -> idle
      - concentric -> eccentric (fast reps; next rep starts immediately)
      - EOF while in concentric
    """
    reps: List[Rep] = []

    WAIT_ECC, IN_ECC, IN_ISO, IN_CONC = 0, 1, 2, 3
    state = WAIT_ECC
    start = None
    saw_conc = False

    def finalize(end_idx: int):
        nonlocal start
        if start is None:
            return
        if end_idx < start:
            return
        if (end_idx - start + 1) < min_rep_frames:
            start = None
            return
        bottom = (start + end_idx) // 2
        reps.append(Rep(rep_id=len(reps) + 1, start=start, end=end_idx, bottom=bottom))
        start = None

    for i, p in enumerate(phases):
        if state == WAIT_ECC:
            if p == "eccentric":
                state = IN_ECC
                start = i
                saw_conc = False
            else:
                continue

        elif state == IN_ECC:
            if p == "eccentric":
                continue
            if p == "isometric":
                state = IN_ISO
                continue
            if p == "concentric":
                state = IN_CONC
                saw_conc = True
                continue
            if p == "idle":
                # aborted
                state = WAIT_ECC
                start = None
                continue

        elif state == IN_ISO:
            if p == "isometric":
                continue
            if p == "concentric":
                state = IN_CONC
                saw_conc = True
                continue
            if p == "eccentric":
                # bounce back
                state = IN_ECC
                continue
            if p == "idle":
                state = WAIT_ECC
                start = None
                continue

        elif state == IN_CONC:
            if p == "concentric" or p == "isometric":
                continue
            if p == "idle":
                finalize(i - 1)
                state = WAIT_ECC
                continue
            if p == "eccentric":
                finalize(i - 1)
                state = IN_ECC
                start = i
                saw_conc = False
                continue
            # unknown: ignore

    if state == IN_CONC and start is not None:
        finalize(len(phases) - 1)

    return reps


# --------------------------
# Scoring (view-specific)
# --------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def score_metric_linear(value: float, good: float, bad: float, higher_is_better: bool) -> float:
    """Map value to [0,100] using a simple linear ramp.

    If higher_is_better:
      value>=good => 100, value<=bad => 0
    Else:
      value<=good => 100, value>=bad => 0
    """
    if higher_is_better:
        if value >= good:
            return 100.0
        if value <= bad:
            return 0.0
        return 100.0 * (value - bad) / (good - bad)
    else:
        if value <= good:
            return 100.0
        if value >= bad:
            return 0.0
        return 100.0 * (bad - value) / (bad - good)


def get_view_weights_and_thresholds(view: str) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Return metric weights and thresholds customized per view.
    
    Views: "front", "back", "side", "front_side", "back_side"
    
    Returns:
        (weights, thresholds) where:
        - weights: dict of metric -> weight (0.0-1.0)
        - thresholds: dict of metric -> {"good": float, "bad": float, "higher_is_better": bool}
    """
    view_lower = view.lower()
    
    # Default thresholds
    default_thresholds = {
        "knee_valgus": {"good": 0.95, "bad": 0.75, "higher_is_better": True},
        "forward_lean": {"good": 25.0, "bad": 50.0, "higher_is_better": False},
        "depth": {"good": 75.0, "bad": 110.0, "higher_is_better": False},
        "squat_depth": {"good": 0.1, "bad": -0.5, "higher_is_better": True},
    }
    
    # SIDE VIEW: Best for forward lean and depth; poor for valgus and lateral shift
    if "side" in view_lower and "front" not in view_lower and "back" not in view_lower:
        weights = {
            "knee_valgus": 0.05,    # Unreliable from side
            "forward_lean": 0.30,   # Excellent visibility
            "depth": 0.30,          # Excellent visibility
            "squat_depth": 0.30,    # Excellent visibility
        }
        thresholds = {
            "knee_valgus": {"good": 0.95, "bad": 0.70, "higher_is_better": True},  # Less strict (unreliable)
            "forward_lean": {"good": 35.0, "bad": 60.0, "higher_is_better": False},
            "depth": {"good": 50.0, "bad": 100.0, "higher_is_better": False},
            "squat_depth": {"good": 0.15, "bad": -0.5, "higher_is_better": True},  # Excellent visibility
        }
    
    # FRONT/BACK VIEW: Best for valgus and hip shift; moderate depth; poor lean
    elif view_lower in ["front", "back"]:
        weights = {
            "knee_valgus": 0.35,    # Excellent visibility
            "forward_lean": 0.05,   # Poor visibility (2D projection)
            "depth": 0.30,          # Moderate visibility
            "squat_depth": 0.30,    # Moderate visibility
        }
        thresholds = {
            "knee_valgus": {"good": 0.97, "bad": 0.80, "higher_is_better": True},  # Stricter (clear view)
            "forward_lean": {"good": 30.0, "bad": 55.0, "higher_is_better": False},
            "depth": {"good": 80.0, "bad": 120.0, "higher_is_better": False},
            "squat_depth": {"good": 0.08, "bad": -0.5, "higher_is_better": True},  # Moderate visibility
        }
    
    # FRONT_SIDE (diagonal front): Balanced view, all metrics moderately visible
    elif "front_side" in view_lower or "front-side" in view_lower:
        weights = {
            "knee_valgus": 0.1,
            "forward_lean": 0.1,
            "depth": 0.30,
            "squat_depth": 0.30,
        }
        thresholds = {
            "knee_valgus": {"good": 0.95, "bad": 0.78, "higher_is_better": True},
            "forward_lean": {"good": 22.0, "bad": 45.0, "higher_is_better": False},
            "depth": {"good": 75.0, "bad": 112.0, "higher_is_better": False},
            "squat_depth": {"good": 0.1, "bad": -0.5, "higher_is_better": True},
        }

    # BACK_SIDE (diagonal back): Similar to front_side
    elif "back_side" in view_lower or "back-side" in view_lower:
        weights = {
            "knee_valgus": 0.1,
            "forward_lean": 0.1,
            "depth": 0.30,
            "squat_depth": 0.30,
        }
        thresholds = {
            "knee_valgus": {"good": 1.2, "bad": 0.78, "higher_is_better": True},
            "forward_lean": {"good": 35.0, "bad": 55.0, "higher_is_better": False},
            "depth": {"good": 75.0, "bad": 115.0, "higher_is_better": False},
            "squat_depth": {"good": 0.1, "bad": -0.5, "higher_is_better": True},
        }

    # DEFAULT: Balanced weights (unknown view)
    else:
        weights = {
            "knee_valgus": 0.25,
            "forward_lean": 0.20,
            "depth": 0.20,
            "squat_depth": 0.20,
        }
        thresholds = default_thresholds
    
    return weights, thresholds


def score_rep_simple(metrics: Dict[str, Optional[float]], view: str = "unknown") -> Dict[str, Any]:
    """
    Score a rep based on metrics, with view-specific weights and thresholds.
    
    Args:
        metrics: Dictionary of computed metrics
        view: Camera view ("front", "back", "side", "front_side", "back_side")
    """
    weights, thresholds = get_view_weights_and_thresholds(view)
    
    scores: Dict[str, float] = {}
    
    # Score each available metric
    if metrics.get("knee_valgus") is not None:
        t = thresholds["knee_valgus"]
        scores["knee_valgus"] = score_metric_linear(
            metrics["knee_valgus"], 
            good=t["good"], 
            bad=t["bad"], 
            higher_is_better=t["higher_is_better"]
        )
    
    if metrics.get("forward_lean") is not None:
        t = thresholds["forward_lean"]
        scores["forward_lean"] = score_metric_linear(
            abs(metrics["forward_lean"]), 
            good=t["good"], 
            bad=t["bad"], 
            higher_is_better=t["higher_is_better"]
        )
    
    if metrics.get("min_knee_angle") is not None:
        t = thresholds["depth"]
        scores["depth"] = score_metric_linear(
            metrics["min_knee_angle"], 
            good=t["good"], 
            bad=t["bad"], 
            higher_is_better=t["higher_is_better"]
        )
    
    if metrics.get("squat_depth") is not None:
        t = thresholds["squat_depth"]
        # Floor at 20: even the shallowest squat gets at least 20% credit instead of 0
        scores["squat_depth"] = max(20.0, score_metric_linear(
            metrics["squat_depth"],
            good=t["good"],
            bad=t["bad"],
            higher_is_better=t["higher_is_better"]
        ))
    
    # Normalize weights to sum to 1.0
    total_weight = sum(w for k, w in weights.items() if k in scores)
    if total_weight > 0:
        weights = {k: (v / total_weight) for k, v in weights.items()}

    weighted_sum = 0.0
    for k, w in weights.items():
        if k in scores:
            weighted_sum += scores[k] * w

    overall = weighted_sum  # No division needed now
    
    return {
        "overall_score": round(float(overall), 1),
        "metric_scores": {k: round(float(v), 1) for k, v in scores.items()},
        "view": view,
        "weights_used": {k: round(v, 2) for k, v in weights.items() if k in scores},
    }


# --------------------------
# Main
# --------------------------

def main() -> int:
    global FEATURES_ROOT, SEGMENTED_ROOT, OUTPUT_DIR

    parser = argparse.ArgumentParser(description="Simple squat AQA (uses reps from 5_temporal_segmentation)")
    parser.add_argument("video_id", nargs="?", help="Video ID (e.g., 25713_3) or '*' to process all videos")
    parser.add_argument("--no-save", action="store_true", help="Do not save output")
    parser.add_argument("--exercise", default="squat", help="Exercise type (default: squat)")
    args = parser.parse_args()

    # Update paths based on exercise
    paths = _build_scoring_paths(args.exercise)
    FEATURES_ROOT = paths["features_root"]
    SEGMENTED_ROOT = paths["segmented_root"]
    OUTPUT_DIR = paths["output_dir"]

    if not args.video_id:
        parser.print_help()
        return 2

    # Handle wildcard: process all videos
    if args.video_id == "*":
        video_list = find_all_video_ids()
        if not video_list:
            print("❌ No videos found in any quality folder")
            return 1
        
        print(f"\n{'='*60}")
        print(f"Processing {len(video_list)} videos...")
        print(f"{'='*60}\n")
        
        processed = 0
        failed = 0
        
        for video_id, source_quality in video_list:
            result = process_single_video(video_id, source_quality, save_output=not args.no_save)
            if result is not None:
                processed += 1
                print(f"✓ [{processed}/{len(video_list)}] {video_id} ({source_quality}) → Score: {result['overall_score']}/100")
            else:
                failed += 1
                print(f"✗ [{processed+failed}/{len(video_list)}] {video_id} ({source_quality}) → Failed")
        
        print(f"\n{'='*60}")
        print(f"Summary: {processed} processed, {failed} failed")
        print(f"{'='*60}\n")
        return 0
    
    # Process single video
    video_id = args.video_id
    
    # Determine source quality from features path
    features_path = find_feature_json(video_id)
    if features_path:
        for quality in ["excellent", "good", "fair"]:
            if quality in features_path:
                source_quality = quality
                break
        else:
            source_quality = "unknown"
    else:
        source_quality = "unknown"
    
    result = process_single_video(video_id, source_quality, save_output=not args.no_save)
    if result is None:
        return 1
    
    print(f"\n{'='*60}")
    print(f"Video: {video_id}")
    print(f"View: {result['view']}")
    print(f"Reps: {result['rep_count']}")
    print(f"Overall score: {result['overall_score']}/100")
    print('='*60)
    
    return 0


def process_single_video(video_id: str, source_quality: str, save_output: bool = True) -> Optional[Dict[str, Any]]:
    """Process a single video and return the result.
    
    Args:
        video_id: Video identifier
        source_quality: Source quality folder (excellent, good, fair, unknown)
        save_output: Whether to save JSON output
    
    Returns:
        Result dictionary or None if processing failed
    """
    # Load features (2.5 output)
    features_path = find_feature_json(video_id)
    if not features_path or not os.path.exists(features_path):
        print(f"❌ Features JSON not found for {video_id}")
        if FEATURES_ROOT.exists():
            available = sorted(FEATURES_ROOT.rglob("*.json"))
            print(f"   Available feature JSON count: {len(available)}")
        return None

    with open(features_path, "r") as f:
        features = json.load(f)

    keypoints = features.get("keypoints_img", [])
    info = features.get("info", {})
    fps = float(info.get("fps", 30.0))
    view = info.get("view", "unknown")

    # Load segmented reps (5_temporal_segmentation output) - REQUIRED
    seg_path = find_segmented_json(video_id)
    if not seg_path or not os.path.exists(seg_path):
        print(f"❌ Segmented JSON not found for {video_id}")
        if SEGMENTED_ROOT.exists():
            available = sorted(SEGMENTED_ROOT.rglob("*_segmented.json"))
            print(f"   Available segmented JSON count: {len(available)}")
        print(f"   Run script 5_temporal_segmentation.py first")
        return None

    with open(seg_path, "r") as f:
        seg = json.load(f)

    reps_from_seg = seg.get("repetitions", [])
    if not reps_from_seg:
        print(f"⚠️  No repetitions found in segmented JSON")
        result = {
            "video_id": video_id,
            "features_json": features_path,
            "segmented_json": seg_path,
            "view": view,
            "fps": fps,
            "frame_count": len(keypoints),
            "rep_count": 0,
            "repetitions": [],
            "overall_score": 0.0,
            "source_quality": source_quality,
            "status": "no_reps_detected",
            "message": "No repetitions were detected during temporal segmentation.",
        }

        if save_output:
            out_path = get_output_path(video_id, result["overall_score"], source_quality)
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

        return result

    # Compute scores for each rep from 5's output
    rep_outputs = []
    for rep_dict in reps_from_seg:
        rep_id = rep_dict.get("rep_id", len(rep_outputs) + 1)
        start = rep_dict.get("start_frame")
        end = rep_dict.get("end_frame")
        bottom = rep_dict.get("bottom_frame", (start + end) // 2)

        if start is None or end is None:
            continue

        frames = keypoints[start:end + 1]

        # compute metrics: average over rep + min knee angle
        valgus_vals = []
        lean_vals = []
        knee_vals = []
        for fr in frames:
            v = knee_valgus_ratio(fr)
            if v is not None:
                valgus_vals.append(v)
            fl = forward_lean_deg(fr)
            if fl is not None:
                lean_vals.append(fl)
            ka = knee_angle(fr)
            if ka is not None:
                knee_vals.append(ka)

        min_knee = float(np.min(knee_vals)) if knee_vals else None

        # Calculate squat depth at the bottom frame
        depth_val = None
        if 0 <= bottom - start < len(frames):
            depth_val = calculate_vertical_depth(frames[bottom - start])
        below_parallel = depth_val is not None and depth_val > 0.0

        metrics = {
            "knee_valgus": float(np.mean(valgus_vals)) if valgus_vals else None,
            "forward_lean": float(np.mean(lean_vals)) if lean_vals else None,
            "min_knee_angle": min_knee,
            "squat_depth": depth_val,
            "below_parallel": below_parallel,
        }

        rep_score = score_rep_simple(metrics, view=view)

        rep_outputs.append({
            "rep_id": int(rep_id),
            "start_frame": int(start),
            "end_frame": int(end),
            "duration_frames": int(end - start + 1),
            "duration_seconds": round((end - start + 1) / fps, 2),
            "bottom_frame": int(bottom),
            "metrics": metrics,
            "score": rep_score,
        })

    if rep_outputs:
        overall = float(np.mean([r["score"]["overall_score"] for r in rep_outputs]))
    else:
        overall = 0.0

    result = {
        "video_id": video_id,
        "features_json": features_path,
        "segmented_json": seg_path,
        "view": view,
        "fps": fps,
        "frame_count": len(keypoints),
        "rep_count": len(rep_outputs),
        "repetitions": rep_outputs,
        "overall_score": round(overall, 1),
        "source_quality": source_quality,
        "status": "ok",
    }

    if save_output:
        out_path = get_output_path(video_id, result["overall_score"], source_quality)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    raise SystemExit(main())

