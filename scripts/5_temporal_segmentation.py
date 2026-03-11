"""
Temporal Segmentation Module (v6.0 - Simplified)

Segments squat motion into idle/eccentric/isometric/concentric phases.

Logic:
  - Hip Y rising (going down in frame)  → eccentric
  - Hip Y falling (going up in frame)   → concentric
  - Near-zero velocity at depth          → isometric
  - Near-zero velocity at standing       → idle

That's it. Smooth, threshold, classify, merge short blips.
"""

import os
import json
import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass
from typing import List, Tuple, Optional
from tqdm import tqdm
from enum import Enum

# --- Paths ---
FEATURES_DIRS = [
    "./squat/extracted_features_clean/excellent",
    "./squat/extracted_features_clean/good",
    "./squat/extracted_features_clean/fair",
]
VIDEO_IDS_TO_PROCESS = ["*"]
OUTPUT_DIRS = {
    "excellent": "./squat/segmented_reps/excellent",
    "good":      "./squat/segmented_reps/good",
    "fair":      "./squat/segmented_reps/fair",
}
VISUALIZATION_DIRS = {
    "excellent": "./squat/visualized_segmentation/excellent",
    "good":      "./squat/visualized_segmentation/good",
    "fair":      "./squat/visualized_segmentation/fair",
}
VIDEO_DIR = "./squat/dataset_videos_all"

PHASE_COLORS = {
    "idle":       (128, 128, 128),
    "eccentric":  (0, 0, 255),
    "isometric":  (255, 255, 0),
    "concentric": (0, 255, 0),
    "unknown":    (50, 50, 50),
}

# --- Landmark indices (MediaPipe) ---
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_SHOULDER, R_SHOULDER = 11, 12

# --- Tuning knobs ---
SMOOTH_WINDOW     = 15      # Savgol smoothing window (odd, frames)
SMOOTH_ORDER      = 3       # Savgol polynomial order
VEL_SMOOTH_SIZE   = 11      # Moving-average window for velocity
MIN_CONFIDENCE    = 0.4     # Discard landmarks below this
CALIBRATION_N     = 60      # Frames to estimate standing height

VEL_THRESH        = 0.006   # |velocity| above this = moving
DEPTH_FRAC_ISO    = 0.25    # Must be ≥25% of max depth for isometric
MIN_PHASE_FRAMES  = 10      # Merge phases shorter than this
MIN_REP_FRAMES    = 15      # Discard reps shorter than this

PHASE_NAMES = {0: "idle", 1: "eccentric", 2: "isometric", 3: "concentric"}


class Phase(Enum):
    IDLE = 0
    ECCENTRIC = 1
    ISOMETRIC = 2
    CONCENTRIC = 3


# ── Helpers ──────────────────────────────────────────────────────────────────

def _interp_nans(arr: np.ndarray) -> np.ndarray:
    """Linear-interpolate NaN gaps in-place, return same array."""
    nans = np.isnan(arr)
    if nans.all():
        arr[:] = 0
        return arr
    if nans.any():
        idx = np.arange(len(arr))
        arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
    return arr


def _angle_at(p1, p2, p3):
    """Angle (degrees) at p2 in the triangle p1-p2-p3."""
    v1, v2 = p1 - p2, p3 - p2
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return np.nan
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return np.degrees(np.arccos(cos_a))


def _to_native(obj):
    """Recursively convert numpy types → Python natives for JSON."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Enum):
        return obj.name.lower()
    return obj


def find_video_file(video_id: str, quality: Optional[str] = None) -> Optional[str]:
    """Locate video file: prefer annotated version, fall back to raw."""
    if quality:
        annotated = f"./squat/visualized_poses_clean/{quality.lower()}/{video_id}_annotated.mp4"
        if os.path.exists(annotated):
            return annotated
    for root, _, files in os.walk(VIDEO_DIR):
        for f in files:
            if os.path.splitext(f)[0] == video_id and f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                return os.path.join(root, f)
    return None


# ── Signal extraction ────────────────────────────────────────────────────────

def extract_signals(keypoints: list, fps: float):
    """
    From raw keypoints, produce three clean 1-D arrays:
      hip_y      – smoothed normalized hip displacement (0 = standing, + = deeper)
      velocity   – smoothed first derivative of hip_y (+ = going down)
      knee_angle – average knee angle per frame
    Also returns body_scale and standing_height for metadata.
    """
    n = len(keypoints)

    # ── Raw extraction ───────────────────────────────────────────────────
    raw_hip_y = np.full(n, np.nan)
    raw_knee  = np.full(n, np.nan)
    confs     = np.zeros(n)

    for i, kp in enumerate(keypoints):
        if kp is None or len(kp) <= max(L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE):
            continue
        hip_conf = (kp[L_HIP][3] + kp[R_HIP][3]) / 2
        confs[i] = hip_conf
        if hip_conf < MIN_CONFIDENCE:
            continue

        # Hip midpoint Y
        raw_hip_y[i] = (kp[L_HIP][1] + kp[R_HIP][1]) / 2

        # Knee angles (average of left & right)
        angles = []
        for (h, k, a) in [(L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)]:
            if min(kp[h][3], kp[k][3], kp[a][3]) >= MIN_CONFIDENCE:
                ang = _angle_at(np.array(kp[h][:3]), np.array(kp[k][:3]), np.array(kp[a][:3]))
                if not np.isnan(ang):
                    angles.append(ang)
        if angles:
            raw_knee[i] = np.mean(angles)

    _interp_nans(raw_hip_y)
    _interp_nans(raw_knee)

    # ── Calibration: standing height & body scale ────────────────────────
    cal = raw_hip_y[:CALIBRATION_N]
    standing_y = np.percentile(cal[~np.isnan(cal)], 25) if np.any(~np.isnan(cal)) else np.nanmin(raw_hip_y)

    # Body scale from first valid frame's torso+leg length
    body_scale = 0.25  # sensible default
    for kp in keypoints[:CALIBRATION_N]:
        if kp is None or len(kp) <= R_ANKLE:
            continue
        if (kp[L_SHOULDER][3] + kp[L_HIP][3] + kp[L_KNEE][3] + kp[L_ANKLE][3]) / 4 < MIN_CONFIDENCE:
            continue
        sh = (np.array(kp[L_SHOULDER][:3]) + np.array(kp[R_SHOULDER][:3])) / 2
        hp = (np.array(kp[L_HIP][:3]) + np.array(kp[R_HIP][:3])) / 2
        kn = (np.array(kp[L_KNEE][:3]) + np.array(kp[R_KNEE][:3])) / 2
        ak = (np.array(kp[L_ANKLE][:3]) + np.array(kp[R_ANKLE][:3])) / 2
        body_scale = (np.linalg.norm(sh - hp) + np.linalg.norm(hp - kn) + np.linalg.norm(kn - ak)) / 3
        break

    # ── Normalize hip displacement (0 = standing, positive = squatting) ──
    hip_y = np.maximum(0, raw_hip_y - standing_y) / max(body_scale, 1e-6)

    # ── Smooth ───────────────────────────────────────────────────────────
    win = min(SMOOTH_WINDOW, n // 2 * 2 + 1)
    if win >= 5:
        hip_y = savgol_filter(hip_y, win, min(SMOOTH_ORDER, win - 1))
        hip_y = np.maximum(0, hip_y)

    # ── Velocity (positive = descending) ─────────────────────────────────
    velocity = np.gradient(hip_y)
    velocity = uniform_filter1d(velocity, size=min(VEL_SMOOTH_SIZE, n), mode='nearest')

    return hip_y, velocity, raw_knee, confs, body_scale, standing_y


# ── Phase classification ─────────────────────────────────────────────────────

def classify_phases(hip_y: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """
    Per-frame phase assignment, then merge short segments.

    Rules (applied in order):
      1. velocity >  VEL_THRESH           → ECCENTRIC   (hips going down)
      2. velocity < -VEL_THRESH           → CONCENTRIC  (hips going up)
      3. hip_y ≥ DEPTH_FRAC_ISO * max_depth → ISOMETRIC (still, but deep)
      4. otherwise                         → IDLE        (still, at top)
    """
    n = len(hip_y)
    max_depth = np.max(hip_y) if n > 0 else 1.0
    depth_cutoff = DEPTH_FRAC_ISO * max_depth if max_depth > 0 else 0

    labels = np.zeros(n, dtype=int)  # default IDLE

    for i in range(n):
        v = velocity[i]
        if v > VEL_THRESH:
            labels[i] = Phase.ECCENTRIC.value
        elif v < -VEL_THRESH:
            labels[i] = Phase.CONCENTRIC.value
        elif hip_y[i] >= depth_cutoff:
            labels[i] = Phase.ISOMETRIC.value
        # else: IDLE (already 0)

    # ── Merge short segments into their neighbours ───────────────────────
    labels = _merge_short_phases(labels, MIN_PHASE_FRAMES)

    # Opening in concentric/isometric is usually a startup artifact or a clip
    # that begins mid-return. Normalize it to idle so the first real descent
    # can still form a valid rep.
    labels = _normalize_leading_phase(labels)
    labels = _merge_short_phases(labels, MIN_PHASE_FRAMES)

    # ── Enforce valid ordering via single forward pass ───────────────────
    # Valid: IDLE→ECC, ECC→ISO, ECC→CON, ISO→CON, CON→IDLE
    # If a transition is invalid, inherit previous label.
    VALID_NEXT = {
        Phase.IDLE.value:       {Phase.IDLE.value, Phase.ECCENTRIC.value},
        Phase.ECCENTRIC.value:  {Phase.ECCENTRIC.value, Phase.ISOMETRIC.value, Phase.CONCENTRIC.value},
        Phase.ISOMETRIC.value:  {Phase.ISOMETRIC.value, Phase.CONCENTRIC.value},
        Phase.CONCENTRIC.value: {Phase.CONCENTRIC.value, Phase.IDLE.value},
    }
    for i in range(1, n):
        if labels[i] not in VALID_NEXT.get(labels[i - 1], {Phase.IDLE.value}):
            labels[i] = labels[i - 1]

    # Merge once more after enforcement to clean up any new short fragments
    labels = _merge_short_phases(labels, MIN_PHASE_FRAMES)

    return labels


def _merge_short_phases(labels: np.ndarray, min_len: int) -> np.ndarray:
    """Replace segments shorter than min_len with their left neighbour."""
    n = len(labels)
    i = 0
    while i < n:
        j = i + 1
        while j < n and labels[j] == labels[i]:
            j += 1
        if (j - i) < min_len and i > 0:
            labels[i:j] = labels[i - 1]
        i = j
    return labels


def _normalize_leading_phase(labels: np.ndarray) -> np.ndarray:
    """Convert impossible opening phases into idle so rep detection can recover."""
    n = len(labels)
    i = 0
    while i < n:
        phase = int(labels[i])
        j = i + 1
        while j < n and labels[j] == phase:
            j += 1
        if phase == Phase.IDLE.value:
            i = j
            continue
        if phase in {Phase.CONCENTRIC.value, Phase.ISOMETRIC.value}:
            labels[i:j] = Phase.IDLE.value
            i = j
            continue
        break
    return labels


# ── Repetition detection ─────────────────────────────────────────────────────

@dataclass
class RepPhase:
    phase_type: str
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_seconds: float
    transition_reason: str = ""

    def to_dict(self):
        return {
            "phase_type": self.phase_type,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "duration_seconds": round(self.duration_seconds, 2),
            "transition_reason": self.transition_reason,
        }


@dataclass
class Repetition:
    rep_id: int
    start_frame: int
    end_frame: int
    phases: List[RepPhase]
    squat_depth_normalized: float
    squat_depth_angle: float
    bottom_frame: int
    bottom_knee_angle: float

    def to_dict(self):
        return {
            "rep_id": self.rep_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.end_frame - self.start_frame,
            "squat_depth_normalized": round(self.squat_depth_normalized, 3),
            "squat_depth_angle": round(self.squat_depth_angle, 2),
            "bottom_knee_angle": round(self.bottom_knee_angle, 2),
            "bottom_frame": self.bottom_frame,
            "phases": [p.to_dict() for p in self.phases],
        }


def detect_reps(labels: np.ndarray, hip_y: np.ndarray, knee_angles: np.ndarray,
                fps: float) -> List[Repetition]:
    """
    Walk through labels and collect reps as eccentric→[isometric]→concentric cycles.
    A rep ends when we return to IDLE or hit the next ECCENTRIC.
    """
    E, I, C, D = Phase.ECCENTRIC.value, Phase.ISOMETRIC.value, Phase.CONCENTRIC.value, Phase.IDLE.value
    reps = []
    n = len(labels)

    # Collect contiguous phase runs: [(phase_id, start, end), ...]
    runs = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and labels[j] == labels[i]:
            j += 1
        runs.append((int(labels[i]), i, j - 1))
        i = j

    # Scan runs for ecc→[iso]→con patterns
    ri = 0
    while ri < len(runs):
        pid, rs, re = runs[ri]
        if pid != E:
            ri += 1
            continue

        # Found eccentric – collect optional isometric + mandatory concentric
        phases_in_rep = [(pid, rs, re)]
        rj = ri + 1
        found_con = False

        while rj < len(runs):
            pj, rjs, rje = runs[rj]
            if pj == I:
                phases_in_rep.append((pj, rjs, rje))
                rj += 1
            elif pj == C:
                phases_in_rep.append((pj, rjs, rje))
                found_con = True
                rj += 1
                break
            else:
                break  # hit IDLE or another ECC without CON → abort this rep

        if not found_con:
            ri += 1
            continue

        rep_start = phases_in_rep[0][1]
        rep_end   = phases_in_rep[-1][2]

        if (rep_end - rep_start + 1) < MIN_REP_FRAMES:
            ri = rj
            continue

        # Bottom frame = max hip displacement in this rep
        seg = hip_y[rep_start:rep_end + 1]
        bottom_off = int(np.argmax(seg))
        bottom_frame = rep_start + bottom_off
        max_depth = float(seg[bottom_off])

        # Knee angle depth
        standing_angle = float(np.nanmedian(knee_angles[:CALIBRATION_N]))
        bottom_angle = float(knee_angles[bottom_frame])
        angle_depth = (standing_angle - bottom_angle) if not (np.isnan(standing_angle) or np.isnan(bottom_angle)) else 0.0

        # Build phase list
        phase_objs = []
        for (ph, ps, pe) in phases_in_rep:
            phase_objs.append(RepPhase(
                phase_type=PHASE_NAMES[ph],
                start_frame=int(ps),
                end_frame=int(pe),
                duration_frames=int(pe - ps + 1),
                duration_seconds=(pe - ps + 1) / fps,
            ))

        reps.append(Repetition(
            rep_id=len(reps) + 1,
            start_frame=int(rep_start),
            end_frame=int(rep_end),
            phases=phase_objs,
            squat_depth_normalized=max_depth,
            squat_depth_angle=angle_depth,
            bottom_frame=int(bottom_frame),
            bottom_knee_angle=bottom_angle if not np.isnan(bottom_angle) else 0.0,
        ))

        ri = rj  # advance past this rep

    return reps


# ── Main segmenter ───────────────────────────────────────────────────────────

def segment_video(keypoints_data: dict, video_id: str) -> dict:
    """Full pipeline: extract → classify → detect reps → package result."""
    keypoints = keypoints_data.get('keypoints_img', [])
    info = keypoints_data.get('info', {})
    fps = info.get('fps', 30.0)
    view = info.get('view', 'unknown')
    quality = info.get('quality_rating', 'Unknown')
    n = len(keypoints)

    try:
        hip_y, velocity, knee_angles, confs, body_scale, standing_y = extract_signals(keypoints, fps)
        labels = classify_phases(hip_y, velocity)
        reps = detect_reps(labels, hip_y, knee_angles, fps)

        phase_names = [PHASE_NAMES.get(int(l), "unknown") for l in labels]

        return _to_native({
            "video_id": video_id,
            "info": {
                "fps": fps, "frame_count": n, "quality_rating": quality,
                "view": view, "total_reps": len(reps),
                "calibration": {"body_scale": body_scale, "standing_hip_height": standing_y},
                "params": {"smooth_window": SMOOTH_WINDOW, "vel_thresh": VEL_THRESH,
                           "min_phase_frames": MIN_PHASE_FRAMES, "depth_frac_iso": DEPTH_FRAC_ISO},
            },
            "frame_phases": phase_names,
            "repetitions": [r.to_dict() for r in reps],
            "signals": {
                "normalized_hip_displacement": hip_y,
                "window_velocity": velocity,
                "knee_angles": knee_angles,
                "landmark_confidence": confs,
            },
        })
    except Exception as e:
        import traceback
        return {"video_id": video_id, "error": str(e), "traceback": traceback.format_exc()}


# ── Visualization ────────────────────────────────────────────────────────────

def create_visualization(video_id: str, seg: dict, quality: str) -> bool:
    """Overlay phase labels, rep info, and signals onto the source video."""
    video_path = find_video_file(video_id, quality)
    out_dir = VISUALIZATION_DIRS.get(quality.lower(), list(VISUALIZATION_DIRS.values())[0])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{video_id}_phases.mp4")

    if not video_path:
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not out.isOpened():
        out_path = out_path.replace('.mp4', '.avi')
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    if not out.isOpened():
        cap.release()
        return False

    phases = seg["frame_phases"]
    reps = seg["repetitions"]
    signals = seg.get("signals", {})
    hip_disp = signals.get("normalized_hip_displacement", [])
    vel = signals.get("window_velocity", [])
    knee = signals.get("knee_angles", [])
    info = seg.get("info", {})

    fi = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fi < len(phases):
                ph = phases[fi]
                col = PHASE_COLORS.get(ph, (255, 255, 255))

                # Phase bar
                cv2.rectangle(frame, (10, 10), (300, 60), (0, 0, 0), -1)
                cv2.rectangle(frame, (12, 12), (298, 58), col, -1)
                cv2.putText(frame, f"Phase: {ph.upper()}", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Current rep
                for r in reps:
                    if r["start_frame"] <= fi <= r["end_frame"]:
                        cv2.rectangle(frame, (w - 200, 10), (w - 10, 60), (0, 0, 0), -1)
                        cv2.putText(frame, f"Rep {r['rep_id']}", (w - 190, 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                        cv2.putText(frame, f"Depth: {r['squat_depth_normalized']:.2f}", (w - 190, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                        if fi == r["bottom_frame"]:
                            cv2.putText(frame, "* BOTTOM *", (w // 2 - 80, h - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        break

                # Signal panel
                py = h - 90
                cv2.rectangle(frame, (10, py), (290, h - 10), (0, 0, 0), -1)
                if fi < len(hip_disp):
                    cv2.putText(frame, f"Hip: {hip_disp[fi]:.3f}", (18, py + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 255, 150), 1)
                if fi < len(vel):
                    vc = (150, 150, 255) if vel[fi] > 0 else (255, 150, 150)
                    cv2.putText(frame, f"Vel: {vel[fi]:+.4f}", (18, py + 44),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, vc, 1)
                if fi < len(knee):
                    cv2.putText(frame, f"Knee: {knee[fi]:.1f} deg", (18, py + 66),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                cv2.putText(frame, f"{fi + 1}/{len(phases)}", (w - 120, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

            out.write(frame)
            fi += 1
    finally:
        cap.release()
        out.release()
    return True


# ── Batch runner ─────────────────────────────────────────────────────────────

def run_segmentation(quality_filter=None, create_vis=True):
    for d in list(OUTPUT_DIRS.values()) + (list(VISUALIZATION_DIRS.values()) if create_vis else []):
        os.makedirs(d, exist_ok=True)

    folders = FEATURES_DIRS if not quality_filter else [f for f in FEATURES_DIRS if quality_filter.lower() in f.lower()]

    json_files, quality_map = [], {}
    for folder in folders:
        if not os.path.exists(folder):
            continue
        quality = "excellent" if "excellent" in folder else "good" if "good" in folder else "fair"
        for f in os.listdir(folder):
            if not f.endswith(".json"):
                continue
            vid = os.path.splitext(f)[0]
            if VIDEO_IDS_TO_PROCESS != ["*"] and vid not in VIDEO_IDS_TO_PROCESS:
                continue
            path = os.path.join(folder, f)
            json_files.append(path)
            quality_map[path] = quality

    if not json_files:
        print("No videos found.")
        return

    print(f"\n{'='*60}")
    print(f"SQUAT TEMPORAL SEGMENTATION v6.0 (simplified)")
    print(f"{'='*60}")
    print(f"Processing {len(json_files)} videos | viz={create_vis}")
    print(f"Params: smooth={SMOOTH_WINDOW}, vel_thresh={VEL_THRESH}, min_phase={MIN_PHASE_FRAMES}\n")

    stats = {"ok": 0, "err": 0, "reps": 0, "viz": 0}

    for jp in tqdm(json_files, desc="Segmenting"):
        vid = os.path.splitext(os.path.basename(jp))[0]
        try:
            with open(jp) as f:
                data = json.load(f)
            result = segment_video(data, vid)
        except Exception as e:
            print(f"\n  ✗ {vid}: {e}")
            stats["err"] += 1
            continue

        if "error" in result:
            print(f"\n  ✗ {vid}: {result['error']}")
            stats["err"] += 1
            continue

        stats["ok"] += 1
        q = result.get("info", {}).get("quality_rating", "unknown").lower()
        if q not in OUTPUT_DIRS:
            q = quality_map.get(jp, "fair")
        stats["reps"] += result["info"]["total_reps"]

        out_dir = OUTPUT_DIRS.get(q, list(OUTPUT_DIRS.values())[0])
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{vid}_segmented.json"), 'w') as f:
            json.dump(result, f, indent=2)

        if create_vis and create_visualization(vid, result, q):
            stats["viz"] += 1

    print(f"\n{'='*60}")
    print(f"Done: {stats['ok']} ok, {stats['err']} err, {stats['reps']} reps, {stats['viz']} viz")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_segmentation(quality_filter=None, create_vis=True)