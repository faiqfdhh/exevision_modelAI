import os
import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Configuration ---
# Point this to the video you want to test
TEST_VIDEO_PATH = r"squat\dataset_videos_all\50801_2.mp4" 

# Model Paths
MODEL_PATH = "./squat/view_classifier/view_classifier_v5_ensemble.pkl"
ENCODER_PATH = "./squat/view_classifier/label_encoder_v5.pkl"
SCALER_PATH = "./squat/view_classifier/scaler_v5.pkl"
MP_MODEL_ASSET = "models/pose_landmarker_heavy.task" # Ensure this path is correct

# --- 1. UPDATED Feature Logic (V5 with 16 features) ---
def calculate_frame_features(lm_norm, lm_world):
    """Extract 16 features per frame (matching v5 training)"""
    # Indices
    NOSE = 0
    L_EYE, R_EYE = 2, 5
    L_SHOULDER, R_SHOULDER = 11, 12
    L_ELBOW, R_ELBOW = 13, 14
    L_WRIST, R_WRIST = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28
    L_HEEL, R_HEEL = 29, 30
    L_TOE, R_TOE = 31, 32

    # A. Normalization (Torso Height)
    mid_sh_y = (lm_norm[L_SHOULDER].y + lm_norm[R_SHOULDER].y) / 2
    mid_hip_y = (lm_norm[L_HIP].y + lm_norm[R_HIP].y) / 2
    torso_h = abs(mid_sh_y - mid_hip_y)
    if torso_h < 0.01: torso_h = 1.0

    # B. Geometric Ratios
    sh_width = abs(lm_norm[L_SHOULDER].x - lm_norm[R_SHOULDER].x) / torso_h
    hip_width = abs(lm_norm[L_HIP].x - lm_norm[R_HIP].x) / torso_h
    
    # C. Aspect Ratio
    xs = [lm_norm[i].x for i in [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE]]
    ys = [lm_norm[i].y for i in [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE]]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0

    # D. Symmetry (Visibility)
    sh_sym = abs(lm_norm[L_SHOULDER].visibility - lm_norm[R_SHOULDER].visibility)
    hip_sym = abs(lm_norm[L_HIP].visibility - lm_norm[R_HIP].visibility)
    knee_sym = abs(lm_norm[L_KNEE].visibility - lm_norm[R_KNEE].visibility)
    total_symmetry = sh_sym + hip_sym + knee_sym

    # E. 3D Depths (World Landmarks)
    sh_rot_z = abs(lm_world[L_SHOULDER].z - lm_world[R_SHOULDER].z)
    avg_hip_z = (lm_world[L_HIP].z + lm_world[R_HIP].z) / 2
    nose_rel_z = lm_world[NOSE].z - avg_hip_z
    
    avg_heel_z = (lm_world[L_HEEL].z + lm_world[R_HEEL].z) / 2
    avg_toe_z = (lm_world[L_TOE].z + lm_world[R_TOE].z) / 2
    heel_toe_diff = avg_heel_z - avg_toe_z
    min_sh_vis = min(lm_norm[L_SHOULDER].visibility, lm_norm[R_SHOULDER].visibility)

    # === NEW FEATURES (matching v5 training) ===
    
    # F. Elbow/Wrist Visibility Asymmetry
    elbow_sym = abs(lm_norm[L_ELBOW].visibility - lm_norm[R_ELBOW].visibility)
    wrist_sym = abs(lm_norm[L_WRIST].visibility - lm_norm[R_WRIST].visibility)
    
    # G. Knee Width Ratio
    knee_width = abs(lm_norm[L_KNEE].x - lm_norm[R_KNEE].x) / torso_h
    
    # H. Ankle Width Ratio
    ankle_width = abs(lm_norm[L_ANKLE].x - lm_norm[R_ANKLE].x) / torso_h
    
    # I. Hip Rotation (World Z-axis)
    hip_rot_z = abs(lm_world[L_HIP].z - lm_world[R_HIP].z)
    
    # J. Eye Visibility
    eye_vis_sum = lm_norm[L_EYE].visibility + lm_norm[R_EYE].visibility
    eye_vis_diff = abs(lm_norm[L_EYE].visibility - lm_norm[R_EYE].visibility)
    
    # K. Torso Lean
    mid_sh_x = (lm_norm[L_SHOULDER].x + lm_norm[R_SHOULDER].x) / 2
    mid_hip_x = (lm_norm[L_HIP].x + lm_norm[R_HIP].x) / 2
    torso_lean = (mid_sh_x - mid_hip_x) / torso_h
    
    return [
        # Original 8
        sh_width, hip_width, aspect_ratio, total_symmetry,
        sh_rot_z, nose_rel_z, heel_toe_diff, min_sh_vis,
        # New 8
        elbow_sym, wrist_sym, knee_width, ankle_width,
        hip_rot_z, eye_vis_sum, eye_vis_diff, torso_lean
    ]

# --- 2. Processing Pipeline ---
def analyze_video(video_path):
    print(f"--- Processing: {os.path.basename(video_path)} ---")
    
    # Load Models
    if not os.path.exists(MODEL_PATH):
        print("Error: Model files not found. Check paths.")
        return

    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Init MediaPipe
    base_options = python.BaseOptions(model_asset_path=MP_MODEL_ASSET)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # Read Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    feats_time_series = []
    frame_count = 0
    max_frames = 90  # First 3 seconds (matching training)

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break

        # Convert for MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        result = landmarker.detect_for_video(mp_image, timestamp)
        
        if result.pose_landmarks:
            lm_norm = result.pose_landmarks[0]
            lm_world = result.pose_world_landmarks[0]
            
            # Extract 16 features
            f = calculate_frame_features(lm_norm, lm_world)
            feats_time_series.append(f)
        
        frame_count += 1

    cap.release()
    landmarker.close()

    if not feats_time_series:
        print("No person detected in video.")
        return

    # Aggregate with 6 stats (matching v5 training)
    arr = np.array(feats_time_series)
    
    mean_feats = np.mean(arr, axis=0)
    std_feats = np.std(arr, axis=0)
    min_feats = np.min(arr, axis=0)
    max_feats = np.max(arr, axis=0)
    p25_feats = np.percentile(arr, 25, axis=0)
    p75_feats = np.percentile(arr, 75, axis=0)
    
    # Result: 16 features × 6 stats = 96 features
    final_features = np.concatenate([mean_feats, std_feats, min_feats, max_feats, p25_feats, p75_feats])
    
    # Scale
    features_scaled = scaler.transform([final_features])

    # Predict
    probs = model.predict_proba(features_scaled)[0]
    pred_idx = np.argmax(probs)
    label = le.inverse_transform([pred_idx])[0]
    confidence = probs[pred_idx]

    print("\n" + "="*40)
    print(f"🎯 PREDICTED VIEW:  {label.upper()}")
    print(f"   CONFIDENCE:      {confidence*100:.2f}%")
    print("="*40)
    
    print("\nClass Probabilities:")
    for cls, prob in zip(le.classes_, probs):
        bar = "█" * int(prob * 20)
        print(f"  {cls:12s} {prob*100:5.1f}% {bar}")

if __name__ == "__main__":
    analyze_video(TEST_VIDEO_PATH)