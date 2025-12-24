import os
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

# --- Configuration ---
FEATURES_ROOT = "./squat/view_classifier/extracted_features"
OUTPUT_MODEL = "./squat/view_classifier/view_classifier_v5_ensemble.pkl"
OUTPUT_ENCODER = "./squat/view_classifier/label_encoder_v5.pkl"
OUTPUT_SCALER = "./squat/view_classifier/scaler_v5.pkl"
OUTPUT_CSV = "./squat/view_classifier/training_features_v5.csv"

CLASS_NAME_MAPPING = {
    "back_side": "side_back",
    "front_side": "side_front"
}

# --- 1. IMPROVED Feature Logic ---
def calculate_frame_features(lm_norm, lm_world):
    """
    Extracts 16 raw features per frame (EXPANDED from 8).
    """
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

    # A. Normalization Scale (Torso Height)
    mid_sh_y = (lm_norm[L_SHOULDER][1] + lm_norm[R_SHOULDER][1]) / 2
    mid_hip_y = (lm_norm[L_HIP][1] + lm_norm[R_HIP][1]) / 2
    torso_h = abs(mid_sh_y - mid_hip_y)
    if torso_h < 0.01: torso_h = 1.0

    # B. Geometric Ratios (ORIGINAL)
    sh_width = abs(lm_norm[L_SHOULDER][0] - lm_norm[R_SHOULDER][0]) / torso_h
    hip_width = abs(lm_norm[L_HIP][0] - lm_norm[R_HIP][0]) / torso_h
    
    # C. Aspect Ratio
    xs = [lm_norm[i][0] for i in [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE]]
    ys = [lm_norm[i][1] for i in [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE]]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0

    # D. Symmetry Scores (ORIGINAL)
    sh_sym = abs(lm_norm[L_SHOULDER][3] - lm_norm[R_SHOULDER][3])
    hip_sym = abs(lm_norm[L_HIP][3] - lm_norm[R_HIP][3])
    knee_sym = abs(lm_norm[L_KNEE][3] - lm_norm[R_KNEE][3])
    total_symmetry = sh_sym + hip_sym + knee_sym

    # E. 3D Depths (ORIGINAL)
    sh_rot_z = abs(lm_world[L_SHOULDER][2] - lm_world[R_SHOULDER][2])
    avg_hip_z = (lm_world[L_HIP][2] + lm_world[R_HIP][2]) / 2
    nose_rel_z = lm_world[NOSE][2] - avg_hip_z
    avg_heel_z = (lm_world[L_HEEL][2] + lm_world[R_HEEL][2]) / 2
    avg_toe_z = (lm_world[L_TOE][2] + lm_world[R_TOE][2]) / 2
    heel_toe_diff = avg_heel_z - avg_toe_z
    min_sh_vis = min(lm_norm[L_SHOULDER][3], lm_norm[R_SHOULDER][3])

    # ===== NEW FEATURES (8 additional) =====
    
    # F. Elbow/Wrist Visibility Asymmetry (arms visible = side view)
    elbow_sym = abs(lm_norm[L_ELBOW][3] - lm_norm[R_ELBOW][3])
    wrist_sym = abs(lm_norm[L_WRIST][3] - lm_norm[R_WRIST][3])
    
    # G. Knee Width Ratio (side views have overlapping knees)
    knee_width = abs(lm_norm[L_KNEE][0] - lm_norm[R_KNEE][0]) / torso_h
    
    # H. Ankle Width Ratio
    ankle_width = abs(lm_norm[L_ANKLE][0] - lm_norm[R_ANKLE][0]) / torso_h
    
    # I. Hip Rotation (World Z-axis)
    hip_rot_z = abs(lm_world[L_HIP][2] - lm_world[R_HIP][2])
    
    # J. Eye Visibility (front = both visible, back = neither, side = one)
    eye_vis_sum = lm_norm[L_EYE][3] + lm_norm[R_EYE][3]
    eye_vis_diff = abs(lm_norm[L_EYE][3] - lm_norm[R_EYE][3])
    
    # K. Torso Lean (X-offset between shoulders and hips)
    mid_sh_x = (lm_norm[L_SHOULDER][0] + lm_norm[R_SHOULDER][0]) / 2
    mid_hip_x = (lm_norm[L_HIP][0] + lm_norm[R_HIP][0]) / 2
    torso_lean = (mid_sh_x - mid_hip_x) / torso_h
    
    # L. Average Body Visibility (quality indicator)
    avg_visibility = np.mean([
        lm_norm[L_SHOULDER][3], lm_norm[R_SHOULDER][3],
        lm_norm[L_HIP][3], lm_norm[R_HIP][3],
        lm_norm[L_KNEE][3], lm_norm[R_KNEE][3]
    ])
    
    return [
        # Original 8
        sh_width, hip_width, aspect_ratio, total_symmetry,
        sh_rot_z, nose_rel_z, heel_toe_diff, min_sh_vis,
        # New 8
        elbow_sym, wrist_sym, knee_width, ankle_width,
        hip_rot_z, eye_vis_sum, eye_vis_diff, torso_lean
    ]

def process_json_file(json_path):
    """
    Extracts TEMPORAL Statistics (Mean, Std, Min, Max) over the video.
    Returns flattened feature vector.
    """
    try:
        with open(json_path, 'r') as f: 
            data = json.load(f)
        kpts_img = data.get("keypoints_img", [])
        kpts_world = data.get("keypoints_world", [])
        if not kpts_img or not kpts_world: 
            return None

        feats_time_series = []
        # Sample more frames (first 3 seconds)
        frames_to_check = min(len(kpts_img), 90)

        for i in range(frames_to_check):
            if kpts_img[i][0][0] == 0.0: 
                continue
            
            f_feats = calculate_frame_features(kpts_img[i], kpts_world[i])
            feats_time_series.append(f_feats)
        
        if len(feats_time_series) < 5:  # Need minimum frames
            return None
        
        # --- IMPROVED TEMPORAL AGGREGATION ---
        arr = np.array(feats_time_series)
        
        mean_feats = np.mean(arr, axis=0)
        std_feats = np.std(arr, axis=0)
        min_feats = np.min(arr, axis=0)
        max_feats = np.max(arr, axis=0)
        
        # Percentiles for robustness
        p25_feats = np.percentile(arr, 25, axis=0)
        p75_feats = np.percentile(arr, 75, axis=0)
        
        # Result: 16 base features × 6 stats = 96 features
        return np.concatenate([mean_feats, std_feats, min_feats, max_feats, p25_feats, p75_feats])

    except Exception as e:
        return None

# --- 2. Improved Augmentation ---
def augment_data(X, y, samples=3):
    """More sophisticated augmentation"""
    print(f"--- Augmenting Data (x{samples+1}) ---")
    X_aug, y_aug = list(X), list(y)
    
    for i in range(len(X)):
        for s in range(samples):
            # Varying noise levels
            noise_scale = 0.01 + (s * 0.01)  # 0.01, 0.02, 0.03
            noise = np.random.normal(0, noise_scale, X[i].shape)
            X_aug.append(X[i] + noise)
            y_aug.append(y[i])
            
    return np.array(X_aug), np.array(y_aug)

# --- 3. Main Pipeline ---
def main():
    print("=" * 60)
    print("VIEW CLASSIFIER TRAINING v5 (Enhanced Features)")
    print("=" * 60)
    
    print("\n--- 1. Loading & Processing Data ---")
    data_rows, labels = [], []
    
    classes = [d for d in os.listdir(FEATURES_ROOT) if os.path.isdir(os.path.join(FEATURES_ROOT, d))]
    print(f"Found classes: {classes}")
    
    for folder_name in classes:
        std_label = CLASS_NAME_MAPPING.get(folder_name, folder_name)
        class_dir = os.path.join(FEATURES_ROOT, folder_name)
        files = [f for f in os.listdir(class_dir) if f.endswith(".json")]
        
        for json_file in tqdm(files, desc=f"{folder_name} -> {std_label}"):
            feat = process_json_file(os.path.join(class_dir, json_file))
            if feat is not None:
                data_rows.append(feat)
                labels.append(std_label)

    # Define Column Names for CSV
    base_cols = [
        "sh_width", "hip_width", "aspect", "symmetry",
        "sh_rot", "nose_z", "heel_diff", "min_sh_vis",
        "elbow_sym", "wrist_sym", "knee_width", "ankle_width",
        "hip_rot", "eye_vis_sum", "eye_vis_diff", "torso_lean"
    ]
    stats = ["mean", "std", "min", "max", "p25", "p75"]
    cols = [f"{c}_{s}" for s in stats for c in base_cols]
    
    df = pd.DataFrame(data_rows, columns=cols)
    df['target'] = labels
    df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Class distribution:\n{df['target'].value_counts()}")

    # --- 4. Training ---
    print("\n--- 2. Building Enhanced Ensemble ---")
    X = df.drop(columns=['target']).values
    y = df['target'].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, OUTPUT_ENCODER)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, OUTPUT_SCALER)
    
    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Augment training data
    X_train_aug, y_train_aug = augment_data(X_train, y_train, samples=4)
    print(f"After augmentation: {len(X_train_aug)} samples")

    # --- IMPROVED MODELS ---
    
    # 1. XGBoost with better hyperparameters
    clf_xgb = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=2,
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # 2. Random Forest
    clf_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # 3. SVM with RBF kernel
    clf_svc = SVC(
        kernel='rbf',
        probability=True,
        C=10.0,
        gamma='scale',
        random_state=42
    )
    
    # 4. Gradient Boosting (additional model)
    clf_gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    
    # Ensemble with 4 models
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', clf_xgb),
            ('rf', clf_rf),
            ('svc', clf_svc),
            ('gb', clf_gb)
        ],
        voting='soft',
        weights=[3, 2, 1, 2]  # XGBoost gets highest weight
    )
    
    print("\nTraining Ensemble (XGB + RF + SVC + GB)...")
    ensemble.fit(X_train_aug, y_train_aug)
    
    # --- 5. Evaluation ---
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n🎯 TEST ACCURACY: {acc*100:.2f}%")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # --- Individual Model Performance ---
    print("\n--- Individual Model Accuracies ---")
    for name, model in ensemble.named_estimators_.items():
        pred = model.predict(X_test)
        ind_acc = accuracy_score(y_test, pred)
        print(f"  {name.upper()}: {ind_acc*100:.2f}%")
    
    joblib.dump(ensemble, OUTPUT_MODEL)
    print(f"\n✅ Ensemble Saved: {OUTPUT_MODEL}")
    print(f"✅ Encoder Saved: {OUTPUT_ENCODER}")
    print(f"✅ Scaler Saved: {OUTPUT_SCALER}")

if __name__ == "__main__":
    main()