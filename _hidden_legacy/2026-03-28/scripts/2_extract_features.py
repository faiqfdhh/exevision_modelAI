import os
import json
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- Suppress MediaPipe CPU Warnings ---
os.environ['GLOG_minloglevel'] = '2'

# --- Configuration ---
DATASET_ROOT = "./squat/dataset_videos_all"
OUTPUT_ROOT = "./squat/extracted_features"
MODEL_PATH = os.path.join('models', 'pose_landmarker_heavy.task')

def get_mediapipe_options():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    return vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )

def process_single_video(vid_path):
    try:
        # Check if output file already exists
        rel_path = os.path.relpath(os.path.dirname(vid_path), DATASET_ROOT)
        output_dir = os.path.join(OUTPUT_ROOT, rel_path)
        vid_id = os.path.splitext(os.path.basename(vid_path))[0]
        save_path = os.path.join(output_dir, f"{vid_id}.json")
        
        if os.path.exists(save_path):
            return vid_path, "Skipped", None
        
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            return vid_path, "Failed to open video", None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0

        data_img_space = []
        data_world_space = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        with vision.PoseLandmarker.create_from_options(get_mediapipe_options()) as landmarker:
            for _ in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                if detection_result.pose_landmarks:
                    frame_img = [[lm.x, lm.y, lm.z, lm.visibility] 
                                for lm in detection_result.pose_landmarks[0]]
                    frame_world = [[lm.x, lm.y, lm.z, lm.visibility] 
                                  for lm in detection_result.pose_world_landmarks[0]]
                else:
                    frame_img = [[0.0, 0.0, 0.0, 0.0] for _ in range(33)]
                    frame_world = [[0.0, 0.0, 0.0, 0.0] for _ in range(33)]
                data_img_space.append(frame_img)
                data_world_space.append(frame_world)

        cap.release()
        
        # Output path already computed at the start
        os.makedirs(output_dir, exist_ok=True)
        
        landmarks_img_np = np.array(data_img_space, dtype=np.float32)
        landmarks_world_np = np.array(data_world_space, dtype=np.float32)
        with open(save_path, 'w') as f:
            json.dump({
                "info": {
                    "fps": fps, 
                    "frame_count": len(data_img_space),
                    "model": "PoseLandmarker_FULL",
                    "processed_on": "CPU_MultiCore",
                    "view_folder": rel_path  # Track which folder it came from
                },
                "keypoints_img": landmarks_img_np.tolist(),
                "keypoints_world": landmarks_world_np.tolist()
            }, f)
        return vid_path, "Success", None
    except Exception as e:
        return vid_path, "Error", str(e)

def run_extraction():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file '{MODEL_PATH}' not found. Download from:")
        print("https://developers.google.com/mediapipe/solutions/vision/pose_landmarker")
        return

    # Collect all videos in dataset_videos_all (recursively)
    video_files = []
    for root, dirs, files in os.walk(DATASET_ROOT):
        for file in files:
            if file.lower().endswith((".mp4", ".mov", ".avi")):
                video_files.append(os.path.join(root, file))
    
    print(f"Found {len(video_files)} videos.")
    print(f"Using {cpu_count()} CPU cores...")

    with Pool(cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(process_single_video, video_files),
            total=len(video_files),
            desc="Processing videos"
        ))
    
    errors = [r for r in results if r[1] == "Error"]
    skipped = [r for r in results if r[1] == "Skipped"]
    
    print(f"\n✓ Processing complete:")
    print(f"  - Successfully processed: {len([r for r in results if r[1] == 'Success'])}")
    print(f"  - Skipped (already exists): {len(skipped)}")
    print(f"  - Failed: {len(errors)}")
    
    if errors:
        print(f"\n⚠️  Failed videos:")
        for vid, status, error in errors[:5]:
            print(f"  - {os.path.basename(vid)}: {error}")

if __name__ == "__main__":
    run_extraction()