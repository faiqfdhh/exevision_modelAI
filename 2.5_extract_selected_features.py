import os
import json
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

# --- Suppress MediaPipe CPU Warnings ---
os.environ['GLOG_minloglevel'] = '2'

# --- Configuration ---
DATASET_ROOT = "./squat/dataset_videos_all"
OUTPUT_ROOT = "./squat/extracted_features1/pose"
MODEL_PATH = os.path.join('models', 'pose_landmarker_heavy.task')

# List of video IDs to process (without extension)
VIDEO_IDS = [
    "25708_2",
    "25737_1",
    "25760_7",
    "25734_3",
    "25742_3"
]

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

def find_video_path(video_id):
    # Search recursively for the video file with the given ID and common extensions
    for root, dirs, files in os.walk(DATASET_ROOT):
        for ext in (".mp4", ".mov", ".avi"):
            filename = f"{video_id}{ext}"
            if filename in files:
                return os.path.join(root, filename)
    return None

def process_single_video(vid_path):
    try:
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
        
        # Save as: extracted_features1/pose/{video_id}.json
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        vid_id = os.path.splitext(os.path.basename(vid_path))[0]
        save_path = os.path.join(OUTPUT_ROOT, f"{vid_id}.json")
        landmarks_img_np = np.array(data_img_space, dtype=np.float32)
        landmarks_world_np = np.array(data_world_space, dtype=np.float32)
        with open(save_path, 'w') as f:
            json.dump({
                "info": {
                    "fps": fps, 
                    "frame_count": len(data_img_space),
                    "model": "PoseLandmarker_FULL",
                    "processed_on": "CPU"
                },
                "keypoints_img": landmarks_img_np.tolist(),
                "keypoints_world": landmarks_world_np.tolist()
            }, f)
        return vid_id, "Success", None
    except Exception as e:
        return vid_path, "Error", str(e)

def run_extraction():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file '{MODEL_PATH}' not found. Download from:")
        print("https://developers.google.com/mediapipe/solutions/vision/pose_landmarker")
        return

    # Find paths for the selected videos
    video_paths = []
    for vid_id in VIDEO_IDS:
        path = find_video_path(vid_id)
        if path:
            video_paths.append(path)
        else:
            print(f"⚠️  Video file for ID '{vid_id}' not found.")

    print(f"Processing {len(video_paths)} videos...")
    results = []
    for vid_path in tqdm(video_paths, desc="Processing selected videos"):
        results.append(process_single_video(vid_path))
    
    errors = [r for r in results if r[1] == "Error"]
    if errors:
        print(f"\n⚠️  {len(errors)} videos failed:")
        for vid, status, error in errors:
            print(f"  - {os.path.basename(vid)}: {error}")

if __name__ == "__main__":
    run_extraction()