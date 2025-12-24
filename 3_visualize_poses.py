import cv2
import json
import numpy as np
import os
from tqdm import tqdm


NAME = "25709_1"

VIDEO_PATH = os.path.join("squat", "dataset_videos_all", f"{NAME}.mp4")
JSON_PATH = os.path.join("squat", "extracted_features", f"{NAME}.json")
OUTPUT_PATH = os.path.join("squat", "visualized_poses", f"{NAME}_annotated.mp4")

# MediaPipe 33 landmarks connections
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),  # Face
    (0, 4), (4, 5), (5, 6), (6, 8),  # Face
    (9, 10),  # Mouth
    (11, 12),  # Shoulders
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),  # Left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),  # Right arm
    (11, 23), (12, 24), (23, 24),  # Torso
    (23, 25), (25, 27), (27, 29), (27, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (28, 32),  # Right leg
]

# Color coding by visibility
def get_color(visibility):
    """Green = high confidence, Red = low confidence"""
    if visibility > 0.7:
        return (0, 255, 0)  # Green
    elif visibility > 0.4:
        return (0, 255, 255)  # Yellow
    else:
        return (0, 0, 255)  # Red

def draw_landmarks(frame, landmarks, h, w):
    """Draw pose landmarks and connections on frame"""
    # Draw connections first (so they appear behind points)
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start_lm = landmarks[start_idx]
        end_lm = landmarks[end_idx]
        
        # Only draw if both points are visible
        if start_lm[3] > 0.3 and end_lm[3] > 0.3:
            start_point = (int(start_lm[0] * w), int(start_lm[1] * h))
            end_point = (int(end_lm[0] * w), int(end_lm[1] * h))
            cv2.line(frame, start_point, end_point, (255, 255, 255), 2)
    
    # Draw landmark points
    for idx, lm in enumerate(landmarks):
        x, y, z, visibility = lm
        
        if visibility > 0.3:  # Only draw visible landmarks
            point = (int(x * w), int(y * h))
            color = get_color(visibility)
            cv2.circle(frame, point, 4, color, -1)
            cv2.circle(frame, point, 5, (255, 255, 255), 1)  # White outline
    
    return frame

def create_annotated_video():
    """Generate video with pose landmarks overlay"""
    
    # Load JSON data
    print(f"Loading keypoints from: {JSON_PATH}")
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    keypoints = data['keypoints_img']
    fps = data['info']['fps']
    frame_count = data['info']['frame_count']
    
    # Open video
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print("❌ Error: Could not open video file")
        return
    
    # Get video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {w}x{h} @ {actual_fps:.2f} FPS")
    print(f"Keypoints: {len(keypoints)} frames")
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))
    
    print(f"Creating annotated video...")
    
    # Process each frame
    frame_idx = 0
    with tqdm(total=min(frame_count, len(keypoints))) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= len(keypoints):
                break
            
            # Get landmarks for current frame
            landmarks = keypoints[frame_idx]
            
            # Calculate average visibility
            avg_visibility = np.mean([lm[3] for lm in landmarks])
            
            # Draw landmarks
            annotated_frame = draw_landmarks(frame.copy(), landmarks, h, w)
            
            # Add info overlay
            info_text = f"Frame: {frame_idx+1}/{frame_count} | Avg Visibility: {avg_visibility:.2f}"
            cv2.putText(annotated_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Color legend
            cv2.putText(annotated_frame, "Green: High Conf | Yellow: Med | Red: Low", 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Write frame
            out.write(annotated_frame)
            
            frame_idx += 1
            pbar.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"✅ Annotated video saved to: {OUTPUT_PATH}")
    print(f"📊 Processed {frame_idx} frames")

if __name__ == "__main__":
    create_annotated_video()