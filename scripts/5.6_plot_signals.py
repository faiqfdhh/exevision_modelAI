"""
Plot segmentation signals for parameter tuning
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

SEGMENTED_DIR = "./squat/segmented_reps"
PLOTS_DIR = "./squat/segmentation_plots"


def plot_video_signals(video_id: str):
    """Create diagnostic plot for a video"""
    
    segment_path = os.path.join(SEGMENTED_DIR, f"{video_id}_segmented.json")
    
    if not os.path.exists(segment_path):
        print(f"Not found: {segment_path}")
        return
    
    with open(segment_path, 'r') as f:
        data = json.load(f)
    
    hip_y = np.array(data["signals"]["hip_y"])
    velocity = np.array(data["signals"]["velocity"])
    frame_phases = data["frame_phases"]
    reps = data["repetitions"]
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    frames = np.arange(len(hip_y))
    
    # Plot 1: Hip Y position
    ax1 = axes[0]
    ax1.plot(frames, hip_y, 'b-', linewidth=1, label='Hip Y')
    ax1.set_ylabel('Hip Y Position')
    ax1.set_title(f'Temporal Segmentation: {video_id}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mark rep bottoms
    for rep in reps:
        ax1.axvline(rep["bottom_frame"], color='red', linestyle='--', alpha=0.7)
        ax1.annotate(f'Rep {rep["rep_id"]}', 
                    (rep["bottom_frame"], hip_y[rep["bottom_frame"]]),
                    textcoords="offset points", xytext=(5, 10))
    
    # Plot 2: Velocity
    ax2 = axes[1]
    ax2.plot(frames, velocity, 'g-', linewidth=1, label='Velocity')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.axhline(0.002, color='gray', linestyle=':', label='Idle threshold')
    ax2.axhline(-0.002, color='gray', linestyle=':')
    ax2.set_ylabel('Velocity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Phase labels
    ax3 = axes[2]
    phase_map = {"idle": 0, "eccentric": 1, "concentric": 2}
    phase_values = [phase_map[p] for p in frame_phases]
    ax3.fill_between(frames, phase_values, alpha=0.5)
    ax3.set_ylabel('Phase')
    ax3.set_xlabel('Frame')
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['Idle', 'Eccentric', 'Concentric'])
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, f"{video_id}_signals.png"), dpi=150)
    plt.close()
    print(f"Saved plot: {video_id}_signals.png")


if __name__ == "__main__":
    # Plot specific videos for tuning
    test_videos = ["25708_2", "25734_3", "25737_1"]
    for vid in test_videos:
        plot_video_signals(vid)
