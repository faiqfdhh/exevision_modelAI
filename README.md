# ExeVision AI - Project Structure

## 📁 Directory Organization

```
exevision_modelAI/
├── src/                              # ⭐ Production/Deployment Code
│   ├── __init__.py                   # Package initialization
│   ├── main.py                       # Main pipeline orchestrator
│   ├── config.py                     # Configuration & data classes
│   ├── pose_extractor.py             # Pose extraction module
│   ├── view_classifier.py            # View classification module
│   └── temporal_segmenter.py         # Temporal segmentation module
│
├── scripts/                          # 🔬 Development/Experimental Scripts
│   ├── 1_organize_data.py            # Data organization
│   ├── 2_extract_features.py         # Original extraction script
│   ├── 2.5_extract_selected_features.py
│   ├── 3_visualize_poses.py          # Pose visualization
│   ├── 4_classify_views              # Original view classification
│   ├── 5_temporal_segmentation.py    # Original segmentation script
│   ├── 5.5_visualize_segmentation.py
│   └── 5.6_plot_signals.py           # Signal plotting
│
├── squat/                            # Data folder
│   ├── dataset_videos_all/           # Source videos
│   ├── extracted_features/           # Extracted pose features
│   ├── extracted_features1/pose/     # Alternative features
│   ├── segmented_reps/               # Segmentation results
│   ├── view_classifier/              # Trained view classifier
│   └── visualized_poses/             # Visualization outputs
│
├── models/                           # MediaPipe models
│   └── pose_landmarker_heavy.task
│
├── progress.md                       # Project progress tracker
└── README.md                         # This file
```

## 🚀 Usage

### Running the Pipeline
```bash
# Make sure you're in the project root
cd "exevision_modelAI"

# Run main pipeline on a video
python -m src.main path/to/video.mp4
```

### Minimal UI Runner (New)
```bash
# Launch simple desktop UI for script-based pipeline execution
python pipeline_ui/app.py
```

This UI lets you run full pipeline, individual stages, or custom stage selections.
Each run is isolated under `pipeline_ui_runs/<run_name>/` and stores per-stage snapshots.

### Importing in Python
```python
# Import from production code
from src import PipelineConfig, PoseExtractor, ViewClassifier, TemporalSegmenter

# Create pipeline
config = PipelineConfig()
extractor = PoseExtractor(config)
result = extractor.extract("path/to/video.mp4")
```

### Development Scripts
The `scripts/` folder contains original experimental and utility scripts:
- Use for data exploration and debugging
- Reference implementation details
- Testing specific stages

## 📋 Module Responsibilities

| Module | Purpose |
|--------|---------|
| `config.py` | Configuration settings & data models |
| `pose_extractor.py` | MediaPipe pose landmark extraction |
| `view_classifier.py` | Camera view classification (front/back/side) |
| `temporal_segmenter.py` | Squat phase & repetition detection |
| `main.py` | Orchestrates all stages into a pipeline |

## 🔄 Pipeline Flow

```
Video → Pose Extraction → View Classification → Temporal Segmentation → Output
                (1)              (2)                      (3)
```

1. **Stage 1**: Extract pose landmarks from video frames
2. **Stage 2**: Classify camera view (front/back/side)
3. **Stage 3**: Segment motion into phases and repetitions
