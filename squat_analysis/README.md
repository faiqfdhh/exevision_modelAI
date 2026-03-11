# Squat Analysis - Rule-Based AQA Framework

Automated Quality Assessment for squat form analysis, modeled after the diving AQA framework.

## Architecture

```
squat_analysis/
├── aqa_metaProgram_squat.py     # Main orchestration script
├── scoring_functions.py          # Quality scoring & percentile calculation
├── microprograms/
│   ├── squat_error_functions.py      # Form error detection functions
│   └── squat_recognition_functions.py # View classification, depth detection
└── README.md
```

## Pipeline Integration

This module integrates with your existing pipeline:

```
[2.5_extract_selected_features.py] → JSON with keypoints
        ↓
[4_classify_views.py] → View classification (optional - integrated here too)
        ↓
[5_temporal_segmentation.py] → Phase labels & repetitions
        ↓
[aqa_metaProgram_squat.py] → Error analysis + Scoring
        ↓
Final Analysis JSON
```

## Usage

### Single Video Analysis
```bash
python squat_analysis/aqa_metaProgram_squat.py 25713_3
```

### Batch Processing
```bash
# All videos
python squat_analysis/aqa_metaProgram_squat.py --all

# Only "good" quality videos
python squat_analysis/aqa_metaProgram_squat.py --all --quality good
```

### From Specific JSON
```bash
python squat_analysis/aqa_metaProgram_squat.py --json ./squat/extracted_features_clean/good/25713_3.json
```

## Microprograms

### Error Detection (`squat_error_functions.py`)

| Function | Description | Returns |
|----------|-------------|---------|
| `knee_valgus_error()` | Knees caving inward | Ratio (< 1.0 = valgus) |
| `knee_forward_travel_error()` | Knees past toes | Distance (+ = past) |
| `knee_angle_asymmetry()` | Left/right difference | Degrees |
| `hip_shift_error()` | Lateral weight shift | Normalized ratio |
| `forward_lean_error()` | Torso lean from vertical | Degrees |
| `hip_hinge_angle()` | Torso-to-thigh angle | Degrees |
| `heel_rise_error()` | Heels lifting | Distance |
| `stance_width()` | Foot placement | Ratio vs hip width |

### Recognition (`squat_recognition_functions.py`)

| Function | Description |
|----------|-------------|
| `classify_view()` | Camera angle classification (front/back/side) |
| `get_scale_factor()` | Body proportion for normalization |
| `get_knee_angle()` | Knee bend angle |
| `compute_hip_displacement_signal()` | Primary depth signal |
| `find_rep_peaks()` | Detect repetition bottoms |
| `detect_squat_depth_category()` | Parallel/below/above |

## Scoring

Scores are computed on a 0-100 scale:

| Grade | Score |
|-------|-------|
| A | 90-100 |
| B | 80-89 |
| C | 70-79 |
| D | 60-69 |
| F | < 60 |

### Scoring Weights
- **Knee Valgus**: 25%
- **Squat Depth**: 25%
- **Forward Lean**: 20%
- **Hip Shift**: 15%
- **Knee Asymmetry**: 15%

## Output Format

```json
{
  "video_id": "25713_3",
  "success": true,
  "quality_level": "good",
  "view": "side",
  "rep_count": 5,
  "repetitions": [
    {
      "rep_id": 1,
      "start_frame": 45,
      "end_frame": 120,
      "min_knee_angle": 85.2,
      "depth_category": "parallel",
      "errors": {
        "overall": {
          "knee_valgus": {"mean": 0.92, "min": 0.85, "max": 0.98},
          "forward_lean": {"mean": 18.5, ...}
        },
        "by_phase": {
          "eccentric": {...},
          "concentric": {...}
        },
        "at_bottom": {...}
      }
    }
  ],
  "scores": {
    "overall_score": 78.5,
    "grade": "C",
    "rep_scores": [...],
    "feedback": {...}
  }
}
```

## Comparison to Dive AQA

| Dive AQA | Squat AQA |
|----------|-----------|
| `abstractSymbols()` | Features from 2.5 script |
| `getDiveInfo_from_symbols()` | `classify_view()`, phase detection |
| Takeoff/Twist/Entry phases | Idle/Eccentric/Concentric phases |
| `applyFeetApartError()` | `knee_valgus_error()` |
| `applyPositionTightnessError()` | `forward_lean_error()` |
| `splash_area_percentage()` | N/A (no water entry) |
| `distance_from_board_score()` | N/A |
| N/A | `knee_forward_travel_error()` |

## Building Distribution Data

For percentile-based scoring, build distribution from your dataset:

```python
from squat_analysis.scoring_functions import build_distribution_data

# Load all your analyzed squat data
squat_data_list = [...]  # List of analysis dicts

build_distribution_data(squat_data_list, "./squat/distribution_data.pkl")
```

## Dependencies

- numpy
- scipy (for signal processing)
- tqdm (for progress bars)

All dependencies should already be installed from your existing pipeline.
