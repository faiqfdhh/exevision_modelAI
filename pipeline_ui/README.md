# ExeVision Minimal Pipeline UI

A minimal desktop UI for running your existing pipeline scripts without changing the underlying implementation.

## What it does

- Runs the existing scripts in order (full, individual, or custom stage selection)
- Lets you choose a single input video directly from file explorer
- **Stop button to terminate processing** at any time during a run
- Creates a **new isolated run folder** for every run
- Keeps your old folders untouched
- Chains outputs between stages automatically inside a run workspace
- Saves each stage output in a separate snapshot folder
- Shows generated annotated MediaPipe overlay video inside the app preview panel

## Input options

- **Use single video file**: pick one video (`.mp4`, `.mov`, `.avi`, `.mkv`, `.flv`) with **Browse**
- **Use dataset folder**: run pipeline on a full dataset directory

When single-video mode is selected, the app copies only that one file into the isolated run workspace before running stages.

## Controls

- **Start Pipeline**: Begin processing with selected stages and input
- **Stop Pipeline**: Terminate the current running pipeline at any point. The button becomes available when a pipeline is running and is disabled when idle.
  - Gracefully stops the current stage
  - Closes the running process
  - Saves partial results
  - Allows you to close the window safely (with confirmation if pipeline is running)

## Video Preview Controls

- **Load**: Load the selected annotated output video into the preview
- **Play/Pause**: Start playback or pause the current video
- **Stop**: Stop playback and reset video to the beginning
- **Fullscreen**: View the video in fullscreen mode
  - Press **ESC** or **Q** to exit fullscreen
  - Video plays at full resolution scaled to fit your screen
  - Click the window close button to exit fullscreen

## Supported stages

1. `2.5_extract_selected_features.py`
2. `4_classify_views.py`
3. `5_temporal_segmentation.py`
4. `8_scoring.py`
5. `analyze_results.py`

## Run

From project root:

```bash
python pipeline_ui/app.py
```

## Output structure

Each run is saved under:

`pipeline_ui_runs/<run_name>/`

With:

- `workspace/` → isolated working environment used by scripts
- `logs/` → one log file per stage
- `stage_outputs/01_<stage_key>/...` → snapshot of that stage outputs

Annotated video overlays are produced by stage `2.5_extract_selected_features.py` and shown in the app preview when available.

This preserves stage-by-stage data flow:

- Stage N writes to `workspace/squat/...`
- Stage N+1 reads from that updated `workspace/squat/...`
- Snapshot copies are stored separately after each stage
