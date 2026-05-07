# Graph Report - .  (2026-05-06)

## Corpus Check
- Large corpus: 6957 files · ~6,102,480 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder, or use --no-semantic to run AST-only.

## Summary
- 968 nodes · 1551 edges · 63 communities (46 shown, 17 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 62 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_BiLSTM Training|BiLSTM Training]]
- [[_COMMUNITY_Annotation Tool UI|Annotation Tool UI]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_App Py|App Py]]
- [[_COMMUNITY_AQA Scoring|AQA Scoring]]
- [[_COMMUNITY_AQA Scoring|AQA Scoring]]
- [[_COMMUNITY_Pose Extraction|Pose Extraction]]
- [[_COMMUNITY_Legacy Implementations|Legacy Implementations]]
- [[_COMMUNITY_ST-GCN Training|ST-GCN Training]]
- [[_COMMUNITY_Temporal Segmentation|Temporal Segmentation]]
- [[_COMMUNITY_ST-GCN Training|ST-GCN Training]]
- [[_COMMUNITY_ST-GCN Training|ST-GCN Training]]
- [[_COMMUNITY_Temporal Segmentation|Temporal Segmentation]]
- [[_COMMUNITY_Temporal Segmentation|Temporal Segmentation]]
- [[_COMMUNITY_Pose Extraction|Pose Extraction]]
- [[_COMMUNITY_Classify Views|Classify Views]]
- [[_COMMUNITY_Temporal Segmentation|Temporal Segmentation]]
- [[_COMMUNITY_BiLSTM Training|BiLSTM Training]]
- [[_COMMUNITY_BiLSTM Training|BiLSTM Training]]
- [[_COMMUNITY_BiLSTM Training|BiLSTM Training]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Pose Extraction|Pose Extraction]]
- [[_COMMUNITY_FastAPI Backend|FastAPI Backend]]
- [[_COMMUNITY_FastAPI Backend|FastAPI Backend]]
- [[_COMMUNITY_Legacy Implementations|Legacy Implementations]]
- [[_COMMUNITY_Annotation Tool UI|Annotation Tool UI]]
- [[_COMMUNITY_Legacy Implementations|Legacy Implementations]]
- [[_COMMUNITY_Legacy Implementations|Legacy Implementations]]
- [[_COMMUNITY_Config Init|Config Init]]
- [[_COMMUNITY_FastAPI Backend|FastAPI Backend]]
- [[_COMMUNITY_Temporal Segmentation|Temporal Segmentation]]
- [[_COMMUNITY_FastAPI Backend|FastAPI Backend]]
- [[_COMMUNITY_Pose Extraction|Pose Extraction]]
- [[_COMMUNITY_Pose Extraction|Pose Extraction]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Annotation Tool UI|Annotation Tool UI]]
- [[_COMMUNITY_Temporal Segmentation|Temporal Segmentation]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]
- [[_COMMUNITY_Feedback Engine|Feedback Engine]]

## God Nodes (most connected - your core abstractions)
1. `AnnotationToolUI` - 64 edges
2. `PipelineRunnerUI` - 52 edges
3. `main()` - 18 edges
4. `FeedbackEngine` - 16 edges
5. `process_single_video()` - 16 edges
6. `SquatStateMachine` - 16 edges
7. `main()` - 15 edges
8. `PoseExtractor` - 13 edges
9. `BiomechanicalAnalyzer` - 13 edges
10. `pad_or_truncate()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `collect_results()` --calls--> `FeedbackEngine`  [INFERRED]
  apps/api/pipeline.py → core/exevision/feedback/engine.py
- `_pipeline_task()` --calls--> `run_pipeline_sync()`  [INFERRED]
  apps/api/main.py → apps/api/pipeline.py
- `train()` --calls--> `load_bilstm_reps()`  [INFERRED]
  core/exevision/training/pretrain_bilstm.py → core/exevision/neural/nn_utils.py
- `train()` --calls--> `load_bilstm_reps()`  [INFERRED]
  core/exevision/training/overhead_press/pretrain_bilstm.py → core/exevision/neural/nn_utils.py
- `train()` --calls--> `load_bilstm_reps()`  [INFERRED]
  core/exevision/training/seated_overhead_press/pretrain_bilstm.py → core/exevision/neural/nn_utils.py

## Communities (63 total, 17 thin omitted)

### Community 0 - "BiLSTM Training"
Cohesion: 0.05
Nodes (64): apply_safety_clamps(), BiLSTMScorer, build_heuristic_vector(), HeuristicGuidedFusion, Heuristic-anchored fusion where neural branches propose a bounded residual corre, Returns:             final_score: clamped prediction in [0, 100]             r, Fine-tuned BiLSTM for temporal quality scoring., Fine-tuned ST-GCN for spatial quality scoring. (+56 more)

### Community 1 - "Annotation Tool UI"
Cohesion: 0.05
Nodes (19): AnnotationToolUI, Rep annotation tool for ExeVision neural training dataset.      Workflow:, Load annotation flag definitions from the exercise config JSON., Destroy existing flag widgets and rebuild from current exercise config., Rebuild annotation flags when exercise selection changes in annotation tab., Sorts the current video_files list by heuristic score (descending)., Search pipeline_ui_runs/ for a completed run that has         segmented + scored, Build annotation payload from a pipeline run for any target video. (+11 more)

### Community 2 - "Feedback Engine"
Cohesion: 0.05
Nodes (40): assign_item_scores(), detect_mismatch(), FeedbackEngine, FeedbackItem, FeedbackItemBuilder, FeedbackResult, get_category_for_metric(), _load_json() (+32 more)

### Community 3 - "App Py"
Cohesion: 0.07
Nodes (13): _build_stages(), _config_file_for_exercise(), get_view_thresholds(), main(), ordered_stages(), PipelineRunnerUI, Return config JSON stem for the given exercise.     seated_overhead_press shares, Find neural fusion scoring JSON for a video. (+5 more)

### Community 4 - "AQA Scoring"
Cohesion: 0.06
Nodes (59): _angle_2d(), _angle_3d(), _basic_phases_from_hip(), _build_body_frame(), _build_scoring_paths(), calculate_torso_tibia_offset(), calculate_vertical_depth(), _conf() (+51 more)

### Community 5 - "AQA Scoring"
Cohesion: 0.05
Nodes (52): analyze_rep_errors(), aqa_metaprogram_squat(), extract_repetitions_from_phases(), find_feature_json(), find_segmented_json(), get_quality_level(), load_feature_data(), load_segmented_data() (+44 more)

### Community 6 - "Pose Extraction"
Cohesion: 0.05
Nodes (50): _already_processed_json_exists(), analyze_visibility_trends(), apply_one_euro_filter(), apply_savgol_filter(), build_mandatory_chain_flags(), _build_paths(), consolidate_foot_region(), create_visualization_report() (+42 more)

### Community 7 - "Legacy Implementations"
Cohesion: 0.06
Nodes (29): BiomechanicalAnalyzer, Temporal Segmentation Module - Biomechanically-sound squat phase detection Fait, Compute the primary control signal: normalized vertical hip displacement, Compute knee bending angle for all frames (secondary signal), Compute smoothed velocity of hip displacement, Compute velocity trends averaged over windows, Identify frames with sufficient landmark confidence, Compute metrics for a temporal window centered at given frame (+21 more)

### Community 8 - "ST-GCN Training"
Cohesion: 0.1
Nodes (18): pad_or_truncate(), Pad (with zeros) or truncate a sequence to fixed length.     Input: (T, ...) nu, apply_joint_masking(), build_dataloader(), _convert_rep_to_stgcn_tensor(), main(), MaskedRepDataset, parse_args() (+10 more)

### Community 9 - "Temporal Segmentation"
Cohesion: 0.09
Nodes (27): Enum, _build_temporal_paths(), create_segmentation_visualization(), _debug_enabled(), _env_flag(), find_video_file(), _get_control_signal(), _get_thresholds() (+19 more)

### Community 10 - "ST-GCN Training"
Cohesion: 0.12
Nodes (14): Dataset, apply_joint_masking(), build_dataloader(), main(), MaskedRepDataset, parse_args(), batch: (B, C, T, J) — mask random valid (frame, joint) positions via Bernoulli s, _resolve_data_dir() (+6 more)

### Community 11 - "ST-GCN Training"
Cohesion: 0.13
Nodes (13): apply_joint_masking(), build_dataloader(), main(), MaskedRepDataset, parse_args(), batch: (B, C, T, J) — mask random valid (frame, joint) positions via Bernoulli s, _resolve_data_dir(), set_seed() (+5 more)

### Community 12 - "Temporal Segmentation"
Cohesion: 0.13
Nodes (11): BiomechanicalAnalyzer, Main segmentation pipeline with comprehensive error handling, Calculates angle-invariant and view-invariant biomechanical metrics.     Uses b, Extract anthropometric measurements from idle frames.         Now includes outl, Compute the primary control signal based on exercise type.                  Fo, Compute knee bending angle for all frames (secondary signal), Compute smoothed velocity of hip displacement.         Positive velocity = movi, Compute velocity trends averaged over windows.         This is the primary inpu (+3 more)

### Community 13 - "Temporal Segmentation"
Cohesion: 0.13
Nodes (11): Advanced temporal segmentation with biomechanical rigor.          Supports mul, Validate that the camera view is processable.         Allow all views including, Route to exercise-specific rep detection., OHP rep detection: CONCENTRIC → [ISOMETRIC] → ECCENTRIC cycle.         Rep star, Detect repetitions from phase sequences.         A rep = eccentric → [isometric, Route to exercise-specific phase-only rep counting., OHP phase-only fallback: CONCENTRIC → [ISOMETRIC] → ECCENTRIC cycle.         Ig, Simple rep counting from phase sequence ONLY.          Counts 1 rep for: (+3 more)

### Community 14 - "Pose Extraction"
Cohesion: 0.21
Nodes (18): _as_path(), _build_feature_for_video(), _compute_velocity(), _discover_feature_index(), _discover_segmented_files(), _extract_active_joints(), _extract_rep_matrix(), _extract_stgcn_rep() (+10 more)

### Community 15 - "Classify Views"
Cohesion: 0.15
Nodes (18): _build_features_dirs(), _classify_frame(), _face_score(), _facing_camera(), get_view_label(), get_view_label_with_probs(), _is_visible(), process_video_classification() (+10 more)

### Community 16 - "Temporal Segmentation"
Cohesion: 0.19
Nodes (8): Post-processing pass to merge short phases into neighbors.         Eliminates a, OHP phase detection. Control signal = inverted wrist Y (positive = wrists rising, Simplified Machine for squat phase detection.          Logic:     1. Positive, Allow IDLE once the control signal returns near the starting position., Enforce strict legal adjacency globally.          Allowed high-level cycle:, Route to exercise-specific phase detection., Simplified phase detection looping over all frames.         Returns array of ph, SquatStateMachine

### Community 17 - "BiLSTM Training"
Cohesion: 0.16
Nodes (10): apply_masking(), BiLSTMPretrainer, build_dataloader(), main(), parse_args(), RepSignalDataset, _resolve_data_dir(), set_seed() (+2 more)

### Community 18 - "BiLSTM Training"
Cohesion: 0.16
Nodes (10): apply_masking(), BiLSTMPretrainer, build_dataloader(), main(), parse_args(), RepSignalDataset, _resolve_data_dir(), set_seed() (+2 more)

### Community 19 - "BiLSTM Training"
Cohesion: 0.16
Nodes (10): apply_masking(), BiLSTMPretrainer, build_dataloader(), main(), parse_args(), RepSignalDataset, _resolve_data_dir(), set_seed() (+2 more)

### Community 20 - "Feedback Engine"
Cohesion: 0.15
Nodes (16): _build_stage_specs(), _cleanup_workspace(), coerce_old_feedback_format(), _delete_input_video(), _prepare_workspace(), ExeVision AI — Pipeline Runner  Mirrors the stage execution logic from apps/de, Create workspace directory tree and copy the input video into it., Ensure each stage produced the expected artifact for the requested video. (+8 more)

### Community 21 - "Pose Extraction"
Cohesion: 0.15
Nodes (9): PoseExtractor, Analyze visibility patterns across frames, Create a comprehensive text and visualization report, Internal video processing logic with quality assessment, Extracts pose landmarks from exercise videos using MediaPipe with quality assess, Check that the MediaPipe model exists, Create MediaPipe PoseLandmarker options, Map landmark index to joint type for threshold selection (+1 more)

### Community 22 - "FastAPI Backend"
Cohesion: 0.18
Nodes (12): get_job(), health(), InferRequest, JobStatus, _normalize_stage_selection(), ExeVision AI — FastAPI Inference Server  Wraps the squat analysis pipeline for, Accept a video URL and enqueue a pipeline run.     Returns immediately with a j, Poll for job status. Once status='done', result contains the full analysis. (+4 more)

### Community 23 - "FastAPI Backend"
Cohesion: 0.15
Nodes (14): _build_kinematic_data(), _build_phase_timeline(), collect_results(), _find_json(), _get_field_mapping(), Glob for a JSON file anywhere under base., Build a phase timeline for one rep, including idle phases inferred from gaps., Build ROM time-series for one rep using hip vertical displacement.      Return (+6 more)

### Community 24 - "Legacy Implementations"
Cohesion: 0.15
Nodes (10): PipelineResult, Configuration and result dataclasses for the ExeVision Pipeline, Single phase within a repetition, Single squat repetition with phase breakdown, Result from temporal segmentation stage, Complete pipeline result for a single video, Convert result to serializable dictionary, Repetition (+2 more)

### Community 25 - "Annotation Tool UI"
Cohesion: 0.25
Nodes (10): main(), Strategic sample selection for ExeVision annotation.  Selects a subset of reps f, Select reps near heuristic score decision boundaries., Equal representation from each view type., Top and bottom of heuristic score range., Select a strategic subset of reps for human annotation.     Returns selected rep, select_boundary_reps(), select_extremes() (+2 more)

### Community 26 - "Legacy Implementations"
Cohesion: 0.24
Nodes (7): Result from view classification stage, ViewResult, Classifies camera view using rule-based geometric analysis, Classify view directly from JSON data (for updating existing JSON files)., Classify the camera view from extracted pose landmarks using rule-based logic., Classify camera view from keypoints using geometric rules.         Returns: 'fr, ViewClassifier

### Community 27 - "Legacy Implementations"
Cohesion: 0.24
Nodes (7): PipelineConfig, Central configuration for the entire pipeline, ExeVisionPipeline, main(), ExeVision AI - Main Pipeline Orchestrator =====================================, Main pipeline that orchestrates all analysis stages, Process a video through the complete pipeline.

### Community 28 - "Config Init"
Cohesion: 0.36
Nodes (5): resolve_run_paths(), RunPaths, _env_bool(), ExeVisionSettings, load_settings()

### Community 29 - "FastAPI Backend"
Cohesion: 0.29
Nodes (7): _fire_callback(), _pipeline_task(), POST the job result to the Next.js callback endpoint.     Includes the Authoriz, Downloads the video and runs the full pipeline. Runs in a background thread., _update_job(), download_video(), Download a video from a URL (e.g. Supabase signed URL) into dest_dir.

### Community 30 - "Temporal Segmentation"
Cohesion: 0.29
Nodes (4): Convert phase ID to name, Extract phase breakdown for a single rep with transition reasons, Single phase within a repetition, RepPhase

### Community 31 - "FastAPI Backend"
Cohesion: 0.33
Nodes (6): _build_stage_cmd(), _get_model_path(), Build the subprocess command for a stage, mirroring app.py arg construction., Run one pipeline stage; returns captured stdout+stderr., Construct exercise-specific model path, with fallback to generic names for compa, _run_stage()

### Community 32 - "Pose Extraction"
Cohesion: 0.33
Nodes (3): ExeVision AI - Production Pipeline Module, Pose Extraction Module - Extracts MediaPipe pose landmarks from videos with qual, View Classification Module - Rule-based camera angle classification Faithful re

### Community 33 - "Pose Extraction"
Cohesion: 0.33
Nodes (4): ExtractionResult, Result from pose extraction stage, Extract pose landmarks from a video with quality assessment.          Args:, Extract pose landmarks and save to JSON file.          Args:             vide

### Community 34 - "Feedback Engine"
Cohesion: 0.5
Nodes (4): _build_feedback_fallback(), Map numeric score to feedback tier labels expected by the frontend., Create a schema-compatible fallback when template/config files are unavailable., _tier_for_score()

### Community 35 - "Annotation Tool UI"
Cohesion: 0.67
Nodes (3): analyze_annotations(), main(), Annotation quality self-check for ExeVision.  After finishing annotations, run t

### Community 36 - "Temporal Segmentation"
Cohesion: 0.5
Nodes (3): Metrics computed over a temporal window, Compute metrics for a temporal window centered at given frame, WindowMetrics

## Knowledge Gaps
- **295 isolated node(s):** `ExeVision AI — FastAPI Inference Server  Wraps the squat analysis pipeline for`, `Validate and normalize requested stages.      Rules:     - Unknown stage name`, `POST the job result to the Next.js callback endpoint.     Includes the Authoriz`, `Downloads the video and runs the full pipeline. Runs in a background thread.`, `Accept a video URL and enqueue a pipeline run.     Returns immediately with a j` (+290 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **17 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `convert_to_serializable()` connect `AQA Scoring` to `Temporal Segmentation`, `Temporal Segmentation`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Why does `AnnotationToolUI` connect `Annotation Tool UI` to `App Py`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `main()` (e.g. with `build_adjacency_matrix()` and `BiLSTMScorer`) actually correct?**
  _`main()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ExeVision AI — FastAPI Inference Server  Wraps the squat analysis pipeline for`, `Validate and normalize requested stages.      Rules:     - Unknown stage name`, `POST the job result to the Next.js callback endpoint.     Includes the Authoriz` to the rest of the system?**
  _295 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `BiLSTM Training` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Should `Annotation Tool UI` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Should `Feedback Engine` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._