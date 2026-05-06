# Graph Report - exevision_modelAI  (2026-05-06)

## Corpus Check
- 154 files · ~6,076,905 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1982 nodes · 3879 edges · 120 communities (98 shown, 22 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 268 edges (avg confidence: 0.65)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1c8df9f0`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 86|Community 86]]
- [[_COMMUNITY_Community 88|Community 88]]
- [[_COMMUNITY_Community 95|Community 95]]
- [[_COMMUNITY_Community 96|Community 96]]
- [[_COMMUNITY_Community 97|Community 97]]
- [[_COMMUNITY_Community 98|Community 98]]
- [[_COMMUNITY_Community 99|Community 99]]
- [[_COMMUNITY_Community 100|Community 100]]
- [[_COMMUNITY_Community 101|Community 101]]
- [[_COMMUNITY_Community 102|Community 102]]
- [[_COMMUNITY_Community 103|Community 103]]
- [[_COMMUNITY_Community 104|Community 104]]
- [[_COMMUNITY_Community 105|Community 105]]
- [[_COMMUNITY_Community 106|Community 106]]
- [[_COMMUNITY_Community 107|Community 107]]
- [[_COMMUNITY_Community 108|Community 108]]
- [[_COMMUNITY_Community 109|Community 109]]
- [[_COMMUNITY_Community 110|Community 110]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 70 edges
2. `AnnotationToolUI` - 61 edges
3. `AnnotationToolUI` - 61 edges
4. `PipelineRunnerUI` - 52 edges
5. `PipelineRunnerUI` - 51 edges
6. `evaluate_metric()` - 48 edges
7. `format_value()` - 44 edges
8. `threshold_text()` - 44 edges
9. `build_rep_summary()` - 44 edges
10. `build_video_summary()` - 44 edges

## Surprising Connections (you probably didn't know these)
- `test_feedback_e2e_squat()` --calls--> `FeedbackEngine`  [INFERRED]
  tests/test_feedback_e2e.py → core/exevision/feedback/engine.py
- `test_control_signal_extraction()` --calls--> `_get_control_signal()`  [INFERRED]
  test_temporal_segmentation_refactor.py → core/exevision/stages/temporal_segmentation.py
- `test_debug_mode()` --calls--> `_debug_enabled()`  [INFERRED]
  test_temporal_segmentation_refactor.py → core/exevision/stages/temporal_segmentation.py
- `test_callback_payload_includes_visualization()` --calls--> `_fire_callback()`  [INFERRED]
  tests/test_visualization_integration.py → apps/api/main.py
- `StageSpec` --uses--> `FeedbackEngine`  [INFERRED]
  apps/api/pipeline.py → core/exevision/feedback/engine.py

## Communities (120 total, 22 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (16): AnnotationToolUI, Rep annotation tool for ExeVision neural training dataset.      Workflow:, Sorts the current video_files list by heuristic score (descending)., Search pipeline_ui_runs/ for a completed run that has         segmented + scored, Build annotation payload from a pipeline run for any target video., Build and save annotation for the currently loaded video from pipeline output., Build and save annotation metadata for any processed video during batch runs., Run Stages 2.5→4→5→8 for a single video, then load results. (+8 more)

### Community 1 - "Community 1"
Cohesion: 0.05
Nodes (72): _angle_2d(), _angle_3d(), _basic_phases_from_hip(), _build_body_frame(), _build_scoring_paths(), calculate_torso_tibia_offset(), calculate_vertical_depth(), _conf() (+64 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (16): AnnotationToolUI, Rep annotation tool for ExeVision neural training dataset.      Workflow:, Sorts the current video_files list by heuristic score (descending)., Search pipeline_ui_runs/ for a completed run that has         segmented + scored, Build annotation payload from a pipeline run for any target video., Build and save annotation for the currently loaded video from pipeline output., Build and save annotation metadata for any processed video during batch runs., Run Stages 2.5→4→5→8 for a single video, then load results. (+8 more)

### Community 3 - "Community 3"
Cohesion: 0.05
Nodes (45): assign_training_weights(), build_records(), evaluate_bilstm(), evaluate_fusion(), evaluate_stgcn(), get_device(), infer_video_score(), load_json() (+37 more)

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (56): applyFeetApartError(), applyPositionTightnessError(), board_end(), calculate_distance_from_platform_for_one_frame(), draw_two_coord(), find_which_side_board_on(), get_splash_from_one_frame(), get_splash_pred_mask() (+48 more)

### Community 5 - "Community 5"
Cohesion: 0.07
Nodes (11): _build_stages(), get_view_thresholds(), main(), ordered_stages(), PipelineRunnerUI, Find neural fusion scoring JSON for a video., Update STAGES when exercise selection changes., Build stage definitions with exercise-specific paths. (+3 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (8): get_view_thresholds(), main(), ordered_stages(), PipelineRunnerUI, Start full pipeline with neural fusion scoring (stages 2.5 → 4 → 5 → 8 → neural_, Find neural fusion scoring JSON for a video., score_severity(), Stage

### Community 7 - "Community 7"
Cohesion: 0.05
Nodes (50): _already_processed_json_exists(), analyze_visibility_trends(), apply_one_euro_filter(), apply_savgol_filter(), build_mandatory_chain_flags(), _build_paths(), consolidate_foot_region(), create_visualization_report() (+42 more)

### Community 8 - "Community 8"
Cohesion: 0.36
Nodes (12): build_rep_summary(), build_video_summary(), detect_base_dir(), evaluate_metric(), format_value(), get_view_thresholds(), iter_score_files(), main() (+4 more)

### Community 9 - "Community 9"
Cohesion: 0.07
Nodes (24): Dataset, apply_masking(), BiLSTMPretrainer, build_dataloader(), main(), parse_args(), RepSignalDataset, _resolve_data_dir() (+16 more)

### Community 10 - "Community 10"
Cohesion: 0.07
Nodes (38): _already_processed_json_exists(), analyze_visibility_trends(), apply_one_euro_filter(), apply_savgol_filter(), build_mandatory_chain_flags(), consolidate_foot_region(), create_visualization_report(), draw_landmarks_enhanced() (+30 more)

### Community 11 - "Community 11"
Cohesion: 0.08
Nodes (35): _load_ohp_config(), _make_frame(), _make_rep(), Unit tests for OHP scoring geometry helpers and metric functions. No video file, Create a list of identical frames for testing., Wrists at same X as shoulders → grip_ratio = 0., Wrists 20% wider than shoulders → grip_ratio ≈ 0.20., Wrists move straight up (no horizontal drift) → deviation = 0. (+27 more)

### Community 12 - "Community 12"
Cohesion: 0.07
Nodes (33): _build_temporal_paths(), create_segmentation_visualization(), _debug_enabled(), _env_flag(), find_video_file(), _get_control_signal(), _hip_y_sequence(), process_video() (+25 more)

### Community 13 - "Community 13"
Cohesion: 0.29
Nodes (19): analyze_common_errors(), analyze_correlations(), analyze_depth_by_view(), analyze_metric_distributions(), analyze_score_distribution(), analyze_zero_depth_issues(), generate_summary_report(), load_all_results() (+11 more)

### Community 14 - "Community 14"
Cohesion: 0.08
Nodes (33): Enum, classify_view(), compute_hip_displacement_signal(), compute_knee_angle_signal(), count_reps(), detect_phase_simple(), find_rep_peaks(), get_hip_height() (+25 more)

### Community 15 - "Community 15"
Cohesion: 0.11
Nodes (32): _angle_2d(), _basic_phases_from_hip(), calculate_torso_tibia_offset(), calculate_vertical_depth(), _conf(), find_all_video_ids(), find_feature_json(), find_segmented_json() (+24 more)

### Community 16 - "Community 16"
Cohesion: 0.11
Nodes (32): Squat Analysis Microprograms  Low-level functions for error detection and squa, aggregate_phase_errors(), calculate_angle(), calculate_angle_3d(), forward_lean_error(), get_all_errors_for_frame(), get_confidence(), get_landmark() (+24 more)

### Community 17 - "Community 17"
Cohesion: 0.12
Nodes (18): assign_item_scores(), detect_mismatch(), FeedbackEngine, get_category_for_metric(), _metric_phrase_tier(), Feedback template engine orchestration for rep and session coaching., Main feedback orchestrator using config + templates + deterministic rendering., Generate per-rep and session-level feedback from score data. (+10 more)

### Community 18 - "Community 18"
Cohesion: 0.13
Nodes (26): Squat Analysis - Rule-Based AQA Framework  Automated Quality Assessment for sq, build_distribution_data(), compute_percentile_score(), compute_rep_score(), compute_set_score(), load_distribution_data(), scoring_functions.py Squat quality scoring functions.  Converts error measure, Score knee valgus measurement.          Args:         valgus_ratio: Knee spre (+18 more)

### Community 19 - "Community 19"
Cohesion: 0.12
Nodes (18): Post-processing pass to merge short phases into neighbors.         Eliminates a, Post-processing pass to merge short phases into neighbors.         Eliminates a, OHP phase detection. Control signal = inverted wrist Y (positive = wrists rising, Simplified Machine for squat phase detection.          Logic:     1. Positive, Only allow IDLE once hips return near beginning-of-video standing position., Enforce strict legal adjacency globally.          Allowed high-level cycle:, Simplified phase detection looping over all frames.         Returns array of ph, Post-processing pass to merge short phases into neighbors.         Eliminates a (+10 more)

### Community 20 - "Community 20"
Cohesion: 0.12
Nodes (14): apply_joint_masking(), build_dataloader(), main(), MaskedRepDataset, parse_args(), batch: (batch, 7, 128, 11)     Randomly mask joint positions across frames., batch: (B, C, T, J) — mask random valid (frame, joint) positions via Bernoulli s, _resolve_data_dir() (+6 more)

### Community 21 - "Community 21"
Cohesion: 0.12
Nodes (13): apply_joint_masking(), build_dataloader(), main(), MaskedRepDataset, parse_args(), batch: (B, C, T, J) — mask random valid (frame, joint) positions via Bernoulli s, _resolve_data_dir(), set_seed() (+5 more)

### Community 22 - "Community 22"
Cohesion: 0.12
Nodes (13): apply_joint_masking(), build_dataloader(), main(), MaskedRepDataset, parse_args(), batch: (batch, 7, 128, 11)     Randomly mask joint positions across frames., _resolve_data_dir(), set_seed() (+5 more)

### Community 23 - "Community 23"
Cohesion: 0.18
Nodes (25): assign_training_weights(), build_records(), evaluate_bilstm(), evaluate_fusion(), evaluate_stgcn(), get_device(), infer_video_score(), load_json() (+17 more)

### Community 24 - "Community 24"
Cohesion: 0.15
Nodes (23): analyze_rep_errors(), aqa_metaprogram_squat(), extract_repetitions_from_phases(), find_feature_json(), find_segmented_json(), get_quality_level(), load_feature_data(), load_segmented_data() (+15 more)

### Community 25 - "Community 25"
Cohesion: 0.1
Nodes (17): BiomechanicalAnalyzer, Calculates angle-invariant and view-invariant biomechanical metrics.     Uses b, Compute smoothed velocity of hip displacement.         Positive velocity = movi, Calculates angle-invariant and view-invariant biomechanical metrics.     Uses b, Identify frames with sufficient landmark confidence, Calculates angle-invariant and view-invariant biomechanical metrics.     Uses b, Fill NaN values with linear interpolation, Compute the primary control signal based on exercise type.                  Fo (+9 more)

### Community 26 - "Community 26"
Cohesion: 0.12
Nodes (13): ExtractionResult, Result from pose extraction stage, PoseExtractor, Analyze visibility patterns across frames, Create a comprehensive text and visualization report, Extract pose landmarks from a video with quality assessment.          Args:, Internal video processing logic with quality assessment, Extract pose landmarks and save to JSON file.          Args:             vide (+5 more)

### Community 27 - "Community 27"
Cohesion: 0.09
Nodes (18): convert_to_serializable(), Convert phase ID to name, Main segmentation pipeline with comprehensive error handling, Convert phase ID to name, Main segmentation pipeline with comprehensive error handling, Recursively convert numpy types to Python native types for JSON serialization, Extract anthropometric measurements from idle frames.         Now includes outl, Recursively convert numpy types to Python native types for JSON serialization (+10 more)

### Community 28 - "Community 28"
Cohesion: 0.14
Nodes (22): discover_videos(), extract_bilstm_rep(), get_device(), infer_rep(), _load_model_state(), _load_stgcn_with_compat(), main(), _normalize_state_dict_keys() (+14 more)

### Community 29 - "Community 29"
Cohesion: 0.16
Nodes (22): _as_path(), build_adjacency_matrix(), _build_feature_for_video(), _compute_velocity(), _discover_feature_index(), _discover_segmented_files(), _extract_active_joints(), _extract_rep_matrix() (+14 more)

### Community 30 - "Community 30"
Cohesion: 0.12
Nodes (16): PipelineResult, Configuration and result dataclasses for the ExeVision Pipeline, Single phase within a repetition, Single squat repetition with phase breakdown, Result from temporal segmentation stage, Complete pipeline result for a single video, Convert result to serializable dictionary, Repetition (+8 more)

### Community 31 - "Community 31"
Cohesion: 0.11
Nodes (16): Detect repetitions from phase sequences.         A rep = eccentric → [isometric, Extract phase breakdown for a single rep with transition reasons, Route to exercise-specific rep detection., OHP rep detection: CONCENTRIC → [ISOMETRIC] → ECCENTRIC cycle.         Rep star, Detect repetitions from phase sequences.         A rep = eccentric → [isometric, Extract phase breakdown for a single rep with transition reasons, Extract phase breakdown for a single rep with transition reasons, Single phase within a repetition (+8 more)

### Community 32 - "Community 32"
Cohesion: 0.17
Nodes (21): _as_path(), _build_feature_for_video(), _compute_velocity(), _discover_feature_index(), _discover_segmented_files(), _extract_active_joints(), _extract_rep_matrix(), _extract_stgcn_rep() (+13 more)

### Community 33 - "Community 33"
Cohesion: 0.14
Nodes (14): PipelineConfig, Result from view classification stage, Central configuration for the entire pipeline, ViewResult, ExeVisionPipeline, main(), ExeVision AI - Main Pipeline Orchestrator =====================================, Main pipeline that orchestrates all analysis stages (+6 more)

### Community 34 - "Community 34"
Cohesion: 0.15
Nodes (11): BiomechanicalAnalyzer, Compute the primary control signal: normalized vertical hip displacement, Compute knee bending angle for all frames (secondary signal), Compute smoothed velocity of hip displacement, Compute velocity trends averaged over windows, Identify frames with sufficient landmark confidence, Calculate angle at p2 formed by p1-p2-p3, Fill NaN values with linear interpolation (+3 more)

### Community 35 - "Community 35"
Cohesion: 0.12
Nodes (19): _fire_callback(), get_job(), health(), InferRequest, JobStatus, _normalize_stage_selection(), _pipeline_task(), ExeVision AI — FastAPI Inference Server  Wraps the squat analysis pipeline for (+11 more)

### Community 36 - "Community 36"
Cohesion: 0.12
Nodes (17): ExtractionResult, PipelineConfig, Configuration and result dataclasses for the ExeVision Pipeline, Result from view classification stage, Single phase within a repetition, Central configuration for the entire pipeline, Result from temporal segmentation stage, Result from pose extraction stage (+9 more)

### Community 37 - "Community 37"
Cohesion: 0.12
Nodes (11): PoseExtractor, Analyze visibility patterns across frames, Create a comprehensive text and visualization report, Extract pose landmarks from a video with quality assessment.          Args:, Internal video processing logic with quality assessment, Extract pose landmarks and save to JSON file.          Args:             vide, Extracts pose landmarks from exercise videos using MediaPipe with quality assess, Check that the MediaPipe model exists (+3 more)

### Community 38 - "Community 38"
Cohesion: 0.15
Nodes (11): BiomechanicalAnalyzer, Compute the primary control signal: normalized vertical hip displacement, Compute knee bending angle for all frames (secondary signal), Compute smoothed velocity of hip displacement, Compute velocity trends averaged over windows, Identify frames with sufficient landmark confidence, Calculate angle at p2 formed by p1-p2-p3, Fill NaN values with linear interpolation (+3 more)

### Community 39 - "Community 39"
Cohesion: 0.15
Nodes (8): Session aggregation helpers for session-level coaching summary., Builds aggregate metrics and trajectory across all reps., SessionAggregator, Tests for SessionAggregator session-level summary logic., TestAverageScore, TestMostImprovedMetric, TestPersistentIssue, TestTrajectoryDetection

### Community 40 - "Community 40"
Cohesion: 0.13
Nodes (10): BiomechanicalAnalyzer, Calculates angle-invariant and view-invariant biomechanical metrics.     Uses b, Extract anthropometric measurements from idle frames.         Now includes outl, Compute the primary control signal: normalized vertical hip displacement., Compute knee bending angle for all frames (secondary signal), Compute smoothed velocity of hip displacement.         Positive velocity = movi, Compute velocity trends averaged over windows.         This is the primary inpu, Identify frames with sufficient landmark confidence (+2 more)

### Community 41 - "Community 41"
Cohesion: 0.14
Nodes (18): _build_features_dirs(), _classify_frame(), _face_score(), _facing_camera(), get_view_label(), get_view_label_with_probs(), _is_visible(), process_video_classification() (+10 more)

### Community 42 - "Community 42"
Cohesion: 0.17
Nodes (18): discover_videos(), get_device(), _load_model_state(), _load_stgcn_with_compat(), main(), _normalize_state_dict_keys(), parse_args(), process_video() (+10 more)

### Community 43 - "Community 43"
Cohesion: 0.16
Nodes (10): apply_masking(), BiLSTMPretrainer, build_dataloader(), main(), parse_args(), RepSignalDataset, _resolve_data_dir(), set_seed() (+2 more)

### Community 44 - "Community 44"
Cohesion: 0.12
Nodes (16): convert_to_serializable(), create_segmentation_visualization(), find_video_file(), process_video(), Temporal Segmentation Module (5.4) Segments squat motion into idle/eccentric/co, Process a single video's keypoints. Returns (video_id, status, result, quality), Create annotated video with phase overlay, rep markers, and signal graphs, Process all extracted features and segment into reps using biomechanical analysi (+8 more)

### Community 45 - "Community 45"
Cohesion: 0.15
Nodes (11): HeuristicGuidedFusion, Heuristic-anchored fusion where neural branches propose a bounded residual corre, Returns:             final_score: clamped prediction in [0, 100]             r, build_adjacency_matrix(), Build the normalized adjacency matrix for the 11-joint skeleton graph.      Re, Build the normalized adjacency matrix for the 11-joint skeleton graph.      Re, mae(), main() (+3 more)

### Community 46 - "Community 46"
Cohesion: 0.18
Nodes (16): create_mock_keypoints(), main(), Test that BiomechanicalAnalyzer accepts exercise parameter., Test that debug mode can be enabled., Test that TemporalSegmenter accepts exercise parameter., Run all verification tests., Create mock MediaPipe keypoints for testing., Test that control signal extraction works for both exercises. (+8 more)

### Community 47 - "Community 47"
Cohesion: 0.14
Nodes (12): Advanced temporal segmentation with biomechanical rigor.          Supports mul, Simple rep counting from phase sequence ONLY.          Counts 1 rep for:, Validate that the camera view is processable.         Allow all views including, Advanced temporal segmentation with biomechanical rigor.          Supports mul, Validate that the camera view is processable.         Allow all views including, Simple rep counting from phase sequence ONLY.          Counts 1 rep for:, Route to exercise-specific phase-only rep counting., OHP phase-only fallback: CONCENTRIC → [ISOMETRIC] → ECCENTRIC cycle.         Ig (+4 more)

### Community 48 - "Community 48"
Cohesion: 0.15
Nodes (5): BiLSTMScorer, Fine-tuned BiLSTM for temporal quality scoring., Fine-tuned ST-GCN for spatial quality scoring., STGCNScorer, TemporalAttention

### Community 49 - "Community 49"
Cohesion: 0.15
Nodes (16): _build_stage_specs(), coerce_old_feedback_format(), _delete_input_video(), _prepare_workspace(), ExeVision AI — Pipeline Runner  Mirrors the stage execution logic from apps/de, Create workspace directory tree and copy the input video into it., Run one pipeline stage; returns captured stdout+stderr., Ensure each stage produced the expected artifact for the requested video. (+8 more)

### Community 50 - "Community 50"
Cohesion: 0.16
Nodes (16): _classify_frame(), _face_score(), _facing_camera(), get_view_label(), get_view_label_with_probs(), _is_visible(), process_video_classification(), Classify camera view. Returns the view with the most frame votes. (+8 more)

### Community 51 - "Community 51"
Cohesion: 0.22
Nodes (6): Computes rep-over-rep deltas and improvement tiers., RepComparator, Tests for RepComparator rep-over-rep progress logic., TestImprovementTier, TestMetricImprovement, TestRepComparison

### Community 52 - "Community 52"
Cohesion: 0.21
Nodes (7): FeedbackResult, Complete feedback output for all reps plus session summary., Handles deterministic phrase selection and template slot replacement., TemplateRenderer, Tests for TemplateRenderer phrase selection and slot filling., TestPhraseSelection, TestSlotFilling

### Community 53 - "Community 53"
Cohesion: 0.17
Nodes (8): Window-based Finite State Machine for squat phase detection, Main FSM loop - classifies each frame, Determine what phase the current window suggests, Handle state transitions with hysteresis and validation, Generate human-readable transition reason, Post-process to remove very short phases, Convert phase ID to string name, SquatStateMachine

### Community 54 - "Community 54"
Cohesion: 0.15
Nodes (10): PipelineResult, Single squat repetition with phase breakdown, Complete pipeline result for a single video, Convert result to serializable dictionary, Repetition, ExeVisionPipeline, main(), ExeVision AI - Main Pipeline Orchestrator ===================================== (+2 more)

### Community 55 - "Community 55"
Cohesion: 0.23
Nodes (9): apply_masking(), BiLSTMPretrainer, build_dataloader(), main(), parse_args(), RepSignalDataset, _resolve_data_dir(), set_seed() (+1 more)

### Community 56 - "Community 56"
Cohesion: 0.15
Nodes (13): _cleanup_workspace(), Remove heavy intermediate artifacts from the workspace after results are collect, Test visualization integration: generate_viz flag handling in the backend.  Te, When generate_viz=False, cleanup should delete visualization directories., Result schema should include visualization_url and visualization_available field, Callback payload should include visualization metadata for success and failure s, Verify that critical path calls to .exists() are all properly guarded., When generate_viz=True, cleanup should NOT delete visualization directories. (+5 more)

### Community 57 - "Community 57"
Cohesion: 0.15
Nodes (14): _build_kinematic_data(), _build_phase_timeline(), collect_results(), _find_json(), _get_field_mapping(), Glob for a JSON file anywhere under base., Build a phase timeline for one rep, including idle phases inferred from gaps., Build ROM time-series for one rep using hip vertical displacement.      Return (+6 more)

### Community 58 - "Community 58"
Cohesion: 0.19
Nodes (8): Simple rep counting from phase sequence ONLY.          Counts 1 rep for:, Single squat repetition with phase breakdown, Advanced temporal segmentation with biomechanical rigor.          Pipeline:, Validate that the camera view is processable.         Allow all views including, Main segmentation pipeline with comprehensive error handling, Detect repetitions from phase sequences.         A rep = eccentric → [isometric, Repetition, TemporalSegmenter

### Community 59 - "Community 59"
Cohesion: 0.22
Nodes (6): Simplified Machine for squat phase detection.          Logic:     1. Positive, Only allow IDLE once hips return near beginning-of-video standing position., Enforce strict legal adjacency globally.          Allowed high-level cycle:, Simplified phase detection looping over all frames.         Returns array of ph, Post-processing pass to merge short phases into neighbors.         Eliminates a, SquatStateMachine

### Community 60 - "Community 60"
Cohesion: 0.19
Nodes (11): apply_safety_clamps(), build_heuristic_vector(), _safe_score(), pad_or_truncate(), Pad (with zeros) or truncate a sequence to fixed length.     Input: (T, ...) nu, Pad (with zeros) or truncate a sequence to fixed length.     Input: (T, ...) nu, extract_bilstm_rep(), infer_rep() (+3 more)

### Community 61 - "Community 61"
Cohesion: 0.21
Nodes (7): Window-based Finite State Machine for squat phase detection, Main FSM loop - classifies each frame, Determine what phase the current window suggests, Handle state transitions with hysteresis and validation, Generate human-readable transition reason, Post-process to remove very short phases, SquatStateMachine

### Community 62 - "Community 62"
Cohesion: 0.25
Nodes (10): main(), Strategic sample selection for ExeVision annotation.  Selects a subset of reps f, Select reps near heuristic score decision boundaries., Equal representation from each view type., Top and bottom of heuristic score range., Select a strategic subset of reps for human annotation.     Returns selected rep, select_boundary_reps(), select_extremes() (+2 more)

### Community 63 - "Community 63"
Cohesion: 0.25
Nodes (10): main(), Strategic sample selection for ExeVision annotation.  Selects a subset of reps f, Select reps near heuristic score decision boundaries., Equal representation from each view type., Top and bottom of heuristic score range., Select a strategic subset of reps for human annotation.     Returns selected rep, select_boundary_reps(), select_extremes() (+2 more)

### Community 64 - "Community 64"
Cohesion: 0.29
Nodes (5): QualityChecker, Detect mismatch between overall score and sub-metric breakdown., Tests for FeedbackEngine quality checks and full orchestration., TestFeedbackEngineIntegration, TestMismatchDetection

### Community 65 - "Community 65"
Cohesion: 0.28
Nodes (5): Classifies camera view using rule-based geometric analysis, Classify view directly from JSON data (for updating existing JSON files)., Classify the camera view from extracted pose landmarks using rule-based logic., Classify camera view from keypoints using geometric rules.         Returns: 'fr, ViewClassifier

### Community 66 - "Community 66"
Cohesion: 0.36
Nodes (5): resolve_run_paths(), RunPaths, _env_bool(), ExeVisionSettings, load_settings()

### Community 67 - "Community 67"
Cohesion: 0.25
Nodes (7): Metrics computed over a temporal window, Metrics computed over a temporal window, Metrics computed over a temporal window, Compute metrics for a temporal window centered at given frame, Compute metrics for a temporal window centered at given frame, Compute metrics for a temporal window centered at given frame, WindowMetrics

### Community 68 - "Community 68"
Cohesion: 0.25
Nodes (8): _build_stage_cmd(), _get_model_path(), Build the subprocess command for a stage, mirroring app.py arg construction., Construct exercise-specific model path, with fallback to generic names for compa, When generate_viz=True, Stage 2.5 and Stage 5 should generate visualizations., When generate_viz=False, both Stage 2.5 and Stage 5 should skip visualization., test_pipeline_respects_generate_viz_true(), test_pipeline_skips_viz_when_generate_viz_false()

### Community 69 - "Community 69"
Cohesion: 0.25
Nodes (4): ExeVision AI - Production Pipeline Module, Pose Extraction Module - Extracts MediaPipe pose landmarks from videos with qual, Temporal Segmentation Module - Biomechanically-sound squat phase detection Fait, View Classification Module - Rule-based camera angle classification Faithful re

### Community 70 - "Community 70"
Cohesion: 0.33
Nodes (4): Segments exercise motion into phases and repetitions using biomechanical FSM, Extract individual repetitions from phase labels, Create a Repetition object with phase breakdown, TemporalSegmenter

### Community 71 - "Community 71"
Cohesion: 0.33
Nodes (4): Segments exercise motion into phases and repetitions using biomechanical FSM, Extract individual repetitions from phase labels, Create a Repetition object with phase breakdown, TemporalSegmenter

### Community 72 - "Community 72"
Cohesion: 0.29
Nodes (4): Extract phase breakdown for a single rep with transition reasons, Single phase within a repetition, Convert phase ID to name, RepPhase

### Community 73 - "Community 73"
Cohesion: 0.4
Nodes (4): compute_improvement_percentage(), get_improvement_tier(), Rep-to-rep comparison utilities for progress messaging., Compare two reps and return aggregate and per-metric improvement metadata.

### Community 74 - "Community 74"
Cohesion: 0.33
Nodes (3): ExeVision AI - Production Pipeline Module, Pose Extraction Module - Extracts MediaPipe pose landmarks from videos with qual, View Classification Module - Rule-based camera angle classification Faithful re

### Community 76 - "Community 76"
Cohesion: 0.7
Nodes (4): mae(), main(), parse_args(), pearson()

### Community 77 - "Community 77"
Cohesion: 0.67
Nodes (3): analyze_annotations(), main(), Annotation quality self-check for ExeVision.  After finishing annotations, run t

### Community 78 - "Community 78"
Cohesion: 0.5
Nodes (3): FeedbackItemBuilder, _load_json(), Helper class for building and scoring feedback items.

### Community 79 - "Community 79"
Cohesion: 0.5
Nodes (4): _build_feedback_fallback(), Map numeric score to feedback tier labels expected by the frontend., Create a schema-compatible fallback when template/config files are unavailable., _tier_for_score()

### Community 81 - "Community 81"
Cohesion: 0.5
Nodes (3): plot_video_signals(), Plot segmentation signals for parameter tuning, Create diagnostic plot for a video

### Community 82 - "Community 82"
Cohesion: 0.67
Nodes (3): analyze_annotations(), main(), Annotation quality self-check for ExeVision.  After finishing annotations, run t

### Community 83 - "Community 83"
Cohesion: 0.67
Nodes (3): FeedbackItem, Individual feedback item with score for color-coding., TypedDict

## Knowledge Gaps
- **611 isolated node(s):** `Create mock MediaPipe keypoints for testing.`, `Test that control signal extraction works for both exercises.`, `Test that exercise-specific thresholds are returned correctly.`, `Test that BiomechanicalAnalyzer accepts exercise parameter.`, `Test that debug mode can be enabled.` (+606 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **22 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SquatPhase` connect `Community 30` to `Community 33`, `Community 70`, `Community 14`, `Community 53`, `Community 26`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **Why does `SquatPhase` connect `Community 36` to `Community 54`, `Community 69`, `Community 14`, `Community 71`?**
  _High betweenness centrality (0.013) - this node is a cross-community bridge._
- **Why does `BiomechanicalAnalyzer` connect `Community 25` to `Community 67`, `Community 27`, `Community 12`, `Community 47`?**
  _High betweenness centrality (0.011) - this node is a cross-community bridge._
- **What connects `Create mock MediaPipe keypoints for testing.`, `Test that control signal extraction works for both exercises.`, `Test that exercise-specific thresholds are returned correctly.` to the rest of the system?**
  _611 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._