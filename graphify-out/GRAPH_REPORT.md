# Graph Report - exevision_modelAI  (2026-05-05)

## Corpus Check
- 153 files · ~6,065,687 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1858 nodes · 3688 edges · 120 communities (98 shown, 22 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 268 edges (avg confidence: 0.65)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `fb68ffac`
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
Nodes (16): AnnotationToolUI, Rep annotation tool for ExeVision neural training dataset.      Workflow:, Sorts the current video_files list by heuristic score (descending)., Search pipeline_ui_runs/ for a completed run that has         segmented + scored, Build annotation payload from a pipeline run for any target video., Build and save annotation for the currently loaded video from pipeline output., Build and save annotation metadata for any processed video during batch runs., Run Stages 2.5→4→5→8 for a single video, then load results. (+8 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (45): assign_training_weights(), build_records(), evaluate_bilstm(), evaluate_fusion(), evaluate_stgcn(), get_device(), infer_video_score(), load_json() (+37 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (56): applyFeetApartError(), applyPositionTightnessError(), board_end(), calculate_distance_from_platform_for_one_frame(), draw_two_coord(), find_which_side_board_on(), get_splash_from_one_frame(), get_splash_pred_mask() (+48 more)

### Community 4 - "Community 4"
Cohesion: 0.07
Nodes (11): _build_stages(), get_view_thresholds(), main(), ordered_stages(), PipelineRunnerUI, Find neural fusion scoring JSON for a video., Update STAGES when exercise selection changes., Build stage definitions with exercise-specific paths. (+3 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (8): get_view_thresholds(), main(), ordered_stages(), PipelineRunnerUI, Start full pipeline with neural fusion scoring (stages 2.5 → 4 → 5 → 8 → neural_, Find neural fusion scoring JSON for a video., score_severity(), Stage

### Community 6 - "Community 6"
Cohesion: 0.05
Nodes (50): _already_processed_json_exists(), analyze_visibility_trends(), apply_one_euro_filter(), apply_savgol_filter(), build_mandatory_chain_flags(), _build_paths(), consolidate_foot_region(), create_visualization_report() (+42 more)

### Community 7 - "Community 7"
Cohesion: 0.36
Nodes (12): build_rep_summary(), build_video_summary(), detect_base_dir(), evaluate_metric(), format_value(), get_view_thresholds(), iter_score_files(), main() (+4 more)

### Community 8 - "Community 8"
Cohesion: 0.07
Nodes (38): _already_processed_json_exists(), analyze_visibility_trends(), apply_one_euro_filter(), apply_savgol_filter(), build_mandatory_chain_flags(), consolidate_foot_region(), create_visualization_report(), draw_landmarks_enhanced() (+30 more)

### Community 9 - "Community 9"
Cohesion: 0.1
Nodes (36): _angle_2d(), _basic_phases_from_hip(), _build_scoring_paths(), calculate_torso_tibia_offset(), calculate_vertical_depth(), _conf(), _fallback_single_json(), find_all_video_ids() (+28 more)

### Community 10 - "Community 10"
Cohesion: 0.29
Nodes (19): analyze_common_errors(), analyze_correlations(), analyze_depth_by_view(), analyze_metric_distributions(), analyze_score_distribution(), analyze_zero_depth_issues(), generate_summary_report(), load_all_results() (+11 more)

### Community 11 - "Community 11"
Cohesion: 0.11
Nodes (32): _angle_2d(), _basic_phases_from_hip(), calculate_torso_tibia_offset(), calculate_vertical_depth(), _conf(), find_all_video_ids(), find_feature_json(), find_segmented_json() (+24 more)

### Community 12 - "Community 12"
Cohesion: 0.11
Nodes (32): Squat Analysis Microprograms  Low-level functions for error detection and squa, aggregate_phase_errors(), calculate_angle(), calculate_angle_3d(), forward_lean_error(), get_all_errors_for_frame(), get_confidence(), get_landmark() (+24 more)

### Community 13 - "Community 13"
Cohesion: 0.09
Nodes (29): _build_temporal_paths(), create_segmentation_visualization(), _debug_enabled(), _env_flag(), find_video_file(), _get_control_signal(), _hip_y_sequence(), process_video() (+21 more)

### Community 14 - "Community 14"
Cohesion: 0.1
Nodes (10): BiLSTMScorer, HeuristicGuidedFusion, Heuristic-anchored fusion where neural branches propose a bounded residual corre, Returns:             final_score: clamped prediction in [0, 100]             r, Fine-tuned BiLSTM for temporal quality scoring., Fine-tuned ST-GCN for spatial quality scoring., STGCNScorer, MultiModalRepDataset (+2 more)

### Community 15 - "Community 15"
Cohesion: 0.13
Nodes (26): classify_view(), Classify camera view from keypoints.          Replicates logic from 4_classify, ViewType, analyze_rep_errors(), aqa_metaprogram_squat(), extract_repetitions_from_phases(), find_feature_json(), find_segmented_json() (+18 more)

### Community 16 - "Community 16"
Cohesion: 0.14
Nodes (25): build_distribution_data(), compute_percentile_score(), compute_rep_score(), compute_set_score(), load_distribution_data(), scoring_functions.py Squat quality scoring functions.  Converts error measure, Score knee valgus measurement.          Args:         valgus_ratio: Knee spre, Score forward torso lean.          Args:         lean_angle: Forward lean in (+17 more)

### Community 17 - "Community 17"
Cohesion: 0.11
Nodes (15): pad_or_truncate(), Pad (with zeros) or truncate a sequence to fixed length.     Input: (T, ...) nu, apply_joint_masking(), build_dataloader(), _convert_rep_to_stgcn_tensor(), main(), MaskedRepDataset, parse_args() (+7 more)

### Community 18 - "Community 18"
Cohesion: 0.12
Nodes (13): apply_joint_masking(), build_dataloader(), main(), MaskedRepDataset, parse_args(), batch: (batch, 7, 128, 11)     Randomly mask joint positions across frames., _resolve_data_dir(), set_seed() (+5 more)

### Community 19 - "Community 19"
Cohesion: 0.12
Nodes (13): apply_joint_masking(), build_dataloader(), main(), MaskedRepDataset, parse_args(), batch: (batch, 7, 128, 11)     Randomly mask joint positions across frames., _resolve_data_dir(), set_seed() (+5 more)

### Community 20 - "Community 20"
Cohesion: 0.12
Nodes (13): apply_joint_masking(), build_dataloader(), main(), MaskedRepDataset, parse_args(), batch: (batch, 7, 128, 11)     Randomly mask joint positions across frames., _resolve_data_dir(), set_seed() (+5 more)

### Community 21 - "Community 21"
Cohesion: 0.18
Nodes (25): assign_training_weights(), build_records(), evaluate_bilstm(), evaluate_fusion(), evaluate_stgcn(), get_device(), infer_video_score(), load_json() (+17 more)

### Community 22 - "Community 22"
Cohesion: 0.1
Nodes (22): compute_knee_angle_signal(), detect_phase_simple(), get_hip_height(), get_knee_angle(), get_leg_length(), get_scale_factor(), get_view_label(), is_at_bottom() (+14 more)

### Community 23 - "Community 23"
Cohesion: 0.13
Nodes (19): ExtractionResult, PipelineConfig, Configuration and result dataclasses for the ExeVision Pipeline, Single phase within a repetition, Central configuration for the entire pipeline, Single squat repetition with phase breakdown, Result from temporal segmentation stage, Result from pose extraction stage (+11 more)

### Community 24 - "Community 24"
Cohesion: 0.12
Nodes (13): ExtractionResult, Result from pose extraction stage, PoseExtractor, Analyze visibility patterns across frames, Create a comprehensive text and visualization report, Extract pose landmarks from a video with quality assessment.          Args:, Internal video processing logic with quality assessment, Extract pose landmarks and save to JSON file.          Args:             vide (+5 more)

### Community 25 - "Community 25"
Cohesion: 0.14
Nodes (22): discover_videos(), extract_bilstm_rep(), get_device(), infer_rep(), _load_model_state(), _load_stgcn_with_compat(), main(), _normalize_state_dict_keys() (+14 more)

### Community 26 - "Community 26"
Cohesion: 0.16
Nodes (22): _as_path(), build_adjacency_matrix(), _build_feature_for_video(), _compute_velocity(), _discover_feature_index(), _discover_segmented_files(), _extract_active_joints(), _extract_rep_matrix() (+14 more)

### Community 27 - "Community 27"
Cohesion: 0.13
Nodes (16): Configuration and result dataclasses for the ExeVision Pipeline, Single phase within a repetition, Single squat repetition with phase breakdown, Result from temporal segmentation stage, Repetition, RepPhase, SegmentationResult, Temporal Segmentation Module - Biomechanically-sound squat phase detection Fait (+8 more)

### Community 28 - "Community 28"
Cohesion: 0.15
Nodes (11): BiomechanicalAnalyzer, Compute the primary control signal: normalized vertical hip displacement, Compute knee bending angle for all frames (secondary signal), Compute smoothed velocity of hip displacement, Compute velocity trends averaged over windows, Identify frames with sufficient landmark confidence, Calculate angle at p2 formed by p1-p2-p3, Fill NaN values with linear interpolation (+3 more)

### Community 29 - "Community 29"
Cohesion: 0.12
Nodes (13): BiomechanicalAnalyzer, Calculates angle-invariant and view-invariant biomechanical metrics.     Uses b, Compute knee bending angle for all frames (secondary signal), Compute smoothed velocity of hip displacement.         Positive velocity = movi, Calculates angle-invariant and view-invariant biomechanical metrics.     Uses b, Calculate angle at p2 formed by p1-p2-p3, Fill NaN values with linear interpolation, Compute the primary control signal based on exercise type.                  Fo (+5 more)

### Community 30 - "Community 30"
Cohesion: 0.12
Nodes (19): _fire_callback(), get_job(), health(), InferRequest, JobStatus, _normalize_stage_selection(), _pipeline_task(), ExeVision AI — FastAPI Inference Server  Wraps the squat analysis pipeline for (+11 more)

### Community 31 - "Community 31"
Cohesion: 0.15
Nodes (11): BiomechanicalAnalyzer, Compute the primary control signal: normalized vertical hip displacement, Compute knee bending angle for all frames (secondary signal), Compute smoothed velocity of hip displacement, Compute velocity trends averaged over windows, Identify frames with sufficient landmark confidence, Calculate angle at p2 formed by p1-p2-p3, Fill NaN values with linear interpolation (+3 more)

### Community 32 - "Community 32"
Cohesion: 0.12
Nodes (11): PoseExtractor, Analyze visibility patterns across frames, Create a comprehensive text and visualization report, Extract pose landmarks from a video with quality assessment.          Args:, Internal video processing logic with quality assessment, Extract pose landmarks and save to JSON file.          Args:             vide, Extracts pose landmarks from exercise videos using MediaPipe with quality assess, Check that the MediaPipe model exists (+3 more)

### Community 33 - "Community 33"
Cohesion: 0.19
Nodes (8): FeedbackEngine, _metric_phrase_tier(), Main feedback orchestrator using config + templates + deterministic rendering., Generate per-rep and session-level feedback from score data., Build win text for narrative and return items with metric tracking., Generate brief tier-appropriate items for metrics >= threshold that are not wins, Generate issue items with text and metric tracking., _resolve_issue_tone_mode()

### Community 34 - "Community 34"
Cohesion: 0.11
Nodes (14): Advanced temporal segmentation with biomechanical rigor.          Supports mul, Simple rep counting from phase sequence ONLY.          Counts 1 rep for:, Validate that the camera view is processable.         Allow all views including, Detect repetitions from phase sequences.         A rep = eccentric → [isometric, Extract phase breakdown for a single rep with transition reasons, Simple rep counting from phase sequence ONLY.          Counts 1 rep for:, Extract phase breakdown for a single rep with transition reasons, Single phase within a repetition (+6 more)

### Community 35 - "Community 35"
Cohesion: 0.14
Nodes (18): _build_features_dirs(), _classify_frame(), _face_score(), _facing_camera(), get_view_label(), get_view_label_with_probs(), _is_visible(), process_video_classification() (+10 more)

### Community 36 - "Community 36"
Cohesion: 0.11
Nodes (14): convert_to_serializable(), Convert phase ID to name, Main segmentation pipeline with comprehensive error handling, Recursively convert numpy types to Python native types for JSON serialization, Single squat repetition with phase breakdown, Extract anthropometric measurements from idle frames.         Now includes outl, Recursively convert numpy types to Python native types for JSON serialization, Single squat repetition with phase breakdown (+6 more)

### Community 37 - "Community 37"
Cohesion: 0.15
Nodes (11): Post-processing pass to merge short phases into neighbors.         Eliminates a, Simplified Machine for squat phase detection.          Logic:     1. Positive, Only allow IDLE once hips return near beginning-of-video standing position., Enforce strict legal adjacency globally.          Allowed high-level cycle:, Simplified phase detection looping over all frames.         Returns array of ph, Post-processing pass to merge short phases into neighbors.         Eliminates a, Simplified Machine for squat phase detection.          Logic:     1. Positive, Only allow IDLE once hips return near beginning-of-video standing position. (+3 more)

### Community 38 - "Community 38"
Cohesion: 0.21
Nodes (18): _as_path(), _build_feature_for_video(), _compute_velocity(), _discover_feature_index(), _discover_segmented_files(), _extract_active_joints(), _extract_rep_matrix(), _extract_stgcn_rep() (+10 more)

### Community 39 - "Community 39"
Cohesion: 0.17
Nodes (18): discover_videos(), get_device(), _load_model_state(), _load_stgcn_with_compat(), main(), _normalize_state_dict_keys(), parse_args(), process_video() (+10 more)

### Community 40 - "Community 40"
Cohesion: 0.16
Nodes (10): apply_masking(), BiLSTMPretrainer, build_dataloader(), main(), parse_args(), RepSignalDataset, _resolve_data_dir(), set_seed() (+2 more)

### Community 41 - "Community 41"
Cohesion: 0.16
Nodes (10): apply_masking(), BiLSTMPretrainer, build_dataloader(), main(), parse_args(), RepSignalDataset, _resolve_data_dir(), set_seed() (+2 more)

### Community 42 - "Community 42"
Cohesion: 0.13
Nodes (18): _build_feedback_fallback(), _build_stage_specs(), coerce_old_feedback_format(), _delete_input_video(), _prepare_workspace(), ExeVision AI — Pipeline Runner  Mirrors the stage execution logic from apps/de, Create workspace directory tree and copy the input video into it., Ensure each stage produced the expected artifact for the requested video. (+10 more)

### Community 43 - "Community 43"
Cohesion: 0.16
Nodes (6): Template rendering helpers for deterministic phrase selection and slot filling., Handles deterministic phrase selection and template slot replacement., TemplateRenderer, Tests for TemplateRenderer phrase selection and slot filling., TestPhraseSelection, TestSlotFilling

### Community 44 - "Community 44"
Cohesion: 0.15
Nodes (9): Compute metrics for a temporal window centered at given frame, Window-based Finite State Machine for squat phase detection, Main FSM loop - classifies each frame, Determine what phase the current window suggests, Handle state transitions with hysteresis and validation, Generate human-readable transition reason, Post-process to remove very short phases, Convert phase ID to string name (+1 more)

### Community 45 - "Community 45"
Cohesion: 0.18
Nodes (16): create_mock_keypoints(), main(), Test that BiomechanicalAnalyzer accepts exercise parameter., Test that debug mode can be enabled., Test that TemporalSegmenter accepts exercise parameter., Run all verification tests., Create mock MediaPipe keypoints for testing., Test that control signal extraction works for both exercises. (+8 more)

### Community 46 - "Community 46"
Cohesion: 0.15
Nodes (9): Compute metrics for a temporal window centered at given frame, Window-based Finite State Machine for squat phase detection, Main FSM loop - classifies each frame, Determine what phase the current window suggests, Handle state transitions with hysteresis and validation, Generate human-readable transition reason, Post-process to remove very short phases, Convert phase ID to string name (+1 more)

### Community 47 - "Community 47"
Cohesion: 0.16
Nodes (16): _classify_frame(), _face_score(), _facing_camera(), get_view_label(), get_view_label_with_probs(), _is_visible(), process_video_classification(), Classify camera view. Returns the view with the most frame votes. (+8 more)

### Community 48 - "Community 48"
Cohesion: 0.22
Nodes (6): Computes rep-over-rep deltas and improvement tiers., RepComparator, Tests for RepComparator rep-over-rep progress logic., TestImprovementTier, TestMetricImprovement, TestRepComparison

### Community 49 - "Community 49"
Cohesion: 0.14
Nodes (15): assign_item_scores(), detect_mismatch(), FeedbackItem, FeedbackResult, get_category_for_metric(), Feedback template engine orchestration for rep and session coaching., Individual feedback item with score for color-coding., Per-rep feedback payload. (+7 more)

### Community 50 - "Community 50"
Cohesion: 0.13
Nodes (15): _cleanup_workspace(), Remove heavy intermediate artifacts from the workspace after results are collect, Test visualization integration: generate_viz flag handling in the backend.  Te, When generate_viz=False, cleanup should delete visualization directories., Result schema should include visualization_url and visualization_available field, Callback payload should include visualization metadata for success and failure s, Verify that critical path calls to .exists() are all properly guarded., When generate_viz=True, Stage 2.5 and Stage 5 should generate visualizations. (+7 more)

### Community 51 - "Community 51"
Cohesion: 0.16
Nodes (14): Enum, convert_to_serializable(), create_segmentation_visualization(), find_video_file(), process_video(), Temporal Segmentation Module (5.4) Segments squat motion into idle/eccentric/co, Process a single video's keypoints. Returns (video_id, status, result, quality), Create annotated video with phase overlay, rep markers, and signal graphs (+6 more)

### Community 52 - "Community 52"
Cohesion: 0.17
Nodes (8): BiomechanicalAnalyzer, Calculates angle-invariant and view-invariant biomechanical metrics.     Uses b, Compute the primary control signal: normalized vertical hip displacement., Compute knee bending angle for all frames (secondary signal), Compute smoothed velocity of hip displacement.         Positive velocity = movi, Compute velocity trends averaged over windows.         This is the primary inpu, Calculate angle at p2 formed by p1-p2-p3, Fill NaN values with linear interpolation

### Community 53 - "Community 53"
Cohesion: 0.17
Nodes (10): PipelineConfig, PipelineResult, Central configuration for the entire pipeline, Complete pipeline result for a single video, Convert result to serializable dictionary, ExeVisionPipeline, main(), ExeVision AI - Main Pipeline Orchestrator ===================================== (+2 more)

### Community 54 - "Community 54"
Cohesion: 0.21
Nodes (10): Dataset, apply_masking(), BiLSTMPretrainer, build_dataloader(), main(), parse_args(), RepSignalDataset, _resolve_data_dir() (+2 more)

### Community 55 - "Community 55"
Cohesion: 0.24
Nodes (7): Builds aggregate metrics and trajectory across all reps., SessionAggregator, Tests for SessionAggregator session-level summary logic., TestAverageScore, TestMostImprovedMetric, TestPersistentIssue, TestTrajectoryDetection

### Community 56 - "Community 56"
Cohesion: 0.15
Nodes (14): _build_kinematic_data(), _build_phase_timeline(), collect_results(), _find_json(), _get_field_mapping(), Glob for a JSON file anywhere under base., Build a phase timeline for one rep, including idle phases inferred from gaps., Build ROM time-series for one rep using hip vertical displacement.      Return (+6 more)

### Community 57 - "Community 57"
Cohesion: 0.22
Nodes (6): Simplified Machine for squat phase detection.          Logic:     1. Positive, Only allow IDLE once hips return near beginning-of-video standing position., Enforce strict legal adjacency globally.          Allowed high-level cycle:, Simplified phase detection looping over all frames.         Returns array of ph, Post-processing pass to merge short phases into neighbors.         Eliminates a, SquatStateMachine

### Community 58 - "Community 58"
Cohesion: 0.18
Nodes (8): PipelineResult, Complete pipeline result for a single video, Convert result to serializable dictionary, ExeVisionPipeline, main(), ExeVision AI - Main Pipeline Orchestrator =====================================, Main pipeline that orchestrates all analysis stages, Process a video through the complete pipeline.

### Community 59 - "Community 59"
Cohesion: 0.25
Nodes (10): main(), Strategic sample selection for ExeVision annotation.  Selects a subset of reps f, Select reps near heuristic score decision boundaries., Equal representation from each view type., Top and bottom of heuristic score range., Select a strategic subset of reps for human annotation.     Returns selected rep, select_boundary_reps(), select_extremes() (+2 more)

### Community 60 - "Community 60"
Cohesion: 0.25
Nodes (7): Result from view classification stage, ViewResult, Classifies camera view using rule-based geometric analysis, Classify view directly from JSON data (for updating existing JSON files)., Classify the camera view from extracted pose landmarks using rule-based logic., Classify camera view from keypoints using geometric rules.         Returns: 'fr, ViewClassifier

### Community 61 - "Community 61"
Cohesion: 0.22
Nodes (7): Result from view classification stage, ViewResult, Classifies camera view using rule-based geometric analysis, Classify view directly from JSON data (for updating existing JSON files)., Classify the camera view from extracted pose landmarks using rule-based logic., Classify camera view from keypoints using geometric rules.         Returns: 'fr, ViewClassifier

### Community 62 - "Community 62"
Cohesion: 0.2
Nodes (6): Single squat repetition with phase breakdown, Extract anthropometric measurements from idle frames.         Now includes outl, Identify frames with sufficient landmark confidence, Main segmentation pipeline with comprehensive error handling, Detect repetitions from phase sequences.         A rep = eccentric → [isometric, Repetition

### Community 63 - "Community 63"
Cohesion: 0.25
Nodes (10): main(), Strategic sample selection for ExeVision annotation.  Selects a subset of reps f, Select reps near heuristic score decision boundaries., Equal representation from each view type., Top and bottom of heuristic score range., Select a strategic subset of reps for human annotation.     Returns selected rep, select_boundary_reps(), select_extremes() (+2 more)

### Community 64 - "Community 64"
Cohesion: 0.29
Nodes (5): QualityChecker, Detect mismatch between overall score and sub-metric breakdown., Tests for FeedbackEngine quality checks and full orchestration., TestFeedbackEngineIntegration, TestMismatchDetection

### Community 65 - "Community 65"
Cohesion: 0.36
Nodes (5): resolve_run_paths(), RunPaths, _env_bool(), ExeVisionSettings, load_settings()

### Community 66 - "Community 66"
Cohesion: 0.29
Nodes (8): compute_hip_displacement_signal(), count_reps(), find_rep_peaks(), Compute normalized hip displacement for all frames.     Primary signal for phas, Find frame indices of squat bottoms (peak hip displacement).          Args:, Count number of repetitions in squat data.          Args:         squat_data:, Segment squat into phases using simplified algorithm.          This is a light, segment_into_phases()

### Community 67 - "Community 67"
Cohesion: 0.36
Nodes (7): apply_safety_clamps(), build_heuristic_vector(), _safe_score(), extract_bilstm_rep(), infer_rep(), Extract BiLSTM sequence (128, 4) for one rep from segmentation data., Infer one rep.      Returns dict with neural_score, residual, and sub-metrics,

### Community 68 - "Community 68"
Cohesion: 0.25
Nodes (8): _build_stage_cmd(), _get_model_path(), Build the subprocess command for a stage, mirroring app.py arg construction., Run one pipeline stage; returns captured stdout+stderr., Construct exercise-specific model path, with fallback to generic names for compa, _run_stage(), When generate_viz=False, both Stage 2.5 and Stage 5 should skip visualization., test_pipeline_skips_viz_when_generate_viz_false()

### Community 69 - "Community 69"
Cohesion: 0.25
Nodes (4): ExeVision AI - Production Pipeline Module, Pose Extraction Module - Extracts MediaPipe pose landmarks from videos with qual, Temporal Segmentation Module - Biomechanically-sound squat phase detection Fait, View Classification Module - Rule-based camera angle classification Faithful re

### Community 70 - "Community 70"
Cohesion: 0.43
Nodes (6): build_adjacency_matrix(), Build the normalized adjacency matrix for the 11-joint skeleton graph.      Re, mae(), main(), parse_args(), pearson()

### Community 71 - "Community 71"
Cohesion: 0.33
Nodes (4): Simple rep counting from phase sequence ONLY.          Counts 1 rep for:, Advanced temporal segmentation with biomechanical rigor.          Pipeline:, Validate that the camera view is processable.         Allow all views including, TemporalSegmenter

### Community 72 - "Community 72"
Cohesion: 0.29
Nodes (4): Extract phase breakdown for a single rep with transition reasons, Single phase within a repetition, Convert phase ID to name, RepPhase

### Community 74 - "Community 74"
Cohesion: 0.4
Nodes (4): compute_improvement_percentage(), get_improvement_tier(), Rep-to-rep comparison utilities for progress messaging., Compare two reps and return aggregate and per-metric improvement metadata.

### Community 75 - "Community 75"
Cohesion: 0.33
Nodes (3): ExeVision AI - Production Pipeline Module, Pose Extraction Module - Extracts MediaPipe pose landmarks from videos with qual, View Classification Module - Rule-based camera angle classification Faithful re

### Community 76 - "Community 76"
Cohesion: 0.33
Nodes (5): Metrics computed over a temporal window, Metrics computed over a temporal window, Compute metrics for a temporal window centered at given frame, Compute metrics for a temporal window centered at given frame, WindowMetrics

### Community 77 - "Community 77"
Cohesion: 0.7
Nodes (4): mae(), main(), parse_args(), pearson()

### Community 78 - "Community 78"
Cohesion: 0.67
Nodes (3): analyze_annotations(), main(), Annotation quality self-check for ExeVision.  After finishing annotations, run t

### Community 79 - "Community 79"
Cohesion: 0.5
Nodes (3): FeedbackItemBuilder, _load_json(), Helper class for building and scoring feedback items.

### Community 81 - "Community 81"
Cohesion: 0.5
Nodes (3): plot_video_signals(), Plot segmentation signals for parameter tuning, Create diagnostic plot for a video

### Community 82 - "Community 82"
Cohesion: 0.5
Nodes (3): Metrics computed over a temporal window, Compute metrics for a temporal window centered at given frame, WindowMetrics

### Community 83 - "Community 83"
Cohesion: 0.67
Nodes (3): analyze_annotations(), main(), Annotation quality self-check for ExeVision.  After finishing annotations, run t

## Knowledge Gaps
- **536 isolated node(s):** `Create mock MediaPipe keypoints for testing.`, `Test that control signal extraction works for both exercises.`, `Test that exercise-specific thresholds are returned correctly.`, `Test that BiomechanicalAnalyzer accepts exercise parameter.`, `Test that debug mode can be enabled.` (+531 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **22 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SquatPhase` connect `Community 27` to `Community 24`, `Community 51`, `Community 44`, `Community 53`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **Why does `SquatPhase` connect `Community 23` to `Community 51`, `Community 69`, `Community 46`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._
- **Why does `MultiModalRepDataset` connect `Community 14` to `Community 70`, `Community 21`, `Community 54`?**
  _High betweenness centrality (0.012) - this node is a cross-community bridge._
- **What connects `Create mock MediaPipe keypoints for testing.`, `Test that control signal extraction works for both exercises.`, `Test that exercise-specific thresholds are returned correctly.` to the rest of the system?**
  _536 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.05 - nodes in this community are weakly interconnected._