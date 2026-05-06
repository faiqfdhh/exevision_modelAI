# Overhead Press Integration & Training Plan

**Created:** 2026-05-04
**Last updated:** 2026-05-04
**Branch:** `multiexercise`
**Status:** Plan finalized — ready for implementation
**Companion docs:**
- [`IMPLEMENTATION_HANDOFF.md`](./IMPLEMENTATION_HANDOFF.md) — generic multi-exercise plumbing (hardcode removal, Dockerfile, model paths)
- [`MULTIEXERCISE_ANALYSIS.md`](./MULTIEXERCISE_ANALYSIS.md) — architectural analysis of multi-exercise readiness

> **Read this if:** You are implementing or training the overhead press (OHP) exercise, OR resuming OHP work mid-stream in a new session.
>
> **Don't read this if:** You only need to wire generic multi-exercise support (use `IMPLEMENTATION_HANDOFF.md` instead).

---

## 0. TL;DR — What This Plan Adds

`IMPLEMENTATION_HANDOFF.md` removes hardcodes so the pipeline *can* run with `--exercise overhead_press`. **This plan tells you what needs to happen so OHP actually produces correct scores end-to-end**, including:

1. OHP-specific phase detection (wrist control signal vs. squat's hip control signal)
2. OHP-specific scoring metrics (shoulder elevation, elbow extension, bar path)
3. Conversion of the FitnessAQA dataset (~2,260 labeled videos) into our annotation format
4. A 3-phase training pipeline: pre-train → FitnessAQA fine-tune → human annotation fine-tune
5. Exercise-aware annotation tool (replacing hardcoded squat flags)

---

## 1. Modularity Contract — The "Exercise Interface"

> **Why this section exists:** Every new exercise after OHP should follow this interface. It is the single source of truth for what "complete exercise support" means.

### 1.1 What an Exercise Must Provide

For any exercise `<X>` to be considered "complete":

| Artifact | Path | Purpose |
|----------|------|---------|
| Config JSON | `core/exevision/config/exercises/<X>.json` | Score brackets, metric thresholds, issue groups, annotation flags |
| Phase detection branch | `core/exevision/stages/temporal_segmentation.py` `_get_control_signal()` | What landmark drives the FSM |
| Scoring branch | `core/exevision/stages/scoring.py` `_score_<X>()` | Per-rep metric computation |
| Pre-trained models | `models/bilstm_<X>.pt`, `stgcn_<X>.pt`, `fusion_<X>.pt` | Or graceful fallback if missing |
| Annotation flags | `<X>.json["annotation_flags"]` | What flags appear in the annotation tool |
| Field mapping | `<X>.json["field_mapping"]` | Bridges scoring output names → feedback config metric names |

### 1.2 What an Exercise Must NOT Touch

- `core/exevision/stages/extract_selected_features.py` — generic MediaPipe extraction, body-agnostic
- `core/exevision/stages/classify_views.py` — generic visibility heuristics, work for any standing/sitting human
- `core/exevision/feedback/engine.py` — config-driven, generic
- `apps/api/main.py` — already accepts `exercise` parameter, validates against config existence
- `apps/api/pipeline.py` — already threads `--exercise` to all stages

> If a change to OHP requires touching files in this list, **stop and reconsider** — it's a sign the work belongs in a config or stage branch, not in shared code.

### 1.3 Where Exercises Diverge (Branch Points)

```
temporal_segmentation.py
└── _get_control_signal(frames, exercise):
    ├── if exercise == "squat":           return hip_y_sequence
    ├── if exercise == "overhead_press":  return mean_wrist_y_sequence
    └── (future)                          return <new exercise signal>

scoring.py
└── _compute_metrics(rep_frames, exercise):
    ├── if exercise == "squat":           return _score_squat(...)
    ├── if exercise == "overhead_press":  return _score_overhead_press(...)
    └── (future)                          return _score_<new>(...)
```

Two and only two branch points. Everything else is config-driven.

---

## 2. OHP-Specific Stage Implementation

### 2.1 Temporal Segmentation (`temporal_segmentation.py`)

**Concept:** OHP is the **vertical inverse** of squat. In squat, hips drop (eccentric) then rise (concentric). In OHP, wrists rise (concentric) then drop (eccentric). The FSM and phase names are unchanged — only the tracked landmark and its sign convention differ.

**Image-coordinate sign reminder:** Y increases downward. So:
- Squat eccentric (hip drops) → hip_y *increases*
- OHP concentric (wrists rise) → wrist_y *decreases*

To keep the FSM's "downward velocity = eccentric" semantics, **invert** the wrist signal:
```python
control_signal = -mean_wrist_y  # so increasing = "movement toward end of rep"
```

**Implementation sketch (add near top of file, before phase detection):**

```python
# MediaPipe wrist indices
L_WRIST, R_WRIST = 15, 16

def _get_control_signal(frames, exercise: str) -> np.ndarray:
    """Returns the 1D signal whose velocity drives FSM transitions."""
    if exercise == "squat":
        return _hip_y_sequence(frames)  # existing logic, extracted into helper
    elif exercise == "overhead_press":
        return _wrist_elevation_sequence(frames)
    raise ValueError(f"No control signal defined for exercise={exercise!r}")

def _wrist_elevation_sequence(frames) -> np.ndarray:
    """For OHP: bar elevation. Returns signal where INCREASING = lowering bar (eccentric)."""
    signal = []
    for f in frames:
        lw = _lm(f, L_WRIST); rw = _lm(f, R_WRIST)
        if lw is None or rw is None:
            signal.append(np.nan)
            continue
        # Invert sign: image y is inverted, so -y = "bar height".
        # Increasing this signal = bar lowering = eccentric (matches squat semantics).
        signal.append((lw[1] + rw[1]) / 2.0)
    return _interpolate_nans(np.array(signal))
```

**Idle baseline calibration:** Squat assumes hips start near baseline. OHP starts with bar at shoulders. Use the *first 60 frames mean* of the control signal as the idle baseline — same calibration logic, just applied to the new signal.

**Phase reset condition:** Squat returns to IDLE when hips return near baseline AND knees are extended. OHP equivalent: wrists return near baseline AND elbow angle > extension threshold (~165°). This requires an OHP branch in the IDLE-return check (~line 130 area where `IDLE_RETURN_MARGIN` and `IDLE_KNEE_EXTENSION_THRESHOLD` are used).

### 2.2 Scoring (`scoring.py`)

**OHP metric definitions:**

| Metric | Computation | When measured |
|--------|------------|---------------|
| `shoulder_elevation` | Angle at shoulder formed by hip→shoulder→elbow vectors | Top of rep (peak elevation) |
| `elbow_extension` | Angle at elbow formed by shoulder→elbow→wrist | Top of rep (lockout frame) |
| `bar_path_deviation` | `max_x_drift / shoulder_width` across rep | Across full rep |
| `forward_lean` | Existing function — angle of hip→shoulder from vertical | Standing variant only |
| `wrist_alignment` | Angle at wrist formed by elbow→wrist→knuckle (or wrist X-deviation if no hand landmarks) | Top of rep |

**Variant detection (standing vs. sitting):**

```python
def _detect_ohp_variant(frames, sample_size: int = 30) -> str:
    """Auto-detect standing vs sitting via lower-body landmark stability."""
    sample = frames[:sample_size]
    hip_confs = [_conf(f, L_HIP) for f in sample] + [_conf(f, R_HIP) for f in sample]
    knee_confs = [_conf(f, L_KNEE) for f in sample] + [_conf(f, R_KNEE) for f in sample]
    mean_hip = np.mean([c for c in hip_confs if c > 0])
    mean_knee = np.mean([c for c in knee_confs if c > 0])
    if mean_knee < 0.4 or (mean_hip > 0.4 and mean_knee < 0.5):
        return "sitting"
    return "standing"
```

**Implementation pattern:**

```python
def _score_overhead_press(rep_frames, view, config) -> dict:
    variant = _detect_ohp_variant(rep_frames)

    metrics = {
        "shoulder_elevation": _peak_shoulder_elevation(rep_frames),
        "elbow_extension":    _peak_elbow_extension(rep_frames),
        "bar_path_deviation": _bar_path_deviation(rep_frames),
        "wrist_alignment":    _peak_wrist_alignment(rep_frames),
    }
    if variant == "standing":
        metrics["forward_lean"] = forward_lean_deg(_top_frame(rep_frames))

    metric_scores = _score_metrics_against_thresholds(metrics, config["metrics"])
    overall = _weighted_overall(metric_scores, view, variant)
    return {
        "variant": variant,
        "metrics": metrics,
        "metric_scores": metric_scores,
        "overall_score": overall,
    }
```

**Dispatch in scoring main:**
```python
if args.exercise == "squat":
    rep_result = _score_squat(rep_frames, view, config)
elif args.exercise == "overhead_press":
    rep_result = _score_overhead_press(rep_frames, view, config)
```

### 2.3 Config Updates (`overhead_press.json`)

Current file has the score brackets and issue groups. Add the missing pieces:

```json
"field_mapping": {
  "metrics_to_feedback": {
    "shoulder_elevation": "shoulder_elevation",
    "elbow_extension":    "elbow_extension",
    "bar_path_deviation": "bar_path_deviation",
    "forward_lean_deg":   "forward_lean",
    "wrist_alignment":    "wrist_alignment"
  }
},
"annotation_flags": {
  "incomplete_lockout": "Incomplete Lockout",
  "elbow_flare":        "Elbow Flare / Winging",
  "forward_lean":       "Excessive Layback",
  "bar_drift":          "Bar Path Drift",
  "wrist_deviation":    "Wrist Bent Back",
  "knee_instability":   "Knee Instability (standing only)"
},
"annotation_metrics": {
  "lockout":   "Lockout Quality",
  "bar_path":  "Bar Path Straightness",
  "smoothness":"Smoothness",
  "control":   "Control"
}
```

---

## 3. FitnessAQA Dataset Integration

### 3.1 Source Layout (Reference)

```
D:\FitnessAQA\Overhead Press\
├── Unlabeled_Dataset-OHP\Unlabeled_Dataset\
│   ├── videos\videos\          (5,490 .mp4 files, naming: {user_id}_{variant}.mp4)
│   ├── videos.zip              (1.72 GB)
│   ├── bar_trajectories_raw.zip (11 MB — 3D barbell coords; investigate format)
│   └── ReadMe.md.docx
└── Labeled_Dataset-OHP\Labeled_Dataset\
    ├── videos\videos\          (~2,260 labeled videos, subset of unlabeled)
    ├── Splits\
    │   ├── train_keys.json     (1,582 IDs)
    │   ├── val_keys.json       (339 IDs)
    │   └── test_keys.json      (339 IDs)
    └── Labels\
        ├── error_elbows.json   ({video_id: [[start_sec, end_sec], ...]})
        └── error_knees.json    (same shape)
```

### 3.2 Critical Insight: Multi-Rep Videos

Each FitnessAQA video contains **multiple reps** (errors span 0–16+ seconds, typical OHP rep is 2–4 seconds → 4–8 reps per video). Error windows are absolute timestamps, not per-rep labels. **You cannot use the labels until you've segmented reps.**

### 3.3 Conversion Pipeline

**Script to create:** `core/exevision/training/prepare_ohp_dataset.py`

**What it does (per video):**

1. Run Stage 2.5 (`extract_selected_features.py`) on the video → MediaPipe poses
2. Run Stage 4 (`classify_views.py`) → view label
3. Run Stage 5 (`temporal_segmentation.py --exercise overhead_press`) → rep boundaries (in seconds)
4. Run Stage 8 (`scoring.py --exercise overhead_press`) → heuristic scores
5. For each detected rep `[rep_start_sec, rep_end_sec]`:
   - Compute `elbow_overlap_ratio` = fraction of rep duration covered by `error_elbows.json[video_id]` windows
   - Compute `knee_overlap_ratio` = same for `error_knees.json[video_id]`
   - **Variant-aware:** if variant is sitting, set `knee_overlap_ratio = 0` (knee labels irrelevant for seated)
   - Derive score:
     ```
     error_score    = 100 × (1 − 0.65 × elbow_overlap_ratio − 0.35 × knee_overlap_ratio)
     final_score    = 0.7 × error_score + 0.3 × heuristic_score
     final_score    = clamp(final_score, 0, 100)
     ```
6. Write per-video annotation JSON to `training_dataset/annotations/videos/{video_id}.json` matching the **exact schema** of existing squat annotations (see `training_dataset/annotations/videos/25713_3.json` for reference). Add an `annotation_source: "fitnessaqa_derived"` field for provenance.

**Pre-populate flags for Phase 3 acceleration:**

```python
flags = {
    "incomplete_lockout": elbow_overlap_ratio > 0.3,
    "elbow_flare":        elbow_overlap_ratio > 0.5,
    "knee_instability":   knee_overlap_ratio > 0.3 and variant == "standing",
    # bar_drift, wrist_deviation, forward_lean: leave False — derived only from human review
}
```

### 3.4 Bar Trajectory Investigation (Optional but Valuable)

Before running step 5 above, **inspect** `bar_trajectories_raw.zip`:

```bash
unzip -l "D:/FitnessAQA/Overhead Press/Unlabeled_Dataset-OHP/Unlabeled_Dataset/bar_trajectories_raw.zip" | head
```

Most likely format: per-video CSV/JSON with `[frame_idx, x, y, z]` rows. If usable:
- Replace wrist-based `bar_path_deviation` with direct 3D X-drift normalized by lift height
- Add `bar_traj_path` to `pipeline_outputs` in the annotation JSON
- More accurate, view-invariant

If unusable (proprietary format, no docs): skip and fall back to wrist-derived bar path. Don't block the pipeline on this.

### 3.5 Splits Reuse

The FitnessAQA splits map directly to our training:

```python
# In finetune_models.py, add OHP split-loading branch
if args.exercise == "overhead_press" and args.use_fitnessaqa_splits:
    train_ids = json.load(open(SPLITS_DIR / "train_keys.json"))
    val_ids   = json.load(open(SPLITS_DIR / "val_keys.json"))
    test_ids  = json.load(open(SPLITS_DIR / "test_keys.json"))
else:
    # Existing stratified split logic
    ...
```

This is the single biggest win from FitnessAQA: **the evaluation methodology is pre-defined**. We don't have to invent train/val/test splits or worry about data leakage.

---

## 4. Three-Phase Training Pipeline

### 4.1 Phase 1 — Self-Supervised Pre-Training

**Scale:** ~5,490 unlabeled videos
**Goal:** Learn OHP-specific pose representations before any score supervision
**Why:** Squat skipped this (small dataset, ~147 reps); OHP has 30× more raw data, so we exploit it.

**Workflow:**

1. Run Stage 2.5 in batch on all 5,490 unlabeled videos:
   ```bash
   for video in /path/to/unlabeled/videos/*.mp4; do
       python core/exevision/stages/extract_selected_features.py \
           filtered --video-id "$(basename ${video%.mp4})" \
           --exercise overhead_press
   done
   ```
2. Train BiLSTM and ST-GCN with self-supervised objectives:
   - **BiLSTM**: next-frame pose prediction (autoregressive)
   - **ST-GCN**: masked joint reconstruction (mask 15% of joints, predict from neighbors)

**Output:** `models/bilstm_ohp_pretrain.pt`, `models/stgcn_ohp_pretrain.pt`

**New script needed:** `core/exevision/training/pretrain_models.py` (≈200 LOC; can be skipped initially if Phase 2 alone gives acceptable results — treat as optional Phase 1 enhancement).

> **Decision point:** If schedule is tight, **skip Phase 1** and go directly Phase 2 → 3. The 2,260 FitnessAQA labels alone may suffice. Phase 1 becomes a "if accuracy is poor, try this" lever.

### 4.2 Phase 2 — FitnessAQA Supervised Fine-Tuning

**Scale:** ~2,260 videos → estimated 6,000–8,000 reps
**Goal:** Train models on derived rep scores from FitnessAQA error windows
**Splits:** Use FitnessAQA `train_keys.json` / `val_keys.json` / `test_keys.json` directly

**Workflow:**

1. Run `prepare_ohp_dataset.py` (Section 3.3) on all 2,260 labeled videos → annotation JSONs in `training_dataset/annotations/videos/`
2. Fine-tune from Phase 1 weights (or from scratch if Phase 1 skipped):
   ```bash
   python core/exevision/training/finetune_models.py \
       --exercise overhead_press \
       --use-fitnessaqa-splits \
       --init-from-pretrain \  # if Phase 1 ran
       --output-suffix phase2
   ```
3. Evaluate on test split:
   ```bash
   python core/exevision/training/evaluate_model.py \
       --exercise overhead_press \
       --checkpoint-suffix phase2
   ```

**Output:** `models/bilstm_ohp_phase2.pt`, `stgcn_ohp_phase2.pt`, `fusion_ohp_phase2.pt`

**Acceptance criterion:** test MAE on derived scores < 15.0. If above, debug rep segmentation quality before proceeding to Phase 3 — bad rep boundaries propagate bad labels.

### 4.3 Phase 3 — Human Annotation Calibration

**Scale:** ~150–200 manually annotated reps
**Goal:** Calibrate Phase 2 model to human perception (FitnessAQA-derived scores are biased toward error presence/absence, not nuanced quality)

**Sample selection (`select_annotation_samples.py --exercise overhead_press`):**

Targeting strategy:
- 60% **uncertainty samples**: reps where Phase 2 predicted 40–70 (model is least confident)
- 20% **error type coverage**: mix of elbow-only, knee-only, both, neither
- 10% **variant coverage**: standing and sitting if dataset has any
- 10% **calibration anchors**: known-clean (FitnessAQA score 100) and known-bad (high error coverage)

**Annotation tool changes (Section 5).**

**Final fine-tuning:**
```bash
python core/exevision/training/finetune_models.py \
    --exercise overhead_press \
    --init-from models/bilstm_ohp_phase2.pt \
    --human-weight 3.0 \
    --output-suffix final  # produces bilstm_overhead_press.pt etc.
```

`--human-weight 3.0` triples the loss weight on human-annotated reps so they dominate calibration without being drowned out by FitnessAQA-derived majority.

**Output:** `models/bilstm_overhead_press.pt`, `stgcn_overhead_press.pt`, `fusion_overhead_press.pt` (production names).

---

## 5. Annotation Tool Refactor (Exercise-Aware)

**Current state (`apps/desktop-ui/app.py:2082`):** `flag_defs` list is hardcoded with squat flags.

**Target state:** Flag list rebuilt from `<exercise>.json["annotation_flags"]` whenever exercise changes.

### 5.1 Refactor Steps

1. **Add exercise selector to annotation tab** (the inference tab already has one — replicate the pattern)
2. **Replace hardcoded `flag_defs` with config load:**
   ```python
   def _build_annotation_flag_defs(self) -> list[tuple[str, str]]:
       cfg_path = CONFIG_DIR / "exercises" / f"{self._annotation_exercise}.json"
       cfg = json.load(open(cfg_path))
       return list(cfg.get("annotation_flags", {}).items())
   ```
3. **Rebuild flag UI on exercise change:**
   - Destroy existing flag widgets in `flags_frame`
   - Recreate from `_build_annotation_flag_defs()`
4. **Same pattern for `annotation_metrics`** (the per-metric score sliders)
5. **FitnessAQA pre-population (OHP only):**
   - On video load, check if `video_id` has entries in cached FitnessAQA error JSONs
   - If yes, pre-tick relevant flags using the rules in Section 3.3
   - Annotator reviews and overrides — bias-blind heuristic-hidden behavior unchanged

### 5.2 Bias-Blind Compatibility

The existing bias-blind design (heuristic score hidden until human submits) is preserved. Pre-populated flags are *suggestions from FitnessAQA labels*, not from our model — they don't bias the human toward our pipeline's errors.

---

## 6. File-by-File Build Order

> **Use this as the handoff checklist.** Each row is independently testable. Tick off as you go.

| # | File | Change | Test Command |
|---|------|--------|--------------|
| 1 | `core/exevision/config/exercises/overhead_press.json` | Add `field_mapping`, `annotation_flags`, `annotation_metrics` | `python -c "import json; json.load(open('core/exevision/config/exercises/overhead_press.json'))"` |
| 2 | `core/exevision/stages/temporal_segmentation.py` | Add `_get_control_signal(exercise)` and `_wrist_elevation_sequence`; branch in main FSM | Run on one OHP video, inspect `_segmented.json` for plausible reps |
| 3 | `core/exevision/stages/scoring.py` | Add `_score_overhead_press()`, `_detect_ohp_variant()`, OHP metric helpers | Run on one OHP video, inspect `aqa_simple.json` for OHP metric keys |
| 4 | `apps/api/pipeline.py` | Verify `result.feedback` works for OHP (likely already works via config) | `curl POST /infer` with OHP video |
| 5 | (optional) Inspect `bar_trajectories_raw.zip` | Decide: use 3D coords or wrist landmarks | Manual |
| 6 | `core/exevision/training/prepare_ohp_dataset.py` | NEW — FitnessAQA → annotation JSONs | Run on 5 videos, inspect output JSONs |
| 7 | Run script on full FitnessAQA labeled set | Generate ~2,260 annotation JSONs | `ls training_dataset/annotations/videos/ \| grep -c "_"` |
| 8 | `core/exevision/training/finetune_models.py` | Add `--use-fitnessaqa-splits` and `--human-weight` flags | Dry-run on 50 reps |
| 9 | (optional Phase 1) `core/exevision/training/pretrain_models.py` | NEW — self-supervised pre-training | Skip unless Phase 2 accuracy poor |
| 10 | Run Phase 2 fine-tuning | Generate `*_ohp_phase2.pt` checkpoints | `evaluate_model.py` test MAE < 15 |
| 11 | `apps/desktop-ui/app.py` | Refactor annotation flags to config-driven; add exercise selector to annotation tab | Launch UI, switch exercise, verify flags rebuild |
| 12 | `core/exevision/analysis/select_annotation_samples.py` | Add `--exercise` parameter; uncertainty-targeting strategy | Dry-run, inspect selected sample list |
| 13 | Human annotation session | ~150–200 OHP reps annotated | `ls training_dataset/annotations/videos/{ohp_ids}.json` |
| 14 | Run Phase 3 fine-tuning | Generate production `*_overhead_press.pt` checkpoints | `evaluate_model.py` test MAE on human holdout < 12 |
| 15 | End-to-end smoke test | API call with OHP video → returns scores + feedback | `curl POST /infer -d '{"exercise":"overhead_press"...}'` |

---

## 7. Resume Instructions (For Mid-Stream Handoff)

> **If you're picking this up in a new session**, read this section first.

### 7.1 Where Am I?

Check progress by inspecting:
```bash
# Are config + stage changes done?
grep -c "_score_overhead_press\|overhead_press" core/exevision/stages/scoring.py
grep -c "overhead_press" core/exevision/stages/temporal_segmentation.py

# Has dataset prep run?
ls training_dataset/annotations/videos/ | grep -E "^[0-9]{5,}_[0-9]+\.json$" | wc -l
# Squat annotations are ~147; if count is much higher, OHP prep has run

# Are OHP models trained?
ls models/bilstm_ohp* models/stgcn_ohp* models/fusion_ohp* 2>/dev/null
ls models/bilstm_overhead_press.pt 2>/dev/null  # final production model

# Has annotation tool been refactored?
grep "annotation_flags" apps/desktop-ui/app.py
```

### 7.2 Continue Where You Left Off

Map progress to the build order in Section 6:

| If you have... | Resume at... |
|---|---|
| Only `overhead_press.json` updates | Step 2 (temporal segmentation) |
| Stages 2 and 3 done, no `prepare_ohp_dataset.py` | Step 6 |
| `prepare_ohp_dataset.py` exists, no annotation JSONs | Step 7 |
| Annotation JSONs exist, no Phase 2 models | Step 8 |
| Phase 2 models exist, no annotation tool refactor | Step 11 |
| Annotation tool refactored, no human annotations | Step 13 |
| Human annotations done, no Phase 3 models | Step 14 |

### 7.3 Verify Plan Is Still Valid

Before resuming, run:
```bash
git log --oneline -20  # see what's been committed
git status              # see uncommitted in-progress work
cat CHANGELOG.md | head -50  # what was the last documented change
```

If the codebase has materially diverged from the assumptions in Sections 1–5 (e.g., a refactor changed how stages are dispatched), **re-read this plan and update Section 1.3 (branch points) before continuing.**

---

## 8. What This Plan Deliberately Does NOT Cover

- **Sitting OHP as a separate exercise.** Handled via auto-detection inside `_score_overhead_press` (Section 2.2). Adding `seated_overhead_press.json` would duplicate work.
- **Real-time inference latency optimization.** Out of scope; handled by existing GCR deployment.
- **Web frontend changes.** None needed — the API contract is unchanged (just pass `"exercise": "overhead_press"`).
- **Re-training squat models.** Squat is frozen at its current state; this plan only adds OHP.
- **Multi-exercise workouts in a single video.** Not supported by current infrastructure; out of scope.

---

## 9. Decision Points & Open Questions

These need a human decision before they can be settled — flag them when resuming:

1. **Phase 1 (self-supervised pre-training): include or skip?**
   - Include if Phase 2 alone underperforms (test MAE > 15).
   - Skip for fastest path to a working OHP model.
   - **Recommendation:** Skip initially; revisit if Phase 2 accuracy is poor.

2. **`bar_trajectories_raw.zip`: integrate or ignore?**
   - Integrate if format is documented and parseable.
   - Ignore if proprietary/undocumented.
   - **Recommendation:** Spend 30 min inspecting; if unclear, fall back to wrist-derived bar path.

3. **Score derivation weights (`0.7 × error_score + 0.3 × heuristic_score`):**
   - These are starting estimates. Tune after Phase 2 by comparing to human annotations on a small validation subset.

4. **Sitting variant: auto-detect or explicit parameter?**
   - **Recommendation:** Auto-detect (Section 2.2). Keeps API surface unchanged.

---

**End of plan.** All other context (existing codebase patterns, hardcode locations, deployment) is in `IMPLEMENTATION_HANDOFF.md` and `MULTIEXERCISE_ANALYSIS.md`.
