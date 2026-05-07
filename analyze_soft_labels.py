#!/usr/bin/env python3
"""Analyze soft label distribution in OHP annotation JSONs."""

import json
from pathlib import Path
import numpy as np

annotation_dir = Path(r"D:\FitnessAQA\ohp_phase2\workspace\annotations")
print(f"Loading annotations from: {annotation_dir}")
print()

elbow_errors = []
knee_errors = []
overall_scores = []
splits = {}
video_count = 0

# Parse all annotation JSONs
for json_file in sorted(annotation_dir.glob("*.json")):
    if json_file.name.startswith("_") or json_file.name == "index.json":
        continue

    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
        video_count += 1

        split = data.get("fitnessaqa_split")
        if split:
            splits[split] = splits.get(split, 0) + 1

        # Extract rep-level error labels
        for rep in data.get("reps", []):
            if "elbow_error_soft" in rep:
                elbow_errors.append(rep["elbow_error_soft"])
            if "knee_error_soft" in rep:
                knee_errors.append(rep["knee_error_soft"])
            if "human_score" in rep:
                overall_scores.append(rep["human_score"])
    except Exception as e:
        print(f"Error parsing {json_file.name}: {e}")

print(f"✓ Processed {video_count} videos, {len(elbow_errors)} total reps")
print()

# === ELBOW ERROR ANALYSIS ===
print("=" * 50)
print("ELBOW ERROR DISTRIBUTION")
print("=" * 50)
elbow_arr = np.array(elbow_errors)
print(f"Count:     {len(elbow_errors)}")
print(f"Mean:      {np.mean(elbow_arr):.4f}")
print(f"Std:       {np.std(elbow_arr):.4f}")
print(f"Min/Max:   {np.min(elbow_arr):.4f} / {np.max(elbow_arr):.4f}")
print(f"Median:    {np.median(elbow_arr):.4f}")

non_zero_elbow = sum(1 for x in elbow_errors if x > 0.001)
pct_elbow = 100 * non_zero_elbow / len(elbow_errors) if elbow_errors else 0
print(f"Non-zero:  {non_zero_elbow} / {len(elbow_errors)} = {pct_elbow:.1f}%")

# Percentile breakdown
if elbow_errors:
    p25, p50, p75, p90 = np.percentile(elbow_arr, [25, 50, 75, 90])
    print(f"Percentiles: 25%={p25:.3f}, 50%={p50:.3f}, 75%={p75:.3f}, 90%={p90:.3f}")

print()

# === KNEE ERROR ANALYSIS ===
print("=" * 50)
print("KNEE ERROR DISTRIBUTION")
print("=" * 50)
knee_arr = np.array(knee_errors)
print(f"Count:     {len(knee_errors)}")
print(f"Mean:      {np.mean(knee_arr):.4f}")
print(f"Std:       {np.std(knee_arr):.4f}")
print(f"Min/Max:   {np.min(knee_arr):.4f} / {np.max(knee_arr):.4f}")
print(f"Median:    {np.median(knee_arr):.4f}")

non_zero_knee = sum(1 for x in knee_errors if x > 0.001)
pct_knee = 100 * non_zero_knee / len(knee_errors) if knee_errors else 0
print(f"Non-zero:  {non_zero_knee} / {len(knee_errors)} = {pct_knee:.1f}%")

if knee_errors:
    p25, p50, p75, p90 = np.percentile(knee_arr, [25, 50, 75, 90])
    print(f"Percentiles: 25%={p25:.3f}, 50%={p50:.3f}, 75%={p75:.3f}, 90%={p90:.3f}")

print()

# === OVERALL SCORE ANALYSIS ===
print("=" * 50)
print("OVERALL SCORE DISTRIBUTION")
print("=" * 50)
if overall_scores:
    score_arr = np.array(overall_scores)
    print(f"Count:     {len(overall_scores)}")
    print(f"Mean:      {np.mean(score_arr):.1f}")
    print(f"Std:       {np.std(score_arr):.1f}")
    print(f"Min/Max:   {np.min(score_arr):.1f} / {np.max(score_arr):.1f}")

print()

# === SPLIT DISTRIBUTION ===
print("=" * 50)
print("SPLIT DISTRIBUTION")
print("=" * 50)
for split in sorted(splits.keys()):
    print(f"{split}: {splits[split]}")

print()

# === INTERPRETATION ===
print("=" * 50)
print("INTERPRETATION")
print("=" * 50)

if pct_elbow < 20:
    print("🔴 ELBOW: SEVERE class imbalance! <20% positive labels")
    print("   → Model learns to predict near-zero for everything → AUC ≈ 0.5")
elif pct_elbow < 40:
    print("🟡 ELBOW: Moderate class imbalance (20-40% positive)")
    print("   → Challenging but learnable if labels are clean")
else:
    print("🟢 ELBOW: Balanced class distribution")

print()

if pct_knee < 20:
    print("🔴 KNEE: SEVERE class imbalance! <20% positive labels")
    print("   → Model learns to predict near-zero for everything → AUC ≈ 0.5")
elif pct_knee < 40:
    print("🟡 KNEE: Moderate class imbalance (20-40% positive)")
    print("   → Challenging but learnable if labels are clean")
else:
    print("🟢 KNEE: Balanced class distribution")

print()
print("RECOMMENDATION:")
if pct_elbow < 20 or pct_knee < 20:
    print("1. Severe class imbalance detected → increase loss weights for error BCE")
    print("   Modify _LAMBDA_ELBOW and _LAMBDA_KNEE in finetune.py (current: 0.3 and 0.2)")
    print("   Suggested: 1.0 and 0.8 to force model to learn minority class")
    print()
    print("2. Consider using pos_weight in BCE loss for class imbalance handling")
    print()
    print("3. Verify FitnessAQA error window labels are correct (not all zeros)")
else:
    print("Class distribution appears reasonable.")
    print("Low AUC likely due to:")
    print("- Soft labels (overlap ratios) don't correlate with visual errors")
    print("- FitnessAQA error window annotations are noisy or mislabeled")
    print("- Model needs more training time or data")
