"""
Annotation quality self-check for ExeVision.

After finishing annotations, run this script to verify your annotations
have enough signal for neural model training.

Usage:
    python scripts/analyze_annotations.py [path_to_annotations.json]

If no path given, defaults to dataset/neural_training/human_annotations.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "dataset" / "neural_training" / "human_annotations.json"


def analyze_annotations(annotations_path: Path) -> None:
    if not annotations_path.exists():
        print(f"Error: {annotations_path} not found.")
        sys.exit(1)

    with open(annotations_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Filter out skipped entries
    data = [a for a in raw if not a.get("skipped", False)]
    skipped = len(raw) - len(data)

    if not data:
        print("No annotations found (all skipped).")
        return

    human = np.array([a["human_score"] for a in data])
    heuristic = np.array([a["heuristic_score"] for a in data])
    disagreement = human - heuristic

    print("=" * 50)
    print("  ExeVision Annotation Quality Report")
    print("=" * 50)
    print(f"  Total annotated:  {len(data)}")
    print(f"  Skipped:          {skipped}")
    print()
    print(f"  Human score range:   {human.min():.0f} – {human.max():.0f}")
    print(f"  Human score mean:    {human.mean():.1f} ± {human.std():.1f}")
    print(f"  Heuristic mean:      {heuristic.mean():.1f} ± {heuristic.std():.1f}")
    print()
    print(f"  Mean disagreement:   {disagreement.mean():+.1f} pts")
    print(f"  Disagreement std:    {disagreement.std():.1f} pts")
    print(f"  Max disagreement:    {np.abs(disagreement).max():.1f} pts")
    print()

    corr = np.corrcoef(human, heuristic)[0, 1]
    print(f"  Correlation (r):     {corr:.3f}")
    print()

    # Score distribution
    bins = [(0, 20), (20, 40), (40, 60), (60, 75), (75, 90), (90, 100)]
    print("  Human score distribution:")
    for lo, hi in bins:
        count = np.sum((human >= lo) & (human < hi + (1 if hi == 100 else 0)))
        bar = "█" * int(count)
        print(f"    {lo:3d}-{hi:3d}: {count:3d}  {bar}")
    print()

    # View distribution
    view_counts: dict[str, int] = defaultdict(int)
    for a in data:
        view_counts[a.get("view", "unknown")] += 1
    print("  View distribution:")
    for view, count in sorted(view_counts.items()):
        print(f"    {view:20s}: {count}")
    print()

    # Error distribution
    error_keys = [
        "insufficient_squat_depth", "lumbar_flexion", "knee_valgus_collapse",
        "excessive_anterior_trunk_lean", "non_vertical_bar_path",
        "insufficient_stance_width"
    ]
    error_counts = {k: 0 for k in error_keys}
    error_severity_sum = {k: 0 for k in error_keys}
    reps_with_errors = 0
    for a in data:
        flags = a.get("flags", {})
        severities = a.get("flag_severities", {})
        if flags:
            has_error = False
            for k in error_keys:
                if flags.get(k, False):
                    error_counts[k] += 1
                    error_severity_sum[k] += severities.get(k, 1) # default to 1 if missing
                    has_error = True
            if has_error:
                reps_with_errors += 1

    if reps_with_errors > 0:
        print(f"  Error distribution ({reps_with_errors} reps had labeled flags):")
        for k, c in error_counts.items():
            avg_sev = error_severity_sum[k] / c if c > 0 else 0
            sev_str = f" (avg sev: {avg_sev:.1f})" if c > 0 else ""
            print(f"    {k:30s}: {c:3d} ({100 * c / len(data):.0f}%){sev_str}")
        print()

    # Warnings
    print("-" * 50)
    issues = 0

    if disagreement.std() < 3.0:
        issues += 1
        print("  ⚠ WARNING: Very low disagreement variance (< 3 pts).")
        print("    Your scores closely match the heuristic.")
        print("    The neural model won't have much to learn.")
        print("    Try watching reps more holistically — assess rhythm/control.")
        print()

    if human.std() < 10.0:
        issues += 1
        print("  ⚠ WARNING: Low human score variance (< 10 pts).")
        print("    You might be scoring too narrowly (e.g., everything 60-80).")
        print("    Use the full 0-100 range more aggressively.")
        print()

    if corr > 0.95:
        issues += 1
        print("  ⚠ WARNING: Very high correlation with heuristic (> 0.95).")
        print("    Your scores are very similar to the rules.")
        print("    Focus more on temporal quality, not just positions.")
        print()

    if len(data) < 50:
        issues += 1
        print("  ⚠ WARNING: Low annotation count (< 50).")
        print("    Consider annotating more reps for better training signal.")
        print()

    if issues == 0:
        print("  ✅ No issues detected. Annotations look good for training!")
        print()
        print("  What you have:")
        print(f"    - Correlation: {corr:.3f} (target: 0.5-0.8)")
        print(f"    - Disagreement std: {disagreement.std():.1f} (target: 8-15)")
        print(f"    - Score range: {human.min():.0f}-{human.max():.0f}")

    print("=" * 50)


def main() -> None:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = DEFAULT_ANNOTATIONS

    analyze_annotations(path)


if __name__ == "__main__":
    main()
