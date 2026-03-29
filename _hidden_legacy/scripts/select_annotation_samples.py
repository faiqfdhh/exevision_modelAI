"""
Strategic sample selection for ExeVision annotation.

Selects a subset of reps from pipeline runs for human annotation,
prioritizing reps near score boundaries, view diversity, score extremes,
and random fill.

Usage (standalone):
    python scripts/select_annotation_samples.py <queue_json_path> [--total 200] [--seed 42]

Also importable:
    from select_annotation_samples import select_samples_for_annotation
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def select_boundary_reps(all_reps: list[dict], n: int = 80) -> list[dict]:
    """Select reps near heuristic score decision boundaries."""
    bands = [(45, 55), (65, 75), (80, 90)]
    selected: list[dict] = []
    per_band = max(1, n // len(bands))

    for lo, hi in bands:
        band_reps = [r for r in all_reps if lo <= r.get("heuristic_score", 0) <= hi]
        random.shuffle(band_reps)
        selected.extend(band_reps[:per_band])

    return selected[:n]


def select_view_stratified(all_reps: list[dict], n: int = 60) -> list[dict]:
    """Equal representation from each view type."""
    by_view: dict[str, list[dict]] = defaultdict(list)
    for r in all_reps:
        by_view[r.get("view", "unknown")].append(r)

    selected: list[dict] = []
    per_view = max(1, n // max(len(by_view), 1))
    for view, reps in by_view.items():
        random.shuffle(reps)
        selected.extend(reps[:per_view])

    return selected[:n]


def select_extremes(all_reps: list[dict], n: int = 40) -> list[dict]:
    """Top and bottom of heuristic score range."""
    scored = sorted(all_reps, key=lambda r: r.get("heuristic_score", 0))
    half = max(1, n // 2)
    bottom = scored[:half]
    top = scored[-half:]
    return bottom + top


def select_samples_for_annotation(
    all_reps: list[dict],
    total: int = 200,
    seed: int = 42,
) -> list[dict]:
    """
    Select a strategic subset of reps for human annotation.
    Returns selected rep dicts.

    Handles pools smaller than `total` gracefully — just returns all available.
    """
    random.seed(seed)

    if len(all_reps) <= total:
        # Pool is small — annotate everything
        result = list(all_reps)
        random.shuffle(result)
        return result

    selected_ids: set[str] = set()
    selected: list[dict] = []

    def _add(reps: list[dict]) -> None:
        for r in reps:
            sid = r["sample_id"]
            if sid not in selected_ids:
                selected_ids.add(sid)
                selected.append(r)

    # Priority 1: Score boundary reps
    _add(select_boundary_reps(all_reps, n=80))

    # Priority 2: View-stratified (from remaining)
    remaining = [r for r in all_reps if r["sample_id"] not in selected_ids]
    _add(select_view_stratified(remaining, n=60))

    # Priority 3: Extremes (from remaining)
    remaining = [r for r in all_reps if r["sample_id"] not in selected_ids]
    _add(select_extremes(remaining, n=40))

    # Priority 4: Random fill
    remaining = [r for r in all_reps if r["sample_id"] not in selected_ids]
    random.shuffle(remaining)
    fill_count = total - len(selected)
    if fill_count > 0:
        _add(remaining[:fill_count])

    return selected[:total]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select annotation samples from a queue JSON.")
    parser.add_argument("queue_json", type=str, help="Path to annotation_queue.json (all reps)")
    parser.add_argument("--total", type=int, default=200, help="Target number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: overwrite input)")
    args = parser.parse_args()

    queue_path = Path(args.queue_json)
    if not queue_path.exists():
        print(f"Error: {queue_path} not found.")
        sys.exit(1)

    with open(queue_path, "r", encoding="utf-8") as f:
        all_reps = json.load(f)

    print(f"Loaded {len(all_reps)} total reps.")
    selected = select_samples_for_annotation(all_reps, total=args.total, seed=args.seed)
    print(f"Selected {len(selected)} reps for annotation.")

    # Summary by priority
    score_dist = defaultdict(int)
    for r in selected:
        s = r.get("heuristic_score", 0)
        if s < 30:
            score_dist["0-30"] += 1
        elif s < 55:
            score_dist["30-55"] += 1
        elif s < 75:
            score_dist["55-75"] += 1
        elif s < 90:
            score_dist["75-90"] += 1
        else:
            score_dist["90-100"] += 1
    print(f"Score distribution: {dict(score_dist)}")

    view_dist = defaultdict(int)
    for r in selected:
        view_dist[r.get("view", "unknown")] += 1
    print(f"View distribution: {dict(view_dist)}")

    out_path = Path(args.output) if args.output else queue_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)
    print(f"Saved selected samples to {out_path}")


if __name__ == "__main__":
    main()
