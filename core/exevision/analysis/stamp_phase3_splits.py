"""Stamp fitnessaqa_split='train'/'val'/'test' into OHP Phase 3 annotation JSONs.

Stratifies at VIDEO level by quality score bucket — exactly matching squat's
stratified_video_split() in core/exevision/training/squat/finetune_models.py.

Bucket edges: [20, 40, 60, 80, 100] (5 buckets)
Per bucket:
  >=3 videos → 70% train, 15% val, 15% test (min 1 each)
  2 videos   → 1 train, 0 val, 1 test
  1 video    → 1 train, 0 val, 0 test

Usage:
    python core/exevision/analysis/stamp_phase3_splits.py
    python core/exevision/analysis/stamp_phase3_splits.py --dry-run
    python core/exevision/analysis/stamp_phase3_splits.py --seed 123
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

_REPO      = Path(__file__).resolve().parents[3]
_ANNO_DIR  = _REPO / "training_dataset" / "ohp_phase3_annotations" / "videos"

BUCKET_EDGES = [20.0, 40.0, 60.0, 80.0, 100.0]


def _quality_bucket(score: float) -> int:
    for i, edge in enumerate(BUCKET_EDGES):
        if score < edge:
            return i
    return len(BUCKET_EDGES) - 1


def _mean_human_score(anno: dict) -> float | None:
    scores = [
        float(r["human_score"])
        for r in (anno.get("reps") or [])
        if r.get("human_score") is not None
    ]
    return sum(scores) / len(scores) if scores else None


def stamp_splits(anno_dir: Path, seed: int, dry_run: bool) -> None:
    rng = random.Random(seed)

    all_paths = sorted(anno_dir.glob("*.json"))
    if not all_paths:
        raise RuntimeError(f"No annotation JSONs in {anno_dir}")

    # Build per-video records with mean quality score
    records: list[dict] = []
    for path in all_paths:
        try:
            anno = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            print(f"  [WARN] could not read {path.name} — skipping")
            continue
        mean_score = _mean_human_score(anno)
        if mean_score is None:
            continue
        n_reps = sum(1 for r in (anno.get("reps") or []) if r.get("human_score") is not None)
        records.append({
            "path":   path,
            "anno":   anno,
            "bucket": _quality_bucket(mean_score),
            "score":  mean_score,
            "n_reps": n_reps,
        })

    if not records:
        raise RuntimeError("No annotated videos found.")

    # Group by bucket
    buckets: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        buckets[rec["bucket"]].append(rec)

    # Assign splits per bucket (squat methodology)
    for b, vids in buckets.items():
        rng.shuffle(vids)
        n = len(vids)
        if n >= 3:
            n_val  = max(1, math.floor(n * 0.15))
            n_test = max(1, math.floor(n * 0.15))
            n_train = n - n_val - n_test
            # Ensure at least 1 in train
            if n_train < 1:
                n_val = max(0, n_val - 1)
                n_train = n - n_val - n_test
        elif n == 2:
            n_train, n_val, n_test = 1, 0, 1
        else:  # n == 1
            n_train, n_val, n_test = 1, 0, 0

        for i, vid in enumerate(vids):
            if i < n_train:
                vid["split"] = "train"
            elif i < n_train + n_val:
                vid["split"] = "val"
            else:
                vid["split"] = "test"

    # Summary
    split_counts = Counter(r["split"] for r in records)
    rep_counts   = Counter()
    for r in records:
        rep_counts[r["split"]] += r["n_reps"]

    print(f"\nSplit summary (seed={seed}):")
    for s in ("train", "val", "test"):
        print(f"  {s}: {split_counts[s]} videos, {rep_counts[s]} reps")

    print("\nPer-bucket breakdown:")
    for b in sorted(buckets):
        lo = 0 if b == 0 else BUCKET_EDGES[b - 1]
        hi = BUCKET_EDGES[b]
        bc = Counter(r["split"] for r in buckets[b])
        print(f"  [{int(lo)}-{int(hi)}): {len(buckets[b])} videos  "
              f"train={bc.get('train',0)} val={bc.get('val',0)} test={bc.get('test',0)}")

    if dry_run:
        print("\n[DRY RUN] No files written. Re-run without --dry-run to stamp.")
        return

    # Write split field in-place
    for rec in records:
        anno = rec["anno"]
        anno["fitnessaqa_split"] = rec["split"]
        rec["path"].write_text(json.dumps(anno, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nStamped {len(records)} files.")
    print("Next steps:")
    print("  1. python core/exevision/training/ohp/finetune_ohp.py --annotation-dir ... --pretrain-bilstm ... --pretrain-stgcn ... --output-dir models/")
    print("  2. python core/exevision/training/ohp/evaluate_ohp.py --annotation-dir ... --model-dir models/ --output results/ohp_eval.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stamp 3-way quality-stratified splits into OHP Phase 3 annotations")
    parser.add_argument("--anno-dir", type=Path, default=_ANNO_DIR)
    parser.add_argument("--seed",     type=int,  default=42)
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()
    stamp_splits(args.anno_dir, args.seed, args.dry_run)


if __name__ == "__main__":
    main()
