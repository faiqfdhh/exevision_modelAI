"""Populate heuristic_score + heuristic_metric_scores into OHP Phase 3 annotations
from existing AQA simple JSONs.

Searches all quality-tier subdirs under the given aqa-root for {video_id}_aqa_simple.json.
Matches reps by rep_id. Only overwrites reps where heuristic_score is 0 or missing.

Usage:
    python core/exevision/analysis/populate_heuristic_scores.py
    python core/exevision/analysis/populate_heuristic_scores.py --dry-run
    python core/exevision/analysis/populate_heuristic_scores.py \\
        --aqa-root "D:/FitnessAQA/ohp_phase3/personal_videos/overhead_press/aqa_analysis_simple"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_REPO      = Path(__file__).resolve().parents[3]
_ANNO_DIR  = _REPO / "training_dataset" / "ohp_phase3_annotations" / "videos"
_AQA_ROOT  = Path("D:/FitnessAQA/ohp_phase3/personal_videos/overhead_press/aqa_analysis_simple")


def _build_aqa_index(aqa_root: Path) -> dict[str, Path]:
    """Scan all subdirs under aqa_root for *_aqa_simple.json, keyed by video_id."""
    index: dict[str, Path] = {}
    for p in aqa_root.rglob("*_aqa_simple.json"):
        # Filename pattern: {video_id}_aqa_simple.json
        video_id = p.name.replace("_aqa_simple.json", "")
        index[video_id] = p
    return index


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def populate(anno_dir: Path, aqa_root: Path, dry_run: bool) -> None:
    if not aqa_root.exists():
        raise RuntimeError(f"AQA root not found: {aqa_root}")

    aqa_index = _build_aqa_index(aqa_root)
    print(f"AQA index: {len(aqa_index)} files found under {aqa_root}")

    anno_paths = sorted(anno_dir.glob("*.json"))
    print(f"Annotation files: {len(anno_paths)}")

    stats = {"populated": 0, "already_set": 0, "aqa_missing": 0, "rep_missing": 0}

    for anno_path in anno_paths:
        anno = _load(anno_path)
        video_id = anno.get("video_id", anno_path.stem)
        reps = anno.get("reps") or []

        aqa_path = aqa_index.get(video_id)
        if aqa_path is None:
            if any(r.get("heuristic_score") in (None, 0, 0.0) for r in reps if r.get("human_score") is not None):
                print(f"  [MISSING AQA] {video_id}")
                stats["aqa_missing"] += 1
            continue

        aqa = _load(aqa_path)
        # Build rep_id → AQA rep lookup (rep_id may be int or str)
        aqa_reps: dict[str, dict] = {}
        for aqa_rep in (aqa.get("repetitions") or []):
            rid = str(aqa_rep.get("rep_id", ""))
            aqa_reps[rid] = aqa_rep

        changed = False
        for rep in reps:
            if rep.get("human_score") is None:
                continue
            h = rep.get("heuristic_score")
            if h is not None and float(h) != 0.0:
                stats["already_set"] += 1
                continue  # already populated

            rid = str(rep.get("rep_id", ""))
            aqa_rep = aqa_reps.get(rid)
            if aqa_rep is None:
                # Try index 0 if single-rep video
                if len(aqa_reps) == 1:
                    aqa_rep = next(iter(aqa_reps.values()))
                else:
                    print(f"  [REP MISSING] {video_id} rep_id={rid}")
                    stats["rep_missing"] += 1
                    continue

            score_block = aqa_rep.get("score") or {}
            overall = score_block.get("overall_score")
            if overall is None:
                overall = aqa.get("overall_score")  # fallback to top-level
            if overall is None:
                continue

            metric_scores = score_block.get("metric_scores") or {}

            rep["heuristic_score"] = float(overall)
            rep["heuristic_metric_scores"] = {k: float(v) for k, v in metric_scores.items() if v is not None}
            stats["populated"] += 1
            changed = True

        if changed and not dry_run:
            anno_path.write_text(json.dumps(anno, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nResults:")
    print(f"  Populated:    {stats['populated']} reps")
    print(f"  Already set:  {stats['already_set']} reps (skipped)")
    print(f"  AQA missing:  {stats['aqa_missing']} videos (no AQA file found)")
    print(f"  Rep missing:  {stats['rep_missing']} reps (AQA file found but rep_id not matched)")
    if dry_run:
        print("\n[DRY RUN] No files written.")
    else:
        print(f"\nDone. Re-run stamp_phase3_splits.py if needed, then re-train.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-dir", type=Path, default=_ANNO_DIR)
    parser.add_argument("--aqa-root", type=Path, default=_AQA_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    populate(args.anno_dir, args.aqa_root, args.dry_run)


if __name__ == "__main__":
    main()
