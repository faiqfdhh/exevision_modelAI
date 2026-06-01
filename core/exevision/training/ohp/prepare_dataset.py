from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

_OHP_TRAIN = Path(__file__).resolve().parent
if str(_OHP_TRAIN) not in sys.path:
    sys.path.insert(0, str(_OHP_TRAIN))

from label_derivation import RepLabels, derive_rep_labels

_DEFAULT_HEURISTIC_SCORE = 50.0
_QUALITY_TIER = "raw_unfiltered"
_EXERCISE = "overhead_press"   # Phase 2 fine-tunes standing OHP only


# ---------------------------------------------------------------------------
# Workspace path helpers
# ---------------------------------------------------------------------------

def _features_path(workspace: Path, video_id: str) -> Path:
    return workspace / _EXERCISE / "extracted_features_clean" / _QUALITY_TIER / f"{video_id}.json"


def _segmented_path(workspace: Path, video_id: str) -> Path:
    return workspace / _EXERCISE / "segmented_reps" / _QUALITY_TIER / f"{video_id}_segmented.json"


def _scoring_path(workspace: Path, video_id: str) -> Path:
    return workspace / _EXERCISE / "aqa_analysis_simple" / _QUALITY_TIER / video_id / f"{video_id}_aqa_simple.json"


# ---------------------------------------------------------------------------
# JSON loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_label_windows(path: Path) -> Dict[str, List[List[float]]]:
    return dict(_load_json(path) or {})


def _load_splits(splits_dir: Path) -> Dict[str, str]:
    """Return {video_id: split_name} for all splits."""
    result: Dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        p = splits_dir / f"{split_name}_keys.json"
        if p.exists():
            for vid in (json.loads(p.read_text()) or []):
                result[str(vid)] = split_name
    return result


# ---------------------------------------------------------------------------
# Rep boundary extraction
# ---------------------------------------------------------------------------

def _extract_reps_from_segmented(seg_data: dict, fps: float) -> List[dict]:
    reps = seg_data.get("repetitions", []) or []
    out = []
    for rep in reps:
        sf = int(rep.get("start_frame", 0))
        ef = int(rep.get("end_frame", sf))
        out.append({"start_frame": sf, "end_frame": ef,
                    "start_sec": sf / fps, "end_sec": ef / fps})
    return out


def _whole_video_rep(feat_data: dict) -> List[dict]:
    kp = feat_data.get("keypoints_img", [])
    total_frames = len(kp) if kp else 1
    fps = float((feat_data.get("info") or {}).get("fps", 30.0))
    return [{"start_frame": 0, "end_frame": total_frames - 1,
             "start_sec": 0.0, "end_sec": (total_frames - 1) / fps}]


# ---------------------------------------------------------------------------
# Per-rep heuristic score extraction
# ---------------------------------------------------------------------------

def _rep_heuristic(scoring_data: Optional[dict], rep_id: int) -> tuple:
    """Return (overall_score, metric_scores_dict, flags_dict) for a rep."""
    _default_flags = {
        "incomplete_lockout": False, "elbow_flare": False, "forward_lean": False,
        "bar_drift": False, "wrist_deviation": False, "knee_instability": False,
    }
    _default_metrics = {"grip_ratio": 0.0, "rom": 0.0, "lockout": 0.0, "elbow_flare": 0.0}
    if scoring_data is None:
        return _DEFAULT_HEURISTIC_SCORE, _default_metrics, _default_flags
    for r in (scoring_data.get("reps") or []):
        if r.get("rep_id") == rep_id:
            return (
                float(r.get("overall_score", _DEFAULT_HEURISTIC_SCORE)),
                r.get("metric_scores") or _default_metrics,
                r.get("flags") or _default_flags,
            )
    return _DEFAULT_HEURISTIC_SCORE, _default_metrics, _default_flags


# ---------------------------------------------------------------------------
# Annotation JSON builder
# ---------------------------------------------------------------------------

def _build_annotation(
    video_id: str,
    workspace: Path,
    feat_data: dict,
    reps_boundaries: List[dict],
    scoring_data: Optional[dict],
    knee_windows: List[List[float]],
    split: str,
) -> dict:
    info = feat_data.get("info") or {}
    fps = float(info.get("fps", 30.0))
    view = str(info.get("view", "unknown"))
    calibration = info.get("calibration") or {}

    reps_out = []
    for i, bounds in enumerate(reps_boundaries):
        rep_id = i + 1
        h_score, h_metrics, h_flags = _rep_heuristic(scoring_data, rep_id)
        labels: RepLabels = derive_rep_labels(
            rep_start_sec=bounds["start_sec"],
            rep_end_sec=bounds["end_sec"],
            knee_windows=knee_windows,
            heuristic_score=h_score,
        )
        reps_out.append({
            "rep_id": rep_id,
            "start_frame": bounds["start_frame"],
            "end_frame": bounds["end_frame"],
            "start_sec": bounds["start_sec"],
            "end_sec": bounds["end_sec"],
            "human_score": labels.overall_score,
            "heuristic_score": h_score,
            "heuristic_metric_scores": h_metrics,
            "flags": h_flags,
            "knee_error": labels.knee_error,
            "annotation_source": "fitnessaqa_derived",
        })

    return {
        "video_id": video_id,
        "exercise": _EXERCISE,
        "pipeline_run": "ohp_phase2",
        "pipeline_outputs": {
            "features_json": str(_features_path(workspace, video_id)),
            "segmented_json": str(_segmented_path(workspace, video_id)),
            "scoring_json": str(_scoring_path(workspace, video_id)),
        },
        "view": view,
        "fps": fps,
        "calibration": calibration,
        "total_reps": len(reps_out),
        "annotation_source": "fitnessaqa_derived",
        "fitnessaqa_split": split,
        "annotated_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
        "reps": reps_out,
    }


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def run_preparation(
    workspace: Path,
    labels_dir: Path,
    splits_dir: Path,
    output_dir: Path,
) -> None:
    workspace = Path(workspace)
    labels_dir = Path(labels_dir)
    splits_dir = Path(splits_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    knee_labels = _load_label_windows(labels_dir / "error_knees.json")
    splits = _load_splits(splits_dir)

    written = skipped = 0
    for video_id, split in sorted(splits.items()):
        feat_path = _features_path(workspace, video_id)
        feat_data = _load_json(feat_path)
        if feat_data is None:
            print(f"  SKIP {video_id}: features JSON not found at {feat_path}")
            skipped += 1
            continue

        fps = float((feat_data.get("info") or {}).get("fps", 30.0))
        seg_data = _load_json(_segmented_path(workspace, video_id))
        reps = (
            _extract_reps_from_segmented(seg_data, fps)
            if seg_data and seg_data.get("repetitions")
            else _whole_video_rep(feat_data)
        )
        if not reps:
            reps = _whole_video_rep(feat_data)

        scoring_data = _load_json(_scoring_path(workspace, video_id))
        knee_windows = knee_labels.get(video_id, [])

        anno = _build_annotation(
            video_id, workspace, feat_data, reps,
            scoring_data, knee_windows, split,
        )
        out_path = output_dir / f"{video_id}.json"
        out_path.write_text(json.dumps(anno, indent=2), encoding="utf-8")

        written += 1

    print(f"\nDone. Written: {written} videos. Skipped: {skipped}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert FitnessAQA knee labels to standing OHP annotation JSONs"
    )
    parser.add_argument("--workspace", required=True, help="Path to ohp_phase2/workspace")
    parser.add_argument("--labels-dir", required=True, help="Path to Labeled_Dataset/Labels")
    parser.add_argument("--splits-dir", required=True, help="Path to Labeled_Dataset/Splits")
    parser.add_argument("--output-dir", required=True, help="Where to write annotation JSONs")
    args = parser.parse_args()
    run_preparation(
        workspace=Path(args.workspace),
        labels_dir=Path(args.labels_dir),
        splits_dir=Path(args.splits_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
