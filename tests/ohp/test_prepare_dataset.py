import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Ensure module paths resolve
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "core" / "exevision" / "training" / "ohp"))
sys.path.insert(0, str(_REPO / "core" / "exevision" / "training"))


@pytest.fixture
def fake_workspace(tmp_path):
    """Build a minimal fake ohp_phase2 workspace for one video 'test_001'."""
    vid = "test_001"
    fps = 30.0
    total_frames = 90

    # features JSON (mimics Stage 2.5 output structure)
    feat_dir = tmp_path / "overhead_press" / "extracted_features_clean" / "raw_unfiltered"
    feat_dir.mkdir(parents=True)
    feat_path = feat_dir / f"{vid}.json"
    feat_path.write_text(json.dumps({
        "info": {"fps": fps, "view": "front", "calibration": {"body_scale": 0.22, "standing_hip_height": 0.57}},
        "keypoints_img": [[[0.5, 0.5, 0.0, 1.0]] * 33] * total_frames,
    }))

    # segmented JSON (1 rep covering full video)
    seg_dir = tmp_path / "overhead_press" / "segmented_reps" / "raw_unfiltered"
    seg_dir.mkdir(parents=True)
    seg_path = seg_dir / f"{vid}_segmented.json"
    seg_path.write_text(json.dumps({
        "info": {"fps": fps},
        "signals": {
            "normalized_hip_displacement": [0.5] * total_frames,
            "window_velocity": [0.0] * total_frames,
            "knee_angles": [160.0] * total_frames,
            "landmark_confidence": [0.99] * total_frames,
        },
        "repetitions": [{"rep_id": 1, "start_frame": 0, "end_frame": total_frames - 1}],
    }))

    # scoring JSON
    score_dir = tmp_path / "overhead_press" / "aqa_analysis_simple" / "raw_unfiltered" / vid
    score_dir.mkdir(parents=True)
    score_path = score_dir / f"{vid}_aqa_simple.json"
    score_path.write_text(json.dumps({
        "reps": [{
            "rep_id": 1,
            "overall_score": 70.0,
            "metric_scores": {"grip_ratio": 80.0, "rom": 75.0, "lockout": 90.0, "elbow_flare": 85.0},
            "flags": {"incomplete_lockout": False, "elbow_flare": False, "forward_lean": False,
                      "bar_drift": False, "wrist_deviation": False, "knee_instability": False},
        }],
    }))

    # seated features (just needs to exist with the right path)
    seated_dir = tmp_path / "seated_overhead_press" / "extracted_features_clean" / "raw_unfiltered"
    seated_dir.mkdir(parents=True)
    (seated_dir / f"{vid}.json").write_text(feat_path.read_text())

    return tmp_path, vid, fps, total_frames


@pytest.fixture
def fake_labels_dir(tmp_path):
    labels = tmp_path / "Labels"
    labels.mkdir()
    (labels / "error_elbows.json").write_text(json.dumps({
        "test_001": [[0.5, 1.5]],   # 1 sec overlap in a 3-sec rep → 1/3
    }))
    (labels / "error_knees.json").write_text(json.dumps({
        "test_001": [],
    }))
    splits = tmp_path / "Splits"
    splits.mkdir()
    (splits / "train_keys.json").write_text(json.dumps(["test_001"]))
    (splits / "val_keys.json").write_text(json.dumps([]))
    (splits / "test_keys.json").write_text(json.dumps([]))
    return tmp_path


def test_prepare_writes_both_variants(fake_workspace, fake_labels_dir, tmp_path):
    ws_root, vid, fps, total_frames = fake_workspace
    out_dir = tmp_path / "annotations"
    out_dir.mkdir()

    from prepare_dataset import run_preparation
    run_preparation(
        workspace=ws_root,
        labels_dir=fake_labels_dir / "Labels",
        splits_dir=fake_labels_dir / "Splits",
        output_dir=out_dir,
    )

    ohp_path = out_dir / f"{vid}.json"
    seated_path = out_dir / f"{vid}_seated.json"
    assert ohp_path.exists(), "OHP annotation not written"
    assert seated_path.exists(), "Seated OHP annotation not written"


def test_ohp_annotation_schema(fake_workspace, fake_labels_dir, tmp_path):
    ws_root, vid, _, _ = fake_workspace
    out_dir = tmp_path / "annotations"
    out_dir.mkdir()

    from prepare_dataset import run_preparation
    run_preparation(
        workspace=ws_root,
        labels_dir=fake_labels_dir / "Labels",
        splits_dir=fake_labels_dir / "Splits",
        output_dir=out_dir,
    )

    data = json.loads((out_dir / f"{vid}.json").read_text())
    assert data["exercise"] == "overhead_press"
    assert data["fitnessaqa_split"] == "train"
    assert data["annotation_source"] == "fitnessaqa_derived"
    assert len(data["reps"]) == 1

    rep = data["reps"][0]
    assert "human_score" in rep
    assert "elbow_error_soft" in rep
    assert "knee_error_soft" in rep
    assert 0.0 <= rep["human_score"] <= 100.0
    assert 0.0 <= rep["elbow_error_soft"] <= 1.0


def test_seated_always_zero_knee(fake_workspace, fake_labels_dir, tmp_path):
    ws_root, vid, _, _ = fake_workspace
    out_dir = tmp_path / "annotations"
    out_dir.mkdir()

    # Give seated a knee error — should still be 0.0
    knee_path = fake_labels_dir / "Labels" / "error_knees.json"
    knee_path.write_text(json.dumps({"test_001": [[0.0, 3.0]]}))

    from prepare_dataset import run_preparation
    run_preparation(
        workspace=ws_root,
        labels_dir=fake_labels_dir / "Labels",
        splits_dir=fake_labels_dir / "Splits",
        output_dir=out_dir,
    )

    seated = json.loads((out_dir / f"{vid}_seated.json").read_text())
    for rep in seated["reps"]:
        assert rep["knee_error_soft"] == 0.0


def test_missing_segmented_fallback(fake_workspace, fake_labels_dir, tmp_path):
    ws_root, vid, fps, total_frames = fake_workspace
    # Remove segmented JSON to trigger fallback
    (ws_root / "overhead_press" / "segmented_reps" / "raw_unfiltered" / f"{vid}_segmented.json").unlink()

    out_dir = tmp_path / "annotations"
    out_dir.mkdir()

    from prepare_dataset import run_preparation
    run_preparation(
        workspace=ws_root,
        labels_dir=fake_labels_dir / "Labels",
        splits_dir=fake_labels_dir / "Splits",
        output_dir=out_dir,
    )

    data = json.loads((out_dir / f"{vid}.json").read_text())
    assert len(data["reps"]) == 1
    rep = data["reps"][0]
    assert rep["start_frame"] == 0
