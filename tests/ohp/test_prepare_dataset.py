import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "core" / "exevision" / "training" / "ohp"))
sys.path.insert(0, str(_REPO / "core" / "exevision" / "training"))


@pytest.fixture
def fake_workspace(tmp_path):
    """Build a minimal fake ohp_phase2 workspace for one video 'test_001'."""
    vid = "test_001"
    fps = 30.0
    total_frames = 90

    feat_dir = tmp_path / "overhead_press" / "extracted_features_clean" / "raw_unfiltered"
    feat_dir.mkdir(parents=True)
    feat_path = feat_dir / f"{vid}.json"
    feat_path.write_text(json.dumps({
        "info": {"fps": fps, "view": "front", "calibration": {"body_scale": 0.22, "standing_hip_height": 0.57}},
        "keypoints_img": [[[0.5, 0.5, 0.0, 1.0]] * 33] * total_frames,
    }))

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

    return tmp_path, vid, fps, total_frames


@pytest.fixture
def fake_labels_dir(tmp_path):
    labels = tmp_path / "Labels"
    labels.mkdir()
    # error_elbows.json no longer used — kept on disk only as a no-op file
    (labels / "error_knees.json").write_text(json.dumps({"test_001": []}))
    splits = tmp_path / "Splits"
    splits.mkdir()
    (splits / "train_keys.json").write_text(json.dumps(["test_001"]))
    (splits / "val_keys.json").write_text(json.dumps([]))
    (splits / "test_keys.json").write_text(json.dumps([]))
    return tmp_path


def test_prepare_writes_standing_only(fake_workspace, fake_labels_dir, tmp_path):
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

    ohp_path = out_dir / f"{vid}.json"
    seated_path = out_dir / f"{vid}_seated.json"
    assert ohp_path.exists(), "Standing OHP annotation not written"
    assert not seated_path.exists(), "Seated annotation should not be generated in Phase 2 v2"


def test_annotation_schema(fake_workspace, fake_labels_dir, tmp_path):
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
    assert "knee_error" in rep
    assert "elbow_error_soft" not in rep, "elbow_error_soft should be removed"
    assert "knee_error_soft" not in rep, "knee_error_soft should be renamed to knee_error"
    assert 0.0 <= rep["human_score"] <= 100.0
    assert rep["knee_error"] in (0.0, 1.0)


def test_knee_error_binary(fake_workspace, fake_labels_dir, tmp_path):
    ws_root, vid, _, _ = fake_workspace
    # Give knee error overlapping the rep
    knee_path = fake_labels_dir / "Labels" / "error_knees.json"
    knee_path.write_text(json.dumps({"test_001": [[0.5, 1.5]]}))

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
    assert data["reps"][0]["knee_error"] == 1.0


def test_missing_segmented_fallback(fake_workspace, fake_labels_dir, tmp_path):
    ws_root, vid, _, _ = fake_workspace
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
    assert data["reps"][0]["start_frame"] == 0
