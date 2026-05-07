import sys
from pathlib import Path

import torch

_NEURAL = Path(__file__).resolve().parents[2] / "core" / "exevision" / "neural"
_TRAIN = Path(__file__).resolve().parents[2] / "core" / "exevision" / "training"
for _p in [str(_NEURAL), str(_NEURAL / "ohp"), str(_TRAIN)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nn_utils import (
    NUM_OHP_BILSTM_CHANNELS, NUM_OHP_ACTIVE_JOINTS,
    FIXED_SEQ_LEN, STGCN_CHANNELS,
    build_adjacency_matrix_ohp,
)
from core.exevision.neural.ohp.models import OHPBiLSTMScorer, OHPSTGCNScorer


BATCH = 4


def test_bilstm_input_channels():
    # OHP BiLSTM uses 8 channels (4 original + 4 new OHP-specific arm/asymmetry signals)
    assert NUM_OHP_BILSTM_CHANNELS == 8


def test_bilstm_output_keys():
    model = OHPBiLSTMScorer()
    x = torch.zeros(BATCH, FIXED_SEQ_LEN, NUM_OHP_BILSTM_CHANNELS)
    out = model(x)
    assert "embedding" in out
    assert "quality" in out
    assert "knee_error" in out
    assert "elbow_error" not in out


def test_bilstm_output_shapes():
    model = OHPBiLSTMScorer()
    x = torch.zeros(BATCH, FIXED_SEQ_LEN, NUM_OHP_BILSTM_CHANNELS)
    out = model(x)
    assert out["quality"].shape == (BATCH,)
    assert out["knee_error"].shape == (BATCH,)


def test_bilstm_quality_range():
    model = OHPBiLSTMScorer()
    x = torch.randn(BATCH, FIXED_SEQ_LEN, NUM_OHP_BILSTM_CHANNELS)
    out = model(x)
    assert out["quality"].min() >= 0.0
    assert out["quality"].max() <= 100.0


def test_bilstm_knee_error_range():
    model = OHPBiLSTMScorer()
    x = torch.randn(BATCH, FIXED_SEQ_LEN, NUM_OHP_BILSTM_CHANNELS)
    out = model(x)
    assert out["knee_error"].min() >= 0.0
    assert out["knee_error"].max() <= 1.0


def test_stgcn_joint_count():
    # OHP ST-GCN uses 10 joints (shoulders, elbows, wrists, hips, knees)
    assert NUM_OHP_ACTIVE_JOINTS == 10


def test_stgcn_output_keys():
    A = torch.tensor(build_adjacency_matrix_ohp())
    model = OHPSTGCNScorer(A)
    x = torch.zeros(BATCH, STGCN_CHANNELS, FIXED_SEQ_LEN, NUM_OHP_ACTIVE_JOINTS)
    out = model(x)
    assert "embedding" in out
    assert "quality" in out
    assert "knee_error" in out
    assert "elbow_error" not in out


def test_stgcn_quality_range():
    A = torch.tensor(build_adjacency_matrix_ohp())
    model = OHPSTGCNScorer(A)
    x = torch.randn(BATCH, STGCN_CHANNELS, FIXED_SEQ_LEN, NUM_OHP_ACTIVE_JOINTS)
    out = model(x)
    assert out["quality"].min() >= 0.0
    assert out["quality"].max() <= 100.0


def test_adjacency_matrix_shape():
    A = build_adjacency_matrix_ohp()
    assert A.shape == (10, 10)
