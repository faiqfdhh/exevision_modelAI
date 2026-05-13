"""
OHP neural inference — legacy module.

All runtime inference logic has been moved to
``core/exevision/stages/neural_fusion_inference.py`` which now uses the
exercise-handler pattern from ``core/exevision/neural/registry.py``.

This module is kept for backward compatibility.  New code should import
directly from the modules below.
"""

from ohp.fusion import build_ohp_fusion
from ohp.heuristic_vec import build_ohp_heuristic_vector
from ohp.models import OHPBiLSTMScorer, OHPSTGCNScorer

