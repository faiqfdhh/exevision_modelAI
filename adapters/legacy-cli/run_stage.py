from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGE_SCRIPT_MAP = {
    "extract": PROJECT_ROOT / "scripts" / "2.5_extract_selected_features.py",
    "classify": PROJECT_ROOT / "scripts" / "4_classify_views.py",
    "segment": PROJECT_ROOT / "scripts" / "5_temporal_segmentation.py",
    "score": PROJECT_ROOT / "scripts" / "8_scoring.py",
    "neural": PROJECT_ROOT / "scripts" / "9_neural_fusion_inference.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Legacy-compatible stage launcher")
    parser.add_argument("stage", choices=sorted(STAGE_SCRIPT_MAP.keys()))
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to stage script")
    ns = parser.parse_args()

    script = STAGE_SCRIPT_MAP[ns.stage]
    if not script.exists():
        print(f"Stage script not found: {script}", file=sys.stderr)
        return 2

    cmd = [sys.executable, str(script), *ns.args]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
