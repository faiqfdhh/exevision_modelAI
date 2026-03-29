from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExeVisionSettings:
    project_root: Path
    storage_root: Path
    models_root: Path
    dataset_root: Path
    runs_root: Path
    use_neural: bool
    device: str



def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}



def load_settings() -> ExeVisionSettings:
    project_root = Path(os.environ.get("EXEVISION_PROJECT_ROOT", Path(__file__).resolve().parents[4]))
    storage_root = Path(os.environ.get("EXEVISION_STORAGE_ROOT", project_root / "pipeline_ui_runs"))
    models_root = Path(os.environ.get("EXEVISION_MODELS_ROOT", project_root / "models"))
    dataset_root = Path(os.environ.get("EXEVISION_DATASET_ROOT", project_root / "squat" / "dataset_videos_all"))
    runs_root = Path(os.environ.get("EXEVISION_RUNS_ROOT", storage_root))
    use_neural = _env_bool("EXEVISION_ENABLE_NEURAL", True)
    device = os.environ.get("EXEVISION_DEVICE", "auto")

    return ExeVisionSettings(
        project_root=project_root,
        storage_root=storage_root,
        models_root=models_root,
        dataset_root=dataset_root,
        runs_root=runs_root,
        use_neural=use_neural,
        device=device,
    )
