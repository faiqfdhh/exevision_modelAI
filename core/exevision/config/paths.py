from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_root: Path
    workspace_root: Path
    logs_root: Path
    stage_outputs_root: Path



def resolve_run_paths(storage_root: Path, run_id: str) -> RunPaths:
    run_root = storage_root / run_id
    return RunPaths(
        run_root=run_root,
        workspace_root=run_root / "workspace",
        logs_root=run_root / "logs",
        stage_outputs_root=run_root / "stage_outputs",
    )
