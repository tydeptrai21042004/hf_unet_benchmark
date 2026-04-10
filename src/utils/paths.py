from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp(fmt: str = "%Y%m%d_%H%M%S") -> str:
    return datetime.now().strftime(fmt)


@dataclass
class ExperimentPaths:
    root: Path
    checkpoints: Path
    logs: Path
    results: Path

    @classmethod
    def create(cls, root: str | Path, experiment_name: Optional[str] = None) -> "ExperimentPaths":
        root = Path(root)
        if experiment_name:
            root = root / experiment_name
        ckpt = ensure_dir(root / "checkpoints")
        logs = ensure_dir(root / "logs")
        results = ensure_dir(root / "results")
        return cls(root=root, checkpoints=ckpt, logs=logs, results=results)


__all__ = ["ensure_dir", "timestamp", "ExperimentPaths"]
