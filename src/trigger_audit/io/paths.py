"""Filesystem path resolution for experiment inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PathLike = str | Path


def ensure_dir(path: PathLike) -> Path:
    """Create a directory (and parents) if missing and return it as a Path."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


@dataclass(frozen=True)
class PathResolver:
    """Resolves the standard data/ and outputs/ subdirectory layout from a root.

    Relative ``data_dir`` / ``outputs_dir`` are resolved against ``root``. Each accessor that
    returns a file path ensures the parent directory exists, so callers can write directly.
    """

    root: Path = Path(".")
    data_dir: Path = Path("data")
    outputs_dir: Path = Path("outputs")

    def _data(self) -> Path:
        return self.data_dir if self.data_dir.is_absolute() else self.root / self.data_dir

    def _outputs(self) -> Path:
        return self.outputs_dir if self.outputs_dir.is_absolute() else self.root / self.outputs_dir

    # --- data inputs ---
    def raw_dir(self) -> Path:
        return ensure_dir(self._data() / "raw")

    def synthetic_dir(self) -> Path:
        return ensure_dir(self._data() / "synthetic")

    def triggers_path(self, name: str = "triggers.jsonl") -> Path:
        return ensure_dir(self._data() / "triggers") / name

    def manifest_path(self, name: str = "trial_manifest.jsonl") -> Path:
        return ensure_dir(self._data() / "manifests") / name

    def shard_path(self, name: str) -> Path:
        return ensure_dir(self._data() / "shards") / name

    # --- outputs ---
    def logs_dir(self) -> Path:
        return ensure_dir(self._outputs() / "logs")

    def final_prompts_dir(self) -> Path:
        return ensure_dir(self._outputs() / "final_prompts")

    def survival_results_path(self, name: str) -> Path:
        return ensure_dir(self._outputs() / "survival_results") / name

    def generation_results_path(self, name: str) -> Path:
        return ensure_dir(self._outputs() / "generation_results") / name

    def tables_dir(self) -> Path:
        return ensure_dir(self._outputs() / "tables")

    def figures_dir(self) -> Path:
        return ensure_dir(self._outputs() / "figures")

    def failure_examples_dir(self) -> Path:
        return ensure_dir(self._outputs() / "failure_examples")
