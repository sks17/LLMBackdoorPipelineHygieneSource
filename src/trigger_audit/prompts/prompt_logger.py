"""Final-prompt logging: persist the exact model-visible prompt for debugging and inspection.

Saving the final prompt (at least for a sample of trials) is non-negotiable: the project
cannot be debugged without seeing the actual input the model received. Sampling is
deterministic per trial id so a given run logs a stable, reproducible subset.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Any

from trigger_audit.io.paths import ensure_dir

PathLike = str | Path


class PromptLogger:
    """Writes final prompt text (and optionally the four logged layers) for sampled trials."""

    def __init__(
        self,
        directory: PathLike,
        *,
        sample_rate: float = 1.0,
        write_layers: bool = False,
    ) -> None:
        self._dir = Path(directory)
        self._sample_rate = sample_rate
        self._write_layers = write_layers

    def should_log(self, trial_id: str) -> bool:
        """Deterministically decide whether this trial is in the logged sample."""
        if self._sample_rate >= 1.0:
            return True
        if self._sample_rate <= 0.0:
            return False
        bucket = (zlib.crc32(trial_id.encode("utf-8")) % 10_000) / 10_000
        return bucket < self._sample_rate

    def log_final_prompt(
        self,
        trial_id: str,
        text: str,
        *,
        layers: dict[str, Any] | None = None,
    ) -> Path | None:
        """Write the final prompt for a sampled trial; return its path, or None if not sampled."""
        if not self.should_log(trial_id):
            return None
        ensure_dir(self._dir)
        path = self._dir / f"{trial_id}.txt"
        path.write_text(text, encoding="utf-8")
        if self._write_layers and layers is not None:
            (self._dir / f"{trial_id}.layers.json").write_text(
                json.dumps(layers, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return path
