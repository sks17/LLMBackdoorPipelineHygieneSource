"""Build the trial manifest: expand the experiment grid and shard it for the cluster."""

from __future__ import annotations

import re
import zlib
from collections.abc import Iterator, Sequence
from itertools import product
from pathlib import Path

from trigger_audit.experiments.survivability_audit.config import SurvivabilityExperimentConfig
from trigger_audit.io.jsonl import write_jsonl
from trigger_audit.io.paths import PathResolver
from trigger_audit.schemas.trials import TrialSpec
from trigger_audit.util.ids import make_trial_id

_UNSAFE = re.compile(r"[^A-Za-z0-9_.\-]+")


def _safe_name(value: str) -> str:
    """Make a model id safe to use in a shard filename."""
    return _UNSAFE.sub("_", value)


class ManifestBuilder:
    """Expands a survivability config into TrialSpec rows and shards them by model.

    The builder is pure over its id lists (no store access), so it is easy to unit test. Trial
    ids are derived from the full tuple, so the same grid always yields the same manifest.
    """

    def __init__(
        self,
        config: SurvivabilityExperimentConfig,
        *,
        base_ids: Sequence[str],
        trigger_ids: Sequence[str],
        model_ids: Sequence[str],
    ) -> None:
        self._config = config
        self._base_ids = list(base_ids)
        self._trigger_ids = list(trigger_ids)
        self._model_ids = list(model_ids)

    def build(self) -> Iterator[TrialSpec]:
        """Yield one TrialSpec per point in the Cartesian product of the grid."""
        cfg = self._config
        for base_id, trigger_id, position, context_length, policy, model_id in product(
            self._base_ids,
            self._trigger_ids,
            cfg.trigger_positions,
            cfg.context_lengths,
            cfg.pipeline_policies,
            self._model_ids,
        ):
            chat_template = cfg.chat_templates.get(model_id)
            trial_id = make_trial_id(
                base_id=base_id,
                trigger_id=trigger_id,
                trigger_position=position.value,
                model_id=model_id,
                context_length=context_length,
                pipeline_policy=policy,
                chat_template=chat_template,
                seed=cfg.seed,
            )
            yield TrialSpec(
                trial_id=trial_id,
                base_id=base_id,
                trigger_id=trigger_id,
                trigger_position=position,
                model_id=model_id,
                context_length=context_length,
                pipeline_policy=policy,
                chat_template=chat_template,
                run_generation=self._select_generation(trial_id),
                seed=cfg.seed,
            )

    def build_list(self) -> list[TrialSpec]:
        """Materialize the full manifest as a list."""
        return list(self.build())

    def write_manifest(self, path: str | Path) -> int:
        """Write the full manifest to a single JSONL file and return the row count."""
        return write_jsonl(path, self.build())

    def shard(self, trials: Sequence[TrialSpec], resolver: PathResolver) -> list[Path]:
        """Group trials by model and write per-model shard files; return the shard paths.

        Sharding by model lets a generation worker load each model once (see the cluster plan).
        """
        by_model: dict[str, list[TrialSpec]] = {}
        for trial in trials:
            by_model.setdefault(trial.model_id, []).append(trial)

        paths: list[Path] = []
        size = max(1, self._config.shard_size)
        for model_id, model_trials in by_model.items():
            model_trials.sort(key=lambda t: t.trial_id)
            for index in range(0, len(model_trials), size):
                chunk = model_trials[index : index + size]
                name = f"{_safe_name(model_id)}_shard_{index // size:04d}.jsonl"
                path = resolver.shard_path(name)
                write_jsonl(path, chunk)
                paths.append(path)
        return paths

    def _select_generation(self, trial_id: str) -> bool:
        """Deterministically flag a trial for generation based on the configured fraction.

        TODO: replace fraction sampling with stratified selection (positive/negative controls,
        delivered/not-delivered, boundary cases) once survival results are available.
        """
        cfg = self._config
        if not cfg.run_generation or cfg.generation_fraction <= 0.0:
            return False
        if cfg.generation_fraction >= 1.0:
            return True
        bucket = (zlib.crc32(trial_id.encode("utf-8")) % 10_000) / 10_000
        return bucket < cfg.generation_fraction
