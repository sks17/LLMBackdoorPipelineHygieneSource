"""Grid expander for probe-detection experiments (Project 2, component F).

Turns a compact :class:`ProbeGridAxes` description into a concrete list of
:class:`~trigger_audit.experiments.probe_detection.config.ProbeDetectionExperimentConfig`
cells -- the Cartesian product of the axes -- with stable, content-derived experiment ids,
then serializes them to per-cell YAMLs a Slurm array can run without collisions. This is the
"changing parameters defines experiments" seam: a researcher instantiates any tier E0-E3
(``docs/PROJECT2_EXPERIMENT_PLAN.md`` Part III) by editing an axes YAML in ``configs/probe/``
and submitting an array, never by writing Python.

The module deliberately depends only on the config schema and the neutral ``generalization``
leaf (:class:`GeneralizationSpec` lives there, not here, so ``config -> grid -> config`` is not
a cycle); it re-exports :class:`GeneralizationSpec` and :func:`partition_by_metadata` for
convenience so callers can reach the whole grid surface from one module.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from trigger_audit.experiments.probe_detection.config import ProbeDetectionExperimentConfig
from trigger_audit.experiments.probe_detection.generalization import (
    GeneralizationSpec,
    partition_by_metadata,
)
from trigger_audit.schemas.probes import PoolingStrategy
from trigger_audit.util.ids import stable_id

__all__ = [
    "GeneralizationSpec",
    "ProbeGridAxes",
    "ProbeModelSpec",
    "expand_probe_grid",
    "partition_by_metadata",
    "write_probe_configs",
]

# Root under which every generated cell's outputs live: one subtree per experiment family and
# per experiment id, so a Slurm array writes results/activations/predictions with no collisions.
_GENERATED_RUN_ROOT = Path("outputs/probe_detection/generated_runs")


def _slug(value: str) -> str:
    """Lowercase, id-safe slug (keep ``[a-z0-9._-]``; collapse everything else to ``_``)."""
    return re.sub(r"[^a-z0-9._-]+", "_", value.lower()).strip("_") or "x"


class ProbeModelSpec(BaseModel):
    """One model axis point: a short label plus how to load it (or that it is offline).

    ``id`` is a short, filesystem-safe label embedded into every cell's ``experiment_id`` so a
    Slurm array can glob exactly one model's cells (extraction shards strictly by model). A
    real ``model_id`` (a Hugging Face id) selects the ``hf`` backend; ``None`` selects the
    dependency-free ``reference`` backend for offline / Tier-0 cells.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    model_id: str | None = None
    revision: str | None = None
    device: str = "cpu"
    trust_remote_code: bool = False


class ProbeGridAxes(BaseModel):
    """A compact description of one experiment tier's grid (Project 2 Part III).

    :func:`expand_probe_grid` takes the Cartesian product over
    ``models x layer_depth_fractions x poolings x aggregations x target_fprs x seeds``, so the
    emitted cell count is exactly the product of the axis lengths. Each inner list of
    ``layer_depth_fractions`` / ``target_fprs`` is *one cell's* set (the depth-fraction band or
    FPR budget for that cell), which is why those two axes are lists-of-lists.

    The data source mirrors :class:`ProbeDetectionExperimentConfig`: set
    ``survival_results_path`` + ``final_tokens_path`` for the delivery-verified measurement path
    (Tier 1+), or leave both ``None`` and set ``synthetic_mode`` for the offline reference path
    (Tier 0). ``generalization`` (optional) threads an E2.x train/test holdout onto every cell.
    """

    model_config = ConfigDict(extra="forbid")

    experiment_family: str
    models: list[ProbeModelSpec]
    layer_depth_fractions: list[list[float]]
    poolings: list[PoolingStrategy]
    aggregations: list[str]
    target_fprs: list[list[float]]
    seeds: list[int]

    # Data source (measurement path xor offline path); mirrors the config fields of the same name.
    survival_results_path: Path | None = None
    final_tokens_path: Path | None = None
    synthetic_mode: Literal["simple", "twins"] | None = None

    # Optional E2.x generalization holdout, threaded verbatim onto every generated cell.
    generalization: GeneralizationSpec | None = None

    @model_validator(mode="after")
    def _check_axes_nonempty(self) -> ProbeGridAxes:
        """Reject an empty axis up front: a Cartesian product with an empty factor is 0 cells."""
        empty = [
            name
            for name, seq in (
                ("models", self.models),
                ("layer_depth_fractions", self.layer_depth_fractions),
                ("poolings", self.poolings),
                ("aggregations", self.aggregations),
                ("target_fprs", self.target_fprs),
                ("seeds", self.seeds),
            )
            if not seq
        ]
        if empty:
            raise ValueError(f"grid axes must all be non-empty; empty: {', '.join(empty)}")
        return self


def expand_probe_grid(axes: ProbeGridAxes) -> list[ProbeDetectionExperimentConfig]:
    """Expand an axes description into one runnable config per grid cell.

    The Cartesian product is iterated in a fixed order
    (``models x layer_depth_fractions x poolings x aggregations x target_fprs x seeds``), so the
    output order and every cell's content-derived ``experiment_id`` are deterministic:
    re-expanding identical axes yields byte-identical ids. Each cell is a complete, standalone
    :class:`ProbeDetectionExperimentConfig` runnable as-is via ``run_probe_experiment``:

    - ``extractor_backend`` is ``"hf"`` when the model spec carries a real ``model_id``, else
      ``"reference"`` (offline).
    - ``layer_depth_fractions`` populate the config so component C resolves them to concrete
      layers at runtime against each model's ``num_layers`` (same relative depth across sizes).
    - ``activations_dir`` / ``results_out`` / ``predictions_out`` point at a per-cell directory
      under ``outputs/probe_detection/generated_runs/<family>/<experiment_id>/`` so an array
      never collides on an output path.
    - ``reuse_store`` is ``True`` so the CPU probe rerun after GPU extraction reuses the store.
    - ``generalization`` (if any) is threaded onto every cell so ``run_probe_experiment`` honors
      the E2.x holdout with no extra wiring.
    """
    family_slug = _slug(axes.experiment_family)
    run_root = _GENERATED_RUN_ROOT / family_slug
    configs: list[ProbeDetectionExperimentConfig] = []

    for model, fractions, pooling, aggregation, fprs, seed in itertools.product(
        axes.models,
        axes.layer_depth_fractions,
        axes.poolings,
        axes.aggregations,
        axes.target_fprs,
        axes.seeds,
    ):
        is_real_model = bool(model.model_id)
        backend = "hf" if is_real_model else "reference"

        # Every coordinate that distinguishes a cell is folded into the hash key, so no two
        # cells collide and re-expansion is identical. The model label is a readable prefix
        # component (not only inside the hash) so a Slurm array can glob one model's cells.
        digest = stable_id(
            axes.experiment_family,
            model.id,
            model.model_id or "",
            model.revision or "",
            fractions,
            pooling.value,
            aggregation,
            fprs,
            seed,
            length=10,
        )
        experiment_id = f"{family_slug}_{_slug(model.id)}_{digest}"
        cell_dir = run_root / experiment_id

        name_parts = [
            axes.experiment_family,
            f"model={model.id}",
            f"pooling={pooling.value}",
            f"agg={aggregation}",
            f"fprs={fprs}",
            f"seed={seed}",
        ]
        if axes.generalization is not None:
            name_parts.append(
                f"generalization={axes.generalization.kind}(applied via config.generalization)"
            )

        configs.append(
            ProbeDetectionExperimentConfig(
                name=" ".join(name_parts),
                experiment_id=experiment_id,
                model_id=model.model_id or "reference-model",
                extractor_backend=backend,
                device=model.device,
                revision=model.revision,
                trust_remote_code=model.trust_remote_code,
                layer_depth_fractions=list(fractions),
                pooling=pooling,
                aggregation=aggregation,
                target_fprs=list(fprs),
                split_seed=seed,
                synthetic_seed=seed,
                synthetic_mode=axes.synthetic_mode or "simple",
                survival_results_path=axes.survival_results_path,
                final_tokens_path=axes.final_tokens_path,
                generalization=axes.generalization,
                reuse_store=True,
                activations_dir=cell_dir / "activations",
                results_out=cell_dir / "results.jsonl",
                predictions_out=cell_dir / "predictions.jsonl",
            )
        )
    return configs


def write_probe_configs(
    configs: Sequence[ProbeDetectionExperimentConfig], out_dir: Path
) -> list[Path]:
    """Write each cell to ``<out_dir>/<experiment_id>.yaml`` and return the paths in order.

    Serialized via ``model_dump(mode="json")`` so every value is YAML-native (paths become
    strings, enums their values, nested models plain dicts); each file therefore round-trips
    cleanly back through ``load_config(path, ProbeDetectionExperimentConfig)``. The
    ``experiment_id`` filename is stable and unique per cell, so re-running the expander
    overwrites in place rather than accumulating stale configs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for config in configs:
        path = out_dir / f"{config.experiment_id}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                config.model_dump(mode="json"),
                handle,
                sort_keys=True,
                default_flow_style=False,
            )
        paths.append(path)
    return paths
