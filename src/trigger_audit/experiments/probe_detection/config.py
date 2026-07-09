"""Configuration schema for the probe-detection experiment (Project 2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from trigger_audit.experiments.probe_detection.generalization import GeneralizationSpec
from trigger_audit.schemas.probes import PoolingStrategy


def _default_layers() -> list[int]:
    return [1, 2, 3, 4]


def _default_target_fprs() -> list[float]:
    return [1e-2, 1e-3]


class ProbeDetectionExperimentConfig(BaseModel):
    """One probe-detection run: extractor, layers, pooling, probe, calibration, and paths.

    Loaded with the shared :func:`trigger_audit.config.loader.load_config` (the repo-wide
    YAML pattern). The dataset comes from one of two sources: ``survival_results_path`` +
    ``final_tokens_path`` joins Project 1's delivery-verified records (the measurement
    path), or -- when both are null -- a deterministic synthetic dataset is built so the
    whole experiment runs offline against the reference extractor (the smoke/test path).
    """

    name: str = "probe_detection"
    experiment_id: str
    model_id: str = "reference-model"

    # Activation extractor: "reference" (offline, deterministic) or "hf" (real model).
    extractor_backend: str = "reference"
    extractor_seed: int = 0
    extractor_hidden_size: int = 64
    extractor_num_layers: int = 4

    # Real-model (hf backend) load knobs; threaded into make_activation_extractor. Harmless
    # for the reference backend (which ignores them), so offline runs are unchanged.
    device: str = "cpu"
    revision: str | None = None
    trust_remote_code: bool = False

    layers: list[int] = Field(default_factory=_default_layers)
    # When set, the concrete probe layers are resolved at runtime from the loaded model's
    # ``num_layers`` (same relative depth across model sizes), overriding ``layers``. ``None``
    # means "use ``layers`` verbatim". Each fraction must lie in [0, 1]; an empty list is
    # invalid. See activations/slicing.resolve_layers_from_fractions.
    layer_depth_fractions: list[float] | None = None
    pooling: PoolingStrategy = PoolingStrategy.MEAN

    # Probe hyperparameters (numpy logistic regression; see probes/linear.py).
    probe_l2: float = 1e-2
    probe_lr: float = 0.5
    probe_max_iter: int = 1000

    target_fprs: list[float] = Field(default_factory=_default_target_fprs)
    aggregation: str = "mean_score"
    aggregation_params: dict[str, Any] = Field(default_factory=dict)

    # base_id-grouped split fractions (test gets the remainder); see dataset.assign_splits.
    train_fraction: float = 0.5
    calibration_fraction: float = 0.25
    split_seed: int = 0

    # E2.x generalization holdout. When set, the run applies this holdout (drop un-held-out
    # rows, hold out one side as TEST, carve a base_id-grouped calibration subset from the
    # train side) instead of the base_id-fraction split. ``None`` keeps every existing run
    # unchanged. Imported from the neutral ``generalization`` leaf module (not ``grid``) to
    # avoid a config -> grid -> config import cycle.
    generalization: GeneralizationSpec | None = None

    # Measurement inputs (Project 1 join). Both null -> synthetic offline dataset.
    survival_results_path: Path | None = None
    final_tokens_path: Path | None = None
    # Which synthetic builder the offline path uses: "simple" (unique base per example,
    # trigger_inserted == label) or "twins" (counterfactual twin pairs + partial-survival
    # negatives). The twins builder is driven by synthetic_n_bases/partial_survival_fraction.
    synthetic_mode: Literal["simple", "twins"] = "simple"
    synthetic_n_examples: int = 60
    synthetic_n_bases: int = 40
    partial_survival_fraction: float = 0.25
    synthetic_seq_len: int = 16
    synthetic_seed: int = 0

    # Extract-once, pool-many store reuse: when True, a layer's pooled features are loaded
    # from the store (matching trial-id order + producer metadata) instead of re-extracted.
    reuse_store: bool = False
    # When set, the runner writes one ProbePrediction per TEST example here (default off, so
    # existing runs are unchanged unless they opt in).
    predictions_out: Path | None = None

    activations_dir: Path = Path("outputs/activations")
    results_out: Path = Path("outputs/probe_detection/results.jsonl")

    @field_validator("layers")
    @classmethod
    def _validate_layers(cls, layers: list[int]) -> list[int]:
        """Reject layer lists that would fail deep in the run with a misleading message.

        A duplicate layer crashes far downstream in ``store.save`` as a spurious "corrupt
        store" row-count error, and a negative layer only surfaces at extract time (after a
        potentially expensive HF model load); catching both here fails fast at config load.
        The upper bound is model-dependent and so is checked once the extractor is built (see
        ``run_probe_experiment``).
        """
        if not layers:
            raise ValueError("layers must not be empty")
        if any(layer < 0 for layer in layers):
            raise ValueError(f"layers must all be non-negative, got {layers}")
        if len(set(layers)) != len(layers):
            raise ValueError(f"layers must be unique, got {layers}")
        return layers

    @field_validator("layer_depth_fractions")
    @classmethod
    def _validate_layer_depth_fractions(cls, fractions: list[float] | None) -> list[float] | None:
        """Reject a malformed depth-fraction request at config load rather than at runtime.

        ``None`` means "use ``layers`` verbatim" and is left untouched. A set-but-empty list
        is a mistake (it would resolve to no layers), and every fraction must be a genuine
        relative depth in ``[0.0, 1.0]`` -- the upper bound of the resolved index is
        model-dependent and checked once the extractor is built (see ``run_probe_experiment``).
        """
        if fractions is None:
            return None
        if not fractions:
            raise ValueError("layer_depth_fractions must be non-empty when set")
        for fraction in fractions:
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"layer_depth_fractions must all be in [0.0, 1.0], got {fractions}"
                )
        return fractions
