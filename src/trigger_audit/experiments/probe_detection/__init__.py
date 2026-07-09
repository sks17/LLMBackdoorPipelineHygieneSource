"""Experiment 2: activation-based trigger detection with delivery-verified labels.

Trains per-layer linear probes on hidden-state activations to detect trigger presence,
calibrates thresholds to target false-positive rates on held-out clean negatives, and
evaluates conditional on verified delivery (labels joined from Project 1's
``SurvivalResult`` records, never from raw insertion).
"""

from trigger_audit.experiments.probe_detection.config import ProbeDetectionExperimentConfig
from trigger_audit.experiments.probe_detection.dataset import (
    assign_splits,
    build_probe_examples,
    build_synthetic_probe_dataset,
    build_synthetic_probe_dataset_with_twins,
)
from trigger_audit.experiments.probe_detection.generalization import (
    GeneralizationSpec,
    assign_generalization_splits,
    partition_by_metadata,
)
from trigger_audit.experiments.probe_detection.grid import (
    ProbeGridAxes,
    ProbeModelSpec,
    expand_probe_grid,
    write_probe_configs,
)
from trigger_audit.experiments.probe_detection.runner import (
    ProbeDetectionRunner,
    run_probe_experiment,
)

__all__ = [
    "GeneralizationSpec",
    "ProbeDetectionExperimentConfig",
    "ProbeDetectionRunner",
    "ProbeGridAxes",
    "ProbeModelSpec",
    "assign_generalization_splits",
    "assign_splits",
    "build_probe_examples",
    "build_synthetic_probe_dataset",
    "build_synthetic_probe_dataset_with_twins",
    "expand_probe_grid",
    "partition_by_metadata",
    "run_probe_experiment",
    "write_probe_configs",
]
