"""Linear probing on activations: numpy-only probes, metrics, calibration, aggregation.

Everything here is deliberately dependency-free beyond numpy (a core dependency), so probe
training, threshold calibration, and evaluation run on CPU-only login nodes and in offline
tests. The interfaces are stable extension points; heavier implementations (sklearn solvers,
GPU probes) can be swapped in behind them later.
"""

from trigger_audit.probes.aggregation import AGGREGATION_REGISTRY, ScoreAggregator
from trigger_audit.probes.calibration import (
    ThresholdCalibration,
    calibrate_threshold,
    wilson_interval,
)
from trigger_audit.probes.linear import LinearProbe
from trigger_audit.probes.metrics import (
    ConfusionCounts,
    auroc,
    confusion_at_threshold,
    threshold_at_fpr,
    tpr_at_fpr,
)

__all__ = [
    "AGGREGATION_REGISTRY",
    "ConfusionCounts",
    "LinearProbe",
    "ScoreAggregator",
    "ThresholdCalibration",
    "auroc",
    "calibrate_threshold",
    "confusion_at_threshold",
    "threshold_at_fpr",
    "tpr_at_fpr",
    "wilson_interval",
]
