"""Probe schemas: activation-probe training examples and calibrated evaluation records.

Project 2 trains linear probes on per-layer hidden-state activations to detect trigger
presence. Its labels are *delivery-verified*: an example is positive only when Project 1's
:class:`~trigger_audit.schemas.results.SurvivalResult` says the trigger reached the final
model-visible tokens (``final_token_trigger_present``), never merely because a trigger was
inserted upstream. These schemas are the data contracts for that experiment.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PoolingStrategy(str, Enum):
    """How per-token activations of one layer are pooled into a single feature vector."""

    LAST_TOKEN = "last_token"
    MEAN = "mean"
    MAX = "max"
    TRIGGER_SPAN = "trigger_span"


class ProbeLabelSource(str, Enum):
    """Where a probe example's label came from.

    ``SURVIVAL_RESULT`` labels are delivery-verified ground truth joined from Project 1;
    ``SYNTHETIC`` labels come from constructed token sequences (offline tests and smoke runs).
    """

    SURVIVAL_RESULT = "survival_result"
    SYNTHETIC = "synthetic"


class ProbeSplit(str, Enum):
    """Which role an example plays: probe training, threshold calibration, or held-out test."""

    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"


class ProbeExample(BaseModel):
    """One labeled example for probe training/evaluation.

    ``label`` means "the trigger was delivered into the final model-visible tokens", not
    "a trigger was inserted somewhere upstream" -- the two diverge exactly when the prompt
    pipeline drops or corrupts the trigger, which is the phenomenon this project measures.
    The optional token span localizes the trigger within the final token sequence (for
    :attr:`PoolingStrategy.TRIGGER_SPAN` pooling). ``metadata["trigger_inserted"]`` records
    whether a trigger was inserted upstream so evaluation can separate clean negatives from
    inserted-but-not-delivered negatives.
    """

    trial_id: str
    base_id: str
    label: bool
    label_source: ProbeLabelSource
    split: ProbeSplit = ProbeSplit.TRAIN
    trigger_token_start: int | None = None
    trigger_token_end: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def trigger_span(self) -> tuple[int, int] | None:
        """Return the half-open trigger token span, or None when either endpoint is unknown."""
        if self.trigger_token_start is None or self.trigger_token_end is None:
            return None
        return (self.trigger_token_start, self.trigger_token_end)


class ProbePrediction(BaseModel):
    """One honest per-trial probe prediction over a TEST-split example.

    Emitted (optionally) by the runner so the inference layer (component G) can
    cluster-bootstrap ``P(fire | delivered)`` over ``base_id`` and compute ``TAR_w/TAR_wo``
    per stratum without re-running the probe. Every field is a plain per-trial fact; no
    aggregation across trials happens here.

    - ``label`` -- delivery-verified positive (the trigger reached the final tokens).
    - ``trigger_inserted`` -- whether a trigger was inserted upstream (raw-layer flag).
    - ``delivered`` -- the delivered-only membership, ``label | ~trigger_inserted``: verified
      positives plus clean (never-inserted) negatives, i.e. every trial whose activations are
      not contaminated by an undelivered trigger fragment.
    - ``clean_negative`` -- ``~label & ~trigger_inserted``: a negative that was never inserted,
      the population the FPR budget actually contracts.
    - ``aggregated_score`` -- the multi-layer aggregated decision score for this trial.
    - ``layer_scores`` -- per-layer decision score keyed by ``str(layer_index)``.
    - ``fired`` -- keyed by ``str(target_fpr)``, whether ``aggregated_score`` met the
      threshold calibrated to that target FPR (``score >= threshold``).
    """

    trial_id: str
    base_id: str
    label: bool
    trigger_inserted: bool
    delivered: bool
    clean_negative: bool
    split: ProbeSplit
    aggregated_score: float
    layer_scores: dict[str, float] = Field(default_factory=dict)
    fired: dict[str, bool] = Field(default_factory=dict)


class LayerProbeMetrics(BaseModel):
    """Evaluation metrics for one layer's probe (or the multi-layer aggregate).

    ``layer_index`` is the hidden-state layer the probe read (0 = embedding layer, matching
    Hugging Face ``hidden_states`` indexing); the sentinel ``-1`` marks the multi-layer
    aggregated score. ``tpr_at_target_fpr`` and ``threshold`` are keyed by the target FPR
    rendered as a string (e.g. ``"0.01"``) so the mapping survives JSON round-trips; the TPR
    is measured at the *calibrated* threshold, i.e. the operating point a deployment would
    actually use, not an oracle point read off the test ROC curve.
    """

    layer_index: int
    auroc: float
    tpr_at_target_fpr: dict[str, float] = Field(default_factory=dict)
    threshold: dict[str, float] = Field(default_factory=dict)
    n_pos: int
    n_neg: int


class AchievedFpr(BaseModel):
    """Empirical false-positive rate achieved on held-out negatives at one calibrated threshold.

    The 95% interval is a Wilson score interval, computed with numpy/stdlib only: the exact
    Clopper-Pearson interval needs the beta inverse CDF (scipy), and scipy is deliberately not
    a base dependency of this package. Wilson is well-behaved at k=0 and k=n, which are the
    common cases at deployment-style target FPRs (1e-2, 1e-3) with modest negative counts.
    """

    target_fpr: float
    achieved_fpr: float
    ci_low: float
    ci_high: float
    n_negatives: int


class ProbeEvaluationResult(BaseModel):
    """End-to-end record of one probe-detection experiment.

    The survival-aware split is this project's differentiator: every metric is reported twice.

    - ``*_all`` -- over ALL test trials, where an inserted-but-not-delivered trigger counts as
      a negative (its fragments may still contaminate the activations).
    - ``*_delivered_only`` -- over test trials restricted to verified outcomes: positives whose
      trigger provably reached the final tokens, plus negatives where no trigger was inserted
      at all. This isolates P(probe fires | trigger delivered), the quantity a monitoring
      deployment actually cares about, from pipeline-induced label noise.

    Thresholds are calibrated on the clean (never-inserted) negatives of the CALIBRATION
    split to the ``target_fprs`` operating points -- the probe monitors clean traffic, so its
    FPR budget is a statement about clean negatives, and partial-survival (inserted-but-
    undelivered) negatives are deliberately kept out of the calibration pool. ``achieved_fprs``
    reports the empirical FPR of those thresholds on ALL test negatives (the deployment-
    pessimistic view, where partial-survival negatives count against the budget), while
    ``achieved_fprs_clean`` reports it on the CLEAN test negatives only (the population the
    budget actually contracts). Both carry a Wilson 95% interval (see :class:`AchievedFpr`).
    """

    experiment_id: str
    model_id: str
    extractor_backend: str
    pooling: PoolingStrategy
    layers: list[int]
    target_fprs: list[float]

    layer_metrics_all: list[LayerProbeMetrics]
    layer_metrics_delivered_only: list[LayerProbeMetrics]

    aggregation: str
    aggregated_metrics_all: LayerProbeMetrics
    aggregated_metrics_delivered_only: LayerProbeMetrics

    achieved_fprs: list[AchievedFpr]
    achieved_fprs_clean: list[AchievedFpr] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
