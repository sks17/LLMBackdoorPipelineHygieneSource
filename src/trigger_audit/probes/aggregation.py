"""Multi-layer score aggregation: combine per-layer probe scores into one detector score.

Open-source probing stacks typically stop at "pick the best single layer"; principled
multi-layer combiners -- min (all layers must agree), product-of-experts, and quantile
aggregation -- are a known gap this module fills. Aggregators are registry-resolved by name
(reusing the generic :class:`~trigger_audit.pipelines.base.Registry`), so a config string
selects the combiner without ``if/elif`` chains.

Interface contract: ``fit(layer_scores, labels)`` is a no-op for the closed-form
aggregators and required only by learned ones (``stacked_logistic``);
``aggregate(layer_scores)`` maps an ``(n_examples, n_layers)`` score matrix to
``(n_examples,)`` combined scores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from trigger_audit.pipelines.base import Registry
from trigger_audit.probes.linear import LinearProbe

AGGREGATION_REGISTRY: Registry[ScoreAggregator] = Registry("aggregation")


def _as_matrix(layer_scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(layer_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError(
            f"expected a 2-D (n_examples, n_layers) score matrix, got shape {scores.shape}"
        )
    return scores


class ScoreAggregator(ABC):
    """Combines per-layer probe scores into a single score per example."""

    name: str = "aggregator"

    def fit(self, layer_scores: np.ndarray, labels: np.ndarray) -> None:
        """Learn combiner parameters.

        Default: validate the score matrix and learn nothing -- the closed-form combiners
        are parameter-free, and only learned aggregators (``stacked_logistic``) override
        this. Keeping ``fit`` in the base interface lets the runner treat both kinds
        uniformly.
        """
        _as_matrix(layer_scores)

    @abstractmethod
    def aggregate(self, layer_scores: np.ndarray) -> np.ndarray:
        """Map an ``(n_examples, n_layers)`` matrix to ``(n_examples,)`` combined scores."""


@AGGREGATION_REGISTRY.register("mean_score")
class MeanScore(ScoreAggregator):
    """Average the per-layer scores (the robust default)."""

    name = "mean_score"

    def aggregate(self, layer_scores: np.ndarray) -> np.ndarray:
        return _as_matrix(layer_scores).mean(axis=1)


@AGGREGATION_REGISTRY.register("max_score")
class MaxScore(ScoreAggregator):
    """Fire if ANY layer fires (highest sensitivity, loosest FPR control per layer)."""

    name = "max_score"

    def aggregate(self, layer_scores: np.ndarray) -> np.ndarray:
        return _as_matrix(layer_scores).max(axis=1)


@AGGREGATION_REGISTRY.register("min_score")
class MinScore(ScoreAggregator):
    """Fire only if ALL layers fire (conservative; suppresses single-layer false positives)."""

    name = "min_score"

    def aggregate(self, layer_scores: np.ndarray) -> np.ndarray:
        return _as_matrix(layer_scores).min(axis=1)


@AGGREGATION_REGISTRY.register("product_of_experts")
class ProductOfExperts(ScoreAggregator):
    """Sum the per-layer logits: the log-odds of a product of independent experts."""

    name = "product_of_experts"

    def aggregate(self, layer_scores: np.ndarray) -> np.ndarray:
        return _as_matrix(layer_scores).sum(axis=1)


@AGGREGATION_REGISTRY.register("quantile")
class QuantileScore(ScoreAggregator):
    """Take a configurable per-example quantile of the layer scores.

    ``q`` interpolates between ``min_score`` (q=0) and ``max_score`` (q=1); the median
    (q=0.5) is a vote-like combiner robust to one aberrant layer.
    """

    name = "quantile"

    def __init__(self, q: float = 0.5) -> None:
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"q must be in [0, 1], got {q}")
        self._q = float(q)

    def aggregate(self, layer_scores: np.ndarray) -> np.ndarray:
        return np.quantile(_as_matrix(layer_scores), self._q, axis=1)


@AGGREGATION_REGISTRY.register("stacked_logistic")
class StackedLogistic(ScoreAggregator):
    """Fit a :class:`LinearProbe` over the per-layer score vectors (learned stacking).

    The only aggregator that needs labels at fit time: it learns per-layer weights, so it
    can down-weight uninformative or inverted layers that hurt the closed-form combiners.

    Stacking-leakage note: the per-layer probes are fit on TRAIN, so their TRAIN scores are
    optimistically separated relative to anything this stacker sees at inference. Fitting the
    stacker on those TRAIN scores would mis-scale its weights and make every layer look
    informative. The runner therefore fits this aggregator on the held-out CALIBRATION split
    scores, not TRAIN. That is the pragmatic choice given the runner's shape; the cleaner
    alternative is out-of-fold stacking (k-fold within TRAIN, score each held-out fold with
    probes fit on the rest, fit the stacker on the assembled out-of-fold matrix, then refit
    the per-layer probes on full TRAIN for inference). Calibration-split fitting reuses the
    same negatives that set the thresholds, a mild adaptivity trade-off OOF would avoid.
    """

    name = "stacked_logistic"

    def __init__(self, *, l2: float = 1e-3, lr: float = 0.5, max_iter: int = 500) -> None:
        self._probe = LinearProbe(l2=l2, lr=lr, max_iter=max_iter)

    def fit(self, layer_scores: np.ndarray, labels: np.ndarray) -> None:
        self._probe.fit(_as_matrix(layer_scores), np.asarray(labels))

    def aggregate(self, layer_scores: np.ndarray) -> np.ndarray:
        if not self._probe.is_fitted:
            raise RuntimeError("stacked_logistic requires fit(layer_scores, labels) first")
        return self._probe.decision_scores(_as_matrix(layer_scores))
