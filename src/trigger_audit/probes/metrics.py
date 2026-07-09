"""Numpy-only detection metrics: AUROC, TPR at a target FPR, confusion counts.

Design intent: for deployment-style monitoring the operating points at low target FPRs
(e.g. 1e-2, 1e-3) matter more than AUROC -- a monitor fires on a threshold, not on a rank
statistic -- so the low-FPR machinery (``threshold_at_fpr``, ``tpr_at_fpr``) is first-class
here and AUROC is reported as a summary alongside, never instead.

The decision rule is fixed everywhere in this package: predict positive iff
``score >= threshold``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _split_by_label(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=bool).ravel()
    if s.shape != y.shape:
        raise ValueError(f"scores ({s.shape}) and labels ({y.shape}) disagree")
    pos, neg = s[y], s[~y]
    if pos.size == 0 or neg.size == 0:
        raise ValueError("metrics require at least one positive and one negative example")
    return pos, neg


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney), tie-aware via average ranks.

    Equivalent to the probability that a random positive outscores a random negative, with
    ties counting one half -- exactly what averaging tied ranks computes, with no O(n^2)
    pairwise loop.
    """
    pos, neg = _split_by_label(scores, labels)
    s = np.concatenate([pos, neg])
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    # Average rank per tie group: unique values -> [start, end] rank range -> midpoint.
    _, inverse, counts = np.unique(sorted_s, return_inverse=True, return_counts=True)
    cumulative: np.ndarray = np.cumsum(counts)
    average_rank_per_group = (cumulative - counts + 1 + cumulative) / 2.0
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = average_rank_per_group[inverse]

    n_pos, n_neg = pos.size, neg.size
    rank_sum_pos = float(ranks[:n_pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def threshold_at_fpr(negative_scores: np.ndarray, target_fpr: float) -> tuple[float, float]:
    """Return ``(threshold, achieved_fpr)``: the lowest threshold with empirical FPR <= target.

    The lowest such threshold maximizes TPR at the constrained operating point. When the
    target is below ``1/n`` (no false positive is affordable) the threshold is placed just
    above the maximum negative score, giving an empirical FPR of exactly 0 -- the honest
    semantics for "we cannot resolve rates that small with this many negatives".
    """
    neg = np.asarray(negative_scores, dtype=np.float64).ravel()
    if neg.size == 0:
        raise ValueError("threshold_at_fpr requires at least one negative score")
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}")

    n = neg.size
    # Nudge by a tiny epsilon before flooring so a target whose intended budget is an exact
    # integer (e.g. 0.29 * 100 = 29) does not lose one allowed false positive to binary
    # float representation (0.29 * 100 == 28.9999...). The nudge is far smaller than the gap
    # to the next achievable budget, so it never admits an unintended extra false positive.
    max_false_positives = math.floor(target_fpr * n + 1e-9)
    sorted_neg = np.sort(neg)
    candidates = np.unique(sorted_neg)
    count_at_or_above = n - np.searchsorted(sorted_neg, candidates, side="left")
    admissible = count_at_or_above <= max_false_positives
    if admissible.any():
        index = int(np.argmax(admissible))  # first admissible = lowest threshold
        return float(candidates[index]), float(count_at_or_above[index]) / n
    return float(np.nextafter(sorted_neg[-1], np.inf)), 0.0


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    """TPR at the lowest threshold whose empirical FPR on these labels' negatives <= target."""
    pos, neg = _split_by_label(scores, labels)
    threshold, _ = threshold_at_fpr(neg, target_fpr)
    return float(np.mean(pos >= threshold))


@dataclass(frozen=True)
class ConfusionCounts:
    """Confusion-matrix counts at a fixed threshold (positive iff score >= threshold)."""

    tp: int
    fp: int
    tn: int
    fn: int


def confusion_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> ConfusionCounts:
    """Count the confusion matrix at ``threshold`` under the package-wide ``>=`` rule."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=bool).ravel()
    if s.shape != y.shape:
        raise ValueError(f"scores ({s.shape}) and labels ({y.shape}) disagree")
    predicted = s >= threshold
    return ConfusionCounts(
        tp=int(np.sum(predicted & y)),
        fp=int(np.sum(predicted & ~y)),
        tn=int(np.sum(~predicted & ~y)),
        fn=int(np.sum(~predicted & y)),
    )
