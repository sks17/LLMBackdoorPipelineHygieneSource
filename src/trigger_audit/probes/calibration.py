"""Threshold calibration to a target false-positive rate on held-out clean negatives.

Design intent: a deployed trigger monitor is defined by its threshold, and the threshold's
contract is a false-positive budget (e.g. 1e-2, 1e-3), not an AUROC. Calibration therefore
chooses the empirical quantile of held-out NEGATIVE scores such that the empirical FPR does
not exceed the target, and reports the achieved FPR with a Wilson score 95% interval.

Wilson (not Clopper-Pearson) on purpose: the exact interval needs the beta inverse CDF,
which lives in scipy, and scipy is deliberately not a base dependency. Wilson is closed-form
in stdlib math, well-behaved at k=0 and k=n, and matches the convention already used by the
analysis layer (``trigger_audit.analysis.stats.wilson_ci``) -- reimplemented here so the
probe layer does not import the pandas-heavy analysis package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trigger_audit.probes.metrics import threshold_at_fpr


@dataclass(frozen=True)
class ThresholdCalibration:
    """A calibrated operating point: the threshold and the empirical FPR it achieves.

    When ``target_fpr < 1/n_negatives`` the empirical quantile cannot resolve the target;
    the threshold is placed just above the maximum negative score, so ``achieved_fpr`` is
    0.0 and the Wilson interval on 0/n honestly reports how little that certifies.
    """

    threshold: float
    achieved_fpr: float
    target_fpr: float
    n_negatives: int


def calibrate_threshold(negative_scores: np.ndarray, target_fpr: float) -> ThresholdCalibration:
    """Choose the lowest threshold whose empirical FPR on held-out negatives <= target.

    The lowest admissible threshold maximizes sensitivity subject to the FPR budget; using
    held-out CALIBRATION negatives (never training scores) keeps the budget honest on new
    data drawn from the same clean distribution.
    """
    neg = np.asarray(negative_scores, dtype=np.float64).ravel()
    threshold, achieved = threshold_at_fpr(neg, target_fpr)
    return ThresholdCalibration(
        threshold=threshold,
        achieved_fpr=achieved,
        target_fpr=float(target_fpr),
        n_negatives=int(neg.size),
    )


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval (default z) for a binomial proportion ``k/n``.

    Used for the achieved-FPR uncertainty at calibrated thresholds. Returns ``(0.0, 1.0)``
    for ``n <= 0`` (no data constrains nothing).
    """
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, n], got k={k}, n={n}")
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))
