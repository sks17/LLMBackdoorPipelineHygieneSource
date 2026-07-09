"""Tests for twin-calibrated semantic thresholds and the Wilson interval helper."""

from __future__ import annotations

import numpy as np
import pytest

from trigger_audit.scoring.semantic import (
    calibrate_semantic_threshold,
    wilson_interval,
)


def test_empirical_fpr_never_exceeds_target_on_absent_twins():
    rng = np.random.default_rng(0)
    absent = rng.random(500)  # entail scores live in [0, 1]
    for target in (0.2, 0.05, 0.01):
        result = calibrate_semantic_threshold(absent, target)
        empirical = float(np.mean(absent >= result.threshold))
        assert empirical <= target
        assert result.achieved_fpr == pytest.approx(empirical)
        assert result.n_absent == 500
        assert result.target_fpr == target


def test_target_below_one_over_n_puts_threshold_above_max():
    absent = np.linspace(0.0, 1.0, 100)
    result = calibrate_semantic_threshold(absent, 0.001)  # 0.001 < 1/100
    assert result.threshold > absent.max()
    assert result.achieved_fpr == 0.0
    assert float(np.mean(absent >= result.threshold)) == 0.0


def test_calibration_requires_absent_scores():
    with pytest.raises(ValueError, match="at least one absent score"):
        calibrate_semantic_threshold(np.array([]), 0.01)


def test_calibration_rejects_out_of_range_target():
    with pytest.raises(ValueError, match="target_fpr must be in"):
        calibrate_semantic_threshold(np.array([0.1, 0.2]), 1.5)


def test_threshold_is_monotone_in_target():
    rng = np.random.default_rng(1)
    absent = rng.random(400)
    loose = calibrate_semantic_threshold(absent, 0.2)
    strict = calibrate_semantic_threshold(absent, 0.02)
    # A tighter FPR budget can never admit a lower threshold.
    assert strict.threshold >= loose.threshold
    assert strict.achieved_fpr <= loose.achieved_fpr


def test_achieved_fpr_interval_contains_empirical_rate_and_stays_in_unit_interval():
    rng = np.random.default_rng(2)
    absent = rng.random(300)
    result = calibrate_semantic_threshold(absent, 0.05)
    low, high = result.achieved_fpr_interval()
    assert 0.0 <= low <= result.achieved_fpr <= high <= 1.0


def test_wilson_interval_contains_empirical_rate_and_stays_in_unit_interval():
    for k, n in [(0, 50), (3, 50), (25, 50), (50, 50)]:
        low, high = wilson_interval(k, n)
        assert 0.0 <= low <= k / n <= high <= 1.0


def test_wilson_interval_at_zero_successes_has_zero_lower_bound():
    low, high = wilson_interval(0, 40)
    assert low == 0.0
    assert 0.0 < high < 0.15


def test_wilson_interval_degenerate_and_invalid_inputs():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError, match="k must be"):
        wilson_interval(5, 3)
