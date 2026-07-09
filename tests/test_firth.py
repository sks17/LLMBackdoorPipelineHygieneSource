"""Tests for the Firth-penalized logistic regression in ``analysis/stats.py``.

The per-cell delivered rates are near-total 0/1 cells, which is exactly the complete separation
where a vanilla logistic MLE diverges to +-inf. Firth's Jeffreys-prior penalty guarantees a finite
maximum-penalized-likelihood estimate. These tests pin that property (finite estimates on a fully
separated cell) and the design-matrix column-keyed result shape.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from trigger_audit.analysis.stats import FirthResult, firth_logit, firth_logit_from_frame


def _all_finite(mapping: dict[str, float]) -> bool:
    return all(math.isfinite(v) for v in mapping.values())


def test_firth_finite_under_complete_separation() -> None:
    # x=0 -> y=0 and x=1 -> y=1 with no overlap: perfect (complete) separation. A vanilla logistic
    # MLE sends the slope to +inf; Firth must return a finite, positive, converged estimate.
    predictor = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    x = np.column_stack([np.ones_like(predictor), predictor])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)

    result = firth_logit(x, y, feature_names=["Intercept", "x"])

    assert isinstance(result, FirthResult)
    assert result.converged
    assert _all_finite(result.params)
    assert _all_finite(result.std_err)
    # The penalized slope is finite (not the +inf a vanilla GLM would produce) and points the right
    # way: higher x predicts the positive outcome.
    assert result.params["x"] > 0.0
    assert math.isfinite(result.params["x"])
    assert result.std_err["x"] > 0.0
    assert result.n == 8


def test_firth_from_frame_is_design_column_keyed() -> None:
    # A fully separated categorical cell: policy 'a' never delivers, policy 'b' always delivers.
    df = pd.DataFrame(
        {
            "pipeline_policy": ["a", "a", "a", "b", "b", "b"],
            "delivered": [0, 0, 0, 1, 1, 1],
        }
    )
    result = firth_logit_from_frame(df, "delivered", ["pipeline_policy"])

    # First sorted level ('a') is the dropped reference; the design carries Intercept + policy=b.
    assert set(result.params) == {"Intercept", "pipeline_policy=b"}
    assert set(result.std_err) == set(result.params)
    assert set(result.p_values) == set(result.params)
    assert _all_finite(result.params)
    assert _all_finite(result.std_err)
    # 'b' delivers where 'a' does not, so the non-reference coefficient is finite and positive.
    assert math.isfinite(result.params["pipeline_policy=b"])
    assert result.params["pipeline_policy=b"] > 0.0
    assert result.n == 6


def test_firth_matches_balanced_logistic_sign_without_separation() -> None:
    # With a non-separated, informative cell Firth still fits and recovers the correct sign, so the
    # penalty is a sensitivity view rather than a distortion.
    predictor = np.array([0, 0, 0, 1, 1, 1, 0, 1], dtype=float)
    x = np.column_stack([np.ones_like(predictor), predictor])
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=float)

    result = firth_logit(x, y, feature_names=["Intercept", "x"])

    assert result.converged
    assert _all_finite(result.params)
    assert result.params["x"] > 0.0
