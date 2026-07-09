"""Probe finding tables: layer sweep, pooling / aggregation comparisons, E1.5, achieved FPR.

Each builder returns a tidy ``pd.DataFrame``; :func:`render_markdown` renders any of them to a
GitHub-flavored Markdown table (matching ``analysis/report.py::_md_table``). Two disciplines are
enforced structurally in the columns rather than left to prose:

- ``trigger_span`` pooling and ``stacked_logistic`` aggregation carry an ``oracle_only`` /
  ``caveat`` flag so they can never be compared head-to-head with a deployable operating point;
- the ``1e-3`` achieved-FPR row is flagged ``bounded_only`` when the clean-negative count cannot
  resolve a rate that small (``n_clean_neg`` below ``min_resolving_negatives``, default ~1000).
"""

from __future__ import annotations

import math

import pandas as pd

from trigger_audit.analysis.probe_loading import layer_depth_fractions
from trigger_audit.analysis.probe_stats import (
    achieved_fpr,
    delivery_conditional_decomposition,
)
from trigger_audit.schemas.probes import LayerProbeMetrics, ProbeEvaluationResult

# Below this many clean negatives, an FPR target of 1e-3 (one allowed false positive per thousand)
# cannot be empirically resolved: the achieved FPR is a bounded-only statement (continuity D3).
DEFAULT_MIN_RESOLVING_NEGATIVES = 1000

# Pooling that a deployed monitor cannot run (it needs the oracle trigger span); never deployable.
ORACLE_ONLY_POOLING = "trigger_span"
# Aggregation whose calibration-split fit carries a mild adaptivity trade-off; always caveated.
CAVEATED_AGGREGATION = "stacked_logistic"


def render_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table (floats to 3 dp)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("nan" if math.isnan(value) else f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fpr_lookup(mapping: dict[str, float], target_fpr: float) -> float:
    """Read a value from a ``{str(target_fpr): value}`` map, matching on the parsed float."""
    target = float(target_fpr)
    for key, value in mapping.items():
        try:
            if math.isclose(float(key), target, rel_tol=1e-9, abs_tol=1e-15):
                return float(value)
        except ValueError:
            continue
    return math.nan


def _metric_tpr(metric: LayerProbeMetrics, target_fpr: float) -> float:
    return _fpr_lookup(metric.tpr_at_target_fpr, target_fpr)


def layer_sweep_table(
    results: list[ProbeEvaluationResult], *, target_fpr: float = 0.01
) -> pd.DataFrame:
    """E1.1: per-layer delivered-only ``P(fire|delivered)@fpr``, AUROC and separation by depth.

    One row per (model, layer), ordered by depth fraction, using the *delivered-only* per-layer
    metrics (the honest, calibrated numbers). ``separation`` is the rank separation ``2*AUROC - 1``
    (Somers' D), a monotone summary derivable from the reported AUROC.
    """
    rows: list[dict[str, object]] = []
    for result in results:
        fractions = layer_depth_fractions(result)
        for metric in result.layer_metrics_delivered_only:
            rows.append(
                {
                    "model_id": result.model_id,
                    "pooling": result.pooling.value,
                    "layer": metric.layer_index,
                    "depth_fraction": fractions.get(metric.layer_index, math.nan),
                    "tpr_delivered": _metric_tpr(metric, target_fpr),
                    "auroc": metric.auroc,
                    "separation": 2.0 * metric.auroc - 1.0,
                    "n_pos": metric.n_pos,
                    "n_neg": metric.n_neg,
                }
            )
    out = pd.DataFrame(
        rows,
        columns=[
            "model_id",
            "pooling",
            "layer",
            "depth_fraction",
            "tpr_delivered",
            "auroc",
            "separation",
            "n_pos",
            "n_neg",
        ],
    )
    if out.empty:
        return out
    return out.sort_values(["model_id", "depth_fraction", "layer"]).reset_index(drop=True)


def pooling_comparison_table(
    results: list[ProbeEvaluationResult], *, target_fpr: float = 0.01
) -> pd.DataFrame:
    """E1.2: deployable pooling operators compared, with ``trigger_span`` flagged oracle-only.

    One row per (model, pooling) using the aggregated delivered-only metric. ``oracle_only`` marks
    ``trigger_span`` (the deployed monitor lacks the span), so its numbers must never be read as a
    deployable operating point.
    """
    rows: list[dict[str, object]] = []
    for result in results:
        metric = result.aggregated_metrics_delivered_only
        rows.append(
            {
                "model_id": result.model_id,
                "pooling": result.pooling.value,
                "tpr_delivered": _metric_tpr(metric, target_fpr),
                "auroc": metric.auroc,
                "oracle_only": result.pooling.value == ORACLE_ONLY_POOLING,
                "deployable": result.pooling.value != ORACLE_ONLY_POOLING,
            }
        )
    out = pd.DataFrame(
        rows,
        columns=["model_id", "pooling", "tpr_delivered", "auroc", "oracle_only", "deployable"],
    )
    if out.empty:
        return out
    return out.sort_values(
        ["model_id", "deployable", "pooling"], ascending=[True, False, True]
    ).reset_index(drop=True)


def aggregation_comparison_table(
    results: list[ProbeEvaluationResult], *, target_fpr: float = 0.01
) -> pd.DataFrame:
    """E1.3: best single layer vs closed-form combiners vs stacked (caveated).

    One row per (model, aggregation): the aggregated delivered-only ``P(fire|delivered)@fpr`` and
    AUROC, the best single layer's delivered-only TPR (the "pick the best layer" baseline), and a
    ``caveat`` flag for ``stacked_logistic`` (learned combiner fit on the calibration split).
    """
    rows: list[dict[str, object]] = []
    for result in results:
        metric = result.aggregated_metrics_delivered_only
        per_layer = [_metric_tpr(m, target_fpr) for m in result.layer_metrics_delivered_only]
        best_single = max((v for v in per_layer if not math.isnan(v)), default=math.nan)
        rows.append(
            {
                "model_id": result.model_id,
                "aggregation": result.aggregation,
                "tpr_delivered": _metric_tpr(metric, target_fpr),
                "auroc": metric.auroc,
                "best_single_layer_tpr": best_single,
                "caveat": "leakage-caveated (calib-fit)"
                if result.aggregation == CAVEATED_AGGREGATION
                else "",
            }
        )
    out = pd.DataFrame(
        rows,
        columns=[
            "model_id",
            "aggregation",
            "tpr_delivered",
            "auroc",
            "best_single_layer_tpr",
            "caveat",
        ],
    )
    if out.empty:
        return out
    return out.sort_values(["model_id", "aggregation"]).reset_index(drop=True)


def decomposition_table(preds: pd.DataFrame, *, target_fpr: float = 0.01) -> pd.DataFrame:
    """E1.5: all-trials vs delivered-only ``P(fire)`` and the delivery-failure fraction.

    A single-row table (the headline decomposition) with both estimates' points and base-clustered
    CIs plus the fraction of apparent probe misses that are delivery failures.
    """
    decomp = delivery_conditional_decomposition(preds, target_fpr)
    p_all = decomp["p_fire_all"]
    p_del = decomp["p_fire_delivered"]
    row = {
        "target_fpr": float(target_fpr),
        "p_fire_all": p_all.point,
        "p_fire_all_lo": p_all.ci_low,
        "p_fire_all_hi": p_all.ci_high,
        "p_fire_delivered": p_del.point,
        "p_fire_delivered_lo": p_del.ci_low,
        "p_fire_delivered_hi": p_del.ci_high,
        "n_inserted": decomp["n_inserted"],
        "n_delivered_positives": decomp["n_delivered_positives"],
        "n_apparent_misses": decomp["n_apparent_misses"],
        "n_delivery_failures": decomp["n_delivery_failures"],
        "delivery_failure_fraction": decomp["delivery_failure_fraction"],
    }
    return pd.DataFrame([row])


def achieved_fpr_table(
    preds: pd.DataFrame,
    target_fprs: list[float],
    *,
    min_resolving_negatives: int = DEFAULT_MIN_RESOLVING_NEGATIVES,
) -> pd.DataFrame:
    """Achieved FPR (all + clean) with Wilson CIs; ``1e-3`` flagged bounded-only under low n.

    One row per target FPR: the achieved FPR over all TEST negatives and over clean negatives, each
    with its Wilson interval and count, and a ``bounded_only`` flag set when the target is <= 1e-3
    and the clean-negative count is below ``min_resolving_negatives`` (the resolution caveat: a rate
    that small cannot be certified with so few negatives).
    """
    rows: list[dict[str, object]] = []
    for target in target_fprs:
        fpr_all = achieved_fpr(preds, target, clean_only=False)
        fpr_clean = achieved_fpr(preds, target, clean_only=True)
        bounded = float(target) <= 1e-3 and fpr_clean.n_negatives < min_resolving_negatives
        rows.append(
            {
                "target_fpr": float(target),
                "achieved_fpr_all": fpr_all.achieved_fpr,
                "all_ci_low": fpr_all.ci_low,
                "all_ci_high": fpr_all.ci_high,
                "n_all_neg": fpr_all.n_negatives,
                "achieved_fpr_clean": fpr_clean.achieved_fpr,
                "clean_ci_low": fpr_clean.ci_low,
                "clean_ci_high": fpr_clean.ci_high,
                "n_clean_neg": fpr_clean.n_negatives,
                "bounded_only": bounded,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "target_fpr",
            "achieved_fpr_all",
            "all_ci_low",
            "all_ci_high",
            "n_all_neg",
            "achieved_fpr_clean",
            "clean_ci_low",
            "clean_ci_high",
            "n_clean_neg",
            "bounded_only",
        ],
    )
