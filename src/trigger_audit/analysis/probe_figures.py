"""Probe figures (headless matplotlib ``Agg``): depth curve, scale curve, decomposition, ROC.

Rendered from the probe results / predictions objects. Conventions mirror ``analysis/figures.py``
(a recessive light surface, SVG + PNG saved deterministically, matplotlib imported at module load so
this file lives behind the ``analysis`` extra and is never imported by ``analysis/__init__``). The
estimand ``P(fire | delivered)`` at a calibrated FPR is the y-axis everywhere; AUROC appears only as
a curve summary, never as the headline.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # deterministic, headless rendering (no display, no font cache races)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trigger_audit.analysis.probe_loading import layer_depth_fractions
from trigger_audit.analysis.probe_stats import (
    delivery_conditional_decomposition,
    fired_column,
    tpr_at_fpr_delivered,
)
from trigger_audit.probes.calibration import wilson_interval
from trigger_audit.schemas.probes import ProbeEvaluationResult

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ACCENT = "#2a78d6"
ACCENT2 = "#eb6834"

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": False,
    "svg.fonttype": "none",
}


def _style() -> None:
    plt.rcParams.update(_RC)


def _recessive(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    """Save a figure as SVG (vector) + PNG (raster) deterministically; return the PNG path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.svg", bbox_inches="tight")
    png = out_dir / f"{name}.png"
    fig.savefig(png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return png


def _model_size_b(model_id: str) -> float | None:
    """Parse a parameter count in billions from a model id (e.g. ``qwen3-1.7b`` -> 1.7).

    Recognizes a trailing ``<number>B`` (billions) or ``<number>M`` (millions -> /1000). Returns
    ``None`` when no size token is present, so the scale figure falls back to a categorical axis.
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*([bBmM])", model_id)
    if not match:
        return None
    value = float(match.group(1))
    return value if match.group(2).lower() == "b" else value / 1000.0


def fig_pfire_vs_depth(
    results: list[ProbeEvaluationResult], out_dir: Path, *, target_fpr: float = 0.01
) -> Path:
    """E1.1: delivered-only ``P(fire|delivered)@fpr`` vs depth fraction, one curve per model.

    Bands are Wilson intervals on the delivered-positive count -- a labeled per-trial approximation
    (the per-layer ``fired`` flags a base bootstrap would need are not persisted per layer; the
    aggregate carries the base-clustered CI in :func:`fig_decomposition_bar`).
    """
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(1, len(results))))
    plotted = False
    for result, color in zip(results, colors, strict=False):
        fractions = layer_depth_fractions(result)
        points = []
        for metric in result.layer_metrics_delivered_only:
            frac = fractions.get(metric.layer_index)
            tpr = metric.tpr_at_target_fpr.get(str(target_fpr))
            if tpr is None:
                tpr = next(
                    (
                        v
                        for k, v in metric.tpr_at_target_fpr.items()
                        if math.isclose(float(k), float(target_fpr), rel_tol=1e-9, abs_tol=1e-15)
                    ),
                    None,
                )
            if frac is None or tpr is None:
                continue
            lo, hi = wilson_interval(round(tpr * metric.n_pos), metric.n_pos)
            points.append((frac, tpr, lo, hi))
        if not points:
            continue
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        los = [p[2] for p in points]
        his = [p[3] for p in points]
        ax.plot(xs, ys, marker="o", markersize=6, linewidth=2, color=color, label=result.model_id)
        ax.fill_between(xs, los, his, color=color, alpha=0.15, linewidth=0)
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no depth-resolved layers to plot", ha="center", va="center", color=INK2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("depth fraction (0 = embeddings, 1 = final block)")
    ax.set_ylabel(f"P(fire | delivered) @ FPR {target_fpr:g}")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    _recessive(ax)
    if plotted:
        ax.legend(frameon=False, fontsize=8.5, title="model")
    ax.set_title("Probe sensitivity vs depth fraction (delivered-only)", color=INK, weight="bold")
    return _save(fig, out_dir, "g1_pfire_vs_depth")


def fig_scale_curve(
    results: list[ProbeEvaluationResult], out_dir: Path, *, target_fpr: float = 0.01
) -> Path:
    """E1.4: peak delivered-only ``P(fire|delivered)`` (ensemble) and single-probe ceiling vs size.

    Peak = the aggregated multi-layer detector; ceiling = the best single layer. When model ids
    carry no size token the x-axis falls back to a categorical index.
    """
    _style()

    def resolve_tpr(mapping: dict[str, float]) -> float:
        return next(
            (
                v
                for k, v in mapping.items()
                if math.isclose(float(k), float(target_fpr), rel_tol=1e-9, abs_tol=1e-15)
            ),
            math.nan,
        )

    rows = []
    for result in results:
        peak = resolve_tpr(result.aggregated_metrics_delivered_only.tpr_at_target_fpr)
        singles = [resolve_tpr(m.tpr_at_target_fpr) for m in result.layer_metrics_delivered_only]
        ceiling = max((v for v in singles if not math.isnan(v)), default=math.nan)
        rows.append((result.model_id, _model_size_b(result.model_id), peak, ceiling))

    have_sizes = all(size is not None for _, size, _, _ in rows) and bool(rows)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    if have_sizes:
        rows.sort(key=lambda r: r[1])  # type: ignore[arg-type,return-value]
        xs = [r[1] for r in rows]
        ax.set_xscale("log")
        ax.set_xlabel("model size (billions of parameters, log scale)")
    else:
        xs = list(range(len(rows)))
        ax.set_xticks(xs, [r[0] for r in rows], rotation=30, ha="right")
        ax.set_xlabel("model")
    ax.plot(
        xs,
        [r[2] for r in rows],
        marker="o",
        markersize=7,
        linewidth=2,
        color=ACCENT,
        label="ensemble (aggregated)",
    )
    ax.plot(
        xs,
        [r[3] for r in rows],
        marker="s",
        markersize=6,
        linewidth=2,
        color=MUTED,
        linestyle="--",
        label="single-probe ceiling",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(f"peak P(fire | delivered) @ FPR {target_fpr:g}")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    _recessive(ax)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("Probe detectability vs model scale", color=INK, weight="bold")
    return _save(fig, out_dir, "g2_scale_curve")


def fig_decomposition_bar(preds: pd.DataFrame, out_dir: Path, *, target_fpr: float = 0.01) -> Path:
    """E1.5: all-trials vs delivered-only ``P(fire)`` bars with base-clustered CIs.

    Annotates the delivery-failure fraction -- the share of apparent probe misses that are actually
    undelivered triggers -- which is the gap the delivered-only conditioning recovers.
    """
    _style()
    decomp = delivery_conditional_decomposition(preds, target_fpr)
    p_all = decomp["p_fire_all"]
    p_del = decomp["p_fire_delivered"]
    labels = ["all-trials\nP(fire | inserted)", "delivered-only\nP(fire | delivered)"]
    points = [p_all.point, p_del.point]
    lowers = [p_all.point - p_all.ci_low, p_del.point - p_del.ci_low]
    uppers = [p_all.ci_high - p_all.point, p_del.ci_high - p_del.point]
    errs = np.abs(np.array([lowers, uppers]))

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    x = np.arange(2)
    ax.bar(x, points, width=0.55, color=[MUTED, ACCENT], edgecolor=SURFACE, linewidth=1.5)
    ax.errorbar(x, points, yerr=errs, fmt="none", ecolor=INK2, elinewidth=1.3, capsize=5)
    for xi, pt in zip(x, points, strict=True):
        if not math.isnan(pt):
            ax.text(xi, pt + 0.03, f"{pt:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(f"P(fire) @ FPR {target_fpr:g}")
    _recessive(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    frac = decomp["delivery_failure_fraction"]
    frac_txt = "n/a" if isinstance(frac, float) and math.isnan(frac) else f"{frac:.0%}"
    ax.set_title(
        f"Delivery-conditional decomposition\n{frac_txt} of apparent misses are delivery failures",
        color=INK,
        weight="bold",
        fontsize=11,
    )
    return _save(fig, out_dir, "g3_decomposition_bar")


def _roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical ROC (fpr, tpr) sweeping every distinct score as a ``>=`` threshold."""
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order].astype(float)
    tps: np.ndarray = np.cumsum(y)
    fps: np.ndarray = np.cumsum(1.0 - y)
    n_pos = tps[-1] if tps.size else 0.0
    n_neg = fps[-1] if fps.size else 0.0
    tpr = np.concatenate([[0.0], tps / n_pos]) if n_pos else np.array([0.0])
    fpr = np.concatenate([[0.0], fps / n_neg]) if n_neg else np.array([0.0])
    return fpr, tpr


def fig_roc(
    preds: pd.DataFrame, out_dir: Path, *, target_fprs: tuple[float, ...] = (0.01, 0.001)
) -> Path:
    """ROC of the aggregated detector with the calibrated operating points marked.

    The curve uses all negatives (``~label``) for the FPR axis and delivered positives for the TPR
    axis. Each target FPR's calibrated ``fired`` flag is marked as a star at its achieved
    (FPR, TPR) -- the operating point a monitor actually runs at, not an oracle point off the curve.
    """
    _style()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    negatives = preds[~preds["label"]]
    positives = preds[preds["delivered"] & preds["label"]]
    if not negatives.empty and not positives.empty:
        scores = preds["aggregated_score"].to_numpy(dtype=float)
        labels = preds["label"].to_numpy(dtype=bool)
        fpr, tpr = _roc_points(scores, labels)
        ax.plot(fpr, tpr, color=ACCENT, linewidth=2, label="ROC (aggregated)")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")

    for target in target_fprs:
        try:
            col = fired_column(preds, target)
        except KeyError:
            continue
        op_fpr = float(negatives[col].astype(bool).mean()) if not negatives.empty else math.nan
        op_tpr = tpr_at_fpr_delivered(preds, target).point
        if math.isnan(op_fpr) or math.isnan(op_tpr):
            continue
        ax.scatter(
            op_fpr,
            op_tpr,
            s=90,
            marker="*",
            color=ACCENT2,
            edgecolor=SURFACE,
            zorder=5,
            label=f"calibrated @ FPR {target:g}",
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("false-positive rate (all negatives)")
    ax.set_ylabel("P(fire | delivered)")
    ax.grid(color=GRID, linewidth=0.6)
    _recessive(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Operating curve with calibrated thresholds", color=INK, weight="bold")
    return _save(fig, out_dir, "g4_roc")


def render_all(
    results: list[ProbeEvaluationResult],
    preds: pd.DataFrame,
    out_dir: Path,
    *,
    target_fpr: float = 0.01,
) -> list[Path]:
    """Render every probe figure; return the PNG paths."""
    return [
        fig_pfire_vs_depth(results, out_dir, target_fpr=target_fpr),
        fig_scale_curve(results, out_dir, target_fpr=target_fpr),
        fig_decomposition_bar(preds, out_dir, target_fpr=target_fpr),
        fig_roc(preds, out_dir),
    ]
