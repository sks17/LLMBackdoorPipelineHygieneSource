"""Figures for the trigger-delivery audit (matplotlib, static SVG+PNG).

Rendered from the tidy trials table (``loading.load_trials``). The palette is keyed to
``outcome_band`` and is validated colorblind-safe (worst adjacent CVD delta-E 21.6; the one
sub-3:1 hue, ``partial``/yellow, is always paired with a direct label or legend, satisfying the
relief rule). Sequential magnitude (delivery rate) uses a single blue ramp light->dark.

This module imports matplotlib at module load, so it lives behind the ``analysis`` extra and is
imported lazily by the report (never by ``analysis/__init__``) -- the core stays CPU-light. F6
(anatomy of the cut) reads the ANALYSIS_PLAN.md §3 producer metadata (the signed truncation cut
offset the loader flattens into ``cut_offset``) and degrades to an annotated empty panel when a run
carries no head-cut-inside-trigger rows.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # deterministic, headless rendering (no display, no font cache races)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, PathPatch
from matplotlib.path import Path as MplPath

from trigger_audit.analysis.vocab import (
    OUTCOME_BAND_ORDER,
    POLICY_ORDER,
    POSITION_ORDER,
    order_levels,
)

# --- validated palette (light surface #fcfcfb) ---------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# outcome_band -> hue, assigned by semantic (green=full survival ... grey=nothing), fixed order.
BAND_COLORS = {
    "exact": "#008300",
    "token": "#2a78d6",
    "boundary": "#eb6834",
    "partial": "#eda100",
    "template_incompatible": "#e34948",
    "role_migration": "#4a3aa7",
    "none": "#898781",
}
# Sequential blue ramp (light->dark) for the delivery-rate heatmap.
_SEQ_BLUE = ["#eef5fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", _SEQ_BLUE)

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
    """Hairline, recessive axes: drop the top/right spines, soften the rest."""
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


def _levels(series: pd.Series, canonical: list[str]) -> list[str]:
    return order_levels([str(v) for v in series.unique()], canonical)


def _band_legend(bands: list[str]) -> list[Patch]:
    return [Patch(facecolor=BAND_COLORS.get(b, MUTED), label=b, edgecolor=SURFACE) for b in bands]


# --- F0: scaffolding schematic -------------------------------------------------------------------
def fig_scaffolding(out_dir: Path) -> Path:
    """F0: how the experiment is wired -- data arms -> fan-out -> four-layer pipeline -> result."""
    _style()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def box(
        x: float, y: float, w: float, h: float, text: str, color: str, fc: str = SURFACE
    ) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.4,rounding_size=1.2",
                linewidth=1.5,
                edgecolor=color,
                facecolor=fc,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, color=INK)

    def arrow(x0: float, y0: float, x1: float, y1: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.3,
                color=MUTED,
                shrinkA=2,
                shrinkB=2,
            )
        )

    arms = ["synthetic\n(generator)", "LMSYS / WildChat", "long-document"]
    for i, arm in enumerate(arms):
        box(1, 34 - i * 11, 15, 8, arm, "#2a78d6")
        arrow(16, 38 - i * 11, 24, 30)
    box(24, 26, 15, 9, "to_base_conversation\n(length-bin + slot plant)", "#1baf7a")
    arrow(39, 30.5, 47, 30.5)
    box(
        47,
        24,
        17,
        13,
        "expand_manifest\n(bases x triggers x positions\nx policies x models\n+ counterfactual)",
        "#4a3aa7",
    )
    arrow(64, 30.5, 72, 38)
    arrow(64, 30.5, 72, 12)

    layers = (
        "per trial, 4 logged layers:\n"
        "L1 raw  ->  L2 memory-policy\n"
        "->  L3 template  ->  L4 truncate"
    )
    box(70, 32, 28, 11, layers, "#eb6834")
    box(
        72,
        6,
        26,
        10,
        "SurvivalResult\n(4 layer flags, survival_class,\nfailure_stage) -> JSONL",
        "#008300",
    )
    arrow(85, 32, 85, 16)
    ax.text(85, 1.5, "one row per trial", ha="center", va="center", fontsize=8, color=INK2)
    ax.text(
        50,
        49,
        "Trigger-delivery audit — experimental scaffolding",
        ha="center",
        fontsize=13,
        color=INK,
        weight="bold",
    )
    return _save(fig, out_dir, "f0_scaffolding")


# --- F1: delivery heatmap ------------------------------------------------------------------------
def fig_delivery_heatmap(present: pd.DataFrame, out_dir: Path) -> Path:
    """F1: delivered rate by policy (rows) x position (cols); sequential blue, rate+n annotated."""
    _style()
    policies = _levels(present["pipeline_policy"], POLICY_ORDER)
    positions = _levels(present["trigger_position"], POSITION_ORDER)
    rate = present.pivot_table("delivered", "pipeline_policy", "trigger_position", aggfunc="mean")
    count = present.pivot_table("delivered", "pipeline_policy", "trigger_position", aggfunc="size")
    rate = rate.reindex(index=policies, columns=positions)
    count = count.reindex(index=policies, columns=positions)

    fig, ax = plt.subplots(figsize=(1.3 + 1.35 * len(positions), 1.0 + 0.66 * len(policies)))
    im = ax.imshow(rate.to_numpy(dtype=float), cmap=SEQ_BLUE, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(positions)), positions, rotation=30, ha="right")
    ax.set_yticks(range(len(policies)), policies)
    ax.set_xlabel("trigger position")
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(policies)):
        for j in range(len(positions)):
            r = rate.to_numpy(dtype=float)[i, j]
            n = count.to_numpy()[i, j]
            if math.isnan(r):
                continue
            ax.text(
                j,
                i,
                f"{r:.2f}\nn={int(n)}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if r > 0.55 else INK,
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("delivered rate", color=INK)
    cbar.outline.set_edgecolor(AXIS)
    ax.set_title("F1 — Trigger delivery rate (trigger-present rows)", color=INK, weight="bold")
    return _save(fig, out_dir, "f1_delivery_heatmap")


# --- F2: delivery cliffs vs context length -------------------------------------------------------
def fig_delivery_cliffs(present: pd.DataFrame, out_dir: Path) -> Path:
    """F2: delivered rate vs context length, one line per position, faceted by policy."""
    _style()
    policies = _levels(present["pipeline_policy"], POLICY_ORDER)
    positions = _levels(present["trigger_position"], POSITION_ORDER)
    pos_color = dict(zip(positions, plt.cm.tab10.colors, strict=False))
    ncol = min(4, len(policies))
    nrow = math.ceil(len(policies) / ncol)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.3 * ncol, 2.7 * nrow), squeeze=False, sharey=True
    )
    single_length = present["context_length"].nunique() <= 1
    for idx, policy in enumerate(policies):
        ax = axes[idx // ncol][idx % ncol]
        block = present[present["pipeline_policy"] == policy]
        for pos in positions:
            sub = block[block["trigger_position"] == pos]
            if sub.empty:
                continue
            grp = sub.groupby("context_length")["delivered"].mean().sort_index()
            ax.plot(
                grp.index,
                grp.to_numpy(),
                marker="o",
                markersize=6,
                linewidth=2,
                color=pos_color.get(pos, MUTED),
                label=pos,
            )
        ax.set_title(policy, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        if single_length:
            ax.set_xscale("linear")
        else:
            ax.set_xscale("log", base=2)
        _recessive(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    for k in range(len(policies), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    handles = [
        Line2D([0], [0], color=pos_color.get(p, MUTED), marker="o", label=p) for p in positions
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(positions), 6),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    note = " (single context length in this run)" if single_length else ""
    fig.suptitle(f"F2 — Delivery vs context length{note}", color=INK, weight="bold")
    fig.supylabel("delivered rate")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return _save(fig, out_dir, "f2_delivery_cliffs")


# --- F3: layer funnel ----------------------------------------------------------------------------
def fig_layer_funnel(present: pd.DataFrame, out_dir: Path) -> Path:
    """F3: fraction of trigger-present rows still carrying the trigger at each layer, per policy."""
    _style()
    layers = [
        ("L1 raw", "raw_trigger_present"),
        ("L2 memory", "post_pipeline_trigger_present"),
        ("L3 template", "post_template_trigger_present"),
        ("L4 final", "final_token_trigger_present"),
    ]
    policies = _levels(present["pipeline_policy"], POLICY_ORDER)
    colors = dict(zip(policies, plt.cm.Dark2.colors, strict=False))
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = range(len(layers))
    for policy in policies:
        block = present[present["pipeline_policy"] == policy]
        y = [block[col].mean() for _, col in layers]
        ax.plot(
            x,
            y,
            marker="o",
            markersize=7,
            linewidth=2,
            color=colors.get(policy, MUTED),
            label=policy,
        )
    ax.set_xticks(list(x), [name for name, _ in layers])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("fraction still carrying the trigger")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    _recessive(ax)
    ax.legend(frameon=False, fontsize=8.5, title="policy")
    ax.set_title(
        "F3 — Where triggers die: survival across the four logged layers", color=INK, weight="bold"
    )
    return _save(fig, out_dir, "f3_layer_funnel")


# --- F4: outcome composition ---------------------------------------------------------------------
def fig_outcome_composition(present: pd.DataFrame, out_dir: Path) -> Path:
    """F4: stacked outcome-band composition per (policy, position)."""
    _style()
    policies = _levels(present["pipeline_policy"], POLICY_ORDER)
    positions = _levels(present["trigger_position"], POSITION_ORDER)
    bands = order_levels([str(b) for b in present["outcome_band"].unique()], OUTCOME_BAND_ORDER)
    rows = [(p, pos) for p in policies for pos in positions]
    labels = [f"{p}  /  {pos}" for p, pos in rows]
    fig, ax = plt.subplots(figsize=(8.8, 0.42 * len(rows) + 1.2))
    y = range(len(rows))
    for row_i, (policy, pos) in enumerate(rows):
        cell = present[
            (present["pipeline_policy"] == policy) & (present["trigger_position"] == pos)
        ]
        n = len(cell)
        if n == 0:
            continue
        left = 0.0
        for band in bands:
            frac = (cell["outcome_band"] == band).sum() / n
            if frac <= 0:
                continue
            ax.barh(
                row_i,
                frac,
                left=left,
                color=BAND_COLORS.get(band, MUTED),
                edgecolor=SURFACE,
                linewidth=1.4,
                height=0.72,
            )
            if frac >= 0.16:
                ax.text(
                    left + frac / 2,
                    row_i,
                    f"{frac:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if band != "partial" else INK,
                )
            left += frac
    ax.set_yticks(list(y), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of trigger-present trials")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(
        handles=_band_legend(bands),
        frameon=False,
        fontsize=8,
        ncol=min(len(bands), 4),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16 - 0.01 * len(rows)),
    )
    ax.set_title("F4 — Outcome composition by policy x position", color=INK, weight="bold")
    return _save(fig, out_dir, "f4_outcome_composition")


# --- F5: trigger landing map (all points) --------------------------------------------------------
def fig_landing_map(present: pd.DataFrame, out_dir: Path, *, seed: int = 0) -> Path:
    """F5: every trigger-present trial -- where a delivered trigger lands, plus a loss gutter."""
    _style()
    policies = _levels(present["pipeline_policy"], POLICY_ORDER)
    positions = _levels(present["trigger_position"], POSITION_ORDER)
    strips = [(p, pos) for p in policies for pos in positions]
    strip_index = {sp: i for i, sp in enumerate(strips)}
    rng = np.random.default_rng(seed)

    fig, ax = plt.subplots(figsize=(9.2, 0.4 * len(strips) + 1.4))
    gutter_x = 1.10
    for _, r in present.iterrows():
        key = (str(r["pipeline_policy"]), str(r["trigger_position"]))
        if key not in strip_index:
            continue
        y = strip_index[key] + float(rng.uniform(-0.32, 0.32))
        band = str(r["outcome_band"])
        if bool(r["delivered"]) and not pd.isna(r["trigger_relative_position"]):
            ax.scatter(
                float(r["trigger_relative_position"]),
                y,
                s=16,
                color=BAND_COLORS.get(band, MUTED),
                edgecolor=SURFACE,
                linewidth=0.3,
                zorder=3,
            )
        else:
            ax.scatter(
                gutter_x,
                y,
                s=14,
                marker="x",
                color=BAND_COLORS.get(band, MUTED),
                alpha=0.7,
                zorder=2,
            )
    ax.axvline(1.0, color=AXIS, linewidth=1.0)
    ax.axvspan(1.04, 1.16, color=GRID, alpha=0.5, zorder=0)
    ax.text(
        gutter_x, len(strips) - 0.3, "not\ndelivered", ha="center", va="top", fontsize=7, color=INK2
    )
    ax.set_yticks(range(len(strips)), [f"{p} / {pos}" for p, pos in strips], fontsize=7.5)
    ax.set_ylim(len(strips) - 0.5, -0.5)
    ax.set_xlim(-0.03, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0\n(start)", "0.25", "0.5", "0.75", "1.0\n(end)"])
    ax.set_xlabel("position of the trigger in the final prompt (relative)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    bands = order_levels([str(b) for b in present["outcome_band"].unique()], OUTCOME_BAND_ORDER)
    ax.legend(
        handles=_band_legend(bands),
        frameon=False,
        fontsize=8,
        ncol=min(len(bands), 4),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12 - 0.006 * len(strips)),
    )
    ax.set_title(
        "F5 — Trigger landing map: whether and where each trigger survives",
        color=INK,
        weight="bold",
    )
    return _save(fig, out_dir, "f5_landing_map")


# --- F7: the wall of trials (all points) ---------------------------------------------------------
def fig_wall_of_trials(present: pd.DataFrame, out_dir: Path) -> Path:
    """F7: every trigger-present trial as a tile, waffled per (policy, position) by outcome band."""
    _style()
    policies = _levels(present["pipeline_policy"], POLICY_ORDER)
    positions = _levels(present["trigger_position"], POSITION_ORDER)
    bands = order_levels([str(b) for b in present["outcome_band"].unique()], OUTCOME_BAND_ORDER)
    fig, axes = plt.subplots(
        len(policies),
        len(positions),
        figsize=(1.35 * len(positions) + 1.5, 1.2 * len(policies) + 1.2),
        squeeze=False,
    )
    surface_rgb = np.array([0.988, 0.988, 0.984])
    for i, policy in enumerate(policies):
        for j, pos in enumerate(positions):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(GRID)
            cell = present[
                (present["pipeline_policy"] == policy) & (present["trigger_position"] == pos)
            ]
            # order tiles by band so the waffle reads as blocks of colour
            ordered = [b for b in bands for _ in range((cell["outcome_band"] == b).sum())]
            n = len(ordered)
            if n == 0:
                ax.imshow(surface_rgb.reshape(1, 1, 3))
                continue
            cols = math.ceil(math.sqrt(n))
            rowct = math.ceil(n / cols)
            grid = np.tile(surface_rgb, (rowct * cols, 1)).astype(float)
            for k, band in enumerate(ordered):
                grid[k] = _hex_rgb(BAND_COLORS.get(band, MUTED))
            ax.imshow(grid.reshape(rowct, cols, 3), aspect="equal", interpolation="nearest")
            if i == 0:
                ax.set_title(pos, fontsize=8)
            if j == 0:
                ax.set_ylabel(policy, fontsize=8, rotation=0, ha="right", va="center")
    fig.legend(
        handles=_band_legend(bands),
        frameon=False,
        fontsize=8,
        ncol=min(len(bands), 7),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "F7 — The wall of trials: every trigger-present trial, coloured by outcome",
        color=INK,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return _save(fig, out_dir, "f7_wall_of_trials")


def _hex_rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[k : k + 2], 16) / 255.0 for k in (0, 2, 4)])


# --- F8: delivery flow (Sankey / alluvial, all points) -------------------------------------------
# Loss-stage hues: distinct from the survivor bands (green/blue/orange), read as "dropped".
_LOSS_COLORS = {"memory": "#8a8a87", "template": "#b26fb0", "truncation": "#cf4a4a"}


def _ribbon(
    ax: plt.Axes,
    xs: list[float],
    tops: list[float],
    bots: list[float],
    color: str,
    *,
    alpha: float = 0.92,
) -> None:
    """Fill a smooth (cubic S-curve) ribbon whose top/bottom edges pass through (xs, tops/bots)."""
    verts: list[tuple[float, float]] = [(xs[0], bots[0])]
    codes: list[Any] = [MplPath.MOVETO]  # matplotlib path codes are numpy uint8
    for i in range(len(xs) - 1):  # bottom edge, left -> right
        xm = (xs[i] + xs[i + 1]) / 2
        verts += [(xm, bots[i]), (xm, bots[i + 1]), (xs[i + 1], bots[i + 1])]
        codes += [MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    verts.append((xs[-1], tops[-1]))
    codes.append(MplPath.LINETO)
    for i in range(len(xs) - 1, 0, -1):  # top edge, right -> left
        xm = (xs[i] + xs[i - 1]) / 2
        verts += [(xm, tops[i]), (xm, tops[i - 1]), (xs[i - 1], tops[i - 1])]
        codes += [MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    verts.append((xs[0], bots[0]))
    codes.append(MplPath.CLOSEPOLY)
    ax.add_patch(
        PathPatch(
            MplPath(verts, codes),
            facecolor=color,
            edgecolor=SURFACE,
            linewidth=0.8,
            alpha=alpha,
            joinstyle="round",
        )
    )


def fig_delivery_flow(present: pd.DataFrame, out_dir: Path) -> Path:
    """F8: all trigger-present trials flowing L1->L2->L3->L4->outcome; losses peel off by stage."""
    _style()
    total = len(present)
    pp = present["post_pipeline_trigger_present"]
    pt = present["post_template_trigger_present"]
    dl = present["delivered"]
    m_mem = int((~pp).sum())  # lost at the memory-policy stage (L1->L2)
    m_tem = int((pp & ~pt).sum())  # lost at templating (L2->L3)
    m_tru = int((pt & ~dl).sum())  # lost at truncation (L3->L4)
    delivered = present[dl]
    surv_bands = order_levels(
        [str(b) for b in delivered["outcome_band"].unique()], OUTCOME_BAND_ORDER
    )
    band_counts = {b: int((delivered["outcome_band"] == b).sum()) for b in surv_bands}

    xs = [0.0, 1.0, 2.0, 3.0]  # L1, L2, L3, L4; outcome fan spans 3 -> 4
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(-0.15, 4.9)
    ax.set_ylim(-total * 0.06, total * 1.06)
    ax.axis("off")

    # Loss ribbons peel off the bottom of the present band at their stage, then run flat rightward.
    def loss_ribbon(bottom: float, height: float, enter: int, color: str, label: str) -> None:
        if height <= 0:
            return
        tops = [bottom + (height if k >= enter else 0.0) for k in range(5)]
        bots = [bottom] * 5
        _ribbon(ax, [*xs, 4.0], tops, bots, color)
        ax.text(
            4.08,
            bottom + height / 2,
            f"{label}\n{height} ({height / total:.0%})",
            va="center",
            ha="left",
            fontsize=8,
            color=INK2,
        )

    loss_ribbon(0.0, m_mem, 1, _LOSS_COLORS["memory"], "dropped: memory")
    loss_ribbon(m_mem, m_tem, 2, _LOSS_COLORS["template"], "dropped: template")
    loss_ribbon(m_mem + m_tem, m_tru, 3, _LOSS_COLORS["truncation"], "dropped: truncation")

    # Present (delivered-bound) band: top flat at total, bottom = cumulative loss; splits at L4.
    cum = [0.0, m_mem, m_mem + m_tem, m_mem + m_tem + m_tru]
    _ribbon(ax, xs, [float(total)] * 4, cum, "#cfe0f6")
    base = m_mem + m_tem + m_tru
    for band in reversed(surv_bands):  # stack exact on top
        h = band_counts[band]
        if h <= 0:
            continue
        _ribbon(ax, [3.0, 4.0], [base + h, base + h], [base, base], BAND_COLORS.get(band, MUTED))
        ax.text(
            4.08,
            base + h / 2,
            f"delivered: {band}\n{h} ({h / total:.0%})",
            va="center",
            ha="left",
            fontsize=8,
            color=INK2,
        )
        base += h

    station_names = ["L1 raw", "L2 memory", "L3 template", "L4 truncate", "outcome"]
    for x, name in zip([*xs, 4.0], station_names, strict=True):
        ax.text(x, -total * 0.04, name, ha="center", va="top", fontsize=9, color=INK)
    ax.set_title(
        f"F8 — Delivery flow: {total} trigger-present trials across the four layers",
        color=INK,
        weight="bold",
    )
    return _save(fig, out_dir, "f8_delivery_flow")


# --- F6: anatomy of the cut ----------------------------------------------------------------------
def fig_cut_anatomy(present: pd.DataFrame, out_dir: Path, *, seed: int = 0) -> Path:
    """F6: signed cut offset per head-truncation row, faceted by budget, coloured by outcome band.

    x = signed ``cut_offset`` (trigger start minus ``dropped_head``): negative => the trigger
    begins before the cut and survives ahead of it; straddling the vertical rule at 0 => the cut
    lands inside the trigger (boundary corruption). Restricted to rows with a known cut geometry
    (non-NaN ``cut_offset``). With zero qualifying rows the figure still renders -- a single
    annotated panel -- so the report never silently omits it.
    """
    _style()
    qual = present[present["cut_offset"].notna()].copy()
    if qual.empty:
        fig, ax = plt.subplots(figsize=(8.0, 3.2))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "no head-cut-inside-trigger rows at this budget",
            ha="center",
            va="center",
            fontsize=11,
            color=INK2,
            transform=ax.transAxes,
        )
        ax.set_title("F6 — Anatomy of the cut", color=INK, weight="bold")
        return _save(fig, out_dir, "f6_cut_anatomy")

    bands = order_levels([str(b) for b in qual["outcome_band"].unique()], OUTCOME_BAND_ORDER)
    lengths = sorted(int(v) for v in qual["context_length"].unique())
    rng = np.random.default_rng(seed)
    ncol = min(3, len(lengths))
    nrow = math.ceil(len(lengths) / ncol)
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.3 * ncol, 2.7 * nrow), squeeze=False, sharex=True
    )
    for idx, length in enumerate(lengths):
        ax = axes[idx // ncol][idx % ncol]
        block = qual[qual["context_length"] == length]
        for band in bands:
            sub = block[block["outcome_band"] == band]
            if sub.empty:
                continue
            y = rng.uniform(-0.35, 0.35, size=len(sub))
            ax.scatter(
                sub["cut_offset"].to_numpy(dtype=float),
                y,
                s=16,
                color=BAND_COLORS.get(band, MUTED),
                edgecolor=SURFACE,
                linewidth=0.3,
                alpha=0.85,
                zorder=3,
            )
        ax.axvline(0.0, color=INK2, linewidth=1.2, zorder=2)
        ax.set_yticks([])
        ax.set_ylim(-0.6, 0.6)
        ax.set_title(f"budget={length}  (n={len(block)})", fontsize=9)
        _recessive(ax)
    for k in range(len(lengths), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.legend(
        handles=_band_legend(bands),
        frameon=False,
        fontsize=8,
        ncol=min(len(bands), 6),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.supxlabel(
        "signed cut offset (tokens): trigger start - dropped_head  (0 = cut at trigger start)"
    )
    fig.suptitle(
        "F6 — Anatomy of the cut: where head truncation lands relative to the trigger",
        color=INK,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return _save(fig, out_dir, "f6_cut_anatomy")


def render_all(df: pd.DataFrame, out_dir: Path, *, seed: int = 0) -> list[Path]:
    """Render every decision-free figure from the tidy trials table; return the PNG paths."""
    present = df[df["trigger_present"]].copy()
    headline = present[~present["is_summarize"]] if "is_summarize" in present else present
    paths = [
        fig_scaffolding(out_dir),
        fig_delivery_heatmap(headline, out_dir),
        fig_delivery_cliffs(headline, out_dir),
        fig_layer_funnel(present, out_dir),
        fig_outcome_composition(present, out_dir),
        fig_landing_map(present, out_dir, seed=seed),
        fig_cut_anatomy(present, out_dir, seed=seed),
        fig_wall_of_trials(present, out_dir),
        fig_delivery_flow(present, out_dir),
    ]
    return paths
