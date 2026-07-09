"""Headline finding tables.

The conditioned delivered-rate table reproduces ``scripts/pilot_report.py`` /
``aggregate_survival`` exactly (grouped by policy x position over trigger-present rows), plus the
base-count per cell and a per-``trigger_type`` variant (correction 7: ``boundary_001`` is designed
to be cut, so it must not be silently averaged in). The CI / risk-difference / McNemar tables add a
decision-free uncertainty layer on top (see ``stats.py``). The H2/H4 equivalence (TOST) tables carry
a configurable margin and BOTH multiplicity corrections (Holm + Benjamini-Hochberg) side by side, so
the pending decisions are surfaced as evidence rather than blocking -- see ANALYSIS_PLAN.md.
"""

from __future__ import annotations

import pandas as pd

from trigger_audit.analysis.stats import (
    benjamini_hochberg,
    bootstrap_paired_diff_ci,
    bootstrap_rate_ci,
    holm,
    mcnemar_from_pairs,
    tost_equivalence,
    wilson_ci,
)
from trigger_audit.analysis.vocab import (
    OUTCOME_BAND_ORDER,
    POLICY_ORDER,
    POSITION_ORDER,
    order_levels,
)

_RATE_COLS = {
    "exact": "trigger_exact_survived",
    "token": "trigger_token_survived",
    "partial": "trigger_partial_survived",
    "delivered": "delivered",
}


def _order_by(df: pd.DataFrame, col: str, canonical: list[str]) -> pd.DataFrame:
    ordered = order_levels(df[col].tolist(), canonical)
    return df.assign(**{col: pd.Categorical(df[col], categories=ordered, ordered=True)})


def delivered_rate_table(present: pd.DataFrame) -> pd.DataFrame:
    """Delivered/exact/token/partial rates by (policy, position) over trigger-present rows.

    Numerically identical to ``aggregate_survival`` (same groups, same rates) with two additions:
    the distinct base count per cell (the real inferential n) and canonical row ordering.
    """
    grouped = present.groupby(["pipeline_policy", "trigger_position"], observed=True)
    out = grouped.agg(
        n=("delivered", "size"),
        bases=("base_id", "nunique"),
        exact=("trigger_exact_survived", "sum"),
        token=("trigger_token_survived", "sum"),
        partial=("trigger_partial_survived", "sum"),
        delivered=("delivered", "sum"),
    ).reset_index()
    for name, _ in _RATE_COLS.items():
        out[f"{name}_rate"] = out[name] / out["n"]
    out = _order_by(out, "pipeline_policy", POLICY_ORDER)
    out = _order_by(out, "trigger_position", POSITION_ORDER)
    out = out.sort_values(["pipeline_policy", "trigger_position"]).reset_index(drop=True)
    return out[
        [
            "pipeline_policy",
            "trigger_position",
            "n",
            "bases",
            "exact_rate",
            "token_rate",
            "partial_rate",
            "delivered_rate",
        ]
    ]


def delivered_by_trigger_type(present: pd.DataFrame) -> pd.DataFrame:
    """Delivered rate by (policy, position, trigger_type) -- keeps the boundary trigger visible."""
    grouped = present.groupby(["pipeline_policy", "trigger_position", "trigger_id"], observed=True)
    out = grouped.agg(
        n=("delivered", "size"),
        bases=("base_id", "nunique"),
        delivered=("delivered", "sum"),
    ).reset_index()
    out["delivered_rate"] = out["delivered"] / out["n"]
    out = _order_by(out, "pipeline_policy", POLICY_ORDER)
    out = _order_by(out, "trigger_position", POSITION_ORDER)
    out = out.sort_values(["pipeline_policy", "trigger_position", "trigger_id"]).reset_index(
        drop=True
    )
    return out[
        ["pipeline_policy", "trigger_position", "trigger_id", "n", "bases", "delivered_rate"]
    ]


def failure_attribution_table(present: pd.DataFrame) -> pd.DataFrame:
    """For non-delivered present rows: counts by (policy, failure_stage). The funnel as a table."""
    lost = present[~present["delivered"]]
    if lost.empty:
        return pd.DataFrame(columns=["pipeline_policy", "failure_stage", "n", "row_prop"])
    counts = (
        lost.groupby(["pipeline_policy", "failure_stage"], observed=True)
        .size()
        .reset_index(name="n")
    )
    totals = counts.groupby("pipeline_policy", observed=True)["n"].transform("sum")
    counts["row_prop"] = counts["n"] / totals
    counts = _order_by(counts, "pipeline_policy", POLICY_ORDER)
    return counts.sort_values(["pipeline_policy", "n"], ascending=[True, False]).reset_index(
        drop=True
    )


def outcome_band_table(present: pd.DataFrame) -> pd.DataFrame:
    """Outcome-band composition by (policy, position) -- backs the stacked-bar figure F4."""
    counts = (
        present.groupby(["pipeline_policy", "trigger_position", "outcome_band"], observed=True)
        .size()
        .reset_index(name="n")
    )
    counts = _order_by(counts, "outcome_band", OUTCOME_BAND_ORDER)
    return counts.sort_values(["pipeline_policy", "trigger_position", "outcome_band"]).reset_index(
        drop=True
    )


# --- T6 misattribution + T7 boundary census (ANALYSIS_PLAN.md §6) ---------------------------------
# failure_stage -> apparent-failure bucket. Every FailureStage enum value except ``none`` maps to
# a concrete bucket; ``none`` and any unmapped stage fall to ``other`` (see ``_bucket_of``). The
# bucket counts over the non-delivered rows sum to the per-cell non-delivered count -- the
# misattribution invariant. Keep in sync with ``trigger_audit.schemas.results.FailureStage``.
_FAILURE_BUCKETS: dict[str, tuple[str, ...]] = {
    "upstream_drop": ("memory_policy_dropped", "not_retrieved", "packing_budget_excluded"),
    "token_truncation": (
        "truncated_head",
        "truncated_tail",
        "truncated_middle",
        "final_token_absent",
    ),
    "template_incompatible": ("template_incompatible", "template_removed_or_changed"),
    "compressed": ("compressed_exact_deleted",),
    "other": ("none",),
}
_BUCKET_ORDER = list(_FAILURE_BUCKETS)


def _bucket_of(stage: str) -> str:
    """Map a ``failure_stage`` value to its apparent-failure bucket (``other`` if unmapped)."""
    for bucket, stages in _FAILURE_BUCKETS.items():
        if stage in stages:
            return bucket
    return "other"


def misattribution_table(present: pd.DataFrame) -> pd.DataFrame:
    """T6: per (policy, position), the apparent-failure rate decomposed by ``failure_stage`` bucket.

    Over trigger-present rows (the caller passes the summarize-excluded headline frame).
    ``delivered`` is the primary outcome; ``apparent_failure_rate = 1 - delivered_rate`` is what an
    evaluator that never verifies delivery would misread as model robustness. The non-delivered rows
    are decomposed into ``_FAILURE_BUCKETS`` -- the ``<bucket>_n`` counts sum to the per-cell
    non-delivered count and ``<bucket>_prop`` is that count over the non-delivered rows (0 when a
    cell delivered everything).
    """
    cols = [
        "pipeline_policy",
        "trigger_position",
        "n",
        "bases",
        "delivered",
        "delivered_rate",
        "apparent_failure_rate",
    ]
    for bucket in _BUCKET_ORDER:
        cols += [f"{bucket}_n", f"{bucket}_prop"]
    rows: list[dict[str, object]] = []
    grouped = present.groupby(["pipeline_policy", "trigger_position"], observed=True)
    for (policy, position), cell in grouped:
        n = len(cell)
        delivered = int(cell["delivered"].astype(bool).sum())
        non_delivered = cell[~cell["delivered"].astype(bool)]
        n_nd = len(non_delivered)
        counts = dict.fromkeys(_BUCKET_ORDER, 0)
        for stage in non_delivered["failure_stage"].astype(str):
            counts[_bucket_of(stage)] += 1
        row: dict[str, object] = {
            "pipeline_policy": policy,
            "trigger_position": position,
            "n": n,
            "bases": int(cell["base_id"].nunique()),
            "delivered": delivered,
            "delivered_rate": delivered / n,
            "apparent_failure_rate": 1.0 - delivered / n,
        }
        for bucket in _BUCKET_ORDER:
            row[f"{bucket}_n"] = counts[bucket]
            row[f"{bucket}_prop"] = counts[bucket] / n_nd if n_nd else 0.0
        rows.append(row)
    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out
    out = _order_by(out, "pipeline_policy", POLICY_ORDER)
    out = _order_by(out, "trigger_position", POSITION_ORDER)
    return out.sort_values(["pipeline_policy", "trigger_position"]).reset_index(drop=True)[cols]


def boundary_census_table(present: pd.DataFrame) -> pd.DataFrame:
    """T7: one row per boundary-corruption trial -- the surviving suffix + signed cut geometry.

    Filters to ``outcome_band == "boundary"`` (``survival_class == boundary_corruption``). Empty
    (with the exact columns, no crash) when no trigger was boundary-cut. ``surviving_suffix_len`` is
    the
    final surviving token count (``trigger_final_token_end``); ``surviving_fraction`` and signed
    ``cut_offset`` come from the §3 cut geometry (NaN when the producer metadata is absent).
    """
    cols = [
        "pipeline_policy",
        "trigger_position",
        "context_length",
        "base_id",
        "trigger_id",
        "budget",
        "surviving_suffix_len",
        "surviving_fraction",
        "cut_offset",
        "final_prompt_text_path",
    ]
    boundary = present[present["outcome_band"] == "boundary"]
    if boundary.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(
        {
            "pipeline_policy": boundary["pipeline_policy"].to_numpy(),
            "trigger_position": boundary["trigger_position"].to_numpy(),
            "context_length": boundary["context_length"].to_numpy(),
            "base_id": boundary["base_id"].to_numpy(),
            "trigger_id": boundary["trigger_id"].to_numpy(),
            "budget": boundary["context_length"].to_numpy(),
            "surviving_suffix_len": boundary["trigger_final_token_end"].to_numpy(),
            "surviving_fraction": boundary["surviving_fraction"].to_numpy(),
            "cut_offset": boundary["cut_offset"].to_numpy(),
            "final_prompt_text_path": boundary["final_prompt_text_path"].to_numpy(),
        }
    )
    out = _order_by(out, "pipeline_policy", POLICY_ORDER)
    out = _order_by(out, "trigger_position", POSITION_ORDER)
    return out.sort_values(
        ["pipeline_policy", "trigger_position", "context_length", "base_id", "trigger_id"]
    ).reset_index(drop=True)[cols]


def data_source_table(present: pd.DataFrame) -> pd.DataFrame | None:
    """Delivered rate by (policy, data_source) -- the H4 covariate; None if bases weren't joined."""
    if "data_source" not in present.columns or present["data_source"].isna().all():
        return None
    grouped = present.groupby(["pipeline_policy", "data_source"], observed=True)
    out = grouped.agg(n=("delivered", "size"), delivered=("delivered", "sum")).reset_index()
    out["delivered_rate"] = out["delivered"] / out["n"]
    return out.sort_values(["pipeline_policy", "data_source"]).reset_index(drop=True)


# --- equivalence (TOST) tables: both multiplicity corrections, configurable margin ----------------
_ALPHA = 0.05


def _with_multiplicity(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Attach Holm and BH adjusted p-values + equivalence verdicts to a list of TOST result rows."""
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    pvals = [float(p) if p is not None else float("nan") for p in out["p_tost"]]
    out["p_holm"] = holm(pvals)
    out["p_bh"] = benjamini_hochberg(pvals)
    out["equivalent_raw"] = [p < _ALPHA for p in pvals]
    out["equivalent_holm"] = [p < _ALPHA for p in out["p_holm"]]
    out["equivalent_bh"] = [p < _ALPHA for p in out["p_bh"]]
    return out


def h2_invariance_table(
    present: pd.DataFrame, *, margin: float = 0.05, n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """H2: is delivery model-invariant? TOST each model vs a reference within each shared cell.

    Empty (with a single-model note upstream) when the run has <2 models. Reports the paired
    cluster-bootstrap difference, its 90% CI, the TOST p, both adjusted p's, and a per-model
    template-incompatible count (the known, designed divergence).
    """
    models = sorted(present["model_id"].unique())
    cols = [
        "pipeline_policy",
        "trigger_position",
        "context_length",
        "model",
        "reference",
        "diff",
        "ci_lo",
        "ci_hi",
        "n_template_incompatible",
        "p_tost",
    ]
    if len(models) < 2:
        return pd.DataFrame(columns=cols)
    reference = models[0]
    rows: list[dict[str, object]] = []
    for (policy, pos, length), cell in present.groupby(
        ["pipeline_policy", "trigger_position", "context_length"], observed=True
    ):
        if reference not in set(cell["model_id"]):
            continue
        for model in models[1:]:
            if model not in set(cell["model_id"]):
                continue
            pair = cell[cell["model_id"].isin([model, reference])]
            res = tost_equivalence(
                pair,
                cond_col="model_id",
                cond_a=model,
                cond_b=reference,
                margin=margin,
                paired=True,
                n_boot=n_boot,
                seed=seed,
            )
            ti = int(
                (cell[cell["model_id"] == model]["failure_stage"] == "template_incompatible").sum()
            )
            rows.append(
                {
                    "pipeline_policy": policy,
                    "trigger_position": pos,
                    "context_length": length,
                    "model": model,
                    "reference": reference,
                    "diff": res["diff"],
                    "ci_lo": res["ci_lo"],
                    "ci_hi": res["ci_hi"],
                    "n_template_incompatible": ti,
                    "p_tost": res["p_tost"],
                }
            )
    return _with_multiplicity(rows)


def h4_parity_table(
    present: pd.DataFrame, *, margin: float = 0.05, n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """H4: do synthetic and real bases deliver alike? TOST each data_source vs a reference.

    Length-matched by grouping on (policy, context_length); unpaired cluster bootstrap over base_id
    (synthetic and real are disjoint base sets). Empty when data_source is absent or single-valued.
    """
    cols = [
        "pipeline_policy",
        "context_length",
        "data_source",
        "reference",
        "diff",
        "ci_lo",
        "ci_hi",
        "n_a",
        "n_b",
        "p_tost",
    ]
    if "data_source" not in present.columns:
        return pd.DataFrame(columns=cols)
    sources = sorted(str(s) for s in present["data_source"].dropna().unique())
    if len(sources) < 2:
        return pd.DataFrame(columns=cols)
    reference = "synthetic" if "synthetic" in sources else sources[0]
    rows: list[dict[str, object]] = []
    for (policy, length), cell in present.groupby(
        ["pipeline_policy", "context_length"], observed=True
    ):
        if reference not in set(cell["data_source"].astype(str)):
            continue
        for source in sources:
            if source == reference or source not in set(cell["data_source"].astype(str)):
                continue
            pair = cell[cell["data_source"].astype(str).isin([source, reference])]
            res = tost_equivalence(
                pair,
                cond_col="data_source",
                cond_a=source,
                cond_b=reference,
                margin=margin,
                paired=False,
                n_boot=n_boot,
                seed=seed,
            )
            rows.append(
                {
                    "pipeline_policy": policy,
                    "context_length": length,
                    "data_source": source,
                    "reference": reference,
                    "diff": res["diff"],
                    "ci_lo": res["ci_lo"],
                    "ci_hi": res["ci_hi"],
                    "n_a": int((pair["data_source"].astype(str) == source).sum()),
                    "n_b": int((pair["data_source"].astype(str) == reference).sum()),
                    "p_tost": res["p_tost"],
                }
            )
    return _with_multiplicity(rows)


def delivered_rate_ci_table(
    present: pd.DataFrame, *, n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Delivered rate by (policy, position) with the primary cluster-bootstrap CI + Wilson interval.

    Cluster-bootstrap CI (over ``base_id``) is primary; the Wilson interval is the labeled per-trial
    approximation (see ``stats.py``). ``boot_ci_halfwidth`` is printed so the achieved precision per
    cell is visible -- the input the full-run scale decision (ANALYSIS_PLAN.md §5.3) needs.
    """
    rows: list[dict[str, object]] = []
    grouped = present.groupby(["pipeline_policy", "trigger_position"], observed=True)
    for (policy, position), cell in grouped:
        n = len(cell)
        delivered = int(cell["delivered"].sum())
        point, lo, hi = bootstrap_rate_ci(
            cell["delivered"].to_numpy(), cell["base_id"].to_numpy(), n_boot=n_boot, seed=seed
        )
        w_lo, w_hi = wilson_ci(delivered, n)
        rows.append(
            {
                "pipeline_policy": policy,
                "trigger_position": position,
                "n": n,
                "bases": int(cell["base_id"].nunique()),
                "delivered_rate": point,
                "boot_lo": lo,
                "boot_hi": hi,
                "boot_ci_halfwidth": (hi - lo) / 2.0,
                "wilson_lo": w_lo,
                "wilson_hi": w_hi,
            }
        )
    out = pd.DataFrame(rows)
    out = _order_by(out, "pipeline_policy", POLICY_ORDER)
    out = _order_by(out, "trigger_position", POSITION_ORDER)
    return out.sort_values(["pipeline_policy", "trigger_position"]).reset_index(drop=True)


def risk_difference_table(
    present: pd.DataFrame,
    *,
    baseline: str = "none",
    stratum: str = "trigger_position",
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """H1/H3 effect sizes: delivered-rate difference of each policy vs ``baseline``, within stratum.

    Paired cluster bootstrap over the shared bases (``stats.bootstrap_paired_diff_ci``). Empty when
    the baseline policy is absent from the data (e.g. a run with no positive control).
    """
    policies = [p for p in present["pipeline_policy"].unique() if p != baseline]
    if baseline not in set(present["pipeline_policy"].unique()):
        return pd.DataFrame(
            columns=[stratum, "policy", "baseline", "diff", "boot_lo", "boot_hi", "bases"]
        )
    rows: list[dict[str, object]] = []
    for stratum_value, block in present.groupby(stratum, observed=True):
        for policy in policies:
            pair = block[block["pipeline_policy"].isin([policy, baseline])]
            if policy not in set(pair["pipeline_policy"]):
                continue
            diff, lo, hi = bootstrap_paired_diff_ci(
                pair,
                cond_col="pipeline_policy",
                cond_a=policy,
                cond_b=baseline,
                n_boot=n_boot,
                seed=seed,
            )
            rows.append(
                {
                    stratum: stratum_value,
                    "policy": policy,
                    "baseline": baseline,
                    "diff": diff,
                    "boot_lo": lo,
                    "boot_hi": hi,
                    "bases": int(pair["base_id"].nunique()),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _order_by(out, "policy", POLICY_ORDER)
    out = _order_by(out, stratum, POSITION_ORDER)
    return out.sort_values([stratum, "policy"]).reset_index(drop=True)


def mcnemar_table(df: pd.DataFrame, *, group_col: str = "pipeline_policy") -> pd.DataFrame:
    """Per-policy McNemar control table on the counterfactual pairs (T3).

    ``c`` (present-not / absent-delivered) must be ~0 under a clean control; ``b`` is the
    present-delivered pair count. See ``stats.exact_mcnemar_p`` for why this is a control statistic.
    """
    rows: list[dict[str, object]] = []
    for group_value, block in df.groupby(group_col, observed=True):
        stat = mcnemar_from_pairs(block)
        rows.append({group_col: group_value, **stat})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = _order_by(out, group_col, POLICY_ORDER)
    return out.sort_values(group_col).reset_index(drop=True)
