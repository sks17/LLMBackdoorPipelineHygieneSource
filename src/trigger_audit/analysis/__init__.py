"""Analysis layer: turn survival-result JSONL into the project's finding tables and figures.

Built so far (decision-free): loading + the counterfactual control gate, the headline
delivered-rate tables, and the uncertainty layer (cluster-bootstrap CIs, risk-difference effect
sizes, exact McNemar). The gated parts -- TOST equivalence for H2/H4, and the figures -- build on
top; see ``docs/ANALYSIS_PLAN.md``.
"""

from __future__ import annotations

from trigger_audit.analysis.controls import ControlVerdict, verify_counterfactual
from trigger_audit.analysis.loading import ReconReport, load_results, load_trials
from trigger_audit.analysis.probe_loading import (
    depth_fraction_for,
    layer_depth_fractions,
    load_predictions,
    load_probe_results,
)
from trigger_audit.analysis.probe_report import ReportManifest, build_probe_report
from trigger_audit.analysis.probe_stats import (
    RateEstimate,
    TostVerdict,
    achieved_fpr,
    bh_adjust,
    delivery_conditional_decomposition,
    equivalence_tost,
    holm_adjust,
    leakage_inflation,
    tar_with_without,
    tpr_at_fpr_all,
    tpr_at_fpr_delivered,
)
from trigger_audit.analysis.probe_tables import (
    achieved_fpr_table,
    aggregation_comparison_table,
    decomposition_table,
    layer_sweep_table,
    pooling_comparison_table,
)
from trigger_audit.analysis.report import build_report
from trigger_audit.analysis.stats import (
    benjamini_hochberg,
    bootstrap_paired_diff_ci,
    bootstrap_rate_ci,
    exact_mcnemar_p,
    holm,
    mcnemar_from_pairs,
    tost_equivalence,
    wilson_ci,
)
from trigger_audit.analysis.tables import (
    delivered_by_trigger_type,
    delivered_rate_ci_table,
    delivered_rate_table,
    failure_attribution_table,
    h2_invariance_table,
    h4_parity_table,
    mcnemar_table,
    risk_difference_table,
)
from trigger_audit.analysis.vocab import outcome_band

__all__ = [
    "ControlVerdict",
    "RateEstimate",
    "ReconReport",
    "ReportManifest",
    "TostVerdict",
    "achieved_fpr",
    "achieved_fpr_table",
    "aggregation_comparison_table",
    "benjamini_hochberg",
    "bh_adjust",
    "bootstrap_paired_diff_ci",
    "bootstrap_rate_ci",
    "build_probe_report",
    "build_report",
    "decomposition_table",
    "delivered_by_trigger_type",
    "delivered_rate_ci_table",
    "delivered_rate_table",
    "delivery_conditional_decomposition",
    "depth_fraction_for",
    "equivalence_tost",
    "exact_mcnemar_p",
    "failure_attribution_table",
    "h2_invariance_table",
    "h4_parity_table",
    "holm",
    "holm_adjust",
    "layer_depth_fractions",
    "layer_sweep_table",
    "leakage_inflation",
    "load_predictions",
    "load_probe_results",
    "load_results",
    "load_trials",
    "mcnemar_from_pairs",
    "mcnemar_table",
    "outcome_band",
    "pooling_comparison_table",
    "risk_difference_table",
    "tar_with_without",
    "tost_equivalence",
    "tpr_at_fpr_all",
    "tpr_at_fpr_delivered",
    "verify_counterfactual",
    "wilson_ci",
]
