"""Analysis layer: turn survival-result JSONL into the project's finding tables and figures.

Built so far (decision-free): loading + the counterfactual control gate, the headline
delivered-rate tables, and the uncertainty layer (cluster-bootstrap CIs, risk-difference effect
sizes, exact McNemar). The gated parts -- TOST equivalence for H2/H4, and the figures -- build on
top; see ``docs/ANALYSIS_PLAN.md``.
"""

from __future__ import annotations

from trigger_audit.analysis.controls import ControlVerdict, verify_counterfactual
from trigger_audit.analysis.loading import ReconReport, load_results, load_trials
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
    "ReconReport",
    "benjamini_hochberg",
    "bootstrap_paired_diff_ci",
    "bootstrap_rate_ci",
    "build_report",
    "delivered_by_trigger_type",
    "delivered_rate_ci_table",
    "delivered_rate_table",
    "exact_mcnemar_p",
    "failure_attribution_table",
    "h2_invariance_table",
    "h4_parity_table",
    "holm",
    "load_results",
    "load_trials",
    "mcnemar_from_pairs",
    "mcnemar_table",
    "outcome_band",
    "risk_difference_table",
    "tost_equivalence",
    "verify_counterfactual",
    "wilson_ci",
]
