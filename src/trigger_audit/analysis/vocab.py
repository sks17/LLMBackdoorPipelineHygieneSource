"""Canonical vocabularies and label/mechanism resolution for the analysis layer.

Levels are DISCOVERED from the data: the ``pipeline_policy`` and ``trigger_position`` strings on a
result are policy-registry ids / enum values, which vary by run and are NOT the pre-registration
prose names (e.g. the data says ``truncate_head``, not ``head_truncation``). This module supplies
only a stable *ordering* for those discovered levels and, when a policies config is provided, the
human display name plus the memory/truncation mechanism behind each policy id.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from trigger_audit.config.settings import PipelinePolicyConfig

# Memory policies whose survival is not a valid delivery measurement: the summarizer is a
# deterministic placeholder stub (it always exact-deletes), and the pre-registration defers these
# pending a semantic-survival scorer. Tagged ``is_summarize`` and excluded from headline tables.
SUMMARIZE_MEMORY_POLICIES = frozenset({"summarize_old_messages", "summary_plus_recent"})

# The single collapse of (survival_class, failure_stage) used by every table and figure, in order.
OUTCOME_BAND_ORDER = [
    "exact",
    "token",
    "boundary",
    "partial",
    "template_incompatible",
    "role_migration",
    "none",
]

# Canonical display orders. Unknown levels are appended (sorted), never dropped -- a new position or
# policy id shows up at the end rather than silently vanishing from a table.
POSITION_ORDER = [
    "prefix",
    "early",
    "middle",
    "late",
    "end",
    "near_boundary",
    "old_turn",
    "recent_turn",
    "system",
    "tool_output",
    "retrieved_doc",
]

# Ordered by mechanism intensity; includes both the registry ids seen in data and the pre-reg names
# so either naming renders in a sensible order. Unknown ids are appended.
POLICY_ORDER = [
    "none",
    "keep_last_n_messages",
    "keep_recent_messages",
    "truncate_head",
    "head_truncation",
    "truncate_tail",
    "tail_truncation",
    "truncate_middle",
    "middle_truncation",
    "summarize_old_messages",
    "summary_plus_recent",
    "rag_baseline",
]

TRIGGER_TYPE_ORDER = [
    "random_canary",
    "natural_phrase",
    "multi_token_phrase",
    "split",
    "boundary",
    "unicode",
]

# survival_class -> outcome band, for the classes that map directly. ``no_survival`` is resolved by
# failure_stage (template_incompatible is split out); ``semantic_survival`` is reserved and unused.
_BAND_BY_CLASS = {
    "exact_survival": "exact",
    "token_survival": "token",
    "boundary_corruption": "boundary",
    "partial_survival": "partial",
    "role_migration": "role_migration",
}


def outcome_band(survival_class: str, failure_stage: str) -> str:
    """Collapse (survival_class, failure_stage) into one ordered band used everywhere.

    ``template_incompatible`` is kept distinct from generic ``none`` because it is a different
    delivery-failure mechanism (nothing was rendered at all), not a token-level drop.
    """
    band = _BAND_BY_CLASS.get(survival_class)
    if band is not None:
        return band
    if failure_stage == "template_incompatible":
        return "template_incompatible"
    return "none"


def order_levels(levels: Iterable[str], canonical: list[str]) -> list[str]:
    """Order the observed ``levels`` by ``canonical``, appending any unknowns in sorted order."""
    present = list(dict.fromkeys(levels))
    known = [c for c in canonical if c in present]
    unknown = sorted(x for x in present if x not in canonical)
    return known + unknown


def policy_mechanism(
    policies: Mapping[str, PipelinePolicyConfig],
) -> dict[str, dict[str, object]]:
    """Map each policy id to its memory/truncation mechanism, summarize flag, and display name.

    Driven by the run's policies config (authoritative) rather than parsing the id string.
    """
    out: dict[str, dict[str, object]] = {}
    for name, cfg in policies.items():
        out[name] = {
            "memory_policy": cfg.memory_policy,
            "truncation_policy": cfg.truncation_policy,
            "is_summarize": cfg.memory_policy in SUMMARIZE_MEMORY_POLICIES,
            "display": cfg.description or name,
        }
    return out
