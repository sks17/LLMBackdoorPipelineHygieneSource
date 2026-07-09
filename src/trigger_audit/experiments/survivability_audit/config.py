"""Configuration schema for the survivability audit experiment grid."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from trigger_audit.schemas.triggers import TriggerPosition


def _default_positions() -> list[TriggerPosition]:
    return [
        TriggerPosition.PREFIX,
        TriggerPosition.MIDDLE,
        TriggerPosition.END,
        TriggerPosition.NEAR_BOUNDARY,
    ]


def _default_policies() -> list[str]:
    return ["none", "truncate_head", "keep_recent_messages", "summarize_old_messages"]


class SurvivabilityExperimentConfig(BaseModel):
    """Grid definition: bases x triggers x positions x context lengths x policies x models.

    One base conversation expands into many trials so conditions are comparable. Paths point at
    the inputs produced by earlier phases (base conversations, triggers, model/policy configs).
    """

    name: str = "survivability_audit"

    base_conversations_path: Path
    triggers_path: Path
    models_config_path: Path
    pipeline_policies_config_path: Path

    model_ids: list[str] = Field(default_factory=list)
    base_ids: list[str] | None = None
    trigger_ids: list[str] | None = None
    context_lengths: list[int] = Field(default_factory=lambda: [1024, 4096, 8192, 16384])
    trigger_positions: list[TriggerPosition] = Field(default_factory=_default_positions)
    pipeline_policies: list[str] = Field(default_factory=_default_policies)
    chat_templates: dict[str, str] = Field(default_factory=dict)
    # Emit a trigger-absent counterfactual twin for every grid point (for paired McNemar analysis);
    # each twin shares all coordinates but ``trigger_present`` and is a scoring sanity control.
    include_counterfactual: bool = False

    shard_size: int = 1000
    seed: int = 0
    run_generation: bool = False
    generation_fraction: float = 0.0
