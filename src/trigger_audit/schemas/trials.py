"""Trial schema: one atomic experiment row in a manifest."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from trigger_audit.schemas.triggers import TriggerPosition


class TrialSpec(BaseModel):
    """One atomic experiment.

    A trial is the tuple (base, trigger, position, model, context length, pipeline policy,
    chat template, seed). One trial maps to one manifest row; many trials make a shard; one
    worker processes a shard. ``context_length`` is the target token budget for the trial.
    """

    trial_id: str
    base_id: str
    trigger_id: str
    trigger_position: TriggerPosition
    model_id: str
    tokenizer_id: str | None = None
    context_length: int
    pipeline_policy: str
    chat_template: str | None = None
    # Counterfactual pairing: ``True`` is the trigger-present row; its ``False`` twin shares every
    # other coordinate and is the scoring sanity control (no trigger inserted -> no_survival).
    # Defaults ``True`` so existing manifests and single-row constructions are unaffected.
    trigger_present: bool = True
    run_generation: bool = False
    seed: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def resolved_tokenizer_id(self) -> str:
        """Return the tokenizer id, defaulting to the model id when unset."""
        return self.tokenizer_id or self.model_id
