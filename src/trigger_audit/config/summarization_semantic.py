"""Typed config for the summarize-then-semantically-score example cell (Task 10).

The summarize policies (`summarize_old_messages` / `summary_plus_recent`) are the one delivery
cell with model dependence at BOTH the producer (the summarizer) and the scorer (the entailment
model), so a runnable example must name every pin: which summarize policy, which pinned
summarizer, which pinned semantic scorer, and the FP-rate budgets τ is calibrated against. This
model composes the already-typed :class:`SummarizerConfig` with a small semantic-scorer spec so
``config.loader.load_config`` parses ``configs/summarization_semantic.example.yaml`` into one typed
object, matching how the survivability example is modeled (a config schema, not a bespoke parser).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from trigger_audit.pipelines.summarizer import SummarizerConfig


class SemanticScorerConfig(BaseModel):
    """Selects and pins the semantic survival scorer (mirrors :func:`make_semantic_scorer`).

    ``backend='reference'`` is the dependency-free offline stand-in (no pin needed -- it is never a
    measurement scorer); ``backend='nli'`` names a real HF NLI checkpoint and MUST pin both
    ``model_id`` and ``revision`` (commit SHA), the same discipline the summarizer producer carries,
    so a semantic-survival row records exactly which scorer decided it.
    """

    backend: Literal["reference", "nli"] = "reference"
    model_id: str | None = None
    revision: str | None = None
    entail_label_index: int | None = None
    max_length: int = 512
    trust_remote_code: bool = False


class SummarizePolicyConfig(BaseModel):
    """Which compression policy carries the paraphrase, and how much recent context is kept."""

    policy: Literal["summarize_old_messages", "summary_plus_recent"] = "summarize_old_messages"
    keep_recent_turns: int = 1


class SummarizationSemanticExampleConfig(BaseModel):
    """A coherent, parseable example of the summarize + semantic-scoring cell.

    Names the summarize policy, the pinned producer (:class:`SummarizerConfig`), the pinned semantic
    scorer, the FP-rate budgets τ is calibrated to on the trigger-absent twins, the natural-phrase
    trigger whose meaning is being tracked, and the gold set used to report precision/recall. It is
    a documentation-grade example: fully typed and validated, not required to be driven by a CLI in
    this wave.
    """

    name: str = "summarization_semantic"

    summarize: SummarizePolicyConfig = Field(default_factory=SummarizePolicyConfig)
    summarizer: SummarizerConfig = Field(default_factory=SummarizerConfig)
    semantic_scorer: SemanticScorerConfig = Field(default_factory=SemanticScorerConfig)

    # FP-rate budgets on the absent-twin null; τ is the smallest threshold with achieved FPR <= one
    # of these (target 0.0 = "no affordable false positive", the strictest operating point).
    target_fprs: list[float] = Field(default_factory=lambda: [0.01, 0.0])
    # Only triggers with propositional content to paraphrase are meaningful here (never a random
    # canary); E2's ``natural_001`` is the instruction-style natural phrase this cell tracks.
    trigger_id: str = "natural_001"
    gold_path: Path = Path("data/gold/semantic_survival.jsonl")
