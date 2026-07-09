"""The summarization semantic-survival cell: the delivery experiment blocked on a semantic scorer.

Compression memory policies (``summarize_old_messages`` / ``summary_plus_recent``) replace old turns
with an LLM-written summary. A trigger in an old turn can then survive as **meaning** (paraphrase)
even when its exact string / token ids do not -- a delivery mode exact/token matching is blind to.
This driver runs that cell end to end per the ``PRE_REGISTRATION.md`` 2026-07-04 amendment:

1. Compress each base's old turns with the pinned summarizer; take the summary region.
2. **Calibrate** the decision threshold ``tau`` on the trigger-ABSENT twins' entail scores (the
   null the audit's validity rests on) at a false-positive-rate budget.
3. Score each trigger-present trial through the verified :func:`score_from_layers` with the pinned
   semantic scorer at ``tau`` -> ``semantic_survival`` where the meaning carries.
4. Report the **required conditional quantities**: ``tau``, the achieved FPR on twins + its Wilson
   interval, and the gold-set precision/recall at ``tau`` -- never a clean 0/1, always "semantic
   delivery under scorer S at FP rate f". Unlike every other (model-agnostic) delivery cell, this
   one is producer x scorer conditional and is reported apart from the main grid.

Runnable fully offline with the reference summarizer + reference semantic scorer (no torch); a real
measurement swaps in the pinned ``HFSummarizer`` + ``NLIEntailmentScorer`` (config placeholders).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from trigger_audit.experiments.survivability_audit.scorer import score_from_layers
from trigger_audit.pipelines.memory_policy import SummarizeOldMessages, Summarizer
from trigger_audit.prompts.chat_template import ChatTemplateRenderer
from trigger_audit.prompts.trigger_insertion import insert_trigger
from trigger_audit.schemas.messages import ChatMessage
from trigger_audit.schemas.results import SurvivalResult
from trigger_audit.schemas.trials import TrialSpec
from trigger_audit.schemas.triggers import TriggerPosition, TriggerSpec
from trigger_audit.scoring.gold import SemanticGoldExample
from trigger_audit.scoring.semantic import (
    SemanticSurvivalScorer,
    calibrate_semantic_threshold,
    wilson_interval,
)
from trigger_audit.tokenization.tokenizer_adapter import TokenizerAdapter

_OLD_TURN = TriggerPosition.OLD_TURN


def summary_region(
    outcome_messages: Sequence[ChatMessage], raw_messages: Sequence[ChatMessage]
) -> str:
    """The summary the policy inserted: the one message content not present in the input messages.

    A memory policy that compresses old turns adds exactly one summary message; every other message
    is carried through unchanged, so the summary is the sole content absent from ``raw_messages``.
    Returns ``""`` when nothing was summarized (no old turns to compress).
    """
    original = {m.content for m in raw_messages}
    return next((m.content for m in outcome_messages if m.content not in original), "")


@dataclass
class SummarizationSemanticReport:
    """The producer x scorer-conditional result of one summarization semantic-survival cell.

    Every field the pre-registration requires reported: the twin-calibrated ``threshold`` with its
    achieved FPR + Wilson interval, the gold-set precision/recall at that threshold, the pinned
    scorer identity, and the per-outcome survival-class counts over the trigger-present trials.
    """

    policy: str
    summarizer_id: str
    scorer_id: str
    scorer_revision: str
    threshold: float
    target_fpr: float
    achieved_fpr: float
    fpr_wilson: tuple[float, float]
    n_absent: int
    gold_precision: float | None
    gold_recall: float | None
    gold_n: int
    n_present: int
    semantic_survival_rate: float
    class_counts: dict[str, int]
    results: list[SurvivalResult] = field(default_factory=list)


def gold_precision_recall(
    scorer: SemanticSurvivalScorer, gold: Sequence[SemanticGoldExample], threshold: float
) -> tuple[float, float, int]:
    """Precision and recall of the scorer at ``threshold`` on the hand-labeled gold set.

    A gold row is predicted survived when its entail score clears ``threshold``; precision/recall
    computed against the human ``survived`` label (the set deliberately includes hard negatives).
    Returns ``(precision, recall, n)``; precision/recall are ``1.0`` for an empty denominator.
    """
    tp = fp = fn = 0
    for row in gold:
        predicted = scorer.assess_semantic(
            row.trigger_text, row.summary_text, threshold=threshold
        ).semantic_survived
        if predicted and row.survived:
            tp += 1
        elif predicted and not row.survived:
            fp += 1
        elif not predicted and row.survived:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return precision, recall, len(gold)


def _summarize(
    messages: list[ChatMessage], *, summarizer: Summarizer, policy: str, keep_recent_turns: int
) -> tuple[list[ChatMessage], str]:
    """Apply the summarize policy with the pinned summarizer; return (messages, summary region)."""
    outcome = SummarizeOldMessages(
        keep_recent_turns=keep_recent_turns, summarizer=summarizer
    ).apply(messages, budget=0, counter=lambda _m: 1)
    return outcome.messages, summary_region(outcome.messages, messages)


def run_summarization_semantic(
    bases: Sequence[Sequence[ChatMessage]],
    trigger: TriggerSpec,
    *,
    adapter: TokenizerAdapter,
    summarizer: Summarizer,
    summarizer_id: str,
    semantic_scorer: SemanticSurvivalScorer,
    target_fpr: float = 0.0,
    keep_recent_turns: int = 1,
    policy: str = "summarize_old_messages",
    gold: Sequence[SemanticGoldExample] | None = None,
) -> SummarizationSemanticReport:
    """Run the summarization semantic cell over ``bases`` and return the conditional report.

    ``bases`` are raw (trigger-free) multi-turn message lists; the driver plants ``trigger`` at
    the old turn for the present arm and leaves the twin trigger-free. The threshold is calibrated
    on the twins, so the counterfactual control is the null that certifies it -- the dependency the
    pre-registration names.
    """
    renderer = ChatTemplateRenderer(adapter, enable_thinking=False)

    # (1) Calibrate tau on the ABSENT twins: the entail score of each twin's summary vs the trigger.
    absent_scores: list[float] = []
    for base in bases:
        _, region = _summarize(
            list(base), summarizer=summarizer, policy=policy, keep_recent_turns=keep_recent_turns
        )
        absent_scores.append(
            semantic_scorer.assess_semantic(trigger.text, region, threshold=1.0).entail_score
        )
    calib = calibrate_semantic_threshold(absent_scores, target_fpr)
    tau = calib.threshold
    n_fp = sum(1 for s in absent_scores if s >= tau)
    fpr_wilson = wilson_interval(n_fp, len(absent_scores)) if absent_scores else (0.0, 0.0)

    # (2) Score the trigger-present trials at tau through the verified layer scorer.
    results: list[SurvivalResult] = []
    for i, base in enumerate(bases):
        raw = insert_trigger(list(base), trigger.text, _OLD_TURN.value)
        messages, region = _summarize(
            raw, summarizer=summarizer, policy=policy, keep_recent_turns=keep_recent_turns
        )
        text = renderer.render(messages)
        trial = TrialSpec(
            trial_id=f"summ_{policy}_{i}",
            base_id=f"summ_base_{i}",
            trigger_id=trigger.trigger_id,
            trigger_position=_OLD_TURN,
            model_id=adapter.tokenizer_id if hasattr(adapter, "tokenizer_id") else "reference",
            context_length=0,
            pipeline_policy=policy,
        )
        result = score_from_layers(
            trial,
            trigger,
            input_ids=adapter.encode(text, add_special_tokens=False),
            trigger_ids=adapter.encode(trigger.text, add_special_tokens=False),
            post_template_text=text,
            raw_present=any(trigger.text in m.content for m in raw),
            post_pipeline_present=any(trigger.text in m.content for m in messages),
            pipeline_meta={"memory_policy": policy},
            semantic_scorer=semantic_scorer,
            summary_region=region,
            semantic_threshold=tau,
        )
        results.append(result)

    # (3) Gold-set precision/recall at tau; aggregate the outcome bands.
    gp, gr, gn = gold_precision_recall(semantic_scorer, gold, tau) if gold else (None, None, 0)
    counts = Counter(r.survival_class.value for r in results)
    n_present = len(results)
    return SummarizationSemanticReport(
        policy=policy,
        summarizer_id=summarizer_id,
        scorer_id=semantic_scorer.scorer_id,
        scorer_revision=semantic_scorer.scorer_revision,
        threshold=tau,
        target_fpr=target_fpr,
        achieved_fpr=calib.achieved_fpr,
        fpr_wilson=fpr_wilson,
        n_absent=len(absent_scores),
        gold_precision=gp,
        gold_recall=gr,
        gold_n=gn,
        n_present=n_present,
        semantic_survival_rate=counts.get("semantic_survival", 0) / n_present if n_present else 0.0,
        class_counts=dict(counts),
        results=results,
    )
