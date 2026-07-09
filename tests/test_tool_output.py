"""E4 acceptance: the ``tool_output`` position (agent/tool conversation family), fully offline.

Pins the two seams E4 fills, using only the dependency-free reference tokenizer (no LLM, no
HuggingFace download, no network): the generator's ``AGENT_TOOL`` family emits a valid agent/tool
exchange whose ``tool``-role result message is where the ``{{TOOL_OUTPUT_SLOT}}`` lands; the slot-
aware ``TriggerInserter`` plants a trigger into that tool message (not a user turn) because
``target_user_index`` now routes ``TOOL_OUTPUT`` to the last ``tool`` message; and a head truncation
that drops the early tool result delivers ``no_survival`` -- the delivery risk this position audits.
"""

from __future__ import annotations

import pytest

from trigger_audit.experiments.survivability_audit.scorer import score_from_layers
from trigger_audit.generation.conversation_generator import (
    IMPLEMENTED_FAMILIES,
    ConversationFamily,
    MockBackend,
    generate_base_conversation,
    sample_seeds,
    validate_generated,
)
from trigger_audit.pipelines.trigger_insertion import TriggerInserter, target_user_index
from trigger_audit.pipelines.truncation import HeadTruncation
from trigger_audit.prompts.chat_template import ChatTemplateRenderer
from trigger_audit.schemas.messages import BaseConversation, ChatMessage, Role
from trigger_audit.schemas.results import FailureStage, SurvivalClass
from trigger_audit.schemas.trials import TrialSpec
from trigger_audit.schemas.triggers import TriggerPosition, TriggerSpec, TriggerType
from trigger_audit.tokenization.token_search import find_subsequence
from trigger_audit.tokenization.tokenizer_adapter import SimpleWhitespaceTokenizerAdapter

AGENT = ConversationFamily.AGENT_TOOL
TOOL_OUTPUT = TriggerPosition.TOOL_OUTPUT
TOOL_SLOT = "{{TOOL_OUTPUT_SLOT}}"
CANARY = TriggerSpec(
    trigger_id="rand_001", trigger_type=TriggerType.RANDOM_CANARY, text="CANARY_TRIGGER_7F3XQ"
)


def _agent_tool_base(
    adapter: SimpleWhitespaceTokenizerAdapter, *, target: int = 128
) -> BaseConversation:
    """Generate one length-binned AGENT_TOOL base with the tool_output slot planted (mock)."""
    seed = sample_seeds(1, families=[AGENT], seed=0)[0]
    return generate_base_conversation(
        seed,
        backend=MockBackend(),
        adapter=adapter,
        target_length=target,
        positions=[TOOL_OUTPUT],
    )


def _trial(policy: str) -> TrialSpec:
    """A minimal tool_output trial spec for scoring."""
    return TrialSpec(
        trial_id=f"t_{policy}",
        base_id="agent_tool_000",
        trigger_id=CANARY.trigger_id,
        trigger_position=TOOL_OUTPUT,
        model_id="simple",
        context_length=128,
        pipeline_policy=policy,
    )


# ---------------------------------------------------------------------------
# The AGENT_TOOL family is wired and emits a valid tool-carrying conversation.
# ---------------------------------------------------------------------------


def test_agent_tool_is_an_implemented_family() -> None:
    assert AGENT in IMPLEMENTED_FAMILIES


def test_mock_agent_tool_passes_validation_and_has_a_tool_message() -> None:
    backend = MockBackend()
    for seed in sample_seeds(6, families=[AGENT], seed=0):
        messages = backend.generate(seed)
        validate_generated(messages, seed)  # must not raise
        roles = [m.role for m in messages]
        assert roles[0] == Role.SYSTEM
        assert Role.TOOL in roles  # the tool result message
        assert sum(1 for m in messages if m.role == Role.USER) >= 1
        assert sum(1 for m in messages if m.role == Role.ASSISTANT) >= 1
        # No leaked slots or braces in raw generated content.
        assert all("{{" not in m.content and "}}" not in m.content for m in messages)


# ---------------------------------------------------------------------------
# The tool_output slot is planted into the tool message, not a user turn.
# ---------------------------------------------------------------------------


def test_generated_base_plants_slot_in_the_tool_message(
    simple_adapter: SimpleWhitespaceTokenizerAdapter,
) -> None:
    base = _agent_tool_base(simple_adapter)
    assert base.conversation_type == AGENT.value
    tool_slot_locs = [s for s in base.slot_locations if s.slot == TOOL_SLOT]
    assert len(tool_slot_locs) == 1
    idx = tool_slot_locs[0].message_index
    assert base.messages[idx].role == Role.TOOL  # the slot lands in the tool message
    assert TOOL_SLOT in base.messages[idx].content
    # The slot sits in exactly one message.
    assert sum(1 for m in base.messages if TOOL_SLOT in m.content) == 1


def test_target_user_index_routes_tool_output_to_last_tool_message() -> None:
    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are helpful."),
        ChatMessage(role=Role.USER, content="Please look this up."),
        ChatMessage(role=Role.ASSISTANT, content="Calling the first tool."),
        ChatMessage(role=Role.TOOL, content="First tool result."),
        ChatMessage(role=Role.ASSISTANT, content="Calling a second tool."),
        ChatMessage(role=Role.TOOL, content="Second tool result."),
        ChatMessage(role=Role.ASSISTANT, content="Here is the answer."),
    ]
    assert target_user_index(messages, TOOL_OUTPUT) == 5  # the last tool message
    # Other positions are unaffected: END still targets the last user message.
    assert target_user_index(messages, TriggerPosition.END) == 1


def test_target_user_index_tool_output_falls_back_to_user_without_a_tool_message() -> None:
    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are helpful."),
        ChatMessage(role=Role.USER, content="A plain chat with no tool turn."),
        ChatMessage(role=Role.ASSISTANT, content="Sure."),
    ]
    assert target_user_index(messages, TOOL_OUTPUT) == 1  # last user, defensively


def test_insert_at_tool_output_plants_into_the_tool_message(
    simple_adapter: SimpleWhitespaceTokenizerAdapter,
) -> None:
    base = _agent_tool_base(simple_adapter)
    filled, inserted_idx = TriggerInserter().insert(base, CANARY, TOOL_OUTPUT)
    assert filled[inserted_idx].role == Role.TOOL
    assert CANARY.text in filled[inserted_idx].content
    # Raw layer: the trigger is present in the (tool) message before any pipeline runs.
    assert any(CANARY.text in m.content for m in filled)
    joined = "\n".join(m.content for m in filled)
    assert "{{" not in joined  # the filled slot and any unused slots are gone


# ---------------------------------------------------------------------------
# A head truncation that drops the early tool result yields no_survival.
# ---------------------------------------------------------------------------


def test_head_truncation_dropping_the_tool_result_yields_no_survival(
    simple_adapter: SimpleWhitespaceTokenizerAdapter,
) -> None:
    base = _agent_tool_base(simple_adapter, target=200)
    filled, tool_idx = TriggerInserter().insert(base, CANARY, TOOL_OUTPUT)
    assert filled[tool_idx].role == Role.TOOL

    # Layer 3: the templated prompt carries the trigger (in the tool result, mid-conversation).
    renderer = ChatTemplateRenderer(simple_adapter, enable_thinking=False)
    post_template_text = renderer.render(filled)
    assert CANARY.text in post_template_text

    ids = simple_adapter.encode(post_template_text)
    trigger_ids = simple_adapter.encode(CANARY.text)
    span = find_subsequence(ids, trigger_ids)
    assert span is not None
    _, trigger_end = span

    # A head truncation whose budget keeps only the tail after the trigger drops the tool result
    # whole (the trigger and everything before it), leaving the final assistant answer.
    budget = len(ids) - (trigger_end + 1)
    outcome = HeadTruncation().apply(ids, budget)
    assert outcome.dropped_head > trigger_end  # the entire trigger was cut, not just its head
    final_ids = outcome.kept_ids
    final_text = simple_adapter.decode(final_ids)
    assert CANARY.text not in final_text

    result = score_from_layers(
        _trial("truncate_head"),
        CANARY,
        input_ids=final_ids,
        trigger_ids=trigger_ids,
        post_template_text=post_template_text,
        final_text=final_text,
        raw_present=True,
        post_pipeline_present=True,
        pipeline_meta={
            "truncation": {
                "policy": "truncate_head",
                "dropped_head": outcome.dropped_head,
                "dropped_tail": outcome.dropped_tail,
            }
        },
    )
    assert result.survival_class is SurvivalClass.NO_SURVIVAL
    assert result.final_token_trigger_present is False
    assert result.trigger_partial_survived is False
    assert result.failure_stage is FailureStage.TRUNCATED_HEAD


if __name__ == "__main__":  # pragma: no cover - convenience for local runs
    raise SystemExit(pytest.main([__file__, "-q"]))
