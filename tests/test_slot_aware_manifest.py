"""Slot-aware manifest expansion: tool_output/retrieved_doc expand only on bases carrying the slot.

A slot-strict position mis-plants (into the last user turn) when its slot is absent, so it must be
skipped for bases that lack it. Every other position is kept for every base -- the validated grid
is unchanged. Covers the ``plantable_positions`` filter and its wiring through ``expand_manifest``.
"""

from __future__ import annotations

from trigger_audit.io.manifest import expand_manifest
from trigger_audit.pipelines.trigger_insertion import plantable_positions
from trigger_audit.schemas.messages import BaseConversation, ChatMessage, Role
from trigger_audit.schemas.triggers import TriggerPosition

CORE = [TriggerPosition.PREFIX, TriggerPosition.END, TriggerPosition.SYSTEM]
ALL_POS = [*CORE, TriggerPosition.TOOL_OUTPUT]


def _chat_base() -> BaseConversation:
    return BaseConversation(
        base_id="chat_1",
        conversation_type="multi_turn_chat",
        domain="d",
        target_token_length=64,
        messages=[
            ChatMessage(role=Role.SYSTEM, content="{{PREFIX_SLOT}} sys"),
            ChatMessage(role=Role.USER, content="{{PREFIX_SLOT}} q {{END_SLOT}}"),
        ],
        expected_user_task="t",
    )


def _agent_tool_base() -> BaseConversation:
    return BaseConversation(
        base_id="agt_1",
        conversation_type="agent_tool",
        domain="d",
        target_token_length=64,
        messages=[
            ChatMessage(role=Role.SYSTEM, content="sys"),
            ChatMessage(role=Role.USER, content="{{PREFIX_SLOT}} look it up {{END_SLOT}}"),
            ChatMessage(role=Role.ASSISTANT, content="calling tool"),
            ChatMessage(role=Role.TOOL, content="{{TOOL_OUTPUT_SLOT}} result rows"),
            ChatMessage(role=Role.ASSISTANT, content="the answer"),
        ],
        expected_user_task="t",
    )


def test_tool_output_kept_only_when_slot_present() -> None:
    assert TriggerPosition.TOOL_OUTPUT in plantable_positions(_agent_tool_base(), ALL_POS)
    assert TriggerPosition.TOOL_OUTPUT not in plantable_positions(_chat_base(), ALL_POS)
    # core positions survive for both bases (never filtered)
    for base in (_chat_base(), _agent_tool_base()):
        kept = plantable_positions(base, ALL_POS)
        assert all(p in kept for p in CORE)


def test_expand_manifest_uses_per_base_positions() -> None:
    chat, agt = _chat_base(), _agent_tool_base()
    base_positions = {
        chat.base_id: plantable_positions(chat, ALL_POS),
        agt.base_id: plantable_positions(agt, ALL_POS),
    }
    trials = expand_manifest(
        [chat.base_id, agt.base_id],
        ["rand_001"],
        ALL_POS,
        ["none"],
        ["m"],
        context_lengths=[512],
        base_positions=base_positions,
    )
    tool_bases = {t.base_id for t in trials if t.trigger_position is TriggerPosition.TOOL_OUTPUT}
    assert tool_bases == {agt.base_id}  # only the agent-tool base expands tool_output
    # both bases still expand every core position
    for base_id in (chat.base_id, agt.base_id):
        got = {t.trigger_position for t in trials if t.base_id == base_id}
        assert set(CORE) <= got


def test_omitting_base_positions_reproduces_unfiltered_grid() -> None:
    # No base_positions -> every base expands every position (backward compatible).
    trials = expand_manifest(["b1"], ["rand_001"], ALL_POS, ["none"], ["m"], context_lengths=[512])
    assert {t.trigger_position for t in trials} == set(ALL_POS)
