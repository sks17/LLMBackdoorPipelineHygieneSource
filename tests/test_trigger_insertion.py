"""Tests for deterministic trigger insertion."""

from __future__ import annotations

from trigger_audit.pipelines.trigger_insertion import TriggerInserter
from trigger_audit.schemas.messages import BaseConversation, ChatMessage, Role
from trigger_audit.schemas.triggers import TriggerPosition, TriggerSpec, TriggerType

TRIGGER = TriggerSpec(
    trigger_id="rand_001", trigger_type=TriggerType.RANDOM_CANARY, text="CANARY_TRIGGER_7F3XQ"
)


def _plain_conversation():
    return BaseConversation(
        base_id="c1",
        conversation_type="single_turn_long_document",
        messages=[
            ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=Role.USER, content="please summarize this document about latency"),
        ],
    )


def test_slot_insertion_replaces_and_strips_other_slots(slotted_conversation):
    messages, index = TriggerInserter().insert(
        slotted_conversation, TRIGGER, TriggerPosition.PREFIX
    )
    assert index == 1
    content = messages[1].content
    assert content.startswith("CANARY_TRIGGER_7F3XQ")
    assert "{{" not in content  # unused MIDDLE/END slots stripped


def test_prefix_positional_insertion():
    messages, index = TriggerInserter().insert(
        _plain_conversation(), TRIGGER, TriggerPosition.PREFIX
    )
    assert index == 1
    assert messages[1].content.startswith("CANARY_TRIGGER_7F3XQ")


def test_end_positional_insertion():
    messages, _ = TriggerInserter().insert(_plain_conversation(), TRIGGER, TriggerPosition.END)
    assert messages[1].content.rstrip().endswith("CANARY_TRIGGER_7F3XQ")


def test_middle_positional_insertion_is_interior():
    messages, _ = TriggerInserter().insert(_plain_conversation(), TRIGGER, TriggerPosition.MIDDLE)
    content = messages[1].content
    assert "CANARY_TRIGGER_7F3XQ" in content
    assert not content.startswith("CANARY_TRIGGER_7F3XQ")
    assert not content.rstrip().endswith("CANARY_TRIGGER_7F3XQ")


def test_system_position_targets_system_message():
    messages, index = TriggerInserter().insert(
        _plain_conversation(), TRIGGER, TriggerPosition.SYSTEM
    )
    assert index == 0
    assert "CANARY_TRIGGER_7F3XQ" in messages[0].content
