"""Shared pytest fixtures and small builders for the trigger_audit test suite."""

from __future__ import annotations

import pytest

from trigger_audit.schemas.messages import BaseConversation, ChatMessage, Role, SlotLocation
from trigger_audit.schemas.triggers import TriggerSpec, TriggerType
from trigger_audit.tokenization.tokenizer_adapter import SimpleWhitespaceTokenizerAdapter


@pytest.fixture
def simple_adapter() -> SimpleWhitespaceTokenizerAdapter:
    """A fresh dependency-free reference tokenizer for each test."""
    return SimpleWhitespaceTokenizerAdapter()


@pytest.fixture
def canary_trigger() -> TriggerSpec:
    """A single-token random canary trigger."""
    return TriggerSpec(
        trigger_id="rand_001",
        trigger_type=TriggerType.RANDOM_CANARY,
        text="CANARY_TRIGGER_7F3XQ",
    )


@pytest.fixture
def slotted_conversation() -> BaseConversation:
    """A single-turn base conversation with prefix/middle/end slots in the user message."""
    user = (
        "{{PREFIX_SLOT}}\n\nHere is a long document about autosave latency and performance "
        "tuning in a note taking application.\n\n{{MIDDLE_SLOT}}\n\nMore details follow about "
        "the save loop and disk flushing behavior.\n\n{{END_SLOT}}\n\nPlease summarize the issues."
    )
    return BaseConversation(
        base_id="conv_000001",
        conversation_type="single_turn_long_document",
        domain="software_debugging",
        target_token_length=64,
        messages=[
            ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=Role.USER, content=user),
        ],
        expected_user_task="summarize the document",
        slot_locations=[
            SlotLocation(slot="{{PREFIX_SLOT}}", message_index=1),
            SlotLocation(slot="{{MIDDLE_SLOT}}", message_index=1),
            SlotLocation(slot="{{END_SLOT}}", message_index=1),
        ],
    )


def word_counter(message: ChatMessage) -> int:
    """Token counter used by memory-policy tests (whitespace word count)."""
    return len(message.content.split())
