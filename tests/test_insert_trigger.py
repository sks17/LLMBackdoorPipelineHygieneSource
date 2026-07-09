"""Tests for the lightweight message-level :func:`insert_trigger` helper."""

from __future__ import annotations

import pytest

from trigger_audit.experiments.survivability_audit import trial_one_spec as t1
from trigger_audit.experiments.survivability_audit import trial_zero_spec as spec
from trigger_audit.prompts.trigger_insertion import insert_trigger
from trigger_audit.schemas.messages import ChatMessage, Role


def test_prefix_inserts_before_first_user_message():
    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=Role.USER, content="summarize this document"),
    ]
    result = insert_trigger(messages, "CANARY_TRIGGER_7F3XQ", "prefix")

    assert result[0].content == "You are a helpful assistant."  # system untouched
    assert result[1].content == "CANARY_TRIGGER_7F3XQ\n\nsummarize this document"


def test_input_list_and_messages_are_not_mutated():
    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=Role.USER, content="summarize this document"),
    ]
    original_contents = [m.content for m in messages]

    result = insert_trigger(messages, "CANARY_TRIGGER_7F3XQ", "prefix")

    assert [m.content for m in messages] == original_contents  # inputs unchanged
    assert result is not messages
    assert result[1] is not messages[1]  # a fresh message object, not the same instance


def test_hard_acceptance_matches_expected_positive_messages():
    # The frozen contract: a correct prefix insertion of the Trial Zero canary into the base
    # conversation must reproduce the spec's expected positive messages exactly.
    result = insert_trigger(spec.base_messages(), spec.TRIGGER.text, "prefix")
    assert result == spec.expected_positive_messages()


def test_end_appends_after_first_user_message():
    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=Role.USER, content="summarize this document"),
    ]
    result = insert_trigger(messages, "CANARY_TRIGGER_7F3XQ", "end")

    assert result[0].content == "You are a helpful assistant."  # system untouched
    assert result[1].content == "summarize this document\n\nCANARY_TRIGGER_7F3XQ"


def test_end_hard_acceptance_matches_expected_end_messages():
    # The frozen Trial One contract: an end insertion of the canary must reproduce the spec's
    # expected end messages exactly.
    result = insert_trigger(spec.base_messages(), spec.TRIGGER.text, "end")
    assert result == t1.expected_end_messages()


def test_end_does_not_mutate_input():
    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=Role.USER, content="summarize this document"),
    ]
    original_contents = [m.content for m in messages]

    insert_trigger(messages, "CANARY_TRIGGER_7F3XQ", "end")

    assert [m.content for m in messages] == original_contents  # inputs unchanged


def _multi_turn_messages() -> list[ChatMessage]:
    return [
        ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=Role.USER, content="first user turn"),
        ChatMessage(role=Role.ASSISTANT, content="ok"),
        ChatMessage(role=Role.USER, content="last user turn"),
    ]


def test_old_turn_targets_first_user_message():
    messages = _multi_turn_messages()
    result = insert_trigger(messages, "CANARY_TRIGGER_7F3XQ", "old_turn")

    assert "CANARY_TRIGGER_7F3XQ" in result[1].content  # first user message
    assert "CANARY_TRIGGER_7F3XQ" not in result[3].content  # last user message untouched


def test_recent_turn_targets_last_user_message():
    messages = _multi_turn_messages()
    result = insert_trigger(messages, "CANARY_TRIGGER_7F3XQ", "recent_turn")

    assert "CANARY_TRIGGER_7F3XQ" in result[3].content  # last user message
    assert "CANARY_TRIGGER_7F3XQ" not in result[1].content  # first user message untouched


def test_turn_positions_do_not_mutate_input():
    messages = _multi_turn_messages()
    original_contents = [m.content for m in messages]

    insert_trigger(messages, "CANARY_TRIGGER_7F3XQ", "old_turn")
    insert_trigger(messages, "CANARY_TRIGGER_7F3XQ", "recent_turn")

    assert [m.content for m in messages] == original_contents  # inputs unchanged


def test_unsupported_position_raises_not_implemented():
    # "middle" is a valid TriggerPosition but not yet wired into the lightweight inserter.
    with pytest.raises(NotImplementedError):
        insert_trigger(spec.base_messages(), spec.TRIGGER.text, "middle")
