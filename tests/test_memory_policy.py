"""Tests for message-level memory policies."""

from __future__ import annotations

from tests.conftest import word_counter

from trigger_audit.experiments.survivability_audit import trial_two_spec as t2
from trigger_audit.pipelines.memory_policy import (
    MEMORY_REGISTRY,
    KeepLastNMessages,
    KeepRecentMessages,
    NoMemoryPolicy,
    SummarizeOldMessages,
)
from trigger_audit.schemas.messages import ChatMessage, Role


def _conversation():
    return [
        ChatMessage(role=Role.SYSTEM, content="you are helpful"),
        ChatMessage(role=Role.USER, content="old turn one two three four five"),
        ChatMessage(role=Role.ASSISTANT, content="ok"),
        ChatMessage(role=Role.USER, content="recent question here please"),
    ]


def test_no_memory_policy_passthrough():
    messages = _conversation()
    outcome = NoMemoryPolicy().apply(messages, budget=0, counter=word_counter)
    assert outcome.messages == messages
    assert outcome.dropped_indices == []


def test_keep_recent_drops_old_turn_keeps_system_and_recent():
    messages = _conversation()
    outcome = KeepRecentMessages().apply(messages, budget=8, counter=word_counter)
    assert outcome.dropped_indices == [1]
    contents = " ".join(m.content for m in outcome.messages)
    assert "old turn one" not in contents
    assert "recent question here please" in contents
    assert "you are helpful" in contents  # system always kept


def test_keep_recent_always_keeps_latest_even_if_over_budget():
    messages = _conversation()
    outcome = KeepRecentMessages().apply(messages, budget=0, counter=word_counter)
    assert "recent question here please" in outcome.messages[-1].content


def test_summarize_replaces_old_turns_with_placeholder():
    messages = _conversation()
    policy = SummarizeOldMessages(keep_recent_turns=1)
    outcome = policy.apply(messages, budget=0, counter=word_counter)
    assert outcome.dropped_indices == [1, 2]
    contents = [m.content for m in outcome.messages]
    assert any("Summary of earlier conversation" in c for c in contents)
    assert "old turn one two three" not in " ".join(contents)
    assert any("recent question here please" in c for c in contents)


def test_keep_last_n_drops_old_turns_as_whole_messages():
    messages = t2.base_messages()  # 6 messages: system + 5 non-system turns
    outcome = KeepLastNMessages(keep_last_n=2).apply(messages, budget=0, counter=lambda _m: 0)

    # System (index 0) always kept; last two non-system messages (indices 4, 5) kept.
    assert [messages.index(m) for m in outcome.messages] == [0, 4, 5]
    assert outcome.dropped_indices == [1, 2, 3]
    assert len(outcome.messages) == 3
    assert outcome.messages[0].role is Role.SYSTEM  # system always kept


def test_keep_last_n_does_not_mutate_input():
    messages = t2.base_messages()
    original_contents = [m.content for m in messages]

    KeepLastNMessages(keep_last_n=2).apply(messages, budget=0, counter=lambda _m: 0)

    assert [m.content for m in messages] == original_contents  # inputs unchanged


def test_registry_resolves_memory_policies():
    assert set(MEMORY_REGISTRY.names()) >= {
        "none",
        "keep_recent_messages",
        "keep_last_n_messages",
        "summarize_old_messages",
        "summary_plus_recent",
    }
    policy = MEMORY_REGISTRY.create("summarize_old_messages", keep_recent_turns=2)
    assert isinstance(policy, SummarizeOldMessages)
    last_n = MEMORY_REGISTRY.create("keep_last_n_messages", keep_last_n=2)
    assert isinstance(last_n, KeepLastNMessages)
