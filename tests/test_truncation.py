"""Tests for token-level truncation policies."""

from __future__ import annotations

from trigger_audit.pipelines.truncation import (
    TRUNCATION_REGISTRY,
    HeadTruncation,
    MiddleTruncation,
    NoTruncation,
    TailTruncation,
)

TOKENS = list(range(10))  # [0, 1, 2, ..., 9]


def test_no_truncation_passthrough():
    outcome = NoTruncation().apply(TOKENS, budget=3)
    assert outcome.kept_ids == TOKENS
    assert not outcome.truncated


def test_head_truncation_keeps_tail():
    outcome = HeadTruncation().apply(TOKENS, budget=4)
    assert outcome.kept_ids == [6, 7, 8, 9]
    assert outcome.dropped_head == 6
    assert outcome.dropped_tail == 0


def test_tail_truncation_keeps_head():
    outcome = TailTruncation().apply(TOKENS, budget=4)
    assert outcome.kept_ids == [0, 1, 2, 3]
    assert outcome.dropped_tail == 6
    assert outcome.dropped_head == 0


def test_middle_truncation_keeps_both_ends():
    outcome = MiddleTruncation().apply(TOKENS, budget=4)
    assert outcome.kept_ids == [0, 1, 8, 9]
    assert outcome.truncated


def test_under_budget_is_noop():
    for policy in (HeadTruncation(), TailTruncation(), MiddleTruncation()):
        outcome = policy.apply([1, 2], budget=10)
        assert outcome.kept_ids == [1, 2]
        assert not outcome.truncated


def test_registry_resolves_names():
    assert set(TRUNCATION_REGISTRY.names()) >= {
        "none",
        "truncate_head",
        "truncate_tail",
        "truncate_middle",
    }
    policy = TRUNCATION_REGISTRY.create("truncate_head")
    assert isinstance(policy, HeadTruncation)
