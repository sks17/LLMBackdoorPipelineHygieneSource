"""Tests for the dependency-free reference tokenizer adapter."""

from __future__ import annotations

from trigger_audit.schemas.messages import ChatMessage, Role
from trigger_audit.tokenization.tokenizer_adapter import SimpleWhitespaceTokenizerAdapter


def test_token_ids_are_stable_across_instances():
    a = SimpleWhitespaceTokenizerAdapter()
    b = SimpleWhitespaceTokenizerAdapter()
    assert a.encode("CANARY_TRIGGER_7F3XQ") == b.encode("CANARY_TRIGGER_7F3XQ")


def test_encode_decode_roundtrip_for_seen_tokens():
    adapter = SimpleWhitespaceTokenizerAdapter()
    ids = adapter.encode("alpha lantern under blue bridge")
    assert adapter.decode(ids) == "alpha lantern under blue bridge"


def test_count_tokens_is_word_count():
    adapter = SimpleWhitespaceTokenizerAdapter()
    assert adapter.count_tokens("one two three") == 3


def test_render_chat_includes_roles_and_generation_prompt():
    adapter = SimpleWhitespaceTokenizerAdapter()
    rendered = adapter.render_chat(
        [
            ChatMessage(role=Role.SYSTEM, content="sys"),
            ChatMessage(role=Role.USER, content="hello"),
        ],
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert "<|system|>" in rendered
    assert "<|user|>" in rendered
    assert rendered.endswith("<|assistant|>\n")


def test_trigger_survives_as_subsequence_of_rendered_prompt():
    adapter = SimpleWhitespaceTokenizerAdapter()
    rendered = adapter.render_chat(
        [ChatMessage(role=Role.USER, content="CANARY_X please")], enable_thinking=False
    )
    final_ids = adapter.encode(rendered)
    trigger_ids = adapter.encode("CANARY_X")
    from trigger_audit.tokenization.token_search import contains_subsequence

    assert contains_subsequence(final_ids, trigger_ids)
