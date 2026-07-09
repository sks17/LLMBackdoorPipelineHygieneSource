"""Tests for the staged composition pipeline (Trial Three's core abstraction).

The load-bearing property: execution order is derived from each policy's ``stage``, not its
position in the declared list, so PRE_TEMPLATE always runs before POST_TEMPLATE and reversing the
declared policy list yields an identical result.
"""

from __future__ import annotations

import pytest

from trigger_audit.experiments.survivability_audit import trial_two_spec as t2
from trigger_audit.pipelines.composition import (
    ComposedPipeline,
    CompositionContext,
    HeadTruncationPolicy,
    KeepRecentMessagesPolicy,
    Stage,
)
from trigger_audit.pipelines.memory_policy import KeepLastNMessages
from trigger_audit.pipelines.truncation import HeadTruncation
from trigger_audit.prompts.chat_template import ChatTemplateRenderer
from trigger_audit.tokenization.tokenizer_adapter import SimpleWhitespaceTokenizerAdapter

_BUDGET = 6


def _renderer(adapter: SimpleWhitespaceTokenizerAdapter) -> ChatTemplateRenderer:
    return ChatTemplateRenderer(
        adapter, enable_thinking=t2.ENABLE_THINKING, add_generation_prompt=t2.ADD_GENERATION_PROMPT
    )


def _pipeline(adapter: SimpleWhitespaceTokenizerAdapter, policies):
    return ComposedPipeline(policies, renderer=_renderer(adapter), adapter=adapter)


def test_pre_runs_before_post_regardless_of_declared_order():
    adapter = SimpleWhitespaceTokenizerAdapter()
    messages = t2.base_messages()
    # Declared post-then-pre: the pipeline must still apply memory (pre) before templating.
    result = _pipeline(
        adapter,
        [
            HeadTruncationPolicy(context_length_target=_BUDGET),
            KeepRecentMessagesPolicy(keep_last_n=2),
        ],
    ).run(messages)

    # Memory (pre) reduced the six-message base to [system, assistant, user] = 3 messages...
    assert len(result.post_messages) == 3
    # ...and the token ids were derived from that reduced render, then head-truncated to the budget.
    reduced = (
        KeepLastNMessages(keep_last_n=2).apply(messages, budget=0, counter=lambda _m: 0).messages
    )
    reduced_ids = adapter.encode(_renderer(adapter).render(reduced), add_special_tokens=False)
    expected = HeadTruncation().apply(reduced_ids, _BUDGET).kept_ids
    assert result.final_token_ids == expected
    assert len(result.final_token_ids) == _BUDGET


def test_reversing_policy_list_yields_identical_result():
    adapter = SimpleWhitespaceTokenizerAdapter()
    messages = t2.base_messages()
    forward = [
        HeadTruncationPolicy(context_length_target=_BUDGET),
        KeepRecentMessagesPolicy(keep_last_n=2),
    ]
    reverse = list(reversed(forward))

    a = _pipeline(adapter, forward).run(messages)
    b = _pipeline(adapter, reverse).run(messages)

    assert a == b  # frozen dataclass value-equality across all four layers + metadata


def test_post_template_policy_without_token_ids_raises():
    ctx = CompositionContext(messages=t2.base_messages(), token_ids=None)
    with pytest.raises(ValueError, match="token_ids"):
        HeadTruncationPolicy(context_length_target=_BUDGET).apply(ctx)


def test_staged_policies_declare_their_stage():
    assert KeepRecentMessagesPolicy(keep_last_n=2).stage is Stage.PRE_TEMPLATE
    assert HeadTruncationPolicy(context_length_target=_BUDGET).stage is Stage.POST_TEMPLATE
