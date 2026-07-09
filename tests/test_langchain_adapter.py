"""Offline and real-tokenizer tests for the LangChain-backed trim policy (``LangChainTrimPolicy``).

These pin three things. First, message-level survival parity: driving the composition pipeline with
LangChain's ``trim_messages`` reproduces the Trial Two outcomes -- ``strategy="last"`` with
``include_system`` behaves like ``keep_last_n`` (old turn dropped, recent turn kept), while
``strategy="first"`` inverts it (old turn kept, recent turn dropped). Second, a characterization of
LangChain's mid-message overflow handling: a single message larger than the budget is dropped whole
by default, but with ``allow_partial=True`` and a ``text_splitter`` its content is split instead
(and, for ``strategy="last"``, the surviving fragment is the message's *suffix* -- the reachable
boundary-corruption shape). The plumbing/characterization tests are offline via
``SimpleWhitespaceTokenizerAdapter``; message-level *selection* (which messages survive) does not
depend on the tokenizer. Third -- the acceptance gate -- the lc_a/lc_b/lc_c/lc_d parity conditions
are re-run against the *real* Qwen3-0.6B tokenizer (skipped when transformers / the tokenizer are
unavailable, like the prior trials), so parity is asserted against verified ground truth and not
merely against the offline reference tokenizer.
"""

from __future__ import annotations

import pytest

from trigger_audit.experiments.survivability_audit import trial_two_spec
from trigger_audit.experiments.survivability_audit.scorer import score_from_layers
from trigger_audit.pipelines.composition import ComposedPipeline, CompositionContext
from trigger_audit.pipelines.langchain_adapter import (
    LangChainTrimPolicy,
    from_langchain,
    to_langchain,
)
from trigger_audit.prompts.chat_template import ChatTemplateRenderer
from trigger_audit.prompts.trigger_insertion import insert_trigger
from trigger_audit.schemas.messages import ChatMessage, Role
from trigger_audit.schemas.results import SurvivalClass
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.tokenization.tokenizer_adapter import (
    HFTokenizerAdapter,
    SimpleWhitespaceTokenizerAdapter,
)

QWEN = "Qwen/Qwen3-0.6B"


def _real_qwen_adapter() -> HFTokenizerAdapter:
    """Build the real Qwen3-0.6B tokenizer adapter, skipping when it is unavailable offline."""
    pytest.importorskip("transformers")
    try:
        return HFTokenizerAdapter(QWEN)
    except Exception as exc:  # offline / no-cache environments should skip, not fail
        pytest.skip(f"{QWEN} tokenizer unavailable: {exc}")


def _run(policy, position, adapter):
    """Run one condition end-to-end through the composition pipeline and score trigger survival."""
    raw = insert_trigger(trial_two_spec.base_messages(), trial_two_spec.TRIGGER.text, position)
    renderer = ChatTemplateRenderer(adapter, enable_thinking=False, add_generation_prompt=True)
    result = ComposedPipeline([policy], renderer=renderer, adapter=adapter).run(raw)
    final_text = adapter.decode(result.final_token_ids)
    return score_from_layers(
        trial_two_spec.trial_spec(
            TriggerPosition.OLD_TURN if position == "old_turn" else TriggerPosition.RECENT_TURN
        ),
        trial_two_spec.TRIGGER,
        input_ids=result.final_token_ids,
        trigger_ids=adapter.encode(trial_two_spec.TRIGGER.text, add_special_tokens=False),
        post_template_text=result.post_template_text,
        final_text=final_text,
        raw_present=any(trial_two_spec.TRIGGER.text in m.content for m in raw),
        post_pipeline_present=any(
            trial_two_spec.TRIGGER.text in m.content for m in result.post_messages
        ),
        trigger_token_span=adapter.locate_token_span(final_text, trial_two_spec.TRIGGER.text),
        pipeline_meta={"memory_policy": "langchain_trim"},
    )


def test_lc_a_last_include_system_old_turn_no_survival():
    """strategy=last keeps [system, recent-assistant, recent-user]; the old-turn trigger drops."""
    policy = LangChainTrimPolicy(
        max_tokens=3, strategy="last", token_counter=len, include_system=True
    )
    result = _run(policy, "old_turn", SimpleWhitespaceTokenizerAdapter())
    assert result.survival_class is SurvivalClass.NO_SURVIVAL


def test_lc_b_last_include_system_recent_turn_exact_survival():
    """strategy=last keeps the recent user turn, so a recent-turn trigger survives exactly."""
    policy = LangChainTrimPolicy(
        max_tokens=3, strategy="last", token_counter=len, include_system=True
    )
    result = _run(policy, "recent_turn", SimpleWhitespaceTokenizerAdapter())
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL


def test_lc_c_first_old_turn_exact_survival():
    """strategy=first keeps indices [0, 1, 2], so the old-turn trigger at index 1 survives."""
    policy = LangChainTrimPolicy(max_tokens=3, strategy="first", token_counter=len)
    result = _run(policy, "old_turn", SimpleWhitespaceTokenizerAdapter())
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL


def test_lc_d_first_recent_turn_no_survival():
    """strategy=first keeps only the opening turns, so the recent-turn trigger is dropped."""
    policy = LangChainTrimPolicy(max_tokens=3, strategy="first", token_counter=len)
    result = _run(policy, "recent_turn", SimpleWhitespaceTokenizerAdapter())
    assert result.survival_class is SurvivalClass.NO_SURVIVAL


def test_lc_a_parity_real_tokenizer_matches_trial_2a():
    """ACCEPTANCE: lc_a on the real Qwen3-0.6B tokenizer reproduces Trial 2A (no_survival)."""
    policy = LangChainTrimPolicy(
        max_tokens=3, strategy="last", token_counter=len, include_system=True
    )
    result = _run(policy, "old_turn", _real_qwen_adapter())
    assert result.post_pipeline_trigger_present is False  # old turn dropped before templating
    assert result.survival_class is SurvivalClass.NO_SURVIVAL  # == Trial 2A ground truth


def test_lc_b_parity_real_tokenizer_matches_trial_2b():
    """ACCEPTANCE: lc_b on the real Qwen3-0.6B tokenizer reproduces Trial 2B (exact_survival)."""
    policy = LangChainTrimPolicy(
        max_tokens=3, strategy="last", token_counter=len, include_system=True
    )
    result = _run(policy, "recent_turn", _real_qwen_adapter())
    assert result.post_pipeline_trigger_present is True  # recent turn kept through the policy
    assert result.final_token_trigger_present is True
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL  # == Trial 2B ground truth


def test_lc_c_first_old_turn_exact_survival_real_tokenizer():
    """lc_c on the real tokenizer: strategy=first keeps [0, 1, 2]; the old-turn trigger survives."""
    policy = LangChainTrimPolicy(max_tokens=3, strategy="first", token_counter=len)
    result = _run(policy, "old_turn", _real_qwen_adapter())
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL


def test_lc_d_first_recent_turn_no_survival_real_tokenizer():
    """lc_d on the real tokenizer: strategy=first drops the recent turn, so the trigger is gone."""
    policy = LangChainTrimPolicy(max_tokens=3, strategy="first", token_counter=len)
    result = _run(policy, "recent_turn", _real_qwen_adapter())
    assert result.survival_class is SurvivalClass.NO_SURVIVAL


def test_lc_e_overflow_message_dropped_whole_then_split_with_text_splitter():
    """Characterize mid-message overflow: dropped whole by default, split with a text_splitter."""
    long_content = "one two three four five six seven eight"

    def word_counter(messages):
        return sum(len(m.content.split()) for m in messages)

    # allow_partial=False (default): the over-budget user message is dropped whole, not truncated.
    drop_adapter = SimpleWhitespaceTokenizerAdapter()
    drop_policy = LangChainTrimPolicy(
        max_tokens=3,
        token_counter=word_counter,
        adapter=drop_adapter,
        strategy="last",
        include_system=True,
        allow_partial=False,
    )
    drop_ctx = CompositionContext(
        messages=[
            ChatMessage(role=Role.SYSTEM, content="sys"),
            ChatMessage(role=Role.USER, content=long_content),
        ]
    )
    drop_policy.apply(drop_ctx)
    drop_contents = [m.content for m in drop_ctx.messages]
    assert long_content not in drop_contents  # the whole message was dropped, never truncated
    assert "sys" in drop_contents  # the system message is preserved
    assert all(m.role is not Role.USER for m in drop_ctx.messages)  # no partial fragment survived

    # allow_partial=True with a text_splitter: the over-budget message's content is split instead.
    split_adapter = SimpleWhitespaceTokenizerAdapter()
    split_policy = LangChainTrimPolicy(
        max_tokens=3,
        token_counter=word_counter,
        adapter=split_adapter,
        strategy="last",
        include_system=True,
        allow_partial=True,
        text_splitter=lambda t: t.split(" "),
    )
    split_ctx = CompositionContext(
        messages=[
            ChatMessage(role=Role.SYSTEM, content="sys"),
            ChatMessage(role=Role.USER, content=long_content),
        ]
    )
    split_policy.apply(split_ctx)
    user_messages = [m for m in split_ctx.messages if m.role is Role.USER]
    assert user_messages  # a fragment of the long message survived
    assert user_messages[0].content != long_content  # content was split, i.e. modified


def test_lc_e_partial_split_keeps_suffix_for_last_prefix_for_first():
    """Pin the split direction: strategy=last keeps the message *suffix*, strategy=first the prefix.

    A suffix-only survivor of a single over-budget message is exactly the boundary-corruption shape
    (a trigger cut so only its trailing fragment reaches the model). This is the concrete evidence
    that boundary corruption is reachable through LangChain -- but only with ``allow_partial=True``
    plus a ``text_splitter`` (see the default-drops-whole assertion in ``test_lc_e`` above).
    """

    def char_counter(messages: object) -> int:
        return sum(len(m.content) for m in messages)  # type: ignore[attr-defined]

    def run_split(strategy: str) -> str:
        policy = LangChainTrimPolicy(
            max_tokens=4,
            token_counter=char_counter,
            strategy=strategy,
            allow_partial=True,
            text_splitter=list,  # split content into single characters
        )
        ctx = CompositionContext(messages=[ChatMessage(role=Role.USER, content="ABCDEFGHIJ")])
        policy.apply(ctx)
        return ctx.messages[0].content

    assert run_split("last") == "GHIJ"  # suffix survives -> boundary-corruption shape
    assert run_split("first") == "ABCD"  # prefix survives


def test_roundtrip_conversion_preserves_content():
    """from_langchain(to_langchain(msgs)) round-trips content and maps roles to their classes."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    msgs = [
        ChatMessage(role=Role.SYSTEM, content="you are a helpful assistant"),
        ChatMessage(role=Role.USER, content="hello there"),
        ChatMessage(role=Role.ASSISTANT, content="hi, how can I help?"),
    ]
    lc = to_langchain(msgs)
    assert isinstance(lc[0], SystemMessage)
    assert isinstance(lc[1], HumanMessage)
    assert isinstance(lc[2], AIMessage)
    assert from_langchain(lc) == msgs
