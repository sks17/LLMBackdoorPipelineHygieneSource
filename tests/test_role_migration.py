"""E3 acceptance: a system-planted trigger that a template merges into a non-system turn.

The schema reserves ``SurvivalClass.ROLE_MIGRATION`` but nothing emitted it. The real case is
Gemma's chat template, which has no system role and merges the system message into the first user
turn: a trigger planted in the system message is *still delivered* to the final tokens, but it now
renders under the user role. The scorer determines the trigger's rendered role deterministically
from the templated text and emits ``ROLE_MIGRATION`` (instead of exact/token survival) when a
system-planted trigger renders inside a non-system turn while still delivered.

Every committed test here is fully offline: it uses the dependency-free
``SimpleWhitespaceTokenizerAdapter`` (plain, or a Gemma-like subclass that merges system into the
first user turn) and constructs the templated text directly. A live-Gemma check is included but
gated on ``transformers`` + the Gemma license, so it skips rather than fails offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from trigger_audit.experiments.survivability_audit.manifest_runner import run_trial
from trigger_audit.experiments.survivability_audit.scorer import (
    rendered_role_of_span,
    score_from_layers,
)
from trigger_audit.io.manifest import expand_manifest
from trigger_audit.io.stores import BaseConversationStore, TriggerStore
from trigger_audit.schemas.messages import BaseConversation, ChatMessage, Role
from trigger_audit.schemas.results import FailureStage, SurvivalClass
from trigger_audit.schemas.triggers import TriggerPosition, TriggerSpec, TriggerType
from trigger_audit.tokenization.tokenizer_adapter import (
    HFTokenizerAdapter,
    SimpleWhitespaceTokenizerAdapter,
)

TRIGGER_TEXT = "CANARY_TRIGGER_7F3XQ"
SYSTEM = TriggerPosition.SYSTEM
PREFIX = TriggerPosition.PREFIX

_REPO = Path(__file__).resolve().parent.parent
_BASE_PATH = _REPO / "data" / "base_conversations" / "base_conversations_000.jsonl"
_TRIGGERS_PATH = _REPO / "data" / "triggers" / "triggers.jsonl"
GEMMA = "google/gemma-3-1b-it"


@pytest.fixture
def trigger() -> TriggerSpec:
    """The single-token random canary used across the survival trials (id ``rand_001``)."""
    return TriggerSpec(
        trigger_id="rand_001",
        trigger_type=TriggerType.RANDOM_CANARY,
        text=TRIGGER_TEXT,
    )


class _MergingAdapter(SimpleWhitespaceTokenizerAdapter):
    """A Gemma-like reference tokenizer: no system role; system merges into the first user turn.

    Mirrors the structural fact that drives role migration -- the system message is not rendered as
    its own turn but concatenated into the first user turn -- without needing the real Gemma
    tokenizer (which is license-gated). Every other turn renders with its own ``<|role|>`` marker.
    """

    def render_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool,
        chat_template: str | None = None,
    ) -> str:
        _ = enable_thinking
        merged_system = "\n\n".join(m.content for m in messages if m.role == Role.SYSTEM)
        emitted_system = False
        parts: list[str] = []
        for message in messages:
            if message.role == Role.SYSTEM:
                continue  # Gemma has no system role.
            content = message.content
            if message.role == Role.USER and merged_system and not emitted_system:
                content = f"{merged_system}\n\n{content}"
                emitted_system = True
            parts.append(f"<|{message.role.value}|>\n{content}\n")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "".join(parts)


def _base() -> BaseConversation:
    """A minimal system+user base conversation for the offline role-migration checks."""
    return BaseConversation(
        base_id="conv_sys",
        conversation_type="multi_turn_chat",
        domain="software_debugging",
        target_token_length=64,
        messages=[
            ChatMessage(
                role=Role.SYSTEM, content="You are a helpful software debugging assistant."
            ),
            ChatMessage(role=Role.USER, content="How do I center a div in CSS?"),
        ],
        expected_user_task="answer the question",
    )


def _system_trial(position: TriggerPosition = SYSTEM):
    """A trigger-present trial planting the canary at ``position`` under the ``none`` policy."""
    return expand_manifest(["conv_sys"], ["rand_001"], [position], ["none"], ["gemma-like"])[0]


# --- rendered_role_of_span: the deterministic rendered-role detector ---------------------------


def test_rendered_role_chatml_system_is_system():
    # Qwen/ChatML keeps a distinct system turn -> a system trigger renders under "system".
    text = f"<|im_start|>system\n{TRIGGER_TEXT} You are helpful.<|im_end|>\n<|im_start|>user\nhi"
    assert rendered_role_of_span(text, TRIGGER_TEXT) == "system"


def test_rendered_role_gemma_merge_is_user():
    # Gemma has no system role: the nearest preceding marker is the user turn -> "user".
    text = f"<bos><start_of_turn>user\n{TRIGGER_TEXT} You are helpful.<end_of_turn>"
    assert rendered_role_of_span(text, TRIGGER_TEXT) == "user"


def test_rendered_role_gemma_model_turn_normalizes_to_assistant():
    # Gemma names the assistant turn "model"; it normalizes to the canonical "assistant".
    text = f"<start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n{TRIGGER_TEXT}"
    assert rendered_role_of_span(text, TRIGGER_TEXT) == "assistant"


def test_rendered_role_llama_style_marker():
    text = f"<|system|>\n{TRIGGER_TEXT} You are helpful.\n<|user|>\nhi\n"
    assert rendered_role_of_span(text, TRIGGER_TEXT) == "system"


def test_rendered_role_absent_or_unmarked_is_none():
    # Trigger absent -> None; and a base-completion prompt (no turn markers) -> None, so an
    # undetermined role can never be mistaken for a migration.
    assert rendered_role_of_span("<|system|>\nno canary here", TRIGGER_TEXT) is None
    assert rendered_role_of_span(f"system: {TRIGGER_TEXT} You are helpful.\n", TRIGGER_TEXT) is None


# --- score_from_layers: classification (constructed templated text, no pipeline) ---------------


def test_score_system_merge_emits_role_migration(simple_adapter, trigger):
    templated = (
        f"<|user|>\n{TRIGGER_TEXT} You are a helpful assistant.\n\n"
        "How do I center a div?\n<|assistant|>\n"
    )
    input_ids = simple_adapter.encode(templated)
    result = score_from_layers(
        _system_trial(),
        trigger,
        input_ids=input_ids,
        trigger_ids=simple_adapter.encode(TRIGGER_TEXT),
        post_template_text=templated,
        final_text=templated,
        raw_present=True,
        post_pipeline_present=True,
        trigger_token_span=simple_adapter.locate_token_span(templated, TRIGGER_TEXT),
    )
    assert result.survival_class is SurvivalClass.ROLE_MIGRATION
    assert result.failure_stage is FailureStage.NONE
    # The layer-attribution flags still record that the trigger truly survived -- only the headline
    # class changes to record that it arrived under a different role.
    assert result.final_token_trigger_present is True
    assert result.trigger_exact_survived is True
    assert result.trigger_token_survived is True


def test_score_system_no_merge_stays_exact_survival(simple_adapter, trigger):
    templated = (
        f"<|system|>\n{TRIGGER_TEXT} You are a helpful assistant.\n"
        "<|user|>\nHow do I center a div?\n<|assistant|>\n"
    )
    input_ids = simple_adapter.encode(templated)
    result = score_from_layers(
        _system_trial(),
        trigger,
        input_ids=input_ids,
        trigger_ids=simple_adapter.encode(TRIGGER_TEXT),
        post_template_text=templated,
        final_text=templated,
        raw_present=True,
        post_pipeline_present=True,
        trigger_token_span=simple_adapter.locate_token_span(templated, TRIGGER_TEXT),
    )
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL


def test_score_system_merge_but_not_delivered_is_not_migration(simple_adapter, trigger):
    # The system trigger migrated into the user turn but was then truncated out of the final tokens:
    # migration requires actual delivery, so this is a plain delivery failure, not ROLE_MIGRATION.
    templated = f"<|user|>\n{TRIGGER_TEXT} You are a helpful assistant.\n<|assistant|>\n"
    final_text = "<|assistant|>\n"
    result = score_from_layers(
        _system_trial(),
        trigger,
        input_ids=simple_adapter.encode(final_text),
        trigger_ids=simple_adapter.encode(TRIGGER_TEXT),
        post_template_text=templated,
        final_text=final_text,
        raw_present=True,
        post_pipeline_present=True,
        trigger_token_span=None,
    )
    assert result.survival_class is SurvivalClass.NO_SURVIVAL
    assert result.final_token_trigger_present is False


# --- run_trial: end-to-end through the verified spine ------------------------------------------


def test_run_trial_gemma_like_merge_emits_role_migration(trigger):
    result = run_trial(
        _system_trial(), base=_base(), trigger=trigger, tokenizer_adapter=_MergingAdapter()
    )
    assert result.survival_class is SurvivalClass.ROLE_MIGRATION
    assert result.final_token_trigger_present is True
    assert result.failure_stage is FailureStage.NONE


def test_run_trial_non_merging_template_stays_exact_survival(trigger):
    result = run_trial(
        _system_trial(),
        base=_base(),
        trigger=trigger,
        tokenizer_adapter=SimpleWhitespaceTokenizerAdapter(),
    )
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL


def test_run_trial_non_system_position_never_migrates(trigger):
    # A prefix-planted trigger renders in the user turn under the merging template too, but because
    # it was not planted in the system message it stays exact_survival: behavior for every
    # non-system position is unchanged.
    result = run_trial(
        _system_trial(PREFIX),
        base=_base(),
        trigger=trigger,
        tokenizer_adapter=_MergingAdapter(),
    )
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL


# --- SurvivalShardRunner: the persisted/cluster path (lead-wired) ------------------------------


def _shard_runner(tmp_path, adapter):
    """A SurvivalShardRunner over conv_sys + rand_001 whose tokenizer is the given adapter."""
    from trigger_audit.config.settings import ModelConfig, PipelinePolicyConfig
    from trigger_audit.experiments.survivability_audit import SurvivalShardRunner
    from trigger_audit.io.jsonl import write_jsonl

    base_path = tmp_path / "bases.jsonl"
    write_jsonl(base_path, [_base()])
    return SurvivalShardRunner(
        base_store=BaseConversationStore(base_path),
        trigger_store=TriggerStore(_TRIGGERS_PATH),
        model_configs={
            "gemma-like": ModelConfig(
                model_id="gemma-like",
                enable_thinking=False,
                max_context_window=4096,
                reserved_generation_tokens=0,
            )
        },
        pipeline_policies={"none": PipelinePolicyConfig(name="none")},
        tokenizer_factory=lambda mc: adapter,
    )


def test_shard_runner_gemma_like_merge_emits_role_migration(tmp_path, trigger):
    # The persisted shard-runner path (runner.py) must emit ROLE_MIGRATION too, not just the
    # manifest_runner path -- this is the wiring the lead added around the build() call.
    runner = _shard_runner(tmp_path, _MergingAdapter())
    result, _ = runner.run_trial(_system_trial())
    assert result.survival_class is SurvivalClass.ROLE_MIGRATION
    assert result.final_token_trigger_present is True
    _ = trigger


def test_shard_runner_non_system_position_never_migrates(tmp_path, trigger):
    runner = _shard_runner(tmp_path, _MergingAdapter())
    result, _ = runner.run_trial(_system_trial(PREFIX))
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL
    _ = trigger


# --- live Gemma (gated): the real system-merge, skipped offline --------------------------------


def test_run_trial_live_gemma_system_trigger_migrates(trigger):
    pytest.importorskip("transformers")
    try:
        adapter = HFTokenizerAdapter(GEMMA)
    except Exception as exc:  # offline / no-license environments should skip, not fail
        pytest.skip(f"{GEMMA} tokenizer unavailable: {exc}")

    base = BaseConversationStore(_BASE_PATH).get("conv_000001")
    canary = TriggerStore(_TRIGGERS_PATH).get("rand_001")
    gemma_trial = expand_manifest(["conv_000001"], ["rand_001"], [SYSTEM], ["none"], [GEMMA])[0]
    result = run_trial(gemma_trial, base=base, trigger=canary, tokenizer_adapter=adapter)
    assert result.survival_class is SurvivalClass.ROLE_MIGRATION
    assert result.final_token_trigger_present is True
    _ = trigger  # fixture parity with the offline tests
