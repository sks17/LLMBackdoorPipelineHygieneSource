"""Task 04c acceptance: Gemma — when the memory policy's output is unrenderable.

Gemma has no system role (system merges into the first user turn) and a strict-alternation template,
so the ``keep_last_n=2`` post-memory shape ``[system, assistant, user]`` is rejected outright. The
finding: a template-agnostic memory policy can produce a sequence a model cannot template, so
delivery fails at the template stage (``TEMPLATE_INCOMPATIBLE``) rather than crashing.

The graceful-capture path is unit-tested offline with a fake rejecting adapter (no Gemma needed);
the real-tokenizer acceptance and template tripwire skip when transformers / Gemma are unavailable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from trigger_audit.experiments.survivability_audit.manifest_runner import run_trial
from trigger_audit.io.manifest import expand_manifest
from trigger_audit.io.stores import BaseConversationStore, TriggerStore
from trigger_audit.pipelines.trigger_insertion import TriggerInserter
from trigger_audit.prompts.chat_template import ChatTemplateRenderer, TemplateRenderError
from trigger_audit.schemas.messages import ChatMessage, Role
from trigger_audit.schemas.results import FailureStage, SurvivalClass
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.tokenization.tokenizer_adapter import (
    HFTokenizerAdapter,
    SimpleWhitespaceTokenizerAdapter,
)

_REPO = Path(__file__).resolve().parent.parent
_BASE_PATH = _REPO / "data" / "base_conversations" / "base_conversations_000.jsonl"
_TRIGGERS_PATH = _REPO / "data" / "triggers" / "triggers.jsonl"
_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gemma_rows"

QWEN = "Qwen/Qwen3-0.6B"
TINY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GEMMA = "google/gemma-3-1b-it"
TRIGGER_TEXT = "CANARY_TRIGGER_7F3XQ"
OLD, RECENT = TriggerPosition.OLD_TURN, TriggerPosition.RECENT_TURN
NONE, KRM = "none", "keep_recent_messages"
GEN = "keep_recent_messages+head_truncation_generous"
TIGHT = "keep_recent_messages+head_truncation_tight"
GEN_T, TIGHT_T = GEN + "_tinyllama", TIGHT + "_tinyllama"
GEN_G, TIGHT_G = GEN + "_gemma", TIGHT + "_gemma"
GEMMA_POLICIES = [NONE, KRM, GEN_G, TIGHT_G]


class TemplateError(Exception):
    """Local stand-in for ``jinja2.exceptions.TemplateError`` (renderer matches by class name)."""


class _RejectingAdapter(SimpleWhitespaceTokenizerAdapter):
    """A tokenizer adapter whose chat template rejects every sequence (simulates Gemma offline)."""

    def render_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool,
        chat_template: str | None = None,
    ) -> str:
        raise TemplateError("Conversation roles must alternate user/assistant/user/assistant/...")


def _base_and_trigger():
    base = BaseConversationStore(_BASE_PATH).get("conv_000001")
    trigger = TriggerStore(_TRIGGERS_PATH).get("rand_001")
    return base, trigger


def _grid() -> list:
    # Three 8-row expansions concatenated (per-model policy ids) = 24 rows.
    qwen = expand_manifest(
        ["conv_000001"], ["rand_001"], [OLD, RECENT], [NONE, KRM, GEN, TIGHT], [QWEN]
    )
    tiny = expand_manifest(
        ["conv_000001"], ["rand_001"], [OLD, RECENT], [NONE, KRM, GEN_T, TIGHT_T], [TINY]
    )
    gemma = expand_manifest(["conv_000001"], ["rand_001"], [OLD, RECENT], GEMMA_POLICIES, [GEMMA])
    return qwen + tiny + gemma


def test_render_wraps_template_error_and_carries_messages():
    messages = [
        ChatMessage(role=Role.SYSTEM, content="s"),
        ChatMessage(role=Role.ASSISTANT, content="a"),
        ChatMessage(role=Role.USER, content="u"),
    ]
    renderer = ChatTemplateRenderer(_RejectingAdapter(), enable_thinking=False)
    with pytest.raises(TemplateRenderError) as excinfo:
        renderer.render(messages)
    assert "alternate" in str(excinfo.value)
    # The offending (post-memory) messages are carried so the runner can attribute presence.
    assert [m.content for m in excinfo.value.messages] == ["s", "a", "u"]


def test_render_does_not_wrap_unrelated_errors():
    class _Boom(SimpleWhitespaceTokenizerAdapter):
        def render_chat(
            self,
            messages: Sequence[ChatMessage],
            *,
            add_generation_prompt: bool = True,
            enable_thinking: bool,
            chat_template: str | None = None,
        ) -> str:
            raise ValueError("unrelated bug")

    renderer = ChatTemplateRenderer(_Boom(), enable_thinking=False)
    with pytest.raises(ValueError, match="unrelated bug"):
        renderer.render([ChatMessage(role=Role.USER, content="u")])


def test_run_trial_records_template_incompatible_offline():
    base, trigger = _base_and_trigger()
    adapter = _RejectingAdapter()
    # keep_recent_messages yields [system, assistant, user], which the rejecting template refuses.
    for position, expect_post_pipeline in ((OLD, False), (RECENT, True)):
        trial = expand_manifest(["conv_000001"], ["rand_001"], [position], [KRM], ["fake-model"])[0]
        result = run_trial(trial, base=base, trigger=trigger, tokenizer_adapter=adapter)
        assert result.survival_class is SurvivalClass.NO_SURVIVAL
        assert result.failure_stage is FailureStage.TEMPLATE_INCOMPATIBLE
        assert result.final_token_trigger_present is False
        assert result.final_prompt_token_count == 0
        assert result.raw_trigger_present is True
        assert result.post_pipeline_trigger_present is expect_post_pipeline
        assert "alternate" in result.metadata["template_error"]


def test_grid_includes_gemma_for_24_rows():
    grid = _grid()
    assert len(grid) == 24
    assert len({t.trial_id for t in grid}) == 24
    gemma_rows = [t for t in grid if t.model_id == GEMMA]
    assert len(gemma_rows) == 8
    assert {t.pipeline_policy for t in gemma_rows} == set(GEMMA_POLICIES)


def test_gemma_fixture_documents_system_merge():
    meta = json.loads((_FIXTURE_DIR / "meta.json").read_text(encoding="utf-8"))
    for name in ("old_none", "recent_none"):
        text = (_FIXTURE_DIR / name / "post_template_text.txt").read_text(encoding="utf-8")
        assert TRIGGER_TEXT in text
        assert "<start_of_turn>system" not in text  # Gemma has no system role
        assert "You are a helpful software debugging assistant." in text  # system merged into user
        assert meta["rows"][name]["system_merged_into_user"] is True


def test_gemma_fixture_matches_live_template():
    pytest.importorskip("transformers")
    try:
        adapter = HFTokenizerAdapter(GEMMA)
    except Exception as exc:  # offline / no-token environments should skip, not fail
        pytest.skip(f"{GEMMA} tokenizer unavailable: {exc}")

    base, trigger = _base_and_trigger()
    renderer = ChatTemplateRenderer(adapter, enable_thinking=False, add_generation_prompt=True)
    for name, position in (("old_none", OLD), ("recent_none", RECENT)):
        raw, _ = TriggerInserter().insert(base, trigger, position)
        text = renderer.render(raw)
        input_ids = adapter.encode(text, add_special_tokens=False)
        fixture_text = (_FIXTURE_DIR / name / "post_template_text.txt").read_text(encoding="utf-8")
        fixture_ids = json.loads(
            (_FIXTURE_DIR / name / "input_ids.json").read_text(encoding="utf-8")
        )
        assert text == fixture_text, f"{name}: Gemma template drifted from the golden fixture"
        assert input_ids == fixture_ids


def test_gemma_rows_acceptance_real_tokenizer():
    pytest.importorskip("transformers")
    try:
        adapters = {QWEN: HFTokenizerAdapter(QWEN), GEMMA: HFTokenizerAdapter(GEMMA)}
    except Exception as exc:  # offline / no-token environments should skip, not fail
        pytest.skip(f"tokenizers unavailable: {exc}")

    base, trigger = _base_and_trigger()
    by_key = {(t.model_id, t.trigger_position, t.pipeline_policy): t for t in _grid()}

    def run(model: str, position: TriggerPosition, policy_id: str):
        return run_trial(
            by_key[(model, position, policy_id)],
            base=base,
            trigger=trigger,
            tokenizer_adapter=adapters[model],
        )

    # Rows 1-2: exact_survival, model-invariant with Qwen3 despite Gemma's system-message merge.
    for position in (OLD, RECENT):
        gemma = run(GEMMA, position, NONE)
        qwen = run(QWEN, position, NONE)
        assert gemma.survival_class is SurvivalClass.EXACT_SURVIVAL
        assert gemma.survival_class is qwen.survival_class
        assert gemma.final_token_trigger_present is True

    # Rows 3-8: the post-memory sequence is unrenderable -> TEMPLATE_INCOMPATIBLE (no crash),
    # divergent from Qwen3/TinyLlama. That divergence is the trial's finding.
    for position, policy_id in (
        (OLD, KRM),
        (RECENT, KRM),
        (OLD, GEN_G),
        (RECENT, GEN_G),
        (RECENT, TIGHT_G),
        (OLD, TIGHT_G),
    ):
        gemma = run(GEMMA, position, policy_id)
        assert gemma.survival_class is SurvivalClass.NO_SURVIVAL
        assert gemma.failure_stage is FailureStage.TEMPLATE_INCOMPATIBLE
        assert gemma.final_prompt_token_count == 0
