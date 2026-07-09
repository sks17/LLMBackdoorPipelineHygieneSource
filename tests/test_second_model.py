"""Task 04b acceptance: second model (TinyLlama) — tokenizer/template diversity.

The 16-row grid is the Task-04a 8 rows for Qwen3 plus 8 for TinyLlama (its per-model tight budget).
The load-bearing claim: rows 3-8 (message-level outcomes) have the same ``survival_class`` for both
models, because the memory policy runs before any template and must not care which model is
downstream. Rows 1-2 are controls, and — thanks to char-offset localization —
``final_token_trigger_present`` is True for TinyLlama too, where the token-id subsequence method
wrongly returned False (its BPE re-tokenizes the trigger at the boundary).

The model-invariance and tripwire checks need the real tokenizers, so they skip when transformers /
the tokenizers are unavailable. The grid-shape and confound-documentation checks run offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trigger_audit.experiments.survivability_audit.manifest_runner import run_trial
from trigger_audit.io.manifest import expand_manifest
from trigger_audit.io.stores import BaseConversationStore, TriggerStore
from trigger_audit.pipelines.trigger_insertion import TriggerInserter
from trigger_audit.prompts.chat_template import ChatTemplateRenderer
from trigger_audit.schemas.results import SurvivalClass
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.tokenization.token_search import find_subsequence
from trigger_audit.tokenization.tokenizer_adapter import HFTokenizerAdapter

_REPO = Path(__file__).resolve().parent.parent
_BASE_PATH = _REPO / "data" / "base_conversations" / "base_conversations_000.jsonl"
_TRIGGERS_PATH = _REPO / "data" / "triggers" / "triggers.jsonl"
_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "tinyllama_rows"

QWEN = "Qwen/Qwen3-0.6B"
TINY = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TRIGGER_TEXT = "CANARY_TRIGGER_7F3XQ"
OLD, RECENT = TriggerPosition.OLD_TURN, TriggerPosition.RECENT_TURN
NONE, KRM = "none", "keep_recent_messages"
GEN = "keep_recent_messages+head_truncation_generous"
TIGHT = "keep_recent_messages+head_truncation_tight"
GEN_T, TIGHT_T = GEN + "_tinyllama", TIGHT + "_tinyllama"
QWEN_POLICIES = [NONE, KRM, GEN, TIGHT]
TINY_POLICIES = [NONE, KRM, GEN_T, TIGHT_T]


def _grid() -> list:
    # Two 8-row expansions concatenated (NOT one product): each model paired with its own tight id.
    return expand_manifest(["conv_000001"], ["rand_001"], [OLD, RECENT], QWEN_POLICIES, [QWEN]) + (
        expand_manifest(["conv_000001"], ["rand_001"], [OLD, RECENT], TINY_POLICIES, [TINY])
    )


def test_grid_is_16_rows_with_per_model_policy_ids():
    grid = _grid()
    assert len(grid) == 16
    assert len({t.trial_id for t in grid}) == 16  # unique per row
    tiny = [t for t in grid if t.model_id == TINY]
    qwen = [t for t in grid if t.model_id == QWEN]
    assert len(tiny) == 8 and len(qwen) == 8
    assert {t.pipeline_policy for t in tiny} == set(TINY_POLICIES)
    assert {t.pipeline_policy for t in qwen} == set(QWEN_POLICIES)


def test_tinyllama_fixture_documents_tokenization_confound():
    meta = json.loads((_FIXTURE_DIR / "meta.json").read_text(encoding="utf-8"))
    for name in ("old_none", "recent_none"):
        text = (_FIXTURE_DIR / name / "post_template_text.txt").read_text(encoding="utf-8")
        assert TRIGGER_TEXT in text  # the trigger string is plainly present
        row = meta["rows"][name]
        assert row["subsequence_span"] is None  # token-id subsequence MISSES it (the confound)
        assert row["offset_span"] is not None  # char-offset localization recovers it


def test_tinyllama_fixture_matches_live_tokenizer():
    pytest.importorskip("transformers")
    try:
        adapter = HFTokenizerAdapter(TINY)
    except Exception as exc:  # offline / no-cache environments should skip, not fail
        pytest.skip(f"{TINY} tokenizer unavailable: {exc}")

    base = BaseConversationStore(_BASE_PATH).get("conv_000001")
    trigger = TriggerStore(_TRIGGERS_PATH).get("rand_001")
    renderer = ChatTemplateRenderer(adapter, enable_thinking=False, add_generation_prompt=True)

    for name, position in (("old_none", OLD), ("recent_none", RECENT)):
        raw, _ = TriggerInserter().insert(base, trigger, position)
        text = renderer.render(raw)
        input_ids = adapter.encode(text, add_special_tokens=False)
        fixture_text = (_FIXTURE_DIR / name / "post_template_text.txt").read_text(encoding="utf-8")
        fixture_ids = json.loads(
            (_FIXTURE_DIR / name / "input_ids.json").read_text(encoding="utf-8")
        )
        assert text == fixture_text, f"{name}: TinyLlama template drifted from the golden fixture"
        assert input_ids == fixture_ids
        # Offset localization recovers the trigger; the token-id subsequence search misses it.
        assert adapter.locate_token_span(text, trigger.text) is not None
        trigger_ids = adapter.encode(trigger.text, add_special_tokens=False)
        assert find_subsequence(input_ids, trigger_ids) is None


def test_rows_3_8_survival_class_is_model_invariant():
    pytest.importorskip("transformers")
    try:
        adapters = {QWEN: HFTokenizerAdapter(QWEN), TINY: HFTokenizerAdapter(TINY)}
    except Exception as exc:  # offline / no-cache environments should skip, not fail
        pytest.skip(f"tokenizers unavailable: {exc}")

    base = BaseConversationStore(_BASE_PATH).get("conv_000001")
    trigger = TriggerStore(_TRIGGERS_PATH).get("rand_001")
    by_key = {(t.model_id, t.trigger_position, t.pipeline_policy): t for t in _grid()}

    def run(model: str, position: TriggerPosition, policy_id: str):
        return run_trial(
            by_key[(model, position, policy_id)],
            base=base,
            trigger=trigger,
            tokenizer_adapter=adapters[model],
        )

    # Rows 3-8 (Qwen policy, TinyLlama policy) — same condition, per-model tight budget.
    conditions = [
        (OLD, KRM, KRM),  # row 3
        (RECENT, KRM, KRM),  # row 4
        (OLD, GEN, GEN_T),  # row 5
        (RECENT, GEN, GEN_T),  # row 6
        (RECENT, TIGHT, TIGHT_T),  # row 7
        (OLD, TIGHT, TIGHT_T),  # row 8
    ]
    for position, qwen_policy, tiny_policy in conditions:
        qwen = run(QWEN, position, qwen_policy)
        tiny = run(TINY, position, tiny_policy)
        assert qwen.survival_class is tiny.survival_class, (position, qwen_policy)
        assert qwen.trigger_partial_survived is False
        assert tiny.trigger_partial_survived is False

    # Rows 1-2 (TinyLlama): exact_survival, and final_token_trigger_present True under offset
    # localization (it would be wrongly False under the token-id subsequence method).
    for position in (OLD, RECENT):
        tiny = run(TINY, position, NONE)
        assert tiny.survival_class is SurvivalClass.EXACT_SURVIVAL
        assert tiny.final_token_trigger_present is True
