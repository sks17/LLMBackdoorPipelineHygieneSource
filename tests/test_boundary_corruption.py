"""Task 05 acceptance: boundary corruption — a trigger cut through the middle.

The project's first ``trigger_partial_survived=True``: a long trigger whose front half head
truncation drops, leaving the back half as the literal prefix of the final input. Two negative
controls (generous window before the trigger; tight window after it) prove the partial predicate
does not false-positive on full survival or full loss.

Scored offline against the checked-in Qwen3-0.6B golden fixture (like Trial Zero); a live tripwire
re-runs the three conditions through the real tokenizer and skips when transformers is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trigger_audit.experiments.survivability_audit.manifest_runner import run_trial
from trigger_audit.experiments.survivability_audit.scorer import score_from_layers
from trigger_audit.io.manifest import expand_manifest
from trigger_audit.io.stores import BaseConversationStore, TriggerStore
from trigger_audit.schemas.results import FailureStage, SurvivalClass
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.tokenization.tokenizer_adapter import HFTokenizerAdapter

_REPO = Path(__file__).resolve().parent.parent
_BASE_PATH = _REPO / "data" / "base_conversations" / "base_conversations_001.jsonl"
_TRIGGERS_PATH = _REPO / "data" / "triggers" / "triggers.jsonl"
_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "boundary"

QWEN = "Qwen/Qwen3-0.6B"
E, N = SurvivalClass.EXACT_SURVIVAL, SurvivalClass.NO_SURVIVAL
B = SurvivalClass.BOUNDARY_CORRUPTION
NONE, HEAD = FailureStage.NONE, FailureStage.TRUNCATED_HEAD

# (condition, policy_id, exact, partial, survival_class, failure_stage)
CONDITIONS = [
    ("generous", "boundary_generous", True, False, E, NONE),  # control: window before trigger
    ("split", "boundary_split", False, True, B, HEAD),  # the test: cut through the middle
    ("tight", "boundary_tight", False, False, N, HEAD),  # control: window after trigger
]


def _meta() -> dict:
    return json.loads((_FIXTURE_DIR / "meta.json").read_text(encoding="utf-8"))


def _load(condition: str) -> tuple[list[int], str]:
    ids = json.loads(
        (_FIXTURE_DIR / condition / "final_input_ids.json").read_text(encoding="utf-8")
    )
    text = (_FIXTURE_DIR / condition / "final_text.txt").read_text(encoding="utf-8")
    return ids, text


def _trial(policy_id: str):
    return expand_manifest(
        ["conv_000002"], ["boundary_001"], [TriggerPosition.PREFIX], [policy_id], [QWEN]
    )[0]


def test_boundary_three_row_table_offline_from_fixture():
    meta = _meta()
    trigger_ids = meta["trigger_ids"]
    none_total = meta["none_token_count"]
    post_template_text = (_FIXTURE_DIR / "post_template_text.txt").read_text(encoding="utf-8")
    trigger = TriggerStore(_TRIGGERS_PATH).get("boundary_001")

    for condition, policy_id, exact, partial, survival_class, failure_stage in CONDITIONS:
        final_ids, final_text = _load(condition)
        dropped_head = none_total - len(final_ids)
        result = score_from_layers(
            _trial(policy_id),
            trigger,
            input_ids=final_ids,
            trigger_ids=trigger_ids,
            post_template_text=post_template_text,
            final_text=final_text,
            raw_present=True,
            post_pipeline_present=True,
            pipeline_meta={
                "truncation": {
                    "policy": "truncate_head",
                    "dropped_head": dropped_head,
                    "dropped_tail": 0,
                }
            },
        )
        assert result.trigger_exact_survived is exact, condition
        assert result.trigger_partial_survived is partial, condition
        assert result.survival_class is survival_class, (condition, result.survival_class)
        assert result.failure_stage is failure_stage, (condition, result.failure_stage)


def test_boundary_split_final_begins_with_a_trigger_suffix():
    _, split_text = _load("split")
    trigger_text = _meta()["trigger_text"]
    # Evidence the cut landed inside the trigger: the final input begins with its trailing fragment,
    # not its prefix, and the full trigger string is absent.
    assert split_text.startswith("_BLUE_BRIDGE_7F3XQ")
    assert not split_text.startswith("CANARY_TRIGGER_BOUNDARY")
    assert trigger_text not in split_text


def test_boundary_fixture_matches_live_tokenizer():
    pytest.importorskip("transformers")
    try:
        adapter = HFTokenizerAdapter(QWEN)
    except Exception as exc:  # offline / no-cache environments should skip, not fail
        pytest.skip(f"{QWEN} tokenizer unavailable: {exc}")

    base = BaseConversationStore(_BASE_PATH).get("conv_000002")
    trigger = TriggerStore(_TRIGGERS_PATH).get("boundary_001")
    for condition, policy_id, exact, partial, survival_class, failure_stage in CONDITIONS:
        result = run_trial(_trial(policy_id), base=base, trigger=trigger, tokenizer_adapter=adapter)
        assert result.trigger_exact_survived is exact, condition
        assert result.trigger_partial_survived is partial, condition
        assert result.survival_class is survival_class, (condition, result.survival_class)
        assert result.failure_stage is failure_stage, (condition, result.failure_stage)
