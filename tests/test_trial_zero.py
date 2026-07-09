"""Trial Zero acceptance test.

Two layers:
1. Offline scoring against the checked-in golden fixture (always runs; no transformers needed):
   exact string on Layer 3, token subsequence on Layer 4, EXACT_SURVIVAL for the positive trial,
   NO_SURVIVAL for the negative control.
2. A live-tokenizer tripwire (skipped when transformers/jinja2/the tokenizer are unavailable):
   re-renders and re-tokenizes Trial Zero and asserts it still matches the frozen fixture. If a
   future transformers version changes the Qwen3 template, this breaks loudly -- by design.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trigger_audit.experiments.survivability_audit import trial_zero_spec as spec
from trigger_audit.experiments.survivability_audit.scorer import score_from_layers
from trigger_audit.schemas.results import FailureStage, SurvivalClass
from trigger_audit.tokenization.token_search import find_subsequence

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_layers(name: str) -> tuple[str, list[int]]:
    directory = FIXTURE_DIR / name
    text = (directory / "post_template_text.txt").read_text(encoding="utf-8")
    input_ids = json.loads((directory / "input_ids.json").read_text(encoding="utf-8"))
    return text, input_ids


def _load_trigger_ids() -> list[int]:
    return json.loads((FIXTURE_DIR / "trial_zero" / "trigger_ids.json").read_text(encoding="utf-8"))


def test_trial_zero_positive_survives(  # exact string + token subsequence + EXACT_SURVIVAL
):
    text, input_ids = _load_layers("trial_zero")
    trigger_ids = _load_trigger_ids()

    assert spec.TRIGGER.text in text  # Layer 3: exact string match
    assert find_subsequence(input_ids, trigger_ids) is not None  # Layer 4: token subsequence

    result = score_from_layers(
        spec.trial_spec(),
        spec.TRIGGER,
        input_ids=input_ids,
        trigger_ids=trigger_ids,
        post_template_text=text,
        raw_present=True,
        post_pipeline_present=True,
    )
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL
    assert result.failure_stage is FailureStage.NONE
    assert result.trigger_token_survived is True
    assert result.final_token_trigger_present is True


def test_trial_zero_negative_control_no_survival():
    text, input_ids = _load_layers("trial_zero_negative")
    trigger_ids = _load_trigger_ids()

    assert spec.TRIGGER.text not in text
    assert find_subsequence(input_ids, trigger_ids) is None

    result = score_from_layers(
        spec.trial_spec(),
        spec.TRIGGER,
        input_ids=input_ids,
        trigger_ids=trigger_ids,
        post_template_text=text,
        raw_present=False,
        post_pipeline_present=False,
    )
    assert result.survival_class is SurvivalClass.NO_SURVIVAL


def test_trial_zero_fixture_matches_live_tokenizer():
    pytest.importorskip("transformers")
    pytest.importorskip("jinja2")
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(spec.TOKENIZER_ID)
    except Exception as exc:  # offline / no-cache environments should skip, not fail
        pytest.skip(f"{spec.TOKENIZER_ID} tokenizer unavailable: {exc}")

    text, input_ids = _load_layers("trial_zero")
    trigger_ids = _load_trigger_ids()

    live_text = tokenizer.apply_chat_template(
        spec.to_payload(spec.expected_positive_messages()),
        tokenize=False,
        add_generation_prompt=spec.ADD_GENERATION_PROMPT,
        enable_thinking=spec.ENABLE_THINKING,
    )
    assert live_text == text, "Qwen3 chat template output drifted from the golden fixture"
    assert tokenizer.encode(live_text, add_special_tokens=False) == input_ids
    assert tokenizer.encode(spec.TRIGGER.text, add_special_tokens=False) == trigger_ids
