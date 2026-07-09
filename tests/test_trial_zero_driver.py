"""Offline tests for the Trial Zero driver and its base-conversation data row.

These run without ``transformers``: the driver is exercised through the dependency-free
``SimpleWhitespaceTokenizerAdapter``, which is enough to prove the positive control survives and
the negative control does not. The live-tokenizer tripwire lives in ``test_trial_zero.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from trigger_audit.experiments.survivability_audit import trial_zero_spec as spec
from trigger_audit.experiments.survivability_audit.trial_zero import run_trial_zero
from trigger_audit.schemas.messages import BaseConversation
from trigger_audit.schemas.results import SurvivalClass
from trigger_audit.tokenization.tokenizer_adapter import SimpleWhitespaceTokenizerAdapter

DATA_FILE = Path(__file__).parent.parent / "data" / "synthetic" / "trial_zero_base.jsonl"


def test_positive_control_exact_survival():
    result = run_trial_zero(tokenizer_adapter=SimpleWhitespaceTokenizerAdapter(), insert=True)
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL
    assert result.trigger_token_survived is True
    assert result.final_token_trigger_present is True


def test_negative_control_no_survival():
    result = run_trial_zero(tokenizer_adapter=SimpleWhitespaceTokenizerAdapter(), insert=False)
    assert result.survival_class is SurvivalClass.NO_SURVIVAL
    assert result.trigger_token_survived is False


def test_base_data_row_matches_spec():
    # The checked-in base row must stay a trigger-free copy of the spec's base conversation.
    rows = [json.loads(line) for line in DATA_FILE.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    conversation = BaseConversation.model_validate(rows[0])
    assert conversation.base_id == "trial_zero_base"
    assert conversation.messages == spec.base_messages()
    assert spec.TRIGGER.text not in json.dumps(rows[0])  # no trigger leaked into the base data
