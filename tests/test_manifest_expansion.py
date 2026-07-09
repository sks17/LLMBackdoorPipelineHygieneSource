"""Acceptance test for the manifest-driven grid (Task 04a).

The grid is one base x one trigger x two positions x four composite policies x one model = 8 rows.
Every row must reproduce an outcome already independently verified in Trials 2-3, so a mismatch
here localizes to the expansion/runner glue, not the primitives underneath.

The suite runs offline through ``SimpleWhitespaceTokenizerAdapter``. The tight truncation budget is
tokenizer-specific (the checked-in config's ``19`` is derived for Qwen3-0.6B), so for the offline
tokenizer the tight budget is re-derived per-adapter and supplied via a temp config; the supervisor
cross-checks the same table against the real Qwen3-0.6B tokenizer with the checked-in config.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from trigger_audit.experiments.survivability_audit import trial_three_spec as t3
from trigger_audit.experiments.survivability_audit.manifest_runner import run_trial
from trigger_audit.io.manifest import expand_manifest
from trigger_audit.io.stores import BaseConversationStore, TriggerStore
from trigger_audit.schemas.results import FailureStage, SurvivalClass
from trigger_audit.schemas.triggers import TriggerPosition

_REPO = Path(__file__).resolve().parent.parent
_BASE_PATH = _REPO / "data" / "base_conversations" / "base_conversations_000.jsonl"
_TRIGGERS_PATH = _REPO / "data" / "triggers" / "triggers.jsonl"

OLD, RECENT = TriggerPosition.OLD_TURN, TriggerPosition.RECENT_TURN
NONE = "none"
KRM = "keep_recent_messages"
GENEROUS = "keep_recent_messages+head_truncation_generous"
TIGHT = "keep_recent_messages+head_truncation_tight"
POLICY_IDS = [NONE, KRM, GENEROUS, TIGHT]
MODEL_IDS = ["Qwen/Qwen3-0.6B"]


def _grid() -> list:
    return expand_manifest(["conv_000001"], ["rand_001"], [OLD, RECENT], POLICY_IDS, MODEL_IDS)


def _write_offline_config(path: Path, tight_budget: int) -> None:
    # Mirror the checked-in composite config but with the offline-derived tight budget.
    config = {
        "policies": [
            {"id": NONE, "steps": []},
            {"id": KRM, "steps": [{"type": "keep_recent_messages", "keep_last_n": 2}]},
            {
                "id": GENEROUS,
                "steps": [
                    {"type": "keep_recent_messages", "keep_last_n": 2},
                    {"type": "head_truncation", "context_length_target": 40960},
                ],
            },
            {
                "id": TIGHT,
                "steps": [
                    {"type": "keep_recent_messages", "keep_last_n": 2},
                    {"type": "head_truncation", "context_length_target": tight_budget},
                ],
            },
        ]
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def test_grid_cardinality_and_stable_unique_ids():
    first = _grid()
    second = _grid()
    assert len(first) == 8  # 1 base x 1 trigger x 2 positions x 4 policies x 1 model
    # Ids are stable across re-expansion and unique per row.
    assert [t.trial_id for t in first] == [t.trial_id for t in second]
    assert len({t.trial_id for t in first}) == 8


def test_grid_reproduces_verified_trial_2_3_table(tmp_path):
    from trigger_audit.tokenization.tokenizer_adapter import SimpleWhitespaceTokenizerAdapter

    base = BaseConversationStore(_BASE_PATH).get("conv_000001")
    trigger = TriggerStore(_TRIGGERS_PATH).get("rand_001")
    adapter = SimpleWhitespaceTokenizerAdapter()
    trials = {(t.trigger_position, t.pipeline_policy): t for t in _grid()}

    # The offline tokenizer's prompt is far shorter than Qwen3's, so the checked-in tight budget is
    # non-binding here. Re-derive it (T - E) from the recent_turn generous run, as Trial 3C does.
    generous_recent = run_trial(
        trials[(RECENT, GENEROUS)], base=base, trigger=trigger, tokenizer_adapter=adapter
    )
    tight_budget = t3.derive_tight_budget(generous_recent)
    config_path = tmp_path / "policies_offline.yaml"
    _write_offline_config(config_path, tight_budget)

    def run(position: TriggerPosition, policy_id: str):
        return run_trial(
            trials[(position, policy_id)],
            base=base,
            trigger=trigger,
            tokenizer_adapter=adapter,
            policies_config_path=config_path,
        )

    # The 8-row acceptance table (position, policy_id) -> expected survival class.
    expected = {
        (OLD, NONE): SurvivalClass.EXACT_SURVIVAL,  # row 1: positive control
        (RECENT, NONE): SurvivalClass.EXACT_SURVIVAL,  # row 2: positive control
        (OLD, KRM): SurvivalClass.NO_SURVIVAL,  # row 3: Trial 2A
        (RECENT, KRM): SurvivalClass.EXACT_SURVIVAL,  # row 4: Trial 2B
        (OLD, GENEROUS): SurvivalClass.NO_SURVIVAL,  # row 5: Trial 3A
        (RECENT, GENEROUS): SurvivalClass.EXACT_SURVIVAL,  # row 6: Trial 3B
        (RECENT, TIGHT): SurvivalClass.NO_SURVIVAL,  # row 7: Trial 3C
        (OLD, TIGHT): SurvivalClass.NO_SURVIVAL,  # row 8: budget-independence
    }
    results = {key: run(*key) for key in expected}
    for key, survival_class in expected.items():
        assert results[key].survival_class is survival_class, key

    # Message/token granularity never yields partial survival on any row.
    for key, result in results.items():
        assert result.trigger_partial_survived is False, key

    # Row 8 (old_turn, tight) reproduces Row 5 (old_turn, generous)'s trigger outcome: the message
    # is gone at Layer 2, so the truncation budget cannot change the trigger's fate.
    row5, row8 = results[(OLD, GENEROUS)], results[(OLD, TIGHT)]
    assert (row8.survival_class, row8.failure_stage, row8.final_token_trigger_present) == (
        row5.survival_class,
        row5.failure_stage,
        row5.final_token_trigger_present,
    )
    assert row8.failure_stage is FailureStage.MEMORY_POLICY_DROPPED
    assert row8.final_token_trigger_present is False
