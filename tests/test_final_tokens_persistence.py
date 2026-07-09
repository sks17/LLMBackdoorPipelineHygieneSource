"""Tests for component A: persisting `final_token_ids` (the probe wave's join input).

Covers `io/final_tokens.py` (the sidecar producer/consumer), the optional inline field on
`SurvivalResult`, and the shard runner's opt-in sidecar/inline persistence -- all against the
offline reference tokenizer used elsewhere in the survival-audit test suite (see conftest.py and
test_pipeline_end_to_end.py).
"""

from __future__ import annotations

from collections.abc import Sequence

from trigger_audit.config.settings import ModelConfig, PipelinePolicyConfig
from trigger_audit.experiments.survivability_audit.runner import SurvivalShardRunner
from trigger_audit.experiments.survivability_audit.scorer import (
    SurvivalResultBuilder,
    score_from_layers,
    template_incompatible_result,
)
from trigger_audit.io.final_tokens import read_final_tokens, write_final_tokens
from trigger_audit.io.jsonl import read_jsonl, write_jsonl
from trigger_audit.io.stores import BaseConversationStore, TriggerStore
from trigger_audit.prompts.chat_template import TemplateRenderError
from trigger_audit.schemas.messages import ChatMessage
from trigger_audit.schemas.trials import TrialSpec
from trigger_audit.schemas.triggers import TriggerPosition, TriggerSpec, TriggerType
from trigger_audit.scoring.survival import SurvivalAssessment
from trigger_audit.tokenization.tokenizer_adapter import SimpleWhitespaceTokenizerAdapter

MODEL_ID = "simple-whitespace"

TRIGGER = TriggerSpec(
    trigger_id="rand_001", trigger_type=TriggerType.RANDOM_CANARY, text="CANARY_TRIGGER_7F3XQ"
)


def _trial(policy: str, position: TriggerPosition, context_length: int) -> TrialSpec:
    return TrialSpec(
        trial_id=f"t_{policy}",
        base_id="conv_000001",
        trigger_id="rand_001",
        trigger_position=position,
        model_id=MODEL_ID,
        context_length=context_length,
        pipeline_policy=policy,
    )


def _build_runner(tmp_path, slotted_conversation, **runner_kwargs):
    base_path = tmp_path / "base.jsonl"
    trig_path = tmp_path / "triggers.jsonl"
    write_jsonl(base_path, [slotted_conversation])
    write_jsonl(trig_path, [TRIGGER])

    model_configs = {
        MODEL_ID: ModelConfig(
            model_id=MODEL_ID,
            enable_thinking=False,
            max_context_window=4096,
            reserved_generation_tokens=0,
        )
    }
    pipeline_policies = {"none": PipelinePolicyConfig(name="none")}
    return SurvivalShardRunner(
        base_store=BaseConversationStore(base_path),
        trigger_store=TriggerStore(trig_path),
        model_configs=model_configs,
        pipeline_policies=pipeline_policies,
        tokenizer_factory=lambda mc: SimpleWhitespaceTokenizerAdapter(),
        **runner_kwargs,
    )


# --- io/final_tokens.py: the sidecar round trip ---


def test_write_read_round_trip(tmp_path):
    path = tmp_path / "final_tokens.jsonl"
    rows: list[tuple[str, Sequence[int] | None]] = [("t1", [1, 2, 3]), ("t2", [4, 5])]
    count = write_final_tokens(path, rows)
    assert count == 2
    assert read_final_tokens(path) == {"t1": [1, 2, 3], "t2": [4, 5]}


def test_write_final_tokens_skips_none_but_keeps_empty_list(tmp_path):
    path = tmp_path / "final_tokens.jsonl"
    count = write_final_tokens(path, [("t1", [1, 2]), ("t2", None), ("t3", [])])
    assert count == 2  # t2 (None) skipped; t3 (empty list) kept
    assert read_final_tokens(path) == {"t1": [1, 2], "t3": []}


# --- SurvivalResultBuilder / score_from_layers: opt-in inline attachment ---


def test_builder_attaches_final_token_ids_when_requested():
    assessment = SurvivalAssessment(
        exact_text_survived=True, token_survived=True, match_start=0, match_end=1, trigger_len=1
    )
    result = SurvivalResultBuilder().build(
        _trial("none", TriggerPosition.PREFIX, 4096),
        TRIGGER,
        assessment,
        final_token_count=3,
        raw_present=True,
        post_pipeline_present=True,
        post_template_present=True,
        final_token_ids=[10, 20, 30],
    )
    assert result.final_token_ids == [10, 20, 30]


def test_builder_default_leaves_final_token_ids_none():
    assessment = SurvivalAssessment(
        exact_text_survived=True, token_survived=True, match_start=0, match_end=1, trigger_len=1
    )
    result = SurvivalResultBuilder().build(
        _trial("none", TriggerPosition.PREFIX, 4096),
        TRIGGER,
        assessment,
        final_token_count=3,
        raw_present=True,
        post_pipeline_present=True,
        post_template_present=True,
    )
    assert result.final_token_ids is None


def test_score_from_layers_attaches_final_token_ids_when_requested():
    trial = _trial("none", TriggerPosition.PREFIX, 4096)
    result = score_from_layers(
        trial,
        TRIGGER,
        input_ids=[1, 2, 3, 4],
        trigger_ids=[2, 3],
        post_template_text="x CANARY_TRIGGER_7F3XQ y",
        raw_present=True,
        post_pipeline_present=True,
        final_token_ids=[1, 2, 3, 4],
    )
    assert result.final_token_ids == [1, 2, 3, 4]


def test_score_from_layers_default_leaves_final_token_ids_none():
    trial = _trial("none", TriggerPosition.PREFIX, 4096)
    result = score_from_layers(
        trial,
        TRIGGER,
        input_ids=[1, 2, 3, 4],
        trigger_ids=[2, 3],
        post_template_text="x CANARY_TRIGGER_7F3XQ y",
        raw_present=True,
        post_pipeline_present=True,
    )
    assert result.final_token_ids is None


def test_template_incompatible_result_has_no_final_token_ids():
    result = template_incompatible_result(
        _trial("none", TriggerPosition.PREFIX, 4096),
        TRIGGER,
        raw_present=False,
        post_pipeline_present=False,
        error="boom",
    )
    assert result.final_token_ids is None


# --- SurvivalShardRunner: sidecar + inline wiring ---


def test_shard_runner_writes_final_tokens_sidecar_containing_trigger_subsequence(
    tmp_path, slotted_conversation
):
    runner = _build_runner(tmp_path, slotted_conversation)
    shard = tmp_path / "shard.jsonl"
    write_jsonl(shard, [_trial("none", TriggerPosition.PREFIX, 4096)])
    survival_out = tmp_path / "survival.jsonl"
    final_tokens_out = tmp_path / "final_tokens.jsonl"

    scored = runner.run(shard, survival_out, final_tokens_out=final_tokens_out)
    assert scored == 1

    survival_rows = read_jsonl(survival_out)
    survival_ids = {row["trial_id"] for row in survival_rows}
    tokens = read_final_tokens(final_tokens_out)
    assert set(tokens) <= survival_ids  # sidecar trial_ids are a subset of survival trial_ids
    assert set(tokens) == survival_ids  # this trial delivered, so it is joinable

    adapter = SimpleWhitespaceTokenizerAdapter()
    trigger_ids = adapter.encode(TRIGGER.text, add_special_tokens=False)
    final_ids = tokens[survival_rows[0]["trial_id"]]
    # The join is usable: the trigger's token-id subsequence is contained in the persisted ids.
    assert any(
        final_ids[i : i + len(trigger_ids)] == trigger_ids
        for i in range(len(final_ids) - len(trigger_ids) + 1)
    )


def test_default_run_writes_no_sidecar_and_no_inline_field(tmp_path, slotted_conversation):
    runner = _build_runner(tmp_path, slotted_conversation)
    shard = tmp_path / "shard.jsonl"
    write_jsonl(shard, [_trial("none", TriggerPosition.PREFIX, 4096)])
    survival_out = tmp_path / "survival.jsonl"

    scored = runner.run(shard, survival_out)
    assert scored == 1
    assert not (tmp_path / "final_tokens.jsonl").exists()

    result, _ = runner.run_trial(_trial("none", TriggerPosition.PREFIX, 4096))
    assert result.final_token_ids is None


def test_persist_final_tokens_inline_populates_result(tmp_path, slotted_conversation):
    runner = _build_runner(tmp_path, slotted_conversation, persist_final_tokens_inline=True)
    result, _ = runner.run_trial(_trial("none", TriggerPosition.PREFIX, 4096))
    assert result.final_token_ids is not None
    assert len(result.final_token_ids) == result.final_prompt_token_count


def test_select_trial_ids_restricts_persistence_to_the_subset(tmp_path, slotted_conversation):
    trial = _trial("none", TriggerPosition.PREFIX, 4096)

    included = _build_runner(
        tmp_path / "a",
        slotted_conversation,
        persist_final_tokens_inline=True,
        select_trial_ids={trial.trial_id},
    )
    result, _ = included.run_trial(trial)
    assert result.final_token_ids is not None

    excluded = _build_runner(
        tmp_path / "b",
        slotted_conversation,
        persist_final_tokens_inline=True,
        select_trial_ids={"some_other_trial_id"},
    )
    result2, _ = excluded.run_trial(trial)
    assert result2.final_token_ids is None


def test_select_trial_ids_restricts_the_sidecar_too(tmp_path, slotted_conversation):
    kept = _trial("none", TriggerPosition.PREFIX, 4096)
    dropped = TrialSpec(**{**kept.model_dump(), "trial_id": "t_dropped"})
    runner = _build_runner(tmp_path, slotted_conversation, select_trial_ids={kept.trial_id})
    shard = tmp_path / "shard.jsonl"
    write_jsonl(shard, [kept, dropped])
    survival_out = tmp_path / "survival.jsonl"
    final_tokens_out = tmp_path / "final_tokens.jsonl"

    scored = runner.run(shard, survival_out, final_tokens_out=final_tokens_out)
    assert scored == 2  # both trials still scored and survive-result rows written
    tokens = read_final_tokens(final_tokens_out)
    assert set(tokens) == {kept.trial_id}  # only the selected trial reaches the sidecar


def test_template_incompatible_trial_skipped_from_sidecar_without_crashing(
    tmp_path, slotted_conversation
):
    # Regression: a trial whose template rejects the produced sequence has no final tokens at
    # all, so it must not appear in the sidecar (empty ids are not joinable) and must not crash
    # the shard run -- parity with the first-class TEMPLATE_INCOMPATIBLE outcome.
    class _RejectingAdapter(SimpleWhitespaceTokenizerAdapter):
        def render_chat(
            self,
            messages: Sequence[ChatMessage],
            *,
            add_generation_prompt: bool = True,
            enable_thinking: bool = False,
            chat_template: str | None = None,
        ) -> str:
            raise TemplateRenderError("roles must alternate", messages=list(messages))

    runner = _build_runner(tmp_path, slotted_conversation)
    runner._tokenizer_factory = lambda mc: _RejectingAdapter()
    shard = tmp_path / "shard.jsonl"
    write_jsonl(shard, [_trial("none", TriggerPosition.PREFIX, 4096)])
    survival_out = tmp_path / "survival.jsonl"
    final_tokens_out = tmp_path / "final_tokens.jsonl"

    scored = runner.run(shard, survival_out, final_tokens_out=final_tokens_out)
    assert scored == 1
    assert read_final_tokens(final_tokens_out) == {}
