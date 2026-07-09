"""End-to-end tests: run the survivability pipeline offline with the reference tokenizer,
and exercise manifest building. These validate that the shared infrastructure composes."""

from __future__ import annotations

from trigger_audit.config.settings import ModelConfig, PipelinePolicyConfig
from trigger_audit.experiments.survivability_audit import ManifestBuilder, SurvivalShardRunner
from trigger_audit.experiments.survivability_audit.config import SurvivabilityExperimentConfig
from trigger_audit.io.jsonl import read_jsonl, write_jsonl
from trigger_audit.io.paths import PathResolver
from trigger_audit.io.stores import BaseConversationStore, TriggerStore
from trigger_audit.schemas.results import FailureStage, SurvivalClass
from trigger_audit.schemas.trials import TrialSpec
from trigger_audit.schemas.triggers import TriggerPosition, TriggerSpec, TriggerType
from trigger_audit.tokenization.tokenizer_adapter import SimpleWhitespaceTokenizerAdapter

MODEL_ID = "simple-whitespace"


def _build_runner(tmp_path, slotted_conversation, scorer=None):
    base_path = tmp_path / "base.jsonl"
    trig_path = tmp_path / "triggers.jsonl"
    write_jsonl(base_path, [slotted_conversation])
    write_jsonl(
        trig_path,
        [
            TriggerSpec(
                trigger_id="rand_001",
                trigger_type=TriggerType.RANDOM_CANARY,
                text="CANARY_TRIGGER_7F3XQ",
            )
        ],
    )

    model_configs = {
        MODEL_ID: ModelConfig(
            model_id=MODEL_ID,
            enable_thinking=False,
            max_context_window=4096,
            reserved_generation_tokens=0,
        )
    }
    pipeline_policies = {
        "none": PipelinePolicyConfig(name="none"),
        "truncate_head": PipelinePolicyConfig(
            name="truncate_head", truncation_policy="truncate_head"
        ),
    }
    runner = SurvivalShardRunner(
        base_store=BaseConversationStore(base_path),
        trigger_store=TriggerStore(trig_path),
        model_configs=model_configs,
        pipeline_policies=pipeline_policies,
        tokenizer_factory=lambda mc: SimpleWhitespaceTokenizerAdapter(),
        scorer=scorer,
    )
    return runner


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


def test_prefix_trigger_survives_under_no_trimming(tmp_path, slotted_conversation):
    runner = _build_runner(tmp_path, slotted_conversation)
    result, generation = runner.run_trial(_trial("none", TriggerPosition.PREFIX, 4096))
    assert result.final_token_trigger_present is True
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL
    assert result.failure_stage is FailureStage.NONE
    assert generation is None


def test_prefix_trigger_dropped_by_head_truncation(tmp_path, slotted_conversation):
    runner = _build_runner(tmp_path, slotted_conversation)
    # A tiny budget forces head truncation to keep only the last few tokens.
    result, _ = runner.run_trial(_trial("truncate_head", TriggerPosition.PREFIX, 3))
    assert result.final_token_trigger_present is False
    assert result.post_template_trigger_present is True  # it reached the template, then was cut
    assert result.failure_stage is FailureStage.TRUNCATED_HEAD


def test_shard_runner_persists_cut_metadata_for_f6(tmp_path, slotted_conversation):
    # F6 ("anatomy of the cut") needs, on every persisted row, the head-drop count and the trigger's
    # pre-truncation token span. Assert a head-truncation trial carries them, and that a no-cut row
    # still reports dropped_head=0 with the trigger's span.
    runner = _build_runner(tmp_path, slotted_conversation)

    cut, _ = runner.run_trial(_trial("truncate_head", TriggerPosition.PREFIX, 3))
    md = cut.metadata
    assert md["truncation_policy"] == "truncate_head"
    assert md["dropped_head"] > 0  # the head was actually cut
    assert md["pretrunc_token_count"] > cut.final_prompt_token_count  # tokens were dropped
    assert md["pretrunc_trigger_span"] is not None and len(md["pretrunc_trigger_span"]) == 2

    nocut, _ = runner.run_trial(_trial("none", TriggerPosition.PREFIX, 4096))
    assert nocut.metadata["dropped_head"] == 0
    assert nocut.metadata["pretrunc_trigger_span"] is not None  # trigger present -> localizable


def test_shard_runner_localizes_trigger_by_offset_span(tmp_path, slotted_conversation):
    # Regression: the shard runner must localize the trigger by CHARACTER offsets, not a token-id
    # subsequence search. A BPE that re-tokenizes the trigger at the context boundary (TinyLlama in
    # the cluster pilot) breaks the subsequence search, so the token metrics would wrongly report a
    # delivered trigger as absent without this.
    from trigger_audit.scoring.survival import (
        USE_SUBSEQUENCE,
        SurvivalScorer,
        TokenSurvivalScorer,
    )

    seen: list[object] = []

    class _SpyScorer(SurvivalScorer):
        def assess(
            self,
            final_ids,
            trigger_ids,
            *,
            final_text=None,
            trigger_text=None,
            trigger_token_span=USE_SUBSEQUENCE,
            require_boundary_cut=None,
        ):
            seen.append(trigger_token_span)
            return TokenSurvivalScorer().assess(
                final_ids,
                trigger_ids,
                final_text=final_text,
                trigger_text=trigger_text,
                trigger_token_span=trigger_token_span,
                require_boundary_cut=require_boundary_cut,
            )

    runner = _build_runner(tmp_path, slotted_conversation, scorer=_SpyScorer())
    runner.run_trial(_trial("none", TriggerPosition.PREFIX, 4096))
    # The runner supplied an offset-localized span (a tuple/None), not the subsequence sentinel.
    assert seen and seen[0] is not USE_SUBSEQUENCE


def test_shard_runner_records_template_incompatible_instead_of_dropping(
    tmp_path, slotted_conversation
):
    # Regression: when a model's chat template rejects the produced sequence (Gemma's strict
    # alternation), the shard runner must record a first-class TEMPLATE_INCOMPATIBLE outcome, not
    # crash/drop the row (which would break counterfactual pairs). Parity with the trial-driver.
    from collections.abc import Sequence

    from trigger_audit.prompts.chat_template import TemplateRenderError
    from trigger_audit.schemas.messages import ChatMessage
    from trigger_audit.schemas.results import FailureStage, SurvivalClass

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
    result, generation = runner.run_trial(_trial("none", TriggerPosition.PREFIX, 4096))
    assert result.survival_class is SurvivalClass.NO_SURVIVAL
    assert result.failure_stage is FailureStage.TEMPLATE_INCOMPATIBLE
    assert result.final_prompt_token_count == 0
    assert generation is None


def test_run_writes_results_file(tmp_path, slotted_conversation):
    runner = _build_runner(tmp_path, slotted_conversation)
    shard = tmp_path / "shard.jsonl"
    write_jsonl(shard, [_trial("none", TriggerPosition.PREFIX, 4096)])
    out = tmp_path / "survival.jsonl"
    scored = runner.run(shard, out)
    assert scored == 1
    rows = read_jsonl(out)
    assert rows[0]["survival_class"] == "exact_survival"


# --- manifest building ---


def _experiment_config(tmp_path) -> SurvivabilityExperimentConfig:
    return SurvivabilityExperimentConfig(
        base_conversations_path=tmp_path / "base.jsonl",
        triggers_path=tmp_path / "triggers.jsonl",
        models_config_path=tmp_path / "models.yaml",
        pipeline_policies_config_path=tmp_path / "policies.yaml",
        context_lengths=[1024, 4096],
        trigger_positions=[TriggerPosition.PREFIX, TriggerPosition.END],
        pipeline_policies=["none", "truncate_head"],
    )


def test_manifest_cardinality_and_determinism(tmp_path):
    builder = ManifestBuilder(
        _experiment_config(tmp_path),
        base_ids=["conv_000001"],
        trigger_ids=["rand_001"],
        model_ids=[MODEL_ID],
    )
    trials = builder.build_list()
    # 1 base x 1 trigger x 2 positions x 2 lengths x 2 policies x 1 model = 8.
    assert len(trials) == 8
    assert len({t.trial_id for t in trials}) == 8  # ids are unique
    assert [t.trial_id for t in trials] == [
        t.trial_id for t in builder.build_list()
    ]  # deterministic


def test_manifest_sharding_groups_by_model(tmp_path):
    builder = ManifestBuilder(
        _experiment_config(tmp_path),
        base_ids=["conv_000001"],
        trigger_ids=["rand_001"],
        model_ids=[MODEL_ID],
    )
    trials = builder.build_list()
    paths = builder.shard(trials, PathResolver(root=tmp_path))
    assert len(paths) == 1
    assert len(read_jsonl(paths[0])) == 8
