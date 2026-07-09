"""Tests for survival scoring and the experiment result builder."""

from __future__ import annotations

from trigger_audit.experiments.survivability_audit.scorer import (
    SurvivalResultBuilder,
    aggregate_survival,
)
from trigger_audit.schemas.results import FailureStage, SurvivalClass
from trigger_audit.schemas.trials import TrialSpec
from trigger_audit.schemas.triggers import TriggerPosition, TriggerSpec, TriggerType
from trigger_audit.scoring.survival import SurvivalAssessment, TokenSurvivalScorer

TRIGGER = TriggerSpec(trigger_id="r1", trigger_type=TriggerType.RANDOM_CANARY, text="CANARY")


def _trial(policy: str, position: TriggerPosition) -> TrialSpec:
    return TrialSpec(
        trial_id="t_1",
        base_id="b1",
        trigger_id="r1",
        trigger_position=position,
        model_id="m",
        context_length=4096,
        pipeline_policy=policy,
    )


# --- scorer ---


def test_scorer_exact_and_token_survival():
    scorer = TokenSurvivalScorer()
    assessment = scorer.assess(
        [9, 1, 2, 3, 9], [1, 2, 3], final_text="x CANARY y", trigger_text="CANARY"
    )
    assert assessment.token_survived
    assert assessment.exact_text_survived
    assert assessment.match_start == 1
    assert assessment.match_end == 4


def test_scorer_token_only_when_text_absent():
    scorer = TokenSurvivalScorer()
    assessment = scorer.assess([1, 2, 3], [1, 2, 3])
    assert assessment.token_survived
    assert not assessment.exact_text_survived


def test_scorer_partial_survival():
    scorer = TokenSurvivalScorer()
    assessment = scorer.assess([9, 1, 2, 9], [1, 2, 3])
    assert not assessment.token_survived
    assert assessment.partial_survived
    assert assessment.matched_len == 2


def test_scorer_no_survival():
    scorer = TokenSurvivalScorer()
    assessment = scorer.assess([7, 8, 9], [1, 2, 3])
    assert not assessment.token_survived
    assert not assessment.partial_survived


# --- result builder ---


def test_builder_exact_survival_maps_to_no_failure():
    assessment = SurvivalAssessment(
        exact_text_survived=True, token_survived=True, match_start=3, match_end=4, trigger_len=1
    )
    result = SurvivalResultBuilder().build(
        _trial("none", TriggerPosition.PREFIX),
        TRIGGER,
        assessment,
        final_token_count=100,
        raw_present=True,
        post_pipeline_present=True,
        post_template_present=True,
    )
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL
    assert result.failure_stage is FailureStage.NONE


def test_builder_memory_drop_failure():
    assessment = SurvivalAssessment(token_survived=False, trigger_len=1)
    result = SurvivalResultBuilder().build(
        _trial("keep_recent_messages", TriggerPosition.OLD_TURN),
        TRIGGER,
        assessment,
        final_token_count=100,
        raw_present=True,
        post_pipeline_present=False,
        post_template_present=False,
        pipeline_meta={"memory_policy": "keep_recent_messages"},
    )
    assert result.survival_class is SurvivalClass.NO_SURVIVAL
    assert result.failure_stage is FailureStage.MEMORY_POLICY_DROPPED


def test_builder_boundary_corruption_on_head_truncation():
    assessment = SurvivalAssessment(
        token_survived=False, partial_survived=True, matched_len=1, trigger_len=2
    )
    result = SurvivalResultBuilder().build(
        _trial("truncate_head", TriggerPosition.PREFIX),
        TRIGGER,
        assessment,
        final_token_count=100,
        raw_present=True,
        post_pipeline_present=True,
        post_template_present=True,
        pipeline_meta={
            "truncation": {"policy": "truncate_head", "dropped_head": 5, "dropped_tail": 0}
        },
    )
    assert result.survival_class is SurvivalClass.BOUNDARY_CORRUPTION
    assert result.failure_stage is FailureStage.TRUNCATED_HEAD


def test_builder_partial_requires_upstream_delivery():
    # A token-overlap "partial" against a trigger that never reached the templated prompt
    # (post_template_present False -- a counterfactual twin, or one dropped whole by memory) is
    # coincidental benign overlap, not survival. It must classify no_survival, and the recorded
    # trigger_partial_survived flag must be False. This is the counterfactual-soundness guarantee.
    assessment = SurvivalAssessment(
        token_survived=False, partial_survived=True, matched_len=1, trigger_len=5
    )
    result = SurvivalResultBuilder().build(
        _trial("none", TriggerPosition.PREFIX),
        TRIGGER,
        assessment,
        final_token_count=100,
        raw_present=False,
        post_pipeline_present=False,
        post_template_present=False,
    )
    assert result.survival_class is SurvivalClass.NO_SURVIVAL
    assert result.trigger_partial_survived is False


def test_builder_summarize_failure_is_compression():
    assessment = SurvivalAssessment(token_survived=False, trigger_len=1)
    result = SurvivalResultBuilder().build(
        _trial("summarize_old_messages", TriggerPosition.OLD_TURN),
        TRIGGER,
        assessment,
        final_token_count=100,
        raw_present=True,
        post_pipeline_present=False,
        post_template_present=False,
        pipeline_meta={"memory_policy": "summarize_old_messages"},
    )
    assert result.failure_stage is FailureStage.COMPRESSED_EXACT_DELETED


# --- aggregation ---


def test_aggregate_survival_rates():
    builder = SurvivalResultBuilder()
    exact = SurvivalAssessment(exact_text_survived=True, token_survived=True, trigger_len=1)
    gone = SurvivalAssessment(token_survived=False, trigger_len=1)
    results = [
        builder.build(
            _trial("none", TriggerPosition.PREFIX),
            TRIGGER,
            exact,
            final_token_count=10,
            raw_present=True,
            post_pipeline_present=True,
            post_template_present=True,
        ),
        builder.build(
            _trial("none", TriggerPosition.PREFIX),
            TRIGGER,
            gone,
            final_token_count=10,
            raw_present=True,
            post_pipeline_present=True,
            post_template_present=True,
        ),
    ]
    rows = aggregate_survival(results)
    assert len(rows) == 1
    row = rows[0]
    assert row["n"] == 2
    assert row["delivered_rate"] == 0.5
    assert row["exact_rate"] == 0.5
