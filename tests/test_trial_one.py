"""Trial One acceptance test: naive head truncation, prefix vs end.

Everything is held constant except ``trigger_position``; the head-truncation budget is derived
from Trial Zero's measured trigger span (not hardcoded). Scored offline against the checked-in
golden fixtures. The key invariant: ``post_template_trigger_present`` is True for both variants
(the trigger is present pre-truncation in both), while ``final_token_trigger_present`` diverges --
so a False ``post_template_trigger_present`` would mean something upstream broke, not truncation.
"""

from __future__ import annotations

import json
from pathlib import Path

from trigger_audit.experiments.survivability_audit import trial_one_spec as t1
from trigger_audit.experiments.survivability_audit import trial_zero_spec as tz
from trigger_audit.experiments.survivability_audit.scorer import score_from_layers
from trigger_audit.pipelines.truncation import HeadTruncation, TruncationOutcome
from trigger_audit.schemas.results import FailureStage, SurvivalClass, SurvivalResult
from trigger_audit.schemas.triggers import TriggerPosition

FIXTURES = Path(__file__).parent / "fixtures"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_ids(path: Path) -> list[int]:
    return json.loads(path.read_text(encoding="utf-8"))


def _target() -> int:
    return json.loads((FIXTURES / "trial_one" / "meta.json").read_text())["context_length_target"]


def _trigger_ids() -> list[int]:
    return _read_ids(FIXTURES / "trial_zero" / "trigger_ids.json")


def _score(
    position: TriggerPosition, full_ids: list[int], full_text: str, final_text: str
) -> tuple[TruncationOutcome, SurvivalResult]:
    target = _target()
    outcome = HeadTruncation().apply(full_ids, target)
    result = score_from_layers(
        t1.trial_spec(position, target),
        tz.TRIGGER,
        input_ids=outcome.kept_ids,
        trigger_ids=_trigger_ids(),
        post_template_text=full_text,  # Layer 3: full, untruncated
        final_text=final_text,  # Layer 4: decoded truncated text
        raw_present=True,
        post_pipeline_present=True,
        pipeline_meta={
            "truncation": {
                "policy": "truncate_head",
                "dropped_head": outcome.dropped_head,
                "dropped_tail": 0,
            }
        },
    )
    return outcome, result


def _prefix_case() -> tuple[TruncationOutcome, SurvivalResult]:
    return _score(
        TriggerPosition.PREFIX,
        _read_ids(FIXTURES / "trial_zero" / "input_ids.json"),
        _read_text(FIXTURES / "trial_zero" / "post_template_text.txt"),
        _read_text(FIXTURES / "trial_one" / "prefix_final_text.txt"),
    )


def _end_case() -> tuple[TruncationOutcome, SurvivalResult]:
    return _score(
        TriggerPosition.END,
        _read_ids(FIXTURES / "trial_one" / "end_input_ids.json"),
        _read_text(FIXTURES / "trial_one" / "end_post_template_text.txt"),
        _read_text(FIXTURES / "trial_one" / "end_final_text.txt"),
    )


def test_trial_one_a_prefix_destroyed_by_head_truncation():
    outcome, result = _prefix_case()
    assert len(outcome.kept_ids) == _target()  # kept exactly the budget
    assert result.final_token_trigger_present is False
    assert result.survival_class is SurvivalClass.NO_SURVIVAL
    assert result.failure_stage is FailureStage.TRUNCATED_HEAD
    # Present before truncation -- localizes the failure to truncation, not upstream.
    assert result.post_template_trigger_present is True
    assert tz.TRIGGER.text not in _read_text(FIXTURES / "trial_one" / "prefix_final_text.txt")


def test_trial_one_b_end_survives_head_truncation():
    outcome, result = _end_case()
    assert len(outcome.kept_ids) == _target()
    assert result.final_token_trigger_present is True
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL
    assert result.post_template_trigger_present is True


def test_only_position_differs_between_variants():
    _, a = _prefix_case()
    _, b = _end_case()
    # Controlled: same target, base, model, policy; manipulated: trigger_position only.
    assert a.context_length == b.context_length
    assert a.pipeline_policy == b.pipeline_policy == t1.PIPELINE_POLICY
    assert a.model_id == b.model_id
    assert a.trigger_position != b.trigger_position
    assert (a.trigger_position, b.trigger_position) == (TriggerPosition.PREFIX, TriggerPosition.END)


def test_post_template_present_true_for_both():
    _, a = _prefix_case()
    _, b = _end_case()
    assert a.post_template_trigger_present is True
    assert b.post_template_trigger_present is True
    # The divergence lives entirely at Layer 4.
    assert a.final_token_trigger_present is False
    assert b.final_token_trigger_present is True
