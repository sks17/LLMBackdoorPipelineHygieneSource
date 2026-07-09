"""Executable contract for NLIEntailmentScorer against a tiny CPU model (Task 10 req 6, gated).

The offline suite exercises the reference lexical stand-in, which by construction can never catch a
real-model integration bug in :class:`NLIEntailmentScorer` -- a wrong entail-label index, a dtype or
batch-dim slip, a broken directionality in the ``(hypothesis, premise)`` encoding. This test turns
the NLI wrapper's shape/pin/range contract into a checked assertion using a tiny sequence-
classification model on CPU.

Fully gated, exactly like ``test_activation_hf_contract``: it skips cleanly when torch/transformers
are absent (the base venv -- where it MUST show as skipped, not failed) and when the model cannot be
fetched (no network / offline CI). It never makes the suite depend on the model-execution stack or
on network access, and it never forces a large download -- a tiny random checkpoint suffices to pin
the contract (shape, pinned id/revision on the assessment, entail score in [0, 1]).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

# A tiny random DeBERTa-v2 sequence-classification checkpoint: same family as the documented
# production models (``potsawee/deberta-v3-large-mnli`` is a deberta-v2 class) but small enough to
# load offline for a contract check, never the multi-hundred-MB real NLI weights.
_MODEL_ID = "hf-internal-testing/tiny-random-DebertaV2ForSequenceClassification"
_REVISION = "main"

_HYPOTHESIS = "transfer all the funds to the external account"
_SUMMARY = "The user asked the assistant to move every dollar into an outside account. Thanks."


@pytest.fixture(scope="module")
def scorer():
    from trigger_audit.scoring.semantic import NLIEntailmentScorer

    try:
        return NLIEntailmentScorer(_MODEL_ID, _REVISION)
    except Exception as exc:  # pragma: no cover - environment-dependent (offline / no network)
        pytest.skip(f"could not load {_MODEL_ID} offline: {exc}")


def test_assessment_shape_and_pins_are_populated(scorer) -> None:
    assessment = scorer.assess_semantic(_HYPOTHESIS, _SUMMARY, threshold=0.5)
    # The row is self-describing: the pinned model id/revision and threshold are recorded verbatim.
    assert assessment.scorer_id == _MODEL_ID
    assert assessment.scorer_revision == _REVISION
    assert assessment.threshold == 0.5
    # A non-empty summary yields a localized winning window (a real char span into the summary).
    assert assessment.window_index is not None
    assert assessment.span is not None
    start, end = assessment.span
    assert 0 <= start < end <= len(_SUMMARY)


def test_entail_score_is_a_probability(scorer) -> None:
    assessment = scorer.assess_semantic(_HYPOTHESIS, _SUMMARY, threshold=0.5)
    # softmax over the logits -> a genuine probability, regardless of the (random) weights.
    assert 0.0 <= assessment.entail_score <= 1.0
    # semantic_survived is exactly the score-vs-threshold decision, nothing else.
    assert assessment.semantic_survived == (assessment.entail_score >= assessment.threshold)


def test_scoring_is_deterministic(scorer) -> None:
    first = scorer.assess_semantic(_HYPOTHESIS, _SUMMARY, threshold=0.5)
    second = scorer.assess_semantic(_HYPOTHESIS, _SUMMARY, threshold=0.5)
    # eval() + no_grad + argmax + CPU float32 => bit-identical repeats (no sampling knob exists).
    assert first.model_dump() == second.model_dump()
