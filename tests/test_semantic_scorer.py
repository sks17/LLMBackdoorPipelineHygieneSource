"""Tests for the reference semantic survival scorer, its segmentation, and the factory."""

from __future__ import annotations

import pytest

from trigger_audit.scoring.semantic import (
    NLIEntailmentScorer,
    ReferenceSemanticScorer,
    SemanticAssessment,
    make_semantic_scorer,
    segment_summary_windows,
)

# A natural-phrase-style trigger with genuine propositional content to paraphrase.
TRIGGER = "transfer all the funds to the external account"
# A window that rewrites the trigger's meaning without sharing verbatim/subsequence tokens.
PARAPHRASE = "Please move every dollar into an outside account."
# A topically-unrelated window sharing no content lemmas with the trigger.
UNRELATED = "The weather today is sunny with a gentle breeze."

THRESHOLD = 0.3


def test_reference_scorer_is_deterministic():
    scorer = ReferenceSemanticScorer()
    first = scorer.assess_semantic(TRIGGER, PARAPHRASE, threshold=THRESHOLD)
    second = scorer.assess_semantic(TRIGGER, PARAPHRASE, threshold=THRESHOLD)
    assert first == second
    assert first.model_dump() == second.model_dump()


def test_reference_scorer_ranks_paraphrase_above_unrelated():
    scorer = ReferenceSemanticScorer()
    survived = scorer.assess_semantic(TRIGGER, PARAPHRASE, threshold=THRESHOLD)
    absent = scorer.assess_semantic(TRIGGER, UNRELATED, threshold=THRESHOLD)
    assert survived.semantic_survived
    assert not absent.semantic_survived
    assert survived.entail_score > absent.entail_score


def test_reference_scorer_records_pin_on_every_assessment():
    scorer = ReferenceSemanticScorer()
    assessment = scorer.assess_semantic(TRIGGER, PARAPHRASE, threshold=THRESHOLD)
    assert assessment.scorer_id == "reference"
    assert assessment.scorer_revision == "reference"
    assert assessment.threshold == THRESHOLD


def test_reference_scorer_localizes_winning_window():
    # A three-sentence summary whose middle sentence carries the paraphrase; the scorer must
    # return the middle window's index and a char span that indexes exactly that substring.
    summary = f"The weather today is sunny. {PARAPHRASE} I had lunch earlier."
    scorer = ReferenceSemanticScorer()
    assessment = scorer.assess_semantic(TRIGGER, summary, threshold=THRESHOLD)
    assert assessment.semantic_survived
    assert assessment.window_index == 1
    assert assessment.span is not None
    start, end = assessment.span
    assert summary[start:end] == PARAPHRASE
    assert "move every dollar" in summary[start:end]


def test_segmentation_spans_index_their_substrings():
    text = "First point. Second point! Third point?"
    spans = segment_summary_windows(text)
    assert [text[s:e] for s, e in spans] == [
        "First point.",
        "Second point!",
        "Third point?",
    ]


def test_empty_and_whitespace_summary_do_not_crash():
    scorer = ReferenceSemanticScorer()
    for summary in ("", "   \n\t  "):
        assessment = scorer.assess_semantic(TRIGGER, summary, threshold=THRESHOLD)
        assert not assessment.semantic_survived
        assert assessment.span is None
        assert assessment.window_index is None
        assert assessment.entail_score == 0.0


def test_factory_returns_reference_backend():
    scorer = make_semantic_scorer("reference")
    assert isinstance(scorer, ReferenceSemanticScorer)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown semantic scorer backend"):
        make_semantic_scorer("bogus")


def test_nli_backend_missing_deps_raises_clear_error(monkeypatch):
    # Force the lazy torch/transformers import to fail so the ImportError path is exercised
    # deterministically offline -- without importing torch or fetching any model.
    from trigger_audit.scoring import semantic

    real_import_module = semantic.importlib.import_module

    def fake_import_module(name: str, *args: object, **kwargs: object) -> object:
        if name in {"torch", "transformers"}:
            raise ImportError(f"simulated missing dependency: {name}")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(semantic.importlib, "import_module", fake_import_module)
    with pytest.raises(ImportError, match=r"\[hf,generate\]"):
        make_semantic_scorer("nli", model_id="some/model", revision="deadbeef")
    with pytest.raises(ImportError, match="torch and transformers"):
        NLIEntailmentScorer("some/model", "deadbeef")


def test_semantic_assessment_defaults_are_survival_negative():
    assessment = SemanticAssessment(
        threshold=0.5, scorer_id="reference", scorer_revision="reference"
    )
    assert not assessment.semantic_survived
    assert assessment.entail_score == 0.0
    assert assessment.span is None
    assert assessment.window_index is None
