"""E5 acceptance: RAG chunk-boundary corruption -> partial_survival.

A distinct mechanism from truncation ``boundary_corruption``: a long, multi-word trigger inside a
corpus document is cut by the *word chunker* so a chunk boundary lands inside the trigger. Retrieval
(top_k=1) returns a single chunk holding only a **fragment** of the trigger; the whole trigger never
appears in any one chunk, so it never reaches the packed prompt. No truncation stage is responsible,
so the scorer emits ``partial_survival`` (not ``boundary_corruption``). Runs fully offline with the
deterministic hash embedding + the whitespace reference tokenizer -- no network, HF, or torch.
"""

from __future__ import annotations

from pathlib import Path

from trigger_audit.experiments.rag_survival.chunk_boundary import (
    DEFAULT_CHUNK_SIZE_WORDS,
    chunk_by_words,
    run_chunk_boundary_delivery,
)
from trigger_audit.io.jsonl import read_jsonl_as
from trigger_audit.schemas.documents import Document
from trigger_audit.schemas.results import SurvivalClass
from trigger_audit.schemas.triggers import TriggerSpec, TriggerType
from trigger_audit.tokenization.tokenizer_adapter import SimpleWhitespaceTokenizerAdapter

_REPO = Path(__file__).resolve().parent.parent
_CORPUS_PATH = _REPO / "data" / "documents" / "chunk_boundary_corpus.jsonl"

# An 8-word trigger authored to straddle the chunk boundary at word 8 of the filled trigger document
# (4 prefix words + words 1-4 of the trigger fill chunk 0; words 5-8 of the trigger open chunk 1).
_TRIGGER = TriggerSpec(
    trigger_id="chunk_trigger_001",
    trigger_type=TriggerType.MULTI_TOKEN_PHRASE,
    text="silver kestrel over the amber canyon at dawn",
    slot="{{RETRIEVED_DOC_SLOT}}",
)
# The retrieved fragment (chunk 1's leading trigger words); its overlap with the query ranks it #1.
_EXPECTED_FRAGMENT = "amber canyon at dawn"
QUESTION = "what happens at the amber canyon at dawn"


def _corpus() -> list[Document]:
    return read_jsonl_as(_CORPUS_PATH, Document)


def _run(trial_id: str) -> object:
    return run_chunk_boundary_delivery(
        documents=_corpus(),
        trigger=_TRIGGER,
        question=QUESTION,
        top_k=1,
        tokenizer_adapter=SimpleWhitespaceTokenizerAdapter(),
        trial_id=trial_id,
    )


def test_chunker_straddles_trigger_across_a_boundary():
    # The filled trigger document is split so no single chunk holds the whole trigger, but each of
    # the straddled chunks holds a proper word-fragment of it.
    doc = next(d for d in _corpus() if d.doc_id == "doc_trigger")
    filled = doc.content.replace(doc.trigger_slot or "", _TRIGGER.text)
    chunks = chunk_by_words(filled, chunk_size_words=DEFAULT_CHUNK_SIZE_WORDS)
    assert all(_TRIGGER.text not in chunk for chunk in chunks)  # whole trigger split apart
    assert any("silver kestrel over the" in chunk for chunk in chunks)  # head fragment
    assert any(_EXPECTED_FRAGMENT in chunk for chunk in chunks)  # tail fragment


def test_retrieved_chunk_holds_a_proper_fragment_not_the_whole_trigger():
    result = _run("chunk_boundary_fragment")
    packed_id = result.packed_chunk_ids[0]
    packed_chunk = next(
        chunk
        for doc in _corpus()
        for i, chunk in enumerate(
            chunk_by_words(
                doc.content.replace(doc.trigger_slot or "\0", _TRIGGER.text),
                chunk_size_words=DEFAULT_CHUNK_SIZE_WORDS,
            )
        )
        if f"{doc.doc_id}::c{i}" == packed_id
    )
    # A proper fragment reached the packed chunk, but the whole trigger did not.
    assert _EXPECTED_FRAGMENT in packed_chunk
    assert _TRIGGER.text not in packed_chunk
    assert result.packed_chunk_ids == ["doc_trigger::c1"]


def test_survival_class_is_partial_survival_with_fragment_metadata():
    result = _run("chunk_boundary_partial")
    # partial_survival, NOT boundary_corruption (no truncation stage), NOT exact/no survival.
    assert result.survival_class is SurvivalClass.PARTIAL_SURVIVAL
    # The whole-trigger flags all stay False; only a fragment survived.
    assert result.trigger_present_in_retrieved is False
    assert result.trigger_present_in_packed is False
    assert result.trigger_present_in_final_tokens is False
    meta = result.metadata
    assert meta["mechanism"] == "chunk_boundary_split"
    assert meta["fragment_present_in_packed"] is True
    assert meta["fragment_text"] == _EXPECTED_FRAGMENT
    assert 0 < meta["fragment_word_count"] < meta["trigger_word_count"]
    assert meta["trigger_word_count"] == 8


def test_ranking_and_classification_are_deterministic_across_runs():
    first = _run("chunk_boundary_a")
    second = _run("chunk_boundary_b")
    assert first.retrieved_chunk_ids == second.retrieved_chunk_ids  # stable plumbing, not flaky
    assert first.survival_class is second.survival_class
    assert first.metadata["fragment_text"] == second.metadata["fragment_text"]
