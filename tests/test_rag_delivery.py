"""Task 06b acceptance: LangChain RAG delivery baseline.

The first exercise of the retrieval stage. With a deterministic embedding the ranking is controlled
by construction: the trigger-bearing document is off-topic to the query, so it ranks last. A high
``top_k`` retrieves and packs it (positive control -> delivered); ``top_k=1`` excludes it (the first
real use of ``failure_stage="not_retrieved"``). Runs offline (deterministic embedding + reference
tokenizer); a real-tokenizer variant skips when transformers is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trigger_audit.experiments.rag_survival import run_rag_delivery
from trigger_audit.io.jsonl import read_jsonl_as
from trigger_audit.io.stores import TriggerStore
from trigger_audit.schemas.documents import Document
from trigger_audit.schemas.results import FailureStage, SurvivalClass
from trigger_audit.tokenization.tokenizer_adapter import (
    HFTokenizerAdapter,
    SimpleWhitespaceTokenizerAdapter,
)

_REPO = Path(__file__).resolve().parent.parent
_CORPUS_PATH = _REPO / "data" / "documents" / "corpus_000.jsonl"
_TRIGGERS_PATH = _REPO / "data" / "triggers" / "triggers.jsonl"
QUESTION = "How do I speed up slow database queries using an index?"


def _corpus() -> list[Document]:
    return read_jsonl_as(_CORPUS_PATH, Document)


def _trigger():
    return TriggerStore(_TRIGGERS_PATH).get("rand_001")


def _run(top_k: int, adapter, trial_id: str):
    return run_rag_delivery(
        documents=_corpus(),
        trigger=_trigger(),
        question=QUESTION,
        top_k=top_k,
        tokenizer_adapter=adapter,
        trial_id=trial_id,
    )


def test_positive_control_delivers_with_all_flags_true():
    result = _run(top_k=5, adapter=SimpleWhitespaceTokenizerAdapter(), trial_id="rag_positive")
    assert result.trigger_present_in_retrieved is True
    assert result.trigger_present_in_packed is True
    assert result.trigger_present_in_final_tokens is True
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL
    assert result.failure_stage is FailureStage.NONE
    assert "doc_trigger" in result.retrieved_chunk_ids
    assert result.packed_chunk_ids == result.retrieved_chunk_ids  # baseline packs all retrieved
    assert result.final_prompt_token_count > 0


def test_excluded_condition_is_not_retrieved():
    result = _run(top_k=1, adapter=SimpleWhitespaceTokenizerAdapter(), trial_id="rag_excluded")
    assert result.trigger_present_in_retrieved is False
    assert result.trigger_present_in_packed is False
    assert result.trigger_present_in_final_tokens is False
    assert result.survival_class is SurvivalClass.NO_SURVIVAL
    assert result.failure_stage is FailureStage.NOT_RETRIEVED
    assert "doc_trigger" not in result.retrieved_chunk_ids
    assert len(result.retrieved_chunk_ids) == 1  # only the top distractor


def test_retrieval_ranking_is_deterministic_across_runs():
    adapter = SimpleWhitespaceTokenizerAdapter()
    first = _run(top_k=1, adapter=adapter, trial_id="rag_a")
    second = _run(top_k=1, adapter=adapter, trial_id="rag_b")
    assert first.retrieved_chunk_ids == second.retrieved_chunk_ids  # stable plumbing, not flaky


def test_positive_control_real_tokenizer():
    pytest.importorskip("transformers")
    try:
        adapter = HFTokenizerAdapter("Qwen/Qwen3-0.6B")
    except Exception as exc:  # offline / no-cache environments should skip, not fail
        pytest.skip(f"Qwen3-0.6B tokenizer unavailable: {exc}")

    result = _run(top_k=5, adapter=adapter, trial_id="rag_positive_qwen")
    assert result.trigger_present_in_final_tokens is True
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL
    assert result.failure_stage is FailureStage.NONE
