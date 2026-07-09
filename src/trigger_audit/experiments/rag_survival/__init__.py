"""Experiment: RAG delivery — whether a trigger in a corpus document reaches the final prompt.

The retrieval analogue of the survivability audit: documents are embedded, retrieved by a top_k
query, packed into a prompt, templated, and tokenized, logging the trigger's presence at each stage.
Imports LangChain (the `frameworks`/`rag` extras), so this package is opt-in and never loaded on the
CPU-light base import path.
"""

from trigger_audit.experiments.rag_survival.chunk_boundary import (
    chunk_by_words,
    run_chunk_boundary_delivery,
)
from trigger_audit.experiments.rag_survival.embedding import DeterministicHashEmbedding
from trigger_audit.experiments.rag_survival.pipeline import run_rag_delivery

__all__ = [
    "DeterministicHashEmbedding",
    "chunk_by_words",
    "run_chunk_boundary_delivery",
    "run_rag_delivery",
]
