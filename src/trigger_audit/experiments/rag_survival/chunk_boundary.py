"""RAG chunk-boundary corruption: a long trigger split by the *chunker* (not truncation).

A distinct delivery-failure mechanism from truncation ``boundary_corruption``. Here a corpus
document carrying a long, multi-word trigger is cut into chunks by a deterministic word chunker so
that a chunk boundary lands *inside* the trigger. Retrieval then returns a single chunk holding only
a **fragment** of the trigger; the whole trigger never appears in any one chunk and so never reaches
the packed prompt. That partial overlap arises with **no truncation stage** responsible, so it is
classified ``partial_survival`` (reserving ``boundary_corruption`` for truncation cuts).

Like the RAG delivery baseline this is a plumbing/delivery check, not a retrieval-quality study: the
deterministic embedding fixes the ranking by construction, and the chunk boundary is positioned
inside the trigger by construction, so the outcome is attributable to the pipeline. Reuses the
Trial 6b :class:`RagDeliveryResult` contract.
"""

from __future__ import annotations

from collections.abc import Sequence

from trigger_audit.experiments.rag_survival.embedding import DeterministicHashEmbedding
from trigger_audit.prompts.chat_template import ChatTemplateRenderer
from trigger_audit.schemas.documents import Document
from trigger_audit.schemas.messages import ChatMessage, Role
from trigger_audit.schemas.results import FailureStage, RagDeliveryResult, SurvivalClass
from trigger_audit.schemas.triggers import TriggerSpec
from trigger_audit.tokenization.tokenizer_adapter import TokenizerAdapter

_SYSTEM_INSTRUCTION = "You are a helpful assistant. Answer the question using only the context."

# Default chunk width for the word chunker. The chunk-boundary corpus is authored so that, with this
# width and a multi-word trigger, a chunk boundary falls strictly inside the trigger's words.
DEFAULT_CHUNK_SIZE_WORDS = 8


def chunk_by_words(text: str, *, chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS) -> list[str]:
    """Split ``text`` into contiguous fixed-size word chunks (whitespace split, no overlap).

    A deterministic, dependency-free chunker: the same text always yields the same chunks, so a
    trigger positioned to straddle a boundary is split identically on every run. Trailing words form
    a final short chunk. This is trigger-agnostic on purpose -- the straddle comes from how the
    corpus positions the trigger, not from the chunker inspecting it.
    """
    if chunk_size_words < 1:
        raise ValueError(f"chunk_size_words must be >= 1, got {chunk_size_words}")
    words = text.split()
    return [
        " ".join(words[start : start + chunk_size_words])
        for start in range(0, len(words), chunk_size_words)
    ]


class _Chunk:
    """A single retrievable chunk of a source document, carrying a deterministic chunk id."""

    def __init__(self, chunk_id: str, doc_id: str, content: str) -> None:
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.content = content


def _fill_trigger(document: Document, trigger: TriggerSpec) -> str:
    """Return the document content with its trigger slot replaced by the trigger text (if any)."""
    if document.trigger_slot and document.trigger_slot in document.content:
        return document.content.replace(document.trigger_slot, trigger.text)
    return document.content


def _chunk_corpus(
    documents: Sequence[Document],
    trigger: TriggerSpec,
    *,
    chunk_size_words: int,
) -> list[_Chunk]:
    """Fill the trigger, then chunk every document into fixed-size word chunks with stable ids.

    Chunk ids are ``<doc_id>::c<n>`` so a fragment is traceable to its source document and position.
    """
    chunks: list[_Chunk] = []
    for document in documents:
        content = _fill_trigger(document, trigger)
        for index, piece in enumerate(chunk_by_words(content, chunk_size_words=chunk_size_words)):
            chunks.append(_Chunk(f"{document.doc_id}::c{index}", document.doc_id, piece))
    return chunks


def _longest_trigger_fragment(trigger_text: str, haystack: str) -> tuple[str, int, int]:
    """Return the longest contiguous run of trigger words present in ``haystack`` (word-aligned).

    Returns ``(fragment_text, fragment_word_count, trigger_word_count)``. A run equal to the full
    trigger means the whole trigger survived; a shorter non-empty run is a proper fragment; an empty
    run means no part of the trigger reached ``haystack``. Matching is on whitespace-joined word
    runs so a fragment is a genuine word-boundary piece of the trigger, not an incidental substring.
    """
    words = trigger_text.split()
    total = len(words)
    for length in range(total, 0, -1):
        for start in range(0, total - length + 1):
            candidate = " ".join(words[start : start + length])
            if candidate in haystack:
                return candidate, length, total
    return "", 0, total


def _pack_prompt(question: str, packed: Sequence[_Chunk]) -> list[ChatMessage]:
    """Assemble a system instruction + a context block (per-chunk id markers) + the question."""
    context = "\n\n".join(f"[{chunk.chunk_id}] {chunk.content}" for chunk in packed)
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        ChatMessage(role=Role.SYSTEM, content=_SYSTEM_INSTRUCTION),
        ChatMessage(role=Role.USER, content=user),
    ]


def _classify(
    whole_in_final: bool,
    fragment_word_count: int,
    trigger_word_count: int,
) -> tuple[SurvivalClass, FailureStage]:
    """Map whole-trigger and fragment presence onto a survival class and failure stage.

    ``partial_survival`` (not ``boundary_corruption``) is emitted when only a proper fragment of the
    trigger reaches the packed prompt: the mechanism is a chunk-boundary split, and no truncation
    stage is responsible. The chunker has no dedicated ``FailureStage`` in the shared schema, so the
    mechanism is recorded in ``metadata`` and the stage is left ``NONE`` (a partial delivery, not a
    clean stage-attributable drop).
    """
    if whole_in_final:
        return SurvivalClass.EXACT_SURVIVAL, FailureStage.NONE
    if 0 < fragment_word_count < trigger_word_count:
        return SurvivalClass.PARTIAL_SURVIVAL, FailureStage.NONE
    return SurvivalClass.NO_SURVIVAL, FailureStage.NOT_RETRIEVED


def run_chunk_boundary_delivery(
    *,
    documents: Sequence[Document],
    trigger: TriggerSpec,
    question: str,
    top_k: int,
    tokenizer_adapter: TokenizerAdapter,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    trial_id: str = "rag_chunk_boundary",
    embedding: DeterministicHashEmbedding | None = None,
) -> RagDeliveryResult:
    """Run one chunk-boundary condition and return its logged :class:`RagDeliveryResult`.

    The trigger-bearing document is filled and chunked by :func:`chunk_by_words` so a boundary lands
    inside the trigger; the corpus chunks are embedded into an in-memory vector store; ``top_k``
    chunks are retrieved and packed; the prompt is templated and tokenized. The three
    ``trigger_present_in_*`` flags track the **whole** trigger (all ``False`` when only a fragment
    survives, mirroring how ``boundary_corruption`` leaves ``trigger_exact_survived=False``); the
    surviving fragment is recorded in ``survival_class=partial_survival`` and in ``metadata``.
    """
    embedding = embedding or DeterministicHashEmbedding()
    chunks = _chunk_corpus(documents, trigger, chunk_size_words=chunk_size_words)

    # Lazy import: the retrieval stack lives behind the `rag`/`frameworks` extras.
    from langchain_core.documents import Document as LangChainDocument
    from langchain_core.vectorstores import InMemoryVectorStore

    store = InMemoryVectorStore(embedding)
    store.add_documents(
        [
            LangChainDocument(page_content=c.content, metadata={"chunk_id": c.chunk_id})
            for c in chunks
        ]
    )
    retrieved = store.as_retriever(search_kwargs={"k": top_k}).invoke(question)
    retrieved_ids = [d.metadata["chunk_id"] for d in retrieved]

    by_id = {c.chunk_id: c for c in chunks}
    packed = [by_id[chunk_id] for chunk_id in retrieved_ids]  # baseline: pack every retrieved chunk
    packed_ids = list(retrieved_ids)
    dropped_ids = [c.chunk_id for c in chunks if c.chunk_id not in set(packed_ids)]

    renderer = ChatTemplateRenderer(
        tokenizer_adapter, enable_thinking=False, add_generation_prompt=True
    )
    text = renderer.render(_pack_prompt(question, packed))
    final_ids = tokenizer_adapter.encode(text, add_special_tokens=False)
    final_text = tokenizer_adapter.decode(final_ids)

    packed_context = "\n\n".join(chunk.content for chunk in packed)
    fragment_text, fragment_words, trigger_words = _longest_trigger_fragment(
        trigger.text, packed_context
    )
    # Whole-trigger presence at each stage (a fragment leaves every whole-trigger flag False).
    in_retrieved = any(trigger.text in by_id[chunk_id].content for chunk_id in retrieved_ids)
    in_packed = trigger.text in packed_context
    in_final = tokenizer_adapter.locate_token_span(final_text, trigger.text) is not None

    survival_class, failure_stage = _classify(in_final, fragment_words, trigger_words)

    return RagDeliveryResult(
        trial_id=trial_id,
        model_id=tokenizer_adapter.tokenizer_id,
        tokenizer_id=tokenizer_adapter.tokenizer_id,
        trigger_id=trigger.trigger_id,
        trigger_text=trigger.text,
        top_k=top_k,
        embedding_id=getattr(embedding, "embedding_id", type(embedding).__name__),
        retrieved_chunk_ids=retrieved_ids,
        packed_chunk_ids=packed_ids,
        dropped_chunk_ids=dropped_ids,
        trigger_present_in_retrieved=in_retrieved,
        trigger_present_in_packed=in_packed,
        trigger_present_in_final_tokens=in_final,
        final_prompt_token_count=len(final_ids),
        survival_class=survival_class,
        failure_stage=failure_stage,
        metadata={
            "mechanism": "chunk_boundary_split",
            "chunk_size_words": chunk_size_words,
            "fragment_present_in_packed": 0 < fragment_words < trigger_words,
            "fragment_text": fragment_text,
            "fragment_word_count": fragment_words,
            "trigger_word_count": trigger_words,
        },
    )
