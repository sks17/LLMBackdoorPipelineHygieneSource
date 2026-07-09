"""RAG delivery baseline: embed -> vector store -> retrieve(top_k) -> pack -> template -> tokenize.

The project's first exercise of the retrieval stage. Deliberately a plumbing/delivery check, not a
retrieval-quality study: with a deterministic embedding the ranking is controlled by construction,
so the outcome is attributable to the pipeline rather than a model's semantic quirks. It logs the
trigger's presence at each stage (retrieved / packed / final tokens), which begins the delivery
decomposition ``P(delivered) = P(retrieved) x P(packed | retrieved) x P(final | packed)``, and it is
the first user of ``failure_stage="not_retrieved"``.
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


def _fill_trigger(documents: Sequence[Document], trigger: TriggerSpec) -> list[Document]:
    """Insert the trigger into the trigger-bearing document's slot; copy the rest unchanged."""
    filled: list[Document] = []
    for document in documents:
        if document.trigger_slot and document.trigger_slot in document.content:
            content = document.content.replace(document.trigger_slot, trigger.text)
            filled.append(document.model_copy(update={"content": content}))
        else:
            filled.append(document.model_copy(deep=True))
    return filled


def _pack_prompt(question: str, packed: Sequence[Document]) -> list[ChatMessage]:
    """Assemble a system instruction + a context block (per-chunk doc_id markers) + the question."""
    context = "\n\n".join(f"[{document.doc_id}] {document.content}" for document in packed)
    user = f"Context:\n{context}\n\nQuestion: {question}"
    return [
        ChatMessage(role=Role.SYSTEM, content=_SYSTEM_INSTRUCTION),
        ChatMessage(role=Role.USER, content=user),
    ]


def run_rag_delivery(
    *,
    documents: Sequence[Document],
    trigger: TriggerSpec,
    question: str,
    top_k: int,
    tokenizer_adapter: TokenizerAdapter,
    trial_id: str = "rag_delivery",
    embedding: DeterministicHashEmbedding | None = None,
) -> RagDeliveryResult:
    """Run one RAG delivery condition and return its logged :class:`RagDeliveryResult`.

    The trigger is inserted into the trigger-bearing document, the corpus is embedded into an
    in-memory vector store, ``top_k`` documents are retrieved, all retrieved chunks are packed into
    the prompt (baseline: no compression), and the prompt is templated and tokenized. Presence is
    logged at the retrieved / packed / final-token stages so a failure is attributable to a stage.
    """
    embedding = embedding or DeterministicHashEmbedding()
    filled = _fill_trigger(documents, trigger)

    # Lazy import: the retrieval stack lives behind the `rag`/`frameworks` extras.
    from langchain_core.documents import Document as LangChainDocument
    from langchain_core.vectorstores import InMemoryVectorStore

    store = InMemoryVectorStore(embedding)
    store.add_documents(
        [LangChainDocument(page_content=d.content, metadata={"doc_id": d.doc_id}) for d in filled]
    )
    retrieved = store.as_retriever(search_kwargs={"k": top_k}).invoke(question)
    retrieved_ids = [d.metadata["doc_id"] for d in retrieved]

    by_id = {d.doc_id: d for d in filled}
    packed = [by_id[doc_id] for doc_id in retrieved_ids]  # baseline: pack every retrieved chunk
    packed_ids = list(retrieved_ids)
    dropped_ids = [d.doc_id for d in filled if d.doc_id not in set(packed_ids)]

    renderer = ChatTemplateRenderer(
        tokenizer_adapter, enable_thinking=False, add_generation_prompt=True
    )
    text = renderer.render(_pack_prompt(question, packed))
    final_ids = tokenizer_adapter.encode(text, add_special_tokens=False)
    final_text = tokenizer_adapter.decode(final_ids)

    in_retrieved = any(trigger.text in by_id[doc_id].content for doc_id in retrieved_ids)
    in_packed = any(trigger.text in document.content for document in packed)
    in_final = tokenizer_adapter.locate_token_span(final_text, trigger.text) is not None

    survival_class, failure_stage = _classify(in_retrieved, in_packed, in_final)

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
    )


def _classify(
    in_retrieved: bool, in_packed: bool, in_final: bool
) -> tuple[SurvivalClass, FailureStage]:
    """Map the per-stage presence flags onto a survival class and the earliest failing stage."""
    if in_final:
        return SurvivalClass.EXACT_SURVIVAL, FailureStage.NONE
    if not in_retrieved:
        return SurvivalClass.NO_SURVIVAL, FailureStage.NOT_RETRIEVED
    if not in_packed:
        return SurvivalClass.NO_SURVIVAL, FailureStage.PACKING_BUDGET_EXCLUDED
    return SurvivalClass.NO_SURVIVAL, FailureStage.FINAL_TOKEN_ABSENT
