# Task 06b — LangChain RAG delivery, baseline (delegated)

**Audience:** an implementing agent (Claude). You will work with **LangChain** heavily (vector
store, embeddings, retriever). Parse the vendored LangChain docs under `docs-main/` for the exact
classes/calls; this brief fixes the architecture and the contract, not the specific API names.

**Purpose:** the **first exercise of the retrieval stage at all** — nothing before this has touched
documents, chunks, or retrieval. Structurally the biggest remaining addition, on par with what Trial
2 was for message-stage policies. Keep it a **plumbing check**, not a retrieval-quality study.

## New artifact type + schemas

1. `src/trigger_audit/schemas/documents.py` → `Document(doc_id: str, content: str, trigger_slot: str | None = None)` — a corpus document, distinct from conversation messages (per `PROJECT_DESCRIPTION`'s RAG example format).
2. A **RAG delivery result** schema (`schemas/results.py` → `RagDeliveryResult`, or a documented extension of `SurvivalResult`) with the retrieval logging fields:
   `retrieved_chunk_ids`, `packed_chunk_ids`, `final_prompt_token_count`, `trigger_present_in_retrieved`, `trigger_present_in_packed`, `trigger_present_in_final_tokens`, plus `survival_class` and `failure_stage`.
   **Reconciliation flag (do not skip):** these fields are specified in the project's RAG *design* (GENERIC_PLAN §11) but are **not yet in the code schema** — this trial is the one that adds and first exercises them. Document them in `docs/DATA_CONTRACTS.md` with an example record, like the other schemas.

## Minimal corpus — `data/documents/corpus_000.jsonl`

- **1 trigger-bearing document** with the trigger at the document prefix (via its `trigger_slot`, e.g. `{{RETRIEVED_DOC_SLOT}}` at the start of `content`) — the positive control per `PROJECT_DESCRIPTION`.
- **4–5 clean distractor documents** — small, realistic, no trigger.

Insert the trigger into the trigger-bearing document (fill its slot) with the existing slot logic, not into any message. Validate the corpus file with a `--schema document` CLI validation.

## Pipeline (LangChain actual components) — log every stage

`embed corpus → vector store → retriever(top_k) → prompt-template pack → chat template → tokenize → score`

- **Embeddings + vector store:** use LangChain's real components. For a reliable plumbing check the
  ranking must be **deterministic and controlled by construction**, not dependent on a model's
  semantic quirks — prefer `InMemoryVectorStore` with a deterministic embedding (or a small real
  embedding with content crafted so the ordering is unambiguous). FAISS is optional (the `rag` extra);
  `InMemoryVectorStore` avoids it for this baseline. State which you used.
- **Retriever:** `top_k` via the retriever's search kwargs.
- **Packer:** assemble retrieved chunks into a prompt (a fixed template: system instruction + packed
  context block with per-chunk `doc_id` markers + the user question), then the model chat template,
  then tokenize. Log `packed_chunk_ids` and `dropped_chunk_ids`.
- **Scoring:** reuse `score_from_layers` / `locate_token_span` for `trigger_present_in_final_tokens`
  and the survival class; compute `trigger_present_in_retrieved` / `trigger_present_in_packed` by
  string presence in the retrieved / packed chunk text.

Add `langchain-core` + `langchain-community` (and the chosen embedding) to the `frameworks`/`rag`
extras; keep imports lazy so the base package still imports without them.

## Conditions

| condition | config | expected |
|-----------|--------|----------|
| positive control | `top_k` high enough that the trigger-bearing doc is trivially retrieved and packed | `trigger_present_in_retrieved/packed/final = True`, `final_token_trigger_present=True`, `failure_stage=none` |
| excluded | `top_k=1` with distractors ranked **higher by construction** | `trigger_present_in_retrieved=False`, `final_token_trigger_present=False`, `failure_stage=not_retrieved` |

The excluded condition is the **first real use of `failure_stage="not_retrieved"`** — the decomposition
`P(delivered) = P(retrieved) × P(packed | retrieved) × …` starts here.

## Acceptance

- Positive control reaches `final_token_trigger_present=True` with `failure_stage=none`, and all three RAG presence flags True.
- The excluded condition reaches `final_token_trigger_present=False` with `failure_stage=not_retrieved` and `trigger_present_in_retrieved=False`.
- Every intermediate stage is logged (`retrieved_chunk_ids`, `packed_chunk_ids`, `final_prompt_token_count`, the three presence flags).
- The ranking is deterministic: re-running the two conditions yields the same retrieved set (assert this, so the plumbing check is not flaky).

## Constraints & verification

- This is plumbing: no reranker, no compression, no chunking sweep — those are later trials. Keep the corpus tiny.
- Reuse `score_from_layers`, `locate_token_span`, `ChatTemplateRenderer`. One header comment per function/class; type hints throughout.
- Full gate green; Trials 0–5 and Task 06a unchanged and passing. Offline tests use a deterministic embedding + the reference tokenizer where possible; the real-tokenizer path skips when unavailable.
- Supervisor verifies: gate green; positive control delivers with all flags True; excluded condition is `not_retrieved`; ranking is stable across re-runs; the new RAG schema + fields are documented in DATA_CONTRACTS.md.

## Scope note (for Saki)

The source outline says the retrieval logging fields "already exist in the schema" — they do not yet
(they're in the GENERIC_PLAN §11 design, not the code). This trial adds them, so it is genuinely the
first to define *and* exercise the RAG delivery contract. Flagging so the added schema is expected,
not a surprise.
