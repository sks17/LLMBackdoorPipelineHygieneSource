# Task 07 — dataset_adapter: ingest real corpora for the H4 arm (delegated)

**Audience:** an implementing agent (Claude). This is the one narrow piece of scaffolding between
the validated pipeline and testing **H4** (do synthetic and real bases behave alike at matched token
length?). Wave 2.

**Goal:** ingest an external conversation/document dataset and normalize it into the **existing**
`BaseConversation` schema (slot form), length-binned to the grid's context lengths, so it flows
through the *unchanged* `expand_manifest` → `run_trial` pipeline. The output is ordinary base
conversations; nothing downstream changes. That is the whole point — H4 validity requires that real
and synthetic bases differ only in *content*, not in how triggers are inserted or scored.

## Datasets (locked in `PRE_REGISTRATION.md`)

`existing_dataset` arm = **LMSYS-Chat-1M** + **WildChat** (real multi-turn user–LLM chat) + **one
long-document corpus** (for the 16k/32k cells the chat sets rarely reach). Safety/red-team sets are
excluded (Project 2+). UltraChat, if used, is a *synthetic-baseline* label, never `existing_dataset`.

**Blocked on format/license (supervisor is obtaining from Saki):** the per-dataset parsers need each
source's actual record structure (role keys, turn nesting, metadata) and its HuggingFace
usage-agreement terms. Build the adapter *structure* now; fill the concrete field mappings against
the real sample records when provided — do not guess the JSON shape.

## Architecture

`src/trigger_audit/io/dataset_adapter.py`:

- `DatasetParser` (ABC): `parse(raw_record: dict) -> list[ChatMessage]` — normalize one source record into our roles (`system/user/assistant/tool`). One concrete parser per source (`LMSYSParser`, `WildChatParser`, `LongDocParser`), each filled from the real format.
- `to_base_conversation(messages, *, base_id, adapter: TokenizerAdapter, target_length: int, positions: list[TriggerPosition]) -> BaseConversation`:
  1. **Length-match** to `target_length` (a grid context length), measured with the *target model's* tokenizer: if short, append structured, non-lorem filler at section boundaries (per `PROJECT_DESCRIPTION` §14); if long, cut at deterministic section boundaries. Record the achieved token count.
  2. **Insert named slots** at the requested controlled positions (`{{PREFIX_SLOT}}`, `{{OLD_TURN_SLOT}}`, `{{RECENT_TURN_SLOT}}`) so the existing slot-aware `TriggerInserter` fills them **identically** to synthetic bases. No trigger text is inserted here — slots only.
  3. Emit a `BaseConversation` (same schema as synthetic) with `conversation_type`, `domain`, `slot_locations`, and a `metadata` note recording `data_source`, source id, and achieved length.
- A small CLI/driver to materialize `data/base_conversations/<source>_<length>_NNN.jsonl` from a HuggingFace dataset (load via the `datasets` library, `hf` extra). Deterministic sampling (fixed seed passed in, not `random`).

Keep `datasets`/HF imports lazy. The adapter emits ordinary base conversations — **do not** add a new schema or a new runner path.

## Validation trial (the H4-readiness check — the acceptance)

- Ingest ~20 real LMSYS conversations, normalize, length-bin to one grid length (e.g. 4k on Qwen3-0.6B), insert slots, and run them through `run_trial` with a prefix canary under `policy="none"`.
- **Assert the survival-class distribution matches the synthetic equivalent under `none`**: every base delivers (`exact_survival`), i.e. real bases flow through the validated pipeline and score identically to synthetic ones when nothing trims. This proves the ingestion produces valid, insertable, scorable bases — the gate to running the real arm at scale.
- Also assert: every emitted base validates against `BaseConversation`; achieved token counts are within a tolerance of the target bin; slots are present and blanked correctly when unused.

## Constraints & verification

- Reuse `BaseConversation`, `TriggerInserter`, `TokenizerAdapter`, `run_trial`. No new schema, no new runner. One header comment per function/class; type hints throughout.
- Respect the source licenses — record the accepted usage terms in `docs/REQUESTED_DOCUMENTATION.md`; do not commit raw dataset content, only the derived base conversations (and keep those git-ignored like other generated data).
- Full gate green; everything else unchanged.
- Supervisor verifies: gate green; the LMSYS validation trial reproduces the synthetic `none`-policy class distribution; the achieved lengths hit their bins; the same `TriggerInserter` fills real-base slots as synthetic ones (H4 insertion symmetry).
