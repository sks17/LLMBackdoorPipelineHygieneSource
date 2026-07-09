# Task 09 — Synthetic conversation generator

## Objective

Build the **synthetic base-conversation generator**: the data-production module that turns
structured, deterministic *seeds* into realistic, harmless multi-turn chats and long documents, and
emits them as ordinary slot-form `BaseConversation` rows — the same schema and slot mechanism the
real-dataset arm produces. This is the critical-path data source for the H1/H2/H3 grid and the
synthetic half of the H4 (synthetic-vs-real) comparison.

**Non-negotiable design constraint (this is why the task exists):** the generator must reuse the
existing `to_base_conversation()` from `src/trigger_audit/io/dataset_adapter.py` for length-binning
and slot-planting. The generator's *only* new job is producing raw role-tagged `messages`
(the conversation content); everything after that — measuring length with the target tokenizer,
growing/cutting to the grid target, planting `{{...}}` slots — is delegated to the shared function
the dataset arm already uses. This guarantees the property H4 depends on: **synthetic and real base
conversations differ only in content, never in how triggers are inserted or scored.**

## Harmless-canary constraint (carries from every prior task)

Harmless canary strings only. The generator plants **slots, never trigger text** — trigger
insertion is done later by our own `TriggerInserter`. The *generation model must never be asked to
produce triggers, canaries, backdoors, or unsafe behavior*; it produces ordinary helpful-assistant
conversations. Validation (below) rejects any generated content that contains a slot-like or
canary-like string.

## Where this fits (context)

- `to_base_conversation(messages, *, base_id, adapter, target_length, positions, data_source,
  source_record_id, domain, conversation_type, expected_user_task, difficulty, ...)` already:
  length-bins messages with `length_match` (structured filler when short, section-boundary cuts when
  long), plants the position→slot mapping via `_plant_slots`, and returns a `BaseConversation` with
  `metadata={data_source, source_record_id, achieved_token_length, length_tolerance, tokenizer_id,
  planted_positions}`. **Call it; do not reimplement any of it.**
- `slot_for_position` / `target_user_index` / `place_in_content`
  (`pipelines/trigger_insertion.py`) are the single source of truth for placement; `_plant_slots`
  already uses them. You do not touch these.
- Schemas: `ChatMessage(role, content, name?, metadata)`, `Role`
  (`system|user|assistant|tool|document`), `BaseConversation`, `SlotLocation`
  (`schemas/messages.py`); `TriggerPosition` (`schemas/triggers.py`).
- `make_length_measurer(adapter, chat_format=...)` builds the measurer; pass it through so binning
  matches the runner exactly.

## Module to create

`src/trigger_audit/generation/conversation_generator.py` (new package
`src/trigger_audit/generation/` with `__init__.py`). Public surface:

### 1. Seed model

```python
class ConversationSeed(BaseModel):
    seed_id: str                 # deterministic, e.g. "synthetic_mtc_software_debugging_0007"
    family: ConversationFamily   # enum: MULTI_TURN_CHAT | SINGLE_TURN_LONG_DOCUMENT
    domain: str                  # rotating harmless domain
    num_user_turns: int          # >=1; for long-document families, 1
    expected_user_task: str      # short natural-language task description
    difficulty: str              # "easy" | "medium" | "hard"
    index: int                   # ordinal within the deterministic sample
```

`ConversationFamily` is an enum. **This first build implements two families only:**
`MULTI_TURN_CHAT` and `SINGLE_TURN_LONG_DOCUMENT` — both are fully wired end-to-end because their
positions (`prefix/middle/end/old_turn/recent_turn/near_boundary`) already plant correctly via
`_plant_slots`. Design the enum + dispatch so `AGENT_TOOL` and `RAG_LIKE` families can be added
later without refactor, but **do not implement them here** (they are coupled to the `tool_output`/
`retrieved_doc` slot-targeting extension, a separate roadmap item — see "Out of scope").

Provide `sample_seeds(count, *, families, domains, seed=0) -> list[ConversationSeed]`:
deterministic (no global RNG; derive every choice from `seed + index` like
`synthetic_chat_records` does), balanced across families × domains × difficulty.

### 2. Generation backend (pluggable)

```python
class GenerationBackend(ABC):
    name: str  # recorded as the `generation_model` covariate on every base
    @abstractmethod
    def generate(self, seed: ConversationSeed) -> list[ChatMessage]: ...
```

The backend returns raw role-tagged messages for the seed — **no slots, no triggers**. Implement:

- **`MockBackend`** — fully deterministic, offline, no LLM. Produces valid, harmless, seed-shaped
  conversations (rotate realistic content by domain, like the existing `synthetic_chat_records`
  filler tone). Used by the entire offline test suite **and** as the fallback when a real backend
  fails validation repeatedly. `name = "mock"`.
- **`OllamaBackend`** — generates via a locally running Ollama server (we have Qwen3 models pulled
  under `~/.ollama/models`). Constructor: `OllamaBackend(model: str, *, host="http://localhost:11434",
  timeout=..., max_attempts=3)`; `name = f"ollama:{model}"`. It builds a **content-only** prompt
  from the seed (ask for a realistic {family} conversation in {domain} about {task}, {num_user_turns}
  user turns, as a JSON array of `{"role","content"}` objects, roles limited to
  system/user/assistant, *explicitly instruct: no placeholders, no bracketed tokens, no all-caps
  code words*), calls `POST /api/chat` (or `/api/generate`) with `stream=false` and
  `options={"temperature":0.7,"seed":<derived>}`, parses the JSON array, and returns `ChatMessage`s.
  On unparseable / invalid output it retries up to `max_attempts`, then raises
  `GenerationError`. Keep the `requests`/`httpx` import lazy and local so the offline test suite
  never imports it.

Leave **documented stubs** (raising `NotImplementedError` with a clear message, mirroring the
dataset `_BlockedParser` pattern) for:
- `TransformersBackend` — for cluster GPU generation (HF `generate`), and
- `ApiBackend` — for fast API models (e.g. Haiku) once an API key/quota is provided.

Both stubs must document exactly what they need to be filled in (tracked in
`docs/REQUESTED_DOCUMENTATION.md`). A `BACKENDS: dict[str, ...]` registry maps names to classes.

### 3. Validation (shared, backend-agnostic)

`validate_generated(messages, seed) -> None` raises `GenerationValidationError` when the content is
unusable, checking at minimum:
- non-empty; every `role` is a valid `Role`; the family's structural contract holds
  (`MULTI_TURN_CHAT`: ≥2 user turns and ≥1 assistant turn; `SINGLE_TURN_LONG_DOCUMENT`: exactly 1
  user turn carrying substantial content);
- a `system` message is present (synthesize a default one if the model omitted it, matching
  `MockChatParser`'s behavior — do not fail for this alone);
- **no slot-like substring** `{{` or `}}` anywhere (would collide with slot planting);
- **no canary-like substring**: reject a small denylist (`CANARY`, `TRIGGER`, `BACKDOOR`) and any
  run matching `\b[A-Z0-9]{6,}\b` (guards against the model emitting a canary-shaped token);
- no message content is empty/whitespace-only.

The generator calls the backend, then `validate_generated`; on failure it retries the backend
(bounded), and as a last resort falls back to `MockBackend` for that seed — **recording the actual
producing backend name in `generation_model`** (so a fallback is never silently attributed to the
real model). Count and `log` fallbacks; never stall.

### 4. Emit

`generate_base_conversation(seed, *, backend, adapter, target_length, positions, chat_format="chat",
measure=None) -> BaseConversation`:
1. `messages = backend.generate(seed)` (with validation + fallback as above);
2. return `to_base_conversation(messages, base_id=seed.seed_id_for(target_length), adapter=adapter,
   target_length=target_length, positions=positions, data_source="synthetic",
   source_record_id=seed.seed_id, conversation_type=<family value>,
   expected_user_task=seed.expected_user_task, domain=seed.domain, difficulty=seed.difficulty,
   measure=measure)`, then **merge** `generation_model=<producing backend name>` and
   `seed_id=seed.seed_id` into the returned base's `metadata` (do not drop the fields
   `to_base_conversation` already set).

`materialize_synthetic_corpus(*, backend, adapter, target_length, positions, count, families,
domains, output_path, seed=0, chat_format="chat") -> list[BaseConversation]`: sample seeds, generate
each base, `write_jsonl`. Base ids follow `synthetic_<length>_NNN` (parallels the dataset arm's
`<source>_<length>_NNN`). Only derived bases are written; keep output under git-ignored `data/`.

### 5. CLI

A `main(argv)` + `python -m trigger_audit.generation.conversation_generator` shim mirroring
`dataset_adapter.main`: flags `--model-id`, `--tokenizer-backend {hf,simple}`, `--target-length`,
`--count`, `--families`, `--domains`, `--positions`, `--generation-backend {mock,ollama}`,
`--generation-model` (the Ollama model tag, e.g. `qwen3:1.7b`), `--chat-format {chat,base}`,
`--seed`, `--output`. Default backend `mock` so the CLI runs with zero external dependencies.

## Tests — `tests/test_conversation_generator.py` (offline, MockBackend only)

- `sample_seeds` is deterministic (same seed → identical seeds) and balanced across families/domains.
- With `MockBackend`, every family produces messages that pass `validate_generated`.
- `validate_generated` rejects: `{{SLOT}}` content, a `CANARY`-containing message, an all-caps
  `A1B2C3` token, an empty message, a `MULTI_TURN_CHAT` with only one user turn.
- **Symmetry test (the important one):** a base produced by `generate_base_conversation` with
  `MockBackend` is structurally interchangeable with one produced by the dataset arm's
  `to_base_conversation` on mock records — same `slot_locations` slots for the same `positions`,
  same metadata keys, `data_source == "synthetic"`, and every planted slot placeholder is present in
  exactly one message's content. Assert that `TriggerInserter().insert(base, trigger, position)`
  fills the planted slot (round-trips) for `prefix`, `end`, `old_turn`, `recent_turn`.
- Length binning: for a small `target_length`, `metadata["achieved_token_length"]` is within
  `metadata["length_tolerance"]` of the target (reuse the reference/simple tokenizer adapter used by
  the other offline tests — **do not require HF or a network pull in the test suite**).
- `generation_model` is recorded and equals the producing backend's `name`; a forced-failure backend
  falls back to mock and records `generation_model == "mock"`.

Use the same offline tokenizer adapter the existing tests use (simple backend). No test may require
Ollama, HF downloads, or network.

## Acceptance gate (all must pass; report exact output)

- `ruff check .` and `ruff format --check .` clean.
- `mypy src` clean (match the repo's existing strictness; keep heavy imports lazy).
- `pytest -q` — full suite green, including the new tests. Report the pass count delta.
- The new module imports with **no** hard dependency on `requests`/`httpx`/`datasets`/`transformers`
  (all lazy/local). Verify `python -c "import trigger_audit.generation.conversation_generator"`
  succeeds in the base environment.

## Out of scope (do NOT do here — these are separate, tracked roadmap items)

- `AGENT_TOOL` / `RAG_LIKE` families and `tool_output` / `retrieved_doc` slot-targeting. These are
  coupled: `_plant_slots`/`target_user_index` currently target user messages for every position, so
  planting a `tool_output`/`retrieved_doc` slot needs an extension to target the last `tool`/
  `document` message. That extension + these two families land together in a later task. Design the
  family enum/dispatch to accommodate them, but do not implement them.
- Live Ollama generation runs / corpus production at scale (a supervised smoke + the cluster wiring
  are separate steps). `OllamaBackend` must be *implemented and correct*, but it is exercised live
  by the supervisor, not by the offline gate.
- `TransformersBackend` / `ApiBackend` bodies (documented stubs only, pending cluster/API access).

## Do not

- Do not defer any implementation to an external file (no `cursor.md`-style handoff). Write the code.
- Do not reimplement length-binning, slot-planting, or placement — call the shared functions.
- Do not ask the generation model for triggers, or plant trigger text — slots only.
- Do not add a network/HF/Ollama dependency to the offline test path.
