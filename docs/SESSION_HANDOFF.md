# Session handoff — 2026-07-02

Self-contained context for a fresh agent picking up Project 1. Read this top to bottom; it
references the durable docs (`EXPERIMENT_DESIGN.md`, `PRE_REGISTRATION.md`, `DATA_CONTRACTS.md`,
`REQUESTED_DOCUMENTATION.md`, `docs/tasks/*`) where detail lives.

---

## 0. How to work in this repo (roles + constraints — read first)

- **Supervisor/delegate model.** The user (Saki) supervises and **delegates coding to subagents**.
  As the lead agent you **decompose experiments into precise coding specs** (write them to
  `docs/tasks/NN-*.md`), **delegate** to subagents, and **verify** their output (re-run the gate +
  a real cross-check) before accepting. Do the coding yourself **only** for tiny bugfixes and
  high-stakes core pieces. Verify, don't trust: an agent reporting "green" is not acceptance.
- **Harmless canary strings only.** Never create harmful backdoor payloads or unsafe model
  behavior. The generation model is never asked to produce triggers/canaries; our own code plants
  slots and inserts canaries. This is a *delivery* audit, not activation/backdoor detection.
- **Global CLAUDE.md rules:** begin every response with "Hello Saki."; never add the agent as a
  commit co-author; never hand-edit `CHANGELOG.md` or files marked auto-generated; prefer quality /
  simplicity / robustness / maintainability over development cost; reproduce bugs E2E before fixing;
  fix lint/test/flakiness you notice even if unrelated.
- **Environment:** Windows, PowerShell primary (5.1 quirks) + a Bash tool. Repo is **not** a git
  repo. Python venv at `./.venv` (3.12.4). Gate = `ruff check .`, `ruff format --check .`,
  `mypy src`, `pytest`. Ollama is installed with `qwen3:0.6b/1.7b/4b/8b`, `qwen2.5:0.5b`, `llama3`
  pulled. Never commit raw dataset content or large generated data (git-ignored under `data/`).

---

## 1. The project in one paragraph

**Project 1 = trigger-delivery / prompt-survivability audit.** Question: when a harmless canary is
placed in raw user input, does it survive the real prompt pipeline — chat templating, memory/trim
policy, truncation, RAG packing, tokenization — into the **final model-visible tokens**? We log four
layers (L1 raw messages → L2 post-memory-policy → L3 post-template text → L4 final token ids after
truncation) and score survival (exact / token / partial / boundary-corruption). The thesis: much of
what looks like "backdoor robustness failure" is actually **delivery failure**, and you must prove
delivery before evaluating activation. Hypotheses: H1 policy affects delivery; H2 model-invariance;
H3 position×policy interaction; H4 synthetic-vs-real at matched length.

---

## 2. What this session accomplished

1. **Scope audit — confirmed we are building the *full* Project 1, not the minimal first version.**
   Recorded in `EXPERIMENT_DESIGN.md` §1a as a table. The mechanism harness is at full fidelity and
   *exceeds* the first prompt (RAG, LangChain trim+RAG, 4-template diversity, counterfactual
   pairing). **Four Project-1 items remain owed** (tracked, not dropped):
   1. **Analysis / finding-tables layer** — the actual scientific output (survival-rate tables by
      policy×position×model, McNemar's on counterfactual pairs, the "backdoor-failure is
      delivery-failure" tables). Only a basic `aggregate_survival` exists. **Essential.**
   2. **Activation / generation phase** — the secondary `P(activation | delivered)` decomposition
      (currently a stub in `runner.py::_maybe_generate`). Optional/secondary per GENERIC_PLAN, but
      completes the headline.
   3. **Extra trigger types + positions** — `natural_phrase`, `unicode`; `system` (role migration),
      `tool_output` (needs agent/tool family + slot-targeting extension).
   4. **RAG chunk-boundary corruption** — distinct mechanism from truncation-boundary.
   *Deferred by decision (not omission):* `summarize_old`/`summary+recent` (need a semantic scorer).

2. **Built the synthetic conversation generator (Task 09).** New module
   `src/trigger_audit/generation/conversation_generator.py` + `tests/test_conversation_generator.py`.
   Delegated to a subagent against `docs/tasks/09-conversation-generator.md`, then verified and
   hardened directly (details in §3). This is the synthetic arm's data source.

3. **Received dataset licenses** for the H4 real arm: `lmsys/lmsys-chat-1m` and `allenai/WildChat`
   (both gated; access now granted). This unblocks the previously-stubbed parsers.

4. **Produced the data-setup plan** (the full plan is §5 below — the primary thing to preserve).

5. **Gate is green:** **185 passed, 0 failed**; `ruff`/`ruff format`/`mypy src` all clean.

---

## 3. The generator — precise state (the code built/hardened this session)

**File:** `src/trigger_audit/generation/conversation_generator.py`. **Tests:**
`tests/test_conversation_generator.py` (24 tests, fully offline — MockBackend + pure parser tests,
no network/HF/Ollama). Uses the `simple_adapter` (`SimpleWhitespaceTokenizerAdapter`) fixture.

**The load-bearing design invariant:** the generator is a **peer of `DatasetParser`**. It only
produces raw role-tagged `messages`, then hands them to the *same*
`trigger_audit.io.dataset_adapter.to_base_conversation()` that the real-dataset arm uses for
length-binning + slot-planting. Result: synthetic and real bases are structurally identical and
differ only in content provenance — this is exactly what H4 validity requires. **Do not
reimplement length-binning/slot-planting; always route through `to_base_conversation`.**

**Public API:**
- `ConversationFamily` enum: `MULTI_TURN_CHAT`, `SINGLE_TURN_LONG_DOCUMENT` (both fully wired),
  `AGENT_TOOL`, `RAG_LIKE` (documented `NotImplementedError` stubs — coupled to the
  `tool_output`/`retrieved_doc` slot-targeting extension, a later task). `IMPLEMENTED_FAMILIES`
  holds the first two.
- `ConversationSeed` (Pydantic): `seed_id, family, domain, num_user_turns, expected_user_task,
  difficulty, index`; `.seed_id_for(target_length)` → `synthetic_<length>_NNN`.
- `sample_seeds(count, *, families, domains, seed, difficulties)` — deterministic, balanced
  (mixed-radix over `seed+index`, no global RNG).
- `validate_generated(messages, seed)` — rejects: empty/whitespace; disallowed roles; `{{`/`}}`
  slot-braces; canary denylist (`canary/trigger/backdoor`); `\b[A-Z0-9]{6,}\b` canary-shaped runs;
  family contract (MTC ≥2 user + ≥1 assistant; SLD exactly 1 substantial user turn). Synthesizes a
  missing system turn rather than failing.
- `GenerationBackend` (ABC, `.name`, `.generate(seed)->list[ChatMessage]`):
  - `MockBackend` (`name="mock"`) — deterministic, offline; drives all tests and is the fallback.
  - `OllamaBackend(model, *, host, timeout, max_attempts, think=False, options=None)`
    (`name=f"ollama:{model}"`) — live local generation. **Hardened this session** (see below).
  - `TransformersBackend`, `ApiBackend` — documented stubs (pending cluster-GPU / API access).
  - `BACKENDS` registry maps names → classes.
- `generate_base_conversation(seed, *, backend, adapter, target_length, positions, chat_format,
  measure)` → `BaseConversation`. Validates with bounded retry, **falls back to MockBackend**, and
  merges `generation_model=<producing backend name>` + `seed_id` into metadata (mock fallback is
  recorded **honestly**, never attributed to the real model). `data_source="synthetic"`.
- `materialize_synthetic_corpus(...)` → writes a JSONL of derived bases (ids `synthetic_<length>_NNN`).
- CLI: `python -m trigger_audit.generation.conversation_generator --model-id ... --tokenizer-backend
  {hf,simple} --target-length ... --count ... --families ... --domains ... --positions ...
  --generation-backend {mock,ollama} --generation-model qwen3:1.7b --chat-format {chat,base}
  --seed ... --output ...` (default backend `mock`, so it runs with zero external deps).

**Hardening applied after live smokes (all in the same file, all tested):**
1. `think=False` on the Ollama request — Qwen3 is a thinking model; `<think>` blocks slow generation
   and pollute JSON. Escape hatch `think=None` for non-thinking models.
2. **Tolerant output parsing** (`_extract_message_dicts` via `json.JSONDecoder.raw_decode`) — accepts
   a JSON array, several concatenated bare objects, a single role-keyed dict
   (`{"system":..,"user":..}`, expanded in order), and fenced / `<think>`-wrapped output. Small
   local models emit all of these.
3. **Family-specific prompts** (`_structure_instructions`) — the SLD prompt now explicitly demands
   "exactly one user message containing a multi-paragraph document + one question"; this took SLD
   real-yield from 0% → ~100% on qwen3:1.7b.
4. **Anti-repetition decoding** (`_DEFAULT_OPTIONS = {temperature:0.7, repeat_penalty:1.3,
   repeat_last_n:256}`, overridable via `options=`) — small models were emitting the same
   conversation several times (passes validation but degenerate). `repeat_last_n:256` spans a full
   short conversation so the penalty sees the block it would duplicate.

**Verified live (local Ollama):** slots plant correctly on real model content
(`{{PREFIX_SLOT}}` on first user turn, `{{RECENT_TURN_SLOT}}` on last); safety holds (no
canary/slot leakage). Real-generation yield with small local models is partial (qwen3:1.7b 4/6,
qwen3:4b 3/6 at count=6) — **expected**: small models under-comply on exact structure; the mock
fallback absorbs it. **The real corpus quality is expected to come from Haiku/larger API models,
with local Qwen as a style-diversity source.**

**Repetition finding (final smoke, qwen3:4b, count=6):** `real=3/6`; on the 3 real multi-turn bases
`distinct/total` messages were 13/20, 15/15, 15/20. So `repeat_penalty=1.3` **reduced but did not
eliminate** repetition — qwen3:4b also **over-produces turns** (15–20 messages when ~5–9 were
requested). These bases are structurally valid (length-binning never adds messages, so the
over-length + repetition is model-origin, not a pipeline bug), but they are degenerate data that
currently **passes `validate_generated`**. **Recommended cheap follow-up (not yet done):** add a
repetition/over-length guard to `validate_generated` — reject a base whose distinct-message ratio is
too low, or whose message count greatly exceeds `1 + 2*num_user_turns` — so degenerate small-model
output falls back to mock instead of entering the corpus. Deferred deliberately during the handoff;
it is a behavior-changing quality gate, not a correctness fix, and does not block anything.

---

## 4. The data arms and their exact contracts

Everything funnels into the **unchanged** `expand_manifest → run_trial` grid. Bases come from four
sources, all producing the **same** slot-form `BaseConversation` via `to_base_conversation`:

| Source | Module / status |
|---|---|
| **Synthetic** | `generation/conversation_generator.py` — **built** (§3). Local Qwen + mock live; API/GPU backends stubbed. |
| **LMSYS-Chat-1M** | `io/dataset_adapter.py::LMSYSParser` — **stub (`NotImplementedError`)**, now unblocked by the license. |
| **WildChat** | `io/dataset_adapter.py::WildChatParser` — **stub**, now unblocked. |
| **Long-document** | `io/dataset_adapter.py::LongDocParser` — **stub**, needs a corpus choice. |

**`io/dataset_adapter.py` (already built + validated with `MockChatParser`):** `DatasetParser` ABC;
`to_base_conversation(messages, *, base_id, adapter, target_length, positions, data_source,
source_record_id, conversation_type, domain, expected_user_task, difficulty, measure)`; `length_match`
(structured filler when short, section-boundary cuts when long); `make_length_measurer`;
`materialize_base_conversations`; `load_raw_records` (deterministic seeded sampling, lazy `datasets`
import); CLI. Emitted metadata: `data_source, source_record_id, achieved_token_length,
length_tolerance, tokenizer_id, planted_positions`.

**Known record formats for the two licensed datasets** (for filling the parsers):
- `lmsys/lmsys-chat-1m`: `conversation` = list of `{"role":"user"|"assistant","content":str}`; plus
  `conversation_id, model, turn, language, openai_moderation, redacted`. No system turns → synthesize
  one (like `MockChatParser`).
- `allenai/WildChat`: `conversation` = list of turns with `role`/`content` **plus** heavy metadata
  (`toxic, redacted, language, country, hashed_ip, timestamp`). Keep **only role+content**.

**Slot placement source of truth** (`pipelines/trigger_insertion.py`): `slot_for_position`,
`target_user_index`, `place_in_content`; slots `{{PREFIX_SLOT}} {{MIDDLE_SLOT}} {{END_SLOT}}
{{OLD_TURN_SLOT}} {{RECENT_TURN_SLOT}} {{BOUNDARY_SLOT}} {{TOOL_OUTPUT_SLOT}} {{RETRIEVED_DOC_SLOT}}`.
NOTE: `tool_output`/`retrieved_doc` currently mis-target user messages — extending
`target_user_index` to target the last `tool`/`document` message is part of the deferred position
work.

---

## 5. THE DATA-SETUP PLAN (verbatim — the primary artifact to preserve)

> This is the plan produced at the end of the session in response to "robustly set up all of the
> data we will be needing to set up trials." It is **pending the user's 244-line deep-research
> feedback**, which was referenced but **not visible** to the agent (see §6) and will likely reshape
> the corpus/scale/sampling choices (flagged 🔶).

**5.1 Real arm — unblocked by the licenses.** Fill `LMSYSParser` + `WildChatParser` against the
record formats in §4 (map role/content, synthesize a system turn, drop all metadata). Both are gated;
the license access is what was blocking them.

**5.2 Safety / privacy / licensing controls (non-negotiable defaults for real human data):**
1. **Drop toxic/flagged conversations** — filter LMSYS `openai_moderation` and WildChat
   `toxic`/`detoxify`. We need only benign carrier conversations; aligns with harmless-only.
2. **Strip all metadata/PII** — keep role+content only; discard IP/country/timestamps; prefer the
   already-`redacted` text.
3. **Never commit raw data** — only derived bases to git-ignored `data/`. Record dataset id +
   revision + license + accepted-terms date in a provenance file.
4. **Slot-collision guard (new, real-arm-specific)** — real conversations contain literal `{{ }}`
   (code) and all-caps tokens that collide with slot-planting / look canary-shaped. Add an
   `is_plantable`/sanitize check (mirror of the generator's `validate_generated`) so such records are
   skipped or cleaned **before** slotting. Without this the real arm silently corrupts.

**5.3 Long-document arm 🔶** — needs a corpus for the 16k/32k single-turn cells. Recommendation
pending feedback: a permissively-licensed long-form set (e.g. GovReport or arXiv/PubMed
summarization inputs) — clean, long, boring, no PII. Hold the final pick for the feedback.

**5.4 Synthetic arm** — built (§3). Gap: API/Haiku + Transformers/GPU backends need access.

**5.5 Trigger library expansion** — current: `random_canary`, `multi_token`, `boundary`
(`data/triggers/triggers.jsonl`: rand_001, multi_001, boundary_001). Add `natural_phrase` + `unicode`
(`split` stays deferred, multi-turn). Small/mechanical.

**5.6 Grid assembly + provenance** — `expand_manifest` fans out bases × triggers × positions ×
lengths × policies × models (+ counterfactual twins), records `data_source` per base (the H4
covariate), then shards for the cluster. This layer exists; it just needs the bases.

**5.7 Sequencing:**
- **Ready now (facts, not opinions):** (1) fill LMSYS + WildChat parsers; (2) add toxic/PII filter +
  slot-collision guard; (3) expand the trigger library.
- **Feedback-gated 🔶:** (4) long-doc corpus + sampling strategy; (5) pilot scale + any
  contamination/dedup rules; (6) materialize pilot corpus + assemble pilot manifest.

---

## 6. Open inputs still needed from the user (blockers for finalizing)

1. **The 244-line "deep-research feedback"** — referenced in the user's message but delivered to the
   agent only as a placeholder (`[ text #2 +244 lines]`); its contents were **never visible**. Ask
   the user to paste it inline or save it to `docs/DEEP_RESEARCH_FEEDBACK.md`. It is the dominant
   missing input and likely reshapes 5.3/5.7 decisions.
2. **Confirm the §5.2 safety defaults** (drop toxic, strip PII) — assume yes unless told otherwise.
3. **WildChat variant** — the user's snippet uses `allenai/WildChat` (~650k). Confirm vs `WildChat-1M`.
4. **Long-doc corpus choice** (5.3), **pilot scale** (5.7).
5. **Generation API access** (Haiku key/quota) for synthetic quality; **cluster** specifics
   (Klone vs Tillicum, account/partition/storage) and **output format** (JSONL vs Parquet) — all
   still open in `REQUESTED_DOCUMENTATION.md`.

**The user's last decision point (unanswered):** whether to start §5.7 items 1–3 now
(feedback-independent) or hold everything until the feedback is provided. A fresh agent should
re-confirm this before writing code.

---

## 7. Immediate next actions for a fresh agent

1. **Get the deep-research feedback** (§6.1) — do not finalize the data plan without it.
2. On the user's go-ahead, **spec + delegate** (supervisor model, `docs/tasks/NN-*.md`):
   - `10-real-dataset-parsers`: fill `LMSYSParser`/`WildChatParser` + the §5.2 safety/PII/slot-guard,
     with offline tests using tiny synthetic records shaped like the real formats (no gated download
     in the test path). Verify with a small **live** gated pull separately.
   - `11-trigger-library`: add `natural_phrase` + `unicode` `TriggerSpec`s (+ any scoring nuance for
     unicode normalization).
   - Then (feedback-gated): long-doc corpus wiring, pilot materialization, pilot manifest + sharding.
3. Keep the four Project-1 completeness items (§2.1) on the roadmap: **analysis layer** is the most
   important un-built deliverable (it is the scientific output).
4. Always verify delegated work: re-run the full gate (`185 passed` baseline) + a real cross-check.

---

## 8. File map

- `docs/EXPERIMENT_DESIGN.md` — master design; §1 current state, **§1a scope-completeness audit**,
  §2 hypotheses, §3 data plan, §4 stats, §5 cluster, §6 open decisions, §7 roadmap.
- `docs/PRE_REGISTRATION.md` — locked design (models, policy grid, counterfactual, deferrals).
- `docs/DATA_CONTRACTS.md` — survival classes (boundary_corruption vs partial), template_incompatible,
  trim_messages facts.
- `docs/REQUESTED_DOCUMENTATION.md` — open decisions; **item 10 now says generation is implemented**;
  items 14–16 are the dataset-format needs (LMSYS/WildChat now licensed; long-doc pending).
- `docs/tasks/00..09-*.md` — delegation briefs (09 = the generator built this session).
- Code: `src/trigger_audit/` — `generation/` (new), `io/dataset_adapter.py`, `pipelines/`,
  `experiments/survivability_audit/` (`manifest_runner.py`, `runner.py`, `config.py`), `scoring/`,
  `tokenization/`, `prompts/`, `schemas/`.
