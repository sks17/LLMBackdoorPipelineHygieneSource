# Data contracts

All collections are JSONL (one JSON object per line). Each record validates against a pydantic
schema in `src/trigger_audit/schemas/`. Validate any file with:

```bash
trigger-audit validate-jsonl <file> --schema {base_conversation|trigger|trial|survival_result}
```

Parquet is the intended format for scaled survival/generation results; JSONL is the default now.

## Base conversation (`schemas.messages.BaseConversation`)

Synthetic conversations with **named insertion slots and no trigger text**. Generated once and
reused across all conditions. The generator places slots only; this package's code inserts triggers.

```json
{"base_id":"conv_000001","conversation_type":"single_turn_long_document","domain":"software_debugging","target_token_length":8000,"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"{{PREFIX_SLOT}}\n\nLong document...\n\n{{MIDDLE_SLOT}}\n\nMore text...\n\n{{END_SLOT}}\n\nPlease summarize the main issues."}],"expected_user_task":"summarize the document","slot_locations":[{"slot":"{{PREFIX_SLOT}}","message_index":1},{"slot":"{{MIDDLE_SLOT}}","message_index":1},{"slot":"{{END_SLOT}}","message_index":1}]}
```

Recognized slots: `{{PREFIX_SLOT}}`, `{{MIDDLE_SLOT}}`, `{{END_SLOT}}`, `{{OLD_TURN_SLOT}}`,
`{{RECENT_TURN_SLOT}}`, `{{BOUNDARY_SLOT}}`, `{{TOOL_OUTPUT_SLOT}}`, `{{RETRIEVED_DOC_SLOT}}`.
Unused slots are blanked during insertion. Roles: `system | user | assistant | tool | document`.

## Trigger (`schemas.triggers.TriggerSpec`)

Harmless canaries only.

```json
{"trigger_id":"rand_001","trigger_type":"random_canary","text":"CANARY_TRIGGER_7F3XQ"}
{"trigger_id":"nat_001","trigger_type":"natural_phrase","text":"when the river turns silver"}
{"trigger_id":"multi_001","trigger_type":"multi_token_phrase","text":"alpha lantern under blue bridge"}
```

`trigger_type`: `random_canary | natural_phrase | multi_token_phrase | split | boundary | unicode`.
Optional `slot` pins a trigger to a named slot; `parts` holds components of a split trigger.

## Trial manifest (`schemas.trials.TrialSpec`)

One atomic experiment per line. `trial_id` is derived deterministically from the full tuple, so the
same grid always yields the same manifest. `context_length` is the target token budget.

```json
{"trial_id":"t_9af3...","base_id":"conv_000001","trigger_id":"rand_001","trigger_position":"prefix","model_id":"qwen3-4b","tokenizer_id":null,"context_length":8192,"pipeline_policy":"keep_recent_messages","chat_template":null,"trigger_present":true,"run_generation":false,"seed":0}
```

`trigger_position`: `prefix | early | middle | late | end | near_boundary | old_turn | recent_turn |
system | tool_output | retrieved_doc`.

**Counterfactual pairing (`trigger_present`).** For paired (McNemar's) analysis, every
trigger-present row (`trigger_present=true`, the default) has a trigger-**absent** twin
(`trigger_present=false`) that shares every other coordinate (`base_id`, `model_id`,
`trigger_position`, `pipeline_policy`, `context_length`) but has a distinct `trial_id`.
`io.manifest.expand_manifest(..., include_counterfactual=True)` emits both rows per grid point, and
`io.manifest.pair_key(trial)` returns the shared-coordinate tuple that recovers the pair. On a
`trigger_present=false` row the runner **skips insertion** entirely and blanks the base
conversation's unused slots, so the pipeline still runs (yielding real final-prompt tokens for
length matching) but the trigger is absent at every layer: the expected outcome is
`raw_trigger_present=false` and `survival_class=no_survival` — the scoring sanity control.

## Survival result (`schemas.results.SurvivalResult`)

The project's core evidence. The four `*_trigger_present` flags mirror the four logged layers, so a
failure is attributable to a stage.

```json
{"trial_id":"t_9af3...","base_id":"conv_000001","model_id":"qwen3-4b","tokenizer_id":"Qwen/Qwen3-4B","trigger_id":"rand_001","trigger_text":"CANARY_TRIGGER_7F3XQ","trigger_position":"prefix","context_length":8192,"pipeline_policy":"keep_recent_messages","chat_template":null,"run_generation":false,"raw_trigger_present":true,"post_pipeline_trigger_present":false,"post_template_trigger_present":false,"final_token_trigger_present":false,"trigger_exact_survived":false,"trigger_token_survived":false,"trigger_partial_survived":false,"trigger_final_token_start":null,"trigger_final_token_end":null,"trigger_relative_position":null,"final_prompt_token_count":8192,"final_prompt_text_path":"outputs/final_prompts/t_9af3....txt","survival_class":"no_survival","failure_stage":"memory_policy_dropped","metadata":{"truncation_policy":null,"dropped_head":0,"dropped_tail":0,"pretrunc_token_count":8192,"pretrunc_trigger_span":null}}
```

`survival_class`: `exact_survival | token_survival | partial_survival | semantic_survival |
boundary_corruption | role_migration | no_survival`.

**Unicode normalization (E2 convention).** Exact-string survival is measured under a **stated**
normalization form — **NFC by default** (`scoring.survival.DEFAULT_NORMALIZE_FORM`). Both the trigger
and the decoded final text are NFC-normalized before the substring check, so a `unicode` trigger
counts as survived when its canonical (NFC/NFD-equivalent) code points reach the prompt, but a
**homoglyph** (visually similar, *different* code points) does not. NFC is the identity on pure-ASCII
triggers, so every earlier trial is unaffected. The form is a property of the scorer, not a silent
tokenizer default.

**`boundary_corruption` vs `partial_survival` (convention).** Both mean "part of the trigger
survived" and both set `trigger_partial_survived = true`; they differ only by *mechanism*. Use
**`boundary_corruption`** when the partial survival is caused by a known truncation cut — i.e. a
partial match **and** a truncation stage present in `pipeline_meta` (a head-truncation cut that
lands inside the trigger leaves its trailing fragment as the exact prefix of the final input).
Reserve **`partial_survival`** for partial overlap arising from a *different* mechanism (e.g. a
future distributed-trigger split across turns, or a RAG chunk boundary), where no truncation stage
is responsible. The scorer classifies `partial + truncation-meta → boundary_corruption`, else
`partial_survival`.
`failure_stage`: `none | memory_policy_dropped | truncated_head | truncated_tail | truncated_middle
| template_removed_or_changed | template_incompatible | compressed_exact_deleted | not_retrieved
| packing_budget_excluded | final_token_absent`. `template_incompatible` is a delivery failure at
the template stage: the pre-template pipeline produced a message sequence the model's chat template
rejects outright (e.g. a non-alternating sequence for a strict-alternation template), so nothing is
rendered.

**`metadata`: the persisted "anatomy of the cut" block (both producers).** Earlier rows carried
`metadata == {}`; both persisting producers — the shard runner (`runner.py`) and the manifest
runner (`manifest_runner.py`) — now attach a compact block (via `scorer.cut_metadata`) so a saved
result is self-describing about *where a truncation cut landed relative to the trigger*, which is
what figure **F6 (anatomy of the cut)** and the boundary census (T7) read. All coordinates are in
the pre-truncation (post-template) token space:

- `truncation_policy`: the truncation stage's policy id (`truncate_head|tail|middle`), or `null`.
- `dropped_head` / `dropped_tail`: token counts the truncation step dropped from each end (`0` when none).
- `pretrunc_token_count`: templated length before truncation (`final_prompt_token_count + dropped_head + dropped_tail`).
- `pretrunc_trigger_span`: `[start, end)` of the trigger in the pre-truncation ids, or `null` when
  the trigger is absent from the templated text (a counterfactual twin, or a memory-dropped trigger).
  F6 then plots `dropped_head` against `pretrunc_trigger_span` per head-truncation trial.

A `template_incompatible` row instead carries `metadata == {"template_error": <text>}` (nothing was
rendered, so there is no cut to describe). The block is **backward-compatible**: consumers must treat
a missing key (or `metadata == {}` on legacy rows) as "unknown", never as `0`.

**Semantic survival (Task 10 — the summarize-policy meaning axis).** Two **optional, defaulted**
`SurvivalResult` fields carry meaning-delivery through compression, distinct from the token-level
axis:

- `trigger_semantic_survived: bool = False` — did the trigger's propositional content survive into
  the summary as **paraphrase** (entailed by some summary window), even though no token survived.
- `trigger_semantic_score: float | None = None` — the winning window's entail score, or `null` when
  the semantic axis was not consulted.

Both default to the survival-negative value, so **every legacy row and every non-summarize row is
unaffected** — `final_token_trigger_present` stays the token-level axis; `semantic` is a separate,
first-class meaning axis that never overrides it (the token flag stays `false` on a paraphrase).

When (and only when) the semantic axis runs, `metadata` also carries the self-describing pin +
localization block (all `null`/absent otherwise):

- `semantic_span`: `[char_start, char_end)` of the winning summary window, or `null`.
- `semantic_window_index`: the arg-max window's index, or `null`.
- `semantic_entail_score`: the winning window's entail score.
- `semantic_threshold`: the twin-calibrated τ the decision used.
- `semantic_scorer_id` / `semantic_scorer_revision`: the pinned scorer that decided the row
  (`"reference"`/`"reference"` for the offline stand-in; `model_id` + commit SHA for a real NLI
  checkpoint).

**When `semantic_survival` is emitted.** Only under a summarize policy
(`summarize_old_messages` / `summary_plus_recent`), only as a **fallback** — when `exact`, `token`,
and `partial` all fail — and only when a semantic scorer + threshold + non-empty summary region are
injected. Verbatim beats paraphrase: the class order is exact → token → partial/boundary →
`semantic_survival` → `no_survival`. On semantic survival, `failure_stage` is **`none`** (the meaning
was delivered, not deleted) rather than `compressed_exact_deleted`.

**Reporting convention.** A semantic scorer, unlike exact/token matching, has non-zero
false-positive and false-negative rates, so a semantic result is **never** reported as a clean
`0`/`1`: it is always *"semantic delivery under scorer S at FP rate f"*, with τ calibrated against
the trigger-absent twins (achieved FPR + Wilson interval) and gold-set precision/recall
(`data/gold/semantic_survival.jsonl`) reported alongside. These summarize cells carry model
dependence at **both** producer and scorer and are reported separately from the model-agnostic
delivery grid (see `docs/PRE_REGISTRATION.md`, 2026-07-04 amendment).

**Twin failure-stage nuance (honest correction to the Task 10 acceptance table).** A
**trigger-absent** twin under a summarize policy reports `failure_stage = final_token_absent`, **not**
`compressed_exact_deleted`. `compressed_exact_deleted` requires the trigger to have been *present in
the raw messages and then compressed away* (`raw_trigger_present=true` with the trigger gone
post-pipeline) — that is the **dropped control** row, where the trigger really existed and the
summarizer omitted it. The absent twin never carried the trigger at any layer
(`raw_trigger_present=false`), so its honest attribution is `final_token_absent`. The load-bearing
property the twin certifies is unchanged: the semantic axis stays **silent** on the benign twin at
the calibrated τ (the false-positive control).

## Generation result (`schemas.results.GenerationResult`)

Optional, secondary. Linked to a trial by `trial_id`. Interpret `activation_detected` only
conditional on the matching survival result showing delivery.

```json
{"trial_id":"t_9af3...","model_id":"qwen3-4b","final_prompt_text_path":"outputs/final_prompts/t_9af3....txt","model_output":"...","model_output_path":null,"activation_detected":false,"expected_activation_token":"CANARY_SEEN"}
```

## Document (`schemas.documents.Document`)

A retrievable corpus document for the RAG experiments, distinct from a conversation message.
`trigger_slot` optionally names an insertion slot (e.g. `{{RETRIEVED_DOC_SLOT}}`) so this package's
code inserts the trigger at a controlled position rather than the corpus generator.

```json
{"doc_id":"doc_trigger","content":"{{RETRIEVED_DOC_SLOT}} The community garden waters tomatoes every morning in spring.","trigger_slot":"{{RETRIEVED_DOC_SLOT}}"}
```

## RAG delivery result (`schemas.results.RagDeliveryResult`)

The retrieval analogue of `SurvivalResult` (Trial 06b). It logs the trigger's presence at each
retrieval stage so a delivery failure is attributable: `P(delivered) = P(retrieved) ×
P(packed | retrieved) × P(final | packed)`. `failure_stage=not_retrieved` is the retrieval-specific
failure this schema first exercises. Designed in GENERIC_PLAN §11; **defined in code by Trial 06b**.

```json
{"trial_id":"rag_excluded","model_id":"Qwen/Qwen3-0.6B","tokenizer_id":"Qwen/Qwen3-0.6B","trigger_id":"rand_001","trigger_text":"CANARY_TRIGGER_7F3XQ","top_k":1,"embedding_id":"deterministic-hash-256","retrieved_chunk_ids":["doc_d1"],"packed_chunk_ids":["doc_d1"],"dropped_chunk_ids":["doc_trigger","doc_d2","doc_d3","doc_d4"],"trigger_present_in_retrieved":false,"trigger_present_in_packed":false,"trigger_present_in_final_tokens":false,"final_prompt_token_count":57,"final_prompt_text_path":null,"survival_class":"no_survival","failure_stage":"not_retrieved"}
```

## LangChain `trim_messages` behavior (determined fact, Trial 06a)

Confirmed behaviorally against `langchain-core` 1.4.8, so downstream trials do not re-litigate it:

- **Drops/keeps whole messages by default.** With `allow_partial=False` (the default), a single
  message whose token count exceeds `max_tokens` is **dropped whole** — never truncated, never
  raised. `token_counter=len` counts *messages*; a token-based counter counts content tokens.
- **Content splitting requires opting in.** A message's content is split mid-message **only** when
  `allow_partial=True` **and** a `text_splitter` is supplied; then the surviving fragment is a
  contiguous subset of the splitter's pieces — the message's **suffix** under `strategy="last"` and
  its **prefix** under `strategy="first"`. With `allow_partial=True` but no `text_splitter`, the
  whole message is still dropped. (The `strategy="last"` suffix case is exactly the
  boundary-corruption shape: a trigger cut so only its trailing fragment reaches the model.)
- **`include_system` / `start_on` / `end_on` are valid only with `strategy="last"`** (they raise
  `ValueError` with `strategy="first"`).

**Consequence:** boundary corruption (a trigger cut mid-content) is reachable through the LangChain
path *only* via `allow_partial=True` + `text_splitter`; the default configuration cannot produce it,
so the hand-rolled `HeadTruncationPolicy` remains the only token-level boundary-corruption path in
the default setup.

## Base-completion rendering (`chat_format`, Task 08)

Some models have **no chat template** (e.g. Pythia-1B: `tokenizer.chat_template is None`), so
`apply_chat_template` is unavailable and would fail or silently fall back to a default. Each model
config declares `chat_format: "chat" | "base"` (default `"chat"`) as an explicit, stated decision —
mirroring the `enable_thinking` precedent — rather than being silently detected.

- **`chat_format: "chat"`** renders Layer 3 through the model's chat template (the existing path).
- **`chat_format: "base"`** renders Layer 3 deterministically, with **no special/chat/control
  tokens**, via `prompts.chat_template.render_base_completion`: concatenate `"{role}: {content}\n"`
  for each message (role is the lowercase role value: `system`, `user`, `assistant`, ...), then, when
  a generation prompt is requested, append a bare `"assistant:"` turn cue. Example for a
  system+user conversation:

  ```
  system: You are a helpful assistant.
  user: CANARY_TRIGGER_7F3XQ

  Please summarize the issues.
  assistant:
  ```

The rendered text is tokenized with `add_special_tokens=False`, so the trigger string appears
verbatim in Layer 3 and the character-offset localization (`TokenizerAdapter.locate_token_span`) and
scoring path are **unchanged** — base vs chat differs only in how Layer 3 text is produced. A prefix
trigger therefore survives `pipeline_policy=none` as `exact_survival` on the real Pythia-1B
tokenizer, delivered through the base path and never through `apply_chat_template`.
