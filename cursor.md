# Cursor agent delegation log

This file is the code leader's scoping and decision log for delegating coding work to the pool of
cursor agents (up to 10, run in parallel). It is a living document: each unit of delegated work is
a **task** labeled `T n.m`, where `n` is the output number (one leader turn that dispatches work)
and `m` is the agent number (`0`–`10`) within that output. Agents in the same output run in
parallel, so their tasks must be **disjoint** — no shared files, no ordering dependencies.

## When work is delegated vs kept in-house

Per the standing rule: delegate to cursor agents when **a lot of code that is not particularly
sensitive** must be written and the packages are already installed; otherwise the leader writes it.
Each entry below records which path was taken and why.

---

## Output 0 — `docs/tasks/00-trial-zero.md` (Trial Zero connective tissue)

**Decision: kept in-house (no agents dispatched).**

Rationale:

- **Small.** The whole task is three tiny modules (`prompts/trigger_insertion.py`,
  `experiments/survivability_audit/trial_zero.py`, one shared helper factored into
  `pipelines/trigger_insertion.py`), two test files, one data row, and two `__init__` exports —
  well under the "a lot of code" bar.
- **Sensitive.** Trial Zero is the project's foundational positive control ("if a trigger cannot
  survive here, nothing downstream is trustworthy"). It is bound by hard acceptance contracts:
  `insert_trigger(base_messages(), TRIGGER.text, "prefix")` must *equal*
  `expected_positive_messages()`, and `run_trial_zero` with the real Qwen3-0.6B tokenizer must
  reproduce the golden fixture (`exact_survival` at token span `(14, 23)`, `no_survival` negative).
  It wires directly into frozen, off-limits APIs (`score_from_layers`, the tokenizer adapters, the
  spec). Parallelizing ~150 lines of tightly-coupled, correctness-critical code across agents would
  add coordination risk without saving meaningful effort.

Delivered by the leader:

| File | Change |
|------|--------|
| `src/trigger_audit/pipelines/trigger_insertion.py` | Factored placement string logic into module-level `place_in_content()`; `TriggerInserter._place` now delegates to it (shared source of truth, no duplication). |
| `src/trigger_audit/prompts/trigger_insertion.py` (new) | Pure `insert_trigger(messages, trigger_text, position="prefix")`; reuses `place_in_content`; only `prefix` supported. |
| `src/trigger_audit/prompts/__init__.py` | Export `insert_trigger`. |
| `data/synthetic/trial_zero_base.jsonl` (new) | Trigger-free `BaseConversation` row (`base_id: trial_zero_base`) matching `base_messages()`; validates against the `base_conversation` schema. |
| `src/trigger_audit/experiments/survivability_audit/trial_zero.py` (new) | `run_trial_zero(*, tokenizer_adapter, insert=True)` composing spec + inserter + renderer + `score_from_layers`; adapter injected for offline/production parity. |
| `src/trigger_audit/experiments/survivability_audit/__init__.py` | Export `run_trial_zero`. |
| `tests/test_insert_trigger.py` (new) | Prefix placement, no-mutation, hard-acceptance equality, unsupported-position error. |
| `tests/test_trial_zero_driver.py` (new) | Offline positive→`EXACT_SURVIVAL`, negative→`NO_SURVIVAL`, base-data-matches-spec. |

Verification (all green):

- `pytest` — full suite passes, including the pre-existing `tests/test_trial_zero.py` unchanged and
  its live Qwen3 tripwire.
- `ruff check src tests`, `ruff format --check src tests`, `mypy` — clean.
- `trigger-audit validate-jsonl data/synthetic/trial_zero_base.jsonl --schema base_conversation` — ok.
- End-to-end with the **real** `Qwen/Qwen3-0.6B` `HFTokenizerAdapter`: positive →
  `exact_survival`, token span `(14, 23)`, 52 tokens; negative → `no_survival`. Matches the golden
  fixture.

---

## Output 1 — `docs/tasks/01-trial-one.md` (Trial One connective tissue)

**Decision: kept in-house (no agents dispatched).** Same reasoning as Output 0: small surface
(one inserter extension, one driver, tests) and highly sensitive — a controlled scientific variant
(head truncation, `trigger_position` the only manipulated variable) with hard acceptance contracts
(`insert_trigger(..., "end") == expected_end_messages()`; real-Qwen3 golden-fixture reproduction)
bound to frozen APIs (`HeadTruncation`, `score_from_layers`'s `final_text`, `trial_one_spec`).

Delivered by the leader:

| File | Change |
|------|--------|
| `src/trigger_audit/prompts/trigger_insertion.py` | Generalized `insert_trigger` to a `{prefix, end}` position map delegating to `place_in_content`; unsupported positions still raise `NotImplementedError`. |
| `src/trigger_audit/experiments/survivability_audit/trial_one.py` (new) | `run_trial_one(*, tokenizer_adapter, trigger_position, context_length_target)` — insert → render full Layer 3 → `HeadTruncation` → score with `final_text=decode(kept_ids)` and `truncate_head` meta. |
| `src/trigger_audit/experiments/survivability_audit/__init__.py` | Export `run_trial_one`. |
| `tests/test_insert_trigger.py` | Added `"end"` placement + hard-acceptance + no-mutation cases; retargeted the unsupported-position test from `"end"` (now supported) to `"middle"`. |
| `tests/test_trial_one_driver.py` (new) | Offline prefix→`NO_SURVIVAL`/`TRUNCATED_HEAD` and end→`EXACT_SURVIVAL`, both cut to the derived budget. |

Verification (all green):

- `pytest` — 84 passed, including the pre-existing `tests/test_trial_one.py` unchanged.
- `ruff check src tests`, `ruff format --check src tests`, `mypy` — clean.
- End-to-end with the **real** `Qwen/Qwen3-0.6B` `HFTokenizerAdapter`: derived
  `context_length_target = 26`; prefix → `no_survival` (`truncated_head`), end → `exact_survival`,
  both truncated to exactly 26 tokens. Matches the golden fixture.

---

## Output 2 — `docs/tasks/02-trial-two.md` (Trial Two: message-level memory policy)

**Decision: kept in-house (no agents dispatched).** Same reasoning as Outputs 0–1: small surface
(one memory policy, one factored helper, one driver, one spec/data row, tests) and highly
sensitive — the project's first message/turn-level trial, whose whole point is to *pin a
structural invariant*: message-granularity policies can never produce partial survival
(`partial_survived is False`). Getting the shared first-vs-last targeting and the count-based
policy exactly right is correctness-critical and bound to frozen APIs (`MemoryOutcome`,
`score_from_layers`, `place_in_content`).

Delivered by the leader:

| File | Change |
|------|--------|
| `src/trigger_audit/pipelines/memory_policy.py` | New `KeepLastNMessages(*, keep_last_n)` registered `keep_last_n_messages` — count-based, keeps all system + last N non-system as whole messages; ignores budget/counter; never mutates content. The budget-based `KeepRecentMessages` is untouched. |
| `src/trigger_audit/pipelines/trigger_insertion.py` | Factored `target_user_index(messages, position)` (first-vs-last user selection); `TriggerInserter._target_index` now delegates to it. |
| `src/trigger_audit/prompts/trigger_insertion.py` | `insert_trigger` now uses `target_user_index` and supports `old_turn`/`recent_turn` alongside `prefix`/`end`. |
| `src/trigger_audit/pipelines/__init__.py` | Export `KeepLastNMessages`. |
| `src/trigger_audit/experiments/survivability_audit/trial_two_spec.py` (new) | Frozen spec: reuses Trial Zero constants, `KEEP_LAST_N = 2`, 6-message multi-turn base, `trial_spec` (`trial_two_a`/`trial_two_b`). |
| `data/synthetic/trial_two_base.jsonl` (new) | Trigger-free multi-turn `BaseConversation` (`trial_two_base`) matching `base_messages()`; validates against the schema. |
| `src/trigger_audit/experiments/survivability_audit/trial_two.py` (new) | `run_trial_two(*, tokenizer_adapter, trigger_position)` — insert → `KeepLastNMessages` (Layer 2) → render/tokenize → score with `memory_policy` meta and message-derived `raw`/`post_pipeline` presence. |
| `src/trigger_audit/experiments/survivability_audit/__init__.py` | Export `run_trial_two`. |
| `tests/test_memory_policy.py` | `keep_last_n=2` on the base keeps `[0, 4, 5]` / drops `[1, 2, 3]`, length 3, system kept, no mutation; registry resolves the new name. |
| `tests/test_insert_trigger.py` | `old_turn` targets first user, `recent_turn` targets last; no mutation. |
| `tests/test_trial_two_driver.py` (new) | Offline old-turn → `NO_SURVIVAL`/`MEMORY_POLICY_DROPPED`, recent-turn → `EXACT_SURVIVAL`; `partial_survived is False` for both. |

Verification (all green):

- `pytest` — 91 passed, including Trial Zero/One tests unchanged.
- `ruff check src tests`, `ruff format --check src tests`, `mypy` — clean.
- `trigger-audit validate-jsonl data/synthetic/trial_two_base.jsonl --schema base_conversation` — ok.
- Real-`Qwen/Qwen3-0.6B` cross-check reproduces the expected table exactly: `trial_two_a`
  (`OLD_TURN`) → raw present / post-pipeline absent / `no_survival` / `memory_policy_dropped`;
  `trial_two_b` (`RECENT_TURN`) → raw & post-pipeline present / `exact_survival`; `partial_survived`
  is `False` for both (the pinned invariant).

---

## Output 3 — `docs/tasks/03-trial-three.md` (composed memory + truncation, staged interface)

**Decision: kept in-house (no agents dispatched).** This is the most architecturally load-bearing
trial so far — it introduces a *new shared abstraction* (`ComposedPipeline` with stage-ordered
policies) plus several hard invariants (reversal invariance, budget-independence of the
memory-dropped condition, the first present→absent Layer 2 vs Layer 4 transition, no partial
survival). Exactly the sensitive/foundational profile that the delegation rule reserves for the
leader.

Delivered by the leader:

| File | Change |
|------|--------|
| `src/trigger_audit/pipelines/composition.py` (new) | `Stage` enum, `CompositionContext`, `StagedPolicy` ABC, `KeepRecentMessagesPolicy` (pre, wraps `KeepLastNMessages`), `HeadTruncationPolicy` (post, wraps `HeadTruncation`; raises if `token_ids is None`), `CompositionResult`, `ComposedPipeline.run` (applies policies in stage order → reversal-invariant). Delegates all policy math; sets `memory_policy`/`truncation` metadata. |
| `src/trigger_audit/pipelines/trigger_insertion.py` | Decoupled *which message* from *where in it*: added `_PREFIX_PLACEMENT` so `RECENT_TURN` targets the **last** user message (via `target_user_index`, unchanged) but is placed at that message's **prefix** — letting a tight tail-keeping head truncation drop the trigger while keeping the question. |
| `src/trigger_audit/pipelines/__init__.py` | Export the composition public names. |
| `src/trigger_audit/experiments/survivability_audit/trial_three_spec.py` (new) | Reuses Trial Two base + Trial Zero constants; `KEEP_LAST_N=2`, `GENEROUS_BUDGET`, `derive_tight_budget(result_b) = T − E` (measured, never hardcoded), `trial_spec(trial_id, position, budget)`. |
| `src/trigger_audit/experiments/survivability_audit/trial_three.py` (new) | `run_trial_three(*, tokenizer_adapter, trigger_position, context_length_target, reverse_chain=False)` — insert → `ComposedPipeline([HeadTruncationPolicy, KeepRecentMessagesPolicy])` (declared post-then-pre, optionally reversed) → score with `post_template_text` (Layer 3), `final_text=decode(final_token_ids)` (Layer 4), and `result.metadata`. |
| `src/trigger_audit/experiments/survivability_audit/__init__.py` | Export `run_trial_three`. |
| `tests/test_composition.py` (new) | Pre-before-post; reversing the policy list yields an identical `CompositionResult`; a `POST_TEMPLATE` policy with `token_ids is None` raises; stages declared correctly. |
| `tests/test_trial_three_driver.py` (new) | a/b/c table; reversal invariance across all three; (a) budget-independent; (c) present→absent; `partial_survived is False` throughout. |

**Naming flag (for Saki):** the staged wrapper is named `KeepRecentMessagesPolicy` (per the task
spec) and sits close to the existing budget-based `KeepRecentMessages`, though it wraps the
count-based `KeepLastNMessages`. Kept the spec's name; a later rename (e.g. `KeepLastNStage`) may be
worth it if the two are ever confused.

Verification (all green):

- `pytest` — 100 passed, including Trials Zero/One/Two unchanged (the `recent_turn` placement change
  is placement-agnostic for Trial Two and membership-only for `test_insert_trigger`).
- `ruff check src tests`, `ruff format --check src tests`, `mypy` — clean.
- Real-`Qwen/Qwen3-0.6B` cross-check: derived tight budget `T − E = 60 − 40 = 20`; `trial_three_a`
  → post-pipeline absent / `no_survival` / `memory_policy_dropped`; `trial_three_b` → post-pipeline
  & final present / `exact_survival`; `trial_three_c` → post-pipeline present but final **absent** /
  `truncated_head`. Reversal-invariant for all three; (a) budget-independent; (c) is the unique
  present→absent transition; `partial_survived` is `False` throughout.

---

## Output 4a — `docs/tasks/04a-manifest-expansion.md` (manifest-driven grid over verified primitives)

**Decision: PARTIALLY DELEGATED — first use of parallel cursor agents.** This task is the most
mechanical so far (glue that reproduces already-verified Trial 2–3 outcomes) and contained two
genuinely independent, non-sensitive pieces touching disjoint files. Those were delegated in
parallel; the sensitive integration core (the canonical runner, CLI, and the 8-row acceptance test)
and all judgment calls were kept in-house.

### Delegated (2 parallel cursor agents, disjoint files)

- **T4a.1** — `pipelines/policy_registry.py` (new `resolve_policy`) + replaced
  `configs/pipeline_policies.example.yaml` with the composite format. Config-driven, delegates all
  policy construction to the frozen staged policies; unknown id/type raises. Delivered clean, ruff
  passing.
- **T4a.2** — `io/manifest.py` (new `expand_manifest`, the Cartesian product) + `make_grid_trial_id`
  appended to `util/ids.py` (sha256, first-12). Delivered clean, ruff passing.

Judgment calls I resolved *in the scopes* so the agents had zero ambiguity: grid `TrialSpec`
uses `context_length=0` (budget lives in the policy id, not this field) and `tokenizer_id=model_id`;
the id function takes plain strings and the caller passes `trigger_position.value` (avoids the
str-Enum `__format__` gotcha).

### In-house (sensitive integration + verification)

| File | Change |
|------|--------|
| `data/base_conversations/base_conversations_000.jsonl` (new) | Slot-form base (`conv_000001`), slots at turn starts, `slot_locations`; validates. |
| `experiments/survivability_audit/manifest_runner.py` (new) | `run_trial(trial, *, base, trigger, tokenizer_adapter, policies_config_path=None)` — `TriggerInserter` → `resolve_policy` → `ComposedPipeline` → `score_from_layers`. `run_trial_three` generalized. (Added the optional `policies_config_path` so tokenizer-specific budgets are testable.) |
| `cli.py` | Rewrote `build-manifest` to expand via `expand_manifest` and write with `write_jsonl` (dropped the superseded `ManifestBuilder`/shard path). Smoke-tested: writes 8 rows. |
| four `__init__.py` | Export `resolve_policy`, `expand_manifest`, `make_grid_trial_id`, `run_trial`. |
| `tests/test_manifest_expansion.py` (new) | Grid cardinality (8), id stability across re-expansion + uniqueness; the 8-row table offline (with a per-adapter–derived tight budget); `partial_survived is False` on every row; row 8 ≡ row 5 trigger outcome. |

### FLAG for Saki (blocking correctness — I changed a spec value)

The task hardcodes the tight-truncation budget as `20` ("derived: Trial 3C, Qwen3-0.6B"). That value
is for Trial 3C's `insert_trigger` placement, which separates the trigger with `\n\n`. **The manifest
uses the slot-aware `TriggerInserter`, whose slot fill separates with a single space — one fewer
token.** Empirically on Qwen3-0.6B for this slot base: recent_turn gives `T=59, E=40`, so the correct
tight budget is `T − E = 19`, not 20. At 20, the last trigger token survives → `boundary_corruption` /
`partial_survived=True`, violating the invariant and breaking row 7. **I set the config budget to 19**
(with a provenance comment) and verified row 7 → `no_survival`, `partial=False` on the real tokenizer.
Consistent with the task's own practice of flagging derived-budget reinterpretations.

### FLAG for Saki (non-blocking): tokenizer-specific budget + stale validate-config

- The tight budget is inherently tokenizer-specific. Offline (`SimpleWhitespaceTokenizerAdapter`) the
  prompt is far shorter, so `19` is non-binding; the acceptance test re-derives the offline budget
  (`T − E`, ≈10) and runs all 8 rows against a temp config. The checked-in `19` is what the CLI and
  the real-tokenizer cross-check use. A future cleanup could make the budget derived rather than
  hardcoded per (tokenizer, base).
- Replacing `pipeline_policies.example.yaml` with the composite format leaves
  `trigger-audit validate-config <that file> --kind pipeline_policies` (old `PipelinePolicyConfig`
  format) stale — it will now error on that file. No test exercises it; it is part of the pre-composition
  scaffolding the task marks as superseded.

### Verification (all green)

- `pytest` — 102 passed, Trials 0–3 unchanged.
- `ruff check src tests`, `ruff format --check src tests`, `mypy` — clean.
- `validate-jsonl data/base_conversations/base_conversations_000.jsonl --schema base_conversation` — ok.
- **Real `Qwen/Qwen3-0.6B` + checked-in config (budget 19): the full 8-row table reproduced** —
  rows 1/2 `exact_survival` (controls), 3 `no_survival`/2A, 4 `exact_survival`/2B, 5 `no_survival`/3A,
  6 `exact_survival`/3B, 7 `no_survival`/3C, 8 `no_survival` (budget-independent). Trial ids stable
  across two expansions and unique; row 8 ≡ row 5 trigger outcome; `partial_survived` False throughout.
- `build-manifest` CLI smoke-tested end to end: expands the grid and writes 8 manifest rows.

---

## Output 4b — `docs/tasks/04b-second-model.md` (second model: tokenizer/template diversity)

**Decision: kept in-house (no agents dispatched).** Unlike 04a, this task's core is a
*foundational, correctness-critical change to a shared scoring primitive* (character-offset trigger
localization threaded through `score_from_layers`) that must keep Trials 0–3 and the 04a table green,
followed by a strict dependency chain (primitive → derive K → config → fixtures → grid → test). That
is the sensitive, sequential profile the delegation rule reserves for the leader, and there was no
cleanly-parallelizable non-sensitive slice worth isolating.

### The tokenization confound (why this is more than "add a model")

TinyLlama's BPE re-tokenizes `CANARY_TRIGGER_7F3XQ` at the context boundary, so
`find_subsequence(final_ids, encode(trigger))` returns `None` even though the string is plainly
present — exactly the confound `PROJECT_DESCRIPTION` §12 warns about. Verified live: on a TinyLlama
render, subsequence = `None` but the char-offset span recovers `CANARY_TRIGGER_7F3XQ`.

Delivered by the leader:

| File | Change |
|------|--------|
| `tokenization/tokenizer_adapter.py` | New `locate_token_span(text, subtext)`: ABC default = token-id subsequence; `HFTokenizerAdapter` override uses `return_offsets_mapping` (half-open span of tokens overlapping the trigger's char span; falls back to subsequence for non-fast tokenizers). The simple adapter inherits the subsequence fallback. |
| `scoring/survival.py` | `TokenSurvivalScorer.assess` gains an optional `trigger_token_span` (Enum sentinel default `USE_SUBSEQUENCE`): when supplied, token metrics come from the offset span; otherwise the legacy subsequence path (Trials 0–3 unchanged). |
| `experiments/survivability_audit/scorer.py` | `score_from_layers` passes `trigger_token_span` through. Backward-compatible default. |
| `experiments/survivability_audit/manifest_runner.py` | Computes `adapter.locate_token_span(final_text, trigger.text)` and passes it — no per-model branch. |
| `configs/models.example.yaml` | Added `tinyllama-1_1b-chat` (`enable_thinking: false`). |
| `configs/pipeline_policies.example.yaml` | Added `…_generous_tinyllama` (non-binding) and `…_tight_tinyllama` (K=19, provenance comment). |
| `scripts/capture_tinyllama_fixture.py` + `tests/fixtures/tinyllama_rows/` | Golden fixtures (post-template text + ids) for rows 1–2, recording that subsequence misses the trigger while the offset span recovers it. |
| `tests/test_second_model.py` | Grid = 16 rows (two `expand_manifest` concatenated, per-model tight ids); confound documented offline; live template tripwire; rows 3–8 `survival_class` model-invariant; TinyLlama rows 1–2 `exact_survival` with `final_token_trigger_present=True`. |

### FLAGS for Saki

- **Offset localization folded into 4b (as the task suggested).** It is a real addition beyond "add a
  model," but foundational and backward-compatible (Qwen3's offset span == its subsequence span, so
  Trials 0–3 and the 04a table are unchanged). Flag if you'd rather it were its own trial.
- **`K_tinyllama = 19` is independently derived, not reused from Qwen3.** From a real TinyLlama
  recent_turn run (offset-based): T=75, E=56 → T−E=19. It numerically equals Qwen3's 19 only because
  the trigger's tail happens to be 19 tokens for both; the provenance comment records the derivation.
- **Partial-survival under offset localization is intentionally not modeled** (the offset-absent path
  reports `partial=False`): the derived tight budgets drop the *whole* trigger span, so no partial run
  arises. If a future trial deliberately cuts mid-trigger on a real tokenizer, boundary-corruption
  detection would need extending here (documented in a code comment).
- **Gemma deferred to 4c** per the pre-flight finding (its strict-alternation / no-system-role
  template cannot render the `keep_last_n=2` post-memory shape). Nothing actioned for it here.

### Verification (all green)

- `pytest` — 106 passed; Trials 0–3 and the 04a rows unchanged.
- `ruff check src tests`, `ruff format --check src tests`, `mypy` — clean.
- Regression: Qwen3 offset span == subsequence span; the real-Qwen3 04a 8-row table reproduces
  unchanged after wiring the offset path.
- **Real tokenizers, 16-row grid:** rows 3–8 `survival_class` identical across Qwen3 and TinyLlama
  (model-invariant); TinyLlama rows 1–2 `exact_survival` with `final_token_trigger_present=True`
  (recovered by the offset span, which subsequence misses); `partial_survived` False on every row.

---

## Output 4c — `docs/tasks/04c-gemma-template.md` (Gemma: unrenderable memory output)

**Decision: kept in-house (no agents dispatched).** The core is a sensitive correctness change to
shared plumbing — a new delivery-failure mode (`TEMPLATE_INCOMPATIBLE`), a typed render error, and a
graceful-capture path in the runner that must not crash and must attribute presence to the
pre-template layers — plus a strict dependency chain (error type → runner capture → config →
fixtures → grid → test). Same sensitive/sequential profile the delegation rule reserves for the
leader.

### The finding

A template-agnostic memory policy can produce a sequence a model's chat template rejects outright.
Gemma-3 has no system role (system merges into the first user turn) and strict user/assistant
alternation, so `keep_last_n=2`'s post-memory `[system, assistant, user]` raises
`jinja2 TemplateError: roles must alternate`. Delivery fails at the template stage — no prompt, no
trigger — a mode neither Qwen3 nor TinyLlama (lenient templates) can produce.

Delivered by the leader:

| File | Change |
|------|--------|
| `schemas/results.py` | New `FailureStage.TEMPLATE_INCOMPATIBLE`. |
| `prompts/chat_template.py` | `TemplateRenderError` (carries the offending `messages`); `render()` catches jinja2 `TemplateError` (matched by class name — no jinja2 import) and re-raises it, letting unrelated exceptions propagate. |
| `experiments/survivability_audit/scorer.py` | `template_incompatible_result(...)` builds the `NO_SURVIVAL` / `TEMPLATE_INCOMPATIBLE` row (`final_prompt_token_count=0`, error text in metadata). |
| `experiments/survivability_audit/manifest_runner.py` | Wraps `ComposedPipeline.run`; on `TemplateRenderError`, records the outcome with `raw`/`post_pipeline` presence computed from the carried post-memory messages. No per-model branch. |
| `prompts/__init__.py` | Export `TemplateRenderError`. |
| `configs/models.example.yaml` | Added `gemma-3-1b-it`. |
| `configs/pipeline_policies.example.yaml` | Added `…_generous_gemma` / `…_tight_gemma` (budgets **inert** — render fails before truncation; kept for grid symmetry, documented). |
| `scripts/capture_gemma_fixture.py` + `tests/fixtures/gemma_rows/` | Rows 1–2 golden fixtures + system-merge evidence. |
| `tests/test_gemma_template.py` | Offline: render wraps `TemplateError` (and carries messages) but not unrelated errors; `run_trial` records `TEMPLATE_INCOMPATIBLE` via a fake rejecting adapter (presence flags correct); 24-row grid; fixture documents system-merge. Live: template tripwire; Gemma rows 1–2 `exact_survival` (model-invariant with Qwen3), rows 3–8 `TEMPLATE_INCOMPATIBLE`, no crash. |
| `RUNNING_EXPERIMENTS.md` | Added Trial 4c entry (the delivery-failure mode) and back-filled the missing Trial 4b entry, newest-first. |

### FLAGS for Saki

- **Gemma tight/generous budgets are inert.** Gemma's keep-recent rows fail at the template stage
  before truncation runs, so those budgets are non-functional placeholders (both set to 40960),
  kept only so Gemma contributes the same 8 (position × policy) rows as the other models. Documented
  in the config comment.
- **Two deferred follow-ups (noted, not built):** (1) *role migration* — a system-position trigger
  migrating into the user turn under Gemma's merge (the `role_migration` class our schema reserves)
  needs the scorer to track the trigger's role raw-vs-post-template; worth its own trial. (2) A
  *Gemma-valid memory shape* (`keep_last_n=3` → `[system, user, assistant, user]`) to confirm
  survival classes match Qwen3 when the template *can* render — isolating the incompatibility to the
  sequence shape, not the policy.

### Verification (all green)

- `pytest` — 113 passed (7 new Gemma tests all ran, none skipped); Trials 0–3, 4a, 4b unchanged.
- `ruff check src tests`, `ruff format --check src tests`, `mypy` — clean.
- **Real Gemma-3-1b-it:** rows 1–2 `exact_survival` with `final_token_trigger_present=True`
  (model-invariant with Qwen3 despite the system-merge); rows 3–8 `no_survival` /
  `template_incompatible` with `final_prompt_token_count=0` and the template error captured — **no
  crash**. Fixture tripwire matches the live template.

---

## Output 5 — `docs/tasks/05-boundary-corruption.md` (boundary corruption: a trigger cut in half)

**Decision: kept in-house (no agents dispatched).** The core is a new correctness-critical scoring
primitive (`head_truncation_boundary_overlap`) wired into the shared scorer that must produce the
project's first `partial_survived=True` while leaving Trials 0–4 unchanged — plus a measured,
never-hardcoded budget derivation. Sensitive and sequential; the leader's to own.

### The finding

A long trigger straddling a head-truncation cut: the front half is dropped and the back half becomes
the literal prefix of the final input. Detected by a precise exact-match predicate — the surviving
fragment is a *suffix of the trigger* appearing as the *exact prefix of the final ids* — not a fuzzy
longest-common-run, so it cannot false-positive on ordinary content.

Delivered by the leader:

| File | Change |
|------|--------|
| `tokenization/token_search.py` | New pure `head_truncation_boundary_overlap(final_ids, trigger_ids)` → smallest `k` (`0<k<len`) s.t. `final_ids` begins with `trigger_ids[k:]`, else None. |
| `tests/test_token_search.py` | 9 exhaustive unit tests (hit, smallest-k, single-token suffix, no-match on full-survival / full-loss / unrelated / suffix-not-at-prefix / fragment-too-long / short triggers). |
| `scoring/survival.py` | When the full trigger is absent (either localization path), apply the predicate; on a hit set `partial=True`, `match_start=0`, `match_end=len-k`. Backward-compatible (prior budgets drop the whole trigger → predicate returns None). |
| `data/triggers/triggers.jsonl` | Added `boundary_001` (the purpose-built long trigger). |
| `data/base_conversations/base_conversations_001.jsonl` | `conv_000002`: Trial Zero's single-turn base in slot form (`{{PREFIX_SLOT}}`), so no memory-policy interaction. Validates. |
| `experiments/survivability_audit/boundary_spec.py` | `derive_split_budget` (`T − S − (E−S)//2`) and `derive_tight_budget` (`T − E`), measured from the `none` run. |
| `configs/pipeline_policies.example.yaml` | Three head-truncation-only ids (`boundary_generous` 40960, `boundary_split` 38, `boundary_tight` 29) with provenance comments (S=14, E=32, T=61). |
| `scripts/capture_boundary_fixture.py` + `tests/fixtures/boundary/` | Golden fixture: none span, three budgets, each condition's final ids + decoded text. |
| `tests/test_boundary_corruption.py` | Offline 3-row table from the fixture + the decoded-suffix evidence check + a live-tokenizer tripwire. |
| `docs/DATA_CONTRACTS.md` | The `boundary_corruption` vs `partial_survival` convention. |

### Verification (all green)

- `pytest` — 125 passed (Trials 0–4 unchanged; +9 predicate tests, +3 boundary tests, all ran).
- `ruff check src tests`, `ruff format --check src tests`, `mypy` — clean.
- `validate-jsonl data/base_conversations/base_conversations_001.jsonl --schema base_conversation` — ok.
- **Real Qwen3-0.6B, three-row table:** `boundary_generous` → `exact_survival` (partial False);
  `boundary_split` → `boundary_corruption`, `truncated_head`, **`partial_survived=True`** (span
  `(0, 9)`), decoded final begins with `_BLUE_BRIDGE_7F3XQ…` (the trigger's back half; full trigger
  absent); `boundary_tight` → `no_survival` (partial False). The two controls prove the predicate is
  a real detector, not one that always fires.
- Added a Trial 5 entry to `RUNNING_EXPERIMENTS.md`.

---

## Output 6 — `docs/tasks/06a-langchain-trim.md` + `06b-langchain-rag.md` (LangChain trim + RAG)

**Decision: HYBRID — delegated 06a, kept 06b in-house.** Both are large, framework-heavy tasks
(the strongest delegation candidates yet), but the tasks demand *behaviorally confirming* LangChain's
actual behavior rather than trusting the brief. So I did the sensitive/investigative core myself
(installed LangChain; probed `trim_messages` and the RAG components; designed + verified the RAG
schemas and the deterministic-ranking corpus), then delegated the one genuinely isolated,
now-well-scoped module (the trim adapter) and kept the new-delivery-contract task (RAG) in-house.

### Environment (precondition — leader set up)

`langchain-core` was not installed. Installed `langchain-core` **1.4.8**, `langchain-community`
4.4.x, `langchain-text-splitters`; added them to the `frameworks`/`rag` pyproject extras and to the
mypy ignore-list (langchain ships no stubs). Because it landed as a 1.x major version, I confirmed
every API behaviorally rather than trusting the 0.3-era brief.

### Behavioral facts I established (the "don't trust the brief" core)

- `trim_messages` parity: `strategy="last", token_counter=len, max_tokens=3, include_system=True`
  → keeps `[0, 4, 5]` (== `keep_last_n=2`). `strategy="first", max_tokens=3` → `[0, 1, 2]`.
- **`include_system`/`start_on`/`end_on` raise with `strategy="first"`** (valid only with `"last"`) —
  so the brief's lc_c/lc_d config had to drop `include_system`.
- Mid-message overflow: `allow_partial=False` drops the whole message (never truncates/raises);
  content splits **only** with `allow_partial=True` + a `text_splitter`. → boundary corruption is
  reachable via LangChain only that way.
- RAG: `InMemoryVectorStore` + a deterministic hash embedding + a crafted corpus (trigger doc
  off-topic) → the trigger doc ranks **last**; `top_k=1` excludes it, `top_k=5` includes it,
  deterministic across runs.

### Delegated — T6a.1 (one cursor agent)

`pipelines/langchain_adapter.py` (`LangChainTrimPolicy`, `ChatMessage↔BaseMessage` conversion,
adapter-backed default counter, lazy imports) + `tests/test_langchain_adapter.py` (lc_a–lc_e +
round-trip). Delivered clean (6 tests pass, ruff clean); it independently re-confirmed my facts and
found a `ToolMessage.tool_call_id` requirement. **On integration I fixed one mypy `union-attr`**
(narrowing the default-counter `adapter`), added the `pipelines/__init__.py` export, and verified.

### In-house — 06b (RAG delivery baseline)

| File | Change |
|------|--------|
| `schemas/documents.py` (new) | `Document(doc_id, content, trigger_slot)`. |
| `schemas/results.py` | `RagDeliveryResult` (retrieval logging fields; first defined in code — GENERIC_PLAN §11 design). |
| `schemas/__init__.py`, `cli.py` | Export `Document`/`RagDeliveryResult`; `validate-jsonl --schema document`. |
| `experiments/rag_survival/` (new package) | `DeterministicHashEmbedding` + `run_rag_delivery` (embed→InMemoryVectorStore→retrieve→pack→template→tokenize→classify). |
| `data/documents/corpus_000.jsonl` (new) | 1 trigger-bearing (off-topic) + 4 on-topic distractors; validates. |
| `tests/test_rag_delivery.py` (new) | Positive control delivers (all flags True); excluded → `not_retrieved`; ranking deterministic; real-tokenizer variant. |

### Docs

`DATA_CONTRACTS.md`: `Document` + `RagDeliveryResult` schemas with example records, and the
`trim_messages` behavior as a determined fact. `RUNNING_EXPERIMENTS.md`: Trial 6a and 6b entries.

### FLAGS for Saki

- **RAG schema was NOT already in code** (the source outline said it was). This trial adds and first
  exercises `RagDeliveryResult` — expected, not a surprise.
- **lc_c/lc_d config corrected:** `strategy="first"` cannot take `include_system` (raises), so those
  conditions drop it; the survival hypothesis still holds ([0,1,2] → old survives, recent dropped).
- **Boundary corruption via LangChain** needs `allow_partial=True` + `text_splitter`; the default
  path can't reach it, so `HeadTruncationPolicy` stays the only default token-level path.

### Verification (all green)

- `pytest` — 135 passed (Trials 0–5 unchanged; +6 trim, +4 RAG). `ruff`, `ruff format`, `mypy` — clean.
- `validate-jsonl … --schema document` — ok (5 rows).
- **Real Qwen3-0.6B:** lc_a `no_survival` (=2A), lc_b `exact_survival` (=2B), lc_c `exact_survival`,
  lc_d `no_survival`; RAG positive control delivers (all flags True, `exact_survival`), excluded →
  `no_survival`/`not_retrieved` (retrieved `doc_d1` only).
