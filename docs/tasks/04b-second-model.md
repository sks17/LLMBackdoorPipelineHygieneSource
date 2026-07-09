# Task 04b — Second model: tokenizer / chat-template diversity (delegated)

**Audience:** an implementing agent (Claude). Builds on Task 04a (manifest expansion).

**Goal:** add a second model so the grid becomes 8 rows × 2 models = **16 rows**, to test that
the message-level (Layer 1→2) outcomes are **model-invariant** — the memory policy runs before any
template, so it must not care which model is downstream. If the second model's rows 3–8 disagree
with Qwen3's on `survival_class`, the runner is leaking template-specific behavior into a
template-agnostic stage — a real bug.

## Two findings from the supervisor's pre-flight (read first — they change the plan)

1. **TinyLlama is the deliberate choice for 4b; Gemma is deferred to a later trial (4c).** A token
   is now configured and Gemma (`gemma-3-270m-it` / `gemma-3-1b-it`) loads. But Gemma's template
   requires **strict user/assistant alternation** and has **no system role**, so it hard-errors
   (`TemplateError: roles must alternate`) on the `[system, assistant, user]` post-memory shape that
   `keep_last_n=2` produces — it cannot template rows 3–8 as-is. That strict-alternation /
   role-migration behavior is a rich, model-specific finding that deserves its own trial. So 4b
   uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` **by design** (its Zephyr template is lenient on that
   shape → a clean invariance test; and its boundary re-tokenization forces the offset fix below) —
   not as a fallback.

2. **The trigger's standalone token IDs are not a subsequence of the templated tokens for
   TinyLlama.** Verified: TinyLlama's BPE re-tokenizes `CANARY_TRIGGER_7F3XQ` at the context
   boundary, so `find_subsequence(final_ids, encode(trigger))` returns `None` even though the
   string is plainly present. Qwen3 happened not to do this. **This breaks the tight-budget
   derivation "the same way as before"**, because `trigger_final_token_start/end` come from that
   subsequence search. This is not a bug to route around — it is exactly the tokenization
   confound `PROJECT_DESCRIPTION` §12 warns about, and Option B is the trial that surfaces it.

## Architecture — implement the tokenizer-agnostic localization first

### 1. Char-offset trigger localization (foundational — do before adding the model)

Add to `TokenizerAdapter`:

```python
def locate_token_span(self, text: str, subtext: str) -> tuple[int, int] | None: ...
```

For `HFTokenizerAdapter`: encode with `return_offsets_mapping=True`, find `subtext`'s character
span in `text`, and return the half-open token span whose offsets overlap it (or `None` if absent).
For `SimpleWhitespaceTokenizerAdapter`: a whitespace-offset implementation (or fall back to
`find_subsequence`). Verified to work for both real tokenizers: TinyLlama → `(15,28)` (where
subsequence gives `None`), Qwen3 → `(9,18)` (identical to subsequence). Both are fast tokenizers.

Wire it into scoring so the **token metrics are tokenizer-agnostic**: the runner computes the
trigger's token span with `adapter.locate_token_span(final_text, trigger.text)` and passes it into
`score_from_layers` (add an optional `trigger_token_span` parameter used for
`token_survived`/`final_token_trigger_present`/`trigger_final_token_start`/`_end`; fall back to
`find_subsequence` when not provided). **Regression bar:** because Qwen3's offset span equals its
subsequence span, Trials 0–3 must stay green unchanged.

### 2. Tokenizer-agnostic tight-budget derivation

Update Trial 3's `derive_tight_budget` path (and its Task-04a config value) to source
`trigger_final_token_end` from `locate_token_span`, so a `composite_tight_<model>` budget can be
derived for **any** tokenizer. Then derive `K_tinyllama` from a real TinyLlama run exactly as
`K=20` was derived for Qwen3 (from the measured span, never hardcoded).

### 3. Second model + two new policy ids

- Add a model-config entry for `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (`enable_thinking: false`; it has no thinking mode, stated explicitly per policy). Pull it locally for the cluster if generation is ever added (tokenizer-only for now).
- Add `keep_recent_messages+head_truncation_generous_tinyllama` and `keep_recent_messages+head_truncation_tight_tinyllama` to the policy config, with `K_generous` non-binding and `K_tight` = the derived TinyLlama value (provenance comment). Do **not** reuse Qwen3's `20`.

### 4. Golden-fixture capture for the second model's template

Capture TinyLlama's `post_template_text` + `input_ids` for rows 1–2 as a tripwire (same pattern as
`scripts/capture_trial_zero_fixture.py`). Do not take the template characterization above on faith —
capture and commit what the live tokenizer actually produces.

## Grid construction (integrates with the built 04a)

Task 04a's `expand_manifest`, `policy_registry`, and `run_trial` are already built and verified —
reuse them unchanged. The tight budget is tokenizer-specific, so the composite policy ids are
**per-model**: the 16-row manifest is the union of two 8-row expansions, each model paired with its
own policy ids (Qwen3 with `…_tight` (K=19), TinyLlama with `…_tight_tinyllama` (derived K)). Build
it as two `expand_manifest(...)` calls concatenated, not one Cartesian product (a full product would
pair Qwen3 with TinyLlama's budget and vice versa). Add the TinyLlama policy ids and model config to
the checked-in YAML; `run_trial` needs no per-model changes.

## Grid & acceptance

Grid: the Task-04a 8 rows for Qwen3 **plus** 8 rows for TinyLlama (its two composite ids replacing
Qwen3's), 16 rows total.

- **Rows 3–8 (TinyLlama) reproduce the same `survival_class` per condition as rows 3–8 (Qwen3)**,
  despite different token IDs and offsets underneath. `survival_class` is string-driven, so this is
  the model-invariance the trial tests.
- **Rows 1–2 (TinyLlama) are `exact_survival`** — and now, with the offset localization,
  `final_token_trigger_present` is `True` for them too (it would have been wrongly `False` under the
  old subsequence method).
- If a `TokenizerAdapter` throws or silently mishandles a template/tokenizer difference, that is the
  bug Option B exists to surface — do **not** special-case a model in the runner.

## Constraints & verification

- No per-model branches in the runner; the model is just a config/adapter choice. Reuse everything from Task 04a. One header comment per function/class; type hints throughout.
- Full gate green; Trials 0–3 and Task-04a rows unchanged and passing.
- Supervisor verifies: gate green; the offset span matches the subsequence span on Qwen3 (regression) and recovers the TinyLlama span the subsequence method misses; rows 3–8 `survival_class` identical across the two models; TinyLlama rows 1–2 `exact_survival` with `final_token_trigger_present=True`.

## For Saki (decisions to confirm)

- **Model choice:** Gemma is now unblocked, but its strict-alternation / no-system-role template cannot render the `keep_last_n=2` post-memory shape, so it is deferred to its own trial (4c). TinyLlama is the deliberate 4b model (lenient template = clean invariance test; boundary re-tokenization = validates the offset fix). See the model-spread recommendation in the session for the full sequencing (4b TinyLlama → 4c Gemma → 4d Pythia base).
- **Scope:** the offset-mapping localization is a real addition beyond "add a model," forced by finding (2). It is foundational and backward-compatible (Qwen3 unchanged), so I folded it into 4b rather than a separate trial — flag if you'd rather it be its own step.
