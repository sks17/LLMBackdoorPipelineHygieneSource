# Task 08 — Fan-out readiness: counterfactual pairing, base-completion, context caps (delegated)

**Audience:** an implementing agent (Claude). Three small, Wave-1-blocking changes to the manifest
grid and runner (per `docs/PRE_REGISTRATION.md`). All build on the accepted Task 04a machinery.

## 1. Counterfactual pairing in the expander

For paired McNemar's analysis, every trigger-present row needs its trigger-**absent** twin.

- Add `trigger_present: bool = True` to `TrialSpec`. Include it in the stable trial-id hash (`make_grid_trial_id`) so the two rows of a pair have distinct ids but share every other coordinate.
- `expand_manifest` emits **both** rows per grid point: one `trigger_present=True`, one `False`. Provide a way to recover the pair (same `base_id`/`model_id`/`trigger_position`/`pipeline_policy`/`context_length`; a helper `pair_key(trial)` returning that tuple is enough).
- `run_trial`: when `trial.trigger_present is False`, **skip insertion** — render/score the base conversation with no trigger. Expected result: `raw_trigger_present=False`, `survival_class=no_survival`. This is the scoring sanity control; the pipeline still runs so the negative row has real final-prompt tokens for length matching.
- Tests: `expand_manifest` yields pairs (2× rows, matched coordinates, distinct ids); a `trigger_present=False` row scores `no_survival` with `raw_trigger_present=False`.

## 2. Base-completion rendering path (Pythia has no chat template)

Pythia-1B has **no chat template**; `apply_chat_template` fails or silently uses a default. The
fan-out includes it, so the renderer needs an explicit base-completion mode.

- Add `chat_format: Literal["chat", "base"] = "chat"` to `ModelConfig` (Pythia → `"base"`), OR detect it (HF tokenizer `chat_template is None`) — prefer the **explicit config flag** so it is a stated decision, not silent behavior (consistent with the `enable_thinking` precedent).
- In base mode, render a **deterministic** base-completion Layer 3: concatenate messages as `"{role}: {content}\n"` per message (or a documented equivalent), no special/chat tokens, then tokenize. Log it as Layer 3 like any other render. Document the exact format in `docs/DATA_CONTRACTS.md`.
- The `HFTokenizerAdapter.locate_token_span` / scoring path is unchanged (offset localization is tokenizer-agnostic).
- Tests: a base-format render of a small conversation contains each message's content and no chat-special tokens; a prefix trigger survives base-completion under `policy=none` (`exact_survival`); Pythia does not route through `apply_chat_template`.

## 3. Model-capped context lengths

Pythia-1B's window is 2048; emitting 4k/8k/16k/32k cells for it is invalid.

- `expand_manifest` (or the experiment config that drives it) takes the model configs and **skips** `(model, context_length)` cells where `context_length > model.max_context_window`. Log how many cells were skipped per model (no silent truncation of the grid).
- Tests: with Pythia (2048) in the model set, no row has `context_length > 2048` for Pythia; Qwen3 models keep all length cells.

## Constraints & verification

- Reuse `expand_manifest`, `run_trial`, `ChatTemplateRenderer`, `ModelConfig`. No duplication. One header comment per function/class; type hints throughout. Keep heavy imports lazy.
- Full gate green; Trials 0–5 and Task 04a–04c unchanged and passing (the `trigger_present` field defaults `True`, so existing manifests/tests are unaffected — verify).
- Supervisor verifies: gate green; expander emits counterfactual pairs and caps Pythia's lengths; a `trigger_present=False` row is `no_survival`; Pythia renders via the base path and a prefix trigger survives `policy=none` on the **real** Pythia-1B tokenizer.

## Note for Saki

This is the last scaffolding before the Wave-1 synthetic fan-out. After it lands (plus the
pre-registration on record), the synthetic grid for `{none, head, tail, keep_recent}` × the 5 models
is shippable to the cluster with no new mechanism — it is Task 04a's `expand_manifest` + `run_trial`
at scale.
