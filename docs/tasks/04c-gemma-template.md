# Task 04c — Gemma: when the memory policy's output is unrenderable (delegated)

**Audience:** an implementing agent (Claude). Builds on Task 04a (manifest) and, ideally, 04b
(offset localization — not required here since Gemma's tokenizer is clean).

**Goal:** add Gemma-3 as a third model. Unlike TinyLlama (a tokenizer-diversity case), Gemma is a
**template-structure** case, and it surfaces a delivery-failure mode neither Qwen3 nor TinyLlama
can: a template-agnostic memory policy can produce a message sequence the target model's template
**rejects outright**, so nothing is delivered — no trigger, no prompt at all.

## Verified facts about Gemma (supervisor pre-flight — build on these)

- `google/gemma-3-1b-it` (and `-270m-it`) load with the configured token. Template: `<bos><start_of_turn>user … <end_of_turn>\n<start_of_turn>model`.
- **No system role:** a system message is merged into the first user turn (system content prepended to it), and the turn tag is `model`, not `assistant`.
- **Strict alternation:** the template raises `TemplateError: Conversation roles must alternate user/assistant/…` on any non-alternating sequence — including `[system, assistant, user]`, the exact post-memory shape `keep_last_n=2` produces on `conv_000001`. So Gemma **cannot template rows 3–8 as built**. (Qwen3 and TinyLlama are lenient on that shape; verified.)
- **Clean tokenizer:** the trigger localizes cleanly (`subseq == offset == (4,14)`), so no offset fix is needed for Gemma specifically.

## Architecture — handle template-render failure as a delivery outcome, not a crash

Today `run_trial` → `ComposedPipeline.run` → `renderer.render(...)` would raise on the Gemma
keep-recent rows and abort the run. Per the project's own guidance ("if the adapter throws, that is
the bug to surface — do not special-case a model"), the finding is that **delivery failed at the
template stage**, so record it.

1. **New failure stage:** add `FailureStage.TEMPLATE_INCOMPATIBLE` to `schemas/results.py`.
2. **Typed render error:** in `ChatTemplateRenderer.render`, catch the underlying template error and re-raise as a defined `TemplateRenderError` (a small exception type in `prompts/chat_template.py`) carrying the message — so callers can distinguish a template rejection from an unrelated bug.
3. **Graceful capture in `run_trial`** (`manifest_runner.py`): wrap the `ComposedPipeline.run` call; on `TemplateRenderError`, return a `SurvivalResult` with `survival_class=NO_SURVIVAL`, `failure_stage=TEMPLATE_INCOMPATIBLE`, `final_token_trigger_present=False`, `final_prompt_token_count=0`, `raw_trigger_present`/`post_pipeline_trigger_present` computed from the messages that existed before templating, and the error text in `metadata`. Do **not** special-case Gemma by name — any model whose template rejects a produced sequence lands here.

Do not "fix" the incompatibility by reshaping Gemma's messages in the runner — the whole point is
that a standard memory policy produced an unrenderable sequence for this model.

## Your tasks

1. `FailureStage.TEMPLATE_INCOMPATIBLE`, `TemplateRenderError`, and the graceful-capture path above (unit test each: a policy that yields `[system, assistant, user]` + Gemma → `TEMPLATE_INCOMPATIBLE`).
2. Add a `gemma-3-1b-it` model config (`enable_thinking: false`).
3. Golden-fixture capture of Gemma's **rows 1–2** template (the `none` policy renders fine — system merges into the first user turn); commit `post_template_text` + `input_ids`. This is the tripwire and the evidence for the system-merge behavior.
4. Extend the manifest grid to include Gemma (its own composite policy ids, per 04b's per-model grid construction). Acceptance test below.

## Acceptance (the point of this trial)

| Gemma rows | policy | expected |
|------------|--------|----------|
| 1–2 | `none` | `exact_survival` — full alternating conversation renders; the user-turn trigger survives. Model-invariant with Qwen3 rows 1–2, **despite** the system-message merge. |
| 3–8 | any `keep_recent_messages…` | `NO_SURVIVAL`, `failure_stage=TEMPLATE_INCOMPATIBLE` — the post-memory `[system, assistant, user]` sequence is unrenderable. **Divergent** from Qwen3/TinyLlama, and that divergence is the finding. |

So the naive "message-stage outcomes are model-invariant" claim **fails** for Gemma — not because
the memory policy behaves differently, but because its output is not renderable by Gemma's template.
Document this in `RUNNING_EXPERIMENTS.md` as a distinct delivery-failure mode.

## Deferred (flagged, not in this task)

- **Role migration (system-position trigger × Gemma).** A trigger placed in the *system* message
  migrates into the user turn under Gemma's merge — the `SurvivalClass.ROLE_MIGRATION` case our
  schema already reserves. Detecting it requires the scorer to track the trigger's *role* (raw vs
  post-template), which is a scorer enhancement worth its own trial. Note it; do not build it here.
- **Gemma-valid memory (`keep_last_n=3` → `[system, user, assistant, user]`).** A follow-up could
  show that *when the template can render*, Gemma's survival classes do match Qwen3 — isolating the
  incompatibility to the sequence shape, not the policy. Optional; mention in the writeup.

## Constraints & verification

- No per-model branches in the runner beyond the generic `TemplateRenderError` handling. Reuse 04a/04b. One header comment per function/class; type hints throughout.
- Full gate green; Trials 0–3, 4a, and (if present) 4b unchanged and passing.
- Supervisor verifies: gate green; Gemma rows 1–2 `exact_survival` against the real tokenizer; Gemma keep-recent rows `TEMPLATE_INCOMPATIBLE`; no crash; the fixture matches the live Gemma template.
