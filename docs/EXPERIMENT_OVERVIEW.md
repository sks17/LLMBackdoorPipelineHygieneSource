# Experiment 1: trigger-delivery / prompt-survivability audit

## What this project is

This project audits whether a **harmless canary trigger** placed into raw user input survives the
real prompt pipeline (chat templating, token-budget enforcement, truncation, summarization, and
later RAG packing) and appears in the **final model-visible input**.

It is **not** backdoor training. The first project audits the pipeline. The question is delivery
validity, not whether a model becomes malicious:

> Before evaluating whether a model is robust to a backdoor trigger, first prove that the trigger
> survived the full context pipeline and reached the model-visible input.

## Why it matters

A model can appear robust to a trigger simply because the trigger never arrived. Treating "no
activation" as "the backdoor failed" is the central confusion this project removes. The useful
quantity is **P(activation | trigger delivered)**, not raw P(activation). Separating these
requires logging the exact final prompt for every trial.

## Primary output

The main artifact is **structured survival metrics**, not model generations. For each trial the
harness records, across the four logged layers, whether the trigger survived:

- **exact** — the trigger string appears verbatim in the final text,
- **token** — the trigger token IDs survive as a contiguous subsequence of the final tokens,
- **partial** — a non-empty proper run of trigger tokens reaches the final input (boundary
  corruption when caused by truncation),
- **none** — the trigger is absent.

Each failure is attributed to a pipeline stage (`failure_stage`): memory-policy drop, head/tail/
middle truncation, template change, compression, or final-token absence. See
[`DATA_CONTRACTS.md`](DATA_CONTRACTS.md) for the exact record shape.

## Model generation is optional and secondary

Generation is run only on a selected subset (controls, delivered/not-delivered, boundary cases).
A harmless activation check (e.g. emit `CANARY_SEEN` when the canary is present) is interpreted
**conditional on verified delivery**. In this scaffold generation is stubbed with a clear TODO;
the pipeline-only audit runs without it.

## Traceability

Every result is traceable to the exact final model-visible prompt. `PromptLogger` persists the
final prompt text (and the four layers) for a deterministic sample of trials, so any survival
verdict or model output can be inspected against the real input.

## Manipulated variables (project 1)

- **Trigger position** — prefix, middle, end, near truncation boundary (extensible to old-turn,
  tool output, retrieved doc).
- **Context length** — e.g. 1k / 4k / 8k / 16k / 32k, to apply budget pressure.
- **Pipeline policy** — none, head/tail/middle truncation, keep-recent-messages, summarize-old,
  summary-plus-recent.
- **Chat template** — per model; the same messages can yield different final tokens.
- **Trigger type** — random canary, natural phrase, multi-token phrase (boundary/unicode later).
- **Model family** — model choice matters less than pipeline choice; the artifact is
  model-agnostic (what input was actually sent?).

## Controlled variables

Same base conversations, trigger strings, tokenizer/template versions, context budget, decoding
settings, and seed across conditions, so survival differences are attributable to the pipeline
rather than to content or version drift.

## What a finding looks like

> Under naive evaluation, prefix triggers appear to fail in long contexts. After logging the final
> model-visible prompts, most of those failures are explained by trigger deletion before
> inference, not by model robustness.

The headline result tables are survival rate by (policy × position) and delivery vs activation vs
activation-given-delivery. The aggregation for the first table is produced by
`trigger-audit score-survival`.
