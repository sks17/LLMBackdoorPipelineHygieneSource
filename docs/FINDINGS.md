# Project 1 — Findings: trigger delivery through LLM context pipelines

**One sentence.** When a harmless canary is placed in raw user input, whether it reaches the model
at all is decided — often destroyed — by ordinary context management (chat templating, memory/trim
policies, truncation, RAG packing, tokenization), so much of what looks like a model *resisting* a
trigger is really the trigger never being *delivered* — and you must prove delivery before you can
claim robustness.

This document explains and justifies every finding. Numbers are the wave-2 cluster run
(**916,200 trials**); figures/tables live in [`outputs/analysis/project1/`](../outputs/analysis/project1)
(`figures/f*.png|svg`, `tables/t*.{csv,md,tex}`). Reproduction: [`docs/REPLICATION.md`](REPLICATION.md).

---

## 1. The audit

A *trigger* is a harmless canary string (e.g. `CANARY_TRIGGER_7F3XQ`). We insert it at a controlled
position in a base conversation, run that conversation through a real prompt pipeline, log the
trigger's presence at **four layers** — L1 raw messages → L2 post-memory-policy → L3 post-template
text → L4 final token ids — and classify whether it *survived* into L4 (the exact tokens the model
would receive). No model weights are involved: delivery is a property of the tokenizer + template +
policies, not the parameters, which is why it is measurable deterministically and at scale. The
scaffolding is drawn in **F0** (`f0_scaffolding`).

**Scale (wave 2).** 4 tokenizers (Qwen3-0.6B, Pythia-1B, TinyLlama-1.1B, Gemma-3-1B) × 4 data arms
(synthetic, long-document, and **real LMSYS-Chat-1M + WildChat**) × 5 policies × up to 7 positions ×
3 context budgets × 5 trigger types × a counterfactual twin = **916,200 trials**.

**Validity, established before any result (this is a precondition, not a finding).**
- **Reconciliation:** 916,200 results join the manifest **1:1** (0 duplicates), every base joins,
  **458,100 matched counterfactual pairs** — `gate.json`, `manifest.json`.
- **Counterfactual control (Gate 0):** every trigger-*absent* twin must be `no_survival`, delivered
  nowhere. **458,100 twins, 0 leaks.** This control is load-bearing: an earlier pilot's 312-twin
  leak exposed a real scorer bug (a natural-phrase trigger's common words coincidentally matching
  benign text), which we fixed by requiring the trigger to have actually reached the templated
  prompt before crediting partial survival. Every rate below **conditions on trigger-present rows**;
  pooling the absent twins would halve every number.

---

## 2. H1 / H3 — pipeline policy and position decide delivery (the dominant effect)

**Claim.** Which context-management policy runs, and where the trigger sits, jointly determine
whether it is delivered — each policy has a characteristic position it silently deletes.

**Evidence** — `t1_delivered_rate`, heatmap **F1** (`f1_delivery_heatmap`), effect sizes
`t_risk_difference`, cliffs **F2** (`f2_delivery_cliffs`):

| policy | prefix | middle | end | old_turn | recent | system |
|---|---|---|---|---|---|---|
| `none` (control) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `keep_recent_messages` | 0.33 | 0.55 | 0.55 | 0.33 | 0.55 | 0.97 |
| `truncate_head` | 0.21 | 0.59 | **0.94** | 0.21 | 0.45 | 0.21 |
| `truncate_tail` | mirror of head — kills **end**, keeps prefix/old_turn |||||
| `truncate_middle` | keeps ends/prefix, kills the **middle** |||||

**Justification.** `none` delivering 1.00 everywhere is the positive control — the plumbing works, so
every deficit below is real deletion. The pattern is mechanistic and predictable: **head truncation**
keeps only the last N tokens, so a *prefix* or *old-turn* trigger (0.21) is dropped from the front
while an *end* trigger (0.94) survives near the generation point; **tail truncation** is the exact
mirror; **middle truncation** removes the center; **keep-recent memory** drops old whole turns, so
prefix/old_turn (0.33) fall while the recent turn and the always-kept system message (0.97) stay. The
effect sizes are near-total and their CIs exclude zero (e.g. `truncate_head − none` at prefix ≈ −0.79
[−0.80, −0.78] in `t_risk_difference`). **Safety reading:** under naive evaluation a prefix trigger
under head truncation looks like the model "resisting"; the layered log shows it was deleted before
the model ever saw it.

---

## 3. H2 — delivery is model-invariant, except where the *template* diverges (Gemma)

**Claim.** Because delivery is deterministic token/template mechanics, it is invariant across models
that share those mechanics — but a model whose *chat template* has different structure breaks that
invariance.

**Evidence** — `t4_h2_model_invariance` (per-cell Δ vs Gemma + TOST, Holm + BH), outcome bands
`t4_outcome_bands` / **F4**:

- Qwen3, Pythia, TinyLlama agree cell-for-cell on the message/token-stage outcomes — the same policy
  × position × budget yields the same delivery, as expected for a tokenizer-only phenomenon.
- **Gemma diverges** on two mechanisms it alone exhibits: **2,700 `template_incompatible`** rows — its
  strict user/assistant alternation *rejects* the message sequence some memory policies produce, so
  **nothing renders** and the trigger is lost before tokenization; and **4,050 `role_migration`** rows
  (below).

**Justification.** Gemma has no system role — it merges the system message into the first user turn.
So delivery is model-invariant *as long as the template can render the produced sequence*; when it
cannot, the failure is a distinct delivery mode (`template_incompatible`), not a token drop. This is
why H2 must be stated as "invariant at the message/token stage, template-conditional at the render
stage," never as a flat "all models behave alike."

---

## 4. H4 — synthetic vs real conversations differ *only through the memory policy*

**Claim.** Synthetic base conversations and real ones (LMSYS/WildChat) deliver alike **except** under
memory policies, where real multi-turn structure changes the outcome — and even that gap closes when
there is no truncation pressure.

**Evidence** — `t5_h4_synthetic_vs_real` (Δ + paired cluster-bootstrap CI + TOST), by data_source
`t5_delivered_by_data_source`:

| policy | budget | lmsys Δ | wildchat Δ | longdoc Δ |
|---|---|---|---|---|
| `keep_recent` | 512 | −0.31 [−0.35,−0.27] | −0.27 [−0.32,−0.23] | +0.41 [+0.37,+0.45] |
| `keep_recent` | 2048 | **−0.01 [−0.06,+0.04]** | **+0.03 [−0.02,+0.07]** | +0.23 |

(Δ = data_source − synthetic; CI excluding 0 ⇒ non-equivalent.)

**Justification.** Real conversations carry **more, longer old turns**, so a keep-recent policy drops
more of them — real delivers *less* than synthetic (Δ ≈ −0.3). Long documents are single-turn (no old
turn to drop), so they deliver *more* (Δ ≈ +0.4). Crucially, at the **2048 no-truncation budget** the
lmsys/wildchat gap collapses to ≈ 0 (CI includes 0): the synthetic-vs-real difference is an
**artifact of how much old content the memory policy discards**, not a fundamental content property.
This is the honest H4 result — the content covariate is real but mechanism-mediated, so a delivery
audit validated on synthetic data generalizes to real conversations wherever memory pressure is
matched.

---

## 5. Four mechanism findings

**Boundary corruption is position-invariant** (`t7_boundary_census`, **F6** `f6_cut_anatomy`; the
targeted sweep in `RUNNING_EXPERIMENTS.md`). When a head cut lands *inside* a trigger's token span,
the front half is dropped and the back half becomes the literal prefix of the final input — a
partial (`boundary_corruption`) survival. Sweeping the cut within ±20 tokens of the span across
prefix/middle/end/old_turn: the outcome depends **only** on where the cut falls relative to the span
(before → whole survives, inside → boundary, after → lost), *identically for every placement*. So
partial-vs-whole survival is a property of the cut geometry, not the trigger's position in the
conversation.

**Role migration — first emission** (`role_migration`, 4,050 rows, Gemma + `system` position only).
A trigger planted in the *system* message renders **inside the user turn** under Gemma's system-merge
while still reaching the final tokens. This is a delivery mode where the trigger survives but its
*role* changes — the schema reserved the class; this is the first run to produce it, and only for the
one model whose template merges roles.

**Tool-output delivery** (`tool_output` position, agent/tool bases only). A trigger placed in a
`tool`-role result message is delivered like any other (0.67 under `none` on the small agent-base
cell), and — because build-manifest is slot-aware — this position is expanded **only** on bases that
have a tool message (Gemma, whose template has no tool role, is correctly excluded). This extends the
audit to agentic prompts.

**Semantic survival — the last uncovered mechanism** (`summarize_old_messages`, separate
producer×scorer-conditional cell; `RUNNING_EXPERIMENTS.md` + `outputs/summarization_semantic/`). When
old turns are **compressed into a summary** rather than dropped or cut, a trigger can survive as
*meaning* even when its exact string and tokens do not. With a paraphrasing summarizer and an
entailment scorer at a threshold τ **calibrated on the trigger-absent twins** (0 false positives,
Wilson [0.00, 0.32]): verbatim summaries → `exact_survival`, paraphrase → **`semantic_survival`** (the
first-ever emissions), drop → `no_survival`. This is the one delivery mode exact/token matching is
blind to, and it closes the last context-management mechanism the audit did not cover.

---

## 6. The headline: "backdoor failure" is largely **delivery failure**

**Claim.** An evaluator that does not verify delivery will attribute a large fraction of apparent
trigger failures to *model robustness* when they are actually *delivery failures* upstream of the
model.

**Evidence** — `t6_misattribution` decomposes every apparent failure (1 − delivered) by stage:
upstream drop (memory policy), token truncation (head/tail/middle), and template-incompatible. For a
cell like head-truncation × prefix, ~79% of trials are "apparent trigger failures" that are **entirely
delivery failures** — the trigger was deleted before tokenization. Because this is the pipeline-only
wave (no generation), the honest claim is an **upper bound on misattribution**: "up to X% of these
cells' apparent failures would be miscredited to model robustness." The full `P(activation | delivered)`
decomposition — actually running the model on delivered triggers — is **Project 2**.

---

## 7. Caveats and scope (stated plainly)

- **Delivery, not activation.** We measure whether the trigger reaches the model, not whether the
  model then does anything. Activation is Project 2.
- **Deterministic outcomes.** Every fully-specified trial is deterministic, so within-cell noise is
  zero; the unit of generalization is the **base conversation**, and all CIs are cluster-bootstrap
  over `base_id` (`n_boot=500` at this scale — point estimates are exact).
- **The McNemar control is degenerate** (`t3_mcnemar_control`): the absent arm is always 0, so it is a
  sanity statistic, not evidence for H1–H4 — the estimation (rates + effect sizes) is primary.
- **The semantic cell is producer × scorer conditional.** Unlike every model-agnostic delivery cell,
  it depends on both the summarizer and the entailment scorer; it is reported **separately**, never a
  clean 0/1 ("semantic delivery under scorer S at FP rate f"), and the offline reference backends here
  establish the machinery — a pinned-NLI + pinned-summarizer run (needs the `[generate]` extra) is the
  measurement form (`PRE_REGISTRATION.md`, 2026-07-04 amendment).
- **Boundary trigger by design.** `boundary_001` is built to be cuttable; it is reported per
  `trigger_type` so its by-design cuttability is not averaged into the headline delivery rate.

---

## 8. Why it matters for AI safety

Backdoor / trigger evaluations routinely conclude "the model was robust to trigger T." Project 1
shows that in a realistic deployment pipeline the trigger frequently **never reaches the model** —
and *which* triggers are lost is a systematic, predictable function of the context-management
strategy, not of the model. Any evaluation that does not log the final model-visible input is
measuring a mixture of delivery and robustness and cannot separate them. The practical takeaway:
**instrument the last layer.** The delivery audit is the necessary first stage; only on *delivered*
triggers does asking "did the model activate?" become meaningful.
