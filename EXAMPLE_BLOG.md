# Your backdoor eval might be measuring the wrong thing: triggers die in the plumbing

*An example write-up of Project 1 — a trigger-delivery audit of LLM context pipelines. This is the
long, natural-language version; a terser, more formal treatment lives in the paper. Every table and
figure below is reproducible from the repository's committed results with a single command.*

---

## The evaluation everyone runs, and the question it skips

Here is a workflow that shows up all over the backdoor and trigger-robustness literature. You have a
model you suspect might carry a **backdoor** — a hidden behavior that fires when a specific **trigger**
appears in the input. (Throughout, a *trigger* is just a specific string; a *backdoor* is a learned
association between that string and some target behavior.) You want to know whether the model is
robust. So you construct inputs that contain the trigger, run them through the model, look at the
outputs, and the model behaves normally. You conclude: the model resisted the trigger. It's robust.

There is a step missing from that story, and it is not a small one. Between "you construct an input
that contains the trigger" and "you run it through the model," there is an entire machine — the
**context pipeline** — that rewrites your input before the model ever sees it. In any real deployment,
raw user text does not go straight into the model. It gets wrapped in a **chat template** (the special
formatting, like `<|user|> … <|assistant|>`, that turns a list of messages into the single string the
model was trained to read). If the conversation is long, a **truncation** policy throws away tokens to
fit the **context window** (the maximum number of tokens the model can attend to). A long-running chat
might run a **memory policy** that keeps only the most recent turns, or replaces old turns with a
summary. A retrieval-augmented (**RAG**) system packs in fetched documents. And at the very end,
everything is **tokenized** — chopped into the integer IDs the model actually consumes.

Every one of those steps can *delete your trigger*. And if the trigger was deleted before the model
saw it, then "the model behaved normally" tells you exactly nothing about whether the model is robust.
You didn't test the model's robustness. You tested whether your trigger survived the plumbing.

This post is about measuring that plumbing directly. We built a harness that plants a harmless canary
string in raw input, runs it through realistic context pipelines, and logs whether the string actually
reaches the final tokens the model would receive. We ran it a little under a million times. The
headline is simple and, I think, important:

> In a realistic pipeline, a trigger is frequently deleted before the model sees it — and *which*
> triggers get deleted is a systematic, predictable function of the context-management policy, not of
> the model. A large share of what looks like a model "resisting" a backdoor is actually the trigger
> never being **delivered**. You cannot separate the two unless you log the final model-visible input.

One more pair of terms up front, because the whole post turns on the distinction. **Delivery** is
whether the trigger reaches the model's actual input. **Activation** is whether the model, having
received the trigger, does the backdoored thing. Almost all backdoor evaluation is implicitly about
activation. This work is about delivery — the step before, which nobody measures, and which turns out
to be doing a lot of the work.

To make it concrete, picture a specific, entirely ordinary evaluation. You have a 4,000-token
conversation, you want to test a trigger, and you put the trigger in the system prompt — a sensible
choice, since system prompts are where instructions "belong." Your serving stack, like most, caps the
prompt at some budget and drops the oldest tokens when it overflows (head truncation). You run 100
such conversations, the model never misbehaves, and you file the result: robust. But if you had logged
the tokens the model actually received, you would have found that in 79 of those 100 conversations the
system prompt had been guillotined off the front before the model saw a single token of it. The model
"resisted" a trigger it was never shown. Your 100-trial evaluation was, unknowingly, a 21-trial
evaluation — and you have no idea which 21. That is the gap this project measures, and it is not a
corner case; it is the default behavior of ordinary infrastructure.

> A note on safety and scope: every "trigger" here is a *harmless canary* — a nonsense marker like
> `CANARY_TRIGGER_7F3XQ`, or a benign instruction-shaped phrase. We never train a backdoored model,
> never construct a harmful payload, and never elicit unsafe behavior. This is a measurement study of
> the pipeline, not an attack.

---

## What "delivery" means, concretely: the four-layer audit

To measure delivery you need to be precise about *where* in the pipeline you're looking, because the
whole point is that a trigger can be present at one stage and gone at the next. So we log the trigger's
presence at four layers, and the design of the whole experiment falls out of that choice.

![F0 — the experimental scaffolding: three data sources feed a length-binning + slot-planting step,
which fans out into a grid of trials, each of which runs a conversation through the four logged layers
and emits one result row.](outputs/analysis/project1/figures/f0_scaffolding.png)

*Figure F0 — the scaffolding.* Read it left to right. On the left are three sources of base
conversations: a **synthetic generator** (deterministic mock conversations we control completely),
**LMSYS-Chat-1M and WildChat** (two large corpora of real human–model conversations), and a
**long-document** source (single-turn conversations built around a long text). All three funnel into
one function, `to_base_conversation`, which does two jobs: it **length-bins** each conversation (pads
or trims it to a target token length, so a "512-token conversation" from any source really is 512
tokens under that model's tokenizer) and it **plants slots** — named placeholders like
`{{PREFIX_SLOT}}` or `{{OLD_TURN_SLOT}}` that mark exactly where a trigger can later be inserted.
Crucially, the synthetic and real arms go through *the same* function, so a synthetic base and a real
base differ only in where their words came from — which is what makes the synthetic-vs-real comparison
later on a fair one.

From there, `expand_manifest` takes the Cartesian product: every base × every trigger × every position
× every policy × every model, plus — the load-bearing part — a **counterfactual twin** for each.

Then each trial runs, and we log four flags, in order:

- **L1 (raw):** is the trigger in the raw messages, right after we insert it? (Always, for a
  trigger-present trial — this is the sanity floor.)
- **L2 (post-memory-policy):** is it still there after the memory policy (keep-recent, summarize) runs?
- **L3 (post-template):** is it in the fully-templated text, after chat formatting?
- **L4 (final tokens):** does it survive into the final token IDs, after truncation — the exact
  sequence the model would be given?

Because the four flags are logged in order, a failure is *attributable to a stage*. If a trigger is
present at L1 but gone at L2, a memory policy ate it. If it survives to L3 but dies at L4, truncation
cut it. If L3 is empty entirely, the template refused to render. Each trial emits one row — a
`SurvivalResult` — carrying those four flags plus a `survival_class` (exactly *how* it survived, or
didn't) and a `failure_stage` (exactly *where* it died). One row per trial; 916,200 rows.

Here is the design, laid out as a table, because the scale is part of the argument — the pattern isn't
one lucky configuration, it's every combination:

| Axis | Levels | n |
|---|---|---|
| model *(tokenizer + template + window)* | Qwen3-0.6B, Pythia-1B, TinyLlama-1.1B, Gemma-3-1B | 4 |
| data source | synthetic, long-document, **LMSYS-Chat-1M**, **WildChat** | 4 |
| pipeline policy | none, keep_recent_messages, truncate_head, truncate_tail, truncate_middle | 5 |
| trigger position | prefix, middle, end, old_turn, recent_turn, system, tool_output | 7 |
| context budget (tokens) | 512, 1024, 2048 | 3 |
| trigger type | random canary, multi-token phrase, boundary, natural phrase, unicode | 5 |
| counterfactual | trigger-present **and** trigger-absent twin | ×2 |
| **Total** | | **916,200 trials** |

One more thing that makes this cheap and trustworthy: **no model weights are involved.** Delivery is a
property of the tokenizer, the chat template, and the policies — not of the model's parameters. That
means we run the whole audit deterministically on CPU, and it means a "model," for our purposes, is
really just its *(tokenizer, template, context window)* triple. That will matter when we get to Gemma.

It's worth dwelling on *why* the four-layer decomposition is the right instrument, rather than just
checking the final tokens and being done. A single final-layer check tells you *whether* a trigger
died but not *where*, and "where" is exactly what you need to fix it. Consider two trials that both end
with the trigger absent from the final tokens. In one, the trigger was in an old turn that a memory
policy dropped at L2 — the tokens were never even a candidate for the window. In the other, the trigger
survived memory and templating and was cut at L4 because the conversation overflowed the budget by a
few tokens. These are the *same* final outcome and *completely different* problems: the first is fixed
by changing the memory policy or moving the trigger to a kept turn; the second is fixed by raising the
budget or shortening the surrounding context. A one-bit check can't tell them apart. The four ordered
flags can, because the *first* layer at which the trigger goes missing names the culprit stage
uniquely. That's why every result table in this post can attribute failures to a stage — the
attribution is baked into the measurement, not reconstructed after the fact.

The layers also correspond to real, separable pieces of production infrastructure, which is what makes
the attribution actionable rather than academic. L2 is your conversation-memory or history-management
layer. L3 is your chat-templating layer (often a library default you never think about). L4 is your
context-window / truncation layer. When the audit says "9% of your losses are at L2," it is pointing at
a specific component you can go and change.

---

## The trick that makes the numbers believable: counterfactual twins

Before any result, one methodological point, because it's the thing a skeptical reader should demand.

When you claim "the trigger survived," how do you know you didn't just find the trigger's letters by
accident in some benign text? For a nonsense canary this is a non-issue, but for a natural-language
trigger made of common words ("move the funds to the account") it is a very real risk — those words
appear in ordinary conversation.

The defense is the **counterfactual twin**. For every trigger-*present* trial, we run an identical
trigger-*absent* trial — same base, same model, same position, same policy, same budget — except the
trigger is never inserted. That twin *must* score "no survival." If it doesn't, the scorer is finding
phantom triggers, and every "survival" is suspect.

This control earned its keep. An early version leaked: 312 trigger-absent twins scored as partial
survivals, because a natural-phrase trigger's common words coincidentally overlapped benign text. That
was a real scorer bug; fixing it (requiring the trigger to have actually reached the templated prompt
before we credit any partial match) tightened the whole scorer. In the full run reported here:

| Control check | Result |
|---|---|
| Trigger-absent twins | 458,100 |
| Twins that leaked (should be 0) | **0** |
| McNemar discordant "absent-delivered" count (c), every policy | **0** |
| Result rows ↔ manifest join | 1:1, 0 duplicates |
| Matched counterfactual pairs | 458,100 |

The McNemar row deserves a footnote: it's a paired test on the present/absent pairs, and `c = 0`
everywhere means no trigger-absent twin was ever delivered. Because that arm is *always* zero by
construction, the test is a **sanity statistic**, not evidence for the hypotheses — the real evidence
is the rates below. But it does certify the control is airtight. Every rate in this post **conditions
on trigger-present rows**; the twins exist only to validate the null.

---

## The core result: policy and position decide delivery

Here is the single most important table. It is the delivered-rate matrix: rows are
context-management policies, columns are trigger positions, each cell is the fraction of
trigger-present trials in which the trigger reached the final tokens. Every core cell rests on 15,120
trials across 1,008 base conversations.

**Table 1 — delivery rate by policy × position.**

| policy | prefix | middle | end | old_turn | recent_turn | system | tool_output |
|---|---|---|---|---|---|---|---|
| `none` *(control)* | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.67 |
| `keep_recent_messages` | **0.33** | 0.55 | 0.55 | **0.33** | 0.55 | 0.97 | 0.67 |
| `truncate_head` | **0.21** | 0.59 | 0.94 | **0.21** | 0.45 | **0.21** | 0.67 |
| `truncate_tail` | 1.00 | 0.44 | **0.25** | 1.00 | 0.74 | 1.00 | **0.11** |
| `truncate_middle` | 1.00 | **0.43** | 0.83 | 1.00 | 0.92 | 1.00 | 0.67 |

The same matrix as a heatmap, for readers who prefer to see the pattern than read it:

![F1 — delivery-rate heatmap: the same numbers as Table 1, coloured white (0) to dark blue (1). The
top row is uniformly dark; every other row has a light band where that policy deletes the
trigger.](outputs/analysis/project1/figures/f1_delivery_heatmap.png)

Let me walk through the table, because the whole argument is in it.

**The top row, `none`, is your control.** "None" means no context management — just template and
tokenize. Every cell is 1.00: a trigger placed *anywhere* is delivered *always*. This is exactly what
should happen, and it matters, because it proves the plumbing works. The pipeline faithfully delivers
triggers when nothing tries to delete them, so every sub-1.00 cell elsewhere is a *real deletion*, not
a bug in our harness. (The one exception, `tool_output` at 0.67, is a small special cell explained
later.)

Now `truncate_head`. Head truncation keeps only the *last* N tokens and throws away the front. The
signature is stark: a **prefix** trigger is delivered 21% of the time and an **old_turn** trigger also
21% — both live near the front, so head truncation guillotines them. The **system** position sits at
0.21 too: the system prompt is at the very front, first against the wall. But **end** survives 94% of
the time, because the end is nearest the generation point, which head truncation always keeps.

Now `truncate_tail`, and you see the *mirror image*. Tail truncation keeps the front and drops the end,
so **end** collapses to 0.25 while **prefix**, **old_turn**, and **system** stay at 1.00. And
`truncate_middle` completes the picture: it keeps both ends and removes the center, so **middle** drops
to 0.43 while everything else stays high. `keep_recent_messages` — a *memory* policy that keeps the
system message plus the most recent whole turns and drops older ones — kills prefix and old_turn (0.33)
while sparing the recent turn (0.55) and the always-kept system message (0.97).

The `tool_output` column is worth its own note, because it's the only cell where even the `none`
control isn't 1.00 (it sits at 0.67, and drops to 0.11 under tail truncation). A `tool_output` trigger
is one planted in the *result of a tool call* — the kind of content a RAG system or an agent's function
call injects into the conversation. It's a "slot-strict" position: it can only exist in conversations
that actually have a tool/agent turn to hold it, so it's a smaller, more specialized cell (n=900 rather
than 15,120), and it lives structurally near the end of an agent exchange. That end-placement is why
tail truncation is so brutal to it (0.11): tool outputs are among the last things added, so a policy
that drops the tail drops them first. For anyone auditing an agentic or retrieval system, this is the
cell to watch — a trigger arriving through a tool result is *more* fragile than one in ordinary chat
text, not less, and a tail-dropping budget will eat it almost every time.

Step back and the finding is this: **every policy has a signature — a specific position it reliably
deletes** — and the four positions form a coherent geometry (front, middle, end, and "old" mapping
cleanly onto which region each policy discards). None of this is the model doing anything. It is
deterministic bookkeeping about which tokens fit in the window. If you evaluated a backdoor by putting
its trigger at the prefix under a head-truncating system, you would see it "fail" 79% of the time — and
you'd be measuring truncation, not the model.

The size of these effects is not subtle. Below are the risk differences — each policy's delivery rate
minus the `none` baseline, for a prefix trigger — with a **cluster-bootstrap** confidence interval (a
way of estimating how much the number would wobble if we'd drawn a *different* set of base
conversations, which is the right notion of uncertainty here because the per-trial outcome itself is
deterministic — there is no within-trial noise, only generalization across conversations):

**Table 2 — effect sizes (delivery rate vs `none`, prefix trigger).**

| policy | Δ delivery | 95% cluster-bootstrap CI |
|---|---|---|
| `truncate_head` | −0.79 | [−0.80, −0.78] |
| `keep_recent_messages` | −0.67 | [−0.69, −0.65] |
| `truncate_tail` | 0.00 | [0.00, 0.00] |
| `truncate_middle` | 0.00 | [0.00, 0.00] |

These are near-total effects with tight intervals. They are not noise, and they are exactly zero where
the mechanism predicts zero (a prefix trigger is untouched by tail or middle truncation).

---

## Does the trigger's *content* matter? No — and that's the point

There's an assumption baked into most backdoor evaluation that's worth dragging into the light: the
belief that whether a trigger survives has something to do with *what the trigger is*. It's a natural
assumption — surely a long, weird string behaves differently from a short natural phrase? We ran five
different trigger types to check: a random canary (`CANARY_TRIGGER_7F3XQ`), a multi-token natural
phrase, a deliberately long "boundary" trigger, an instruction-shaped natural phrase, and a Unicode
trigger with accented characters. Here is what they deliver at, for one representative cell —
truncate_head at the prefix:

**Table 2b — delivery rate by trigger type (truncate_head × prefix).**

| trigger type | trigger | delivered rate | n |
|---|---|---|---|
| random canary | `rand_001` | 0.2057 | 3,024 |
| multi-token phrase | `multi_001` | 0.2057 | 3,024 |
| boundary (long) | `boundary_001` | 0.2057 | 3,024 |
| natural phrase | `natural_001` | 0.2057 | 3,024 |
| unicode | `unicode_001` | 0.2057 | 3,024 |

They are identical to four decimal places. This is not a coincidence and it is not a rounding
artifact; it is the whole thesis in miniature. Delivery is decided by the *pipeline* — where the cut
falls, which turn gets dropped — and the pipeline does not care what the trigger says. Whether your
canary is a nonsense string or a polished natural-language instruction, if it sits at the prefix under
head truncation, it is delivered exactly 20.6% of the time, because that is the fraction of base
conversations short enough that the prefix survives the cut. The trigger's semantics are irrelevant to
its delivery. (Semantics come back in exactly one place — summarization, where a paraphrase can carry
meaning without carrying letters — and we get there at the end. Everywhere else, delivery is blind to
content.)

This has a sharp implication for how you should read *any* backdoor result. If two triggers with
totally different content show the same "robustness" in your eval, the natural inference is "the model
generalizes across triggers." The delivery audit offers a deflationary alternative: maybe they show
the same robustness because they were *deleted at the same rate*, and the model never saw either of
them. You cannot distinguish "the model resists both" from "the pipeline drops both" without logging
delivery.

---

## Where, exactly, do triggers die?

Table 1 tells you *that* triggers die and under *which* policy. The next question is *where in the
pipeline* they die — which layer does the killing — because different stages imply different
mitigations. Two views of that.

First, the aggregate flow: of all 458,100 trigger-present trials, where did they end up?

**Table 3 — outcome of every trigger-present trial (the delivery flow).**

| outcome | stage | count | share |
|---|---|---|---|
| delivered — exact | reached L4 verbatim | 323,346 | 71% |
| delivered — role migration | reached L4, role changed | 4,050 | 1% |
| dropped — memory | died at L2 (memory policy) | 39,391 | 9% |
| dropped — template | died at L3 (template) | 3,300 | 1% |
| dropped — truncation | died at L4 (truncation) | 88,013 | 19% |

![F8 — delivery flow: a Sankey diagram of the same 458,100 trials fanning from raw messages to
outcomes, band width proportional to count.](outputs/analysis/project1/figures/f8_delivery_flow.png)

Roughly **28% of triggers never reach the model**, and that 28% is spread across three distinct
pipeline stages: 9% killed at the memory stage, 1% at the template stage, 19% at truncation. Nearly a
fifth of *all* triggers are killed by truncation alone.

Second, the per-policy view — *which* stage each policy uses to do its killing. This is the failure
attribution table: for each policy, of the trials whose trigger was lost, what fraction died at each
stage.

**Table 4 — failure attribution (of each policy's non-delivered trials, where they died).**

| policy | dominant failure stage | share of its failures | other |
|---|---|---|---|
| `keep_recent_messages` | memory_policy_dropped | 93% | 7% template_incompatible *(Gemma)* |
| `truncate_head` | truncated_head | 99% | — |
| `truncate_tail` | truncated_tail | 99% | — |
| `truncate_middle` | truncated_middle | 98% | — |

This is diagnostically clean: memory policies kill triggers at the *message* level (the
`memory_policy_dropped` stage), early; truncation policies kill at the *token* level, late. The same
story is visible in the funnel figure, where you can watch each policy's line take its drop at a
different stage:

![F3 — layer funnel: for each policy, the fraction of triggers still present at each of the four
layers. The keep-recent line drops at the memory stage; the truncation lines stay flat until the final
token stage.](outputs/analysis/project1/figures/f3_layer_funnel.png)

The practical upshot: if triggers are dying at the memory stage, you change your memory policy; if
they're dying at truncation, you change your budget. The four-layer log tells you which. And notice the
lone 7% in the keep-recent row — that's `template_incompatible`, and it's entirely Gemma. Hold that
thought.

---

## The reframe: "backdoor failure" is mostly *delivery* failure

Now put the pieces together, because this is the point of the whole exercise.

Suppose you're evaluating a backdoor. You place its trigger at the prefix — a completely natural
choice — and your deployment head-truncates long prompts, also natural. From Table 1, that cell
delivers 21% of the time. So **79% of your evaluation trials never delivered the trigger at all.** If
you don't log the final input, all 79% look like "the model didn't respond to the trigger" — like
robustness. They are deletions.

The misattribution table makes this quantitative. For each cell it takes the apparent-failure rate (1 −
delivered) and decomposes it into *where* the failure actually happened: upstream drop (memory), token
truncation, template-incompatibility.

**Table 5 — misattribution: apparent "robustness" that is really delivery failure.**

| policy × position | delivered | apparent failure | of which: memory | truncation |
|---|---|---|---|---|
| `truncate_head` × prefix | 0.21 | **0.79** | 0.00 | **0.79** |
| `truncate_head` × old_turn | 0.21 | 0.79 | 0.00 | 0.79 |
| `truncate_head` × end | 0.94 | 0.06 | 0.00 | 0.06 |
| `keep_recent` × prefix | 0.33 | 0.67 | **0.67** | 0.00 |
| `keep_recent` × middle | 0.55 | 0.45 | 0.45 | 0.00 |

Read the first row: for a prefix trigger under head truncation, **79% of trials are apparent failures,
and 100% of that 79% is delivery failure** — the trigger was deleted by truncation, not resisted by
the model. The `keep_recent × prefix` row tells the same story through a different stage (67% apparent
failure, all of it memory drop). And notice the contrast within a single policy: `truncate_head × end`
has only 6% apparent failure, because the end survives. *Same model, same trigger, same policy* — the
apparent "robustness" swings from 6% to 79% purely on where you put the trigger.

I want to be careful about the claim, because it's easy to overstate. This is a **delivery** audit; we
did not run the models and check activation. So the honest statement is an **upper bound on
misattribution**: an evaluator who doesn't verify delivery would attribute *up to* that fraction of a
cell's apparent failures to model robustness when they are actually delivery failures. The complement —
of the triggers that *are* delivered, how many activate — is the `P(activation | delivered)` question,
and that's the next project, because it requires running the models. But even as an upper bound, the
message stands: an evaluation that doesn't instrument the last layer is measuring a blend of delivery
and robustness and cannot separate them.

---

## The model that breaks the pattern

I said delivery is a property of the *(tokenizer, template, window)* triple, not the weights. A strong
prediction follows: models that share those mechanics should agree cell-for-cell. And mostly they do —
Qwen3, Pythia, and TinyLlama produce the same delivery outcomes for the same policy × position ×
budget, exactly as a tokenizer-only phenomenon should. The interesting cases are where the prediction
*breaks*, and there's exactly one: **Gemma**.

We test this with **equivalence testing** (TOST — two one-sided tests): rather than asking "are two
models different?", TOST asks "are they the *same* within a margin?" (here ±5 percentage points). Most
cells pass — the models are equivalent. Gemma's don't. Here are two representative cells (the
difference is Gemma minus another model; a CI excluding zero means genuinely different):

**Table 6 — H2, where Gemma diverges (per-cell delivery-rate difference vs Gemma).**

| cell | model | Δ vs Gemma | 95% CI |
|---|---|---|---|
| keep_recent / end / 512 | Pythia-1B | −0.23 | [−0.33, −0.12] |
| keep_recent / end / 512 | Qwen3-0.6B | −0.21 | [−0.32, −0.11] |
| keep_recent / end / 512 | TinyLlama | −0.24 | [−0.34, −0.14] |

And Gemma alone produces two outcome classes no other model does:

| Gemma-only outcome | count | why |
|---|---|---|
| `template_incompatible` | 2,700 | template refuses to render some memory-policy shapes |
| `role_migration` | 4,050 | system-planted trigger renders inside the user turn |

Here's what's going on. Gemma's chat template has **no system role**: where other templates render a
system message as its own turn, Gemma *merges* it into the first user turn. So a trigger you planted in
the system message gets rendered *inside a user turn*. It's still delivered — it reaches the final
tokens — but its *role* has changed. That's `role_migration`: survival with a change of provenance, and
the first time this project ever emits that class. For a safety researcher it's a genuinely interesting
delivery mode: a trigger that was "in the system prompt" arrives labeled as user content, which matters
for any defense that trusts the system role differently from user text.

Gemma's template *also* demands strict user/assistant alternation. Some memory policies, after dropping
old turns, produce a sequence that doesn't alternate — and Gemma's template *refuses to render it at
all*, raising an error rather than producing a string. Our harness records this as its own delivery
failure (`template_incompatible`): the trigger is lost not because a token was cut, but because
*nothing was rendered*. The composition figure shows both rare classes as thin coloured slivers on the
`system` and `keep_recent` rows:

![F4 — outcome composition: for each policy × position, the share of trials in each outcome band. Most
bars are green (exact) and grey (none); the system rows carry a blue role-migration sliver and the
keep-recent rows a red template-incompatible sliver — both
Gemma-only.](outputs/analysis/project1/figures/f4_outcome_composition.png)

The lesson generalizes past Gemma: delivery is model-invariant *only as long as the template can render
the sequence the policies produced*. When template structure differs — no system role, strict
alternation — you get new delivery failure modes the other models don't have. "We validated this on
model X" does not transfer across template families. You have to check the template, not just the
tokenizer.

---

## But does it hold on *real* conversations?

A fair objection: our synthetic conversations are ones we built. Maybe the whole pattern is an artifact
of clean synthetic data, and real human logs would behave differently. This is the **H4** question, and
it's why we pulled two large corpora of real conversations — **LMSYS-Chat-1M** and **WildChat** — ran
them through the identical grid, and compared. (We drop toxic/flagged conversations and strip all
metadata and PII; only the benign role-and-content skeleton enters the audit.)

First, the delivery rates by data source:

**Table 7 — delivery rate by policy × data source.**

| policy | synthetic | LMSYS | WildChat | long-doc |
|---|---|---|---|---|
| `none` | 0.99 | 1.00 | 1.00 | 1.00 |
| `keep_recent_messages` | 0.65 | **0.45** | **0.50** | 1.00 |
| `truncate_head` | 0.45 | 0.44 | 0.44 | 0.35 |
| `truncate_tail` | 0.68 | 0.75 | 0.75 | 0.77 |
| `truncate_middle` | 0.90 | 0.87 | 0.83 | 0.86 |

Look at the `none` row: all four sources deliver ~1.00 — real and synthetic are indistinguishable when
nothing is deleting. Look at the truncation rows: they track closely across sources (the differences
are single-digit percentage points). The one row where sources genuinely diverge is
`keep_recent_messages`: synthetic 0.65, but LMSYS 0.45 and WildChat 0.50 — real conversations deliver
meaningfully *less* under the memory policy. Long documents go the other way (1.00), because they're
single-turn — there are no old turns to drop.

Now the formal comparison — real minus synthetic, with confidence intervals, at a tight budget and a
loose one:

**Table 8 — H4: real vs synthetic under keep-recent (Δ = source − synthetic).**

| budget | LMSYS − synth | WildChat − synth | long-doc − synth |
|---|---|---|---|
| 512 | −0.31 [−0.35, −0.27] | −0.27 [−0.32, −0.23] | +0.41 [+0.37, +0.45] |
| 2048 | **−0.01 [−0.06, +0.04]** | **+0.03 [−0.02, +0.07]** | +0.23 [+0.19, +0.26] |

Read the top row: under keep-recent at a tight budget, real conversations deliver ~30 percentage points
*less* than synthetic. Why? Real human conversations carry **more, and longer, old turns**, and
keep-recent's whole job is to drop old turns — there's simply more old material to throw away, so more
triggers planted in old turns get dropped.

Now the bottom row, and this is the part I find reassuring about the method. At the **2048 budget —
where nothing is truncated — the LMSYS and WildChat differences collapse to essentially zero** (both
CIs comfortably include 0). Synthetic and real are *not* fundamentally different in how their triggers
get delivered. The gap under keep-recent is *entirely* about how much old content the memory policy
happens to discard — remove the pressure that discards it, and the gap disappears. The
synthetic-vs-real difference is a **mechanism-mediated artifact**, not a content property.

Practically, that's a license: a delivery audit validated on synthetic conversations generalizes to
real ones, *as long as you match the memory pressure*. The synthetic arm isn't cheating; it's a
faithful stand-in wherever the amount of droppable history is matched.

---

## Two more mechanisms, because they're neat and they matter

**Boundary corruption, and why it's position-invariant.** Everything so far treated a trigger as
surviving or not — a binary. But truncation doesn't respect trigger boundaries; a cut can land *in the
middle of a trigger*, dropping its front half and leaving its back half as the literal first tokens of
the prompt. We call that **boundary corruption** — a partial survival. Does it depend on *where* the
trigger sits in the conversation? We ran a targeted sweep: for a trigger of known length, we placed the
truncation cut at a range of offsets within ±20 tokens of the trigger's span, at every position, and
asked what survives.

**Table 9 — boundary sweep: outcome by where the cut lands (each row identical across prefix / middle /
end / old_turn).**

| cut lands… | outcome | rate | partial-survival flag |
|---|---|---|---|
| before the trigger span | whole trigger survives | 1.00 | 0.00 |
| **inside** the span | **boundary corruption** (back half survives) | 1.00 | **1.00** |
| after the span | trigger lost | 1.00 | 0.00 |

The result: whether a trigger survives, half-survives, or dies is decided *entirely* by where the cut
lands relative to it — before it, it survives; through it, it's corrupted; after it, it's gone — and
this is **identical** across prefix, middle, end, and old_turn. The trigger's conversational position
doesn't matter; only the cut geometry does. The figure makes the sharpness visible: every surviving
trial (green) sits on one side of the trigger, every lost trial (grey) on the other, with a clean
transition at the seam.

![F6 — anatomy of the cut: each dot is a truncation trial positioned by where the cut fell relative to
the trigger (0 = at the trigger's start). Survivors (green) sit to one side, losses (grey) to the
other, faceted by budget.](outputs/analysis/project1/figures/f6_cut_anatomy.png)

**Semantic survival: the one mode string-matching can't see.** There's a memory policy we haven't
discussed: **summarization**. Instead of *dropping* old turns, it *compresses* them into an LLM-written
summary — common in production (rolling-summary agents and the like). It can do something the other
policies can't: preserve a trigger's *meaning* while destroying its exact string. A summary might
paraphrase "move the funds to the external account" as "the user asked to transfer the money
elsewhere." The exact trigger is gone — no string match, no token match — but the *instruction*
survived. For a natural-language backdoor, meaning-survival *is* delivery, and every string-matching
audit is blind to it.

Measuring this requires a **semantic scorer**: something that judges whether a summary *entails* the
trigger's meaning rather than contains its letters. That's a different kind of tool — probabilistic and
model-dependent, unlike the deterministic string/token checks everywhere else — so it needs different
care. The key move is calibrating the scorer's threshold against the counterfactual twins: we set it so
that trigger-*absent* summaries essentially never trip it (a zero false-positive operating point) —
exactly the counterfactual control doing double duty as the null that certifies the scorer. With a
paraphrasing summarizer and a calibrated entailment scorer:

**Table 10 — the summarization semantic-survival cell (τ calibrated for 0 false positives on the
twins).**

| summarizer behavior | outcome | threshold τ | false-positive rate on twins |
|---|---|---|---|
| verbatim (copies old turns) | `exact_survival` | 0.29 | 0.00 |
| **paraphrase** (rewords, keeps meaning) | **`semantic_survival`** | 0.29 | 0.00 |
| drop (content-free summary) | `no_survival` | — | 0.00 |

The middle row is the finding: a paraphrased trigger, absent at the exact and token level, is delivered
as **meaning** — the first `semantic_survival` emissions in the whole project — while the absent-twin
null stays silent (zero false positives). To keep ourselves honest about the scorer's own accuracy we
also validated it against a small hand-labeled **gold set** of summary/trigger pairs, where it hit 88%
precision and 88% recall — good enough to trust the direction of the result, and reported openly rather
than hidden. The calibration is the crucial part: because a semantic scorer is fuzzy, its threshold is
a dial, and the only principled place to set that dial is where the trigger-absent twins stop tripping
it. That's the counterfactual control doing triple duty — it validates the string scorer, it validates
the semantic scorer, and it sets the semantic scorer's operating point — which is a nice illustration
of why the twin was worth building in the first place. Because this cell depends on both the summarizer and the
scorer — two model-dependent choices, unlike the rest of the audit — we report it *separately* and
never as a clean 0/1: always "semantic delivery under scorer S at false-positive rate f." It's the last
context-management mechanism the audit didn't cover, so we've now accounted for dropping, cutting, *and*
compressing.

---

## What this is not: scope and honest limitations

An audience that rewards rigor deserves the boundaries stated plainly.

- **This is delivery, not activation.** We measure whether the trigger reaches the model, full stop. We
  do not run the models and check whether they do anything. That's the natural next project, and the
  one that turns "up to X% could be misattributed" into "X% actually was." Everything here is a
  necessary *first* stage, not the whole story.
- **The outcomes are deterministic.** A fully-specified trial has no randomness — same inputs, same
  result. So the "uncertainty" in the confidence intervals isn't within-trial noise (there is none);
  it's *generalization* to new conversations. That's why every interval is a cluster bootstrap over
  base conversations, and why the unit that matters is the conversation, not the trial. (At this scale
  the bootstrap uses 500 resamples; the point estimates are exact.)
- **The McNemar control is degenerate.** Because the trigger-absent arm is always zero by construction,
  the paired test is a *sanity check*, not evidence — the rates and effect sizes are the evidence.
- **The semantic cell is conditional.** Unlike every other, model-agnostic cell, semantic survival
  depends on the specific summarizer and scorer. We report it apart, with its calibration and its
  false-positive rate. The offline version establishes the machinery; a pinned-model measurement run is
  the scientific form.
- **One trigger is built to be cut.** The long "boundary" trigger exists specifically to be cuttable,
  so we report per trigger-type rather than averaging its by-design cuttability into the headline.

None of these undercut the main result. They bound it. And bounding your own claim is how you earn the
reader's trust in the part you didn't bound.

---

## What this changes about how you should run an eval

Abstract findings are cheap; let me make this operational. If you build, run, or read backdoor and
trigger evaluations, here is what the delivery audit says you should actually do differently.

**Log the final token IDs, and diff them against your raw input.** This is the whole ballgame, and
it's nearly free. Your harness already builds the exact sequence it hands the model; capture it, and
before you record any result, check that your trigger's tokens are actually in there. One boolean per
trial — `trigger_delivered` — converts an uninterpretable "the model didn't respond" into two clean
buckets: "delivered but no activation" (a real robustness signal) and "never delivered" (a
measurement artifact you must exclude). Without that boolean you are averaging the two together and
calling the average "robustness." A delivery flag is the single highest-leverage line of
instrumentation you can add, and most harnesses don't have it.

**Report robustness conditioned on delivery, never marginally.** The number that means something is
`P(no activation | delivered)` — of the triggers the model actually received, how many it resisted.
The number most papers report is `P(no activation)` marginally, which silently folds in every trigger
the pipeline ate. Those two numbers can differ by tens of percentage points, as Table 5 shows: a cell
with 79% "apparent robustness" can have 0% real robustness once you condition on delivery, because
100% of the apparent robustness was truncation. Publishing the marginal number without the delivery
denominator is, at this point, a known bug.

**Sweep position, not just presence.** Table 1 shows the same trigger swinging from 21% to 94%
delivery purely on where it sits. So "we tested the trigger" is underspecified — you tested it *at a
position*, and your result is entangled with your serving stack's truncation direction. If you want a
result that generalizes, either sweep the trigger across positions (prefix, middle, end, old turn) or
state explicitly which position you used and which truncation your stack applies, so a reader can tell
whether your "robust" is a robustness fact or a geometry fact.

**Match your eval's memory pressure to production.** The synthetic-vs-real result (Table 7, Table 8)
says the thing that determines delivery under a memory policy is *how much droppable history there
is*. If your eval uses short, clean conversations but production runs long ones with lots of old
turns, your eval will *over*-deliver triggers relative to production and overstate how often the model
gets to see them — or under-deliver, if it's the reverse. The fix isn't "use real data" (synthetic is
fine); it's "match the amount of history the memory policy has to throw away."

**Check the template, not just the tokenizer.** The Gemma result (Table 6) says delivery is
model-portable *until the chat template's structure changes*. Before you reuse an eval across model
families, render a few trigger-bearing prompts through each family's actual template and look: does a
system-role trigger stay in the system role, or migrate? Does every memory-policy output render, or do
some raise? A trigger that "worked on Llama" can role-migrate or fail-to-render on Gemma, and your eval
won't tell you unless you look at the rendered string.

**Don't trust string-match delivery under summarization.** If any part of your stack summarizes or
compresses history (rolling-summary agents, "memory" features), a trigger can be delivered as *meaning*
while its exact string is gone — invisible to a substring check. For natural-language triggers under
those policies, you need a semantic delivery check, calibrated (as in Table 10) so it essentially never
fires on trigger-absent inputs. Otherwise you'll score a meaning-preserved trigger as "not delivered"
and, again, miscredit the model.

Every one of these is cheap relative to the cost of a wrong robustness claim, and every one of them
falls directly out of a single design choice: instrument the last layer.

---

## The takeaway: instrument the last layer

If there's one thing to carry away, it's a habit, not a number.

Backdoor and trigger evaluations routinely conclude "the model was robust to trigger T." This work
shows that in any realistic pipeline, T is frequently *deleted before the model sees it* — by
truncation, by memory management, by a template that won't render, by a summarizer that paraphrases it
away — and that *which* triggers are lost is a systematic, predictable function of the
context-management strategy rather than of the model. In our grid, ~28% of triggers never reached the
model, split across three pipeline stages in patterns you can read straight off Table 1.

So the practical prescription is simple: **log the final model-visible input.** If your harness doesn't
record the exact token sequence the model was given, you are measuring an unknown mixture of delivery
and robustness, and you have no way to separate them. Run the delivery audit first — it's cheap, no
weights, deterministic, a million trials on a CPU. And only on the triggers that *survive* does the
interesting question — did the model actually activate? — even become well-posed. That question is
where we go next.

---

*Reproducibility: every table and figure regenerates from the repository's committed results with a
single analysis command; the small experiments (the cut sweep, the summarization cell) run locally in
seconds, and the full grid runs as a CPU-only cluster job. The findings document lists, for each claim,
the exact artifact that proves it. Figures shown: F0 (scaffolding), F1 (delivery heatmap), F3 (layer
funnel), F4 (outcome composition), F6 (anatomy of the cut), F8 (delivery flow); the paper additionally
includes F2 (delivery cliffs vs context length), F5 (trigger landing map), F7 (the wall of trials), and
the full statistical tables with every confidence interval.*
