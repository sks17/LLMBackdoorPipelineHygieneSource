# Guide: writing the Project-1 blog post (LessWrong)

A working guide for turning the findings into a LessWrong post. **This is not the post** — it is the
scaffold: the argument arc, section-by-section guidance, which figure/table goes where, and the
tone. The precise, hedged, numbers-first version lives in the **paper**; the blog's job is the
*clear natural-language explanation of the same findings*.

Source material: [`docs/FINDINGS.md`](FINDINGS.md) (every claim + justification + the exact numbers)
and the artifacts in [`outputs/analysis/project1/`](../outputs/analysis/project1) (`figures/f*.png`,
`tables/t*.md`).

---

## Audience and register

- **Who:** LessWrong readers fluent in AI safety — they know what a backdoor, a trigger, and a
  context window are. Do **not** re-explain those. Do explain *our* specific mechanism (the
  four-layer delivery audit) because it is the novel framing.
- **Register:** clear natural language **beats** polished prose. Prefer "the trigger was deleted
  before the model saw it" over "an epistemically-fraught deletion event upstream of inference."
  Short sentences. One idea per paragraph. Define a term the first time, then reuse it.
- **What to keep vs. defer to the paper:** the blog carries the *story and the intuition* — why this
  matters, one clean number per claim, one figure per claim. The paper carries the full tables, the
  CIs, the TOST margins, the multiplicity correction, the reconciliation counts. **Rule of thumb:**
  if a sentence needs a confidence interval to be honest, put the point estimate in the blog and the
  interval in the paper — but never state a number in the blog the paper can't back.
- **Length:** aim ~2,000–3,000 words. It is a findings post, not a methods paper.

---

## The one-sentence thesis (put it in the first paragraph)

> Whether a trigger reaches the model at all is decided by ordinary context management, so a lot of
> what looks like a model *resisting* a backdoor is really the trigger never being *delivered* — and
> you can't tell the two apart unless you log the final prompt.

Everything in the post serves this sentence. If a paragraph doesn't advance it, cut it.

---

## Section-by-section arc

### 1. Motivation — the hook (no figure)
Open with the failure of the standard evaluation. A researcher inserts trigger T, the model doesn't
misbehave, they write "robust to T." **Ask the question they didn't:** did T actually reach the
model? In a real deployment the prompt goes through templating, memory trimming, truncation, maybe
RAG — any of which can silently delete T. Frame the whole post as: *before you can measure robustness,
prove delivery.* Name the safety stakes plainly: every un-instrumented backdoor eval is measuring a
mixture of delivery and robustness and can't separate them.

### 2. What we actually did — the four-layer audit → **Figure F0** (`f0_scaffolding`)
Explain the method in ~5 sentences: insert a harmless canary at a controlled spot, run it through a
real pipeline, log its presence at four layers (raw → post-memory → post-template → final tokens),
and check whether it survives to the tokens the model would receive. Stress the two things that make
this trustworthy and cheap: (a) **no model weights** — delivery is tokenizer/template/policy
mechanics, so it's deterministic and runs ~1M trials on CPU; (b) the **counterfactual twin** — every
trigger-present trial has a trigger-absent copy that must score "no survival," which is how we know a
"survival" isn't a coincidence. **Use F0 here** — it's the mental model for the whole post.

### 3. The core result — policy × position deletes triggers → **Figure F1** (`f1_delivery_heatmap`), table snippet from `t1_delivered_rate`
This is the heart. Show F1 (the heatmap). Walk the reader through the pattern in words: with no
context management everything is delivered (the top row is all 1.00 — that's the control that proves
the plumbing works); then head truncation wipes out prefix and old-turn triggers but keeps the end;
tail truncation is the mirror; keep-recent memory drops old turns. **Give one concrete number:** under
head truncation a prefix trigger is delivered ~21% of the time vs ~94% at the end. Land the
interpretation: this is not the model doing anything — it's deterministic deletion, and it's
*predictable* from the policy. Optionally add **F2** (`f2_delivery_cliffs`) if you want to show the
budget dependence, but F1 alone carries the section.

### 4. The reframe — "backdoor failure" is mostly *delivery* failure → table `t6_misattribution`
The money paragraph. Take the head-truncation × prefix cell: ~79% of trials are "apparent trigger
failures," and essentially all of that is delivery failure — the trigger was deleted upstream. State
it as the upper bound it is: an evaluator not checking delivery would miscredit up to that fraction to
"model robustness." Keep this tight — it's the sentence people will quote. (Full decomposition table
→ paper; blog gets the one number.)

### 5. Where it gets interesting — the model that breaks the pattern → **Figure F4** (`f4_outcome_composition`)
Most models agree cell-for-cell (delivery is tokenizer mechanics, not weights). Gemma is the
exception, and it's a clean AI-safety-relevant point: its chat template has no system role and demands
strict alternation, so (a) some memory-policy outputs **won't render at all** (`template_incompatible`
— the trigger dies before tokenization) and (b) a trigger planted in the *system* message **migrates
into the user turn** (`role_migration`). Use F4 to show the outcome bands differ for Gemma. The
takeaway: delivery is model-invariant *until the template structure differs* — so "we tested on model
X" doesn't transfer across template families.

### 6. Does this hold on real conversations? → **Figure F2 or a `t5_h4` snippet**
Readers will ask whether synthetic bases are cheating. Answer with H4: we ran the same grid on real
LMSYS and WildChat conversations. Synthetic and real deliver **alike except under memory policies**,
where real conversations (more, longer old turns) deliver less — and even that gap **closes at the
no-truncation budget**. Interpretation in one line: the synthetic-vs-real difference is an artifact of
*how much old content the memory policy discards*, not a fundamental content gap, so the audit
generalizes. One number: under keep-recent, real delivers ~30 pp less than synthetic at a tight
budget, ~0 pp at a loose one.

### 7. Two mechanisms worth a paragraph each (optional figures)
- **Boundary corruption is position-invariant** → **Figure F6** (`f6_cut_anatomy`). When a cut lands
  *inside* the trigger, its back half becomes the literal start of the prompt — a half-survival. The
  neat result: this depends only on where the cut falls relative to the trigger, identically wherever
  the trigger sits. Nice, crisp, visual.
- **Semantic survival** (no figure needed; a 3-row table). When old turns are *summarized* instead of
  dropped, the trigger can survive as **meaning** even when its exact words don't — the one delivery
  mode string-matching can't see. Explain the honesty knob plainly: a paraphrase scorer has false
  positives, so we calibrate its threshold on the trigger-absent twins (zero false positives) and
  never report a clean 0/1. This is a good place to preview Project 2 (this needs a real model, so it
  bridges to activation).

### 8. Scope and honesty (short, but do not skip — this audience rewards it)
State the boundaries in plain terms: this measures **delivery, not activation** (Project 2 runs the
model); outcomes are deterministic so the "uncertainty" is generalization across conversations, not
noise; the semantic cell depends on the scorer and is reported separately. Naming limits *increases*
credibility with this audience.

### 9. Takeaway (no figure)
Close on the actionable version of the thesis: **instrument the last layer.** If your backdoor eval
doesn't log the final model-visible tokens, you're measuring a mix of delivery and robustness. The
delivery audit is the necessary first stage; only on delivered triggers is "did it activate?"
meaningful. End by pointing to the repo (reproducible) and the paper (precise).

---

## Figure/table placement cheat-sheet

| Where | Asset | Why |
|---|---|---|
| §2 method | **F0** `f0_scaffolding` | the mental model of the audit |
| §3 core result | **F1** `f1_delivery_heatmap` (+ optional F2) | the policy×position deletion pattern |
| §4 reframe | `t6_misattribution` (one number) | the headline safety claim |
| §5 Gemma | **F4** `f4_outcome_composition` | the model that breaks invariance |
| §6 real data | `t5_h4_synthetic_vs_real` (one row) | H4 generalization |
| §7 boundary | **F6** `f6_cut_anatomy` | position-invariant half-survival |
| §7 semantic | 3-row summarizer table | meaning-level survival |
| appendix/paper only | **F3** funnel, **F5** landing map, **F7** wall, **F8** Sankey; all full tables | detail the blog doesn't need |

**Figure hygiene for the post:** use the `.png` for embedding, keep the `.svg` for the paper; every
figure needs a one-line caption in plain English stating *what to look at* ("top row all blue = the
control; the dark cells are where the policy deletes the trigger"), a note that rates condition on
trigger-present rows, and the per-panel n. Do not embed F5/F7 as-is — at ~1M trials they're rendered
on a labelled sample; say so in the caption or leave them to the paper.

---

## Things to get right (this audience will notice)

- **Don't overclaim.** "Delivery failure explains a large share of apparent robustness" — not "backdoor
  evals are all wrong." The upper-bound framing (§4) is the honest one.
- **Keep the control visible.** Mention the counterfactual twins early; it's *why* the reader should
  believe a "survival" is real. It also pre-empts the obvious "how do you know it's not noise?"
- **One number per claim, not five.** The blog persuades with the pattern + one clean figure; the paper
  proves it with the table. Resist dumping CIs into prose.
- **Name the limitation before the reader does** (delivery≠activation, deterministic, semantic cell is
  scorer-conditional). On LessWrong, stating your own scope boundaries is a credibility multiplier.
- **Link the repo and say it's reproducible** — point at `docs/REPLICATION.md`. This audience checks.
