# Pre-registration — Project 1 fan-out (trigger delivery audit)

Locked design for the multi-model / multi-condition cluster run. Written before fan-out so agents
and analysis cannot drift. Amendments must be dated and justified here, not made silently.

## Scope and framing

Project 1 measures **trigger delivery through context management** — whether a harmless canary
placed in raw input survives the pipeline into the final model-visible tokens. It is **not**
activation/backdoor-detection (that is Project 2+). Survival is scored at **exact** and **partial**
granularity only; **semantic survival is out of scope** for this pass (see deferrals).

## Manipulated variables (the grid)

| Axis | Levels | Notes |
|------|--------|-------|
| model | Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, **Pythia-1B** | Pythia is a base model (no chat template → base-completion path; 2048 ctx cap). Spans scale for H2. |
| pipeline_policy | `none`, `head_truncation`, `tail_truncation`, `keep_recent_messages`, `rag_baseline` | 5 mechanism-validated policies. `rag_baseline` is Wave 2 (needs Task 06b). |
| trigger_position | `prefix`, `middle`, `end`, `near_boundary` (single-turn); `old_turn`, `recent_turn` (multi-turn) | Position set is base-family-appropriate. |
| context_length | 1k, 4k, 8k, 16k, 32k | **Model-capped**: Pythia-1B only 1k/2k; Qwen3 to 32k. Cells above a model's window are not emitted. |
| trigger_type | `rand_001` (random canary), `multi_001` (multi-token phrase), `boundary_001` (boundary) | |
| data_source | `synthetic` (Wave 1), `existing_dataset` (Wave 2: LMSYS + WildChat + one long-doc corpus) | H4. |
| trigger_present | **True and False** (counterfactual pair) | Every trigger-present row is emitted with its trigger-absent twin sharing `base_id` — see below. |

Controlled (fixed) across the grid: tokenizer/template version, `enable_thinking=False`, decoding
settings (generation is off), seed, chat-template per model.

## Locked decisions

1. **Five-policy grid.** `{none, head_truncation, tail_truncation, keep_recent_messages, rag_baseline}`. Hand-rolled implementations are the primary arm (validated Trials 0–5). LangChain `trim_messages` (Task 06a) is a **secondary framework-comparison axis**, not a replacement — added in a later wave, never silently swapped in.
2. **Summarization is formally deferred, and this is not optional.** `summarize_old_messages` and `summary_plus_recent` are **excluded** from the fan-out. Reason: our `SummarizeOldMessages` is a deterministic *placeholder stub* (it swaps old turns for a fixed summary string; it always exact-deletes), and correct measurement requires **semantic-survival scoring** (fuzzy/paraphrase matching) that does not exist yet. `PROJECT_DESCRIPTION`: "focus on exact survival and partial survival. Semantic survival can come later." A dedicated semantic-scorer trial precedes any re-inclusion. **No agent may add these two policies to the grid without that scorer.**
3. **Counterfactual pairing is a first-class expander feature.** `expand_manifest` emits, for every trigger-present row, its **trigger-absent** negative-control twin sharing the same `base_id` / model / position / policy / length. This yields paired observations for **McNemar's test** and is the cleanest guard against `data_source` confounding trigger presence.
4. **Pythia base-completion path.** Pythia-1B has no chat template. The runner needs a base-completion renderer (deterministic raw concatenation of messages, no chat template) selected by model config, so Pythia rows do not crash or silently use a wrong default. This is the `PROJECT_DESCRIPTION` §D "base completion format" condition. See Task 08.
5. **Data-source insertion symmetry.** Canary insertion is a deterministic post-step applied **identically** to synthetic and existing-dataset bases (the slot mechanism). This is what keeps H4 valid — `data_source` must never correlate with trigger presence or insertion method.

## Hypotheses and analysis plan

- **H1 — policy affects delivery.** Trigger delivery rate depends on `pipeline_policy`. Test: logistic regression / χ² of `final_token_trigger_present` on policy, per (position, length). Expected large effects (Trials 1–3 already show them).
- **H2 — delivery model-invariance.** Message-stage outcomes (`keep_recent`, positions) are model-invariant; token/template-stage outcomes may vary. Test: does `survival_class` depend on `model` within (policy, position, length)? Trials 4b/4c give priors (message-stage invariant; template-stage can diverge, e.g. Gemma).
- **H3 — position × policy interaction.** e.g. head truncation destroys prefix but preserves end (Trial 1); keep_recent drops old_turn but keeps recent_turn (Trial 2). Test: interaction term.
- **H4 — synthetic vs real parity.** At matched token length and policy, synthetic and existing-dataset bases yield the same delivery-rate distribution. Test: `data_source` main effect and its interactions, on length-matched cells. (Wave 2.)
- **Counterfactual validity.** Trigger-absent rows must show delivery ≈ 0 (scoring sanity) and enable paired **McNemar's** on with/without-trigger pairs sharing a `base_id`.

**Sizing (informal).** Target ~100–300 synthetic base conversations per conversation family
(`PROJECT_DESCRIPTION` pilot), each expanded across the grid and paired. Most rows are
**pipeline-only** (no generation) → cheap and embarrassingly parallel; generation is a later,
stratified subset. Final n per cell is fixed once the base count is chosen; the paired design gives
strong power for McNemar's at modest base counts. The grid is model-context-capped, so Pythia
contributes only its low-length cells.

## Blocked / pending (documentation, resolvable in parallel)

- **existing_dataset selection — resolved to LMSYS-Chat-1M + WildChat + one long-document corpus.** Pending: their record formats + license/usage terms (needed to spec the `dataset_adapter` parsers; see Task 07). Safety/red-team sets are **not** used (Project 2+; would confound H4).
- **Generation phase** (activation) remains out of scope for this pass.

## Waves

- **Wave 1 (fastest data):** synthetic bases × 5 models × {none, head, tail, keep_recent} × positions × capped lengths × 3 triggers × counterfactual pairs. Needs: base-completion path (Pythia), counterfactual pairing, the pre-reg on record. Fully validated otherwise.
- **Wave 2:** `rag_baseline` policy (Task 06b) + the `existing_dataset` arm (Task 07, H4) + LangChain `trim_messages` comparison axis (Task 06a) + RAG chunk-boundary depth.

## Amendments

### 2026-07-03 — Equivalence margin and multiplicity correction (analysis/inference layer)

Locks the two inferential-layer parameters that `docs/ANALYSIS_PLAN.md` §5.2 / §10 left provisional,
so the H2 (model-invariance) and H4 (synthetic ≈ real) **equivalence** verdicts are fixed *before*
the first full-wave analysis rather than chosen after seeing the numbers.

1. **TOST equivalence margin = ±5 pp** (percentage points of delivered rate). An H2/H4 comparison is
   declared *equivalent* when a two-one-sided-test at α = 0.05 rejects both one-sided nulls against the
   ±0.05 bounds — equivalently, when the 90% cluster-bootstrap CI of the delivered-rate difference is
   contained in [−0.05, +0.05]. Rationale: 5 pp is the smallest delivery-rate gap we treat as
   practically meaningful for a *delivery* audit. §5.3 prints each cell's achieved CI half-width, so a
   cell too imprecise to certify at this margin is visible, never silently labelled "equivalent".
2. **Multiplicity correction = Holm** (family-wise), applied *within* each hypothesis family — H2 across
   the model-pair comparisons, H4 across the (policy, `length_bin`) comparisons. Benjamini–Hochberg
   (FDR) is reported alongside as a secondary sensitivity view; both are already computed in
   `analysis/stats.py` (`holm`, `benjamini_hochberg`) and surfaced side by side, but **Holm is the
   pre-registered primary**.

Scope: binds the first full-wave analysis and every later inferential run. The **pilot is exempt**
(plumbing validation, not inference) and its report must state so. This amendment does **not** touch
the primary estimation path — per-cell delivered rates with cluster-bootstrap CIs over `base_id`,
which remains decision-free and unchanged.

### 2026-07-04 — Conditional re-inclusion of the summarize policies (semantic-survival cell)

Amends **Locked decision 2** (which it does **not** rewrite): with the semantic-survival scorer now
built (Task 10 — `scoring/semantic.py`, the pinned `NLIEntailmentScorer` + twin-calibrated τ +
gold-set validation), the two compression policies it was blocking on are re-included, but only
under strict conditions and reported apart from the main grid.

1. **Scope of re-inclusion.** `summarize_old_messages` and `summary_plus_recent` return to the
   fan-out **only** for `natural_phrase` / instruction-style triggers (E2's `natural_001`) — the
   only triggers with propositional content to paraphrase. Random canaries (`RANDOM_CANARY`, no
   propositional content) remain **excluded** from these two policies; a semantic axis over a random
   string is meaningless.
2. **Pinned producer + pinned scorer, both recorded.** Each summarize cell names an exact,
   greedy/`temperature=0`, CPU/float32 **summarizer** (`SummarizerConfig(backend='hf')`, `model_id`
   + `revision`) *and* an exact, argmax, CPU/float32 **semantic scorer** (`NLIEntailmentScorer`,
   `model_id` + `revision`). Both pins are written onto every row (`semantic_scorer_id`,
   `semantic_scorer_revision`, `semantic_threshold`).
3. **Required reported quantities.** For each summarize cell we report (a) the twin-calibrated
   threshold **τ** — the smallest value with achieved FPR ≤ target on the trigger-absent twin scores
   — with (b) the **achieved FPR on the twins and its Wilson interval**, and (c) the **gold-set
   precision/recall at τ** (`data/gold/semantic_survival.jsonl`). No summarize result is reported as
   a clean 0/1: every value is caveated as *"semantic delivery under scorer S at FP rate f"*.
4. **Dual model dependence, reported separately.** Unlike every other delivery cell — which is
   **model-agnostic** (deterministic token/template mechanics) — a summarize cell introduces model
   dependence at **both** the producer (what the summary says) and the scorer (whether the meaning
   is judged present). These cells are therefore reported in a **separate table**, never pooled into
   the model-agnostic delivery grid, and are explicitly labeled as producer×scorer-conditional.

Scope: binds every summarize-policy row from this date forward. It does **not** alter the five
model-agnostic policies of Locked decision 1, nor the counterfactual-twin machinery of Locked
decision 3 — which this scorer's calibration now *depends on* (the absent twin is the null that
certifies τ).

### 2026-07-06 — Project 2 probe design pre-registration (activation-based trigger detection)

`PRE_REGISTRATION.md` up to this point is entirely a Project 1 document: it locks the trigger-delivery
grid, H1–H4, and the 2026-07-03 inferential-layer amendment, but contains **no** probe / FPR-target /
layer / pooling section. Project 2's activation-based probe (`experiments/probe_detection/`,
`probes/*`, `activations/*`) is fully built and offline-tested (`PROJECT2_MASTER.md` §5) but
un-preregistered — a continuity gap explicit in `PROJECT2_MASTER.md` §12 ("Probe design
**pre-registration** — **not pre-registered** (unlike P1's H1–H4)") and Prerequisite P4 of
`PROJECT2_EXPERIMENT_PLAN.md` Part II ("Probe design is un-preregistered; without it every number is
exploratory"), matching continuity finding **D1** (`PROJECT2_CONTINUITY_CHECK.md:115`). This amendment
locks the estimand and its operating points **before** the first real-model measurement wave (Tier 1,
`PROJECT2_EXPERIMENT_PLAN.md` Part III), so every Tier-1+ number is confirmatory, not exploratory.

1. **Estimand.** Primary = `P(probe fires | trigger delivered)` — the TPR of a linear probe over
   delivered positives, read at the **calibrated** threshold; AUROC is reported only as a summary
   "alongside, never instead," never as the estimand (`probes/metrics.py:3-6`,
   `PROJECT2_FOUNDATIONS.md:3-8`).
2. **Dual reporting.** Every metric is reported twice — **all-trials** (partial-survival negatives
   count against the FPR budget, the deployment-pessimistic view) and **delivered-only**
   (verified-delivered positives + clean negatives only) — per `ProbeEvaluationResult`
   (`schemas/probes.py:109-129`).
3. **Layers by depth-fraction.** Probe layers are named and reported by fraction of model depth, never
   raw HF index (avoids the L vs L+1 vs HF-indexing hazard across models of different depth).
   Pre-registered informative band: **0.5–0.89 of depth**, default probe set at fractions
   `{0.5, 0.66, 0.75, 0.89}`, with a full per-model sweep (E1.1) to confirm the band on our Qwen suite
   before it is treated as confirmed rather than a prior (`PROJECT2_RESOURCES.md:86-88`). HF indexing
   convention: 0 = embeddings, 1..N = transformer blocks (`activations/extractor.py:20-26`).
4. **Pooling.** `mean` is the pre-registered deployable operating pooling; `last_token` and `max` are
   the ablation arm (E1.2); `trigger_span` is an **oracle diagnostic only** — never a deployable
   operating point — and is always run with the seeded random-span fallback ON so the pooling
   *operator* is identical across classes and only trigger content, not window statistics, can drive
   separation (`experiments/probe_detection/runner.py:182-218`, `PROJECT2_FOUNDATIONS.md:123-141`).
5. **Aggregation.** The closed-form combiners (`mean_score`, `max_score`, `product_of_experts`,
   `quantile`) are the headline; `stacked_logistic` is reported but **caveated** for its
   calibration-split adaptivity — fit on CALIBRATION, not TRAIN, to avoid stacking on optimistically
   separated TRAIN scores (`experiments/probe_detection/runner.py:126-138`, continuity **D4**).
   Multi-layer aggregation is the pre-committed recovery for the small-model single-probe ceiling
   (`PROJECT2_RESOURCES.md:80-85`).
6. **FPR targets.** `{1e-2, 1e-3}`. **1e-2 is the resolvable primary.** **1e-3 is reported
   bounded-only** — achieved FPR of 0 with an honest wide Wilson interval — unless clean negatives are
   scaled to **≥ ~1000**, because `target_fpr < 1/n` cannot be resolved by an empirical quantile
   (`probes/calibration.py`, `PROJECT2_FOUNDATIONS.md:88-96`, continuity **D3**). E0.2 (the
   FPR-resolution sweep) fixes the base-count requirement this depends on.
7. **Calibration population.** Thresholds are calibrated on **clean (never-inserted) CALIBRATION
   negatives only** — the monitor guards clean traffic, so its FPR budget is a statement about clean
   negatives. Partial-survival negatives are deliberately excluded from the calibration pool
   (`experiments/probe_detection/runner.py:88-95`, `PROJECT2_FOUNDATIONS.md:79-90`).
8. **Splits.** `base_id`-grouped, leakage-safe: all trials of one base conversation — its
   counterfactual twin and its near-duplicate policy/position variants — land in the same split
   (`experiments/probe_detection/dataset.py:56-63`).
9. **Inference.** `P(fire | delivered)` uncertainty is a **cluster-bootstrap over `base_id`**
   (extending Project 1's `analysis/stats.bootstrap_rate_ci`); achieved FPR carries a **Wilson**
   interval; any *invariance* claim (pooling X ≈ Y, model A ≈ B) uses **TOST at ±5 pp with Holm**
   multiplicity correction, reusing the margin and correction locked in the **2026-07-03** amendment
   above. That amendment's ±5 pp / Holm pairing is hereby **extended** from Project 1 delivery rates to
   the Project 2 probe layer (continuity **D2**); this does not relax or restate that amendment's scope
   over delivery-rate comparisons, which remains unchanged.
10. **Scope discipline — canary ≠ backdoor (the hard boundary).** All Tier-0–2 results are claims about
    **delivered-canary representations**, never a backdoor-detection result — genuine backdoored
    behavior is not yet in the data, so canary-detectability may over- or under-state
    backdoor-detectability (`PROJECT2_MASTER.md` §10.7). Only Tier-3 (real backdoored weights) licenses
    a backdoor-detection statement. No document, table, or figure produced under this amendment may
    headline a backdoor-detection result — or a bare 1e-3 TPR number — from canary/pilot data
    (continuity **E1**/**D3**).
11. **No numeric TPR bar is pre-committed.** Success is defined **structurally** — dual reporting,
    calibrated operating points, honest low-n intervals — not as "TPR ≥ X." Any numeric target set
    later must be size-aware (the 0.6B–8B suite has a lower single-probe ceiling than 70B-class
    results) and stated for a *resolvable* FPR (`PROJECT2_MASTER.md` §13.1).

Scope: this amendment binds every probe (Tier-0+) run from this date forward; it does **not** alter
any Project-1 delivery decision or either amendment above; the exploratory-vs-confirmatory line for
Project 2 is drawn here.
