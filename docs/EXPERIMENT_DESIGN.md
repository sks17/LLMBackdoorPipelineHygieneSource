# Experiment design & readiness — Project 1 (trigger delivery audit)

**What this is.** The single reviewable document that turns a validated pipeline into a ready-to-run
experiment: the data plan (synthetic generation + real corpora), the statistical design, the
parallelization/cluster plan, the open decisions, and the precise list of what I need from you to
finalize the runnable spec. It consolidates and points to the companions:
`PRE_REGISTRATION.md` (locked decisions), `DATA_CONTRACTS.md`, `CLUSTER_EXECUTION_PLAN.md`,
`REQUESTED_DOCUMENTATION.md`, and the `docs/tasks/*` specs.

---

## 1. Where we are

**Mechanism validation is complete.** The pipeline — insert canary → memory/trim policy (Layer 1→2)
→ chat template (Layer 3) → tokenize (Layer 4) → truncation → survival scoring — is built and
verified against ground truth across: no-op, head/tail truncation, keep-recent memory, staged
composition, token-level boundary corruption, manifest expansion, a second model + Gemma template
diversity, LangChain `trim_messages`, a LangChain RAG baseline, and the dataset-ingestion adapter.
161 tests, gate green.

**What is NOT built yet (this document's subject):**
1. the **data** — a synthetic base-conversation corpus at scale, and the real-corpus parsers;
2. the **fan-out** — the at-scale manifest + sharding + Slurm submission;
3. the **analysis** — aggregation and the statistical tests.

None of these is new *mechanism* work; they are the data-production and orchestration layers.

---

## 1a. Project-1 completeness — are we building the whole thing?

Short answer: **yes, and beyond it in several dimensions** — but there are four genuine Project-1
items still owed, so we track them here rather than accidentally ship a limited version. The
`FIRST_PROMPT` scaffolding task described a deliberately minimal "first version"; the target is the
**full Project 1** from `GENERIC_PLAN` §15 (final shape) + §4 (manipulated variables). Audit:

| Full-Project-1 element | Status | Note |
|------------------------|--------|------|
| Reusable audit harness | ✅ built | |
| Matrix: context length × position × policy | ✅ built | grid + manifest |
| Survival metrics: exact / token / partial / boundary | ✅ built | |
| Final-prompt logger | ✅ built | |
| Truncation: none / head / tail / **middle** | ✅ built | `MiddleTruncation` exists |
| Memory: keep-recent | ✅ built | |
| Staged composition (policy ordering) | ✅ built | |
| Chat-template diversity | ✅ built | Qwen(ChatML) / Gemma / TinyLlama(Zephyr) / Pythia(base) — **exceeds** first prompt |
| Model diversity | ✅ built | 5-model suite |
| Counterfactual pairing | ✅ built | **exceeds** — validity improvement |
| RAG baseline (retrieve → pack → deliver) | ✅ built | **exceeds** first prompt (RAG wasn't in it) |
| LangChain comparison (trim + RAG) | ✅ built | **exceeds** first prompt |
| Two data arms (synthetic + existing_dataset) | ◐ real-parser stubs await formats | dataset_adapter built |
| **Synthetic conversation corpus** | 🔨 building now | the generator |
| **Trigger library**: random / multi-token / boundary | ◐ partial | **missing** `natural_phrase`, `unicode` (split = later multi-turn) |
| **Positions**: `system`, `tool_output` | ◐ missing | `system` → role-migration (needs scorer role-tracking); `tool_output` → needs agent/tool family |
| **RAG chunk-boundary corruption** | ◐ planned | advisor's trial 10; in scope for a complete Project 1 |
| **Analysis / finding tables** | ◐ partial | have basic `aggregate_survival`; **missing** McNemar's, delivery-vs-robustness, failure-explanation tables — *the actual scientific output* |
| **Activation / generation phase** | ◐ stubbed | the `P(activation \| delivered)` decomposition GENERIC_PLAN calls "where the science starts" — *secondary/optional* per the plan |
| summarize_old / summary+recent | ⏸ **deferred by decision** | needs a semantic scorer; pre-registered, not an omission |

**The four things still owed for the *entire* Project 1** (added to the §7 roadmap, not dropped):
1. **Analysis / finding-tables layer** — *essential*; it is the output of the delivery audit (survival-rate tables by policy × position × model, McNemar's on counterfactual pairs, the "how often is backdoor-failure actually delivery-failure" tables). Currently only a basic per-cell rate aggregation exists.
2. **Activation / generation phase** — the secondary `P(activation | delivered)` layer. GENERIC_PLAN is explicit this is optional and that delivery-audit-alone is "enough for a strong first project," but a stratified generation subset completes the headline decomposition. Include as the secondary layer.
3. **Extra trigger types + positions** — `natural_phrase`, `unicode`; `system` (role migration) and `tool_output` (agent/tool family). Cheap completeness additions.
4. **RAG chunk-boundary corruption** — a distinct mechanism (chunking splits vs truncation splits); in scope for a complete Project 1. (RAG reranker/compression depth is Project-1-later / Project-2.)

**Verdict:** the mechanism harness is complete at full Project-1 fidelity and exceeds the minimal
first version. The remaining work is data production + the analysis/activation *outputs* + a few
completeness additions — all now explicitly on the roadmap.

---

## 2. The experiment, restated

**Question.** When a harmless canary is placed in raw input, does it survive the real context
pipeline into the final model-visible tokens — measured across models, policies, positions, context
lengths, trigger types, and data sources? (Delivery audit, **not** activation/backdoor detection —
that is Project 2+.)

**Survival is scored** at exact and partial (boundary) granularity via the four logged layers.
**Semantic survival is out of scope** this pass (see §8).

**Hypotheses (pre-registered):**
- **H1** — pipeline policy significantly affects delivery rate.
- **H2** — delivery is model-invariant at the message stage; token/template stages may vary (priors: message-stage invariant; Gemma's strict-alternation template *rejects* some sequences).
- **H3** — trigger position × policy interaction (head truncation kills prefix, spares end; keep-recent drops old turns, keeps recent).
- **H4** — synthetic and real bases behave alike at matched token length and policy.
- **Counterfactual validity** — trigger-absent twins deliver ≈ 0; paired McNemar's on with/without pairs sharing a `base_id`.

---

## 3. Data plan

Two arms. **Both** get canaries inserted by the *same* deterministic post-step (the slot
mechanism) — this identical insertion is what keeps `data_source` from confounding trigger
presence, and is the single most important validity property of the whole design.

### 3a. Synthetic arm — the conversation generator (the missing data source)

The generator is a **structured data-production pipeline**, not "ask an LLM for realistic chats."
The discipline:

- **Sample a structured seed**, then generate from it: `domain × conversation_type × persona × goal × tone × target_length × slot_positions × difficulty × distractor_topic`. This gives controlled, reproducible variation instead of an LLM's default distribution.
- **The generator places named slots only** (`{{PREFIX_SLOT}}`, `{{OLD_TURN_SLOT}}`, `{{RECENT_TURN_SLOT}}`, …) and **never** writes trigger text. Our code inserts canaries afterward (the same `TriggerInserter` used for real data). This is non-negotiable for H4.
- **Strict JSON output** validated against `BaseConversation`; reject on invalid JSON, missing/dup slots, mention of "trigger/canary/backdoor", too-short/nonsense content, malformed roles, or pre-existing trigger-like strings.
- **Length matching** reuses the built `dataset_adapter.to_base_conversation` (tokenize with the target model's tokenizer; append structured, non-lorem filler at section boundaries if short; cut at boundaries if long) so a base lands in a grid context bin (1k/4k/8k/16k/32k).
- **Conversation families** (per GENERIC_PLAN): `single_turn_long_document`, `multi_turn_chat`, `agent_tool`, (`rag_like` for Wave 2). Delivery audit leans on the first two.

**Generation models — mix them, on purpose.** You plan to use local Qwen (0.6/1.7/4/8B), fast API
models (Haiku), and other fast local models. Do it, and **record `generation_model` as a covariate
on every base**. Rationale:
- If *all* synthetic bases come from one model, the "synthetic" arm becomes that model's writing
  style, which can confound H4 (synthetic-vs-real would partly measure one model's style vs real
  humans). Mixing generators diversifies content style and lets us check that survival is invariant
  to the generator (a nice secondary robustness check).
- The **generation model is orthogonal to the model-under-test**: it shapes *content/style* (Layer
  1), while the model-under-test shapes *templating/tokenization* (Layers 3–4). So generate with
  whatever is cheap and diverse; audit with the fixed 5-model suite.
- Practical split: bulk generation on the **local Qwen sizes** (cheap, private, cluster-resident) +
  a slice from **Haiku / a stronger fast model** for higher-quality long-document and agent-tool
  conversations where small models struggle. Keep a fixed sampling seed per shard for
  reproducibility.

**Counterfactual pairing** is already wired (Task 08): each base is expanded with a trigger-present
row and its trigger-absent twin.

### 3b. Existing-dataset arm — real corpora (H4)

Selected (locked in pre-reg): **LMSYS-Chat-1M + WildChat** (real user–LLM multi-turn traffic — the
exact "does synthetic match real" comparison H4 needs) **+ one long-document corpus** (for the
16k/32k cells the chat sets rarely reach). **Deferred:** all safety/red-team/jailbreak sets
(HarmBench, JailbreakBench, SORRY-Bench, HH-RLHF, PAIR) → Project 2+ (activation detection); their
distinctive prompt style would confound `data_source`. **UltraChat**, if used at all, is a
*synthetic-baseline* label, never the "real" arm (it is LLM-generated).

The `dataset_adapter` is **built and structurally validated**; the per-source parsers
(`LMSYSParser`, `WildChatParser`, `LongDocParser`) are honest `NotImplementedError` stubs
(REQUESTED_DOCUMENTATION items 14–16) **blocked on the real record formats + licenses** — see §6.

### 3c. Data integrity / confound controls

- **Insertion symmetry** — identical deterministic slot-fill on synthetic and real bases.
- **Don't let the trigger distribution be too clean** — canaries are inserted on *both* arms (and
  the counterfactual gives trigger-absent rows on both), so `data_source` never correlates with
  trigger presence or insertion method.
- **Generation-model style** — recorded as a covariate; mixed generators avoid single-style bias.
- **A small hand-written gold set** (~200–500 bases) is worth keeping aside to detect
  dataset-artifact effects (does survival track content or an artifact of one generator?).

---

## 4. Statistical design

### The grid

| Axis | Levels (Wave 1) | Notes |
|------|-----------------|-------|
| model | Qwen3-0.6B/1.7B/4B/8B, Pythia-1B | Pythia = base-completion path, 2048-capped |
| pipeline_policy | none, head_truncation, tail_truncation, keep_recent_messages | + `rag_baseline` in Wave 2 |
| trigger_position | prefix, middle, end, near_boundary (single-turn); old_turn, recent_turn (multi-turn) | base-family-appropriate |
| context_length | 1k, 4k, 8k, 16k, 32k | model-capped (Pythia → 1k/2k only) |
| trigger_type | rand_001, multi_001, boundary_001 | random / multi-token / boundary |
| data_source | synthetic (Wave 1), existing_dataset (Wave 2) | H4 |
| trigger_present | True + False (counterfactual pair) | McNemar's |
| generation_model | (synthetic only) recorded covariate | Qwen sizes / Haiku / other |

**Controlled/fixed:** tokenizer+template version, `enable_thinking=False`, decoding settings
(generation off), seed, per-model chat template.

### Tests

- **H1/H3** — logistic regression of `final_token_trigger_present` on policy × position (+ length, model), or χ² per cell. Effects are already large in Trials 1–3.
- **H2** — does `survival_class` depend on `model` within (policy, position, length)? Priors say message-stage invariant; watch template-stage divergence.
- **H4** — `data_source` main effect + interactions on **length-matched** cells only.
- **Counterfactual** — paired **McNemar's** on with/without-trigger rows sharing a `base_id`; trigger-absent delivery must be ≈ 0 (scoring sanity).

### Sizing / power

n per cell = number of base conversations feeding it. The paired (counterfactual) design gives
strong power at modest base counts. Suggested:
- **Cluster smoke / pilot:** ~20–50 synthetic bases per family → validate the end-to-end
  shard→Slurm→results→aggregate loop cheaply.
- **Full run:** ~100–300 bases per family (GENERIC_PLAN pilot scale) → publishable power.

Most rows are **pipeline-only** (no model generation) → CPU-cheap and embarrassingly parallel.
Generation (activation) is a later, *stratified* GPU subset, not the whole grid.

---

## 5. Parallelization & cluster

- **Unit of work:** one trial = one manifest row; many rows = one shard; one worker = one shard. **Do not** submit one job per trial. Shard **by model** for the (later) generation phase so each GPU worker loads its model once.
- **Slurm job arrays** on **Hyak**: the survival (pipeline-only) phase is **CPU-only** → fits **Klone** idle/checkpoint partitions cheaply; the optional generation phase needs **Tillicum** GPUs (≥1 GPU/job, billed). Conda module; project storage (`/gscratch/<group>` on Klone, `/gpfs/projects/<group>` on Tillicum), not home (10 GB). Template at `scripts/slurm/`.
- **Outputs:** JSONL now → **Parquet** for the scaled survival/generation results; save final-prompt text for a deterministic **sample** (`--log-prompts`) so any result is traceable.
- **Determinism/reproducibility:** pin `transformers`/`tokenizers` versions (token boundaries drive truncation), fix all seeds (generation sampling + pipeline), record versions in each result.

---

## 6. Open decisions & exactly what I need from you

**Decided (on record):** the 5-model suite; the 5-policy grid with summarization deferred; datasets
selected (LMSYS + WildChat + long-doc); counterfactual pairing; exact+partial scoring only.

**Still needed to finalize the runnable spec — the precise "please provide" list:**

| # | What I need | Why it blocks | Form it should take |
|---|-------------|---------------|---------------------|
| A | **LMSYS-Chat-1M + WildChat: one sample record each + the HF usage-agreement/license page** | The `dataset_adapter` parsers are stubs; can't ingest real data without the true JSON shape and confirmed license | a pasted sample JSON record + the license URL/text per source |
| B | **Long-document corpus choice** (which corpus) + its record format + license | The 16k/32k real cells have no source without it | corpus name + one sample record + license |
| C | **Generation-model access** for the synthetic corpus | The generator can't run without endpoints | which local models (Qwen sizes, via transformers/ollama), which API models (Haiku — API key/quota?), and where generation runs (local box vs cluster GPU) |
| D | **Cluster specifics** | Can't wire Slurm without them | Klone vs Tillicum; account; partition/QoS; storage path; max walltime; GPU type (if generation) |
| E | **Scale target** | Fixes n and shard count | # synthetic bases per family for the pilot vs full; shard size |
| F | **Output format + retention** | Fixes the writer + storage budget | JSONL vs Parquet for results; keep all final prompts or a sample fraction |
| G | **Generation (activation) in scope for the first run?** | Determines whether Wave 1 is pipeline-only | yes/no; if yes, the harmless activation behavior (e.g. emit `CANARY_SEEN`) and the GPU budget |
| H | **Trigger set confirmation** | Locks the trigger axis | confirm {random, multi-token, boundary}; add unicode/natural-phrase now or later? |
| I | **Institutional/IRB + licensing constraints** | Compliance before a real run | any PI/IRB constraints on synthetic conversations or dataset redistribution |

I can start building the **synthetic generator** and the **fan-out config** immediately with only
**C** (generation access) and **E** (scale). **A/B** unblock the H4 real arm. **D/F/G** are needed
before the actual cluster submission.

---

## 7. Path to ready-to-run (ordered)

1. **Conversation generator** — structured-seed → mixed-model generation → validate → slotted `BaseConversation` corpus (reuses `dataset_adapter.to_base_conversation` for length-matching). *Blocks on C, E.*
2. **Finalize the 5 model configs** — Qwen3 sizes + Pythia (`chat_format=base`), verified context windows/tokenizer ids.
3. **Wave-1 experiment config + manifest** — the grid above with `include_counterfactual=True`; `expand_manifest` → shards. *Uses the built manifest layer.*
4. **Slurm submission + smoke run** — a small pilot end-to-end on the cluster (shard → worker → results → aggregate), then scale.
5. **Aggregation + analysis** — the survival-rate tables (policy × position × model), McNemar's on counterfactual pairs, logistic regression for H1/H3, the H2/H4 comparisons; figures + failure-example dumps.
6. **Wave 2** — fill the `dataset_adapter` real parsers (on A/B) → H4 real arm; RAG at scale; the LangChain `trim_messages` comparison axis; RAG chunk-boundary depth.

---

## 8. Risks & things to keep in view

- **Confounds:** `data_source` × trigger presence (controlled by symmetric insertion + counterfactual); synthetic writing style (controlled by mixed generators + generation_model covariate + the gold set).
- **Tokenizer parity** (a real finding, not hypothetical): LangChain's trimmer/splitter count in *their* tokenizer's units; survival is scored in the *model's* tokenizer. An offset that looks split "on paper" may not land where predicted. We already made trigger localization tokenizer-agnostic (char-offset), but any LangChain trim/RAG condition must point the framework at our tokenizer or record the mismatch.
- **Semantic survival deferred:** `summarize_old`/`summary+recent` are excluded until a fuzzy/paraphrase scorer exists (our current summarizer is a placeholder stub; running it now would emit misleading "always deleted"). This is locked in the pre-reg — do not re-add without the scorer.
- **Model context caps:** Pythia-1B is 2048; its high-length cells are (correctly) not emitted. Any base model added later needs the same base-completion path + cap.
- **Generation cost:** the activation phase is the only expensive part; keep it a stratified subset (controls + delivered/not-delivered + boundary cases), never the whole grid.
- **Boundary/partial finding:** `boundary_corruption` (truncation cut) vs `partial_survival` (other mechanisms, e.g. a future RAG chunk boundary) is a fixed convention in `DATA_CONTRACTS.md`; keep it.

---

## 9. My recommendation

Start the **synthetic generator** now (it's the critical-path data source and maps directly onto
your local-Qwen + Haiku plan), in parallel with you sending the **dataset formats/licenses (A/B)**
so the real arm comes online in the same wave. With the generator + the built manifest layer + the
model configs, the Wave-1 synthetic cluster run is a short hop away — and it is fully statistically
specified above, needing only your **C/D/E/F/G** answers to become a literal, submittable job.
