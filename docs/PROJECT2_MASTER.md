# Project 2 — Master Reference: Activation-Based Trigger Detection

**Status.** This is the single consolidated reference for Project 2. It is a *synthesis* document:
it gathers every facet of Project 2 scattered across the prior descriptions
(`GENERIC_PLAN.md`), the two dedicated design docs (`PROJECT2_FOUNDATIONS.md`,
`PROJECT2_RESOURCES.md`), the coded foundation (`src/trigger_audit/{activations,probes}/`,
`experiments/probe_detection/`, `schemas/probes.py`), the cross-cutting docs
(`EXPERIMENT_DESIGN.md`, `PRE_REGISTRATION.md`, `SESSION_HANDOFF.md`, `DATA_CONTRACTS.md`,
`ONE_SHOT_PLAN.md`, `CLUSTER_EXECUTION_PLAN.md`, `ANALYSIS_PLAN.md`), and the vendored external
corpus in `third_party/`. It is the input to the experiment-design work
(`PROJECT2_EXPERIMENTS_REVIEW.md`), the continuity audit (`PROJECT2_CONTINUITY_CHECK.md`), and the
final experiment plan (`PROJECT2_EXPERIMENT_PLAN.md`).

Every load-bearing claim is cited to its source as `file:line`. Where sources disagree or a facet
is unbuilt, this document says so explicitly rather than papering over it — §4 and §12 collect those.

---

## 0. One paragraph

Project 1 proved *delivery*: whether a trigger inserted into raw input survives context management
into the final model-visible tokens (`P(delivered)`). Project 2 asks the follow-on question at the
level of the model's internal representation: **given that a trigger was delivered, does a linear
probe trained on the model's per-layer hidden-state activations fire — at a threshold calibrated to
a target false-positive rate?** The estimand is `P(probe fires | trigger delivered)` at deployment
operating points (FPR ≤ 1e-2, 1e-3), *not* AUROC, because "an AUROC says nothing about the threshold
a monitor actually runs at" (`PROJECT2_FOUNDATIONS.md:8`). The differentiator versus all prior public
probing work is that labels are **delivery-verified** — an example is positive only when Project 1's
`SurvivalResult.final_token_trigger_present` is true, never merely because a trigger was inserted
upstream (`PROJECT2_FOUNDATIONS.md:51-55`). A complete, offline-tested harness for this already exists;
what remains is real-model measurement and the experiments that would validate it.

---

## 1. Motivation and scientific framing

### 1.1 The lineage: why "conditional on delivery" is the whole point

`GENERIC_PLAN.md` — the Project-1 charter — never specifies a probe, but it establishes the
conditional logic Project 2 inherits and exists to complete:

- "The important metric is: `P(activation | trigger delivered)` not just `P(activation)` … if the
  trigger was removed by the pipeline, a non-activation result says nothing about model robustness"
  (`GENERIC_PLAN.md:76-80`).
- "The most important confusion to avoid is treating 'no activation' as 'the backdoor failed.' It may
  simply mean 'the pipeline deleted the trigger.'" (`GENERIC_PLAN.md:313`).
- "That third column [activation given delivery] is where the science starts. The misleading number
  is raw activation. The useful number is activation conditional on verified delivery."
  (`GENERIC_PLAN.md:419-420`).

The core scientific claim the two-project program makes is therefore: **much of what looks like
"backdoor-robustness failure" is actually delivery failure, and you must prove delivery before
attributing anything to the model** (`SESSION_HANDOFF.md:37-39`; `EXPERIMENT_OVERVIEW.md:18-19`).
Project 1 built the delivery proof; Project 2 supplies the "activation given delivery" measurement —
but with a stronger instrument than a behavioral string match (see §4).

### 1.2 Why a probe, and why calibrated to an FPR

The deployment framing is a **monitor**: a lightweight classifier on the residual stream that a
serving system could run to flag "this request contains an (active) backdoor trigger." A monitor is
defined by its *threshold*, and the threshold's contract is a false-positive budget, not a rank
statistic (`probes/calibration.py:3-6`; `probes/metrics.py:3-6`). Hence the estimand is TPR at a
calibrated FPR, and AUROC is reported only as a scalar summary "alongside, never instead"
(`probes/metrics.py:5-6`).

### 1.3 Why activations rather than behavior

The behavioral alternative — make the model emit a canary string and string-match it — was Project
1's *optional* dependent variable (`GENERIC_PLAN.md:58-59, 296-300`). A representational probe is a
different, stronger object for three reasons that the source material makes explicit:
1. It can fire even when the behavior does not surface in the sampled output (the trigger is
   *present in the representation* but the harmless canary emission is suppressed or truncated).
2. It targets *genuinely backdoored checkpoints* whose behavior is adversarially hidden, which is the
   eventual measurement target (`PROJECT2_FOUNDATIONS.md:149-150`).
3. It is the object the external literature actually studies (deception/sleeper-agent probes), giving
   us method priors and adversarial stress tests (`PROJECT2_RESOURCES.md:63-95`).

---

## 2. Relationship to Project 1 (the data contract Project 2 stands on)

Project 2 consumes Project 1 outputs only; it adds no new pipeline. The exact fields it depends on,
each verified against `schemas/probes.py`, `experiments/probe_detection/dataset.py`, and the
Project-1 data contract:

| Project-1 field (source) | Project-2 use |
|---|---|
| `SurvivalResult.final_token_trigger_present` (`DATA_CONTRACTS.md:67`) | **The probe label** — delivery-verified positive/negative (`dataset.py:34`). |
| `SurvivalResult.raw_trigger_present` | `ProbeExample.metadata["trigger_inserted"]` — separates the two negative populations (`dataset.py:41`, `PROJECT2_FOUNDATIONS.md:64-65`). |
| `base_id` | **Leakage-safe split grouping unit** (`dataset.py:49-95`). Counterfactual twins share a `base_id`. |
| `trigger_final_token_start` / `trigger_final_token_end` | The trigger's final-token span, for `TRIGGER_SPAN` pooling (`dataset.py:37-38`). |
| `trial_id` | Join key from result → probe example → final token ids. |
| `pipeline_policy`, `survival_class` | Carried into `ProbeExample.metadata` as stratification covariates (`dataset.py:42-43`). |
| `final_prompt_text_path` / the final token ids | **The activation input** — the probe must extract over *exactly* the Layer-4 tokens Project 1 verified delivery into (`PROJECT2_RESOURCES.md:26-28`). See the §4/§12 sparsity gap. |

**The three populations.** Because delivery is verified, Project 2 partitions examples into three
groups that no insertion-labeled dataset can distinguish (`runner.py:84-95`,
`PROJECT2_FOUNDATIONS.md:57-65`):
1. **Delivered positives** — trigger inserted *and* reached the final tokens (`label=True`).
2. **Clean negatives** — no trigger ever inserted (`trigger_inserted=False`, `label=False`).
3. **Partial-survival negatives** — trigger inserted but dropped/corrupted upstream, so `label=False`
   yet fragments may still contaminate the activations (`trigger_inserted=True`, `label=False`).

The handling of population (3) is Project 2's central methodological content (§8).

---

## 3. The estimand and success criteria

- **Primary estimand:** `P(probe fires | trigger delivered)` = TPR over delivered positives, measured
  **at the calibrated threshold**, at FPR budgets `[1e-2, 1e-3]` (`config.py:17-18`,
  `PROJECT2_FOUNDATIONS.md:3-7`).
- **Dual reporting (the differentiator).** Every metric is reported twice
  (`ProbeEvaluationResult`, `schemas/probes.py:109-129`):
  - **all-trials** — partial-survival negatives count against the budget (deployment-pessimistic);
  - **delivered-only** — verified-delivered positives + clean negatives only, isolating the estimand
    from pipeline-induced label noise.
- **Achieved FPR reported twice** (`achieved_fprs` over all test negatives; `achieved_fprs_clean` over
  clean test negatives), each with a Wilson 95% interval (`schemas/probes.py:145-146`,
  `PROJECT2_FOUNDATIONS.md:92-98`).
- **TPR is read at the calibrated operating point, never off the test ROC curve** — an honest
  deployment number (`schemas/probes.py:80-82`, `runner.py:239-250`).
- **No numeric success bar is pre-registered.** Success is defined *structurally* — dual reporting,
  calibrated operating points, honest low-n intervals — not as "TPR ≥ X". Setting a target sensitivity
  is an open decision (§13). The paper priors imply the 0.6B–8B suite has a *lower single-probe
  ceiling* than 70B-class results, so any numeric target must be size-aware
  (`PROJECT2_RESOURCES.md:80-85`).

---

## 4. Two conceptions of "the activation phase" — and which one Project 2 is

This is the most important continuity fact in the whole program, and any reader must internalize it.

**Two incompatible definitions of the "follow-on / activation phase" coexist in the repo:**

- **(A) Behavioral** — make the model emit `CANARY_SEEN` under temperature-0 decoding and string-match
  it, producing a `GenerationResult` (`DATA_CONTRACTS.md:115-122`, `EXPERIMENT_OVERVIEW.md:39-41`,
  `EXPERIMENT_DESIGN.md:213`, `ARCHITECTURE.md:77-78`). This is the *older* framing, sketched but never
  built (`runner.py::_maybe_generate` is a stub, `SESSION_HANDOFF.md:52-54`).
- **(B) Representational** — train linear probes on hidden-state activations, producing a
  `ProbeEvaluationResult` (`PROJECT2_FOUNDATIONS.md` throughout; the built code). Only
  `ANALYSIS_PLAN.md:75` bridges the two ("the activation/**linear-probe** phase").

**Project 2, as scoped here and as built, is conception (B).** The `CANARY_SEEN`/`GenerationResult`
machinery is a parallel, largely-superseded path. The two share the *conditioning logic*
(`P(· | delivered)`) and the same stratified-subset selection problem (§9), but they produce different
artifacts and answer different questions. Where this document says "activation," it means **hidden
states fed to a probe**, never a behavioral emission. (In `GENERIC_PLAN.md`, "activation" always means
the behavioral emission, and "Layer 1–4" means prompt-logging stages, not transformer layers —
`GENERIC_PLAN.md:251-273, 296-300`; do not import those tables as probe results.)

The two conceptions are *complementary*, not contradictory: a fully realized program could report
`P(probe fires | delivered)` and `P(behavioral activation | delivered)` side by side and study their
disagreement (a probe firing while the behavior does not is itself a finding). But Project 2's
committed deliverable is the probe.

---

## 5. The coded foundation (what already exists and is tested)

All modules below are additive, typed, `ruff`/`mypy`-clean, and tested **fully offline** against the
reference extractor; no real-model numbers exist yet (`PROJECT2_FOUNDATIONS.md:10-12, 145-146`).

### 5.1 Module map (`PROJECT2_FOUNDATIONS.md:14-29`)

| Module | Responsibility |
|---|---|
| `schemas/probes.py` | Data contracts: `ProbeExample`, `LayerProbeMetrics`, `AchievedFpr`, `ProbeEvaluationResult`, `PoolingStrategy`, `ProbeLabelSource`, `ProbeSplit`. |
| `activations/extractor.py` | `ActivationExtractor` ABC; `HFActivationExtractor` (real) + `ReferenceActivationExtractor` (offline twin). |
| `activations/pooling.py` | Per-layer token pooling: `last_token` / `mean` / `max` / `trigger_span`. |
| `activations/store.py` | `.npz` + JSONL-manifest persistence of pooled feature matrices. |
| `probes/linear.py` | Numpy-only L2 logistic-regression probe (`fit`/`predict_proba`/`decision_scores`/`save`/`load`). |
| `probes/metrics.py` | Tie-aware AUROC, `threshold_at_fpr`, `tpr_at_fpr`, confusion counts. |
| `probes/calibration.py` | Empirical-quantile threshold calibration + Wilson 95% CI. |
| `probes/aggregation.py` | Registry of multi-layer score aggregators. |
| `experiments/probe_detection/{config,dataset,runner}.py` | Config schema, dataset build/split, end-to-end runner. |
| CLI | `trigger-audit run-probe-experiment CONFIG_YAML` (`configs/probe_detection.example.yaml` runs offline out of the box). |

### 5.2 The extractor twin (mirrors `TokenizerAdapter`)

- **`HFActivationExtractor`** (`extractor.py:139-199`): lazily imports `torch`/`transformers` in
  `__init__` (base package stays importable on CPU login nodes), loads `AutoModelForCausalLM` with
  `output_hidden_states=True`, returns each requested layer as float32 numpy from a **single forward
  pass** (`extractor.py:188-199`). Layer indexing matches HF: index 0 = embeddings, 1..N = block
  outputs, so a config's `layers` list is portable across backends (`extractor.py:20-26`).
  **Batch size 1, CPU default** (`extractor.py:192`).
- **`ReferenceActivationExtractor`** (`extractor.py:58-136`): deterministic, numpy-only; token ids hash
  (with seed) into a lazily generated embedding table, each layer applies fixed random mixing + tanh
  **with a residual back to the embeddings**. The residual guarantees "which token ids are present
  stays linearly recoverable at every layer" — exactly the property the probe tests exercise. It is
  **not** a real model and is never used for measurement (`PROJECT2_FOUNDATIONS.md:39-44`).

### 5.3 The runner's end-to-end loop (`runner.py:76-175`)

For each configured layer: extract → pool → fit `LinearProbe` on **TRAIN** → calibrate thresholds to
each target FPR on the **clean CALIBRATION negatives** → evaluate on **TEST** twice (all trials;
delivered-only). Then stack per-layer scores, fit the aggregator on **CALIBRATION** scores (not TRAIN;
§8.4), and report the aggregate + achieved FPRs (all + clean). Token ids arrive via an injected
`token_provider`, decoupling the runner from where final tokens come from (`runner.py:59-74`).

### 5.4 Config surface (`config.py`, `configs/probe_detection.example.yaml`)

Knobs: `extractor_backend` (`reference`/`hf`), `layers`, `pooling`, `probe_l2`/`lr`/`max_iter`,
`target_fprs`, `aggregation` (+`aggregation_params`), `train_fraction`/`calibration_fraction`/
`split_seed`, and the measurement inputs `survival_results_path` + `final_tokens_path` (both null →
deterministic synthetic offline dataset). Defaults probe layers `[1,2,3,4]`, `mean` pooling,
`target_fprs=[1e-2,1e-3]`, `mean_score` aggregation.

### 5.5 What the tests already prove (`tests/test_probe_end_to_end.py` and unit tests)

AUROC > 0.9 on the reference backend; calibrated FPR within 2× target; delivered-only metrics fully
populated; run determinism; artifact persistence + probe save/load round-trip reproduces scores;
`TRIGGER_SPAN` operator-confound handling (spanless examples get a seeded random span; verified on
pooled features); **clean-only calibration** (contaminating the calibration pool raises the threshold —
the harness rejects a calibration split with no clean negative); both all/clean achieved FPRs reported
over the right counts; stacker fit on CALIBRATION not TRAIN. **Gap:** the synthetic smoke sets
`trigger_inserted == label` and a unique `base_id` per example (`dataset.py:130-139`), so it never
exercises population (3) (partial-survival negatives) *except* in the hand-built fault-injection tests —
that regime is real-data-only.

---

## 6. Methodological design decisions and their rationale

Each of these is a deliberate, documented choice; the experiments in the companion doc are designed to
*stress* them.

1. **Delivery-verified labels, not insertion labels** — labeling by insertion "trains and evaluates
   the probe on falsehoods" (`PROJECT2_FOUNDATIONS.md:51-55`). This is the repo's unique angle and
   gap #1/#5 that no external resource fills (`PROJECT2_RESOURCES.md:99-112`).
2. **`base_id`-grouped, leakage-safe splits** — all trials of one base (its counterfactual twins and
   near-duplicate policy/position variants) land in the same split, or the probe memorizes base
   content and reports inflated metrics (`dataset.py:56-63`, `PROJECT2_FOUNDATIONS.md:68-75`).
3. **Calibrate on clean CALIBRATION negatives only** — the monitor guards clean traffic, so the FPR
   budget is a statement about clean negatives; partial-survival negatives are a third population whose
   trigger-contaminated scores would bias thresholds high and delivered-only TPR low, worst exactly in
   the partial-survival regimes this repo studies (`runner.py:88-95, 285-297`,
   `PROJECT2_FOUNDATIONS.md:79-90`).
4. **Stacking leakage → fit the learned aggregator on CALIBRATION, not TRAIN** — per-layer probes fit
   on TRAIN are optimistically separated; fitting the stacker on those inflated scores mis-scales its
   weights. Out-of-fold stacking is named as the cleaner alternative; calibration-split fitting is the
   pragmatic choice with a mild adaptivity trade-off (`runner.py:126-135`,
   `PROJECT2_FOUNDATIONS.md:111-121`).
5. **`TRIGGER_SPAN` is an oracle diagnostic, never a deployable operating point** — the span is
   information a deployed monitor lacks; its numbers must never be compared head-to-head with
   `mean`/`last_token`. The **operator confound**: span-pooling ~3 tokens vs mean-pooling a whole
   sequence produces different feature statistics *even with no trigger content*, so spanless examples
   are pooled over a seeded random span of the median trigger length to hold the pooling operator
   identical across classes (`runner.py:182-218`, `PROJECT2_FOUNDATIONS.md:123-141`).
6. **Wilson intervals, empirical-quantile thresholds, low-n honesty** — when `target_fpr < 1/n` the
   quantile cannot resolve the target; the threshold is placed just above the max negative score,
   achieved FPR is exactly 0, and Wilson on `0/n` states honestly how little that certifies. Wilson
   over Clopper–Pearson to stay scipy-free (`probes/metrics.py:54-81`, `calibration.py:25-71`,
   `PROJECT2_FOUNDATIONS.md:88-96`).
7. **Extraction stack = vanilla `transformers` + `output_hidden_states=True`; adopt no
   interpretability framework** — every target model is a standard HF causal LM; numerical fidelity to
   the exact pipeline Project 1 audits matters more here than anywhere (`PROJECT2_RESOURCES.md:23-28`).
   nnsight is "consider later" only if a causal-intervention phase is added; TransformerLens is skipped
   on numerical-fidelity grounds despite now supporting Qwen3 (`PROJECT2_RESOURCES.md:50-57, 114-120`).
8. **Own the probe harness, structurally shaped by Apollo's `deception-detection`** but do not vendor
   it (no license) (`PROJECT2_RESOURCES.md:29-33`).

---

## 7. Model suite, layers, and compute

- **Pre-registered model suite:** Qwen3-0.6B/1.7B/4B/8B + Pythia-1B; Gemma/TinyLlama used in template
  trials (`PRE_REGISTRATION.md:17`, `PROJECT2_RESOURCES.md:16-17`). **Note the suite mismatch:**
  `GENERIC_PLAN.md` names only Qwen3-1.7B/4B/8B + Pythia-1B + OLMo, no 0.6B/Gemma/TinyLlama
  (`GENERIC_PLAN.md:910-914`); the fuller suite is a Project-2 artifact.
- **The size sweep is a Project-2 instrument.** The four Qwen sizes "share one tokenizer+template, so
  they are redundant for delivery; they matter only for the later *activation* phase (weights)"
  (`ONE_SHOT_PLAN.md:41`). This is a key justification: probe-detectability-vs-scale is *the* reason
  the size axis exists.
- **Layer prior:** the informative band for Qwen sits at roughly **68–89% of depth**; linear probes
  match or beat MLP heads; the whole pipeline runs on an **8 GB GPU** (arXiv 2606.02628,
  `PROJECT2_RESOURCES.md:86-88`). This seeds default layer selection. Multi-layer ensembling is the
  cheapest recovery of the lower single-probe ceiling at small scale (arXiv 2604.13386,
  `PROJECT2_RESOURCES.md:80-85`).
- **Compute:** activation extraction is the GPU phase (Tillicum, ≥1 GPU/job, billed —
  `EXPERIMENT_DESIGN.md:192`, `CLUSTER_EXECUTION_PLAN.md:66-68`); **shard strictly by model** so each
  worker loads weights once (`CLUSTER_EXECUTION_PLAN.md:72`); deterministic decoding/extraction for
  reproducibility (`CLUSTER_EXECUTION_PLAN.md:78`). The probe *analysis* is CPU-light and can run
  locally on the pooled feature matrices; only extraction needs the GPU.

---

## 8. External resources and method recipes (survey summary)

Full verdicts and licenses in `PROJECT2_RESOURCES.md`; pins in `third_party/README.md`. The operative
conclusions:

- **Adopt:** BackdoorLLM (MIT; triggered/clean corpora + attack recipes; `B4G/` is the Qwen2.5 LoRA
  fine-tune template we adapt — no Qwen backdoored weights exist upstream), Liars' Bench (Apache-2.0
  code / CC-BY data; a **soft-trigger sleeper-agent LoRA** scenario, the closest public analog to
  canary detection), and two MIT reference implementations size-matched to our suite:
  `caught-in-the-act` (Qwen 1.5B–14B deception probes) and `HallucinationPatternDetection`
  (Qwen2.5-7B per-layer linear probe on an 8 GB GPU) (`PROJECT2_RESOURCES.md:66-71`).
- **Study-only (no license / adversarial):** Apollo `deception-detection` (structural template),
  `llms-backdoor-detection` (sleeper backdoors in Llama-3 caught by linear probes),
  `obfuscated-activations` (**attacks that make latent/backdoor probes fail** — the stress test our
  probe must survive), DemonAgent (encrypted + fragmented/distributed triggers)
  (`PROJECT2_RESOURCES.md:67-76`).
- **Four paper recipes** (`PROJECT2_RESOURCES.md:80-95`):
  1. *Multi-layer ensembling* (2604.13386): layerwise L2 logistic probes; double-fault selection +
     stacking recovers AUROC; ~5% AUROC per 10× params → small models have a lower ceiling.
  2. *Mid-layer decodability, Qwen2.5* (2606.02628): linear ≥ MLP; informative band ~68–89% depth;
     8 GB GPU.
  3. *Pressure-testing probes* (2605.27958): near-perfect clean AUROC collapses under stylistic shift;
     single-direction probes fail cross-domain, k≥5-dim probes + style-augmented training recover it →
     train across delivery styles Project 1 already manufactures.
  4. *Production probes for Gemini* (2601.11516): short→long-context shift is the hardest production
     failure; hybrid probe + prompted-classifier cascades hit low-FPR points → long-context RAG cells
     are first-class eval strata.

> A concrete per-repo parameter digest (layers/pooling/probe archs/attack families) is folded into
> `PROJECT2_EXPERIMENTS_REVIEW.md` §external, where it grounds specific experiments; this section is
> the decision-level summary.

---

## 9. The stratified-subset selection problem (shared with the behavioral path)

Activation extraction is expensive, so it must run on a **stratified subset**, not the whole grid:
controls (positive/negative), delivered-prefix, not-delivered-prefix, boundary-corruption, and a
stratified sample (`EXPERIMENT_DESIGN.md:240`, `CLUSTER_EXECUTION_PLAN.md:73-77`). **This selection
mechanism is a TODO** — `ManifestBuilder` currently flags generation by a blind deterministic fraction
(`run_generation`); stratified selection *from survival results* is unbuilt
(`CLUSTER_EXECUTION_PLAN.md:75-77`). Project 2 needs this both for the probe wave and (if pursued) the
behavioral wave. The probe's leakage-safe split further requires that stratification respect `base_id`
grouping.

---

## 10. Threats to validity and known confounds (carried into the adversarial review)

1. **Delivery/positional confound** — trigger use is lower mid-long-context even when verified present;
   "delivery and positional underuse are separate failure modes" (`GENERIC_PLAN.md:458`). A probe must
   not conflate them.
2. **Operator confound in span pooling** — handled by the random-span fallback (§6.5); still the single
   easiest way to fake a probe result if disabled.
3. **Tokenizer/vocab drift (inherited P1 pitfall)** — `tokenizer_id` on a result row is a config
   *alias*, not a loadable HF id; policy/position strings are registry ids, not pre-reg names. Any
   probe code that re-tokenizes a trigger or groups by policy must resolve aliases via the
   models/policies config (`ANALYSIS_PLAN.md:112, 158, 163`).
4. **Stylistic / distribution shift** — near-perfect clean AUROC collapses under shift; the delivery
   styles P1 manufactures (template/truncation/RAG) are the built-in shift axes to train and test
   across (`PROJECT2_RESOURCES.md:89-92`).
5. **Long-context shift** — the hardest production failure mode; long-context RAG cells are first-class
   strata (`PROJECT2_RESOURCES.md:93-95`).
6. **Adversarial obfuscation** — `obfuscated-activations` demonstrates attacks that defeat latent
   probes; a credible detector must be evaluated against them (`PROJECT2_RESOURCES.md:74`).
7. **Real-positive absence** — all current positives are harmless canaries; genuine backdoored
   behavior is not yet in the data, so canary-detectability may over- or under-state
   backdoor-detectability (`PROJECT2_FOUNDATIONS.md:149-150`).
8. **Small-model ceiling** — the 0.6B–8B suite has a lower single-probe AUROC ceiling than the
   70B-class literature; targets and aggregation must be size-aware (`PROJECT2_RESOURCES.md:80-85`).

---

## 11. Data and artifact contracts (for the probe wave)

- **Input join:** `survival_results_path` (JSONL of `SurvivalResult`) + `final_tokens_path` (JSONL of
  `{trial_id, final_token_ids}`) → `build_probe_examples` + a `trial_id → token ids` map
  (`runner.py:319-337`, `config.py:58-60`).
- **Persisted artifacts:** pooled per-layer feature matrices (`.npz` + trial-id manifest via
  `activations/store.py`), one saved `LinearProbe` per layer, and one `ProbeEvaluationResult` row
  (`runner.py:109-115, 347`). Every reported number is recomputable from the stored features + probe
  (a determinism/traceability test asserts this).
- **Provenance to record for a real run:** model revision, tokenizer HF id, extractor backend,
  `transformers`/`torch` versions, input file hashes, seed — mirroring the P1 analysis manifest
  discipline.

---

## 12. What is deliberately deferred or not yet built

| Item | State | Source |
|---|---|---|
| Real HF extraction validated on GPU weights | **unexercised** — lands with the cluster phase | `PROJECT2_FOUNDATIONS.md:145-146` |
| Real backdoored Qwen-small/TinyLlama weights | **must be fine-tuned** via the B4G MIT recipe (planned task, not done) | `PROJECT2_RESOURCES.md:123-125` |
| `final_token_ids` / final-prompt persistence at 100% | **gap** — only the ~2% `--log-prompts` sample carries the prompt path today | `ANALYSIS_PLAN.md:177`, `CLUSTER_EXECUTION_PLAN.md:82-83` |
| Stratified-subset selection from survival results | **TODO** — currently a blind deterministic fraction | `CLUSTER_EXECUTION_PLAN.md:75-77` |
| `device`/`revision` wired through the probe config → HF extractor | **gap** — `run_probe_experiment` does not pass them | `runner.py:302-308` vs `extractor.py:147-154` |
| Activation batching / extract-once-pool-many reuse | **not built** — extractor is batch-1; the runner re-extracts per run | `extractor.py:192`, `runner.py:189-201` |
| Semantic / distributed / split triggers | **out of scope** — span pooling assumes a contiguous span | `PROJECT2_FOUNDATIONS.md:151-152` |
| nnsight/nnterp intervention backend | deferred; implements `ActivationExtractor` later | `PROJECT2_FOUNDATIONS.md:147-148` |
| Probe design **pre-registration** | **not pre-registered** (unlike P1's H1–H4) | continuity finding; `PRE_REGISTRATION.md` has no probe section |
| Inference discipline (cluster-bootstrap over `base_id`, TOST/equivalence) ported to P2 | **unstated** — P2F reports Wilson CIs on FPR only | continuity finding |

---

## 13. Open decisions for Saki (before a real-model wave)

1. **Numeric success bar** — do we set a target TPR@FPR (size-aware) or keep success purely structural?
2. **Real backdoored weights** — approve fine-tuning small-Qwen/TinyLlama LoRAs via B4G now, or run the
   first wave on harmless canaries only and treat backdoored checkpoints as a later measurement target?
3. **Final-prompt persistence** — log 100% of final token ids in the next run (unblocks the probe join
   cleanly) vs re-derive final prompts from manifest+bases for the probed subset.
4. **Pre-registration of the probe design** — write layers/pooling/aggregation/FPR targets and the
   evaluation contract into a dated `PRE_REGISTRATION.md` amendment before the first measurement wave.
5. **Inference layer** — extend P1's cluster-bootstrap-over-`base_id` + equivalence testing to the
   probe estimates, or keep Wilson-on-FPR as the only uncertainty statement.
6. **Behavioral path** — is `P(behavioral activation | delivered)` (the `CANARY_SEEN` wave) still in
   scope as a complement, or fully superseded by the probe?

---

*End of master reference. Companion documents: `PROJECT2_EXPERIMENTS_REVIEW.md` (experiments +
justification + adversarial review), `PROJECT2_CONTINUITY_CHECK.md` (continuity audit),
`PROJECT2_EXPERIMENT_PLAN.md` (final experiment plan + argument).*
