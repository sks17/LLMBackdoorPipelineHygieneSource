# Project 2 — external resource evaluation

Which open-source repositories and papers are worth building on for Project 2 (activation-based
trigger detection: linear probes on per-layer hidden states of open-weight LLMs, calibrated to a
target false-positive rate, evaluated **conditional on verified trigger delivery** from Project 1)?

This document records the survey verdicts and the reasoning behind them. The additive
infrastructure that implements the resulting design lives in the repo (see
[`PROJECT2_FOUNDATIONS.md`](PROJECT2_FOUNDATIONS.md)); local pruned snapshots of the adopted
resources live in [`third_party/`](../third_party/README.md) with commit pins and reuse rules.
Survey date: 2026-07-03, with a same-day **verification pass** that re-checked every
license/support claim against primary sources (GitHub API, shipped release tags, paper HTML).
Two first-pass errors were found and are corrected below, marked **[corrected]**: TransformerLens
*does* support Qwen3, and BackdoorLLM is MIT (not Apache-2.0) with no Qwen backdoored weights.
Verdicts are
grounded in this repo's constraints: the pre-registered model suite (Qwen3 0.6B/1.7B/4B/8B +
Pythia-1B, with Gemma/TinyLlama used in template trials), a CPU-light torch-free base install,
and the delivery-conditional evaluation contract (`P(probe fires | trigger delivered)`, never raw
detection rate — the same delivery/activation separation Project 1 exists to enforce).

## Bottom line

1. **Activation extraction: vanilla `transformers` + `output_hidden_states=True`. Adopt no
   interpretability framework.** Every target model is a standard HF causal LM; one forward pass
   returns every residual-stream layer with exact numerical fidelity to the pipeline this repo
   already audits. That fidelity matters here more than anywhere: the probe must see activations
   over *exactly* the final token IDs Project 1 logged (Layer 4), not a reimplementation's
   approximation of them.
2. **Probe harness: own it (already built), shaped by Apollo's `deception-detection`.** The
   Apollo repo is the best structural template (config → activations → per-layer probes →
   serialized results), but it is Llama-centered, has no surfaced license, and lacks the pieces
   this project is actually about (FPR calibration, delivery-conditional labels, multi-layer
   aggregation). We mirror its shape in-repo rather than vendoring it.
3. **Triggered/clean data: BackdoorLLM first, Liars' Bench soft-trigger second.** Both are
   permissively licensed and give real backdoored models/data to probe, which is the step after
   the harness is trusted on reference data.
4. **Method recipes from papers:** layerwise L2 logistic probes with multi-layer ensembling
   (arXiv 2604.13386), a Qwen mid/late-layer band prior (~68–89% depth; arXiv 2606.02628),
   diverse-delivery training and multi-dimensional probes to survive distribution shift
   (arXiv 2605.27958), and long-context shift as the expected hardest failure mode
   (arXiv 2601.11516).

## Activation-access libraries

Baseline to beat: `AutoModelForCausalLM(...)(input_ids, output_hidden_states=True)` — zero extra
dependencies, works for Qwen3/Pythia/TinyLlama/Gemma alike.

| Library | License / activity | Verdict | Why |
|---|---|---|---|
| [nnsight](https://github.com/ndif-team/nnsight) | MIT; very active (v0.7.0, 2026-05) | **consider (later)** | Best-in-class tracing/intervention on any HF model; earns its place only if a causal-intervention phase (activation patching, steering) is added. Overkill for read-only extraction. |
| [nnterp](https://github.com/ndif-team/nnterp) | MIT; young, active | skip | Standardizes module names across architectures — heterogeneity we don't have; drags in nnsight. |
| [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | MIT; very active | skip (viable) | **[corrected]** The first pass claimed no Qwen3 support; verification shows `supported_models.py` in shipped v3.5.1 lists Qwen3 0.6B–14B, so TransformerLens *is* viable for our suite. The skip stands on the remaining grounds: its hooked reimplementation/weight-processing path risks numerical divergence from the exact HF pipeline Project 1 audits, and read-only extraction needs none of its intervention machinery. |
| [pyvene](https://github.com/stanfordnlp/pyvene) | Apache-2.0; last release 2025-05 | skip | Center of gravity is trainable/causal interventions; config-dict indirection is overhead for reads. |
| [SAELens](https://github.com/jbloomAus/SAELens) | MIT; very active | skip | Trains SAEs, not linear probes — no overlap with the deliverable. Only relevant to a follow-up "are trigger features sparse?" question. |
| [baukit](https://github.com/davidbau/baukit) | **no license file**; dormant since 2024-02 | skip | Right philosophy (`TraceDict` forward hooks, online stats) but legally all-rights-reserved; if sub-layer hooks are ever needed, write the ~30-line equivalent in-house. |
| [NeuroX](https://github.com/fdalvi/NeuroX) | BSD-3; dormant (~2021 API) | skip (mine ideas) | The only whole-pipeline probe toolkit, which validates our design — but 2020-era transformers API, and its probe core is a few lines we need to own anyway for FPR calibration. |
| [tuned-lens](https://github.com/AlignmentResearch/tuned-lens) | MIT; pre-1.0, low activity | skip | Orthogonal diagnostic (prediction trajectories). Notably, its own pipeline consumes plain HF hidden states — further evidence vanilla extraction suffices. |

Consequence for this repo: `HFActivationExtractor` is a thin lazy-import wrapper over
`transformers`, twinned with a deterministic dependency-free `ReferenceActivationExtractor` for
offline tests — the same pattern as `TokenizerAdapter`.

## Probe harnesses, benchmarks, and triggered data

| Resource | License | Verdict | Role for Project 2 |
|---|---|---|---|
| [ApolloResearch/deception-detection](https://github.com/ApolloResearch/deception-detection) | **none** (verified via GitHub API) | **template, study-only** (vendored: `third_party/study_only/deception-detection/`) | Closest existing white-box probe train/eval harness (config → `Experiment` → last-token probes → released weights). Mirror its *structure* in-repo; do not copy code — no license means all rights reserved. |
| [Cadenza-Labs/liars-bench](https://github.com/Cadenza-Labs/liars-bench) | code Apache-2.0, data CC BY 4.0 | **adopt** (vendored: `third_party/liars-bench/`) | Evaluation testbed with explicit Qwen-2.5 coverage and — most relevant — a **soft-trigger (sleeper-agent LoRA) scenario**: the closest public analog to canary-trigger detection. Source of triggered/clean eval and calibration data. |
| [bboylyg/BackdoorLLM](https://github.com/bboylyg/BackdoorLLM) | **MIT** (verified; first pass wrongly said Apache-2.0) | **adopt** (vendored: `third_party/backdoorllm/`) | Primary source of triggered-vs-clean corpora and attack recipes (DPA: BadNet/VPI/Sleeper/MTBA/CTBA fixed-string triggers map directly onto canary presence; plus WPA/HSA/CoTA) and DefenseBox. **No Qwen backdoored weights exist upstream** (HF org ships 25 Llama2 LoRAs only); the `B4G/` subproject fine-tunes Qwen2.5-7B-Instruct LoRAs — that recipe is what we adapt for the Qwen-small suite. |
| [ztimothy96/caught-in-the-act](https://github.com/ztimothy96/caught-in-the-act) | MIT | **adopt (reference impl)** (vendored: `third_party/caught-in-the-act/`) | Independent code for arXiv 2508.19505 — deception probes on **Qwen 1.5B–14B, exactly our size class** (the paper itself ships no official code). Directly informs layer/token selection and the "20–100 deception directions" multi-dimensional finding. |
| [Ezharjan/HallucinationPatternDetection](https://github.com/Ezharjan/HallucinationPatternDetection) | MIT | **adopt (reference impl)** (vendored: `third_party/hallucinationpatterndetection/`) | Code + probe weights for arXiv 2606.02628: minimal per-layer linear-probe pipeline on Qwen2.5-7B-class on an 8 GB GPU. Concrete template for our extraction→pool→logistic loop and the mid-layer band prior. |
| [SasankYadati/llms-backdoor-detection](https://github.com/SasankYadati/llms-backdoor-detection) | **none** | **study-only** (vendored: `third_party/study_only/`) | Closest direct match found anywhere: injects sleeper-style backdoors into Llama-3 and detects them via linear probes on activations. Study the design; reimplement independently. |
| [Kojk-AI/sleeper-agent-probe](https://github.com/Kojk-AI/sleeper-agent-probe) | **none** | **study-only** | Replicates "simple probes catch sleeper agents" on Mistral-7B; validates the basic premise. |
| [LukeBailey181/obfuscated-activations](https://github.com/LukeBailey181/obfuscated-activations) | **none** | **study-only (adversarial)** | Attacks that make backdoor/latent probes fail — the stress-test our shipped probe must survive. |
| [DPamK/BadAgent](https://github.com/DPamK/BadAgent) | **none** (verified) | consider (reference only) | Agent-trace poisoning pipeline, but early pure-prompt agent targets, undocumented triggers, no license, known eval bugs. Mine poison JSON formats only. |
| [whfeLingYu/DemonAgent](https://github.com/whfeLingYu/DemonAgent) | **none** (verified) | consider (stress set) | Encrypted + fragmented/distributed triggers — where string matching fails but a hidden-state probe may still fire. AgentBackdoorEval as an adversarial robustness suite, not a dependency. |

## Method recipes from the paper survey

- **arXiv 2604.13386 (multi-layer ensembling):** layerwise L2-regularized logistic probes;
  stacking complementary layers (double-fault selection) recovers large AUROC losses; probe
  accuracy scales ~5% AUROC per 10× params. Two implications: our 0.6B–8B suite has a lower
  single-probe ceiling than the 70B-class results imply, and multi-layer aggregation is the
  cheapest recovery — hence the aggregation registry (mean/max/min/product-of-experts/quantile/
  stacked-logistic) in `probes/aggregation.py`.
- **arXiv 2606.02628 (mid-layer decodability, Qwen2.5):** linear probes match or beat MLP heads;
  the informative band for Qwen sits at roughly 68–89% of depth; the whole pipeline runs on an
  8 GB GPU. Seeds our default layer selection and confirms cluster feasibility.
- **arXiv 2605.27958 (pressure-testing probes):** near-perfect clean AUROC collapses under
  stylistic shift; single-direction probes fail cross-domain while k≥5-dimensional probes and
  style-augmented training recover it. For us: train across delivery styles (template/truncation/
  RAG variants Project 1 already manufactures) and do not bet on one scalar direction.
- **arXiv 2601.11516 (production probes for Gemini):** short→long-context shift is the hardest
  production failure; hybrid probe + prompted-classifier cascades hit low-FPR operating points.
  Long-context RAG cells from Project 1 are therefore first-class eval strata, not an afterthought.

## Gaps no external resource fills (built here instead)

The survey confirmed the deep-research brief's conclusion: nothing public ships the pieces that
are most specific to this project. These are now in-repo foundations:

1. **Trigger-presence labels grounded in delivery** — public probe weights label deception, not
   "trigger present in final tokens". Ours come from `SurvivalResult.final_token_trigger_present`
   (delivery-verified), with counterfactual twins as negatives.
2. **Leakage-safe splits** — `base_id`-grouped train/calibration/test assignment so a twin pair
   never straddles splits.
3. **Target-FPR threshold calibration** — empirical quantile of held-out clean scores with a
   Wilson 95% interval on achieved FPR (numpy-only).
4. **Multi-layer aggregators** (min, product-of-experts, quantile, stacked logistic) — absent as
   reusable code everywhere we looked.
5. **Survival-aware evaluation** — metrics reported over all trials *and* delivered-only trials,
   so probe misses are never conflated with delivery failures.

## Verification pass (2026-07-03) — corrections to the first survey

An independent re-check of single-source claims overturned two verdicts and closed the open items:

- **TransformerLens *does* support Qwen3** (0.6B–14B in `supported_models.py`, shipped in v3.5.1).
  The first pass's "no Qwen3" disqualification was **wrong**; the skip now rests only on
  numerical-fidelity risk vs the exact HF path, not on model coverage.
- **BackdoorLLM is MIT, not Apache-2.0** (verified via GitHub API). Still adopted; license is
  even more permissive than recorded.
- **BackdoorLLM ships no Qwen backdoored weights** — 25 Llama2 LoRAs on its HF org, nothing else.
  Resolved: we fine-tune our own small-Qwen/TinyLlama LoRAs with its MIT recipe (`B4G/` is the
  Qwen2.5 template). This is now a planned task, not an open question.
- **Apollo `deception-detection` has no license** (confirmed null). Downgraded from "adopt as
  template" to **study-only** — structure may inform ours, but no code/data may be copied.
- **`caught-in-the-act` (MIT)** — the first pass missed arXiv 2508.19505 entirely; its Qwen
  1.5B–14B coverage is the best size-matched public probe evidence we have, and an MIT reference
  implementation exists. Promoted to a first-class reference.
- Code URLs located and vendored for arXiv 2606.02628 and 2508.19505; full PDFs of all eight core
  papers are in `third_party/papers/`. The exact double-fault selection + stacking procedure from
  arXiv 2604.13386 is extracted in `PROJECT2_FOUNDATIONS.md`'s layer-selection notes.

All vendored copies, pins, and licenses are catalogued in
[`../third_party/README.md`](../third_party/README.md).
