# Project 2 — Experiments to Validate the Probe, with Justification and Adversarial Review

**Purpose.** This is the working document that turns `PROJECT2_MASTER.md` into a concrete, tiered
experiment program, justifies every design choice from first principles and the source material, and
then **adversarially reviews the program as an expert referee would** — attacking each experiment for
the ways it could produce a true-looking but wrong conclusion, and stating the mitigation. The final,
reconciled version (after the continuity pass) is `PROJECT2_EXPERIMENT_PLAN.md`; this document is
deliberately more argumentative and shows the reasoning, including the objections that survive.

Grounding: the coded seams are `experiments/probe_detection/{config,dataset,runner}.py`,
`activations/extractor.py`, `probes/{linear,metrics,calibration,aggregation}.py`,
`schemas/probes.py`; the external method parameters are from `third_party/` (caught-in-the-act,
Apollo `deception-detection`, HallucinationPatternDetection, obfuscated-activations,
llms-backdoor-detection, liars-bench, BackdoorLLM/B4G, DemonAgent) and the four core papers.

---

## Part 0 — What "validate Project 2" means

Project 2's deliverable is a *claim of the form*: "On model M, a linear probe on hidden states
detects a delivered trigger with TPR = t at a calibrated FPR ≤ f, and this holds under the
distribution shifts and adversaries a deployment would face." Validating it therefore has **four
distinct burdens**, and the experiment tiers map onto them one-to-one:

1. **Instrument validity** — the harness measures what it claims (calibration is honest, splits don't
   leak, pooling doesn't fake separation). *No real weights needed.* → **Tier 0**.
2. **Existence & shape** — on real weights, a delivered trigger *is* linearly decodable, and where
   (layer/pooling/scale). → **Tier 1**.
3. **Robustness** — the detector survives the distribution shifts the deployment creates (delivery
   style, context length, trigger type, model). → **Tier 2**.
4. **Reality & adversariality** — it detects *genuine trained backdoors*, not just benign canaries, and
   survives an adversary who trains to evade it. → **Tiers 3–4**.

A program that stops at Tier 1 (the trap most probing papers fall into) proves only that a benign
canary string is linearly recoverable — which the *reference extractor already guarantees by
construction* and is nearly content-free as a backdoor-detection claim. The scientific weight is in
Tiers 2–4. This ordering is itself a design decision defended in the adversarial review (§AR-0).

---

## Part 1 — The experiment program

Notation: each experiment states **What / How (seam) / Why (justification)**. `P(fire|delivered)` is
always TPR over delivered positives at the calibrated threshold; FPR budgets are `{1e-2, 1e-3}` unless
noted. All inference clusters on `base_id` (Part 6).

### Tier 0 — Instrument validity (offline, reference extractor, CPU)

**E0.1 — Recoverability & determinism gate.**
*What:* On the reference extractor, per-layer probe AUROC > 0.9 and byte-identical results across
invocations. *How:* the existing `test_probe_end_to_end.py` battery, promoted to a reported gate.
*Why:* a floor sanity check that the plumbing (extract→pool→fit→calibrate→evaluate) is wired and
deterministic. **This is a gate, not evidence** — the reference extractor makes token presence
linearly recoverable by construction (`extractor.py:70-72`), so a *failure* here is informative but a
*pass* proves only plumbing.

**E0.2 — Calibration honesty & low-n resolution.**
*What:* Sweep the number of clean calibration negatives `n ∈ {30, 100, 300, 1000, 3000}`; for each,
report achieved FPR (all + clean) with Wilson CI at targets `{1e-2, 1e-3}`, and confirm the
`target_fpr < 1/n` regime yields achieved FPR exactly 0 with the honest wide Wilson interval.
*How:* `calibrate_threshold` + `wilson_interval` (`calibration.py`), driven over synthetic negatives.
*Why:* the monitor's contract *is* the FPR (`calibration.py:3-6`); we must show it is honest and,
critically, **surface the resolution limit**: FPR 1e-3 is unmeasurable below ~1000 clean negatives.
This experiment produces the number that decides how many clean bases a real wave needs (Part 6).

**E0.3 — Leakage-safety ablation (base_id grouping).**
*What:* On a dataset that actually contains counterfactual twins, compare `base_id`-grouped splitting
vs example-level random splitting; report the AUROC/TPR *inflation* from leakage.
*How:* `assign_splits` (grouped) vs a random-split variant; requires real P1 data or a synthetic set
**rebuilt so twins share a `base_id`** — the shipped synthetic generator gives every example a unique
`base_id` (`dataset.py:130-139`) and so *cannot* exhibit the leak (a bug to fix for this experiment).
*Why:* the grouping rule is asserted to prevent inflation (`PROJECT2_FOUNDATIONS.md:68-75`); an
ablation that *quantifies* the inflation is what turns an assertion into evidence, and it is the
cleanest demonstration that Project 2's split discipline matters.

**E0.4 — Pooling operator-confound ablation.**
*What:* Under `TRIGGER_SPAN` pooling, compare random-span fallback ON vs OFF on **trigger-free** data;
show that OFF manufactures class separation from pooling statistics alone.
*How:* the `_random_span` fallback path (`runner.py:182-218`); disable it to reproduce the confound.
*Why:* proves the operator confound is real and the control necessary
(`PROJECT2_FOUNDATIONS.md:131-137`); guards against a whole class of illusory span-pooling results.

**E0.5 — Three-population calibration ablation.**
*What:* Calibrate on clean-only vs a clean+partial-survival mix; show the mixed pool raises the
threshold and depresses delivered-only TPR.
*How:* the fault-injection pattern in `test_calibration_uses_clean_negatives_only`, elevated to a
reported ablation over realistic partial-survival score distributions.
*Why:* justifies the clean-only calibration decision empirically, precisely in the partial-survival
regime the repo studies (`runner.py:88-95`).

### Tier 1 — Existence & shape on real weights (GPU, harmless canaries)

Inputs: Project 1's real `SurvivalResult` rows + the exact final token ids (Part 6 gap), real Qwen3
weights via `HFActivationExtractor`.

**E1.1 — Layer sweep per model.**
*What:* `P(fire|delivered)` and AUROC vs **depth-fraction** for Qwen3-0.6B/1.7B/4B/8B + Pythia-1B; also
the model-free separation-ratio diagnostic per layer (centroid-distance / within-class spread).
*How:* `layers` = a depth-fraction grid; one forward pass returns all layers (`extractor.py:188-199`).
*Why:* the mid/late band prior (~68–89% depth; peak ~2/3) is a Qwen2.5 finding
(`PROJECT2_RESOURCES.md:86-88`, caught-in-the-act peaks ~2/3 depth) that must be *confirmed* on our
suite, not assumed; the separation ratio is a cheap probe-free cross-check (HallucinationPatternDetection).

**E1.2 — Pooling ablation.**
*What:* `last_token` vs `mean` vs `max`; `trigger_span` reported **separately as an oracle diagnostic**.
*How:* `PoolingStrategy` (`pooling.py`); span pooling with the operator-confound control.
*Why:* pooling convention is a documented silent validity bug — last-token (caught-in-the-act,
llms-backdoor-detection) vs mean (Apollo `mlr`, HallucinationPatternDetection) change results; a
standardized ablation prevents cherry-picking and fixes the deployable operating point.

**E1.3 — Multi-layer aggregation.**
*What:* best single layer vs `mean_score`/`max_score`/`product_of_experts`/`quantile`/
`stacked_logistic`; measure the recovery.
*How:* `AGGREGATION_REGISTRY` (`aggregation.py`); stacker fit on CALIBRATION (`runner.py:126-135`).
*Why:* small models have a lower single-probe ceiling and multi-layer ensembling is the cheapest
recovery (~5% AUROC/10× params; double-fault selection — `PROJECT2_RESOURCES.md:80-85`; Apollo
averages the middle half).

**E1.4 — Scale curve & representational redundancy.**
*What:* single-probe ceiling *and* INLP direction-count vs model size (0.6B→8B).
*How:* per-model best-layer AUROC; INLP loop (iterative probe → nullspace projection → retrain) from
caught-in-the-act.
*Why:* directly tests the "small models are inherently more ablatable" prediction (caught-in-the-act:
100+ directions at 14B collapsing to chance after ~20–40 rounds at small scale); the direction count is
a scalar "how obfuscatable is this backdoor" that feeds Tier 4.

**E1.5 — Delivery-conditional decomposition (the program's headline).**
*What:* report `P(fire)` all-trials vs `P(fire|delivered)`, and quantify how much apparent "probe miss"
is actually delivery failure.
*How:* the dual `*_all` / `*_delivered_only` metrics already in `ProbeEvaluationResult`.
*Why:* this *is* the thesis (`GENERIC_PLAN.md:419-420`) — the number no insertion-labeled probe study
can compute, and the reason Project 2 sits on Project 1.

### Tier 2 — Robustness to the shifts the deployment creates (GPU, canaries)

**E2.1 — Delivery-style generalization.**
*What:* train on a subset of pipeline policies/positions, test on held-out ones (e.g. train on `none`
+ `truncate_head`, test on `keep_recent` + RAG).
*Why:* near-perfect clean AUROC collapses under stylistic shift; single-direction probes fail
cross-domain, k≥5-dim + style-augmented training recover it (`PROJECT2_RESOURCES.md:89-92`). Project 1
*manufactures* these styles for free — the built-in shift axis.

**E2.2 — Long-context generalization.**
*What:* train short-context, test on long-context / RAG cells; report `P(fire|delivered)` vs context
length.
*Why:* short→long-context shift is the single hardest production failure (`PROJECT2_RESOURCES.md:93-95`);
long-context RAG cells are first-class strata, not an afterthought.

**E2.3 — Trigger-type & distributed-trigger generalization (the discriminating test).**
*What:* train on `rand`/`multi` canaries, test on `unicode`/`natural`/`boundary`; and — the headline —
`P(fire|delivered)` on **distributed/composite triggers (CTBA: `2024`+`BadMagic`+`OpenAI` scattered;
MTBA: varying)** relative to a **regex/audit baseline that provably scores ~0** there.
*Why:* single-token triggers are string-detectable, so beating a null model on them proves little; the
value proposition of a hidden-state probe is exactly the CTBA/DemonAgent regime where no contiguous
surface trigger exists (`backdoorllm .../ctba`, DemonAgent fragmentation). Note this stresses the
contiguous-span assumption of `TRIGGER_SPAN` pooling and argues for mean/trajectory pooling here.

**E2.4 — Cross-model transfer.**
*What:* probe trained on model A, evaluated on model B.
*Why:* transfer is both a robustness probe and a deployment scenario (llms-backdoor-detection shows
cross-model transfer). **Caveat that becomes a design constraint:** different-size Qwen models have
different `hidden_size`, so a probe vector is not dimension-compatible across sizes — transfer is only
literal within one width, or requires a learned projection / shared-subspace method. Same-width transfer
(e.g. two checkpoints of one size) is the clean version.

### Tier 3 — Real backdoored models (GPU; requires fine-tuning — the eventual target)

**E3.1 — Install & measure real backdoors on the small suite.**
*What:* fine-tune LoRA backdoors on Qwen3-small / TinyLlama across the DPA taxonomy
(BadNet/VPI/MTBA/CTBA/Sleeper) and B4G functional tasks (safety/model-identity/style/access-lock);
measure `P(fire|delivered)` vs clean-FPR — i.e. **TAR_w / TAR_wo** (`B4G/README.md:17`).
*How:* the B4G MIT LoRA recipe (`B4G/src/b4g/training.py`), targets `q/k/v/o/gate/up/down`,
label-masked prompt tokens; liars-bench QLoRA config as a starting point.
*Why:* current positives are harmless canaries; probing genuine backdoored checkpoints is the stated
measurement target (`PROJECT2_FOUNDATIONS.md:149-150`). `TAR_w/TAR_wo` is structurally identical to our
delivery-conditional estimand, so the metric transfers directly.

**E3.2 — Poison-rate × trigger-complexity sweep.**
*What:* grid poison rate (e.g. 875 vs 200 poisoned) × trigger complexity (single-token vs complex/bio)
per model size; find the detectability floor.
*Why:* obfuscated-activations shows *natural* obfuscation — low poison rate or complex trigger, no
adversary — already defeats probes; small models are predicted to hit the floor far earlier (Tier 1.4).
This is the cheapest real-world evasion and must be characterized.

**E3.3 — Behavioral vs representational agreement.**
*What:* jointly report `P(probe fires|delivered)` and `P(behavioral activation|delivered)` (the
`CANARY_SEEN`/`GenerationResult` wave), and study their disagreement.
*Why:* reconciles the two "activation phase" conceptions (`PROJECT2_MASTER.md §4`); a probe firing
while the behavior is suppressed is a *positive result for the probe* and a distinct scientific finding
(the representation carries the trigger even when output does not).

**E3.4 — Baseline battery.**
*What:* the shipped probe vs (a) Apollo's detector zoo (`lr/mms/cmms/lat/mlp/mlr`) as probe baselines,
(b) DefenseBox defenses (CleanGen/CROW/decoding/pruning/quantization), (c) a regex/audit null on
distributed triggers.
*Why:* a detector is only interesting relative to baselines; DefenseBox shows generation/decoding
defenses collapse or backfire on jailbreak-style backdoors (`DefenseBox/README.md:42-66`) while an
internal-state probe should not — that gap is the probe's reason to exist, so a jailbreak-style task
must be in the battery.

### Tier 4 — Adversarial robustness (obfuscation-aware; make-or-break)

**E4.1 — Natural obfuscation curve.** degradation of `P(fire|delivered)` vs poison rate and trigger
complexity, *no probe access* (subsumes E3.2; reported as an attack). *Why:* the cheapest attack needs
no knowledge of the probe.

**E4.2 — Explicit obfuscation (probe-in-the-loop).**
*What:* co-train a backdoor with an activation-space obfuscation loss against our probe
(Gaussian/MMD, Mahalanobis, probe-ensemble losses); report `P(fire|delivered)` collapse.
*How:* obfuscated-activations loss constructors (`backdoors_obfuscation.py:526-616`); the paper defeats
*even a 48-probe orthogonal ensemble* (`:119-130`).
*Why:* any robustness claim not measured against an adversary that trained to evade *this* probe is
unfalsifiable marketing. Scope honestly: the aim is to **characterize degradation**, not to "prove
robust" — the arms race has no clean terminus.

**E4.3 — Inference-time suffix attacks.** GCG / FLRT suffixes against the probe (obfuscated-activations
`optimize/{gcg,flrt}_utils.py`). *Why:* a no-retraining attack surface a deployment faces.

**E4.4 — Defense: obfuscation-augmented training + detection-point sweep.**
*What:* OAT-style co-training of probe on obfuscated examples; sweep detection point
(`last-prompt-token` / `prompt` / `generation`). *Why:* the paper's own counter (OAT); obfuscation is
position-specific, so a point that one attack spares may still detect (`obfuscated-activations` OAT).
Caveat: OAT can overfit the specific attack — report held-out-attack generalization, not train-attack
recovery.

### Part 6 — Cross-cutting: statistics, provenance, and prerequisites

- **Inference:** cluster-bootstrap over `base_id` for every rate CI (port P1's machinery), Wilson for
  achieved FPR, and TOST/equivalence when a claim is *invariance* (e.g. "pooling X ≈ pooling Y"),
  reusing the pre-registered ±5 pp / Holm scheme. P2F currently reports Wilson-on-FPR only — this is an
  extension, not present today.
- **Provenance:** model revision, tokenizer HF id (resolved, not the alias), extractor backend,
  `torch`/`transformers` versions, input hashes, seed — per row.
- **Hard prerequisites** (block a real wave): (i) 100% final-token-id logging or a re-derivation path
  (the `final_tokens_path` join, Part-6 gap); (ii) `device`/`revision` threaded into the probe config →
  HF extractor; (iii) the stratified-subset selector (currently a TODO); (iv) a Gate-0 counterfactual
  check on the probed subset so probe labels don't inherit a scorer leak; (v) a dated pre-registration
  of layers/pooling/aggregation/FPR targets before measurement.

---

## Part 2 — Adversarial review (expert-referee mode)

Each objection is stated at its strongest, then the mitigation that keeps the experiment honest. Where
an objection **survives**, it is marked ⚠ and carried into the final plan as a scoped limitation.

**AR-0 — "You are measuring canary-string recoverability and calling it backdoor detection." ⚠**
The reference extractor makes token presence linearly recoverable *by construction*
(`extractor.py:70-72`), and even on real weights a benign canary is a salient literal substring; a
trained backdoor's latent "trigger recognized → deploy behavior" state is a different object. Tiers 0–2
could all pass while the detector is useless on real backdoors.
*Mitigation:* this is exactly why the program is tiered and why **Tier 3 is mandatory, not optional** —
the canary tiers validate the *instrument and its robustness*, and the claim they license is explicitly
bounded to "detecting a delivered canary's representation." The generalization from canary to backdoor
is an *empirical question E3.1 answers*, never assumed. **Survives** as the program's central scoping
caveat: no backdoor-detection claim may be made from Tiers 0–2 alone.

**AR-1 — "FPR 1e-3 is unmeasurable at your scale." ⚠**
The clustering unit is `base_id`, and the target is ~100–300 bases/family. Achieved FPR at 1e-3 needs
≥~1000 *independent* clean negatives to resolve at all; below that, calibration honestly returns FPR 0
with a Wilson interval spanning up to ~[0, 0.004] (E0.2). So the headline 1e-3 operating point may be
**certifiable only as "≤ some loose bound,"** not measured.
*Mitigation:* E0.2 surfaces the exact resolution limit before any real wave; report 1e-2 as the
*resolvable* operating point and 1e-3 as bounded-only unless the clean-negative pool is scaled
(more clean bases, or trial-level negatives with an explicit note that they are correlated within base).
**Survives** — it constrains what can be claimed and drives the base-count decision.

**AR-2 — "The delivered-only positive set is a collider / selection artifact."**
Conditioning on delivery selects trials, and delivery correlates with position/length/policy; a probe
evaluated only on delivered positives may be reading those covariates, not the trigger.
*Mitigation:* the clean negatives are drawn to match the delivered positives' base/policy/position
composition (matched strata), and E2.1/E2.2 explicitly test whether the probe generalizes *across*
those covariates. The counterfactual twin design means each delivered positive has a same-base negative,
controlling base content. Not fully dissolved, but bounded by the matched-strata + twin design.

**AR-3 — "base_id grouping can't be demonstrated on the shipped synthetic data."**
E0.3's inflation demonstration is impossible on the synthetic generator, which assigns a unique
`base_id` per example (`dataset.py:130-139`) — there are no twins to leak.
*Mitigation:* fix the generator (or use real P1 data) so twins share a base for this experiment; noted
as a prerequisite. This is a genuine gap in the current test scaffolding, not just the experiment.

**AR-4 — "Cross-model transfer among Qwen sizes is dimensionally impossible."**
Different sizes have different `hidden_size`; a probe vector cannot be applied across widths (E2.4).
*Mitigation:* restrict literal transfer to same-width pairs; for cross-width, use a learned projection
or shared-subspace method and label it a *different* experiment. The naive "train on 8B, test on 0.6B"
is malformed and must not be reported as transfer.

**AR-5 — "Small backdoors may not install." ⚠**
The B4G recipe is validated on 7–9B models; a reliable conditional backdoor may not train into a 0.6B
model, so Tier 3 positives might not exist at the small end.
*Mitigation:* verify backdoor ASR (`TAR_w`) as a *precondition* before probing — if the backdoor didn't
install (low ASR), the model is excluded from Tier 3 and reported as "backdoor did not install at this
scale," itself a finding. **Survives** as a Tier-3 feasibility risk at 0.6B–1.7B.

**AR-6 — "Obfuscation experiments are circular and unfalsifiable."**
E4.2 trains an attack against our probe, then E4.4 trains our probe against the attack — an arms race
with no defined winner; either side can always take the last move.
*Mitigation:* fix the protocol *before* running (attacker and defender compute budgets, held-out
attacks for evaluating the defense) and frame the deliverable as a **degradation curve and a
transferability-of-defense number**, never "robust." OAT recovery must be measured on *held-out* attack
families (E4.4 caveat). **Survives** as an explicit scoping of what Tier 4 can conclude.

**AR-7 — "Probe strength is CoT/scratchpad leakage, not a latent backdoor state." ⚠**
llms-backdoor-detection loses ~0.15 AUROC when the trigger-naming scratchpad is removed; if Project 1
transcripts contain reasoning that echoes the trigger, the probe may read CoT text.
*Mitigation:* run every real-model experiment with reasoning trimmed (Apollo's `trim_reasoning`
convention) and report trimmed vs untrimmed as an ablation. **Survives** as a mandatory ablation.

**AR-8 — "The stacked aggregator double-uses the calibration split."**
The stacker is fit on CALIBRATION scores and the thresholds are calibrated on CALIBRATION negatives
(`runner.py:126-138`) — the same data sets weights and operating point, a mild adaptivity leak the code
itself flags.
*Mitigation:* report `stacked_logistic` with a caveat, and offer an out-of-fold variant as the clean
comparison; do not headline a stacked number without the OOF check. Closed-form aggregators
(mean/max/PoE/quantile) are unaffected and are the safer headline.

**AR-9 — "Layer-k is not comparable across sources."**
caught-in-the-act indexes L post-block layers (no embedding); HallucinationPatternDetection indexes L+1
(with embedding); our extractor uses HF indexing (0 = embedding). Comparing "layer 18" across them is
meaningless.
*Mitigation:* report **depth-fraction**, not raw index, everywhere (E1.1), and state the convention
explicitly. Purely a discipline fix, fully resolvable.

**AR-10 — "Determinism/AUROC>0.9 on the reference extractor is presented as a result."**
It is trivially true by construction and says nothing scientific.
*Mitigation:* E0.1 is labeled a *gate*, and no scientific claim is drawn from it (see AR-0). The
reference extractor's role is plumbing and offline CI, never measurement — the code and docs are already
strict about this (`PROJECT2_FOUNDATIONS.md:44`).

**AR-11 — "Probe labels inherit Project 1 scorer errors."**
The label is `final_token_trigger_present`; the pilot once had a 312-twin control leak from a real
scorer bug. A mislabeled positive/negative corrupts everything downstream.
*Mitigation:* run P1's Gate-0 counterfactual control on the *probed subset* as a precondition (Part 6),
and exclude any subset that leaks. The label's trustworthiness is a checkable gate, not an assumption.

**AR-12 — "Distributed-trigger detection may be beyond a linear probe on a contiguous readout."**
CTBA/DemonAgent triggers have no contiguous span; last-token or span pooling may miss them even if the
information is present distributed across positions.
*Mitigation:* E2.3 uses mean/trajectory pooling for these and reports the pooling dependence explicitly;
a negative result here (linear probe *cannot* catch distributed triggers) is a legitimate and important
finding, not a failure of the experiment.

**Global reviewer verdict.** The program is sound *if and only if* the claims are scoped to the tier
that earned them. The dangerous failure mode is a headline of the form "linear probes detect backdoors
in small Qwen models at FPR 1e-3" drawn from Tier-1 canary data at pilot scale — which AR-0, AR-1, and
AR-7 each independently falsify. The surviving caveats (⚠ AR-0/1/5/6/7) are not fatal; they are the
honest boundary of what each experiment concludes, and they are carried verbatim into
`PROJECT2_EXPERIMENT_PLAN.md`.

---

*Next: `PROJECT2_CONTINUITY_CHECK.md` re-reads all sources + this draft for contradictions and unmet
dependencies; then `PROJECT2_EXPERIMENT_PLAN.md` is the reconciled final artifact.*
