# Project 2 — Experiment Plan and Argument (Final)

**What this is.** The final, reconciled deliverable for Project 2 (activation-based trigger detection):
(1) the complete set of experiments that should be run to validate it, and (2) a rock-solid argument
for why *this* set — why each experiment is necessary, why the set is sufficient, and what each
experiment is and is not allowed to conclude. It supersedes the argumentative draft
(`PROJECT2_EXPERIMENTS_REVIEW.md`) and folds in the continuity prerequisites
(`PROJECT2_CONTINUITY_CHECK.md`). Background and the coded foundation are in `PROJECT2_MASTER.md`.

The estimand throughout: **`P(probe fires | trigger delivered)`** — the true-positive rate of a linear
probe on hidden-state activations, measured at a threshold calibrated to a target false-positive budget,
conditional on Project 1 having verified the trigger reached the final model-visible tokens.

---

## Part I — The argument (why this program, in five moves)

**Move 1 — The quantity that matters is a calibrated operating point, not AUROC.** A deployed trigger
monitor is *a threshold*, and its contract to an operator is a false-positive budget, not a rank
statistic — "an AUROC says nothing about the threshold a monitor actually runs at"
(`PROJECT2_FOUNDATIONS.md:8`). Every experiment therefore reports TPR at a calibrated FPR with an
honest interval, and AUROC only as a summary. This is why the harness is built around
`threshold_at_fpr` + Wilson intervals, not around ROC area.

**Move 2 — Conditioning on verified delivery is the necessary, unique contribution.** The program's
thesis, inherited from Project 1, is that *much of what looks like backdoor-robustness failure is
delivery failure, and you must prove delivery before attributing anything to the model*
(`GENERIC_PLAN.md:76-80, 419-420`; `SESSION_HANDOFF.md:37-39`). Labeling a probe by *insertion* — the
universal practice in the public literature — "trains and evaluates the probe on falsehoods"
(`PROJECT2_FOUNDATIONS.md:51-55`), because an inserted-but-dropped trigger is not a positive. Delivery
verification is the one thing no external resource supplies (`PROJECT2_RESOURCES.md:99-112`) and the
reason Project 2 is a distinct contribution rather than a re-run of deception-probe work. It forces
three populations no insertion-labeled study can separate — delivered positives, clean negatives, and
partial-survival negatives — and the correct handling of the third is the program's methodological core.

**Move 3 — Validity is earned in tiers, and every claim is scoped to the tier that earned it.** There
are four independent burdens — the instrument is honest; the signal exists and has a shape; it survives
distribution shift; it detects *real* backdoors and *adversaries*. A probing study that stops after
"clean AUROC ≈ 1.0" has discharged only the second burden on the easiest data and proven almost nothing
about deployment. The tiers below discharge the burdens in order, and **no claim may borrow validity
from a tier it did not run** (the central discipline; see Part IV).

**Move 4 — Canary data validates the instrument; only trained backdoors and adversaries validate the
detector.** All current positives are harmless canaries, and the reference extractor makes token
presence linearly recoverable *by construction* (`extractor.py:70-72`). So Tiers 0–2 can prove the
harness is sound and robust to pipeline-induced shift *without licensing any statement about real
backdoors*. The generalization from "a delivered canary's representation is decodable" to "a trained
backdoor's latent state is decodable" is an empirical question — answered only by Tier 3 (real
backdoored weights) and stress-tested only by Tier 4 (obfuscation). This is why Tier 3 is mandatory, not
optional, and why the plan refuses a backdoor-detection headline from canary data.

**Move 5 — Every design choice traces to a documented failure mode or a literature prior.** Nothing here
is a free parameter: `base_id`-grouped splits answer twin-leakage; clean-only calibration answers the
partial-survival third population; depth-fraction reporting answers a layer-index convention hazard;
multi-layer aggregation answers the small-model single-probe ceiling; the obfuscation tier answers the
demonstrated fact that linear, MLP, SAE, and even 48-probe orthogonal ensembles are all individually
evadable. The experiments are the minimal set that touches every one of these.

Taken together: the program measures the *right quantity* (Move 1), makes the *only contribution that is
ours* (Move 2), earns validity *without over-claiming* (Moves 3–4), and leaves *no design choice
unjustified* (Move 5). That is the argument.

---

## Part II — Prerequisites (must clear before any real-model tier)

From the continuity audit (`PROJECT2_CONTINUITY_CHECK.md §G`). The first five block a real wave:

| # | Prerequisite | Why it blocks | Fix size |
|---|---|---|---|
| P1 | Persist `final_token_ids` for the probed subset | The probe join has no producer today (data exists in-pipeline but is dropped from `SurvivalResult`) | small |
| P2 | Build the `base_id`-aware stratified subset selector | Selection is a blind fraction now; extraction can't run on the right subset | medium |
| P3 | Thread `device`/`revision`/`trust_remote_code` into the probe config → HF extractor | No GPU selection / no revision pin for a reproducible run | small |
| P4 | Dated `PRE_REGISTRATION.md` amendment: layers (depth-fraction), pooling, aggregation, FPR targets, dual reporting, base_id-clustered inference | Probe design is un-preregistered; without it every number is exploratory | doc |
| P5 | Run Project 1's Gate-0 counterfactual control on the probed subset | Probe labels inherit any scorer leak; must be verified clean | reuse |
| P6 | Extend the synthetic generator (shared-base twins + partial-survival negatives) | Tier-0 offline experiments can't otherwise be demonstrated | small |
| P7 | Port base_id cluster-bootstrap; set base counts from E0.2; scope 1e-3 | Inference discipline + the FPR-resolution limit | small |

---

## Part III — The experiments (every one, with what/why and the claim it licenses)

Convention: layers are reported by **depth-fraction**, never raw index (avoids the L vs L+1 vs
HF-indexing hazard). All rate CIs cluster on `base_id`. "Licenses" = the exact claim a clean result
permits — and nothing broader.

### Tier 0 — Instrument validity (offline, reference extractor, CPU; no real weights)

| ID | Experiment | Why (justification) | Licenses |
|---|---|---|---|
| **E0.1** | Recoverability & determinism **gate**: per-layer AUROC > 0.9 and byte-identical reruns on the reference extractor | Plumbing sanity; reference extractor is recoverable by construction | "the harness runs and is deterministic" — a gate, **not evidence** |
| **E0.2** | Calibration honesty & **FPR resolution**: sweep clean-negative count `n∈{30,100,300,1000,3000}`; report achieved FPR (all + clean) + Wilson CI at `{1e-2,1e-3}`; confirm `target<1/n ⇒ FPR=0` with honest wide interval | The monitor's contract is the FPR; must be honest, and the resolution limit sets the base-count budget (P7) | "FPR `f` is resolvable at ≥`n(f)` clean negatives; 1e-3 needs ~1000" |
| **E0.3** | **Leakage ablation**: `base_id`-grouped vs example-level random split on data with real twins; report the AUROC/TPR inflation | Turns the split-discipline assertion into a measured number (needs P6) | "example-level splitting inflates metrics by Δ; grouping is necessary" |
| **E0.4** | **Operator-confound ablation**: `TRIGGER_SPAN` random-span fallback ON vs OFF on trigger-free data | Proves span pooling can fake separation from pooling statistics alone | "the random-span control is necessary; span numbers without it are invalid" |
| **E0.5** | **Three-population calibration ablation**: clean-only vs clean+partial-survival calibration pool | Justifies clean-only calibration precisely in the partial-survival regime | "mixing partial-survival negatives biases thresholds high and TPR low" |

### Tier 1 — Existence & shape on real weights (GPU; harmless canaries; delivery-verified)

| ID | Experiment | Why | Licenses |
|---|---|---|---|
| **E1.1** | **Layer sweep** per model (Qwen3-0.6/1.7/4/8B + Pythia-1B): `P(fire|delivered)`, AUROC, and separation-ratio vs depth-fraction | Confirm (not assume) the mid/late band prior (~68–89% depth; peak ~2/3) on *our* suite | "for model M, the informative band is [a,b]·L" (canary) |
| **E1.2** | **Pooling ablation**: last_token vs mean vs max; `trigger_span` reported separately as an oracle diagnostic | Pooling convention is a silent validity bug across the literature | "the deployable operating pooling is X; span is oracle-only" |
| **E1.3** | **Multi-layer aggregation**: best-single vs mean/max/PoE/quantile/stacked; recovery measured | Small-model single-probe ceiling; ensembling is the cheapest recovery | "aggregation recovers Δ AUROC/TPR over best single layer" |
| **E1.4** | **Scale curve & INLP direction-count** vs model size 0.6B→8B | Tests "small models are inherently more ablatable" (feeds Tier 4) | "single-probe ceiling and redundancy grow with scale as …" |
| **E1.5** | **Delivery-conditional decomposition** (the headline): `P(fire)` all-trials vs `P(fire|delivered)`; fraction of apparent misses that are delivery failures | This is the thesis and the number no insertion-labeled study can compute | "X% of apparent probe failures are delivery failures, not model failures" |

### Tier 2 — Robustness to deployment-induced shift (GPU; canaries)

| ID | Experiment | Why | Licenses |
|---|---|---|---|
| **E2.1** | **Delivery-style generalization**: train on a subset of policies/positions, test on held-out ones | Clean AUROC collapses under stylistic shift; P1 manufactures these styles for free | "the probe (does/doesn't) generalize across pipeline styles" |
| **E2.2** | **Long-context generalization**: short-context train → long-context/RAG test; `P(fire|delivered)` vs context length | Short→long shift is the hardest production failure mode | "the probe holds to context length L / degrades beyond L" |
| **E2.3** | **Trigger-type & distributed-trigger** generalization: train rand/multi, test unicode/natural/boundary; and CTBA/MTBA distributed triggers vs a regex/audit null that scores ~0 | The discriminating test — string-detectable triggers prove little; distributed is where a probe earns its keep | "the probe detects distributed triggers a string matcher cannot" (or a negative finding) |
| **E2.4** | **Cross-model transfer** (same-width only; cross-width flagged as a separate projection experiment) | Transfer is both robustness and a deployment scenario; different widths are dimensionally incompatible | "a probe transfers across same-width checkpoints with TPR …" |

### Tier 3 — Real backdoored models (GPU; requires fine-tuning — the measurement target)

| ID | Experiment | Why | Licenses |
|---|---|---|---|
| **E3.1** | **Install & measure real backdoors** on small Qwen/TinyLlama via the B4G LoRA recipe across the DPA taxonomy (BadNet/VPI/MTBA/CTBA/Sleeper) + B4G tasks; report `P(fire|delivered)` vs clean-FPR (= TAR_w/TAR_wo). **Precondition:** verify backdoor ASR installed | Current positives are canaries; genuine backdoors are the stated target; TAR_w/TAR_wo *is* the delivery-conditional estimand | **the first real backdoor-detection claim**: "TPR t @ FPR f on installed backdoor type T, model M" |
| **E3.2** | **Poison-rate × trigger-complexity sweep**; find the detectability floor per model size | Natural obfuscation (low poison / complex trigger) defeats probes with no adversary; small models hit the floor early | "detectability floor at poison rate p / complexity c for model M" |
| **E3.3** | **Behavioral vs representational agreement**: `P(probe fires|delivered)` vs `P(CANARY behavior|delivered)`; study disagreement | Reconciles the two "activation phase" conceptions; a probe firing while behavior is suppressed is a positive result | "the representation carries the trigger even when the behavior is suppressed in X% of cases" |
| **E3.4** | **Baseline battery**: probe vs Apollo detector zoo (lr/mms/cmms/lat/mlp/mlr), vs DefenseBox (CleanGen/CROW/decoding), vs regex null — including a **jailbreak-style** trigger task | A detector is only interesting vs baselines; generation/decoding defenses collapse on jailbreak backdoors where an internal probe should not | "the probe beats defense D by Δ on jailbreak-style backdoors" |

### Tier 4 — Adversarial robustness (GPU; obfuscation-aware; make-or-break)

| ID | Experiment | Why | Licenses |
|---|---|---|---|
| **E4.1** | **Natural-obfuscation curve**: `P(fire|delivered)` vs poison rate & trigger complexity, no probe access | The cheapest real attack needs no knowledge of the probe | "detection degrades to t' under natural obfuscation" |
| **E4.2** | **Explicit obfuscation (probe-in-the-loop)**: co-train a backdoor with an activation-space loss (Gaussian/MMD, Mahalanobis, probe-ensemble) against our probe; report collapse | Any robustness claim not tested against an evasion-trained adversary is unfalsifiable; the attack defeats even 48-probe orthogonal ensembles | "under a probe-aware adversary of budget B, detection collapses to t''" — **degradation, never 'robust'** |
| **E4.3** | **Inference-time suffix attacks** (GCG/FLRT) against the probe | A no-retraining attack surface a deployment faces | "suffix attacks reduce detection to t'''" |
| **E4.4** | **Defense: obfuscation-augmented (OAT) training + detection-point sweep** (last-prompt-token / prompt / generation); evaluate on **held-out** attack families | The paper's own counter; obfuscation is position-specific; must test defense transfer, not train-attack recovery | "OAT recovers Δ against held-out attacks at detection point P" |

### Cross-cutting (applies to every real-model experiment)

- **Inference:** cluster-bootstrap `P(fire|delivered)` over `base_id`; Wilson for achieved FPR; TOST
  (±5 pp / Holm, per the P1 amendment) for any *invariance* claim (e.g. "pooling X ≈ Y", "model A ≈ B").
- **Mandatory ablations:** reasoning-trimmed vs untrimmed (CoT-leakage control, AR-7); matched-strata
  clean negatives (collider control, AR-2).
- **Provenance per row:** model revision, resolved HF tokenizer id, extractor backend, torch/transformers
  versions, input hashes, seed.

---

## Part IV — Why this set is *sufficient* and each member is *necessary*

**Necessity (drop-one argument).** Remove any experiment and a specific, nameable validity hole opens:

- Drop **E0.2** → you quote a 1e-3 TPR the data cannot resolve (D3).
- Drop **E0.3/E0.4/E0.5** → the three headline methodological choices (grouping, span control, clean
  calibration) revert to unfalsified assertions.
- Drop **E1.1/E1.2** → layer and pooling become undisclosed researcher degrees of freedom (cherry-pick).
- Drop **E1.5** → you have a probe paper, not *this* project; the delivery-conditional number is the
  contribution.
- Drop **E2.x** → a clean-data-only result that the shift literature says collapses in deployment.
- Drop **E3.x** → the entire program remains a canary study with zero backdoor-detection evidence (E1).
- Drop **E4.x** → a robustness claim refuted by construction by obfuscated-activations.

**Sufficiency (coverage argument).** The four validity burdens (Part I, Move 3) are each discharged:
instrument (Tier 0), existence/shape (Tier 1), robustness (Tier 2), reality + adversariality (Tiers
3–4). The known threat catalog (`PROJECT2_MASTER.md §10`) maps onto experiments with no gaps:
delivery/positional confound → E1.5/E2.2; operator confound → E0.4; tokenizer/vocab drift → prerequisites
B2/B3; stylistic shift → E2.1; long-context → E2.2; obfuscation → E4; real-positive absence → E3;
small-model ceiling → E1.4/E1.3. Every threat has an owner; every experiment kills a threat.

**Scoping (the discipline that makes it rigorous).** Each row's "Licenses" column is a hard boundary.
The dangerous headline — "linear probes detect backdoors in small Qwen at FPR 1e-3" — is forbidden
because it would require E3 evidence (backdoor, not canary), E0.2 resolution (1e-3 measurable), and an
E4 adversary (robustness) simultaneously; no single tier grants it. This scoping is not modesty; it is
what separates a result that replicates from one that evaporates on contact with a real deployment.

---

## Part V — Success criteria and what would falsify the program

- **Instrument success (Tier 0):** achieved FPR ≤ target within CI at the resolvable operating points;
  measured leakage inflation; both confounds demonstrated and controlled.
- **Existence success (Tier 1):** a depth-fraction band with `P(fire|delivered)` materially above the
  calibrated FPR for canaries on ≥ the mid-size models; the delivery-conditional decomposition (E1.5)
  quantifies real misattribution.
- **Detector success (Tier 3):** `TAR_w` materially exceeds `TAR_wo` on *installed* backdoors at the
  resolvable FPR, for at least the distributed-trigger types a string matcher misses (E2.3/E3.1).
- **Robustness characterization (Tier 4):** a reported degradation curve and a held-out-attack OAT
  recovery number — success here is *honest characterization*, not a robustness proof.
- **Falsification / negative results that are still findings:** the backdoor does not install at 0.6B
  (AR-5); the linear probe cannot catch distributed triggers (AR-12); probe strength was CoT leakage
  (AR-7); natural obfuscation collapses detection at deployable poison rates (E3.2/E4.1). Each is a
  legitimate, publishable outcome — the program is designed so that a negative result is informative,
  not a null.

**No numeric TPR target is pre-committed** (open decision; `PROJECT2_MASTER.md §13.1`). Any target set
later must be size-aware (the small suite has a lower single-probe ceiling) and stated for a *resolvable*
FPR.

---

## Appendix — traceability

Estimand, contracts, and the coded foundation: `PROJECT2_MASTER.md`. Full justification + the expert
adversarial review with surviving caveats: `PROJECT2_EXPERIMENTS_REVIEW.md`. Contradictions, field-name
drift, and the prerequisite checklist: `PROJECT2_CONTINUITY_CHECK.md`. External method parameters and
attack families: `PROJECT2_RESOURCES.md` + `third_party/README.md`. Code seams:
`experiments/probe_detection/{config,dataset,runner}.py`, `activations/extractor.py`,
`probes/{linear,metrics,calibration,aggregation}.py`, `schemas/probes.py`.
