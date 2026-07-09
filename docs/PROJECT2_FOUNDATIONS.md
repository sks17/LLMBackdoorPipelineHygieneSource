# Project 2 Foundations: Activation-Based Trigger Detection

Project 1 answers "does the trigger reach the final model-visible tokens?". Project 2 asks
the follow-on question: **given that a trigger was delivered, does a linear probe on the
model's hidden-state activations fire — at a threshold calibrated to a target false-positive
rate?** Formally, the target quantity is `P(probe fires | trigger delivered)` at operating
points like FPR ≤ 1e-2 and 1e-3, which is what a deployment-style monitor cares about (an
AUROC says nothing about the threshold a monitor actually runs at).

This document describes the purely additive foundations landed for Project 2. Everything
runs and is tested fully offline; real-model measurement is deliberately deferred (see the
final section).

## Module map (additions to `src/trigger_audit/`)

| Module | Responsibility | Reusable? |
|--------|----------------|-----------|
| `schemas/probes.py` | Data contracts: `ProbeExample`, `LayerProbeMetrics`, `AchievedFpr`, `ProbeEvaluationResult`, `PoolingStrategy` | shared |
| `activations/extractor.py` | `ActivationExtractor` interface; reference + HF implementations | shared |
| `activations/pooling.py` | Per-layer token pooling (last-token / mean / max / trigger-span) | shared |
| `activations/store.py` | `.npz` + JSONL-manifest persistence of pooled feature matrices | shared |
| `probes/linear.py` | Numpy-only L2 logistic-regression probe (`fit`/`predict_proba`/`decision_scores`/`save`/`load`) | shared |
| `probes/metrics.py` | Tie-aware AUROC, `threshold_at_fpr`, `tpr_at_fpr`, confusion counts | shared |
| `probes/calibration.py` | Empirical-quantile threshold calibration + Wilson 95% CI | shared |
| `probes/aggregation.py` | Registry of multi-layer score aggregators | shared |
| `experiments/probe_detection/` | Config, dataset construction/splitting, end-to-end runner | experiment |

The CLI gains one thin command: `trigger-audit run-probe-experiment CONFIG_YAML`
(see `configs/probe_detection.example.yaml`, which runs offline out of the box).

## The reference-vs-HF extractor twin pattern

`ActivationExtractor` mirrors the `TokenizerAdapter` pattern exactly:

- `HFActivationExtractor` — the production path. Lazily imports `torch`/`transformers`
  inside `__init__` (the base package stays importable on CPU-only login nodes), loads
  `AutoModelForCausalLM` with `output_hidden_states=True`, and returns selected layers as
  float32 numpy. Requires the `[hf]` + `[generate]` extras.
- `ReferenceActivationExtractor` — deterministic and dependency-free (numpy only). Token
  ids hash (with the seed) into a lazily generated random embedding table, so the vocabulary
  is unbounded; each layer applies fixed random linear mixing + tanh with a residual back to
  the embeddings, plus a small sinusoidal position encoding. The residual guarantees that
  which token ids are present stays **linearly recoverable** at every layer, which is what
  the probe tests exercise. It is NOT a real model and is never used for measurement.

Layer indexing matches HF `hidden_states`: index 0 is the embedding layer, 1..`num_layers`
are block outputs — so a config's `layers` list is portable between backends.

## The survival-aware labeling contract

Probe labels come from `SurvivalResult.final_token_trigger_present` — **delivery-verified
ground truth** — never from whether a trigger was inserted upstream. This is the repo's
unique angle: an inserted trigger that the prompt pipeline dropped is *not* a positive, and
treating it as one (the common practice when labeling by insertion) trains and evaluates
the probe on falsehoods.

Evaluation is therefore reported twice in `ProbeEvaluationResult`:

- **all trials** — inserted-but-undelivered trials count as negatives (their fragments may
  contaminate activations; this is the pessimistic, deployment-realistic view);
- **delivered-only** — verified-delivered positives plus clean (never-inserted) negatives,
  isolating `P(probe fires | trigger delivered)` from pipeline-induced label noise.

`ProbeExample.metadata["trigger_inserted"]` carries the raw-layer insertion flag that
separates the two negative populations.

## The base_id-grouped split rule

`assign_splits` groups by `base_id`: **all trials sharing a base conversation land in the
same split.** Project 1 expands one base into many trials (positions × lengths × policies,
plus counterfactual twins differing only by trigger presence). An example-level random
split would place near-duplicate contexts — sometimes exact twins — on both sides of the
train/test line, letting the probe memorize base content and report inflated metrics.
Group shuffling is seeded, so the assignment is deterministic given (examples, fractions,
seed).

## Calibration semantics

`calibrate_threshold(negative_scores, target_fpr)` picks the **lowest threshold whose
empirical FPR on the calibration negatives is ≤ the target** (the empirical-quantile rule;
lowest admissible threshold maximizes sensitivity within the budget). The runner calibrates
on the **clean (never-inserted) negatives of the CALIBRATION split only** — the probe
monitors clean traffic, so its FPR budget is a statement about clean negatives. Partial-
survival negatives (inserted but not delivered) are a *third* population, neither clean nor
delivered; letting their trigger-contaminated scores into the calibration-negative pool
would bias thresholds high and the delivered-only TPR low, worst exactly in the partial-
survival regimes this repo studies. `_validate_splits` therefore requires at least one clean
calibration negative. When `target_fpr < 1/n` the quantile cannot resolve the target: the
threshold is placed just above the maximum negative score, achieved FPR is exactly 0, and
the Wilson interval on `0/n` states honestly how little that certifies.

Achieved FPR is reported twice: `achieved_fprs` over **all** test negatives (the
deployment-pessimistic view, where partial-survival negatives count against the budget) and
`achieved_fprs_clean` over the **clean** test negatives only (the population the budget
actually contracts). Both use a Wilson score 95% interval — Wilson rather than
Clopper–Pearson because the exact interval needs scipy's beta inverse CDF and the base
package is deliberately scipy-free. Reported TPRs are measured **at the calibrated
thresholds** (deployment operating points), not read off the test ROC curve.

## Multi-layer aggregation registry

`AGGREGATION_REGISTRY` (reusing the generic `Registry` from `pipelines/base.py`) resolves
combiners by config string: `mean_score`, `max_score`, `min_score`, `product_of_experts`
(sum of per-layer logits), `quantile` (configurable `q`), and `stacked_logistic` (a learned
`LinearProbe` over the per-layer score vectors). Aggregators expose a no-op-by-default
`fit(layer_scores, labels)` and `aggregate(layer_scores) -> scores`, so learned and
closed-form combiners are interchangeable. Min/PoE/quantile aggregators for probe ensembles
are a known gap in the open-source probing ecosystem; this registry fills it behind one
interface.

**Stacking leakage.** The per-layer probes are fit on TRAIN, so their TRAIN scores are
optimistically separated relative to anything the stacker will see at inference. The runner
therefore fits `stacked_logistic` on the held-out **CALIBRATION** split scores, not TRAIN,
so the stacker's weights are learned on an honest score distribution and it can actually
tell informative layers from overfit ones. The statistically cleaner alternative is
out-of-fold stacking (k-fold within TRAIN, score each held-out fold with probes fit on the
rest, fit the stacker on the assembled out-of-fold matrix, then refit the per-layer probes
on full TRAIN for inference); calibration-split fitting is the pragmatic choice given the
runner's shape and reuses the same negatives that set the thresholds, a mild adaptivity
trade-off OOF would avoid. Closed-form aggregators (mean/max/min/PoE/quantile) ignore `fit`
and are unaffected.

## Pooling and the TRIGGER_SPAN oracle diagnostic

`TRIGGER_SPAN` pooling averages activations over the ground-truth trigger token span. That
span is **oracle information a deployed monitor does not have**, so `trigger_span` results
are a **localization diagnostic** (does trigger content concentrate in the span?), *not* a
deployable operating point — never compare its numbers head-to-head with `mean`/`last_token`
as if all three were deployable. Two further subtleties the runner handles:

- **Operator confound.** Span-pooling a ~3-token window and mean-pooling a whole sequence
  produce different feature statistics (a short mean has far higher variance) *even with no
  trigger content*, which alone can separate the classes. So for spanless examples the
  runner pools a **seeded random span of the median trigger length** rather than falling
  back to `mean` — the pooling operator is then identical across all classes and only
  trigger *content* can drive separation. The trial ids that received a random fallback span
  are recorded in `metadata["trigger_span_random_fallback_trial_ids"]` for audit.
- **Who carries a span.** It is *not* "negatives lack spans": clean negatives never carry a
  span, but partial-survival **negatives do** (Project 1's scorer localizes the surviving
  fragment). Only the random-span fallback keeps those three populations on a common
  operator.

## What is deliberately deferred

- **Real HF extraction validation on GPU** — `HFActivationExtractor` exists but has not been
  exercised against real weights; that lands with the cluster generation phase.
- **nnsight/nnterp integration** — a richer intervention/extraction backend can implement
  `ActivationExtractor` later without touching callers.
- **Real backdoored-model data (BackdoorLLM)** — current positives are harmless canaries;
  probing genuinely backdoored checkpoints is the eventual measurement target.
- **Semantic / distributed triggers** — trigger-span pooling assumes a contiguous token
  span; split and semantic triggers need span sets and are out of scope here.
- **sklearn/scipy upgrades** — the numpy probe and Wilson intervals are deliberate; sklearn
  solvers and exact intervals can be swapped in behind the same interfaces if the analysis
  extra is ever promoted.
