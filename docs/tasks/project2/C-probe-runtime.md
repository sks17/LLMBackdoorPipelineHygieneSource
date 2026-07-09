# Spec C — Probe runtime: device/revision threading, model slicing, per-trial predictions, store reuse

This is the load-bearing runtime component: it makes the linear probe **actually run on real model
activations** reproducibly, lets us **slice models by depth-fraction**, and emits the per-trial
predictions the honest inference layer (component G) needs. Prerequisites P3/B4 (device/revision),
plus the depth-fraction slicing (plan Move-5 / continuity A6), per-trial predictions (needed for
`base_id` cluster-bootstrap, continuity D2), and extract-once store reuse (continuity C3).

## Files you own (edit only these + tests)

- `src/trigger_audit/experiments/probe_detection/config.py`
- `src/trigger_audit/experiments/probe_detection/runner.py`
- `src/trigger_audit/activations/slicing.py` — **new**
- `src/trigger_audit/activations/store.py`
- `src/trigger_audit/schemas/probes.py`
- `src/trigger_audit/activations/__init__.py`, `experiments/probe_detection/__init__.py` — exports
- Tests: extend `tests/test_probe_end_to_end.py`, `tests/test_activation_store.py`,
  `tests/test_probe_config.py`; add `tests/test_activation_slicing.py`,
  `tests/test_probe_predictions.py`.

Depends on component A (`SurvivalResult.final_token_ids`) and component D
(`build_synthetic_probe_dataset_with_twins`) — both land before you. Do NOT edit `cli.py` (component F
adds the CLI flags; you expose the config fields and runner behavior they call). Do NOT edit
`dataset.py` or `selection.py`.

## 1. Config threading (P3 / B4)

Add to `ProbeDetectionExperimentConfig` (`config.py`):
- `device: str = "cpu"`
- `revision: str | None = None`
- `trust_remote_code: bool = False`
- `layer_depth_fractions: list[float] | None = None` — when set, the actual layer indices are
  **resolved at runtime** from the loaded/known `num_layers` (see §2), overriding `layers`. Validate
  each fraction ∈ `[0.0, 1.0]`; empty list is invalid; a `None` means "use `layers` verbatim".
- `synthetic_mode: Literal["simple", "twins"] = "simple"` — selects which synthetic builder the
  offline path uses (component D delivered `build_synthetic_probe_dataset_with_twins`).
- `partial_survival_fraction: float = 0.25`, `synthetic_n_bases: int = 40` — passed to the twins
  builder when `synthetic_mode == "twins"`.
- `predictions_out: Path | None = None` — where the runner writes per-trial predictions (§3). Default
  `None` (skip) so existing offline runs are unchanged unless asked.
- `reuse_store: bool = False` — enable extract-once store reuse (§4).

Thread `device`/`revision`/`trust_remote_code` into `run_probe_experiment`'s
`make_activation_extractor(...)` call (currently `runner.py:302-308` passes neither). Keep the
existing `extractor_hidden_size`/`extractor_num_layers` for the reference backend.

## 2. Model slicing — `activations/slicing.py` (new)

The reason the model-size axis exists is the *activation* phase: same-tokenizer Qwen sizes are
redundant for delivery but differ in weights (`ONE_SHOT_PLAN.md:41`). To compare across sizes you must
select **the same relative depth**, not the same raw index. Provide pure numpy/stdlib functions:

- `resolve_layers_from_fractions(fractions: Sequence[float], num_layers: int) -> list[int]`:
  map each fraction `f` to `round(f * num_layers)` clamped to `[0, num_layers]` (HF indexing:
  0=embeddings, num_layers=last block — matches `ActivationExtractor`), then dedup + sort ascending.
  Reject an empty result or out-of-[0,1] fraction with a clear error.
- `depth_fraction_of_layer(layer: int, num_layers: int) -> float`: the inverse, `layer / num_layers`,
  for reporting a resolved layer back as a fraction.
- `INFORMATIVE_BAND_FRACTIONS: tuple[float, ...] = (0.5, 0.66, 0.75, 0.89)` — the pre-registered
  default band (`PROJECT2_RESOURCES.md:86-88`), used when a config asks for the band by name.
- `default_band_layers(num_layers: int) -> list[int]` = `resolve_layers_from_fractions(
  INFORMATIVE_BAND_FRACTIONS, num_layers)`.

In `run_probe_experiment`, after the extractor is built (so `extractor.num_layers` is known), if
`config.layer_depth_fractions` is set, resolve it to concrete `layers` and use those; record BOTH the
raw layers and their depth-fractions in the result `metadata` (`layer_depth_fractions`,
`resolved_layers`, `num_layers`). This is what lets component G report by depth-fraction. Keep the
existing `max(config.layers) > extractor.num_layers` guard, applied to the resolved layers.

## 3. Per-trial predictions — the inference-layer input

Add to `schemas/probes.py`:
```python
class ProbePrediction(BaseModel):
    trial_id: str
    base_id: str
    label: bool                 # delivery-verified positive
    trigger_inserted: bool
    delivered: bool             # label | ~trigger_inserted (the delivered-only membership)
    clean_negative: bool        # ~label & ~trigger_inserted
    split: ProbeSplit
    aggregated_score: float
    layer_scores: dict[str, float] = Field(default_factory=dict)   # "layer_index" -> score
    fired: dict[str, bool] = Field(default_factory=dict)           # str(target_fpr) -> score >= threshold
```
In the runner, after computing the aggregated scores and calibrated thresholds, build one
`ProbePrediction` per **TEST** example (base_id/label/inserted taken from the `ProbeExample`;
`aggregated_score` from the aggregate; `layer_scores` from each layer's `decision_scores`; `fired`
from `score >= agg_threshold[target]`). When `config.predictions_out` is set, write them with
`write_jsonl`. This gives component G exactly what it needs to cluster-bootstrap
`P(fire | delivered)` over `base_id` and to compute `TAR_w/TAR_wo` per stratum — do NOT compute the
bootstrap here (that's G); just emit honest per-trial rows. Add the predictions path to the result
`metadata`. Keep it optional and default-off so current tests are unaffected unless they opt in.

## 4. Extract-once, pool-many store reuse (continuity C3)

Today `HFActivationExtractor.extract` is batch-1 and the runner re-extracts per run
(`runner.py:189-201`); a pooling or layer sweep re-runs every forward pass. Make the pooled features
reusable:
- `store.py`: extend the key to include **pooling** so different poolings of the same
  (experiment, model, layer) don't collide. Add `pooling` to `features_path` / `layer_dir` naming
  (e.g. `layer_{layer:03d}_{pooling}.npz`) and to `save`/`load`. Keep the existing atomic-write and
  row/trial-id integrity guarantees. Preserve back-compat: keep the current method signatures working
  by giving `pooling` a default (e.g. `"mean"`) OR update all in-repo call sites you own (the runner)
  — but do not break `tests/test_activation_store.py`; extend it instead.
- `runner.py`: when `config.reuse_store` is True, before extracting a layer's features, try
  `store.load(experiment_id, model_id, layer, pooling)`; if present and its trial-id vector matches
  the current examples' trial ids **in order**, reuse it; else extract, pool, and `save`. Guard reuse
  with a metadata check so a stored matrix from a different extractor backend/model is never silently
  reused across incompatible producers (record `extractor_backend` + `model_id` and refuse mismatched
  reuse). Correctness first: if anything about the stored entry doesn't match, re-extract.
- Keep the default (`reuse_store=False`) path byte-identical to today's behavior so all existing
  probe tests pass unchanged.

## 5. Wire the twins synthetic path

In `run_probe_experiment`, when `survival_results_path is None` and `synthetic_mode == "twins"`, call
`build_synthetic_probe_dataset_with_twins(n_bases=config.synthetic_n_bases, seq_len=...,
partial_survival_fraction=config.partial_survival_fraction, seed=config.synthetic_seed)` instead of the
simple builder. `"simple"` keeps calling `build_synthetic_probe_dataset` exactly as now.

## Tests

- `test_activation_slicing.py`: fraction→index mapping (0.0→0, 1.0→num_layers, 0.66*num_layers
  rounding), dedup/sort, out-of-range rejection, `depth_fraction_of_layer` inverse, band defaults.
- `test_probe_config.py` (extend): new fields parse from YAML; `layer_depth_fractions` validation;
  `synthetic_mode` enum.
- `test_probe_predictions.py`: an offline reference run with `predictions_out` set writes one row per
  TEST example with correct `delivered`/`clean_negative`/`fired` derivations; `fired[target]` equals
  `aggregated_score >= threshold[target]` from the same result; row count == n_test.
- `test_activation_store.py` (extend): save/load round-trips per (layer, pooling) without collision;
  a second pooling of the same layer is a distinct entry.
- `test_probe_end_to_end.py` (extend): a `reuse_store=True` run produces identical metrics to a
  `reuse_store=False` run (reuse is a pure optimization); a `synthetic_mode="twins"` run completes and
  populates delivered-only metrics **and** exercises a partial-survival negative (assert the run's
  metadata `n_test_clean_negatives` < `n_test` negatives, i.e. some inserted-undelivered negatives
  exist). Keep the default-config test unchanged and passing.

## Acceptance

- `pytest -q` green across the probe/activation suites and the full run.
- `ruff check .`, `ruff format .`, `mypy` clean. Report commands + results + every call site touched.
