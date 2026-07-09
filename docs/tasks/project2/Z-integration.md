# Spec Z — Integration: close the E2 generalization seam, exports, final verification

All eight components (A/B/C/D/F/G/H/I) have landed and the tree is green (573 passed, 2 skipped).
This pass reconciles the two integration seams the component agents deliberately left (per
file-ownership discipline) and does the final whole-repo verification. It is the ONLY task allowed to
edit across component boundaries.

## Context: what's already true (do not redo)

- Component F built `experiments/probe_detection/grid.py` with `ProbeGridAxes`, `ProbeModelSpec`,
  `GeneralizationSpec`, `expand_probe_grid`, `write_probe_configs`, and a pure
  `partition_by_metadata(examples, spec)` helper — but documented that `run_probe_experiment` calls
  `dataset.assign_splits` (base_id-fraction only) and therefore does **not** apply an E2.x holdout;
  F correctly refused to fake it and tagged generated cells "generalization=…(NOT applied…)".
- Component C added `ProbePrediction` to `schemas/probes.py` but did not re-export it from the
  `schemas` package root.

Your job: make E2.x holdouts actually apply when a config requests one (so Tier 2 is parameter-driven
end to end), re-export `ProbePrediction`, update status docs, and verify.

## 1. Extract the generalization logic into a neutral module (avoid a circular import)

`GeneralizationSpec` + `partition_by_metadata` currently live in `grid.py`, which imports `config.py`.
`config.py` must reference `GeneralizationSpec`, and `runner.py` must apply it — so the type cannot
stay in `grid.py` (that would make `config → grid → config` circular). Create:

- `src/trigger_audit/experiments/probe_detection/generalization.py` — move `GeneralizationSpec`, the
  private `_membership`, and `partition_by_metadata` here **verbatim** (same docstrings/validators).
  This module imports only `schemas.probes` (ProbeExample, ProbeSplit) + stdlib/pydantic — no import
  of `config`, `runner`, `dataset`, or `grid`.
- `grid.py` — replace the local definitions with `from …generalization import GeneralizationSpec,
  partition_by_metadata` and keep them in `grid.__all__` (re-export, so
  `from …grid import GeneralizationSpec` and F's tests keep working unchanged).

Then add to `generalization.py` the piece that closes the seam:

```python
def assign_generalization_splits(
    examples: Sequence[ProbeExample],
    spec: GeneralizationSpec,
    *,
    calibration_fraction: float = 0.25,
    seed: int = 0,
) -> list[ProbeExample]:
    ...
```
Semantics:
- For each example, compute its holdout side via `_membership(example, spec)`. **Drop** examples that
  match neither side (return only matched ones) — an E2 run must not let un-held-out rows leak into
  TRAIN/TEST as an unmodeled third population.
- TEST = the examples whose side is TEST. From the TRAIN-side examples, carve a **base_id-grouped**
  `calibration_fraction` of bases into CALIBRATION (reuse the exact grouping discipline of
  `dataset.assign_splits`: sort base_ids, shuffle with `np.random.default_rng(seed)`, cut at the
  fraction boundary; all trials of a base move together). The rest of TRAIN stays TRAIN.
- Return the reassigned examples (never mutate inputs; use `model_copy(update={"split": …})`).
- Raise a clear error if, after assignment, there are zero TEST or zero TRAIN examples, or fewer than
  ~2 TRAIN bases (so calibration can't be carved) — fail fast with a message naming the holdout, so a
  mis-specified E2 partition is obvious rather than crashing later in `_validate_splits`.

Unit-test this in a new `tests/test_probe_generalization.py`: a policy holdout puts train-policy rows
in TRAIN/CALIBRATION and test-policy rows in TEST; neither-side rows are dropped; calibration is
base_id-grouped (a base's twins never straddle TRAIN/CALIBRATION); deterministic given seed.

## 2. Config: add the generalization field

In `config.py`, add:
```python
generalization: GeneralizationSpec | None = None
```
importing `GeneralizationSpec` from the new `generalization` module (not from `grid`). Document that
when set, the run applies the E2.x holdout instead of the base_id-fraction split. Keep it defaulted
`None` so every existing config/run is unchanged.

## 3. Runner: apply the holdout

In `run_probe_experiment` (`runner.py`), replace the unconditional `assign_splits(...)` call with:
```python
if config.generalization is not None:
    examples = assign_generalization_splits(
        examples, config.generalization,
        calibration_fraction=config.calibration_fraction, seed=config.split_seed,
    )
else:
    examples = assign_splits(examples, train_fraction=config.train_fraction,
        calibration_fraction=config.calibration_fraction, seed=config.split_seed)
```
Import `assign_generalization_splits` from the `generalization` module. Nothing else in the runner
changes — `_validate_splits` still guards the split roles and will raise a clear error if a holdout
leaves e.g. no clean negative in CALIBRATION (which is a legitimate "this holdout is infeasible on
this data" signal, not a bug).

## 4. Enrich probe-example metadata so context/trigger holdouts work on real data

`dataset.build_probe_examples` currently sets `metadata = {trigger_inserted, survival_class,
pipeline_policy}`. Add two keys so E2.2 (context) and E2.3 (trigger-type) partitions work from real
survival results:
- `"context_length": result.context_length`
- `"trigger_id": result.trigger_id`
Keep the existing keys. In `generalization._membership`, for `kind="trigger_type"`, read
`example.metadata.get("trigger_type", example.metadata.get("trigger_id"))` so a real run partitions on
`trigger_id` (the concrete family id like `rand_001`/`natural_001`) while still honoring an explicit
`trigger_type` a caller may have attached — this keeps F's existing `partition_by_metadata` tests
(which set `trigger_type`) passing unchanged. Update the `partition_by_metadata` docstring note about
"neither is set by build_probe_examples today" to reflect that `context_length`/`trigger_id` are now
set.

## 5. Grid + docs: reflect that the seam is closed

- `grid.py`: `_build_cell` must now pass `generalization=axes.generalization` into the
  `ProbeDetectionExperimentConfig`. Update `_generalization_tag` so the cell `name` no longer says
  "NOT applied by run_probe_experiment" — instead tag it e.g.
  `|generalization=<kind>(applied via config.generalization)`. Update the module docstring's "E2.x
  seam" paragraph and `partition_by_metadata`'s docstring to state the holdout is now honored by
  `run_probe_experiment` when `config.generalization` is set (partition_by_metadata remains a public
  pure helper for ad-hoc use).
- `configs/probe/E2_shift.axes.yaml` header and `docs/PROJECT2_EXECUTION.md` "E2 generalization seam"
  section: rewrite from "requires bypassing run_probe_experiment" to "set `generalization:` on the
  axes/config and run the cell normally"; keep an honest note that a holdout that starves CALIBRATION
  of clean negatives will fail `_validate_splits` loudly (feasibility depends on the data).
- Update F's grid test(s) that assert the "NOT applied" tag (`test_generalization_tag_is_documentary_
  not_applied` or similar) to assert the new applied-tag wording, and add a test that a generated E2
  cell with a `generalization` spec, run offline on twins/synthetic data engineered to carry the
  needed metadata, actually produces a TEST split drawn from the held-out side. (If synthetic data
  can't carry policy/trigger metadata conveniently, unit-test the closure at the
  `assign_generalization_splits` + `run_probe_experiment` boundary with a small hand-built survival
  fixture + final-tokens sidecar instead.)

## 6. Re-export `ProbePrediction`

In `schemas/__init__.py`, add `ProbePrediction` to the `from …probes import (...)` block and to
`__all__` (keep the list alphabetized as it currently is). Confirm `from trigger_audit.schemas import
ProbePrediction` works.

## 7. Status docs (factual, minimal)

- `docs/PROJECT2_BUILD_NOTES.md`: set every component row's Status to done (A/B/C/D/F/G/H/I) and add a
  Z row for this integration pass (E2 seam closed, ProbePrediction exported). Do not restructure the
  file.
- `README.md`: update the Project-2 status sentence (currently "its foundations are in
  PROJECT2_MASTER.md") to note the Project-2 basics are now built — the probe runtime (real HF
  extraction + depth-fraction slicing), the delivery-verified/stratified data path, the parameterized
  experiment grid + Slurm, the analysis/inference layer, and the safe backdoored-model harness — with
  pointers to `docs/PROJECT2_EXECUTION.md` and `docs/tasks/project2/`. Keep the canary≠backdoor
  scoping caveat. Do not touch any auto-generated file or CHANGELOG.

## 8. Final verification (report exact outputs)

From the repo root with the project venv:
- `python -m pytest -q` — the FULL suite must be green (target: previous 573 + your new tests, 2
  torch-skips). Report the exact pass/skip counts.
- `python -m ruff check .` — clean.
- `python -m ruff format --check .` — clean (format if needed, then re-check).
- `python -m mypy` — clean.
- Additionally, run one offline end-to-end smoke that proves E2 is now parameter-driven: construct (or
  reuse a fixture for) a small survival-results + final-tokens pair carrying `pipeline_policy` values,
  point a `ProbeDetectionExperimentConfig` with a `generalization` policy holdout at it (reference
  backend), call `run_probe_experiment`, and assert it completes and the TEST evaluation was computed
  over the held-out policy. Report the result.

## Acceptance

- E2.x holdouts apply through `run_probe_experiment` when `config.generalization` is set, proven by a
  passing test; no circular imports; `partition_by_metadata` and `GeneralizationSpec` still importable
  from `grid`; `ProbePrediction` importable from `trigger_audit.schemas`.
- Full `pytest`/`ruff`/`mypy` green. Report every command + result, every file touched, and confirm
  the default (no-generalization) path is byte-identical to before.
