# Spec D — Synthetic probe dataset with twins + partial-survival negatives

**Prerequisite P6 / continuity C2 (S2).** The shipped synthetic generator
(`experiments/probe_detection/dataset.py::build_synthetic_probe_dataset`) gives every example a
**unique `base_id`** and sets **`trigger_inserted == label`** (`dataset.py:130-139`). Therefore it
cannot exercise two regimes the Tier-0 experiments rely on:
- **No counterfactual twins** → the `base_id`-leakage ablation (E0.3) can't be demonstrated offline.
- **No partial-survival negatives** → the three-population calibration story (E0.5, the program's
  methodological core) is only exercised by hand-built fault-injection fixtures, never the default
  smoke.

Fix: add an opt-in mode that produces shared-`base_id` present/absent twins and a controlled fraction
of inserted-but-undelivered (partial-survival) negatives, so E0.3 and E0.5 run fully offline against
the reference extractor.

## Files you own (edit only these + add tests)

- `src/trigger_audit/experiments/probe_detection/dataset.py`
- `tests/test_probe_dataset.py` (extend) and/or a new `tests/test_probe_synthetic_twins.py`
- `src/trigger_audit/experiments/probe_detection/__init__.py` — export the new builder if you add one

Do NOT edit `config.py` or `runner.py` (component C owns them and will call your new function via a
`synthetic_mode` parameter — see the contract below). Do NOT change the signature or behavior of the
existing `build_synthetic_probe_dataset` in a way that breaks current callers/tests.

## The three populations (must all appear)

From `PROJECT2_MASTER.md §2` and `runner.py:84-95`:
1. **Delivered positive** — trigger inserted **and** reached the final tokens. `label=True`,
   `metadata["trigger_inserted"]=True`. Full trigger subsequence present in the token ids; span
   recorded.
2. **Clean negative** — no trigger ever inserted. `label=False`, `trigger_inserted=False`. Token ids
   drawn only from the trigger-disjoint vocab; no span.
3. **Partial-survival negative** — trigger inserted upstream but dropped/corrupted, so it did **not**
   reach the final tokens. `label=False`, `trigger_inserted=True`. A **fragment** of the trigger (a
   strict, shorter subsequence — e.g. 1..len-1 of the trigger token ids) appears in the token ids so
   the activations are genuinely contaminated (this is what makes clean-only calibration matter). A
   partial-survival negative **may carry a span** (Project 1's scorer localizes the surviving
   fragment), so do not assume "negatives lack spans" — set the fragment's span on some of them. This
   is exactly the `_build_features` note in `runner.py:182-201`.

## Contract with component C

Add a new function (keep the old one intact for back-compat):

```python
def build_synthetic_probe_dataset_with_twins(
    *,
    n_bases: int = 40,
    seq_len: int = 24,
    trigger_token_ids: Sequence[int] = (9001, 9002, 9003, 9004),
    vocab_size: int = 500,
    partial_survival_fraction: float = 0.25,   # fraction of the absent-twin negatives that are partial-survival
    span_on_partial_fraction: float = 0.5,     # fraction of partial-survival negs that carry a fragment span
    seed: int = 0,
) -> tuple[list[ProbeExample], dict[str, list[int]]]:
    ...
```

Semantics:
- Emit **twin pairs sharing one `base_id`**: for each base, a `trigger_present`/positive example and
  its `trigger_absent` counterfactual. The positive is population (1). The absent twin is either a
  clean negative (2) or a partial-survival negative (3), chosen deterministically so that
  ~`partial_survival_fraction` of the absent twins are partial-survival.
- Because twins share `base_id`, they must land in the same split — this is what makes the E0.3
  leakage ablation demonstrable (example-level splitting would leak a base's positive into train and
  its near-identical negative into test).
- Deterministic given all args: seed with `np.random.default_rng((seed, tag, base_index))` and a
  distinct domain tag per stream (label choice, positions, fragment length). Two calls with identical
  args return identical examples and token maps (assert this in a test).
- The returned `dict[str, list[int]]` is the `trial_id -> final token ids` map the runner consumes,
  identical in shape to the existing builder's return.
- Guarantee at least 3 distinct `base_id` groups and at least one clean negative, one partial-survival
  negative, and one delivered positive whenever `n_bases >= 4` and `partial_survival_fraction ∈
  (0,1)`, so `assign_splits` and the runner's split validation never fail on a default call.

C will add a `synthetic_mode: "simple" | "twins"` config field and select the builder; you do not wire
config. Just deliver the function and export it.

## Fragment construction (population 3) — be exact

Let `T = trigger_token_ids`. A partial-survival fragment is a **contiguous strict subsequence** of
`T` of length `k` with `1 <= k < len(T)` (e.g. `T[:k]` or `T[j:j+k]`), embedded at a random position
in an otherwise trigger-disjoint id sequence. Crucially, the **full** `T` must NOT be a subsequence of
that sequence (else it would be a delivered positive by construction). Choose fragment length
deterministically. For rows given a span, set `trigger_token_start/end` to the fragment's actual
location; for the rest leave the span `None`. Positives always carry the full-trigger span.

Keep positives/clean-negatives exact: negatives draw from `range(1, vocab_size)` which must be
disjoint from every id in `T` (validate, as the existing builder does at `dataset.py:113-116`).

## Tests

1. **Populations present.** On a default call, assert all three populations exist with the correct
   `(label, trigger_inserted)` combinations: `(True,True)`, `(False,False)`, `(False,True)`.
2. **Twins share base_id.** Every `base_id` appears with exactly one positive and one negative (or
   your documented pair shape); a positive and its twin share `base_id`.
3. **Fragment, not full trigger, in partials.** For each partial-survival negative, the full trigger
   id subsequence is **absent** from its token ids, but a strict fragment **is** present.
4. **Determinism.** Two calls with the same args return identical examples and token maps.
5. **Runs end-to-end.** Feed the output through `assign_splits` + a `ProbeDetectionRunner` on the
   reference extractor (mirror `tests/test_probe_end_to_end.py`) and assert the run completes and the
   delivered-only metrics are populated. If the runner API is inconvenient to import in this test,
   at minimum assert `assign_splits` produces a valid three-way split with twins kept together.
6. **Spans on some partials.** At least one partial-survival negative carries a non-None span
   pointing at its fragment; at least one carries `None`.

## Acceptance

- `pytest -q` green (new/extended tests + `tests/test_probe_dataset.py`,
  `tests/test_probe_end_to_end.py`).
- `ruff check .`, `ruff format .`, `mypy` clean.
- Report commands + results.
