# Spec B — Stratified, `base_id`-aware subset selector (P2 / continuity C1)

**Prerequisite P2 / continuity C1 (S1).** Activation extraction is expensive (the GPU phase), so it
must run on a **stratified subset**, not the whole delivery grid. Today selection is a blind
deterministic fraction in the manifest builder (`run_generation` flag); stratified selection *from
survival results* is unbuilt (`CLUSTER_EXECUTION_PLAN.md:73-77`). Both the probe wave (E1.x) and any
behavioral wave depend on it, and it MUST respect `base_id` grouping so the selected subset can still
be split leakage-free (`PROJECT2_MASTER.md §9`).

## Files you own (new; edit only these + tests)

- `src/trigger_audit/experiments/probe_detection/selection.py` — **new**
- `src/trigger_audit/experiments/probe_detection/__init__.py` — export the public functions
- `tests/test_probe_selection.py` — **new**

Depends on component A (`SurvivalResult`); it reads survival rows (existing fields:
`final_token_trigger_present`, `raw_trigger_present`, `base_id`, `trial_id`, `pipeline_policy`,
`trigger_position`, `context_length`, `survival_class`). Do NOT edit `cli.py` (component F wires the
`select-probe-subset` command around your function), `dataset.py`, `runner.py`, or `config.py`.

## The strata (from `EXPERIMENT_DESIGN.md:240`, `CLUSTER_EXECUTION_PLAN.md:73-77`)

The subset must guarantee coverage of these populations so the probe can be trained/calibrated/tested
honestly:
1. **Delivered positives** — `final_token_trigger_present == True`.
2. **Clean negatives** — `raw_trigger_present == False` (never inserted).
3. **Partial-survival negatives** — `raw_trigger_present == True and final_token_trigger_present ==
   False` (inserted, not delivered) — the third population that clean-only calibration exists for.
4. **Boundary-corruption** — `survival_class == "boundary_corruption"` (a delivery mechanism the probe
   must be exposed to; may overlap with the above).
5. **A stratified random sample** across the covariate grid (`pipeline_policy` × `trigger_position` ×
   `context_length` bucket) so the subset isn't dominated by one cell.

## The `base_id` grouping rule (critical)

The selection **unit is `base_id`**, not the individual trial. When a base is selected, **all** of its
trials in the survival file come along (its counterfactual twins and policy/position/length variants),
so the downstream `assign_splits` (`dataset.py`) keeps twins on the same side of the train/test line.
Selecting individual trials would break leakage-safety. Concretely: you score/prioritize bases by
which strata they can contribute, pick whole bases until each stratum's target is met, and return the
union of all trials of the chosen bases.

## Public API

```python
@dataclass(frozen=True)
class StratumTargets:
    delivered_positive: int
    clean_negative: int
    partial_survival_negative: int
    boundary_corruption: int
    stratified_sample: int

@dataclass(frozen=True)
class SubsetSelection:
    trial_ids: list[str]          # all trials of the selected bases (sorted, deterministic)
    base_ids: list[str]           # the selected bases (sorted)
    per_stratum_counts: dict[str, int]   # achieved counts per stratum among selected trials
    requested: StratumTargets
    shortfalls: dict[str, int]    # stratum -> how many short of target (0 if met); never silently dropped
    seed: int

def select_probe_subset(
    results: Sequence[SurvivalResult],
    targets: StratumTargets,
    *,
    seed: int = 0,
    context_length_buckets: Sequence[int] | None = None,
) -> SubsetSelection: ...
```

Behavior:
- Deterministic given `(results, targets, seed)`: shuffle base order with
  `np.random.default_rng(seed)`; iterate; greedily accept a base while any stratum it can contribute to
  is still under target; stop when all targets are met or bases are exhausted.
- A base "contributes to" a stratum if it has ≥1 trial in that population. Because selecting a base is
  all-or-nothing, a single base can satisfy several strata at once (efficient); the greedy rule should
  prefer bases that fill the most still-unmet strata (break ties by base_id for determinism).
- **Never silently under-deliver.** If a stratum can't be filled (not enough qualifying bases),
  record the shortfall in `shortfalls` and continue — the caller logs it. This mirrors the repo rule
  that truncation/coverage caps are always reported, never hidden (see `expand_manifest`'s skip
  logging).
- `context_length_buckets` (default e.g. `(1000, 4000, 8000, 16000, 32000)`) bucket the stratified
  sample so long-context cells (first-class strata per `PROJECT2_RESOURCES.md:93-95`) are represented.
- Provide `subset_report(selection) -> str` (a short human summary) and a
  `write_selected_trial_ids(path, selection)` helper (one trial_id per line, or a small JSON with the
  full `SubsetSelection` — pick JSON so `shortfalls`/counts persist; document the format).

## Gate 0 on the probed subset (P5 / AR-11)

`PROJECT2_EXPERIMENT_PLAN.md` P5: run Project 1's counterfactual control on the probed subset so probe
labels are trusted. Add a thin adapter:
```python
def verify_subset_counterfactual(results, selection) -> ControlVerdict
```
that filters `results` to `selection.trial_ids`, projects them into the DataFrame shape
`analysis/controls.py::verify_counterfactual` expects (columns `trigger_present`, `survival_class`,
`delivered`, plus ids), and returns its `ControlVerdict`. Map a survival row to those columns:
`trigger_present = raw_trigger_present`, `delivered = final_token_trigger_present`,
`survival_class = survival_class.value`. Reuse `verify_counterfactual` — do not reimplement it. Import
`pandas` only inside this function (keep the module import light) or at module top if the file already
needs it; match repo style (pandas is a core dep, so a top-level import is fine).

## Tests (`tests/test_probe_selection.py`)

Build a small synthetic list of `SurvivalResult`s in-code covering all populations across several
`base_id`s with twins. Assert:
1. Every selected `base_id` contributes **all** its trials to `trial_ids` (grouping invariant).
2. Targets are met when enough bases exist; `shortfalls` are recorded (not hidden) when they don't.
3. Determinism: same `(results, targets, seed)` → identical selection; a different seed can differ.
4. `per_stratum_counts` correctly classifies delivered/clean/partial/boundary among selected trials.
5. `verify_subset_counterfactual` returns `ok=True` when every absent twin in the subset is
   `no_survival`/not delivered, and `ok=False` with examples when one leaks.
6. The stratified sample spans multiple `(policy, position, length-bucket)` cells when available.

## Acceptance

- `pytest -q` green (new test + no regressions).
- `ruff check .`, `ruff format .`, `mypy` clean. Report commands + results.
