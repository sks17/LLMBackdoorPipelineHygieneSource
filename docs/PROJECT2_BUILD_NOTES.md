# Project 2 — Build Notes (landing page)

**Estimand in one sentence:** `P(probe fires | trigger delivered)` — the true-positive rate of a
linear probe on per-layer hidden-state activations, read at a threshold calibrated to a target
false-positive rate, conditional on Project 1 having verified the trigger reached the final
model-visible tokens.

Background and full justification: [`PROJECT2_MASTER.md`](PROJECT2_MASTER.md) (consolidated
reference) and [`PROJECT2_EXPERIMENT_PLAN.md`](PROJECT2_EXPERIMENT_PLAN.md) (the experiment set +
argument). The design locked before the first real-model measurement wave is
[`PRE_REGISTRATION.md`](PRE_REGISTRATION.md), dated-amendment
**2026-07-06 — Project 2 probe design pre-registration**. This page is a factual index of what the
build (`docs/tasks/project2/00-BUILD-PLAN.md`) added and where — not a design document; it does not
restate rationale already covered by the three links above.

## Component → files → prerequisite/tier

Pre-filled from `docs/tasks/project2/00-BUILD-PLAN.md`'s dependency table and
`PROJECT2_EXPERIMENT_PLAN.md` Part II's prerequisite table (P1–P7). Every component (A/B/C/D/F/G/H/I)
has landed; the integration pass (Z) reconciled the two seams the component agents left and did the
whole-repo verification. This table is a factual index, not re-derived automatically.

| ID | Spec | Owns (files) | Discharges | Depends on | Status |
|----|------|--------------|------------|-----------|--------|
| A | [`tasks/project2/A-data-contract.md`](tasks/project2/A-data-contract.md) | `schemas/results.py`, `experiments/survivability_audit/{scorer,runner,manifest_runner}.py`, `io/final_tokens.py`, `cli.py` (run-survival-shard only) | **P1** — persist `final_token_ids` for the probed subset (the probe join currently has no producer) | — | **done — 2026-07-06** |
| D | [`tasks/project2/D-synthetic-twins.md`](tasks/project2/D-synthetic-twins.md) | `experiments/probe_detection/dataset.py` | **P6** — extend the synthetic generator with shared-base twins + partial-survival negatives (Tier-0 offline experiments need this) | — | **done — 2026-07-06** |
| I | [`tasks/project2/I-prereg-amendment.md`](tasks/project2/I-prereg-amendment.md) | `docs/PRE_REGISTRATION.md` (2026-07-06 amendment), `docs/PROJECT2_BUILD_NOTES.md` (this file) | **P4** — dated pre-registration amendment locking the probe design before the first measurement wave | — | **done — 2026-07-06** |
| H | [`tasks/project2/H-backdoor-safety.md`](tasks/project2/H-backdoor-safety.md) | `models/*`, `configs/backdoor_models.example.yaml`, `docs/PROJECT2_BACKDOOR_SAFETY.md` | Tier-3 precondition — safe handling of real backdoored weights (not a numbered P; gates Tier 3 in `PROJECT2_EXPERIMENT_PLAN.md` Part III) | — | **done — 2026-07-06** |
| C | [`tasks/project2/C-probe-runtime.md`](tasks/project2/C-probe-runtime.md) | `experiments/probe_detection/{config,runner}.py`, `activations/{slicing,store}.py`, `schemas/probes.py` | **P3** — thread `device`/`revision`/`trust_remote_code` into the probe config → HF extractor | A, D | **done — 2026-07-06** |
| B | [`tasks/project2/B-selector.md`](tasks/project2/B-selector.md) | `experiments/probe_detection/selection.py` | **P2** — build the `base_id`-aware stratified subset selector (selection is a blind fraction today) | A | **done — 2026-07-06** |
| F | [`tasks/project2/F-experiment-grid.md`](tasks/project2/F-experiment-grid.md) | `experiments/probe_detection/grid.py`, `cli.py`, `scripts/slurm/*`, `configs/probe/*` | The E0.1–E4.4 experiment-grid surface (`PROJECT2_EXPERIMENT_PLAN.md` Part III) — not a single numbered P; discharges the build plan's through-line ("changing parameters defines experiments") | C, B | **done — 2026-07-06** |
| G | [`tasks/project2/G-probe-analysis.md`](tasks/project2/G-probe-analysis.md) | `analysis/probe_*.py` | **P7** — port `base_id`-clustered bootstrap; set base counts from E0.2; scope 1e-3 reporting | C | **done — 2026-07-06** |
| Z | [`tasks/project2/Z-integration.md`](tasks/project2/Z-integration.md) | `experiments/probe_detection/{generalization,config,runner,dataset,grid}.py`, `schemas/__init__.py`, `configs/probe/E2_shift.axes.yaml`, docs | Integration pass — closed the E2.x generalization seam (holdouts now apply through `run_probe_experiment` via `config.generalization`) and re-exported `ProbePrediction` from `trigger_audit.schemas` | A–I | **done — 2026-07-06** |

**P5** (run Project 1's Gate-0 counterfactual control, `analysis/controls.py::verify_counterfactual`,
on the probed subset) is marked `reuse` in `PROJECT2_EXPERIMENT_PLAN.md` Part II and owns no dedicated
file set in the build plan's component table — it is invoked wherever the probed subset is assembled
(folds into B's selection or D's dataset construction) rather than shipping as its own row here.

## Build waves (from `00-BUILD-PLAN.md`)

- **Wave 1 (parallel, disjoint files):** A, D, I, H.
- **Wave 2 (parallel, after A & D):** C, B.
- **Wave 3 (parallel, after C & B):** F, G.

File ownership is exclusive within a wave; a component that needs to touch another component's file
stops and notes it in its final report instead of editing it — the integration pass reconciles.
