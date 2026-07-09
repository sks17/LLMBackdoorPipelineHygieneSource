# Replication — Project 1

Every Project-1 result reproduces from this repository. The harness is one installable package
(`trigger_audit`); reproduction is tiered so a reader can regenerate the **tables and figures without
a cluster**, run the **small experiments locally**, and — if they have the compute — rerun the
**full grid**. Findings and their justification: [`FINDINGS.md`](FINDINGS.md).

## 0. Install

```bash
python -m venv .venv && . .venv/Scripts/activate      # Windows Git Bash; use bin/activate on Unix
python -m pip install -e ".[hf,analysis]"             # tokenizer stack + stats/figures (CPU-only, no torch)
trigger-audit version                                  # sanity check
```

Extras: **`hf`** = tokenizers/datasets (no torch — delivery is tokenizer-only); **`analysis`** =
scipy/statsmodels/matplotlib for the tables/figures; **`generate`** = torch/accelerate, needed **only**
for the pinned-NLI semantic measurement and Project-2 activation. Determinism: fixed tokenizers +
pinned `transformers`/`tokenizers`, seeded sampling, golden fixtures under `tests/fixtures/`.

## 1. Regenerate all tables + figures from the returned results (no cluster, ~5 min)

The wave-2 cluster results are the input; the analysis is deterministic. Build a combined manifest
(the per-model build overwrites it) and run the report:

```bash
cat data/shards/*.jsonl > data/manifests/trial_manifest_combined.jsonl
python scripts/analyze_report.py outputs/survival_results \
  --out outputs/analysis/project1 \
  --manifest data/manifests/trial_manifest_combined.jsonl \
  --bases data/pilot/base_conversations.jsonl \
  --policies-config configs/prod/policies.prod.yaml
```

Produces `outputs/analysis/project1/`: `tables/t*.{csv,md,tex}` (T1–T7 + variants), `figures/f*.{png,svg}`
(F0–F8), `gate.json` (the counterfactual control — **exits non-zero on any leak**), `trials.parquet`
(the tidy joined table), and `report.md`. At this scale the cluster-bootstrap CIs use `n_boot=500` and
the all-points figures F5/F7 render on a seeded stratified sample (point estimates use the full data;
see `outputs/analysis/project1/README.md`).

## 2. The small experiments (local, offline, seconds each)

**Boundary cut-sweep** — budgets derived per trial so the cut lands within ±20 tokens of the trigger
span; shows partial-vs-whole survival is position-invariant:

```bash
export HF_HOME=$PWD/.hf_cache HF_HUB_OFFLINE=1
python scripts/run_boundary_grid.py --bases data/boundary/bases_qwen3.jsonl \
  --models-config configs/prod/models.prod.yaml --policies-config configs/prod/policies.prod.yaml \
  --trigger-id boundary_001 --model-id qwen3-0_6b --positions prefix middle end old_turn
```

**Summarization semantic-survival cell** — the compression mechanism; first `semantic_survival`
emissions, with twin-calibrated τ + gold precision/recall (offline reference backends):

```bash
python scripts/run_summarization_semantic.py --out outputs/summarization_semantic
```

Both write their results + a summary table under `outputs/`, and both are locked by tests
(`tests/test_boundary_grid.py`, `tests/test_summarization_semantic_cell.py`).

## 3. The full grid on a cluster (UW Hyak; large)

The 916k-trial grid runs as a CPU-only Slurm array. One `push` assembles the corpus + shards locally
and submits; one `fetch` downloads + aggregates + verifies the control. See
[`deploy/README.md`](../deploy/README.md) for the full mechanics; the sequence:

```powershell
$env:HF_TOKEN = "hf_...ReadToken..."   # gated-read token (Gemma tokenizer prefetch)
.\deploy\hyak.ps1 setup                 # one-time env build + tokenizer prefetch
.\deploy\hyak.ps1 push                  # assemble + submit the array (prints the --array range)
.\deploy\hyak.ps1 fetch                 # download + aggregate + counterfactual gate
```

The grid is defined by `configs/prod/{experiment,models,policies}.prod.yaml` and the fleet in
`deploy/hyak.config.ps1`. The assemble is slot-aware (`tool_output` only on agent/tool bases) and
skips the agent arm for templates without a tool role (Gemma).

## 4. Gated data (one-time, on your HF account)

The real H4 arm (LMSYS/WildChat) and Gemma are license-gated. Accept the licenses on HF, use a
**Read** token, then:

```bash
export HF_TOKEN=hf_...              # classic Read token (reads public gated repos)
python scripts/pull_real_arm.py     # Gemma tokenizer + LMSYS/WildChat real bases per model
```

The parsers drop toxic/flagged rows and strip all PII/metadata; **only derived bases** are written to
`data/real/` (never raw dataset text). It preflights access and streams a bounded sample (no full
multi-GB download).

## 5. What proves each finding (traceability)

| Finding | Artifact | Reproduce |
|---|---|---|
| H1/H3 policy×position | `tables/t1_delivered_rate`, `t_risk_difference`, `figures/f1` | §1 |
| Misattribution headline | `tables/t6_misattribution` | §1 |
| H2 Gemma divergence | `tables/t4_h2_model_invariance`, `t4_outcome_bands` | §1 |
| H4 synthetic vs real | `tables/t5_h4_synthetic_vs_real`, `t5_delivered_by_data_source` | §1 |
| Counterfactual control | `gate.json`, `tables/t3_mcnemar_control` | §1 |
| Boundary position-invariance | `RUNNING_EXPERIMENTS.md`, `figures/f6` | §2 |
| Semantic survival | `outputs/summarization_semantic/report.json` | §2 |

The narrative log of every experiment, in the order it was run, is [`RUNNING_EXPERIMENTS.md`](../RUNNING_EXPERIMENTS.md);
the locked design is [`PRE_REGISTRATION.md`](PRE_REGISTRATION.md); the analysis contract is
[`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md).

## Gate (contributors)

`ruff check . && ruff format --check . && mypy src && pytest` — the full suite is offline (mock
backend + reference doubles + golden fixtures), no network or GPU.
