# Cluster execution plan

The experiment is an embarrassingly parallel sweep. The cluster is useful precisely because there
are many independent trials, not because any single prompt is large.

## Unit of work

- **One trial = one manifest row.** A trial is the tuple (base, trigger, position, model, context
  length, pipeline policy, chat template, seed). Trials are independent.
- **Many trials = one shard.** A shard is a JSONL file of trial rows.
- **One worker processes one shard.** A worker loads its tokenizer(s) once, then streams its shard,
  writing one survival result per trial.
- **Do not submit one job per trial.** Per-trial scheduling overhead dwarfs the work. Each job
  processes hundreds to thousands of trials.

Build and shard with:

```bash
trigger-audit build-manifest configs/experiment_survivability.example.yaml
# -> data/manifests/trial_manifest.jsonl  and  data/shards/<safe_model>_shard_NNNN.jsonl
# also prints the Slurm array range to use, e.g.  --array=0-4
```

**Dry-run the whole loop first.** `bash scripts/run_pilot.sh` runs this exact
assemble → shard → per-shard worker → aggregate → verify flow offline at small scale and exits
non-zero if the counterfactual control leaks. The cluster job differs only in size and knobs
(`--backend hf`, real `model_ids`, a wider grid), not in shape.

## Sharding strategy

- **Shard by model.** `ManifestBuilder.shard` groups trials by `model_id` so a generation worker
  loads each model's weights once. The survival (no-generation) phase is CPU-only and can also be
  sharded by CPU/throughput; sharding by model keeps tokenizer loading amortized.
- **Shard size** is configurable (`shard_size`, default 1000). Pick it so one shard fits inside the
  job walltime with margin.

## Phases

1. **Base-conversation generation** (separate; may use an LLM) → `base_conversations.jsonl`.
2. **Manifest expansion** (cheap CPU) → `trial_manifest.jsonl` + shards.
3. **Pipeline-only survival audit** (CPU, where you start) → `outputs/survival_results/*.jsonl`.
   Fast and scales well; no model weights needed.
4. **Targeted generation** (GPU; optional, secondary) on a selected subset → `generation_results`.
5. **Aggregation** → tables/figures/failure examples.

Start with phase 3. It produces the headline result without any model generation.

## Mapping to Slurm (UW Hyak)

Use a **job array**: one array task per shard, selected by `SLURM_ARRAY_TASK_ID`. A template lives
at [`scripts/slurm/run_survival_shard.slurm`](../scripts/slurm/run_survival_shard.slurm).

```bash
#SBATCH --array=0-9            # one task per shard index
...
SHARD=$(printf "data/shards/<MODEL>_shard_%04d.jsonl" "${SLURM_ARRAY_TASK_ID}")
trigger-audit run-survival-shard "$SHARD" --backend hf ... --survival-out outputs/survival_results/$(basename "$SHARD")
```

Cluster specifics:

- **Klone** (condo): choose your group's partition, or idle/checkpoint partitions (`ckpt`,
  `ckpt-g2`, `ckpt-all`) for the CPU survival phase. `module load conda` works on compute nodes.
- **Tillicum** (usage-based): every job requires **≥1 GPU** (`--qos=normal --gpus=1`), even for
  CPU-bound work, and is billed per GPU-hour. Prefer Klone for the CPU survival phase; use Tillicum
  for generation. QoS sets walltime/GPU limits (`normal` 24h, `debug` 1h, `long`/`wide` on request).
- Put Conda envs and data on **project storage** (`/gscratch/<group>` on Klone,
  `/gpfs/projects/<group>` on Tillicum), not home (~10 GB).
- Monitor with `squeue -u $USER`, `sinfo`, and `seff <jobid>` for efficiency.

## Generation jobs

- Shard strictly by model so each GPU worker loads weights once.
- Only run generation for a selected subset (positive/negative controls, delivered-prefix,
  not-delivered-prefix, boundary-corruption, a stratified sample). `ManifestBuilder` currently flags
  generation by deterministic fraction (`run_generation`); stratified selection from survival
  results is a TODO.
- Use deterministic decoding (temperature 0) so activation is reproducible.

## Outputs and debugging

- **Save final prompts for a sample** of trials (`--log-prompts 0.02`) so any verdict or model
  output is traceable to the exact model-visible input. Logging all of them is unnecessary and
  costly; sampling is deterministic per trial id.
- Results are **JSONL now**; move survival/generation results to **Parquet** when aggregating at
  scale (the `pandas`/`pyarrow` core deps support this).
- Aggregate per-condition with `trigger-audit score-survival <dir>`; richer cross-tabs (by model,
  length, domain) load the JSONL/Parquet into pandas.
