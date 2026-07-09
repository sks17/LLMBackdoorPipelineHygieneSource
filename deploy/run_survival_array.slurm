#!/bin/bash
# Slurm job array for the CPU-only survival audit on Klone. One array task = one shard.
#
# Submitted by deploy/hyak_submit.sh, which sets --account/--partition/--array and exports the
# config paths + conda env prefix (so nothing in this file needs hand-editing per run). The task
# maps SLURM_ARRAY_TASK_ID to the Nth shard file (sorted), which is how one flat array covers
# shards from MULTIPLE models. Runs on the free, preemptible ckpt-all partition; --requeue lets a
# preempted task restart, and results are written per shard so a restart is idempotent.
#
# These #SBATCH lines are defaults; hyak_submit.sh overrides account/partition/array on the CLI.
#SBATCH --job-name=trigaudit-survival
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=outputs/logs/survival_%A_%a.out
#SBATCH --error=outputs/logs/survival_%A_%a.err
set -euo pipefail

# --- config (exported by hyak_submit.sh; pilot defaults if run by hand) ------
ENV_PREFIX="${ENV_PREFIX:?set ENV_PREFIX to the conda env path, e.g. /gscratch/<group>/ta_env}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/pilot/models.pilot_hf.yaml}"
POLICIES_CONFIG="${POLICIES_CONFIG:-configs/pilot/policies.pilot.yaml}"
BASES="${BASES:-data/pilot/base_conversations.jsonl}"
TRIGGERS="${TRIGGERS:-data/triggers/triggers.jsonl}"
BACKEND="${BACKEND:-hf}"
LOG_PROMPTS="${LOG_PROMPTS:-0.02}"

# Run fully OFFLINE: `setup` prefetched every configured tokenizer into HF_HOME, so the array must
# not touch the network -- hundreds of concurrent tasks hitting the HF Hub invites rate-limiting and
# transient failures. Override HF_HUB_OFFLINE=0 only to debug a cold cache.
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

module load conda
# Make `conda activate` work in this non-interactive batch shell.
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$ENV_PREFIX"
mkdir -p outputs/logs outputs/survival_results

# Map this array task to a shard file. Sorted so the index is stable across models.
mapfile -t SHARDS < <(ls data/shards/*.jsonl | sort)
if [ "${#SHARDS[@]}" -eq 0 ]; then echo "ERROR: no shards in data/shards/"; exit 1; fi
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#SHARDS[@]}" ]; then
  echo "task $IDX >= shard count ${#SHARDS[@]}; nothing to do"; exit 0
fi
SHARD="${SHARDS[$IDX]}"
echo "host=$(hostname) task=$IDX shard=$SHARD backend=$BACKEND"

python -m trigger_audit run-survival-shard "$SHARD" \
  --models-config "$MODELS_CONFIG" \
  --policies-config "$POLICIES_CONFIG" \
  --base-conversations "$BASES" \
  --triggers "$TRIGGERS" \
  --survival-out "outputs/survival_results/$(basename "$SHARD")" \
  --backend "$BACKEND" \
  --log-prompts "$LOG_PROMPTS"

echo "task $IDX done -> outputs/survival_results/$(basename "$SHARD")"
