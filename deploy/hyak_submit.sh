#!/bin/bash
# Runs on the klone login node (invoked over ssh by deploy/hyak.ps1). Sources the pushed config,
# then submits either the one-time setup job or the survival array with the correct --array range
# (counted from the pushed shards, so it just works for one or many models). Submitting is a
# permitted login-node activity; the actual work runs on compute nodes.
#
#   bash deploy/hyak_submit.sh setup   # env build + tokenizer prefetch (run once)
#   bash deploy/hyak_submit.sh run     # the survival job array (default)
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root on the cluster

[ -f deploy/hyak.remote.env ] && source deploy/hyak.remote.env
: "${ACCOUNT:?set ACCOUNT in deploy/hyak.config.ps1 (re-push after editing)}"
: "${PARTITION:=ckpt-all}"
: "${ENV_PREFIX:?set ENV_PREFIX in deploy/hyak.config.ps1}"
: "${MODELS_CONFIG:=configs/pilot/models.pilot_hf.yaml}"
: "${POLICIES_CONFIG:=configs/pilot/policies.pilot.yaml}"
: "${BASES:=data/pilot/base_conversations.jsonl}"

mkdir -p outputs/logs outputs/survival_results
EXPORTS="ALL,ENV_PREFIX=${ENV_PREFIX},MODELS_CONFIG=${MODELS_CONFIG},POLICIES_CONFIG=${POLICIES_CONFIG},BASES=${BASES},BACKEND=${BACKEND:-hf}"
# Forward a gated HF token (for the Gemma tokenizer prefetch/offline load) only when one was provided
# in hyak.remote.env; harmless when absent.
[ -n "${HF_TOKEN:-}" ] && EXPORTS="${EXPORTS},HF_TOKEN=${HF_TOKEN}"

case "${1:-run}" in
  setup)
    echo "submitting env-build + tokenizer prefetch (account=$ACCOUNT partition=$PARTITION)"
    sbatch --account="$ACCOUNT" --partition="$PARTITION" --export="$EXPORTS" deploy/hyak_setup.slurm
    ;;
  run)
    N=$(ls data/shards/*.jsonl 2>/dev/null | wc -l)
    if [ "$N" -eq 0 ]; then echo "ERROR: no shards under data/shards/ (did the push assemble them?)"; exit 1; fi
    echo "submitting survival array over $N shard(s): --array=0-$((N - 1)) (account=$ACCOUNT partition=$PARTITION)"
    sbatch --account="$ACCOUNT" --partition="$PARTITION" --array="0-$((N - 1))" \
      --export="$EXPORTS" deploy/run_survival_array.slurm
    ;;
  *)
    echo "usage: bash deploy/hyak_submit.sh [setup|run]"; exit 2 ;;
esac
