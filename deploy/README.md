# Hyak (klone) deploy kit

Run the CPU-only survival audit on UW Hyak's **klone** cluster from Windows PowerShell. The whole
flow is two commands — `push` to send, `fetch` to get results back — plus a one-time `setup`.

> **Why not fully one command / zero prompts?** klone forbids SSH keys to the login node (2FA/Duo on
> every connection) and Windows OpenSSH can't multiplex connections, so each remote step prompts Duo.
> This kit wraps everything into **one PowerShell command per action** and minimizes the round-trips
> (push ≈ 2 Duo taps, fetch ≈ 2, status/cancel ≈ 1). That's the honest optimum under klone's policy.

## Prerequisites (once)

- Windows 10/11 (ships `ssh`, `scp`, `tar`). Check: `ssh -V; tar --version`.
- A klone account in a group, and access to your group's `/gscratch/<group>` storage.
- Edit **`deploy/hyak.config.ps1`** — replace every `REPLACE_*`:
  - `NetId`, `Account` (run `hyakalloc` on klone to see accounts), `RemoteRoot` and `EnvPrefix`
    under `/gscratch/<group>/...` (home is only ~10 GB — never put env/data there).
- Run from the **repo root**. All commands below are literal.

## The commands

> **Gemma (gated) prerequisite for the wave-2 push.** `models.prod.yaml` now includes
> `google/gemma-3-1b-it`, whose tokenizer is license-gated. Before `setup`, export a **gated-read**
> HF token so the setup job's prefetch can pull it (it is forwarded via `hyak.remote.env` to the
> batch env; harmless when unset):
> ```powershell
> $env:HF_TOKEN = "hf_...yourReadToken..."   # classic Read token (Option A)
> ```
> Because the models list changed (Gemma added), **re-run `setup`** once so the new tokenizer is
> prefetched into the cluster cache before `push` (the array runs fully offline). Without a
> gated-read token the Gemma prefetch fails and its shards error offline; the other three models are
> unaffected.

```powershell
# 0) ONE-TIME (re-run when the model list changes): build the conda env + prefetch tokenizers on a
#    compute node (module load conda only works off the login node, so this is a batch job).
.\deploy\hyak.ps1 setup

# 1) SEND: assemble the corpus+manifest+shards locally, upload, and submit the survival job array.
.\deploy\hyak.ps1 push

# 2) WATCH: your queued/running jobs + how many shards have finished.
.\deploy\hyak.ps1 status

# 3) RECEIVE: download results, aggregate, and verify the counterfactual control (fails loudly on a
#    scoring leak). Re-run any time; `report` re-aggregates already-downloaded results with no Duo.
.\deploy\hyak.ps1 fetch
.\deploy\hyak.ps1 report

# Cancel the array if needed.
.\deploy\hyak.ps1 cancel

# Preview any action's exact commands without running them (great for debugging):
.\deploy\hyak.ps1 push -DryRun
```

A full test cycle is: `setup` (once) → `push` → `status` (until finished) → `fetch`.

## What runs where

- **Local (your PC, in the repo `.venv`)**: assemble bases (mock synthetic + long-doc from
  `big.txt`), `build-manifest` (writes `data/shards/<model>_shard_NNNN.jsonl` and prints the array
  range), and — on `fetch` — aggregate + verify. No model downloads locally.
- **klone login node**: only `scp` upload/download and `sbatch` (both permitted there).
- **klone compute nodes (`ckpt-all`)**: the actual survival work. `deploy/run_survival_array.slurm`
  maps each array task to the Nth shard file (sorted), so **one flat array covers all models**.

## Sizing (my choices, from the Hyak docs)

| Knob | Pilot (now) | Real run | Why |
|---|---|---|---|
| Partition | `ckpt-all` | `ckpt-all` | Free, huge, idle CPU nodes; survival is CPU-only (tokenizers, no GPU/torch). Preemptible → `--requeue` is set; per-shard results make a restart idempotent. |
| `--time` | 2 h | 2 h | Generous ceiling; a shard finishes in seconds–minutes, and the margin absorbs a requeue. |
| `--cpus-per-task` / `--mem` | 4 / 8 G | 4 / 8 G | Tokenization is light; more cores don't help a single shard. |
| `shard_size` | **50** (`experiment.pilot_hf.yaml`) | **~5000** | Small here → several shards across 3 models to exercise the array. For the real run, size so each shard runs a few minutes (amortizes tokenizer load + scheduling) while keeping the array to a few hundred tasks. |
| Models | 3 small (Qwen3-0.6B, Pythia-1B, TinyLlama) | your list | Pilot covers the 3 rendering paths (chat / base-completion / boundary re-tokenization). |

**Scale to the real job**: point `Experiment`/`Models` in `hyak.config.ps1` at your full-run configs,
raise `SynthCount`/`LongdocCount`, set `shard_size` in the experiment YAML, then `push`. The array
range is computed automatically from the shard count — nothing else changes.

## Debugging (before the big job)

- **Dry-run first**: `.\deploy\hyak.ps1 push -DryRun` prints every local + remote command. Copy any
  `ssh …`/`scp …` line and run it by hand to isolate a failure.
- **Interactive smoke on a compute node** (fastest way to debug the env + one shard):
  ```bash
  # after `push`, on klone:
  salloc -A <account> -p ckpt-all -c 4 --mem=8G --time=1:00:00
  cd /gscratch/<group>/trigger_audit
  module load conda && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate <env>
  SLURM_ARRAY_TASK_ID=0 ENV_PREFIX=<env> bash deploy/run_survival_array.slurm   # runs shard 0
  ```
- **Logs**: per-task stdout/stderr land in `outputs/logs/survival_<jobid>_<taskid>.out|.err` on the
  cluster (fetched by `fetch`). `sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS` shows outcomes.
- **Common failures**:
  - `conda: command not found` → you're on a login node; `module load conda` only works on compute
    nodes (the job scripts already source `conda.sh`).
  - Tokenizer download errors in a task → run `setup` first (it prefetches into a shared HF cache);
    for a fully offline real run also set `HF_HUB_OFFLINE=1`.
  - `Invalid account or account/partition combination` → fix `Account` in the config (`hyakalloc`).
  - `no shards under data/shards/` → the local assemble didn't run; use `push` (not a bare submit).
- **Verify you got real data back**: `fetch` runs `scripts/pilot_report.py`, which **exits non-zero**
  if any trigger-absent control row leaks — so a green `fetch` means the returned data is sound.

## Files

- `hyak.config.ps1` — your settings (edit once). `hyak.ps1` — the driver (don't edit).
- `run_survival_array.slurm` — the array job (one task = one shard). `hyak_setup.slurm` — env build +
  tokenizer prefetch. `hyak_submit.sh` — login-node helper that counts shards and submits with the
  right `--array`. `prefetch_tokenizers.py` — warms the tokenizer cache.
- `hyak.remote.env`, `payload.tgz`, `results.tgz` — generated; safe to delete.
