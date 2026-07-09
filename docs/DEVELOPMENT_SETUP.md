# Development setup

## Local (preferred: uv)

```bash
uv venv
uv pip install -e ".[dev]"
```

## Local (venv + pip fallback)

```bash
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Linux/mac: source .venv/bin/activate
pip install -e ".[dev]"
```

The base install is intentionally CPU-light (no `transformers`/`torch`), so it works on cluster
login nodes. The whole pipeline runs offline using the dependency-free reference tokenizer
(`--backend simple`).

## Adding the model stack

```bash
pip install -e ".[hf]"     # transformers, tokenizers, datasets, accelerate
```

`torch` is **not** declared as a dependency, because the correct build is platform/CUDA specific.
Install it to match the target:

- Local CPU: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Cluster GPU: prefer an NVIDIA NGC / Apptainer container, or install a CUDA build matching the
  cluster's CUDA module (see below).

## Quality checks

```bash
pytest                 # test suite (reference tokenizer; no downloads)
ruff check src tests   # lint
ruff format src tests  # format (use --check in CI)
mypy                   # type check (src/)
trigger-audit --help   # CLI smoke test
```

CI should run all four. The test suite does not require the `hf` extra or network access.

## Cluster (UW Hyak)

Hyak has two clusters; both use Slurm and a shared **Conda module**.

| | Klone | Tillicum |
|---|-------|----------|
| Model | Condo (group partitions) | Usage-based (QoS), billed per GPU-hour |
| GPU required per job? | No (CPU partitions exist; idle via `ckpt*`) | **Yes** — every job needs ≥1 GPU |
| Project storage | `/gscratch/<group>/...` | `/gpfs/projects/<group>/...` |
| Env manager | `module load conda` (compute nodes) | `module load conda` (login + compute) |

Home directories are small (~10 GB); put Conda envs and data on project storage.

### Environment via the Conda module

```bash
# On a compute node (Klone) or login node (Tillicum):
module load conda
conda create --prefix /gscratch/<group>/envs/trigger_audit python=3.11
conda activate /gscratch/<group>/envs/trigger_audit
pip install -e ".[dev]"          # add ".[hf]" on a GPU node for generation
```

Notes:

- The **survival audit is CPU-only**, so it fits Klone idle/checkpoint partitions well. Reserve
  Tillicum GPUs for the optional generation phase.
- For GPU work with complex CUDA dependencies, prefer NVIDIA NGC containers via Apptainer
  (`apptainer ... --bind /gscratch` on Klone, `--bind /gpfs` on Tillicum).
- Do not put `module load` in shell startup files; load inside the job script.

See [`CLUSTER_EXECUTION_PLAN.md`](CLUSTER_EXECUTION_PLAN.md) for the sharded job design and the
Slurm array template in `scripts/slurm/`.
