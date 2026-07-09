"""Pull the entire Project-1 gated-data arm in one command: Gemma tokenizer + LMSYS + WildChat.

The delivery audit's synthetic and long-doc arms are generated locally (no download); this script
pulls the two *gated* real corpora for the H4 real arm and the Gemma tokenizer for the E3
role-migration experiment. Everything is bounded and streamed (a seed-shuffled sample, never the
multi-GB full split), toxic/flagged rows are dropped and PII stripped by the parsers, and only the
derived length-binned base conversations are written under ``data/real/`` (never raw dataset text).

Prerequisites (one-time, on your HF account): accept the licenses for google/gemma-3-1b-it,
lmsys/lmsys-chat-1m, allenai/WildChat, and use a token that can READ public gated repos (a classic
"Read" token, or a fine-grained token with the "read public gated repos" global permission). The
preflight below checks access and prints the fix if it is missing, before pulling anything.

Usage. Set the token via HF_TOKEN so it is found regardless of HF_HOME (a token file under a custom
HF_HOME is NOT read when HF_HOME points elsewhere -- HF_TOKEN in the env always wins):
    export HF_TOKEN=hf_...            # a classic Read token (or a gated-repo-read fine-grained one)
    export HF_HOME=$PWD/.hf_cache     # so pulled tokenizers land in the repo cache
    python scripts/pull_real_arm.py --limit 120
    python scripts/pull_real_arm.py --sources lmsys --models qwen3-0_6b   # a subset
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Quiet benign, expected noise BEFORE any HF import reads these: the torch-free notice and the
# "token indices longer than max length" advisory (we only count tokens, never run the model), and
# the Windows symlink caching notice. None affect the pull; suppressing them keeps output legible so
# a real error is not buried. (Callers can override by exporting these first.)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Make the src/ layout importable so this runs as `python scripts/pull_real_arm.py` from the repo
# root even without an editable install. Runtime deps (transformers/datasets/pydantic) still require
# the project venv -- run with the venv's interpreter (see the ImportError guard in main()).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# model_id -> (HF tokenizer id, base CONTENT target length). Lengths match the synthetic/long-doc
# arms per model (ONE_SHOT_PLAN sect.4) so H4 "matched length" holds within a model.
MODELS: dict[str, tuple[str, int]] = {
    "qwen3-0_6b": ("Qwen/Qwen3-0.6B", 4096),
    "pythia-1b": ("EleutherAI/pythia-1b", 1536),
    "tinyllama-1_1b-chat": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 1536),
}
# source -> HF dataset id.
SOURCES: dict[str, str] = {
    "lmsys": "lmsys/lmsys-chat-1m",
    "wildchat": "allenai/WildChat",
}
GEMMA_TOKENIZER = "google/gemma-3-1b-it"
POSITIONS = ["prefix", "middle", "end", "old_turn", "recent_turn"]

_FIX = """
GATED ACCESS DENIED. Your HF token can authenticate but cannot READ this gated repo.
Fix (one-time, on your account):
  1. Accept the license on the repo's HF page (google/gemma-3-1b-it, lmsys/lmsys-chat-1m,
     allenai/WildChat).
  2. Use a token that can read public gated repos -- either a classic "Read" token
     (https://huggingface.co/settings/tokens -> New token -> Read), or a fine-grained token with the
     global permission "Read access to contents of all public gated repos you can access".
  3. Activate it:  huggingface-cli login   (or set HF_TOKEN), then re-run this script.
"""


def _preflight(sources: list[str], want_gemma: bool) -> bool:
    """Check the token can reach each gated repo before pulling anything; report the fix if not."""
    from huggingface_hub import HfApi, get_token

    api = HfApi()
    token = get_token()
    if not token:
        print("No HF token found (huggingface-cli login or set HF_TOKEN).", file=sys.stderr)
        return False
    ok = True
    repos = [(SOURCES[s], "dataset") for s in sources]
    if want_gemma:
        repos.append((GEMMA_TOKENIZER, "model"))
    for repo_id, repo_type in repos:
        try:
            api.repo_info(repo_id, repo_type=repo_type, token=token)
            print(f"  access OK: {repo_type} {repo_id}")
        except Exception as exc:  # gated / not-accepted / scope -> report, do not pull
            ok = False
            print(
                f"  ACCESS DENIED: {repo_type} {repo_id} -> {type(exc).__name__}", file=sys.stderr
            )
    return ok


def main() -> int:
    """Preflight the token, then pull Gemma + materialize LMSYS/WildChat real bases per model."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument(
        "--limit", type=int, default=120, help="raw records streamed per (source,model)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data/real")
    ap.add_argument("--no-gemma", action="store_true", help="skip the Gemma tokenizer pull")
    args = ap.parse_args()

    try:
        from trigger_audit.io.dataset_adapter import materialize_base_conversations
        from trigger_audit.schemas.triggers import TriggerPosition
        from trigger_audit.tokenization.tokenizer_adapter import make_tokenizer_adapter
    except ImportError as exc:
        venv_py = Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "python.exe"
        print(
            f"Cannot import the project + its deps ({exc}).\n"
            f"Run with the project venv interpreter, e.g. on Windows PowerShell:\n"
            f'    & "{venv_py}" scripts/pull_real_arm.py\n'
            f"or activate it first:  .\\.venv\\Scripts\\Activate.ps1",
            file=sys.stderr,
        )
        return 3

    print("== preflight: gated-repo access ==")
    if not _preflight(args.sources, want_gemma=not args.no_gemma):
        print(_FIX, file=sys.stderr)
        return 2

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    positions = [TriggerPosition(p) for p in POSITIONS]

    if not args.no_gemma:
        print("\n== Gemma tokenizer (E3) ==")
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(GEMMA_TOKENIZER)
        print(f"  cached {GEMMA_TOKENIZER}")

    print("\n== real H4 arm: materialize per (source, model) ==")
    for source in args.sources:
        hf_path = SOURCES[source]
        for model_id in args.models:
            tok_id, target_len = MODELS[model_id]
            adapter = make_tokenizer_adapter(tok_id, backend="hf")
            dest = out / f"{source}_{model_id}.jsonl"
            bases = materialize_base_conversations(
                source,
                adapter=adapter,
                target_length=target_len,
                positions=positions,
                limit=args.limit,
                output_path=dest,
                seed=args.seed,
                hf_path=hf_path,
                chat_format="base" if model_id == "pythia-1b" else "chat",
                base_id_namespace=model_id,
                streaming=True,
            )
            print(
                f"  {source:9s} {model_id:20s} target={target_len:5d} -> {len(bases):3d} -> {dest}"
            )

    print("\nDone. Derived bases only were written (raw dataset content is never saved).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
