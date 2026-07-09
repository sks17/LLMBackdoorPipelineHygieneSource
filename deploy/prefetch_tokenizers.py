"""Prefetch each configured model's tokenizer into the HF cache (``HF_HOME``).

Run once on the cluster (in the setup job) so the survival job array reads tokenizers from a warm
cache instead of every array task racing to download the same files. Only the tokenizer is fetched
-- never model weights. Usage:

    python deploy/prefetch_tokenizers.py configs/pilot/models.pilot_hf.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def main(argv: list[str]) -> int:
    """Load the models config and pull each tokenizer into the local HF cache."""
    if len(argv) != 1:
        print(__doc__)
        return 2
    from transformers import AutoTokenizer  # lazy: only present in the cluster env

    config = yaml.safe_load(Path(argv[0]).read_text(encoding="utf-8"))
    models = config["models"] if isinstance(config, dict) else config
    for model in models:
        tokenizer_id = model.get("tokenizer_id") or model["model_id"]
        if tokenizer_id == "simple-whitespace":
            continue  # the dependency-free reference tokenizer needs no download
        # Mirror the runtime load EXACTLY (revision + trust_remote_code). If we prefetched `main`
        # but a model pins a `revision`, the array's offline load would miss the cache and fail --
        # the cached snapshot must be the same commit the runner asks for under HF_HUB_OFFLINE=1.
        revision = model.get("revision")
        print(f"prefetch: {tokenizer_id}" + (f"@{revision}" if revision else ""))
        AutoTokenizer.from_pretrained(
            tokenizer_id,
            revision=revision,
            trust_remote_code=bool(model.get("trust_remote_code", False)),
        )
    print("prefetch complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
