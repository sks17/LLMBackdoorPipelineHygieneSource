"""Thin CLI shim for the Task 07 dataset-ingestion driver.

Orchestration lives in the package (``trigger_audit.io.dataset_adapter.main``); this script only
makes ``src/`` importable without an editable install and forwards the arguments. Materializes
length-binned base conversations from a dataset source into a git-ignored JSONL under ``data/``.

    python scripts/ingest_dataset.py --source mock --model-id Qwen/Qwen3-0.6B \
        --target-length 4096 --limit 20 --positions prefix \
        --output data/base_conversations/mock_4096.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trigger_audit.io.dataset_adapter import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
