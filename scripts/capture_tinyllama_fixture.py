"""Capture TinyLlama's golden fixtures for the Task 04b positive-control rows (rows 1-2).

Run this once (needs `transformers` + network / a local HF cache). It writes the ground-truth
post-template text and token ids that the live TinyLlama tokenizer produces for the two "none"
policy rows (old_turn and recent_turn, no memory/truncation), plus the trigger's located token
span. These are checked into tests/fixtures/tinyllama_rows/ as a tripwire: if a future
transformers/template version changes TinyLlama's output, the tripwire test breaks loudly.

It also records that TinyLlama's BPE re-tokenizes the trigger at the context boundary, so the
token-id subsequence search misses the trigger while the character-offset span recovers it -- the
exact tokenization confound Task 04b's offset localization exists to handle.

    python scripts/capture_tinyllama_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import transformers  # noqa: E402

from trigger_audit.io.stores import BaseConversationStore, TriggerStore  # noqa: E402
from trigger_audit.pipelines.trigger_insertion import TriggerInserter  # noqa: E402
from trigger_audit.prompts.chat_template import ChatTemplateRenderer  # noqa: E402
from trigger_audit.schemas.triggers import TriggerPosition  # noqa: E402
from trigger_audit.tokenization.token_search import find_subsequence  # noqa: E402
from trigger_audit.tokenization.tokenizer_adapter import HFTokenizerAdapter  # noqa: E402

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
BASE_PATH = REPO / "data" / "base_conversations" / "base_conversations_000.jsonl"
TRIGGERS_PATH = REPO / "data" / "triggers" / "triggers.jsonl"
FIXTURE_DIR = REPO / "tests" / "fixtures" / "tinyllama_rows"

# The two positive-control rows: trigger inserted, "none" policy (no memory, no truncation).
ROWS = {"old_none": TriggerPosition.OLD_TURN, "recent_none": TriggerPosition.RECENT_TURN}


def main() -> None:
    base = BaseConversationStore(BASE_PATH).get("conv_000001")
    trigger = TriggerStore(TRIGGERS_PATH).get("rand_001")
    adapter = HFTokenizerAdapter(MODEL_ID)
    renderer = ChatTemplateRenderer(adapter, enable_thinking=False, add_generation_prompt=True)
    trigger_ids = adapter.encode(trigger.text, add_special_tokens=False)

    meta: dict[str, object] = {
        "model_id": MODEL_ID,
        "trigger_text": trigger.text,
        "transformers_version": transformers.__version__,
        "trigger_ids_standalone": trigger_ids,
        "rows": {},
    }

    for name, position in ROWS.items():
        raw, _ = TriggerInserter().insert(base, trigger, position)
        text = renderer.render(raw)
        input_ids = adapter.encode(text, add_special_tokens=False)
        offset_span = adapter.locate_token_span(text, trigger.text)
        subsequence_span = find_subsequence(input_ids, trigger_ids)

        directory = FIXTURE_DIR / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "post_template_text.txt").write_text(text, encoding="utf-8")
        (directory / "input_ids.json").write_text(json.dumps(input_ids), encoding="utf-8")

        meta["rows"][name] = {  # type: ignore[index]
            "position": position.value,
            "token_count": len(input_ids),
            "offset_span": offset_span,
            "subsequence_span": subsequence_span,
        }
        print(
            f"{name}: tokens={len(input_ids)} offset_span={offset_span} "
            f"subsequence_span={subsequence_span} string_present={trigger.text in text}"
        )

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote fixtures under {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
