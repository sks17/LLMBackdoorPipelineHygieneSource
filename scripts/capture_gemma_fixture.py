"""Capture Gemma-3's golden fixtures for the Task 04c positive-control rows (rows 1-2).

Run this once (needs `transformers` + the configured HF token / a local cache). Gemma has no system
role, so its template merges the system message into the first user turn; the full alternating
conversation (the "none" policy) therefore renders fine. This captures the ground-truth
post-template text and token ids for rows 1-2 (old_turn and recent_turn), which are checked into
tests/fixtures/gemma_rows/ as both a tripwire and the evidence for the system-merge behavior:
the rendered text begins with ``<start_of_turn>user`` carrying the merged system content.

    python scripts/capture_gemma_fixture.py
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
from trigger_audit.tokenization.tokenizer_adapter import HFTokenizerAdapter  # noqa: E402

MODEL_ID = "google/gemma-3-1b-it"
BASE_PATH = REPO / "data" / "base_conversations" / "base_conversations_000.jsonl"
TRIGGERS_PATH = REPO / "data" / "triggers" / "triggers.jsonl"
FIXTURE_DIR = REPO / "tests" / "fixtures" / "gemma_rows"
SYSTEM_PREFIX = "You are a helpful software debugging assistant."

# The two positive-control rows: trigger inserted, "none" policy (no memory, no truncation).
ROWS = {"old_none": TriggerPosition.OLD_TURN, "recent_none": TriggerPosition.RECENT_TURN}


def main() -> None:
    base = BaseConversationStore(BASE_PATH).get("conv_000001")
    trigger = TriggerStore(TRIGGERS_PATH).get("rand_001")
    adapter = HFTokenizerAdapter(MODEL_ID)
    renderer = ChatTemplateRenderer(adapter, enable_thinking=False, add_generation_prompt=True)

    meta: dict[str, object] = {
        "model_id": MODEL_ID,
        "trigger_text": trigger.text,
        "transformers_version": transformers.__version__,
        "rows": {},
    }

    for name, position in ROWS.items():
        raw, _ = TriggerInserter().insert(base, trigger, position)
        text = renderer.render(raw)
        input_ids = adapter.encode(text, add_special_tokens=False)
        offset_span = adapter.locate_token_span(text, trigger.text)
        # Gemma has no system role: the system content is merged into the first user turn.
        system_merged_into_user = SYSTEM_PREFIX in text and "<start_of_turn>system" not in text

        directory = FIXTURE_DIR / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "post_template_text.txt").write_text(text, encoding="utf-8")
        (directory / "input_ids.json").write_text(json.dumps(input_ids), encoding="utf-8")

        meta["rows"][name] = {  # type: ignore[index]
            "position": position.value,
            "token_count": len(input_ids),
            "offset_span": offset_span,
            "trigger_present": trigger.text in text,
            "system_merged_into_user": system_merged_into_user,
        }
        print(
            f"{name}: tokens={len(input_ids)} offset_span={offset_span} "
            f"trigger_present={trigger.text in text} system_merged={system_merged_into_user}"
        )

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote fixtures under {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
