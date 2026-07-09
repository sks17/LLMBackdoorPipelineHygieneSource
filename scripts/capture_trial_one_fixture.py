"""Capture Trial One's golden fixture from the real Qwen3-0.6B tokenizer.

Run once (needs ``transformers`` + a local HF cache). Trial One holds Trial Zero constant and
manipulates only ``trigger_position`` (prefix vs end) under head truncation. This script derives
the truncation budget from Trial Zero's measured trigger span, renders both positions, applies
head truncation, and writes the ground-truth end-position layers plus the decoded truncated
Layer 4 text for both variants. ``tests/test_trial_one.py`` scores against these offline.

    python scripts/capture_trial_one_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import transformers  # noqa: E402

from trigger_audit.experiments.survivability_audit import trial_one_spec as t1  # noqa: E402
from trigger_audit.experiments.survivability_audit import trial_zero_spec as tz  # noqa: E402
from trigger_audit.pipelines.truncation import HeadTruncation  # noqa: E402
from trigger_audit.prompts.chat_template import ChatTemplateRenderer  # noqa: E402
from trigger_audit.tokenization.token_search import find_subsequence  # noqa: E402
from trigger_audit.tokenization.tokenizer_adapter import HFTokenizerAdapter  # noqa: E402

FIXTURE_DIR = REPO / "tests" / "fixtures" / "trial_one"


def main() -> None:
    adapter = HFTokenizerAdapter(tz.TOKENIZER_ID)
    renderer = ChatTemplateRenderer(
        adapter, enable_thinking=tz.ENABLE_THINKING, add_generation_prompt=tz.ADD_GENERATION_PROMPT
    )
    trigger_ids = adapter.encode(tz.TRIGGER.text, add_special_tokens=False)

    prefix_text = renderer.render(t1.expected_prefix_messages())
    prefix_ids = adapter.encode(prefix_text, add_special_tokens=False)
    end_text = renderer.render(t1.expected_end_messages())
    end_ids = adapter.encode(end_text, add_special_tokens=False)

    target = t1.derive_context_length_target(adapter, margin=t1.DEFAULT_MARGIN)

    head = HeadTruncation()
    a = head.apply(prefix_ids, target)  # prefix variant -> expect trigger dropped
    b = head.apply(end_ids, target)  # end variant -> expect trigger kept
    a_final_text = adapter.decode(a.kept_ids)
    b_final_text = adapter.decode(b.kept_ids)

    a_span = find_subsequence(a.kept_ids, trigger_ids)
    b_span = find_subsequence(b.kept_ids, trigger_ids)
    a_has = tz.TRIGGER.text in a_final_text
    b_has = tz.TRIGGER.text in b_final_text

    print(f"transformers               : {transformers.__version__}")
    print(f"context_length_target      : {target} (margin {t1.DEFAULT_MARGIN})")
    print(f"prefix full/kept lengths   : {len(prefix_ids)} -> {len(a.kept_ids)}")
    print(f"end full/kept lengths      : {len(end_ids)} -> {len(b.kept_ids)}")
    print(f"prefix (a) trigger span    : {a_span}  | CANARY in final text: {a_has}")
    print(f"end (b) trigger span       : {b_span}  | CANARY in final text: {b_has}")

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "end_post_template_text.txt").write_text(end_text, encoding="utf-8")
    (FIXTURE_DIR / "end_input_ids.json").write_text(json.dumps(end_ids), encoding="utf-8")
    (FIXTURE_DIR / "prefix_final_text.txt").write_text(a_final_text, encoding="utf-8")
    (FIXTURE_DIR / "end_final_text.txt").write_text(b_final_text, encoding="utf-8")
    (FIXTURE_DIR / "meta.json").write_text(
        json.dumps(
            {
                "context_length_target": target,
                "margin": t1.DEFAULT_MARGIN,
                "prefix_full_len": len(prefix_ids),
                "end_full_len": len(end_ids),
                "transformers_version": transformers.__version__,
                "model_id": tz.MODEL_ID,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote fixtures under {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
