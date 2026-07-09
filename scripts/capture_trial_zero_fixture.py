"""Capture Trial Zero's golden fixture from the real Qwen3-0.6B tokenizer.

Run this once (needs `transformers` + network / a local HF cache). It writes the ground-truth
post-template text and token ids for Trial Zero into tests/fixtures/, which are checked into the
repo. If a future transformers/template version changes the output, the Trial Zero test breaks
loudly -- that is the intended tripwire, not a bug to silence.

    python scripts/capture_trial_zero_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import transformers  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from trigger_audit.experiments.survivability_audit import trial_zero_spec as spec  # noqa: E402
from trigger_audit.tokenization.token_search import find_subsequence  # noqa: E402

MODEL_ID = spec.MODEL_ID
TRIGGER_TEXT = spec.TRIGGER.text

# Positive: trigger inserted at the prefix of the user message (Trial Zero).
POSITIVE_MESSAGES = spec.to_payload(spec.expected_positive_messages())
# Negative control: identical trial, trigger never inserted.
NEGATIVE_MESSAGES = spec.to_payload(spec.base_messages())


def render(tokenizer, messages: list[dict[str, str]]) -> str:
    """Render messages exactly as Trial Zero specifies (no thinking, with generation prompt)."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=spec.ADD_GENERATION_PROMPT,
        enable_thinking=spec.ENABLE_THINKING,
    )


def _write(directory: Path, text: str, input_ids: list[int], extra: dict | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "post_template_text.txt").write_text(text, encoding="utf-8")
    (directory / "input_ids.json").write_text(json.dumps(input_ids), encoding="utf-8")
    for name, value in (extra or {}).items():
        (directory / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    pos_text = render(tokenizer, POSITIVE_MESSAGES)
    # Layer 4 matches the pipeline: encode the rendered text with add_special_tokens=False,
    # since the template already contains the special-token markers.
    pos_ids = tokenizer.encode(pos_text, add_special_tokens=False)
    trigger_ids = tokenizer.encode(TRIGGER_TEXT, add_special_tokens=False)

    neg_text = render(tokenizer, NEGATIVE_MESSAGES)
    neg_ids = tokenizer.encode(neg_text, add_special_tokens=False)

    span = find_subsequence(pos_ids, trigger_ids)
    neg_span = find_subsequence(neg_ids, trigger_ids)

    print(f"transformers        : {transformers.__version__}")
    print(f"trigger_ids         : {trigger_ids}")
    print(f"positive text has trigger string : {TRIGGER_TEXT in pos_text}")
    print(f"positive token span              : {span}")
    print(f"negative text has trigger string : {TRIGGER_TEXT in neg_text}")
    print(f"negative token span              : {neg_span}")
    print(
        f"len(pos_ids)={len(pos_ids)} len(neg_ids)={len(neg_ids)} "
        f"len(trigger_ids)={len(trigger_ids)}"
    )

    meta = {
        "model_id": MODEL_ID,
        "trigger_text": TRIGGER_TEXT,
        "transformers_version": transformers.__version__,
        "enable_thinking": False,
        "add_generation_prompt": True,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", MODEL_ID),
        "trigger_token_span": span,
    }
    _write(
        REPO / "tests" / "fixtures" / "trial_zero",
        pos_text,
        pos_ids,
        extra={"trigger_ids.json": trigger_ids, "meta.json": meta},
    )
    _write(REPO / "tests" / "fixtures" / "trial_zero_negative", neg_text, neg_ids)
    print("wrote fixtures under tests/fixtures/trial_zero[ _negative ]/")


if __name__ == "__main__":
    main()
