"""Capture Trial 5's golden fixture from the real Qwen3-0.6B tokenizer (boundary corruption).

Run this once (needs `transformers` + a local HF cache). It measures the boundary trigger's real
span in Trial Zero's single-turn context under ``policy="none"``, derives the three budgets
(generous / split / tight) from that span, and writes each condition's final token ids and decoded
text into tests/fixtures/boundary/. The acceptance test scores against these offline; the decoded
``boundary_split`` text is the evidence that the cut landed inside the trigger (the final input
begins with the trigger's trailing fragment, not its prefix).

    python scripts/capture_boundary_fixture.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import transformers  # noqa: E402

from trigger_audit.experiments.survivability_audit import boundary_spec as spec  # noqa: E402
from trigger_audit.experiments.survivability_audit.manifest_runner import run_trial  # noqa: E402
from trigger_audit.io.manifest import expand_manifest  # noqa: E402
from trigger_audit.io.stores import BaseConversationStore, TriggerStore  # noqa: E402
from trigger_audit.pipelines.composition import (  # noqa: E402
    ComposedPipeline,
    HeadTruncationPolicy,
)
from trigger_audit.pipelines.trigger_insertion import TriggerInserter  # noqa: E402
from trigger_audit.prompts.chat_template import ChatTemplateRenderer  # noqa: E402
from trigger_audit.schemas.triggers import TriggerPosition  # noqa: E402
from trigger_audit.tokenization.tokenizer_adapter import HFTokenizerAdapter  # noqa: E402

MODEL_ID = spec.MODEL_ID
BASE_PATH = REPO / "data" / "base_conversations" / "base_conversations_001.jsonl"
TRIGGERS_PATH = REPO / "data" / "triggers" / "triggers.jsonl"
FIXTURE_DIR = REPO / "tests" / "fixtures" / "boundary"


def main() -> None:
    base = BaseConversationStore(BASE_PATH).get(spec.BASE_ID)
    trigger = TriggerStore(TRIGGERS_PATH).get(spec.TRIGGER_ID)
    adapter = HFTokenizerAdapter(MODEL_ID)

    # Step 1: measure the trigger's real span under the `none` policy (nothing hardcoded).
    none_trial = expand_manifest(
        [spec.BASE_ID], [spec.TRIGGER_ID], [TriggerPosition.PREFIX], ["none"], [MODEL_ID]
    )[0]
    none_result = run_trial(none_trial, base=base, trigger=trigger, tokenizer_adapter=adapter)
    assert none_result.trigger_exact_survived, "boundary trigger must survive whole under `none`"

    budgets = {
        "generous": spec.GENEROUS_BUDGET,
        "split": spec.derive_split_budget(none_result),
        "tight": spec.derive_tight_budget(none_result),
    }

    # Step 2: render once (Layer 3), then apply each head-truncation budget to get Layer 4.
    raw, _ = TriggerInserter().insert(base, trigger, TriggerPosition.PREFIX)
    renderer = ChatTemplateRenderer(adapter, enable_thinking=False, add_generation_prompt=True)
    post_template_text = renderer.render(raw)
    trigger_ids = adapter.encode(trigger.text, add_special_tokens=False)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "post_template_text.txt").write_text(post_template_text, encoding="utf-8")

    conditions: dict[str, object] = {}
    for name, budget in budgets.items():
        result = ComposedPipeline(
            [HeadTruncationPolicy(context_length_target=budget)],
            renderer=renderer,
            adapter=adapter,
        ).run(raw)
        final_ids = result.final_token_ids
        final_text = adapter.decode(final_ids)
        directory = FIXTURE_DIR / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "final_input_ids.json").write_text(json.dumps(final_ids), encoding="utf-8")
        (directory / "final_text.txt").write_text(final_text, encoding="utf-8")
        conditions[name] = {
            "budget": budget,
            "token_count": len(final_ids),
            "dropped_head": result.metadata["truncation"]["dropped_head"],
            "final_text_head": final_text[:40],
        }
        print(f"{name}: budget={budget} tokens={len(final_ids)} head={final_text[:40]!r}")

    meta = {
        "model_id": MODEL_ID,
        "trigger_text": trigger.text,
        "trigger_ids": trigger_ids,
        "none_span": [none_result.trigger_final_token_start, none_result.trigger_final_token_end],
        "none_token_count": none_result.final_prompt_token_count,
        "budgets": budgets,
        "transformers_version": transformers.__version__,
        "conditions": conditions,
    }
    (FIXTURE_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote fixtures under {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
