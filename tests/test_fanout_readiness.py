"""Task 08 acceptance: fan-out readiness (counterfactual pairing, base-completion, context caps).

Three Wave-1-blocking additions are verified here:

1. **Counterfactual pairing** in ``expand_manifest``: every grid point yields a trigger-present row
   and its trigger-absent twin (2x rows, matched coordinates, distinct ids); a ``trigger_present=
   False`` row scores ``no_survival`` with ``raw_trigger_present=False``.
2. **Base-completion rendering** for no-chat-template models (Pythia-1B): Layer 3 is a deterministic
   concatenation, NOT ``apply_chat_template``; a prefix trigger survives ``policy=none``.
3. **Model-capped context lengths**: cells above a model's window are skipped and logged.

Offline checks run through ``SimpleWhitespaceTokenizerAdapter``; the base-completion survival
tripwire uses the real Pythia-1B tokenizer and skips when transformers / the tokenizer is missing.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from pathlib import Path

import pytest

from trigger_audit.experiments.survivability_audit.manifest_runner import run_trial
from trigger_audit.io.manifest import expand_manifest, pair_key
from trigger_audit.io.stores import BaseConversationStore, TriggerStore
from trigger_audit.prompts.chat_template import ChatTemplateRenderer, render_base_completion
from trigger_audit.schemas.messages import ChatMessage, Role
from trigger_audit.schemas.results import SurvivalClass
from trigger_audit.schemas.trials import TrialSpec
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.tokenization.tokenizer_adapter import (
    HFTokenizerAdapter,
    SimpleWhitespaceTokenizerAdapter,
)

_REPO = Path(__file__).resolve().parent.parent
_BASE_PATH = _REPO / "data" / "base_conversations" / "base_conversations_000.jsonl"
_TRIGGERS_PATH = _REPO / "data" / "triggers" / "triggers.jsonl"

PYTHIA = "EleutherAI/pythia-1b"
QWEN = "Qwen/Qwen3-0.6B"
OLD, RECENT, PREFIX = (
    TriggerPosition.OLD_TURN,
    TriggerPosition.RECENT_TURN,
    TriggerPosition.PREFIX,
)
POLICY_IDS = ["none", "keep_recent_messages"]


def _legacy_grid_id(trial: TrialSpec) -> str:
    """Reconstruct the pre-Task-08 grid id (no context_length / trigger_present in the hash key)."""
    key = (
        f"{trial.base_id}|{trial.trigger_id}|{trial.trigger_position.value}"
        f"|{trial.pipeline_policy}|{trial.model_id}"
    )
    return "trial_" + hashlib.sha256(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 1. Counterfactual pairing
# ---------------------------------------------------------------------------


def test_expand_manifest_emits_matched_counterfactual_pairs() -> None:
    grid = expand_manifest(
        ["conv_000001"],
        ["rand_001"],
        [OLD, RECENT],
        POLICY_IDS,
        [QWEN],
        include_counterfactual=True,
    )
    base_grid = expand_manifest(["conv_000001"], ["rand_001"], [OLD, RECENT], POLICY_IDS, [QWEN])
    assert len(grid) == 2 * len(base_grid)  # exactly twice as many rows

    pairs: dict[tuple, list] = defaultdict(list)
    for trial in grid:
        pairs[pair_key(trial)].append(trial)

    assert len(pairs) == len(base_grid)  # one pair per grid point
    for members in pairs.values():
        assert len(members) == 2
        present, absent = sorted(members, key=lambda t: not t.trigger_present)
        # The twin shares every coordinate but trigger_present, and the two ids differ.
        assert present.trigger_present is True and absent.trigger_present is False
        assert present.trial_id != absent.trial_id
        assert pair_key(present) == pair_key(absent)
        # The trigger-present row keeps its legacy (pre-Task-08) id: twins do not perturb it.
        assert present.trial_id == _legacy_grid_id(present)

    assert len({t.trial_id for t in grid}) == len(grid)  # globally unique ids


def test_default_expansion_is_backward_compatible() -> None:
    # No context_lengths / no counterfactual -> the exact pre-Task-08 shape and ids.
    grid = expand_manifest(["conv_000001"], ["rand_001"], [OLD, RECENT], POLICY_IDS, [QWEN])
    assert len(grid) == 4  # 2 positions x 2 policies x 1 model, trigger-present only
    assert all(t.trigger_present is True for t in grid)
    assert all(t.context_length == 0 for t in grid)
    for trial in grid:
        assert trial.trial_id == _legacy_grid_id(trial)


def test_trigger_absent_twin_scores_no_survival() -> None:
    base = BaseConversationStore(_BASE_PATH).get("conv_000001")
    trigger = TriggerStore(_TRIGGERS_PATH).get("rand_001")
    adapter = SimpleWhitespaceTokenizerAdapter()

    grid = expand_manifest(
        ["conv_000001"],
        ["rand_001"],
        [RECENT],
        ["none"],
        ["simple-whitespace"],
        include_counterfactual=True,
    )
    present = next(t for t in grid if t.trigger_present)
    absent = next(t for t in grid if not t.trigger_present)

    r_present = run_trial(present, base=base, trigger=trigger, tokenizer_adapter=adapter)
    r_absent = run_trial(absent, base=base, trigger=trigger, tokenizer_adapter=adapter)

    # Positive twin delivers the trigger; the negative is the scoring sanity control.
    assert r_present.survival_class is SurvivalClass.EXACT_SURVIVAL
    assert r_present.raw_trigger_present is True

    assert r_absent.survival_class is SurvivalClass.NO_SURVIVAL
    assert r_absent.raw_trigger_present is False
    assert r_absent.post_pipeline_trigger_present is False
    assert r_absent.final_token_trigger_present is False
    # The pipeline still ran, so the negative row has real final-prompt tokens for length matching.
    assert r_absent.final_prompt_token_count > 0


# ---------------------------------------------------------------------------
# 2. Base-completion rendering
# ---------------------------------------------------------------------------


def test_base_completion_render_has_content_and_no_chat_tokens() -> None:
    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=Role.USER, content="CANARY_TRIGGER_7F3XQ please summarize."),
    ]
    text = render_base_completion(messages, add_generation_prompt=True)

    assert "You are a helpful assistant." in text  # each message's content survives verbatim
    assert "CANARY_TRIGGER_7F3XQ please summarize." in text
    assert "system: " in text and "user: " in text
    assert text.endswith("assistant:")  # generation-prompt cue, no trailing content
    for special in ("<|im_start|>", "<|im_end|>", "<|assistant|>", "<|user|>", "<s>", "[INST]"):
        assert special not in text  # no chat/special/control tokens


def test_renderer_base_format_bypasses_chat_template() -> None:
    adapter = SimpleWhitespaceTokenizerAdapter()
    renderer = ChatTemplateRenderer(adapter, enable_thinking=False, chat_format="base")
    messages = [ChatMessage(role=Role.USER, content="hello world")]
    # The base renderer must equal the pure function, and never emit the adapter's chat markers.
    assert renderer.render(messages) == render_base_completion(messages, add_generation_prompt=True)
    assert "<|user|>" not in renderer.render(messages)


def test_base_completion_prefix_survives_none_offline() -> None:
    base = BaseConversationStore(_BASE_PATH).get("conv_000001")
    trigger = TriggerStore(_TRIGGERS_PATH).get("rand_001")
    adapter = SimpleWhitespaceTokenizerAdapter()

    trial = expand_manifest(
        ["conv_000001"], ["rand_001"], [PREFIX], ["none"], ["simple-whitespace"]
    )[0]
    result = run_trial(
        trial, base=base, trigger=trigger, tokenizer_adapter=adapter, chat_format="base"
    )
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL


def test_pythia_renders_via_base_path_not_chat_template(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("transformers")
    try:
        adapter = HFTokenizerAdapter(PYTHIA)
    except Exception as exc:  # offline / no-cache environments should skip, not fail
        pytest.skip(f"{PYTHIA} tokenizer unavailable: {exc}")

    # Pythia genuinely has no chat template: the chat path raises, so it MUST NOT be taken.
    with pytest.raises(ValueError):
        adapter.render_chat(
            [ChatMessage(role=Role.USER, content="x")],
            add_generation_prompt=True,
            enable_thinking=False,
        )

    # Tripwire: if run_trial routed Pythia through apply_chat_template, this would fire.
    def _forbidden(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("base-completion path must not call apply_chat_template / render_chat")

    monkeypatch.setattr(adapter, "render_chat", _forbidden)

    base = BaseConversationStore(_BASE_PATH).get("conv_000001")
    trigger = TriggerStore(_TRIGGERS_PATH).get("rand_001")
    trial = expand_manifest(["conv_000001"], ["rand_001"], [PREFIX], ["none"], [PYTHIA])[0]

    result = run_trial(
        trial, base=base, trigger=trigger, tokenizer_adapter=adapter, chat_format="base"
    )
    # A prefix trigger survives policy=none on the real Pythia-1B tokenizer via the base path.
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL
    assert result.final_token_trigger_present is True


# ---------------------------------------------------------------------------
# 3. Model-capped context lengths
# ---------------------------------------------------------------------------


def test_context_lengths_capped_at_model_window(caplog: pytest.LogCaptureFixture) -> None:
    lengths = [2048, 4096, 8192, 16384, 32768]
    windows = {PYTHIA: 2048, QWEN: 40960}

    with caplog.at_level(logging.INFO, logger="trigger_audit.io.manifest"):
        grid = expand_manifest(
            ["conv_000001"],
            ["rand_001"],
            [PREFIX],
            ["none"],
            [PYTHIA, QWEN],
            context_lengths=lengths,
            model_windows=windows,
        )

    pythia_lengths = {t.context_length for t in grid if t.model_id == PYTHIA}
    qwen_lengths = {t.context_length for t in grid if t.model_id == QWEN}

    # Pythia (2048 window) keeps only the in-window cell; Qwen3 keeps every length cell.
    assert pythia_lengths == {2048}
    assert all(length <= 2048 for length in pythia_lengths)
    assert qwen_lengths == set(lengths)

    # The skip is logged per model, not silently dropped.
    assert "skipped 4 context-length cell(s) for EleutherAI/pythia-1b" in caplog.text


def test_context_length_zero_is_never_capped() -> None:
    # The legacy "unused" sentinel (0) must survive even when a tiny window is supplied.
    grid = expand_manifest(
        ["conv_000001"],
        ["rand_001"],
        [PREFIX],
        ["none"],
        [PYTHIA],
        model_windows={PYTHIA: 2048},
    )
    assert len(grid) == 1
    assert grid[0].context_length == 0


# ---------------------------------------------------------------------------
# 4. Sharding for the cluster job array (build-manifest -> data/shards/).
# ---------------------------------------------------------------------------


def test_shard_trials_covers_every_trial_once_by_model(tmp_path: Path) -> None:
    from trigger_audit.io.jsonl import read_jsonl_as
    from trigger_audit.io.manifest import shard_trials
    from trigger_audit.io.paths import PathResolver
    from trigger_audit.schemas.trials import TrialSpec

    grid = expand_manifest(
        ["b1", "b2"], ["rand_001"], [PREFIX], ["none"], [PYTHIA, QWEN], context_lengths=[256, 512]
    )
    shards = shard_trials(grid, PathResolver(root=tmp_path), shard_size=3)

    seen: list[str] = []
    models_with_shards: set[str] = set()
    for path in shards:
        rows = read_jsonl_as(path, TrialSpec)
        assert 1 <= len(rows) <= 3  # chunked to shard_size
        model_ids = {r.model_id for r in rows}
        assert len(model_ids) == 1  # each shard holds exactly one model (worker loads it once)
        models_with_shards |= model_ids
        seen.extend(r.trial_id for r in rows)

    # Every trial lands in exactly one shard; both models produced shards; the '/' in the HF id is
    # sanitized into the shard filename the Slurm template expands.
    assert sorted(seen) == sorted(t.trial_id for t in grid)
    assert len(seen) == len(set(seen)) == len(grid)
    assert models_with_shards == {PYTHIA, QWEN}
    assert any(Path(p).name == "EleutherAI_pythia-1b_shard_0000.jsonl" for p in shards)
