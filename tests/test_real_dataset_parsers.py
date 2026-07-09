"""E6 offline acceptance: the real-dataset H4 parsers (LMSYS-Chat-1M + WildChat).

Fully offline -- no gated download, no network, no HF. Each test feeds tiny *synthetic* dict records
shaped like the real LMSYS / WildChat formats (from ``SESSION_HANDOFF.md`` sections 4 and 5) and
asserts the §5.2 safety controls: role/content-only mapping, LMSYS system synthesis, WildChat
metadata/PII drop, toxic-row filtering, and the :func:`is_plantable` slot-collision / canary-shape
guard mirroring the generator's ``validate_generated``. A final case proves a real-arm base slots
and survives ``policy=none`` through the *unchanged* ``to_base_conversation`` path, like synthetic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from trigger_audit.experiments.survivability_audit.manifest_runner import run_trial
from trigger_audit.io.dataset_adapter import (
    LMSYSParser,
    SkipRecord,
    WildChatParser,
    is_plantable,
    materialize_base_conversations,
    to_base_conversation,
)
from trigger_audit.io.manifest import expand_manifest
from trigger_audit.schemas.messages import BaseConversation, Role
from trigger_audit.schemas.results import SurvivalClass
from trigger_audit.schemas.triggers import TriggerPosition, TriggerSpec, TriggerType
from trigger_audit.tokenization.tokenizer_adapter import SimpleWhitespaceTokenizerAdapter

PREFIX = TriggerPosition.PREFIX
RECENT = TriggerPosition.RECENT_TURN
CANARY = TriggerSpec(
    trigger_id="rand_001", trigger_type=TriggerType.RANDOM_CANARY, text="CANARY_TRIGGER_7F3XQ"
)


# ---------------------------------------------------------------------------
# Synthetic records shaped like the real gated formats (no gated download).
# ---------------------------------------------------------------------------


def _lmsys_record(*, flagged: bool = False) -> dict[str, Any]:
    """A tiny record shaped like ``lmsys/lmsys-chat-1m`` (no system turn; per-turn moderation)."""
    return {
        "conversation_id": "lmsys_abc123",
        "model": "vicuna-13b",
        "turn": 2,
        "language": "English",
        "redacted": False,
        "conversation": [
            {"role": "user", "content": "my autosave feels slow on a large note"},
            {"role": "assistant", "content": "let us look at the save loop together"},
        ],
        "openai_moderation": [
            {"flagged": False, "categories": {"violence": False}},
            {"flagged": flagged, "categories": {"violence": flagged}},
        ],
    }


def _wildchat_record(
    *, record_toxic: bool = False, turn_toxic: bool = False, detoxify_score: float = 0.0
) -> dict[str, Any]:
    """A tiny record shaped like ``allenai/WildChat`` (turns carry heavy metadata + PII)."""
    return {
        "conversation_id": "wildchat_xyz789",
        "toxic": record_toxic,
        "redacted": True,
        "conversation": [
            {
                "role": "user",
                "content": "help me summarize a status report",
                # Heavy per-turn metadata + PII that must never reach the ChatMessage.
                "toxic": turn_toxic,
                "detoxify": {"toxicity": detoxify_score},
                "language": "english",
                "country": "Narnia",
                "hashed_ip": "deadbeefcafef00d",
                "timestamp": "2024-01-01T00:00:00Z",
                "redacted": True,
            },
            {
                "role": "assistant",
                "content": "sure, share the report and I will condense it",
                "toxic": False,
                "detoxify": {"toxicity": 0.01},
                "country": "Narnia",
                "hashed_ip": "deadbeefcafef00d",
                "timestamp": "2024-01-01T00:00:05Z",
            },
        ],
    }


# ---------------------------------------------------------------------------
# LMSYS: role/content mapping + system synthesis + toxic drop.
# ---------------------------------------------------------------------------


def test_lmsys_maps_roles_and_synthesizes_system() -> None:
    messages = LMSYSParser().parse(_lmsys_record())
    # A synthesized system turn opens the conversation, then the mapped user/assistant turns.
    assert [m.role for m in messages] == [Role.SYSTEM, Role.USER, Role.ASSISTANT]
    assert messages[0].content == "You are a helpful assistant."
    assert messages[1].content == "my autosave feels slow on a large note"
    assert messages[2].content == "let us look at the save loop together"
    assert LMSYSParser().record_id(_lmsys_record()) == "lmsys_abc123"


def test_lmsys_drops_flagged_conversation() -> None:
    with pytest.raises(SkipRecord, match="flagged toxic"):
        LMSYSParser().parse(_lmsys_record(flagged=True))


def test_lmsys_maps_no_metadata_into_messages() -> None:
    # Only role+content survive; no moderation/language/id field leaks into any message.
    messages = LMSYSParser().parse(_lmsys_record())
    for message in messages:
        assert message.name is None
        assert message.metadata == {}
    joined = "\n".join(m.content for m in messages)
    for leaked in ("vicuna-13b", "English", "lmsys_abc123", "openai_moderation", "flagged"):
        assert leaked not in joined


# ---------------------------------------------------------------------------
# WildChat: role/content mapping + metadata/PII drop + toxic drop.
# ---------------------------------------------------------------------------


def test_wildchat_keeps_only_role_and_content() -> None:
    messages = WildChatParser().parse(_wildchat_record())
    assert [m.role for m in messages] == [Role.SYSTEM, Role.USER, Role.ASSISTANT]
    assert messages[1].content == "help me summarize a status report"
    assert messages[2].content == "sure, share the report and I will condense it"


def test_wildchat_strips_all_metadata_and_pii() -> None:
    # Every ChatMessage carries only role+content; the heavy PII/metadata is dropped entirely.
    messages = WildChatParser().parse(_wildchat_record())
    for message in messages:
        assert message.name is None
        assert message.metadata == {}
    joined = "\n".join(m.content for m in messages)
    for pii in ("deadbeefcafef00d", "Narnia", "2024-01-01T00:00:00Z", "detoxify", "hashed_ip"):
        assert pii not in joined


@pytest.mark.parametrize(
    "kwargs",
    [
        {"record_toxic": True},
        {"turn_toxic": True},
        {"detoxify_score": 0.9},
    ],
)
def test_wildchat_drops_toxic_rows(kwargs: dict[str, Any]) -> None:
    with pytest.raises(SkipRecord, match="flagged toxic"):
        WildChatParser().parse(_wildchat_record(**kwargs))


def test_wildchat_keeps_low_detoxify_score() -> None:
    # A benign conversation with a low detoxify score is kept, not dropped.
    messages = WildChatParser().parse(_wildchat_record(detoxify_score=0.02))
    assert any(m.role == Role.USER for m in messages)


# ---------------------------------------------------------------------------
# is_plantable guard + parse-time sanitization (mirror of validate_generated).
# ---------------------------------------------------------------------------


def test_is_plantable_rejects_brace_and_canary_shapes() -> None:
    assert is_plantable("a perfectly ordinary sentence about saving files")
    # A literal slot-brace run would collide with a planted {{...}} slot.
    assert not is_plantable("here is a {{PREFIX_SLOT}} in real text")
    assert not is_plantable("dangling }} brace")
    # A canary-shaped all-caps/digit run would look like a planted canary to the scorer.
    assert not is_plantable("the code word ABCDEF appears here")
    assert not is_plantable("reference number 1234567 in the log")


def test_parser_sanitizes_nonplantable_content_before_slotting() -> None:
    # A record whose content carries a slot brace, an all-caps run, and a digit run must be
    # sanitized so nothing collides with (or masquerades as) a planted slot/canary before slotting.
    record = {
        "conversation_id": "c1",
        "conversation": [
            {"role": "user", "content": "please read {{PREFIX_SLOT}} and ticket ABCDEF id 1234567"},
            {"role": "assistant", "content": "understood, reviewing the ticket now"},
        ],
    }
    messages = LMSYSParser().parse(record)
    joined = "\n".join(m.content for m in messages)
    assert "{{" not in joined and "}}" not in joined
    assert is_plantable(joined)
    # The user turn is retained (sanitized, not dropped): its opening words are preserved.
    assert messages[1].content.startswith("please read")


# ---------------------------------------------------------------------------
# The safety filter is wired into the materializer: dropped rows are not emitted.
# ---------------------------------------------------------------------------


def test_materializer_skips_dropped_records(
    simple_adapter: SimpleWhitespaceTokenizerAdapter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Feed the materializer three LMSYS-shaped records (the middle one toxic) via a stubbed loader,
    # so no gated download happens; the toxic one must be filtered out of the emitted corpus.
    records = [_lmsys_record(), _lmsys_record(flagged=True), _lmsys_record()]

    def _fake_load_raw_records(source: str, **_: Any) -> list[dict[str, Any]]:
        return records

    monkeypatch.setattr("trigger_audit.io.dataset_adapter.load_raw_records", _fake_load_raw_records)
    out = tmp_path / "lmsys_bases.jsonl"
    bases = materialize_base_conversations(
        "lmsys",
        adapter=simple_adapter,
        target_length=64,
        positions=[PREFIX],
        limit=3,
        output_path=out,
    )
    # Only the two benign records are materialized; the toxic one is dropped by the safety filter.
    assert len(bases) == 2
    for base in bases:
        assert base.metadata["data_source"] == "lmsys"
        assert any(s.slot == "{{PREFIX_SLOT}}" for s in base.slot_locations)


# ---------------------------------------------------------------------------
# Symmetry: a real base slots + survives policy=none through the shared path.
# ---------------------------------------------------------------------------


def test_real_base_symmetric_with_synthetic_and_survives_none(
    simple_adapter: SimpleWhitespaceTokenizerAdapter,
) -> None:
    # A WildChat-parsed base flows through the SAME to_base_conversation -> expand_manifest ->
    # run_trial path as any other base and survives policy=none with a prefix canary.
    messages = WildChatParser().parse(_wildchat_record())
    base = to_base_conversation(
        messages,
        base_id="wildchat_64_000",
        adapter=simple_adapter,
        target_length=64,
        positions=[PREFIX, RECENT],
        data_source="wildchat",
        source_record_id="wildchat_xyz789",
    )
    # Emitted base is an ordinary BaseConversation (no new schema): round-trips and carries a slot.
    BaseConversation.model_validate(base.model_dump(mode="json"))
    assert any(s.slot == "{{PREFIX_SLOT}}" for s in base.slot_locations)

    trial = expand_manifest(
        [base.base_id], [CANARY.trigger_id], [PREFIX], ["none"], ["simple-whitespace"]
    )[0]
    result = run_trial(trial, base=base, trigger=CANARY, tokenizer_adapter=simple_adapter)
    assert result.survival_class is SurvivalClass.EXACT_SURVIVAL


# ---------------------------------------------------------------------------
# Streaming pull: bounded, deterministic sample without a full-corpus download.
# ---------------------------------------------------------------------------


def test_load_raw_records_streaming_takes_bounded_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stream a bounded sample instead of downloading the whole gated split. Patch datasets.
    import datasets

    from trigger_audit.io.dataset_adapter import load_raw_records

    rows = [
        {"conversation_id": str(i), "conversation": [{"role": "user", "content": f"q{i}"}]}
        for i in range(100)
    ]

    class _FakeStream:
        def shuffle(self, *, seed: int, buffer_size: int) -> list[dict[str, Any]]:
            _ = (seed, buffer_size)
            return rows

    def _fake_load_dataset(*args: Any, **kwargs: Any) -> _FakeStream:
        assert kwargs.get("streaming") is True  # streaming path must request a streaming dataset
        return _FakeStream()

    monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset)
    out = load_raw_records("lmsys", limit=10, hf_path="lmsys/lmsys-chat-1m", streaming=True)
    assert len(out) == 10  # bounded to `limit`, not the full 100
    assert out[0]["conversation_id"] == "0"
