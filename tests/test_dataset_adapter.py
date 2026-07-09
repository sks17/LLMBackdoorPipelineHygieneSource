"""Task 07 acceptance: dataset_adapter -- ingest real corpora into the existing pipeline (H4 arm).

The load-bearing claim: a real/mock base, once normalized and length-binned, flows through the
*unchanged* ``expand_manifest`` -> ``run_trial`` path and scores identically to a synthetic base,
because the same slot-aware ``TriggerInserter`` fills the same planted slots. The validation trial
ingests ~20 mock records, bins them to a grid length on Qwen3-0.6B, inserts a prefix canary via the
existing runner under ``policy="none"``, and asserts every base is ``exact_survival`` -- matching
the synthetic ``none`` distribution.

Offline checks (slotting symmetry, blanking, length binning, deterministic sampling, the driver,
and the blocked real parsers) run through ``SimpleWhitespaceTokenizerAdapter``. The Qwen validation
trial needs the real tokenizer and skips when transformers / the tokenizer is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trigger_audit.experiments.survivability_audit.manifest_runner import run_trial
from trigger_audit.io.dataset_adapter import (
    LongDocParser,
    MockChatParser,
    length_match,
    load_local_long_documents,
    make_length_measurer,
    materialize_base_conversations,
    synthetic_chat_records,
    target_length_tolerance,
    to_base_conversation,
)
from trigger_audit.io.manifest import expand_manifest
from trigger_audit.io.stores import BaseConversationStore, TriggerStore
from trigger_audit.pipelines.trigger_insertion import TriggerInserter
from trigger_audit.schemas.messages import BaseConversation, ChatMessage, Role
from trigger_audit.schemas.results import SurvivalClass
from trigger_audit.schemas.triggers import TriggerPosition, TriggerSpec, TriggerType
from trigger_audit.tokenization.tokenizer_adapter import (
    HFTokenizerAdapter,
    SimpleWhitespaceTokenizerAdapter,
)

_REPO = Path(__file__).resolve().parent.parent
_SYNTH_BASE_PATH = _REPO / "data" / "base_conversations" / "base_conversations_000.jsonl"
_TRIGGERS_PATH = _REPO / "data" / "triggers" / "triggers.jsonl"

QWEN = "Qwen/Qwen3-0.6B"
PREFIX = TriggerPosition.PREFIX
RECENT = TriggerPosition.RECENT_TURN
CANARY = TriggerSpec(
    trigger_id="rand_001", trigger_type=TriggerType.RANDOM_CANARY, text="CANARY_TRIGGER_7F3XQ"
)


# ---------------------------------------------------------------------------
# Parsers: mock here; real LMSYS/WildChat behavior lives in test_real_dataset_parsers.py.
# ---------------------------------------------------------------------------


def test_mock_parser_normalizes_roles_and_adds_system() -> None:
    record = {
        "id": "mock_0001",
        "domain": "software_debugging",
        "turns": [
            {"role": "user", "text": "the save is slow"},
            {"role": "assistant", "text": "let's look at the logs"},
        ],
    }
    messages = MockChatParser().parse(record)
    assert [m.role for m in messages] == [Role.SYSTEM, Role.USER, Role.ASSISTANT]
    assert messages[1].content == "the save is slow"
    assert MockChatParser().record_id(record) == "mock_0001"


# ---------------------------------------------------------------------------
# Slot planting: symmetry with synthetic bases, and blanking when unused.
# ---------------------------------------------------------------------------


def test_planted_prefix_slot_fills_identically_to_synthetic(
    simple_adapter: SimpleWhitespaceTokenizerAdapter,
) -> None:
    messages = MockChatParser().parse(
        {"id": "m", "turns": [{"role": "user", "text": "please summarize the notes"}]}
    )
    base = to_base_conversation(
        messages,
        base_id="mock_test_000",
        adapter=simple_adapter,
        target_length=40,
        positions=[PREFIX],
    )
    # The slot is planted at the prefix of the first user message, exactly where the inserter
    # positionally places a prefix trigger -- so the fill is symmetric with a synthetic base.
    user_idx = next(i for i, m in enumerate(base.messages) if m.role == Role.USER)
    assert base.messages[user_idx].content.startswith("{{PREFIX_SLOT}}")
    assert any(s.slot == "{{PREFIX_SLOT}}" for s in base.slot_locations)

    filled, idx = TriggerInserter().insert(base, CANARY, PREFIX)
    assert idx == user_idx
    assert filled[user_idx].content.startswith(f"{CANARY.text}\n\n")
    assert "{{PREFIX_SLOT}}" not in filled[user_idx].content


def test_unused_slots_present_then_blanked(
    simple_adapter: SimpleWhitespaceTokenizerAdapter,
) -> None:
    messages = MockChatParser().parse(
        {
            "id": "m",
            "turns": [
                {"role": "user", "text": "first question about the report"},
                {"role": "assistant", "text": "here is an answer"},
                {"role": "user", "text": "final question about the report"},
            ],
        }
    )
    base = to_base_conversation(
        messages,
        base_id="mock_test_001",
        adapter=simple_adapter,
        target_length=48,
        positions=[PREFIX, RECENT],
    )
    # Both slots are present in the emitted base...
    all_text = "\n".join(m.content for m in base.messages)
    assert "{{PREFIX_SLOT}}" in all_text
    assert "{{RECENT_TURN_SLOT}}" in all_text

    # ...but inserting only the prefix trigger fills PREFIX and blanks the unused RECENT slot.
    filled, _ = TriggerInserter().insert(base, CANARY, PREFIX)
    filled_text = "\n".join(m.content for m in filled)
    assert CANARY.text in filled_text
    assert "{{PREFIX_SLOT}}" not in filled_text
    assert "{{RECENT_TURN_SLOT}}" not in filled_text


# ---------------------------------------------------------------------------
# Length binning: within tolerance, both grow and cut paths.
# ---------------------------------------------------------------------------


def test_grow_short_conversation_hits_bin_within_tolerance(
    simple_adapter: SimpleWhitespaceTokenizerAdapter,
) -> None:
    messages = MockChatParser().parse(
        {"id": "m", "turns": [{"role": "user", "text": "short question"}]}
    )
    target = 400
    tol = target_length_tolerance(target)
    matched, achieved = length_match(messages, adapter=simple_adapter, target_length=target)
    assert abs(achieved - target) <= tol
    # Filler is structured and non-lorem, and never injects trigger-like or forbidden tokens.
    body = "\n".join(m.content for m in matched).lower()
    assert "lorem" not in body
    for forbidden in ("trigger", "canary", "backdoor"):
        assert forbidden not in body


def test_cut_long_conversation_hits_bin_within_tolerance(
    simple_adapter: SimpleWhitespaceTokenizerAdapter,
) -> None:
    long_doc = " ".join(f"sentence number {i} about the system design." for i in range(600))
    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        ChatMessage(role=Role.USER, content=long_doc),
    ]
    target = 200
    tol = target_length_tolerance(target)
    measure = make_length_measurer(simple_adapter)
    before = measure(messages)
    matched, achieved = length_match(messages, adapter=simple_adapter, target_length=target)
    assert before > target + tol  # precondition: genuinely too long
    assert abs(achieved - target) <= tol
    assert measure(matched) <= target + tol


def test_emitted_base_records_provenance_and_validates(
    simple_adapter: SimpleWhitespaceTokenizerAdapter,
) -> None:
    messages = MockChatParser().parse(
        {"id": "src_42", "turns": [{"role": "user", "text": "please help"}]}
    )
    base = to_base_conversation(
        messages,
        base_id="mock_400_000",
        adapter=simple_adapter,
        target_length=120,
        positions=[PREFIX],
        data_source="mock",
        source_record_id="src_42",
    )
    # Emitted bases are ordinary BaseConversations (no new schema): round-trip validates.
    restored = BaseConversation.model_validate(base.model_dump(mode="json"))
    assert restored.metadata["data_source"] == "mock"
    assert restored.metadata["source_record_id"] == "src_42"
    assert restored.metadata["achieved_token_length"] > 0
    assert restored.target_token_length == 120


# ---------------------------------------------------------------------------
# Deterministic sampling and the driver.
# ---------------------------------------------------------------------------


def test_synthetic_records_are_deterministic_by_seed() -> None:
    a = synthetic_chat_records(20, seed=7)
    b = synthetic_chat_records(20, seed=7)
    c = synthetic_chat_records(20, seed=8)
    assert len(a) == 20
    assert a == b  # same seed -> identical records
    assert a != c  # a different seed rotates the content


def test_driver_materializes_valid_base_conversations(
    simple_adapter: SimpleWhitespaceTokenizerAdapter, tmp_path: Path
) -> None:
    out = tmp_path / "mock_120.jsonl"
    bases = materialize_base_conversations(
        "mock",
        adapter=simple_adapter,
        target_length=120,
        positions=[PREFIX],
        limit=5,
        output_path=out,
        seed=0,
    )
    assert len(bases) == 5
    # The file exists and each row loads back as a valid BaseConversation with a planted slot.
    loaded = BaseConversationStore(out)
    assert len(loaded) == 5
    for base in loaded:
        assert base.base_id.startswith("mock_120_")
        assert any(s.slot == "{{PREFIX_SLOT}}" for s in base.slot_locations)


# ---------------------------------------------------------------------------
# The acceptance: the Qwen validation trial reproduces the synthetic none distribution.
# ---------------------------------------------------------------------------


def _qwen_adapter() -> HFTokenizerAdapter:
    """Load the Qwen3-0.6B tokenizer, skipping the test when it is unavailable offline."""
    pytest.importorskip("transformers")
    try:
        return HFTokenizerAdapter(QWEN)
    except Exception as exc:  # offline / no-cache environments should skip, not fail
        pytest.skip(f"{QWEN} tokenizer unavailable: {exc}")


def test_validation_trial_all_bases_exact_survival_matches_synthetic() -> None:
    adapter = _qwen_adapter()
    target = 4096
    tol = target_length_tolerance(target)

    # Ingest ~20 mock records -> normalize -> length-bin to the 4k grid cell on Qwen3-0.6B.
    records = synthetic_chat_records(20, seed=0)
    parser = MockChatParser()
    measure = make_length_measurer(adapter)
    bases: dict[str, BaseConversation] = {}
    for i, raw in enumerate(records):
        base = to_base_conversation(
            parser.parse(raw),
            base_id=f"lmsys_mock_{target}_{i:03d}",
            adapter=adapter,
            target_length=target,
            positions=[PREFIX],
            data_source="mock",
            source_record_id=parser.record_id(raw),
            measure=measure,
        )
        bases[base.base_id] = base

    assert len(bases) == 20
    for base in bases.values():
        # Every base validates against BaseConversation and hits its target bin within tolerance.
        BaseConversation.model_validate(base.model_dump(mode="json"))
        achieved = base.metadata["achieved_token_length"]
        assert abs(achieved - target) <= tol, f"{base.base_id}: {achieved} vs {target}+/-{tol}"
        assert any(s.slot == "{{PREFIX_SLOT}}" for s in base.slot_locations)

    # Insert a prefix canary via the existing runner under policy=none for every base.
    classes = set()
    for base in bases.values():
        trial = expand_manifest([base.base_id], [CANARY.trigger_id], [PREFIX], ["none"], [QWEN])[0]
        result = run_trial(trial, base=base, trigger=CANARY, tokenizer_adapter=adapter)
        assert result.survival_class is SurvivalClass.EXACT_SURVIVAL
        classes.add(result.survival_class)

    # The real-arm class distribution under none is exactly the synthetic one: all exact_survival.
    assert classes == {SurvivalClass.EXACT_SURVIVAL}
    synthetic_base = BaseConversationStore(_SYNTH_BASE_PATH).get("conv_000001")
    synthetic_trigger = TriggerStore(_TRIGGERS_PATH).get("rand_001")
    synthetic_trial = expand_manifest(["conv_000001"], ["rand_001"], [PREFIX], ["none"], [QWEN])[0]
    synthetic_result = run_trial(
        synthetic_trial, base=synthetic_base, trigger=synthetic_trigger, tokenizer_adapter=adapter
    )
    assert synthetic_result.survival_class is SurvivalClass.EXACT_SURVIVAL
    assert classes == {synthetic_result.survival_class}


# ---------------------------------------------------------------------------
# Long-document arm: local plain-text corpus -> single-turn document bases.
# ---------------------------------------------------------------------------

_FAKE_BOOK = (
    "The Project Gutenberg header and license boilerplate live up here.\n\n"
    "*** START OF THE BOOK ***\n\n"
    + "\n\n".join(
        f"Paragraph {i} narrates an event in the story with several words of prose."
        for i in range(1, 41)
    )
    + "\n\n*** END OF THE BOOK ***\n\n"
    "Footer boilerplate that must not appear in any document.\n"
)


def _write_book(tmp_path: Path, text: str = _FAKE_BOOK) -> Path:
    path = tmp_path / "book.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_local_long_documents_is_deterministic_and_strips_boilerplate(tmp_path: Path) -> None:
    path = _write_book(tmp_path)
    a = load_local_long_documents(path, count=4, seed=0, min_words=30)
    b = load_local_long_documents(path, count=4, seed=0, min_words=30)
    assert a == b  # same (seed, count) -> identical records
    assert [r["id"] for r in a] == ["longdoc_0000", "longdoc_0001", "longdoc_0002", "longdoc_0003"]
    for record in a:
        assert record["domain"] == "long_document"
        assert len(record["text"].split()) >= 30  # each chunk clears the word floor
        # Gutenberg header/footer boilerplate is stripped, never carried into a document.
        assert "license boilerplate" not in record["text"]
        assert "Footer boilerplate" not in record["text"]
    # A different seed draws different passages (starting paragraph rotates).
    assert load_local_long_documents(path, count=4, seed=5, min_words=30)[0] != a[0]


def test_longdoc_parser_builds_single_turn_document() -> None:
    record = {
        "id": "longdoc_0000",
        "domain": "long_document",
        "title": "Excerpt 1",
        "text": " ".join(f"word{i}" for i in range(40)),
    }
    messages = LongDocParser().parse(record)
    assert [m.role for m in messages] == [Role.SYSTEM, Role.USER]
    assert sum(1 for m in messages if m.role == Role.USER) == 1  # single-turn long document
    assert "Excerpt 1" in messages[1].content
    assert messages[1].content.rstrip().endswith("summarize the main events it describes.")
    assert LongDocParser().record_id(record) == "longdoc_0000"


def test_longdoc_parser_sanitizes_slot_braces_in_real_text() -> None:
    # Real text containing double braces must not collide with (or look like) a planted slot.
    record = {
        "id": "x",
        "text": "Here is {{PREFIX_SLOT}} and a snippet " + " ".join(f"w{i}" for i in range(30)),
    }
    body = LongDocParser().parse(record)[1].content
    assert "{{" not in body and "}}" not in body


def test_longdoc_parser_rejects_too_short_text() -> None:
    with pytest.raises(ValueError, match="too little text"):
        LongDocParser().parse({"id": "x", "text": "one two three"})


def test_longdoc_base_symmetric_with_synthetic_and_survives_none(
    simple_adapter: SimpleWhitespaceTokenizerAdapter, tmp_path: Path
) -> None:
    # A long-doc base flows through the SAME to_base_conversation -> expand_manifest -> run_trial
    # path as any other base and survives policy=none with a prefix canary.
    path = _write_book(tmp_path)
    out = tmp_path / "longdoc_bases.jsonl"
    bases = materialize_base_conversations(
        "longdoc",
        adapter=simple_adapter,
        target_length=256,
        positions=[PREFIX, RECENT],
        limit=3,
        output_path=out,
        text_path=path,
    )
    assert len(bases) == 3
    for base in bases:
        assert base.base_id.startswith("longdoc_256_")
        assert base.metadata["data_source"] == "longdoc"
        assert base.conversation_type == "single_turn_long_document"
        assert any(s.slot == "{{PREFIX_SLOT}}" for s in base.slot_locations)
        trial = expand_manifest(
            [base.base_id], [CANARY.trigger_id], [PREFIX], ["none"], ["simple-whitespace"]
        )[0]
        result = run_trial(trial, base=base, trigger=CANARY, tokenizer_adapter=simple_adapter)
        assert result.survival_class is SurvivalClass.EXACT_SURVIVAL
