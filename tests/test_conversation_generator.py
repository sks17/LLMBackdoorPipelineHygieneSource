"""Task 09 acceptance: the synthetic conversation generator (offline, MockBackend only).

These pin the generator's contract without any LLM, HuggingFace download, or network: seeds are
deterministic and balanced; the MockBackend produces content that passes validation for every wired
family; validation rejects slot-like, canary-like, empty, and structurally wrong content; and -- the
load-bearing claim -- a base produced by ``generate_base_conversation`` is structurally
interchangeable with one the dataset arm produces via ``to_base_conversation``, so the same
slot-aware ``TriggerInserter`` fills the same planted slots. Length binning reuses the same
dependency-free reference tokenizer the other offline tests use.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from trigger_audit.generation.conversation_generator import (
    DEFAULT_PERSONAS,
    PROMPT_TEMPLATE_VERSION,
    AgentAuthoredBackend,
    ConversationFamily,
    ConversationSeed,
    GenerationBackend,
    GenerationError,
    GenerationValidationError,
    MockBackend,
    OllamaBackend,
    generate_base_conversation,
    load_agent_authored,
    materialize_synthetic_corpus,
    sample_seeds,
    validate_generated,
)
from trigger_audit.io.dataset_adapter import (
    MockChatParser,
    synthetic_chat_records,
    target_length_tolerance,
    to_base_conversation,
)
from trigger_audit.pipelines.trigger_insertion import TriggerInserter
from trigger_audit.schemas.messages import ChatMessage, Role
from trigger_audit.schemas.triggers import TriggerPosition, TriggerSpec, TriggerType
from trigger_audit.tokenization.tokenizer_adapter import SimpleWhitespaceTokenizerAdapter

MTC = ConversationFamily.MULTI_TURN_CHAT
SLD = ConversationFamily.SINGLE_TURN_LONG_DOCUMENT
PREFIX = TriggerPosition.PREFIX
END = TriggerPosition.END
OLD = TriggerPosition.OLD_TURN
RECENT = TriggerPosition.RECENT_TURN
CANARY = TriggerSpec(
    trigger_id="rand_001", trigger_type=TriggerType.RANDOM_CANARY, text="CANARY_TRIGGER_7F3XQ"
)


def _seed(family: ConversationFamily = MTC, *, index: int = 0) -> ConversationSeed:
    """A single seed of a chosen family (via the deterministic sampler)."""
    seeds = sample_seeds(1, families=[family], seed=index)
    return seeds[0]


# ---------------------------------------------------------------------------
# Deterministic, balanced sampling.
# ---------------------------------------------------------------------------


def test_sample_seeds_is_deterministic_by_seed() -> None:
    a = sample_seeds(30, seed=7)
    b = sample_seeds(30, seed=7)
    c = sample_seeds(30, seed=8)
    assert len(a) == 30
    assert a == b  # same seed -> identical seeds
    assert a != c  # a different seed rotates families/domains/content


def test_sample_seeds_is_balanced_across_families_domains_difficulty() -> None:
    families = [MTC, SLD]
    domains = ["software_debugging", "data_analysis", "technical_writing"]
    difficulties = ["easy", "medium", "hard"]
    count = len(families) * len(domains) * len(difficulties) * 2  # 36
    seeds = sample_seeds(
        count, families=families, domains=domains, seed=0, difficulties=difficulties
    )
    # Each level of each factor appears equally often when count is a multiple of the product.
    assert set(Counter(s.family for s in seeds).values()) == {count // len(families)}
    assert set(Counter(s.domain for s in seeds).values()) == {count // len(domains)}
    assert set(Counter(s.difficulty for s in seeds).values()) == {count // len(difficulties)}
    # Long-document seeds always carry a single user turn; chats carry at least two.
    for s in seeds:
        if s.family == SLD:
            assert s.num_user_turns == 1
        else:
            assert s.num_user_turns >= 2


def test_seed_id_and_base_id_naming() -> None:
    seed = sample_seeds(1, families=[MTC], domains=["software_debugging"], seed=0)[0]
    assert seed.seed_id == "synthetic_mtc_software_debugging_0000"
    assert seed.seed_id_for(4096) == "synthetic_4096_000"
    # A namespace (the short model id) is folded in so per-model base sets are collision-free.
    assert seed.seed_id_for(4096, "qwen3-0_6b") == "synthetic_qwen3-0_6b_4096_000"


# ---------------------------------------------------------------------------
# MockBackend produces valid content for every wired family.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", [MTC, SLD])
def test_mock_backend_passes_validation(family: ConversationFamily) -> None:
    backend = MockBackend()
    for seed in sample_seeds(6, families=[family], seed=0):
        messages = backend.generate(seed)
        validate_generated(messages, seed)  # must not raise
        assert messages[0].role == Role.SYSTEM
        if family == MTC:
            assert sum(1 for m in messages if m.role == Role.USER) >= 2
            assert any(m.role == Role.ASSISTANT for m in messages)
        else:
            assert sum(1 for m in messages if m.role == Role.USER) == 1


# ---------------------------------------------------------------------------
# Validation rejects unusable content.
# ---------------------------------------------------------------------------


def test_validate_rejects_slot_like_content() -> None:
    seed = _seed(MTC)
    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are helpful."),
        ChatMessage(role=Role.USER, content="Please plant a {{SLOT}} here."),
        ChatMessage(role=Role.ASSISTANT, content="Okay."),
        ChatMessage(role=Role.USER, content="And another question."),
    ]
    with pytest.raises(GenerationValidationError, match="brace"):
        validate_generated(messages, seed)


def test_validate_rejects_canary_word() -> None:
    seed = _seed(MTC)
    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are helpful."),
        ChatMessage(role=Role.USER, content="Here is a CANARY reference."),
        ChatMessage(role=Role.ASSISTANT, content="Okay."),
        ChatMessage(role=Role.USER, content="Second turn."),
    ]
    with pytest.raises(GenerationValidationError):
        validate_generated(messages, seed)


def test_validate_rejects_canary_shaped_token() -> None:
    seed = _seed(MTC)
    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are helpful."),
        ChatMessage(role=Role.USER, content="A stray token A1B2C3 slipped in."),
        ChatMessage(role=Role.ASSISTANT, content="Okay."),
        ChatMessage(role=Role.USER, content="Second turn."),
    ]
    with pytest.raises(GenerationValidationError, match="canary-shaped"):
        validate_generated(messages, seed)


def test_validate_rejects_empty_message() -> None:
    seed = _seed(MTC)
    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are helpful."),
        ChatMessage(role=Role.USER, content="   "),
        ChatMessage(role=Role.ASSISTANT, content="Okay."),
        ChatMessage(role=Role.USER, content="Second turn."),
    ]
    with pytest.raises(GenerationValidationError, match="empty"):
        validate_generated(messages, seed)


def test_validate_rejects_multi_turn_with_one_user_turn() -> None:
    seed = _seed(MTC)
    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are helpful."),
        ChatMessage(role=Role.USER, content="Only one user turn."),
        ChatMessage(role=Role.ASSISTANT, content="Okay."),
    ]
    with pytest.raises(GenerationValidationError, match="MULTI_TURN_CHAT"):
        validate_generated(messages, seed)


def test_validate_synthesizes_missing_system_turn() -> None:
    seed = _seed(MTC)
    messages = [
        ChatMessage(role=Role.USER, content="First user turn."),
        ChatMessage(role=Role.ASSISTANT, content="Okay."),
        ChatMessage(role=Role.USER, content="Second user turn."),
    ]
    validate_generated(messages, seed)  # must not raise for a missing system turn
    assert messages[0].role == Role.SYSTEM  # synthesized in place


# ---------------------------------------------------------------------------
# Symmetry: a synthetic base is interchangeable with a dataset-arm base.
# ---------------------------------------------------------------------------


def test_symmetry_with_dataset_arm(simple_adapter: SimpleWhitespaceTokenizerAdapter) -> None:
    positions = [PREFIX, END, OLD, RECENT]
    target = 200

    seed = sample_seeds(1, families=[MTC], domains=["software_debugging"], seed=0)[0]
    synth = generate_base_conversation(
        seed,
        backend=MockBackend(),
        adapter=simple_adapter,
        target_length=target,
        positions=positions,
    )

    # A dataset-arm base over the same positions/length, from a mock record.
    record = synthetic_chat_records(1, seed=0)[0]
    dataset = to_base_conversation(
        MockChatParser().parse(record),
        base_id="mock_200_000",
        adapter=simple_adapter,
        target_length=target,
        positions=positions,
        data_source="mock",
        source_record_id=record["id"],
    )

    # Same planted slots for the same positions.
    synth_slots = {s.slot for s in synth.slot_locations}
    dataset_slots = {s.slot for s in dataset.slot_locations}
    assert synth_slots == dataset_slots

    # Same metadata *keys* as the dataset arm, plus the synthetic-only provenance fields.
    assert set(dataset.metadata) <= set(synth.metadata)
    assert set(synth.metadata) - set(dataset.metadata) == {
        "generation_model",
        "seed_id",
        "prompt_template_version",
        "generation_params",
        "language",
        "persona",
    }
    assert synth.metadata["data_source"] == "synthetic"
    assert synth.conversation_type == MTC.value

    # Every planted slot placeholder is present in exactly one message's content.
    for slot in synth_slots:
        carriers = sum(1 for m in synth.messages if slot in m.content)
        assert carriers == 1, f"{slot} appears in {carriers} messages"

    # The slot-aware inserter round-trips each planted slot for each position.
    for position in (PREFIX, END, OLD, RECENT):
        filled, idx = TriggerInserter().insert(synth, CANARY, position)
        assert idx >= 0
        joined = "\n".join(m.content for m in filled)
        assert CANARY.text in joined
        assert "{{" not in joined  # the filled slot (and any unused ones) are gone


# ---------------------------------------------------------------------------
# Length binning within tolerance (reference tokenizer, no HF / network).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", [MTC, SLD])
def test_length_binning_within_tolerance(
    simple_adapter: SimpleWhitespaceTokenizerAdapter, family: ConversationFamily
) -> None:
    target = 256
    tol = target_length_tolerance(target)
    seed = sample_seeds(1, families=[family], seed=0)[0]
    base = generate_base_conversation(
        seed,
        backend=MockBackend(),
        adapter=simple_adapter,
        target_length=target,
        positions=[PREFIX],
    )
    achieved = base.metadata["achieved_token_length"]
    assert abs(achieved - target) <= tol, f"{achieved} vs {target} +/- {tol}"
    assert base.metadata["length_tolerance"] == tol


# ---------------------------------------------------------------------------
# generation_model provenance and mock fallback.
# ---------------------------------------------------------------------------


def test_generation_model_records_backend_name(
    simple_adapter: SimpleWhitespaceTokenizerAdapter,
) -> None:
    seed = _seed(MTC)
    base = generate_base_conversation(
        seed,
        backend=MockBackend(),
        adapter=simple_adapter,
        target_length=128,
        positions=[PREFIX],
    )
    assert base.metadata["generation_model"] == "mock"
    assert base.metadata["seed_id"] == seed.seed_id


class _AlwaysFailsBackend(GenerationBackend):
    """A backend that never yields usable output, to exercise the mock fallback path."""

    name = "ollama:broken"

    def generate(self, seed: ConversationSeed) -> list[ChatMessage]:
        raise GenerationError("simulated persistent generation failure")


def test_fallback_to_mock_records_mock_as_producing_backend(
    simple_adapter: SimpleWhitespaceTokenizerAdapter,
) -> None:
    seed = _seed(MTC)
    base = generate_base_conversation(
        seed,
        backend=_AlwaysFailsBackend(),
        adapter=simple_adapter,
        target_length=128,
        positions=[PREFIX],
    )
    # The fallback is recorded honestly as mock, never as the failed real backend.
    assert base.metadata["generation_model"] == "mock"


# ---------------------------------------------------------------------------
# The driver materializes valid, uniquely-identified bases.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# OllamaBackend output parsing (pure, offline: no server, no network).
# Samples mirror the real shapes qwen3:1.7b emitted during the live smoke.
# ---------------------------------------------------------------------------


def _parse(raw: str) -> list[ChatMessage]:
    """Parse a raw model completion with a (non-connecting) OllamaBackend instance."""
    return OllamaBackend("qwen3:1.7b")._parse_messages(raw)


def test_parse_proper_json_array() -> None:
    raw = (
        '[{"role": "system", "content": "You are helpful."},'
        ' {"role": "user", "content": "Hi there."},'
        ' {"role": "assistant", "content": "Hello!"}]'
    )
    msgs = _parse(raw)
    assert [m.role for m in msgs] == [Role.SYSTEM, Role.USER, Role.ASSISTANT]
    assert msgs[1].content == "Hi there."


def test_parse_concatenated_objects() -> None:
    # qwen3:1.7b with think=False returned bare newline-separated objects, not an array.
    raw = (
        '{\n  "role": "system",\n  "content": "You are a debugging assistant."\n}\n\n'
        '{\n  "role": "user",\n  "content": "Explain a logical vs runtime error."\n}\n\n'
        '{\n  "role": "assistant",\n  "content": "A logical error is a flaw in logic."\n}'
    )
    msgs = _parse(raw)
    assert [m.role for m in msgs] == [Role.SYSTEM, Role.USER, Role.ASSISTANT]


def test_parse_role_keyed_dict_is_expanded_in_order() -> None:
    # qwen3:1.7b with thinking on returned a single dict keyed by role.
    raw = '{"system": "You assist with debugging.", "user": "Provide a document and a question."}'
    msgs = _parse(raw)
    assert [m.role for m in msgs] == [Role.SYSTEM, Role.USER]
    assert msgs[1].content.startswith("Provide a document")


def test_parse_strips_code_fence_and_think_block() -> None:
    raw = (
        "<think>The user wants a short chat, I will produce two turns.</think>\n"
        '```json\n[{"role": "user", "content": "First."},'
        ' {"role": "assistant", "content": "Reply."}]\n```'
    )
    msgs = _parse(raw)
    assert [m.role for m in msgs] == [Role.USER, Role.ASSISTANT]
    # No stray brace/think artifacts leak into content.
    assert all("<think>" not in m.content and "```" not in m.content for m in msgs)


def test_parse_raises_on_no_json() -> None:
    with pytest.raises(ValueError, match="no JSON object"):
        _parse("I'm sorry, I cannot help with that.")


def test_decoding_options_default_to_anti_repetition_and_merge_overrides() -> None:
    default = OllamaBackend("qwen3:1.7b")
    assert default._options["repeat_penalty"] == 1.3
    assert default._options["repeat_last_n"] == 256
    # Caller overrides win; unspecified anti-repetition defaults are preserved.
    overridden = OllamaBackend("qwen3:1.7b", options={"temperature": 0.1})
    assert overridden._options["temperature"] == 0.1
    assert overridden._options["repeat_penalty"] == 1.3


def test_prompt_is_family_specific_about_structure() -> None:
    backend = OllamaBackend("qwen3:1.7b")
    sld = sample_seeds(1, families=[SLD], seed=0)[0]
    sld_prompt = backend._build_prompt(sld)
    assert "exactly one user message" in sld_prompt  # the fix for the 0% SLD yield
    assert "JSON array" in sld_prompt

    mtc = sample_seeds(1, families=[MTC], seed=0)[0]
    mtc_prompt = backend._build_prompt(mtc)
    assert f"exactly {max(2, mtc.num_user_turns)} user messages" in mtc_prompt


def test_materialize_writes_valid_synthetic_bases(
    simple_adapter: SimpleWhitespaceTokenizerAdapter, tmp_path: Path
) -> None:
    out = tmp_path / "synthetic_128.jsonl"
    bases = materialize_synthetic_corpus(
        backend=MockBackend(),
        adapter=simple_adapter,
        target_length=128,
        positions=[PREFIX],
        count=6,
        output_path=out,
        seed=0,
    )
    assert len(bases) == 6
    assert out.exists()
    ids = {b.base_id for b in bases}
    assert len(ids) == 6  # unique base ids
    for b in bases:
        assert b.base_id.startswith("synthetic_128_")
        assert b.metadata["data_source"] == "synthetic"
        assert any(s.slot == "{{PREFIX_SLOT}}" for s in b.slot_locations)


# ---------------------------------------------------------------------------
# Persona is a balanced, first-class sampling factor and provenance covariate.
# ---------------------------------------------------------------------------


def test_seeds_carry_balanced_personas() -> None:
    personas = list(DEFAULT_PERSONAS)
    count = len(personas) * 4  # a multiple of the (1 x 1 x 1 x |personas|) product
    seeds = sample_seeds(
        count,
        families=[MTC],
        domains=["software_debugging"],
        seed=0,
        difficulties=["easy"],
        personas=personas,
    )
    # Every persona level appears equally often; each seed carries a persona and locale.
    assert set(Counter(s.persona for s in seeds).values()) == {count // len(personas)}
    assert all(s.persona in personas for s in seeds)
    assert all(s.locale == "en" for s in seeds)


def test_persona_does_not_perturb_family_domain_difficulty_assignment() -> None:
    # Persona is the slowest mixed-radix factor, so the other factors are byte-for-byte unchanged.
    seeds = sample_seeds(30, seed=0)
    assert seeds[0].seed_id == "synthetic_mtc_software_debugging_0000"  # unchanged from Task 09


# ---------------------------------------------------------------------------
# Quality gate: degenerate (over-produced / repetitive) output is rejected.
# ---------------------------------------------------------------------------


def test_validate_rejects_gross_turn_over_production() -> None:
    seed = _seed(MTC)  # expected ~5 messages for an easy chat; the limit is well under a dozen
    messages = [ChatMessage(role=Role.SYSTEM, content="You are helpful.")]
    for i in range(12):  # 12 extra turns -> grossly over the over-production bound
        role = Role.USER if i % 2 == 0 else Role.ASSISTANT
        messages.append(ChatMessage(role=role, content=f"Distinct turn number {i} about the plan."))
    with pytest.raises(GenerationValidationError, match="over-producing"):
        validate_generated(messages, seed)


def test_validate_rejects_repetitive_conversation() -> None:
    seed = _seed(
        MTC
    )  # 6 messages stays within the over-production bound, so repetition is the fault
    question = "How should I approach the task here?"
    reply = "Let us work through it carefully together."
    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are helpful."),
        ChatMessage(role=Role.USER, content=question),
        ChatMessage(role=Role.ASSISTANT, content=reply),
        ChatMessage(role=Role.USER, content=question),
        ChatMessage(role=Role.ASSISTANT, content=reply),
        ChatMessage(role=Role.USER, content=question),
    ]
    with pytest.raises(GenerationValidationError, match="repeating"):
        validate_generated(messages, seed)


@pytest.mark.parametrize("family", [MTC, SLD])
def test_mock_backend_passes_quality_gate_across_difficulties(family: ConversationFamily) -> None:
    # 15 seeds span easy/medium/hard; mock output is never degenerate.
    for seed in sample_seeds(15, families=[family], seed=0):
        validate_generated(MockBackend().generate(seed), seed)  # must not raise


# ---------------------------------------------------------------------------
# Rich provenance recorded per base.
# ---------------------------------------------------------------------------


def test_base_records_rich_provenance(
    simple_adapter: SimpleWhitespaceTokenizerAdapter,
) -> None:
    seed = sample_seeds(1, families=[MTC], seed=0)[0]
    base = generate_base_conversation(
        seed,
        backend=MockBackend(),
        adapter=simple_adapter,
        target_length=128,
        positions=[PREFIX],
    )
    md = base.metadata
    assert md["generation_model"] == "mock"
    assert md["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert md["generation_params"] == {"deterministic": True}
    assert md["language"] == seed.locale == "en"
    assert md["persona"] == seed.persona


def test_ollama_backend_reports_decoding_provenance() -> None:
    backend = OllamaBackend("qwen3:1.7b")
    prov = backend.provenance()
    assert prov["model"] == "qwen3:1.7b"
    assert prov["think"] is False
    assert prov["options"]["repeat_penalty"] == 1.3


# ---------------------------------------------------------------------------
# AgentAuthoredBackend: the strong-generator "Haiku via agents" path.
# ---------------------------------------------------------------------------


def _authored_mtc() -> list[ChatMessage]:
    """A valid, harmless multi-turn chat as a Claude subagent would author it."""
    return [
        ChatMessage(role=Role.SYSTEM, content="You are a meticulous engineering assistant."),
        ChatMessage(
            role=Role.USER,
            content="My deploy script fails intermittently on the cache warm-up step.",
        ),
        ChatMessage(
            role=Role.ASSISTANT,
            content="Let us isolate whether the warm-up races the readiness probe.",
        ),
        ChatMessage(
            role=Role.USER,
            content="It only fails when the cache is cold. What should I check first?",
        ),
        ChatMessage(
            role=Role.ASSISTANT,
            content="Start by gating traffic on an explicit readiness signal.",
        ),
    ]


def test_agent_authored_backend_serves_and_records_provenance(
    simple_adapter: SimpleWhitespaceTokenizerAdapter,
) -> None:
    seed = sample_seeds(1, families=[MTC], seed=0)[0]
    backend = AgentAuthoredBackend({seed.seed_id: _authored_mtc()}, label="claude-opus-4-8")
    assert backend.name == "agent:claude-opus-4-8"  # honest, namespaced provenance
    base = generate_base_conversation(
        seed,
        backend=backend,
        adapter=simple_adapter,
        target_length=128,
        positions=[PREFIX],
    )
    assert base.metadata["generation_model"] == "agent:claude-opus-4-8"
    assert base.metadata["generation_params"]["authored_by"] == "claude-opus-4-8"


def test_agent_authored_backend_falls_back_to_mock_for_unauthored_seed(
    simple_adapter: SimpleWhitespaceTokenizerAdapter,
) -> None:
    seed = sample_seeds(1, families=[MTC], seed=0)[0]
    backend = AgentAuthoredBackend({}, label="claude")  # nothing authored for this seed
    base = generate_base_conversation(
        seed,
        backend=backend,
        adapter=simple_adapter,
        target_length=128,
        positions=[PREFIX],
    )
    # An unauthored seed becomes an honest mock fallback, never attributed to the agent.
    assert base.metadata["generation_model"] == "mock"


def test_load_agent_authored_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "authored.jsonl"
    rows = [
        {
            "seed_id": "synthetic_mtc_software_debugging_0000",
            "messages": [
                {"role": "user", "content": "Hello there."},
                {"role": "assistant", "content": "Hi, how can I help?"},
            ],
        }
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    authored = load_agent_authored(path)
    assert set(authored) == {"synthetic_mtc_software_debugging_0000"}
    msgs = authored["synthetic_mtc_software_debugging_0000"]
    assert [m.role for m in msgs] == [Role.USER, Role.ASSISTANT]
    assert msgs[0].content == "Hello there."


# ---------------------------------------------------------------------------
# Generation report sidecar (run-level provenance).
# ---------------------------------------------------------------------------


def test_materialize_writes_generation_report(
    simple_adapter: SimpleWhitespaceTokenizerAdapter, tmp_path: Path
) -> None:
    out = tmp_path / "synthetic_128.jsonl"
    materialize_synthetic_corpus(
        backend=MockBackend(),
        adapter=simple_adapter,
        target_length=128,
        positions=[PREFIX],
        count=6,
        output_path=out,
        seed=0,
    )
    report_path = tmp_path / "synthetic_128.report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["backend"] == "mock"
    assert report["prompt_template_version"] == PROMPT_TEMPLATE_VERSION
    assert report["count"] == 6
    assert report["fallbacks"] == 0
    assert report["exact_duplicate_bases"] == 0
    assert report["generation_params"] == {"deterministic": True}
    assert "generated_at" in report


def test_default_families_exclude_agent_tool() -> None:
    """A synthetic pull with no explicit families must NOT include AGENT_TOOL.

    Regression guard: AGENT_TOOL's `tool`-role shape is rejected by strict-alternation templates
    (Gemma), so it is opt-in. `sample_seeds` and the two corpus resolvers must default to the
    multi-turn + long-doc set only (DEFAULT_SAMPLE_FAMILIES), never IMPLEMENTED_FAMILIES.
    """
    from trigger_audit.generation.conversation_generator import (
        DEFAULT_SAMPLE_FAMILIES,
        ConversationFamily,
        sample_seeds,
    )

    assert ConversationFamily.AGENT_TOOL not in DEFAULT_SAMPLE_FAMILIES
    fams = {s.family for s in sample_seeds(30)}  # no families= -> default
    assert ConversationFamily.AGENT_TOOL not in fams
    assert fams <= set(DEFAULT_SAMPLE_FAMILIES)
