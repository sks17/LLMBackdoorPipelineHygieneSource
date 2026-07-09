"""Synthetic base-conversation generator: deterministic seeds -> harmless slot-form bases.

This is the synthetic arm's data source and the peer of :mod:`trigger_audit.io.dataset_adapter`.
A :class:`GenerationBackend` turns a deterministic :class:`ConversationSeed` into raw role-tagged
:class:`ChatMessage` content (an ordinary, harmless helpful-assistant conversation -- **no slots and
no trigger text**); everything after that -- measuring length with the target tokenizer, growing or
cutting to the grid target, and planting the named ``{{...}}`` insertion slots -- is delegated to
the *shared* :func:`trigger_audit.io.dataset_adapter.to_base_conversation`, exactly as the real
dataset arm does. That delegation is the whole point: it guarantees the property H4 depends on --
synthetic and real base conversations differ only in *content*, never in how triggers are inserted
or scored.

The generation model is never asked to produce triggers, canaries, or unsafe behavior; it produces
plain conversations, and :func:`validate_generated` rejects any output that carries a slot-like
(``{{`` / ``}}``) or canary-like (denylisted word or an all-caps/digit run) token, or that is
degenerate (grossly over-produced or highly repetitive), before it is ever handed to the slot
planter. Only the ``MULTI_TURN_CHAT``, ``SINGLE_TURN_LONG_DOCUMENT``, and ``AGENT_TOOL`` families
and the :class:`MockBackend` / :class:`OllamaBackend` / :class:`AgentAuthoredBackend` are
implemented in this build; the ``RAG_LIKE`` family and the :class:`TransformersBackend` /
:class:`ApiBackend` are documented stubs (like the dataset adapter's ``_BlockedParser``), so the
enum and dispatch accommodate them without a later refactor. Heavy/network imports (``requests``)
stay lazy and local so importing this module -- and running the offline test suite -- pulls in
nothing external.

Each emitted base carries rich provenance -- the producing backend name, the prompt-template
version, the decoding parameters, the language, and the persona -- because the synthetic-data
literature treats provenance as a first-class field (a base must be able to answer *who* generated
it, from *what* seed, under *which* prompt contract, with *what* parameters). Persona is a balanced,
first-class sampling factor for diversity rather than incidental flavor. See
``docs/DEEP_RESEARCH_FEEDBACK.md`` for the recommendations these choices implement.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import zlib
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from trigger_audit.io.dataset_adapter import (
    LengthMeasurer,
    make_length_measurer,
    to_base_conversation,
)
from trigger_audit.io.jsonl import read_jsonl, write_jsonl
from trigger_audit.prompts.chat_template import ChatFormat
from trigger_audit.schemas.messages import BaseConversation, ChatMessage, Role
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.tokenization.tokenizer_adapter import TokenizerAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Families, domains, and the deterministic seed.
# ---------------------------------------------------------------------------


class ConversationFamily(str, Enum):
    """Structural family of a synthetic base conversation.

    The value doubles as the emitted ``BaseConversation.conversation_type``, so it matches the
    labels :func:`dataset_adapter._infer_conversation_type` produces for real bases
    (``multi_turn_chat`` / ``single_turn_long_document``). ``MULTI_TURN_CHAT``,
    ``SINGLE_TURN_LONG_DOCUMENT``, and ``AGENT_TOOL`` are wired end to end here (``AGENT_TOOL``
    lands the ``tool_output`` slot-targeting extension); ``RAG_LIKE`` is declared so the enum and
    dispatch already know about it, but it is coupled to the ``retrieved_doc`` slot-targeting
    extension and raises ``NotImplementedError`` until that lands (see ``docs/tasks/09`` "Out of
    scope").
    """

    MULTI_TURN_CHAT = "multi_turn_chat"
    SINGLE_TURN_LONG_DOCUMENT = "single_turn_long_document"
    AGENT_TOOL = "agent_tool"
    RAG_LIKE = "rag_like"


# The families this build actually generates and validates end to end.
IMPLEMENTED_FAMILIES: tuple[ConversationFamily, ...] = (
    ConversationFamily.MULTI_TURN_CHAT,
    ConversationFamily.SINGLE_TURN_LONG_DOCUMENT,
    ConversationFamily.AGENT_TOOL,
)

# Families sampled by default when a caller does not name a set. AGENT_TOOL (E4) is IMPLEMENTED and
# opt-in via ``families=[...]``, but is deliberately NOT in the default so the validated prod grid
# (the config-driven assemble passes no ``--families``) keeps its multi-turn + long-doc composition
# and is not silently enlarged with agent/tool bases.
DEFAULT_SAMPLE_FAMILIES: tuple[ConversationFamily, ...] = (
    ConversationFamily.MULTI_TURN_CHAT,
    ConversationFamily.SINGLE_TURN_LONG_DOCUMENT,
)

# Short, stable family tag used in the human-readable seed id (e.g. ``synthetic_mtc_...``).
_FAMILY_ABBREV: dict[ConversationFamily, str] = {
    ConversationFamily.MULTI_TURN_CHAT: "mtc",
    ConversationFamily.SINGLE_TURN_LONG_DOCUMENT: "sld",
    ConversationFamily.AGENT_TOOL: "agt",
    ConversationFamily.RAG_LIKE: "rag",
}

# Default rotating harmless domains -- the same set the dataset arm's mock records use.
DEFAULT_DOMAINS: tuple[str, ...] = (
    "software_debugging",
    "data_analysis",
    "technical_writing",
    "project_planning",
    "customer_support",
)

# Difficulty rotation; also drives the multi-turn depth so harder seeds run longer conversations.
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")
_TURNS_BY_DIFFICULTY: dict[str, int] = {"easy": 2, "medium": 3, "hard": 4}

# Default persona bank. Persona-driven prompting is a well-supported lever for conversational
# diversity -- a persona collection surfaces perspectives a model's own defaults miss --
# so persona is a *balanced, first-class* sampling factor (a controlled variable) and a recorded
# provenance covariate, not incidental flavor. Personas are harmless and generic.
DEFAULT_PERSONAS: tuple[str, ...] = (
    "a junior engineer who is new to the codebase",
    "a busy product manager who wants the summary first",
    "a meticulous senior developer who asks precise follow-up questions",
    "a support specialist relaying a customer's problem in their own words",
    "a data analyst who is comfortable with technical detail",
)

# Single default language for this build. Kept as an explicit field (not hard-coded) so a later
# multilingual arm only has to populate it -- the diversity literature treats locale/language as a
# sampling quota, and provenance should carry it even when it is currently constant.
DEFAULT_LOCALE = "en"

# Bump when the generation prompts or structural instructions change, so provenance pins the
# exact prompt contract that produced it (the synthetic-data literature calls for recording the
# prompt-template version alongside the model and decoding parameters). ``"1"`` was the initial
# Task-09 build; ``"2"`` adds persona grounding to the prompts.
PROMPT_TEMPLATE_VERSION = "2"


class ConversationSeed(BaseModel):
    """A deterministic, content-free description of one base conversation to generate.

    Every field is derived from ``seed + index`` in :func:`sample_seeds` (no global RNG), so a fixed
    sampling seed always yields the same seeds. The seed carries *what* to generate (family, domain,
    task, depth) but never any conversation text -- that is the backend's job.
    """

    seed_id: str
    family: ConversationFamily
    domain: str
    num_user_turns: int = Field(ge=1)
    expected_user_task: str
    difficulty: str
    persona: str = DEFAULT_PERSONAS[0]
    locale: str = DEFAULT_LOCALE
    index: int

    def seed_id_for(self, target_length: int, namespace: str = "") -> str:
        """Return this seed's base id at a grid length: ``synthetic[_<namespace>]_<length>_NNN``.

        Parallels the dataset arm's ``<source>[_<namespace>]_<length>_NNN`` (``NNN`` is the seed's
        ordinal within the deterministic sample), so synthetic and real base ids share one naming
        scheme. ``namespace`` (typically the short model id) makes per-model base sets -- each
        length-matched to a different tokenizer -- coexist collision-free in one combined store.
        """
        ns = f"_{namespace}" if namespace else ""
        return f"synthetic{ns}_{target_length}_{self.index:03d}"


def _pretty(domain: str) -> str:
    """Render a snake_case domain name as a human-readable phrase (underscores to spaces)."""
    return domain.replace("_", " ")


# Per-domain task phrase; unknown domains fall back to a generic phrase built from the name.
_DOMAIN_TASKS: dict[str, str] = {
    "software_debugging": "diagnosing a slow autosave in a note-taking app",
    "data_analysis": "summarizing a large quarterly metrics report",
    "technical_writing": "drafting a concise design document for a cache",
    "project_planning": "sequencing tasks for a two-week sprint",
    "customer_support": "investigating an intermittent export failure",
}


def _task_phrase(domain: str) -> str:
    """Return a short harmless task phrase for a domain."""
    return _DOMAIN_TASKS.get(domain, f"a {_pretty(domain)} task")


def _expected_task(family: ConversationFamily, domain: str) -> str:
    """Compose a short natural-language task description for a seed."""
    task = _task_phrase(domain)
    if family == ConversationFamily.SINGLE_TURN_LONG_DOCUMENT:
        return f"read a long document about {_pretty(domain)} and answer a question about {task}"
    if family == ConversationFamily.AGENT_TOOL:
        return f"call a tool to look up information about {task} and answer the user"
    return f"help the user with {task} over a short chat"


def _num_user_turns(family: ConversationFamily, difficulty: str) -> int:
    """Long-document/agent-tool families use a single user turn; chats deepen with difficulty."""
    if family in (
        ConversationFamily.SINGLE_TURN_LONG_DOCUMENT,
        ConversationFamily.AGENT_TOOL,
    ):
        return 1
    return _TURNS_BY_DIFFICULTY.get(difficulty, 2)


def sample_seeds(
    count: int,
    *,
    families: Sequence[ConversationFamily] | None = None,
    domains: Sequence[str] | None = None,
    seed: int = 0,
    difficulties: Sequence[str] = DIFFICULTIES,
    personas: Sequence[str] = DEFAULT_PERSONAS,
    locale: str = DEFAULT_LOCALE,
) -> list[ConversationSeed]:
    """Return ``count`` deterministic seeds balanced across family/domain/difficulty/persona.

    Every choice is a pure function of ``seed + index`` via a mixed-radix rotation (family varies
    fastest, then domain, then difficulty, then persona -- persona is added as the slowest factor so
    it never perturbs the family/domain/difficulty assignment), like the dataset arm's
    ``synthetic_chat_records`` -- so a fixed ``seed`` reproduces the exact seeds, and when ``count``
    is a multiple of ``len(families) * len(domains) * len(difficulties) * len(personas)`` every
    combination appears equally often.
    """
    fams = tuple(families) if families is not None else DEFAULT_SAMPLE_FAMILIES
    doms = tuple(domains) if domains is not None else DEFAULT_DOMAINS
    diffs = tuple(difficulties)
    pers = tuple(personas)
    if not fams or not doms or not diffs or not pers:
        raise ValueError("sample_seeds needs at least one family, domain, difficulty, and persona")
    f, d, k, p = len(fams), len(doms), len(diffs), len(pers)
    seeds: list[ConversationSeed] = []
    for i in range(count):
        pick = seed + i
        family = ConversationFamily(fams[pick % f])
        domain = doms[(pick // f) % d]
        difficulty = diffs[(pick // (f * d)) % k]
        persona = pers[(pick // (f * d * k)) % p]
        seeds.append(
            ConversationSeed(
                seed_id=f"synthetic_{_FAMILY_ABBREV[family]}_{domain}_{i:04d}",
                family=family,
                domain=domain,
                num_user_turns=_num_user_turns(family, difficulty),
                expected_user_task=_expected_task(family, domain),
                difficulty=difficulty,
                persona=persona,
                locale=locale,
                index=i,
            )
        )
    return seeds


# ---------------------------------------------------------------------------
# Errors.
# ---------------------------------------------------------------------------


class GenerationError(RuntimeError):
    """A backend could not produce parseable output within its retry budget."""


class GenerationValidationError(ValueError):
    """Generated content is unusable for slot planting (empty, malformed, or trigger-shaped)."""


# ---------------------------------------------------------------------------
# Validation (shared, backend-agnostic).
# ---------------------------------------------------------------------------

# Case-insensitive denylist of canary-like words the generation model must never emit.
_FORBIDDEN_WORDS: tuple[str, ...] = ("canary", "trigger", "backdoor")
# A run of six-plus upper-case letters/digits looks like a planted canary token; reject it.
_CANARY_SHAPED = re.compile(r"\b[A-Z0-9]{6,}\b")
_DEFAULT_SYSTEM = "You are a helpful assistant."
# The single user turn of a long-document base must carry real content, not a one-liner.
_MIN_DOCUMENT_WORDS = 20

# Quality gates that reject *degenerate* output (as opposed to malformed output) so it falls back to
# the deterministic mock backend rather than entering the corpus. Small local models can (a) grossly
# over-produce turns and (b) repeat the same exchange several times; both pass the structural
# contract but are degenerate data. Rejecting them here routes the seed to the mock fallback
# (recorded honestly). This implements the "layered quality filtering / no mode collapse /
# duplication checks" release stage the synthetic-data literature treats as mandatory
# (see docs/DEEP_RESEARCH_FEEDBACK.md).
#
# A base may run up to this multiple of its expected message count before it is judged an
# over-production (generous, so a model that adds one extra summarizing turn is not rejected).
_MAX_MESSAGE_OVERPRODUCTION_FACTOR = 1.5
# A base whose distinct non-system messages fall below this fraction of its non-system messages is
# judged a repetition/mode collapse.
_MIN_DISTINCT_MESSAGE_RATIO = 0.5
# The distinct-ratio check only applies once there are at least this many non-system messages, so a
# legitimately short exchange is never flagged on too little evidence.
_REPETITION_MIN_MESSAGES = 4

# Roles a family's content may use. The chat/long-document families are plain
# system/user/assistant; ``AGENT_TOOL`` additionally allows the ``tool`` role (the result message
# the ``{{TOOL_OUTPUT_SLOT}}`` is planted into); the ``document`` role stays reserved for the
# (stubbed) RAG_LIKE family.
_ALLOWED_ROLES: dict[ConversationFamily, frozenset[Role]] = {
    ConversationFamily.MULTI_TURN_CHAT: frozenset({Role.SYSTEM, Role.USER, Role.ASSISTANT}),
    ConversationFamily.SINGLE_TURN_LONG_DOCUMENT: frozenset(
        {Role.SYSTEM, Role.USER, Role.ASSISTANT}
    ),
    ConversationFamily.AGENT_TOOL: frozenset({Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.TOOL}),
}


def _require_implemented_family(seed: ConversationSeed) -> None:
    """Raise a clear ``NotImplementedError`` for a family this build does not generate."""
    if seed.family not in IMPLEMENTED_FAMILIES:
        raise NotImplementedError(
            f"The {seed.family.value!r} family is a documented stub in this build: it is coupled "
            "to the tool_output/retrieved_doc slot-targeting extension and lands in a later task "
            "(see docs/tasks/09 'Out of scope'). Only "
            f"{[fam.value for fam in IMPLEMENTED_FAMILIES]} are implemented."
        )


def validate_generated(messages: list[ChatMessage], seed: ConversationSeed) -> None:
    """Validate raw generated messages, raising :class:`GenerationValidationError` when unusable.

    Enforces, at minimum: a non-empty message list; only the family's allowed roles; no empty or
    whitespace-only content; no slot-like ``{{`` / ``}}`` substring (which would collide with slot
    planting); no canary-like word or all-caps/digit run; and the family's structural contract
    (``MULTI_TURN_CHAT``: >=2 user and >=1 assistant turns; ``SINGLE_TURN_LONG_DOCUMENT``: exactly
    one substantial user turn). A missing ``system`` turn is *synthesized in place* (mirroring
    :class:`MockChatParser`) rather than treated as a failure.
    """
    _require_implemented_family(seed)
    if not messages:
        raise GenerationValidationError(f"{seed.seed_id}: no messages were generated")

    # Synthesize a default system turn when the model omitted one (do not fail for this alone).
    if not any(m.role == Role.SYSTEM for m in messages):
        messages.insert(0, ChatMessage(role=Role.SYSTEM, content=_DEFAULT_SYSTEM))

    allowed = _ALLOWED_ROLES[seed.family]
    for i, message in enumerate(messages):
        if message.role not in allowed:
            raise GenerationValidationError(
                f"{seed.seed_id}: message {i} has role {message.role.value!r}, which is not "
                f"allowed for the {seed.family.value!r} family"
            )
        content = message.content
        if not content.strip():
            raise GenerationValidationError(
                f"{seed.seed_id}: message {i} is empty or whitespace-only"
            )
        if "{{" in content or "}}" in content:
            raise GenerationValidationError(
                f"{seed.seed_id}: message {i} contains a slot-like brace token, which would "
                "collide with slot planting"
            )
        lowered = content.lower()
        for word in _FORBIDDEN_WORDS:
            if word in lowered:
                raise GenerationValidationError(
                    f"{seed.seed_id}: message {i} contains a canary-like word {word!r}"
                )
        shaped = _CANARY_SHAPED.search(content)
        if shaped is not None:
            raise GenerationValidationError(
                f"{seed.seed_id}: message {i} contains a canary-shaped token {shaped.group()!r}"
            )

    _validate_family_contract(messages, seed)
    _validate_quality(messages, seed)


def _validate_family_contract(messages: Sequence[ChatMessage], seed: ConversationSeed) -> None:
    """Check the per-family structural contract."""
    user_turns = sum(1 for m in messages if m.role == Role.USER)
    assistant_turns = sum(1 for m in messages if m.role == Role.ASSISTANT)
    if seed.family == ConversationFamily.MULTI_TURN_CHAT:
        if user_turns < 2 or assistant_turns < 1:
            raise GenerationValidationError(
                f"{seed.seed_id}: MULTI_TURN_CHAT needs >=2 user and >=1 assistant turns; got "
                f"{user_turns} user / {assistant_turns} assistant"
            )
    elif seed.family == ConversationFamily.SINGLE_TURN_LONG_DOCUMENT:
        if user_turns != 1:
            raise GenerationValidationError(
                f"{seed.seed_id}: SINGLE_TURN_LONG_DOCUMENT needs exactly 1 user turn; got "
                f"{user_turns}"
            )
        document = next(m for m in messages if m.role == Role.USER)
        if len(document.content.split()) < _MIN_DOCUMENT_WORDS:
            raise GenerationValidationError(
                f"{seed.seed_id}: SINGLE_TURN_LONG_DOCUMENT user turn is not substantial "
                f"(needs >= {_MIN_DOCUMENT_WORDS} words)"
            )
    elif seed.family == ConversationFamily.AGENT_TOOL:
        tool_turns = sum(1 for m in messages if m.role == Role.TOOL)
        if user_turns < 1 or assistant_turns < 1 or tool_turns < 1:
            raise GenerationValidationError(
                f"{seed.seed_id}: AGENT_TOOL needs >=1 user, >=1 assistant, and >=1 tool turns "
                f"(the tool result carries the tool_output slot); got {user_turns} user / "
                f"{assistant_turns} assistant / {tool_turns} tool"
            )


def _expected_message_count(seed: ConversationSeed) -> int:
    """Return the nominal message count a well-formed base for this seed carries.

    A ``MULTI_TURN_CHAT`` base is one system turn plus a user/assistant pair per user turn; a
    long-document base is a system turn plus one user turn plus one assistant answer; an
    ``AGENT_TOOL`` base is system + user + assistant(tool-call) + tool(result) + assistant(answer).
    Used only to bound gross over-production -- not to require an exact count.
    """
    if seed.family == ConversationFamily.SINGLE_TURN_LONG_DOCUMENT:
        return 3
    if seed.family == ConversationFamily.AGENT_TOOL:
        return 5
    return 1 + 2 * max(2, seed.num_user_turns)


def _validate_quality(messages: Sequence[ChatMessage], seed: ConversationSeed) -> None:
    """Reject degenerate output: gross turn over-production or whole-conversation repetition.

    Both failure modes pass the structural contract yet are unusable data; raising
    :class:`GenerationValidationError` here routes the seed to the mock fallback in
    :func:`_generate_validated`. Applied after the structural checks, so a genuinely malformed base
    fails on the more specific error first.
    """
    expected = _expected_message_count(seed)
    max_messages = int(_MAX_MESSAGE_OVERPRODUCTION_FACTOR * expected) + 1
    if len(messages) > max_messages:
        raise GenerationValidationError(
            f"{seed.seed_id}: {len(messages)} messages far exceeds the expected ~{expected} for "
            f"this seed (limit {max_messages}); likely a model over-producing turns"
        )
    body = [m.content.strip() for m in messages if m.role != Role.SYSTEM]
    if len(body) >= _REPETITION_MIN_MESSAGES:
        distinct = len(set(body))
        ratio = distinct / len(body)
        if ratio < _MIN_DISTINCT_MESSAGE_RATIO:
            raise GenerationValidationError(
                f"{seed.seed_id}: only {distinct}/{len(body)} non-system messages are distinct "
                f"(ratio {ratio:.2f} < {_MIN_DISTINCT_MESSAGE_RATIO}); a model repeating the "
                "same exchange"
            )


# ---------------------------------------------------------------------------
# Backends.
# ---------------------------------------------------------------------------


class GenerationBackend(ABC):
    """Produces raw role-tagged messages for a seed -- no slots, no triggers.

    ``name`` is recorded as the ``generation_model`` covariate on every emitted base, so the
    producing backend is always auditable (a mock fallback is never attributed to a real model).
    """

    name: str

    @abstractmethod
    def generate(self, seed: ConversationSeed) -> list[ChatMessage]:
        """Return the raw conversation content for ``seed`` (ordinary, harmless, slot-free)."""

    def provenance(self) -> dict[str, Any]:
        """Return this backend's decoding/config provenance (the per-base ``generation_params``).

        The default is empty; concrete backends report the knobs that would change their output
        (model tag, decoding options, authoring label) so a base can be reproduced or attributed.
        """
        return {}


# --- MockBackend: deterministic, offline content. --------------------------

# Harmless, forbidden-token-free content banks. Rotated by seed index so consecutive seeds of the
# same domain/family still differ, while staying fully deterministic.
_USER_OPENERS: tuple[str, ...] = (
    "I could use some help with {task}.",
    "I'm working on {task} and would like a second opinion.",
    "Can you walk me through {task}?",
)
_USER_FOLLOWUPS: tuple[str, ...] = (
    "Here is some more detail from the notes I gathered this morning.",
    "For extra context, this has come up a few times over the past week.",
    "One more thing: the behavior is consistent across several examples.",
    "I also put together a short summary of the relevant background.",
)
_USER_CLOSERS: tuple[str, ...] = (
    "Could you outline the most likely next steps?",
    "What would you check first, and why?",
    "Please give me a short, ordered plan I can follow.",
)
_ASSISTANT_REPLIES: tuple[str, ...] = (
    "Sure, let's work through it step by step.",
    "Good question. A few things stand out that are worth checking.",
    "Understood. That helps narrow down where to look next.",
    "Here is how I would approach it, starting with the simplest cause.",
)
_DOC_PARAGRAPHS: tuple[str, ...] = (
    "The document opens with an overview of the situation and the goals for the work ahead.",
    "It then describes the main components involved and how they interact under normal use.",
    "A section lists the observed symptoms, how often they occur, and how to repeat them.",
    "The proposed approach is broken into stages, each with a clear entry and exit condition.",
    "Finally, it records the open questions and the measurements needed to resolve them.",
)


def _system_for(domain: str, persona: str = "") -> str:
    """A domain-flavored system prompt, matching the tone of the dataset arm's mock records.

    When a persona is supplied it is woven in as who the assistant is helping, so the deterministic
    mock fallback is persona-varied too (persona is a diversity covariate, not just a prompt input).
    """
    base = f"You are a helpful {_pretty(domain)} assistant."
    return f"{base} You are helping {persona}." if persona else base


def _assistant_reply(index: int) -> str:
    """Pick a deterministic assistant reply."""
    return _ASSISTANT_REPLIES[index % len(_ASSISTANT_REPLIES)]


def _user_turn_texts(seed: ConversationSeed, turns: int) -> list[str]:
    """Build the ``turns`` user messages: opener, optional follow-ups, then a closing request."""
    task = _task_phrase(seed.domain)
    opener = _USER_OPENERS[seed.index % len(_USER_OPENERS)].format(task=task)
    closer = _USER_CLOSERS[seed.index % len(_USER_CLOSERS)]
    if turns <= 2:
        return [opener, closer]
    middles = [_USER_FOLLOWUPS[(seed.index + j) % len(_USER_FOLLOWUPS)] for j in range(turns - 2)]
    return [opener, *middles, closer]


def _build_multi_turn_chat(seed: ConversationSeed) -> list[ChatMessage]:
    """Build a system turn plus ``num_user_turns`` user/assistant exchanges."""
    turns = max(2, seed.num_user_turns)
    messages = [ChatMessage(role=Role.SYSTEM, content=_system_for(seed.domain, seed.persona))]
    for k, user_text in enumerate(_user_turn_texts(seed, turns)):
        messages.append(ChatMessage(role=Role.USER, content=user_text))
        messages.append(ChatMessage(role=Role.ASSISTANT, content=_assistant_reply(seed.index + k)))
    return messages


def _build_single_turn_long_document(seed: ConversationSeed) -> list[ChatMessage]:
    """Build a system turn plus one user turn carrying a multi-paragraph document and a question."""
    title = f"Reference document on {_pretty(seed.domain)}: {_task_phrase(seed.domain)}."
    count = len(_DOC_PARAGRAPHS)
    paragraphs = [_DOC_PARAGRAPHS[(seed.index + j) % count] for j in range(count)]
    question = _USER_CLOSERS[seed.index % len(_USER_CLOSERS)]
    body = "\n\n".join([title, *paragraphs, f"Based on the document above, {question.lower()}"])
    return [
        ChatMessage(role=Role.SYSTEM, content=_system_for(seed.domain, seed.persona)),
        ChatMessage(role=Role.USER, content=body),
    ]


# Harmless AGENT_TOOL content banks (an agent looks something up with a tool, then answers). Kept
# free of forbidden words, all-caps runs, and braces so the generated content passes validation; the
# result message is the one the ``{{TOOL_OUTPUT_SLOT}}`` is later planted into.
_TOOL_NAMES: tuple[str, ...] = (
    "lookup_records",
    "search_docs",
    "fetch_metrics",
    "query_tickets",
    "read_reference",
)
_AGENT_USER_REQUESTS: tuple[str, ...] = (
    "Can you look up the latest details on {task} before we decide anything?",
    "Please check the reference data for {task} and tell me what it says.",
    "I need the current numbers for {task}; could you pull them up for me?",
)
_AGENT_TOOL_CALLS: tuple[str, ...] = (
    "Let me call the {tool} tool to gather the relevant details first.",
    "I will query the {tool} tool so we are working from the real data.",
    "Give me a moment while I run the {tool} tool to fetch that.",
)
_AGENT_TOOL_RESULTS: tuple[str, ...] = (
    "The reference entry lists three related items, each with a short status note and an owner, "
    "and it records when the last update happened for every one of them.",
    "The lookup returned a summary describing the current situation, the two most recent changes, "
    "and a short list of the follow-up items that are still open.",
    "The tool reported a small set of results: a handful of rows, each pairing a label with a "
    "value and a brief comment about how it was measured.",
)
_AGENT_FINAL_ANSWERS: tuple[str, ...] = (
    "Based on what the tool returned, here is a short summary and a sensible next step for you.",
    "With those details in hand, the likely explanation and the first thing to check are clear.",
    "Given the results above, I would proceed with the plan I have outlined for you here.",
)


def _build_agent_tool(seed: ConversationSeed) -> list[ChatMessage]:
    """Build an agent/tool exchange: system, user, assistant(tool-call), tool(result), assistant.

    The single ``tool``-role message carries the result the ``{{TOOL_OUTPUT_SLOT}}`` is later
    planted into; a closing assistant answer follows so the tool result sits mid-conversation,
    where a head truncation can drop it while the final answer survives -- exactly the delivery
    risk the ``tool_output`` position audits. All content is harmless and free of slots/canaries.
    """
    task = _task_phrase(seed.domain)
    tool_name = _TOOL_NAMES[seed.index % len(_TOOL_NAMES)]
    user = _AGENT_USER_REQUESTS[seed.index % len(_AGENT_USER_REQUESTS)].format(task=task)
    call = _AGENT_TOOL_CALLS[seed.index % len(_AGENT_TOOL_CALLS)].format(tool=tool_name)
    result = _AGENT_TOOL_RESULTS[seed.index % len(_AGENT_TOOL_RESULTS)]
    answer = _AGENT_FINAL_ANSWERS[seed.index % len(_AGENT_FINAL_ANSWERS)]
    return [
        ChatMessage(role=Role.SYSTEM, content=_system_for(seed.domain, seed.persona)),
        ChatMessage(role=Role.USER, content=user),
        ChatMessage(role=Role.ASSISTANT, content=call, metadata={"tool_name": tool_name}),
        ChatMessage(role=Role.TOOL, content=result, name=tool_name),
        ChatMessage(role=Role.ASSISTANT, content=answer),
    ]


_MOCK_FAMILY_BUILDERS: dict[ConversationFamily, Callable[[ConversationSeed], list[ChatMessage]]] = {
    ConversationFamily.MULTI_TURN_CHAT: _build_multi_turn_chat,
    ConversationFamily.SINGLE_TURN_LONG_DOCUMENT: _build_single_turn_long_document,
    ConversationFamily.AGENT_TOOL: _build_agent_tool,
}


class MockBackend(GenerationBackend):
    """Fully deterministic, offline backend used by the whole test suite and as the fallback.

    Produces valid, harmless, seed-shaped conversations by rotating realistic content per domain and
    seed index. It requires no LLM and no network, so it always succeeds for an implemented family.
    """

    name = "mock"

    def generate(self, seed: ConversationSeed) -> list[ChatMessage]:
        builder = _MOCK_FAMILY_BUILDERS.get(seed.family)
        if builder is None:
            _require_implemented_family(seed)  # raises the documented stub error
        assert builder is not None  # narrowed: implemented families always have a builder
        return builder(seed)

    def provenance(self) -> dict[str, Any]:
        return {"deterministic": True}


# --- OllamaBackend: local Ollama server generation. ------------------------


def _strip_wrappers(text: str) -> str:
    """Remove ``<think>`` reasoning blocks and a ```json ...``` fence from a model completion.

    Local instruct models wrap the payload in ways that defeat a naive parse: thinking models
    (e.g. Qwen3) prepend ``<think>...</think>``, and many fence the JSON. Both are stripped so the
    tolerant object sweep below sees just the data. Thinking is also disabled at the API level
    (``think=False``); this strip is a belt-and-suspenders for a thinking-on run.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fence is not None:
        cleaned = fence.group(1).strip()
    return cleaned


def _iter_json_values(text: str) -> list[Any]:
    """Pull every top-level JSON value out of ``text`` in order, tolerating junk between them.

    Uses ``JSONDecoder.raw_decode`` to consume successive values, so it handles a proper array,
    several concatenated objects (``{...}\\n{...}``), or a single object -- exactly the shapes small
    local models emit -- and correctly ignores braces inside strings. Unparseable stretches are
    skipped a character at a time rather than aborting the whole parse.
    """
    decoder = json.JSONDecoder()
    values: list[Any] = []
    idx, n = 0, len(text)
    while idx < n:
        while idx < n and text[idx] in " \t\r\n,":
            idx += 1
        if idx >= n:
            break
        try:
            value, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1  # skip a stray character and keep looking for the next value
            continue
        values.append(value)
        idx = end
    return values


def _extract_message_dicts(text: str) -> list[dict[str, Any]]:
    """Extract the message objects from a model completion, tolerant of the common output shapes.

    Accepts a JSON array of objects, several concatenated objects, or a single object (which a
    caller may expand from a role-keyed dict). Raises ``ValueError`` when no object is found so the
    backend retries.
    """
    dicts: list[dict[str, Any]] = []
    for value in _iter_json_values(_strip_wrappers(text)):
        if isinstance(value, list):
            dicts.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            dicts.append(value)
    if not dicts:
        raise ValueError("no JSON object(s) found in model output")
    return dicts


def _role_or_none(key: str) -> Role | None:
    """Return the :class:`Role` a dict key names, or ``None`` when the key is not a role."""
    try:
        return Role(key.strip().lower())
    except ValueError:
        return None


# Default Ollama decoding options. ``repeat_penalty`` with a wide ``repeat_last_n`` discourages the
# whole-conversation repetition small local models fall into (a base whose one exchange is emitted
# several times passes structural validation but is degenerate data); ``repeat_last_n`` spans a full
# short conversation so the penalty actually sees the earlier block it would otherwise duplicate.
# ``seed`` is injected per attempt in ``_chat``. Override via ``OllamaBackend(options=...)``.
_DEFAULT_OPTIONS: dict[str, Any] = {
    "temperature": 0.7,
    "repeat_penalty": 1.3,
    "repeat_last_n": 256,
}


# The output-format half of the prompt: strict JSON and the harmless-content guardrails (no
# placeholders/braces/all-caps) so the model never emits a slot or a canary-shaped token that
# validation would reject. The allowed-role clause is family-aware because the AGENT_TOOL family
# additionally needs the ``tool`` role for its result message, while the chat/document families do
# not (allowing it there would only invite messages their validators reject).
def _json_format_instructions(seed: ConversationSeed) -> str:
    """Return the JSON output-format instruction, listing the roles ``seed``'s family may use."""
    roles = (
        "system, user, assistant, and tool"
        if seed.family == ConversationFamily.AGENT_TOOL
        else "system, user, and assistant"
    )
    return (
        'Return ONLY a JSON array of objects shaped like {"role": ..., "content": ...}. Allowed '
        f"roles are {roles}. Do not include any placeholders, bracketed or curly-brace tokens, "
        "template variables, or all-caps code words. Write natural prose only."
    )


class OllamaBackend(GenerationBackend):
    """Generates conversations via a locally running Ollama server (e.g. a pulled Qwen3 model).

    The prompt is strictly *content-only*: it asks for a realistic, harmless conversation for the
    seed and explicitly forbids placeholders, bracketed tokens, and all-caps code words, so the
    model is never asked to produce a trigger or canary. Output is parsed as a JSON array of
    ``{"role", "content"}`` objects; on unparseable or empty output the call retries up to
    ``max_attempts`` and then raises :class:`GenerationError`. ``requests`` is imported lazily
    inside :meth:`_chat`, so the offline suite never imports it.
    """

    def __init__(
        self,
        model: str,
        *,
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        max_attempts: int = 3,
        think: bool | None = False,
        options: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._max_attempts = max_attempts
        # Disable model "thinking" by default: our generation fleet is Qwen3 (a thinking model), and
        # <think> reasoning both slows generation and pollutes JSON parsing. Pass ``think=None`` to
        # omit the field for a non-thinking model that rejects it.
        self._think = think
        # Decoding options merged over the anti-repetition defaults (caller overrides win).
        self._options = {**_DEFAULT_OPTIONS, **(options or {})}
        self.name = f"ollama:{model}"

    def generate(self, seed: ConversationSeed) -> list[ChatMessage]:
        _require_implemented_family(seed)
        prompt = self._build_prompt(seed)
        base_seed = zlib.crc32(seed.seed_id.encode("utf-8")) & 0x7FFFFFFF
        last_error = "no attempt made"
        for attempt in range(self._max_attempts):
            try:
                raw = self._chat(prompt, seed=(base_seed + attempt) & 0x7FFFFFFF)
                messages = self._parse_messages(raw)
                if messages:
                    return messages
                last_error = "model returned an empty message list"
            except Exception as exc:  # any backend/parse failure is a retryable generation error
                last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Ollama backend %s attempt %d/%d for seed %s failed: %s",
                self.name,
                attempt + 1,
                self._max_attempts,
                seed.seed_id,
                last_error,
            )
        raise GenerationError(
            f"{self.name} failed to produce parseable output for seed {seed.seed_id} after "
            f"{self._max_attempts} attempts (last error: {last_error})"
        )

    def _build_prompt(self, seed: ConversationSeed) -> str:
        """Build the content-only prompt (never asks for triggers, canaries, or placeholders).

        The structural instruction is family-specific and explicit about the exact message shape the
        family contract requires, because local instruct models otherwise read
        "single_turn_long_document" as "have a conversation" (many user turns) rather than "one user
        message carrying a long document" -- which the validator then rejects, forcing a fallback.
        """
        return f"{self._structure_instructions(seed)}\n\n{_json_format_instructions(seed)}"

    def _structure_instructions(self, seed: ConversationSeed) -> str:
        """Return the family-specific structural instruction (exact message shape and counts).

        Grounds the conversation in the seed's persona (a diversity lever) while keeping the exact
        message-shape contract the family validator enforces.
        """
        domain, task = _pretty(seed.domain), seed.expected_user_task
        persona = f"The user is {seed.persona}. "
        if seed.family == ConversationFamily.AGENT_TOOL:
            return (
                f"{persona}Write a realistic, harmless agent/tool exchange in the domain of "
                f"{domain} about {task}. Produce exactly five messages in order: (1) one short "
                "system message; (2) one user message asking for information that requires a "
                "lookup; (3) one assistant message saying it will call a tool to fetch that "
                "information; (4) one tool message (role 'tool') containing the tool's returned "
                "result as plain prose; (5) one assistant message that answers the user using the "
                "tool result."
            )
        if seed.family == ConversationFamily.SINGLE_TURN_LONG_DOCUMENT:
            return (
                f"{persona}Write a realistic, harmless long-document reading task in the domain of "
                f"{domain} about {task}. Produce exactly three messages in order: (1) one short "
                "system message; (2) exactly one user message that contains a multi-paragraph "
                "reference document of at least four paragraphs followed by a question about "
                "it; (3) one assistant message that answers the question. There must be exactly "
                "one user message in total -- do not split the document across several user turns."
            )
        n = max(2, seed.num_user_turns)
        return (
            f"{persona}Write a realistic, harmless multi-turn chat in the domain of {domain} about "
            f"{task}. Produce exactly one opening system message, then exactly {n} user messages, "
            "each immediately followed by exactly one assistant reply, strictly alternating user "
            f"then assistant (a total of {1 + 2 * n} messages). Keep each message a short, natural "
            "turn."
        )

    def _chat(self, prompt: str, *, seed: int) -> str:
        """POST the prompt to Ollama ``/api/chat`` and return the assistant text (non-streaming)."""
        import requests  # lazy: only for a live run; never imported by the offline suite

        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {**self._options, "seed": seed},
        }
        if self._think is not None:
            body["think"] = self._think
        response = requests.post(f"{self._host}/api/chat", json=body, timeout=self._timeout)
        response.raise_for_status()
        payload = response.json()
        return str(payload["message"]["content"])

    def _parse_messages(self, raw: str) -> list[ChatMessage]:
        """Parse the model output into messages, tolerant of the shapes local models emit.

        Each extracted object is either a ``{"role", "content"}`` message or a role-keyed dict
        (``{"system": ..., "user": ...}``) that is expanded into one message per role key in order.
        An invalid role in a ``{"role", "content"}`` object raises, triggering a retry.
        """
        messages: list[ChatMessage] = []
        for item in _extract_message_dicts(raw):
            if "role" in item and "content" in item:
                role = Role(str(item["role"]).strip().lower())
                messages.append(ChatMessage(role=role, content=str(item["content"])))
                continue
            # A role-keyed dict: expand each valid-role key (in insertion order) into a message.
            for key, value in item.items():
                role_or_none = _role_or_none(key)
                if role_or_none is not None and isinstance(value, str) and value.strip():
                    messages.append(ChatMessage(role=role_or_none, content=value))
        return messages

    def provenance(self) -> dict[str, Any]:
        return {"model": self._model, "think": self._think, "options": dict(self._options)}


# --- AgentAuthoredBackend: strong-generator content authored out-of-band by an agent. ---------


class AgentAuthoredBackend(GenerationBackend):
    """Serves harmless conversations authored out-of-band by an agent (e.g. a Claude subagent).

    We do not have hosted-API access, so the strong-generator role the synthetic-data literature
    assigns to a frontier model (Claude Haiku in the write-up) is filled by *agent-authored*
    conversations rather than an API backend: an agent writes ordinary role-tagged conversations for
    sampled seeds to a JSONL file, and this backend serves them by seed id. Served content is *not*
    trusted -- it goes through the same :func:`validate_generated` guardrails (slot/canary/quality)
    as any other backend, so an unusable authored conversation still falls back to the mock backend.
    Seeds with no authored content raise :class:`GenerationError`, which the orchestrator turns into
    the same honest mock fallback. ``label`` names the authoring agent/model (e.g.
    ``claude-opus-4-8``) so provenance stays truthful.

    Authored content should respect the seed's requested shape -- in particular ``num_user_turns``
    for a chat -- because the quality gate bounds a base against its seed's *expected* size;
    a conversation much richer than its seed's difficulty implies is treated as over-production and
    falls back to mock (author a matching seed difficulty, or split it across seeds, instead).
    """

    def __init__(
        self, authored: Mapping[str, Sequence[ChatMessage]], *, label: str = "agent"
    ) -> None:
        self._authored: dict[str, list[ChatMessage]] = {
            sid: list(msgs) for sid, msgs in authored.items()
        }
        self._label = label
        # Keep provenance names namespaced under "agent:" without double-prefixing an "agent"-tagged
        # label, so ``generation_model`` reads e.g. ``agent:claude-opus-4-8``.
        self.name = label if label.startswith("agent") else f"agent:{label}"

    def generate(self, seed: ConversationSeed) -> list[ChatMessage]:
        _require_implemented_family(seed)
        authored = self._authored.get(seed.seed_id)
        if authored is None:
            raise GenerationError(
                f"{self.name} has no authored conversation for seed {seed.seed_id!r}; author one "
                "(or accept the mock fallback for this seed)"
            )
        # Hand back deep copies so downstream length-binning/slotting never mutates the loaded map.
        return [m.model_copy(deep=True) for m in authored]

    def provenance(self) -> dict[str, Any]:
        return {"authored_by": self._label, "authored_seed_count": len(self._authored)}


def load_agent_authored(path: str | Path) -> dict[str, list[ChatMessage]]:
    """Load an agent-authored conversation file into a ``seed_id -> messages`` map.

    Each JSONL row is ``{"seed_id": str, "messages": [{"role", "content"}, ...]}`` -- what an
    agent writes when it authors the strong-generator corpus. Content is *not* validated
    here; the generation orchestrator validates each served conversation (and falls back to mock)
    per seed, so a malformed authored row degrades gracefully rather than aborting the load.
    """
    authored: dict[str, list[ChatMessage]] = {}
    for row in read_jsonl(path):
        seed_id = str(row["seed_id"])
        authored[seed_id] = [
            ChatMessage(role=Role(str(m["role"]).strip().lower()), content=str(m["content"]))
            for m in row["messages"]
        ]
    return authored


# --- Documented stubs (mirroring the dataset adapter's ``_BlockedParser``). -


class _BlockedBackend(GenerationBackend):
    """Shared base for backends blocked on external access; :meth:`generate` names what is missing.

    Like ``dataset_adapter._BlockedParser``, this can never be mistaken for a working backend: it
    raises a ``NotImplementedError`` describing exactly what must be provided before it can be
    filled in (tracked in ``docs/REQUESTED_DOCUMENTATION.md``).
    """

    _requirement: ClassVar[str]

    def generate(self, seed: ConversationSeed) -> list[ChatMessage]:
        raise NotImplementedError(
            f"{type(self).__name__} is a documented stub: {self._requirement} Fill in generate() "
            "once that access lands (tracked in docs/REQUESTED_DOCUMENTATION.md)."
        )


class TransformersBackend(_BlockedBackend):
    """Cluster GPU generation via HuggingFace ``generate`` -- BLOCKED on model/GPU access (stub).

    The recommended fill is an open, checkpointed, openly-licensed suite -- **Pythia** (Apache-2.0,
    70M-12B, 154 checkpoints/model, dataloader reconstruction) -- because when the generator is part
    of the scientific claim, reproducibility and transparency outweigh raw dialogue quality (see
    ``docs/DEEP_RESEARCH_FEEDBACK.md``). It is a documented stub because ``torch`` is not installed
    here and generation needs a GPU allocation.
    """

    name = "transformers"
    _requirement = (
        "generating on the cluster via a HuggingFace causal LM (`AutoModelForCausalLM.generate`, "
        "recommended: a Pythia checkpoint) needs `torch`, the approved model revision, and a GPU "
        "allocation, none of which are wired here yet."
    )


class ApiBackend(_BlockedBackend):
    """Hosted API generation (e.g. Claude Haiku) -- use AgentAuthoredBackend instead (stub).

    We do not have hosted-API access, so the strong-generator role is filled by
    :class:`AgentAuthoredBackend` (Claude-authored conversations served from a file), not this
    backend. This stub is kept only for a future first-party API integration.
    """

    name = "api"
    _requirement = (
        "generating via a hosted API model needs a chosen provider, a model id, and an API "
        "key/quota, which are not provisioned -- the AgentAuthoredBackend is used instead."
    )


# Registry of backend name -> class. The stubs resolve to documented ``NotImplementedError``s;
# ``AgentAuthoredBackend`` needs an authored-content map, so the CLI constructs it explicitly.
BACKENDS: dict[str, type[GenerationBackend]] = {
    "mock": MockBackend,
    "ollama": OllamaBackend,
    "agent": AgentAuthoredBackend,
    "transformers": TransformersBackend,
    "api": ApiBackend,
}


# ---------------------------------------------------------------------------
# Generation orchestration (validation + bounded retry + mock fallback).
# ---------------------------------------------------------------------------

# Backend attempts before falling back to the mock backend for a seed.
_MAX_BACKEND_ATTEMPTS = 3


def _generate_validated(
    seed: ConversationSeed,
    *,
    backend: GenerationBackend,
    fallback: GenerationBackend | None = None,
) -> tuple[list[ChatMessage], GenerationBackend]:
    """Return ``(messages, producing_backend)`` for a seed, with retry and mock fallback.

    Calls ``backend.generate`` then :func:`validate_generated`, retrying a bounded number of times
    on either a :class:`GenerationError` or a :class:`GenerationValidationError`, and as a last
    resort generates with the mock backend -- returning the *actual* producing backend (its name and
    decoding provenance) so a fallback is never silently attributed to the real model.
    """
    errors: list[str] = []
    for attempt in range(_MAX_BACKEND_ATTEMPTS):
        try:
            messages = backend.generate(seed)
            validate_generated(messages, seed)
            return messages, backend
        except (GenerationError, GenerationValidationError) as exc:
            errors.append(str(exc))
            logger.warning(
                "Backend %s failed for seed %s (attempt %d/%d): %s",
                backend.name,
                seed.seed_id,
                attempt + 1,
                _MAX_BACKEND_ATTEMPTS,
                exc,
            )

    fb = fallback if fallback is not None else MockBackend()
    messages = fb.generate(seed)
    validate_generated(messages, seed)
    logger.warning(
        "Falling back to %s for seed %s after %d failed attempt(s) (last error: %s)",
        fb.name,
        seed.seed_id,
        _MAX_BACKEND_ATTEMPTS,
        errors[-1] if errors else "unknown",
    )
    return messages, fb


def generate_base_conversation(
    seed: ConversationSeed,
    *,
    backend: GenerationBackend,
    adapter: TokenizerAdapter,
    target_length: int,
    positions: Sequence[TriggerPosition],
    chat_format: ChatFormat = "chat",
    measure: LengthMeasurer | None = None,
    base_id_namespace: str = "",
) -> BaseConversation:
    """Generate one slot-form :class:`BaseConversation` for a seed via the shared emit path.

    The backend produces raw content (validated, with mock fallback); everything else -- length
    binning to ``target_length`` and planting the ``positions`` slots -- is delegated to the shared
    :func:`to_base_conversation`, so a synthetic base is inserted into and scored exactly like a
    real one. Rich provenance is merged into the metadata (without dropping the fields
    ``to_base_conversation`` already set): the producing backend's ``name``, the ``seed_id``, the
    ``prompt_template_version``, the backend's decoding ``generation_params``, the ``language``, and
    the ``persona`` -- so a base can answer who generated it, from what seed, under which
    prompt contract, and with what parameters (the provenance-as-first-class-field recommendation).
    """
    measure_fn = (
        measure if measure is not None else make_length_measurer(adapter, chat_format=chat_format)
    )
    messages, producing = _generate_validated(seed, backend=backend)
    base = to_base_conversation(
        messages,
        base_id=seed.seed_id_for(target_length, base_id_namespace),
        adapter=adapter,
        target_length=target_length,
        positions=positions,
        data_source="synthetic",
        source_record_id=seed.seed_id,
        conversation_type=seed.family.value,
        expected_user_task=seed.expected_user_task,
        domain=seed.domain,
        difficulty=seed.difficulty,
        measure=measure_fn,
    )
    base.metadata["generation_model"] = producing.name
    base.metadata["seed_id"] = seed.seed_id
    base.metadata["prompt_template_version"] = PROMPT_TEMPLATE_VERSION
    base.metadata["generation_params"] = producing.provenance()
    base.metadata["language"] = seed.locale
    base.metadata["persona"] = seed.persona
    return base


def _exact_duplicate_bases(bases: Sequence[BaseConversation]) -> int:
    """Count bases whose full (role, content) message signature exactly repeats another base's.

    A cheap corpus-level duplication check (the literature's release gate targets a low duplicate
    rate). Deterministic mock content is unique per seed, so a non-zero count flags a real backend
    returning identical output across seeds. Near-duplicate detection needs embeddings and is
    deferred to the analysis layer.
    """
    signatures = [tuple((m.role.value, m.content) for m in b.messages) for b in bases]
    return sum(c - 1 for c in Counter(signatures).values() if c > 1)


def _write_generation_report(
    output_path: str | Path,
    *,
    backend: GenerationBackend,
    bases: Sequence[BaseConversation],
    fallbacks: int,
    target_length: int,
    positions: Sequence[TriggerPosition],
    families: Sequence[ConversationFamily],
    domains: Sequence[str],
    seed: int,
) -> Path:
    """Write a ``<output>.report.json`` sidecar recording the full generation condition.

    The synthetic-data literature calls for publishing a generation report (model, prompt-template
    version, decoding parameters, filters, seed, and dedup rate) alongside the corpus, so a run is
    reproducible and auditable. Only run-level provenance is written here -- never raw prompts or
    conversation content.
    """
    duplicates = _exact_duplicate_bases(bases)
    report = {
        "backend": backend.name,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "generation_params": backend.provenance(),
        "count": len(bases),
        "fallbacks": fallbacks,
        "fallback_rate": round(fallbacks / len(bases), 4) if bases else 0.0,
        "exact_duplicate_bases": duplicates,
        "target_length": target_length,
        "positions": [p.value for p in positions],
        "families": [f.value for f in families],
        "domains": list(domains),
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = Path(output_path).with_name(Path(output_path).stem + ".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def materialize_synthetic_corpus(
    *,
    backend: GenerationBackend,
    adapter: TokenizerAdapter,
    target_length: int,
    positions: Sequence[TriggerPosition],
    count: int,
    families: Sequence[ConversationFamily] | None = None,
    domains: Sequence[str] | None = None,
    output_path: str | Path,
    seed: int = 0,
    chat_format: ChatFormat = "chat",
    write_report: bool = True,
    base_id_namespace: str = "",
) -> list[BaseConversation]:
    """Sample seeds, generate each base, and write the derived bases to a JSONL file.

    Base ids follow ``synthetic[_<namespace>]_<length>_NNN`` (parallel to the dataset arm's
    ``<source>[_<namespace>]_<length>_NNN``); ``base_id_namespace`` (the short model id) makes
    per-model, per-tokenizer base sets coexist in one combined store. Only the derived base
    conversations are written -- never any raw
    generation prompt -- so keep the output path under the git-ignored ``data/`` tree; fallbacks are
    logged; generation never stalls. When ``write_report`` is set (the default), a
    ``<output>.report.json`` provenance sidecar is written next to the corpus.
    """
    measure = make_length_measurer(adapter, chat_format=chat_format)
    # Default to the multi-turn + long-doc set, NOT every IMPLEMENTED family: AGENT_TOOL is opt-in
    # (its `tool`-role shape is rejected by strict-alternation templates such as Gemma's, and it is
    # generated by a separate explicit --families agent_tool call), so a plain synthetic pull must
    # not silently include it. Matches sample_seeds' default.
    resolved_families = tuple(families) if families is not None else DEFAULT_SAMPLE_FAMILIES
    resolved_domains = tuple(domains) if domains is not None else DEFAULT_DOMAINS
    seeds = sample_seeds(count, families=resolved_families, domains=resolved_domains, seed=seed)
    bases: list[BaseConversation] = []
    fallbacks = 0
    for one in seeds:
        base = generate_base_conversation(
            one,
            backend=backend,
            adapter=adapter,
            target_length=target_length,
            positions=positions,
            chat_format=chat_format,
            measure=measure,
            base_id_namespace=base_id_namespace,
        )
        if base.metadata.get("generation_model") != backend.name:
            fallbacks += 1
        bases.append(base)
    write_jsonl(output_path, bases)
    if write_report:
        report_path = _write_generation_report(
            output_path,
            backend=backend,
            bases=bases,
            fallbacks=fallbacks,
            target_length=target_length,
            positions=positions,
            families=resolved_families,
            domains=resolved_domains,
            seed=seed,
        )
        logger.info("Wrote generation report -> %s", report_path)
    logger.info(
        "Materialized %d synthetic base conversations (%d mock fallback(s)) -> %s",
        len(bases),
        fallbacks,
        output_path,
    )
    return bases


# ---------------------------------------------------------------------------
# CLI (mirrors dataset_adapter.main).
# ---------------------------------------------------------------------------


def _make_backend(
    name: str, model: str | None, *, authored_path: str | None = None
) -> GenerationBackend:
    """Construct the requested generation backend for the CLI."""
    if name == "mock":
        return MockBackend()
    if name == "ollama":
        if not model:
            raise ValueError(
                "--generation-model is required for the ollama backend (e.g. qwen3:1.7b)"
            )
        return OllamaBackend(model)
    if name == "agent":
        if not authored_path:
            raise ValueError(
                "--authored-path is required for the agent backend (a JSONL of {seed_id, messages} "
                "authored by an agent)"
            )
        return AgentAuthoredBackend(load_agent_authored(authored_path), label=model or "agent")
    raise ValueError(f"Unknown generation backend {name!r}; known: {sorted(BACKENDS)}")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the synthetic-generation driver."""
    parser = argparse.ArgumentParser(
        description="Materialize synthetic base conversations for the survivability grid."
    )
    parser.add_argument("--model-id", required=True, help="Target tokenizer/model id for binning")
    parser.add_argument("--tokenizer-backend", default="hf", choices=("hf", "simple"))
    parser.add_argument("--target-length", type=int, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        choices=[fam.value for fam in ConversationFamily],
        help="Families to sample (default: the implemented families)",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=None,
        help="Harmless domains to rotate (default: the built-in domain set)",
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=["prefix"],
        help="Trigger positions to plant slots for (e.g. prefix old_turn recent_turn)",
    )
    parser.add_argument("--generation-backend", default="mock", choices=("mock", "ollama", "agent"))
    parser.add_argument(
        "--generation-model",
        default=None,
        help="Ollama model tag (e.g. qwen3:1.7b), or the provenance label for the agent backend",
    )
    parser.add_argument(
        "--authored-path",
        default=None,
        help="Agent-authored conversations JSONL ({seed_id, messages}) for the agent backend",
    )
    parser.add_argument("--chat-format", default="chat", choices=("chat", "base"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--base-id-namespace",
        default="",
        help="Tag folded into every base id (typically the short model id) so per-model, "
        "per-tokenizer base sets coexist collision-free in one combined store",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path (keep under data/)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: sample seeds, generate synthetic bases, and write JSONL."""
    from trigger_audit.tokenization.tokenizer_adapter import make_tokenizer_adapter

    args = _build_arg_parser().parse_args(argv)
    adapter = make_tokenizer_adapter(args.model_id, backend=args.tokenizer_backend)
    positions = [TriggerPosition(p) for p in args.positions]
    # No --families -> the default multi-turn + long-doc set (NOT every implemented family):
    # AGENT_TOOL is opt-in via an explicit --families agent_tool, so a plain synthetic pull never
    # silently emits its `tool`-role shape (which strict-alternation templates like Gemma's reject).
    families = (
        [ConversationFamily(f) for f in args.families]
        if args.families
        else list(DEFAULT_SAMPLE_FAMILIES)
    )
    domains = list(args.domains) if args.domains else list(DEFAULT_DOMAINS)
    backend = _make_backend(
        args.generation_backend, args.generation_model, authored_path=args.authored_path
    )
    bases = materialize_synthetic_corpus(
        backend=backend,
        adapter=adapter,
        target_length=args.target_length,
        positions=positions,
        count=args.count,
        families=families,
        domains=domains,
        output_path=args.output,
        seed=args.seed,
        chat_format=args.chat_format,
        base_id_namespace=args.base_id_namespace,
    )
    print(f"Wrote {len(bases)} synthetic base conversations to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(main())
