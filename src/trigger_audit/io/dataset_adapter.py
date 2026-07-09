"""Ingest external corpora into the existing ``BaseConversation`` schema for the H4 arm.

This is the one narrow bridge between real conversation/document datasets and the *unchanged*
``expand_manifest`` -> ``run_trial`` pipeline. A source-specific :class:`DatasetParser` normalizes
one raw record into our roles; :func:`to_base_conversation` then length-bins the normalized
messages to a grid context length (measured with the *target model's* tokenizer) and plants the
same named insertion slots synthetic bases carry, so the slot-aware ``TriggerInserter`` fills real
and synthetic bases identically. The output is an ordinary ``BaseConversation`` -- no new schema,
no new runner path -- which is exactly what H4 validity requires: real and synthetic bases must
differ only in *content*, not in how triggers are inserted or scored.

The real LMSYS/WildChat/long-document parsers are blocked on each source's actual record format
(role keys, turn nesting, metadata) and usage terms; they are documented ``NotImplementedError``
stubs here rather than guessed JSON shapes. :class:`MockChatParser` is a fully working synthetic
parser used to validate the ingestion end to end. Heavy imports (``datasets``) stay lazy.
"""

from __future__ import annotations

import argparse
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

from trigger_audit.io.jsonl import write_jsonl
from trigger_audit.pipelines.trigger_insertion import (
    place_in_content,
    slot_for_position,
    target_user_index,
)
from trigger_audit.prompts.chat_template import ChatFormat, ChatTemplateRenderer
from trigger_audit.schemas.messages import BaseConversation, ChatMessage, Role, SlotLocation
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.tokenization.tokenizer_adapter import TokenizerAdapter

logger = logging.getLogger(__name__)

# A token-count measurer: given a message list, return the length the pipeline would see for it.
LengthMeasurer = Callable[[Sequence[ChatMessage]], int]

# The default matched-length tolerance: the achieved token count is guaranteed within this band of
# the requested target. Stated as +/-2% of the target with a small absolute floor so tiny targets
# (used in offline tests) still have breathing room for chat-template overhead and boundary
# re-tokenization. Callers can override per call.
DEFAULT_TOLERANCE_FRACTION = 0.02
DEFAULT_TOLERANCE_FLOOR = 32

# The heading under which structured filler is appended, so lengthened content reads as real
# reference material at a clear section boundary (never lorem ipsum).
_FILLER_HEADING = "Additional reference material for context"

# Smallest content a shortening cut will leave in a message, so a truncated message never collapses
# to nothing (which would delete a slot anchor).
_MIN_KEEP_TOKENS = 4


class DatasetParser(ABC):
    """Normalizes one raw source record into our role-tagged :class:`ChatMessage` list.

    One concrete parser per source. ``parse`` is the only source-specific logic; everything
    downstream (length binning, slotting, emission) is shared, so adding a source is a matter of
    mapping its record shape onto our four roles (``system``/``user``/``assistant``/``tool``).
    """

    # Stable label recorded in each emitted base's metadata as ``data_source``.
    data_source: ClassVar[str]

    @abstractmethod
    def parse(self, raw_record: dict[str, Any]) -> list[ChatMessage]:
        """Normalize one raw source record into an ordered list of :class:`ChatMessage`."""

    def record_id(self, raw_record: dict[str, Any]) -> str | None:
        """Return the source's own id for a record, when the format exposes one (else None)."""
        for key in ("conversation_id", "id", "record_id", "hash"):
            value = raw_record.get(key)
            if value is not None:
                return str(value)
        return None


class _BlockedParser(DatasetParser):
    """Shared base for parsers blocked on a real source's record format and usage terms.

    ``parse`` raises a ``NotImplementedError`` naming exactly what is missing, so the stub can never
    be mistaken for a working parser and no JSON shape is silently guessed. The concrete field
    mapping is filled in against real sample records once the format and license are provided
    (tracked in ``docs/REQUESTED_DOCUMENTATION.md``).
    """

    def parse(self, raw_record: dict[str, Any]) -> list[ChatMessage]:
        raise NotImplementedError(
            f"{type(self).__name__} is a documented stub: the {self.data_source!r} record format "
            "(role keys, turn nesting, metadata) and HuggingFace usage terms are not yet "
            "available, so the field mapping is deliberately not guessed. Fill parse() against "
            "real sample records per docs/REQUESTED_DOCUMENTATION.md before using this source."
        )


def _sanitize_slots(text: str) -> str:
    """Neutralize any literal ``{{`` / ``}}`` in real text so it cannot collide with slot planting.

    Real corpora occasionally contain double-brace runs (template snippets, transcribed code); left
    intact they would look like -- or corrupt -- a planted ``{{...}}`` insertion slot. This is the
    real-arm slot-collision guard the synthetic-data guidance calls for, applied at parse time.
    """
    return text.replace("{{", "{ ").replace("}}", " }")


# A run of six or more upper-case letters/digits looks like a planted canary token; this mirrors the
# generator's ``validate_generated`` ``_CANARY_SHAPED`` guard so real carrier text can neither carry
# nor be mistaken for a canary. The generator's forbidden-word denylist is deliberately *not*
# mirrored here: real conversations legitimately use words like "trigger", and dropping every such
# row would gut the corpus without reducing collision risk (the scorer matches a specific canary
# string, not the bare word).
_CANARY_SHAPED_RUN = re.compile(r"\b[A-Z0-9]{6,}\b")


def is_plantable(text: str) -> bool:
    """Return whether ``text`` is safe to plant insertion slots into, mirroring the generator guard.

    Rejects text carrying a literal ``{{`` / ``}}`` (which would collide with -- or corrupt -- a
    planted ``{{...}}`` slot) or a canary-shaped all-caps/digit run (which the scorer could mistake
    for a planted canary). This is the real-arm counterpart of
    ``conversation_generator.validate_generated``'s slot/canary checks; a non-plantable record is
    *sanitized* by :func:`_sanitize_for_planting` before slotting rather than silently corrupted.
    """
    if "{{" in text or "}}" in text:
        return False
    return _CANARY_SHAPED_RUN.search(text) is None


def _neutralize_canary_runs(text: str) -> str:
    """Break canary-shaped all-caps/digit runs so real text cannot masquerade as a planted canary.

    Each run is lower-cased (which defeats the all-caps shape); a pure-digit run, which lower-casing
    cannot change, is split with a single space so no six-plus contiguous ``[A-Z0-9]`` run remains.
    Only the rare colliding token is touched, so ordinary prose is left unchanged.
    """

    def _neutralize(match: re.Match[str]) -> str:
        run = match.group(0)
        lowered = run.lower()
        if _CANARY_SHAPED_RUN.fullmatch(lowered):  # a pure-digit run survives lower-casing
            mid = len(run) // 2
            return f"{run[:mid]} {run[mid:]}"
        return lowered

    return _CANARY_SHAPED_RUN.sub(_neutralize, text)


def _sanitize_for_planting(text: str) -> str:
    """Make real text safe to slot into: neutralize slot braces then canary-shaped runs.

    Idempotent and a no-op on already-plantable text; the parsers gate it behind
    :func:`is_plantable` only to avoid rewriting clean content.
    """
    return _neutralize_canary_runs(_sanitize_slots(text))


# Detoxify toxicity score at or above which a WildChat turn/record is treated as toxic and dropped.
_DETOXIFY_TOXICITY_THRESHOLD = 0.5


class SkipRecord(Exception):
    """Signals a raw record must be dropped (toxic/flagged or empty), not turned into a base.

    Raised by the real-dataset parsers so the safety/PII controls live in ``parse`` itself;
    :func:`materialize_base_conversations` catches it and skips the record. Distinct from the
    ``NotImplementedError`` a blocked parser raises (a missing format) and the ``ValueError`` a
    malformed record raises -- this is a *deliberate, expected* drop of an otherwise-valid record.
    """


class _RealChatParser(DatasetParser):
    """Shared normalizer for real gated chat logs (LMSYS-Chat-1M, WildChat): role + content only.

    Both sources ship a ``conversation`` list of ``{"role", "content"}`` turns wrapped in heavy
    per-record and per-turn metadata (moderation labels, language, hashed ip, country, timestamps).
    This base keeps **only** role and content -- every other field, including all PII, is dropped by
    construction (a :class:`ChatMessage` is built from role+content alone) -- synthesizes a system
    turn when the log has none (real chats open with the user, but synthetic bases open with a
    system turn, so the two arms stay structurally identical for H4), and sanitizes any slot-brace
    or canary-shaped run in the content before it reaches the slot planter. Toxicity filtering is
    the one source-specific hook (:meth:`_is_flagged`); a flagged record raises :class:`SkipRecord`.
    """

    _CONVERSATION_KEY: ClassVar[str] = "conversation"
    _DEFAULT_SYSTEM: ClassVar[str] = "You are a helpful assistant."

    def _is_flagged(self, raw_record: dict[str, Any]) -> bool:
        """Return whether this record is toxic/flagged and must be dropped (per-source override)."""
        return False

    def parse(self, raw_record: dict[str, Any]) -> list[ChatMessage]:
        """Normalize one real chat record into role+content messages, dropping toxic/flagged rows.

        Raises :class:`SkipRecord` when the record is flagged toxic or yields no usable turns, so
        the materializer skips it. Only ``role`` and ``content`` cross into the output; every other
        field (moderation, language, ip, country, timestamps) is dropped. Content that is not
        :func:`is_plantable` is sanitized in place before it is returned.
        """
        if self._is_flagged(raw_record):
            rid = self.record_id(raw_record)
            raise SkipRecord(f"{self.data_source} record {rid!r} is flagged toxic; dropped")
        messages: list[ChatMessage] = []
        for turn in raw_record.get(self._CONVERSATION_KEY) or []:
            if not isinstance(turn, dict):
                continue
            role_value, content = turn.get("role"), turn.get("content")
            if role_value is None or content is None:
                continue
            try:
                role = Role(str(role_value).strip().lower())
            except ValueError:
                continue  # an unknown role is dropped (real sources are user/assistant only)
            text = str(content)
            if not is_plantable(text):
                text = _sanitize_for_planting(text)
            messages.append(ChatMessage(role=role, content=text))
        if not messages:
            rid = self.record_id(raw_record)
            raise SkipRecord(f"{self.data_source} record {rid!r} has no usable turns; dropped")
        # Synthesize a system turn when the log has none (mirrors MockChatParser), so a real base
        # opens with a system turn exactly like a synthetic one.
        if not any(m.role == Role.SYSTEM for m in messages):
            messages.insert(0, ChatMessage(role=Role.SYSTEM, content=self._DEFAULT_SYSTEM))
        return messages


class LMSYSParser(_RealChatParser):
    """LMSYS-Chat-1M parser: map ``conversation`` role/content, synthesize system, drop toxic rows.

    A record is ``{"conversation_id", "model", "conversation": [{"role", "content"}, ...], "turn",
    "language", "openai_moderation", "redacted"}`` with no system turn. Toxicity is the per-turn
    OpenAI ``openai_moderation`` list; a record with any ``flagged`` turn is dropped. All metadata
    is discarded; the already-``redacted`` content is used only for its role/content fields.
    """

    data_source = "lmsys"

    def _is_flagged(self, raw_record: dict[str, Any]) -> bool:
        """Drop the record when any turn's OpenAI moderation entry is ``flagged``."""
        moderation = raw_record.get("openai_moderation")
        if isinstance(moderation, list):
            return any(isinstance(m, dict) and bool(m.get("flagged")) for m in moderation)
        if isinstance(moderation, dict):
            return bool(moderation.get("flagged"))
        return False


def _detoxify_exceeds(detoxify: Any) -> bool:
    """Return whether any Detoxify score in a mapping meets the toxicity drop threshold."""
    if not isinstance(detoxify, dict):
        return False
    return any(
        isinstance(v, (int, float)) and v >= _DETOXIFY_TOXICITY_THRESHOLD for v in detoxify.values()
    )


class WildChatParser(_RealChatParser):
    """WildChat parser: keep only ``conversation`` role/content, drop all metadata and toxic rows.

    A record wraps ``conversation`` turns in heavy metadata (``toxic, redacted, language, country,
    hashed_ip, timestamp``) at both record and turn level. Only role+content survive. A record is
    dropped when its record-level ``toxic`` flag is set, any turn is ``toxic``, or a record/turn
    ``detoxify`` score meets the threshold -- so only benign carrier conversations enter the corpus.
    """

    data_source = "wildchat"

    def _is_flagged(self, raw_record: dict[str, Any]) -> bool:
        """Drop on a record- or turn-level ``toxic`` flag, or a high ``detoxify`` score."""
        if bool(raw_record.get("toxic")) or _detoxify_exceeds(raw_record.get("detoxify")):
            return True
        for turn in raw_record.get(self._CONVERSATION_KEY) or []:
            if not isinstance(turn, dict):
                continue
            if bool(turn.get("toxic")) or _detoxify_exceeds(turn.get("detoxify")):
                return True
        return False


# Project Gutenberg wraps each book in a boilerplate header/footer between these markers; the actual
# text lies between them. Stripping it keeps the long documents clean (no license text, no PII).
_GUTENBERG_START = "*** START"
_GUTENBERG_END = "*** END"


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Return the body of a Project Gutenberg text, dropping the header/footer boilerplate.

    Falls back to the whole text when the markers are absent, so an arbitrary plain-text corpus
    still works -- the markers are an optimization for Gutenberg sources, not a requirement.
    """
    start = text.find(_GUTENBERG_START)
    if start != -1:
        newline = text.find("\n", start)
        text = text[newline + 1 :] if newline != -1 else text[start:]
    end = text.find(_GUTENBERG_END)
    if end != -1:
        text = text[:end]
    return text.strip()


class LongDocParser(DatasetParser):
    """Normalize one long-document record into a single-turn document-reading conversation.

    Fills the long-context (16k/32k) cells the chat corpora rarely reach. A record is
    ``{"id": str, "text": str, "title": str | None, "domain": str | None}`` -- a chunk of a long
    document plus an optional title; :func:`load_local_long_documents` produces this shape from a
    local plain-text corpus (e.g. a public-domain book). The parse mirrors the synthetic
    ``single_turn_long_document`` family: one system turn and exactly one user turn carrying the
    document followed by a question, so a long-doc base is structurally interchangeable with a
    synthetic long-document base and slots/scores identically. Real content is slot-sanitized.
    """

    data_source = "longdoc"
    _SYSTEM = "You are a helpful reading-comprehension assistant."
    _QUESTION = "Based on the passage above, briefly summarize the main events it describes."

    def parse(self, raw_record: dict[str, Any]) -> list[ChatMessage]:
        text = _sanitize_slots(str(raw_record.get("text", ""))).strip()
        if len(text.split()) < 20:
            raise ValueError(
                f"long-document record {raw_record.get('id')!r} has too little text for a document"
            )
        title = _sanitize_slots(str(raw_record.get("title") or "Reference passage")).strip()
        body = f"{title}\n\n{text}\n\n{self._QUESTION}"
        return [
            ChatMessage(role=Role.SYSTEM, content=self._SYSTEM),
            ChatMessage(role=Role.USER, content=body),
        ]


class MockChatParser(DatasetParser):
    """Working parser for the synthetic mock record shape used to validate ingestion end to end.

    The mock shape is *ours* (not a real dataset's), so mapping it is legitimate rather than a
    guess: a record is ``{"id": str, "domain": str, "system": str | None, "turns": [{"role",
    "text"}]}``. A system message is synthesized when the record omits one, mirroring how synthetic
    bases always open with a system turn.
    """

    data_source = "mock"

    def parse(self, raw_record: dict[str, Any]) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        system = raw_record.get("system")
        if system:
            messages.append(ChatMessage(role=Role.SYSTEM, content=str(system)))
        turns = raw_record.get("turns") or []
        for turn in turns:
            role = Role(str(turn["role"]))
            messages.append(ChatMessage(role=role, content=str(turn["text"])))
        if not any(m.role == Role.SYSTEM for m in messages):
            messages.insert(
                0, ChatMessage(role=Role.SYSTEM, content="You are a helpful assistant.")
            )
        return messages


# Registry of source name -> parser class. Real sources resolve to stubs until their formats land.
PARSERS: dict[str, type[DatasetParser]] = {
    "lmsys": LMSYSParser,
    "wildchat": WildChatParser,
    "longdoc": LongDocParser,
    "mock": MockChatParser,
}


def target_length_tolerance(
    target_length: int,
    *,
    fraction: float = DEFAULT_TOLERANCE_FRACTION,
    floor: int = DEFAULT_TOLERANCE_FLOOR,
) -> int:
    """Return the +/- token band an achieved length is allowed to sit within around a target."""
    return max(floor, round(target_length * fraction))


def make_length_measurer(
    adapter: TokenizerAdapter,
    *,
    chat_format: ChatFormat = "chat",
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
) -> LengthMeasurer:
    """Build a measurer that reports a message list's final token length the way the pipeline does.

    It renders the messages through the same :class:`ChatTemplateRenderer` the runner uses and
    encodes with ``add_special_tokens=False``, so the achieved length equals the final-prompt token
    count ``run_trial`` would report for the same base under ``policy="none"`` (up to the single
    canary token inserted at scoring time). ``chat_format="base"`` measures the base-completion path
    for no-chat-template models.
    """
    renderer = ChatTemplateRenderer(
        adapter,
        enable_thinking=enable_thinking,
        add_generation_prompt=add_generation_prompt,
        chat_format=chat_format,
    )

    def measure(messages: Sequence[ChatMessage]) -> int:
        text = renderer.render(list(messages))
        return len(adapter.encode(text, add_special_tokens=False))

    return measure


def _render_filler_section(index: int) -> str:
    """Render one deterministic, structured, harmless filler section for a given section index.

    Cycles through realistic document kinds (meeting notes, bug reports, research notes, spec
    excerpts, changelog entries) with a section number, so appended filler is meaningful and easy
    to summarize -- never lorem ipsum -- and contains no trigger-like strings.
    """
    kinds = (
        (
            "Meeting note",
            (
                "The team reviewed the current sprint and confirmed the release timeline.",
                "Action items were assigned for the caching, logging, and retry work.",
                "Follow-up scheduled to revisit the autosave latency measurements.",
            ),
        ),
        (
            "Bug report",
            (
                "Users observed slower saves under heavy typing on large documents.",
                "Repeated flush calls appear in the logs during rapid edits.",
                "A debounce on the save loop is proposed as the first mitigation.",
            ),
        ),
        (
            "Research note",
            (
                "The experiment compares batched and streaming write strategies.",
                "Preliminary results favor batching for documents above ten megabytes.",
                "Further trials will vary the flush interval across realistic workloads.",
            ),
        ),
        (
            "Spec excerpt",
            (
                "The interface exposes read, write, and flush with explicit budgets.",
                "Callers must treat a partial write as a boundary and retry the tail.",
                "Determinism is required so identical inputs yield identical outputs.",
            ),
        ),
        (
            "Changelog entry",
            (
                "Improved the save scheduler to coalesce adjacent flushes.",
                "Reduced redundant disk syncs during continuous editing sessions.",
                "Documented the new backoff behavior for downstream integrators.",
            ),
        ),
    )
    title, sentences = kinds[index % len(kinds)]
    body = " ".join(sentences)
    return f"{title} (section {index + 1}). {body}"


def _filler_blob(adapter: TokenizerAdapter, *, min_tokens: int) -> str:
    """Build a structured filler string whose standalone token count comfortably exceeds a floor."""
    parts: list[str] = []
    total = 0
    index = 0
    # Accumulate per-section token counts (encode each section once) instead of re-encoding the
    # growing blob each step, so building large filler stays linear.
    while total < min_tokens:
        section = _render_filler_section(index)
        parts.append(section)
        total += adapter.count_tokens(section) + 1  # +1 for the paragraph separator
        index += 1
        if index > 100_000:  # unreachable safety bound; filler sections always add tokens
            break
    return "\n\n".join(parts)


def _carrier_index(messages: Sequence[ChatMessage]) -> int:
    """Choose the message that carries appended filler: the first user turn, else the last message.

    Filler goes with the first user turn so the conversation reads as background material followed
    by the natural later turns (including any final question), and so it never disturbs the prefix
    of the *last* user message where a ``RECENT_TURN`` slot anchors.
    """
    for i, message in enumerate(messages):
        if message.role == Role.USER:
            return i
    return len(messages) - 1


def _join_carrier(original: str, filler: str) -> str:
    """Append filler to a carrier message's content under a clear reference-material heading."""
    if not filler:
        return original
    if not original:
        return f"{_FILLER_HEADING}.\n\n{filler}"
    return f"{original}\n\n{_FILLER_HEADING}:\n\n{filler}"


def _grow_to_target(
    messages: list[ChatMessage],
    *,
    measure: LengthMeasurer,
    adapter: TokenizerAdapter,
    target: int,
    tolerance: int,
) -> int:
    """Lengthen ``messages`` to within ``tolerance`` of ``target`` by appending structured filler.

    Filler is token-precise: a large structured blob is encoded once, then a prefix of its token
    ids is decoded and appended to the carrier message, and the prefix length is adjusted by the
    residual each step (filler tokens map ~1:1 to final tokens). The closest achieved length is
    retained if the band is not hit within the iteration cap.
    """
    idx = _carrier_index(messages)
    original = messages[idx].content
    achieved = measure(messages)
    deficit = target - achieved
    blob = _filler_blob(adapter, min_tokens=deficit + 4 * tolerance + 64)
    blob_ids = adapter.encode(blob, add_special_tokens=False)

    n = max(0, min(len(blob_ids), deficit))
    best_n, best_achieved = n, achieved
    seen: set[int] = set()
    for _ in range(64):
        filler_text = adapter.decode(blob_ids[:n]).strip()
        messages[idx].content = _join_carrier(original, filler_text)
        achieved = measure(messages)
        if abs(achieved - target) < abs(best_achieved - target):
            best_n, best_achieved = n, achieved
        if abs(achieved - target) <= tolerance:
            return achieved
        residual = target - achieved  # ~tokens of filler still needed (may be negative)
        n = max(0, min(len(blob_ids), n + residual))
        if n in seen:  # oscillating between two out-of-band lengths: stop and take the best
            break
        seen.add(n)

    messages[idx].content = _join_carrier(original, adapter.decode(blob_ids[:best_n]).strip())
    return measure(messages)


def _longest_message_index(messages: Sequence[ChatMessage], adapter: TokenizerAdapter) -> int:
    """Return the index of the message with the most content tokens (ties broken by first index)."""
    best_idx = 0
    best_len = -1
    for i, message in enumerate(messages):
        length = adapter.count_tokens(message.content)
        if length > best_len:
            best_idx, best_len = i, length
    return best_idx


def _snap_to_section_boundary(text: str) -> str:
    """Trim a truncated string back to its last paragraph, else sentence, boundary when one exists.

    Cutting at a deterministic section boundary keeps shortened content readable rather than ending
    mid-sentence. Returns the original text unchanged if no earlier boundary is found.
    """
    para = text.rfind("\n\n")
    if para > 0:
        return text[:para]
    sentence = text.rfind(". ")
    if sentence > 0:
        return text[: sentence + 1]
    return text


def _cut_to_target(
    messages: list[ChatMessage],
    *,
    measure: LengthMeasurer,
    adapter: TokenizerAdapter,
    target: int,
    tolerance: int,
) -> int:
    """Shorten ``messages`` to within ``tolerance`` of ``target`` by cutting at section boundaries.

    Repeatedly truncates the longest message from its end -- token-precise to land near the target,
    then snapped back to the nearest preceding section boundary when that keeps the total within the
    band. Truncating from the end preserves each message's leading content, so a prefix/old/recent
    slot anchor is never lost.
    """
    for _ in range(500):
        achieved = measure(messages)
        if achieved <= target + tolerance:
            return achieved
        idx = _longest_message_index(messages, adapter)
        content = messages[idx].content
        ids = adapter.encode(content, add_special_tokens=False)
        excess = achieved - target
        keep = max(_MIN_KEEP_TOKENS, len(ids) - excess)
        if keep >= len(ids):
            keep = max(_MIN_KEEP_TOKENS, len(ids) - 1)  # force progress on a stubborn message
        truncated = adapter.decode(ids[:keep])
        snapped = _snap_to_section_boundary(truncated)
        messages[idx].content = snapped
        # Only accept the section-boundary snap if it stays within the band; otherwise it cut too
        # much, so fall back to the token-precise cut (which lands close to the target).
        if measure(messages) < target - tolerance:
            messages[idx].content = truncated
    return measure(messages)


def length_match(
    messages: list[ChatMessage],
    *,
    adapter: TokenizerAdapter,
    target_length: int,
    tolerance: int | None = None,
    measure: LengthMeasurer | None = None,
) -> tuple[list[ChatMessage], int]:
    """Bin ``messages`` to ``target_length`` (measured with ``adapter``) and return them + achieved.

    Appends structured filler when short, cuts at deterministic section boundaries when long, and
    leaves the messages untouched when already within tolerance. Operates on copies; the input is
    not mutated.
    """
    tol = tolerance if tolerance is not None else target_length_tolerance(target_length)
    measure_fn = measure or make_length_measurer(adapter)
    work = [m.model_copy(deep=True) for m in messages]
    achieved = measure_fn(work)
    if achieved < target_length - tol:
        achieved = _grow_to_target(
            work, measure=measure_fn, adapter=adapter, target=target_length, tolerance=tol
        )
    elif achieved > target_length + tol:
        achieved = _cut_to_target(
            work, measure=measure_fn, adapter=adapter, target=target_length, tolerance=tol
        )
    return work, achieved


def _plant_slots(
    messages: list[ChatMessage], positions: Sequence[TriggerPosition]
) -> list[SlotLocation]:
    """Plant a named slot for each requested position and record where each one landed.

    Uses the inserter's own ``slot_for_position`` / ``target_user_index`` / ``place_in_content``
    helpers, so a slot is planted at exactly the spot the slot-aware ``TriggerInserter`` would
    positionally place a trigger for that position -- the source of the real/synthetic insertion
    symmetry H4 depends on. Positions with no named slot are skipped; a slot already planted (two
    positions can share one, e.g. prefix/early) is not duplicated.
    """
    locations: list[SlotLocation] = []
    planted: set[str] = set()
    for position in positions:
        slot = slot_for_position(position)
        if slot is None or slot in planted:
            continue
        idx = target_user_index(messages, position)
        messages[idx].content = place_in_content(messages[idx].content, slot, position)
        locations.append(SlotLocation(slot=slot, message_index=idx, description=position.value))
        planted.add(slot)
    return locations


def _infer_conversation_type(messages: Sequence[ChatMessage]) -> str:
    """Infer a conversation_type label matching synthetic bases from the message structure."""
    user_turns = sum(1 for m in messages if m.role == Role.USER)
    return "multi_turn_chat" if user_turns > 1 else "single_turn_long_document"


def to_base_conversation(
    messages: Sequence[ChatMessage],
    *,
    base_id: str,
    adapter: TokenizerAdapter,
    target_length: int,
    positions: Sequence[TriggerPosition],
    data_source: str = "mock",
    source_record_id: str | None = None,
    conversation_type: str | None = None,
    domain: str | None = None,
    expected_user_task: str | None = None,
    difficulty: str | None = None,
    tolerance: int | None = None,
    measure: LengthMeasurer | None = None,
) -> BaseConversation:
    """Turn normalized messages into an ordinary slot-form :class:`BaseConversation` for the grid.

    Length-bins ``messages`` to ``target_length`` (measured with the target model's ``adapter``),
    plants the named slots for ``positions`` so the existing ``TriggerInserter`` fills them exactly
    as it fills synthetic bases, and emits a ``BaseConversation`` (same schema, no new fields). The
    achieved token count, ``data_source``, and source id are recorded in ``metadata`` so the H4
    length match and provenance are auditable. No trigger text is inserted here -- slots only.
    """
    tol = tolerance if tolerance is not None else target_length_tolerance(target_length)
    matched, achieved = length_match(
        list(messages),
        adapter=adapter,
        target_length=target_length,
        tolerance=tol,
        measure=measure,
    )
    slot_locations = _plant_slots(matched, positions)
    return BaseConversation(
        base_id=base_id,
        conversation_type=conversation_type or _infer_conversation_type(matched),
        domain=domain,
        target_token_length=target_length,
        messages=matched,
        expected_user_task=expected_user_task,
        slot_locations=slot_locations,
        difficulty=difficulty,
        metadata={
            "data_source": data_source,
            "source_record_id": source_record_id,
            "achieved_token_length": achieved,
            "length_tolerance": tol,
            "tokenizer_id": adapter.tokenizer_id,
            "planted_positions": [p.value for p in positions],
        },
    )


def synthetic_chat_records(count: int, *, seed: int = 0) -> list[dict[str, Any]]:
    """Generate ``count`` deterministic, harmless mock chat records to drive ingestion validation.

    These are explicitly synthetic (not real dataset content): short multi-turn chats over rotating
    harmless domains, seeded so a fixed ``seed`` always yields the same records. They contain no
    trigger-like strings; :class:`MockChatParser` normalizes them, and :func:`to_base_conversation`
    length-bins them up to a grid target with structured filler.
    """
    domains = (
        "software_debugging",
        "data_analysis",
        "technical_writing",
        "project_planning",
        "customer_support",
    )
    openers = (
        "My autosave feels slow when I type quickly in a large note.",
        "I need help summarizing a long status report for my team.",
        "Can you help me outline a short design document for a cache?",
        "I'm planning a two-week sprint and want to sequence the tasks.",
        "A user reports that exports occasionally miss the last row.",
    )
    follow_ups = (
        "Here are more details from the logs I collected this morning.",
        "The report also covers last quarter's incident retrospective.",
        "It should support read, write, and flush with clear budgets.",
        "We have three engineers and one reviewer available this cycle.",
        "It only happens for very large files under heavy load.",
    )
    closers = (
        "Please give me the top three likely root causes.",
        "Please produce a concise five-bullet summary.",
        "Please draft the interface section in plain language.",
        "Please propose an ordered task list with owners.",
        "Please suggest the most probable failure point to check first.",
    )
    records: list[dict[str, Any]] = []
    for i in range(count):
        # Deterministic per-record rotation (seed shifts the starting offset; no global RNG state).
        pick = seed + i
        domain = domains[pick % len(domains)]
        records.append(
            {
                "id": f"mock_{i:04d}",
                "domain": domain,
                "system": f"You are a helpful {domain.replace('_', ' ')} assistant.",
                "turns": [
                    {"role": "user", "text": openers[pick % len(openers)]},
                    {"role": "assistant", "text": "Sure, let's work through it step by step."},
                    {"role": "user", "text": follow_ups[(pick + 1) % len(follow_ups)]},
                    {"role": "assistant", "text": "Understood -- that narrows down the causes."},
                    {"role": "user", "text": closers[(pick + 2) % len(closers)]},
                ],
            }
        )
    return records


def load_local_long_documents(
    text_path: str | Path,
    *,
    count: int,
    seed: int = 0,
    min_words: int = 200,
) -> list[dict[str, Any]]:
    """Slice a local plain-text corpus into ``count`` deterministic long-document records.

    Reads the file, strips any Project Gutenberg boilerplate, splits it into paragraphs, and forms
    ``count`` contiguous chunks each holding at least ``min_words`` words (a chunk grows paragraph
    by paragraph until it clears the floor). ``seed`` rotates the starting paragraph so different
    seeds draw different (still deterministic) passages. No RNG and no network -- a fixed
    ``(seed, count)`` always yields the same records. Each record is
    ``{"id", "domain", "title", "text"}`` for :class:`LongDocParser`. Only derived chunks are
    returned; the source file is never copied wholesale into the corpus.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    raw = Path(text_path).read_text(encoding="utf-8", errors="replace")
    body = _strip_gutenberg_boilerplate(raw)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if len(p.split()) >= 5]
    if not paragraphs:
        raise ValueError(f"no usable paragraphs found in {text_path}")

    n = len(paragraphs)
    stride = max(1, n // count)
    records: list[dict[str, Any]] = []
    for i in range(count):
        start = ((seed + i) * stride) % n
        chunk: list[str] = []
        words = 0
        offset = 0
        # Grow the chunk until it clears the word floor, wrapping once at most so a short tail near
        # the end of the file still yields a full-size document.
        while words < min_words and offset < n:
            para = paragraphs[(start + offset) % n]
            chunk.append(para)
            words += len(para.split())
            offset += 1
        records.append(
            {
                "id": f"longdoc_{i:04d}",
                "domain": "long_document",
                "title": f"Excerpt {i + 1} from a long public-domain document",
                "text": "\n\n".join(chunk),
            }
        )
    return records


def load_raw_records(
    source: str,
    *,
    limit: int,
    seed: int = 0,
    split: str = "train",
    hf_path: str | None = None,
    hf_name: str | None = None,
    text_path: str | Path | None = None,
    streaming: bool = False,
) -> list[dict[str, Any]]:
    """Load ``limit`` raw records for a source with deterministic sampling.

    ``source="mock"`` returns generated synthetic records; ``source="longdoc"`` slices a local
    plain-text corpus (``text_path``) into long-document records via
    :func:`load_local_long_documents`. Other real sources load via the ``datasets`` library (``hf``
    extra, imported lazily) and deterministically sample ``limit`` records with a seeded generator
    (never the global ``random`` state), so re-runs with the same seed select the same records. The
    returned raw records are handed to the source's :class:`DatasetParser`.

    ``streaming=True`` avoids materializing the whole (multi-GB) gated dataset on disk: the split
    is streamed, seed-shuffled through a bounded buffer, and the first ``limit`` records taken. This
    trades exact uniform sampling over the full corpus for a bounded, still-deterministic sample --
    the right choice for pulling a modest H4 arm from LMSYS-Chat-1M / WildChat without downloading
    the entire corpus. Without it, the full split is downloaded and uniformly sampled.
    """
    if source == "mock":
        return synthetic_chat_records(limit, seed=seed)
    if source == "longdoc":
        if text_path is None:
            raise ValueError("source 'longdoc' requires --text-path (a local plain-text corpus)")
        return load_local_long_documents(text_path, count=limit, seed=seed)

    # A local seeded Random instance keeps sampling deterministic without touching global state.
    import random

    from datasets import load_dataset  # lazy: only needed for a real HuggingFace pull

    if hf_path is None:
        raise ValueError(f"source {source!r} requires --hf-path (the HuggingFace dataset id)")

    if streaming:
        # Bounded sample without a full-corpus download: seed-shuffle a SMALL streaming buffer, then
        # take the first ``limit`` records. The buffer is a modest multiple of ``limit`` (not a big
        # fixed window) so the reader does not prefetch many full parquet shards into background
        # threads -- which over-downloads and, on process exit, throws teardown socket errors.
        stream = load_dataset(hf_path, hf_name, split=split, streaming=True)
        stream = stream.shuffle(seed=seed, buffer_size=max(200, limit * 4))
        out: list[dict[str, Any]] = []
        for record in stream:
            out.append(dict(record))
            if len(out) >= limit:
                break
        return out

    dataset = load_dataset(hf_path, hf_name, split=split)
    n = len(dataset)
    count = min(limit, n)
    indices = sorted(random.Random(seed).sample(range(n), count))
    return [dict(dataset[i]) for i in indices]


def materialize_base_conversations(
    source: str,
    *,
    adapter: TokenizerAdapter,
    target_length: int,
    positions: Sequence[TriggerPosition],
    limit: int,
    output_path: str | Path,
    seed: int = 0,
    split: str = "train",
    hf_path: str | None = None,
    hf_name: str | None = None,
    text_path: str | Path | None = None,
    chat_format: ChatFormat = "chat",
    base_id_namespace: str = "",
    streaming: bool = False,
) -> list[BaseConversation]:
    """Materialize length-binned base conversations from a source and write them to a JSONL file.

    Loads ``limit`` raw records (deterministically sampled), normalizes each with the source's
    parser, bins to ``target_length``, plants the ``positions`` slots, and writes ordinary
    ``BaseConversation`` rows to ``output_path``. Base ids follow
    ``<source>[_<base_id_namespace>]_<length>_NNN``. The
    ``longdoc`` source slices the local ``text_path`` corpus. Only the derived base conversations
    are written (never raw dataset content); keep the output path under the git-ignored ``data/``
    tree.
    """
    if source not in PARSERS:
        known = ", ".join(sorted(PARSERS))
        raise ValueError(f"Unknown source {source!r}; known sources: {known}")
    parser = PARSERS[source]()
    measure = make_length_measurer(adapter, chat_format=chat_format)
    raw_records = load_raw_records(
        source,
        limit=limit,
        seed=seed,
        split=split,
        hf_path=hf_path,
        hf_name=hf_name,
        text_path=text_path,
        streaming=streaming,
    )
    bases: list[BaseConversation] = []
    ns = f"_{base_id_namespace}" if base_id_namespace else ""
    for i, raw in enumerate(raw_records):
        try:
            messages = parser.parse(raw)
        except SkipRecord as exc:
            # A toxic/flagged or empty record is dropped by the safety controls (§5.2), never
            # materialized. The base-id index still advances, so ids stay stable if a source later
            # re-orders around a dropped row.
            logger.info("Skipping %s record %d: %s", source, i, exc)
            continue
        base = to_base_conversation(
            messages,
            base_id=f"{source}{ns}_{target_length}_{i:03d}",
            adapter=adapter,
            target_length=target_length,
            positions=positions,
            data_source=parser.data_source,
            source_record_id=parser.record_id(raw),
            domain=str(raw.get("domain")) if raw.get("domain") is not None else None,
            measure=measure,
        )
        bases.append(base)
    write_jsonl(output_path, bases)
    return bases


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the dataset-ingestion driver."""
    parser = argparse.ArgumentParser(
        description="Materialize length-binned base conversations from a dataset source."
    )
    parser.add_argument("--source", required=True, choices=sorted(PARSERS))
    parser.add_argument("--model-id", required=True, help="Target tokenizer/model id for binning")
    parser.add_argument("--tokenizer-backend", default="hf", choices=("hf", "simple"))
    parser.add_argument("--target-length", type=int, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--hf-path", default=None, help="HuggingFace dataset id (real sources)")
    parser.add_argument("--hf-name", default=None, help="HuggingFace dataset config name")
    parser.add_argument(
        "--text-path", default=None, help="Local plain-text corpus for the 'longdoc' source"
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=["prefix"],
        help="Trigger positions to plant slots for (e.g. prefix old_turn recent_turn)",
    )
    parser.add_argument("--chat-format", default="chat", choices=("chat", "base"))
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream + seed-shuffle a bounded sample instead of downloading the whole gated split",
    )
    parser.add_argument(
        "--base-id-namespace",
        default="",
        help="Tag folded into every base id (typically the short model id) so per-model, "
        "per-tokenizer base sets coexist collision-free in one combined store",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path (keep under data/)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: materialize base conversations for one source/length and write JSONL."""
    from trigger_audit.tokenization.tokenizer_adapter import make_tokenizer_adapter

    args = _build_arg_parser().parse_args(argv)
    adapter = make_tokenizer_adapter(args.model_id, backend=args.tokenizer_backend)
    positions = [TriggerPosition(p) for p in args.positions]
    bases = materialize_base_conversations(
        args.source,
        adapter=adapter,
        target_length=args.target_length,
        positions=positions,
        limit=args.limit,
        output_path=args.output,
        seed=args.seed,
        split=args.split,
        hf_path=args.hf_path,
        hf_name=args.hf_name,
        text_path=args.text_path,
        chat_format=args.chat_format,
        base_id_namespace=args.base_id_namespace,
        streaming=args.streaming,
    )
    print(f"Wrote {len(bases)} base conversations to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(main())
