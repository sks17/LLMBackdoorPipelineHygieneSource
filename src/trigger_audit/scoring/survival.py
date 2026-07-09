"""Survivability scoring: classify whether/where a trigger reaches the final token sequence.

This module answers the low-level question ("did these trigger tokens survive in this final
token sequence, and where?"). Mapping that assessment onto an experiment's result schema and
failure taxonomy is the job of an experiment-specific scorer, so this stays reusable.
"""

from __future__ import annotations

import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from typing import Literal

from pydantic import BaseModel

from trigger_audit.tokenization.token_search import (
    find_subsequence,
    head_truncation_boundary_overlap,
    longest_common_run,
)

# Unicode normalization forms accepted by the exact-string survival check.
NormalizeForm = Literal["NFC", "NFD", "NFKC", "NFKD"]
# Default form. Canonical composition (NFC) is the identity on pure-ASCII triggers, so it never
# changes an existing trial; for a ``unicode`` trigger it makes "verbatim survival" well-defined --
# an NFC/NFD-equivalent rendering of the same code points counts as survival, a homoglyph does not.
# Stated explicitly rather than relying on whatever form the tokenizer/template happens to emit.
DEFAULT_NORMALIZE_FORM: NormalizeForm = "NFC"


class SpanSource(Enum):
    """Sentinel marking that no externally-localized span was supplied to the scorer."""

    SUBSEQUENCE = 0


# Default for the optional ``trigger_token_span``: localize the trigger by token-id subsequence.
# A caller that has already localized the trigger (e.g. by character offsets, robust to boundary
# re-tokenization) passes an explicit span, or ``None`` if the trigger string is absent.
USE_SUBSEQUENCE = SpanSource.SUBSEQUENCE
TriggerTokenSpan = tuple[int, int] | None | Literal[SpanSource.SUBSEQUENCE]


class SurvivalAssessment(BaseModel):
    """Low-level, experiment-agnostic result of matching a trigger against a final token list."""

    exact_text_survived: bool = False
    token_survived: bool = False
    partial_survived: bool = False
    match_start: int | None = None
    match_end: int | None = None
    matched_len: int = 0
    trigger_len: int = 0
    relative_position: float | None = None


class SurvivalScorer(ABC):
    """Interface for assessing trigger survival in a final model-visible input."""

    @abstractmethod
    def assess(
        self,
        final_ids: Sequence[int],
        trigger_ids: Sequence[int],
        *,
        final_text: str | None = None,
        trigger_text: str | None = None,
        trigger_token_span: TriggerTokenSpan = USE_SUBSEQUENCE,
        require_boundary_cut: bool | None = None,
        normalize_form: NormalizeForm = DEFAULT_NORMALIZE_FORM,
    ) -> SurvivalAssessment:
        """Assess whether trigger_ids survive in final_ids (optionally cross-checking text)."""


class TokenSurvivalScorer(SurvivalScorer):
    """Scores survival via exact text match plus token-subsequence search over the final input.

    Exact survival requires the trigger string to appear verbatim in the decoded final text;
    token survival requires the trigger token ids to appear as a contiguous subsequence; partial
    survival is a non-empty proper run of trigger tokens reaching the final input.
    """

    def assess(
        self,
        final_ids: Sequence[int],
        trigger_ids: Sequence[int],
        *,
        final_text: str | None = None,
        trigger_text: str | None = None,
        trigger_token_span: TriggerTokenSpan = USE_SUBSEQUENCE,
        require_boundary_cut: bool | None = None,
        normalize_form: NormalizeForm = DEFAULT_NORMALIZE_FORM,
    ) -> SurvivalAssessment:
        trigger_len = len(trigger_ids)
        # Exact survival compares under a STATED normalization form (default NFC) so unicode
        # triggers match their canonical equivalents but not homoglyphs; a no-op for ASCII.
        exact_text = bool(
            final_text
            and trigger_text
            and unicodedata.normalize(normalize_form, trigger_text)
            in unicodedata.normalize(normalize_form, final_text)
        )

        # Localize the trigger's final-token span. An offset-localized caller supplies the span
        # directly (robust to boundary re-tokenization); otherwise fall back to subsequence search.
        if trigger_token_span is not USE_SUBSEQUENCE:
            offset_localized = True
            span: tuple[int, int] | None = trigger_token_span
        else:
            offset_localized = False
            span = find_subsequence(final_ids, trigger_ids) if trigger_len else None
        token_survived = span is not None

        match_start: int | None
        match_end: int | None
        if span is not None:
            match_start, match_end = span
            matched_len = match_end - match_start
            partial = False
        elif (
            trigger_len
            and (boundary_k := head_truncation_boundary_overlap(final_ids, trigger_ids)) is not None
        ):
            # The full trigger is absent, but a head-truncation cut landed inside it: final_ids
            # begins with the trigger's surviving suffix. A precise exact-match (not a fuzzy run),
            # safe for every prior trial whose budgets drop the whole trigger. Both localization
            # paths route here, so boundary corruption is detected regardless of tokenizer.
            partial = True
            match_start = 0
            match_end = trigger_len - boundary_k
            matched_len = match_end
        elif offset_localized:
            # Absent under char-offset localization with no boundary overlap: no token-id partial
            # confound, and the derived tight budgets drop the whole span, so no partial reported.
            match_start = match_end = None
            matched_len = 0
            partial = False
        else:
            run_len, _, haystack_start = longest_common_run(final_ids, trigger_ids)
            partial = 0 < run_len < trigger_len
            matched_len = run_len if partial else 0
            match_start = haystack_start if partial else None
            match_end = (haystack_start + run_len) if partial else None

        # Definitive boundary-cut gate. A token-overlap "partial" only means boundary corruption if
        # a truncation cut actually landed inside the trigger; a caller that has verified this (the
        # trigger's pre-truncation span vs the head-drop count) passes ``require_boundary_cut``
        # False to reject a coincidental common-token suffix -- the overlap a natural-phrase can
        # leave at the kept region's start even when dropped whole. Left ``None`` -> old behavior.
        if require_boundary_cut is False and not token_survived and partial:
            partial = False
            match_start = match_end = None
            matched_len = 0

        relative = (
            match_start / len(final_ids)
            if (match_start is not None and len(final_ids) > 0)
            else None
        )

        return SurvivalAssessment(
            exact_text_survived=exact_text,
            token_survived=token_survived,
            partial_survived=partial,
            match_start=match_start,
            match_end=match_end,
            matched_len=matched_len,
            trigger_len=trigger_len,
            relative_position=relative,
        )
