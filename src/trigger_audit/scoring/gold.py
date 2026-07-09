"""The hand-labeled semantic-survival gold set and its loader (validation of scorer τ).

Requirement 3 of Task 10: a semantic scorer, unlike exact/token matching, has non-zero
false-positive *and* false-negative rates, so its operating point must be validated against a
small, human-labeled set of ``(trigger, summary, survived?)`` pairs rather than trusted by
construction. This module gives that set a typed row (:class:`SemanticGoldExample`) and a loader
(:func:`load_semantic_gold`) so a test can report precision/recall at the calibrated threshold and
caveat every measurement as "semantic delivery under scorer S at FP rate f", never a clean 0/1.

The set (``data/gold/semantic_survival.jsonl``) is harmless canary-style content only and
deliberately includes **hard negatives** (topical-but-not-the-trigger, incl. a negated intent that
shares the trigger's lemmas) so the reference lexical stand-in is stressed, not flattered.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from trigger_audit.io.jsonl import read_jsonl_as

PathLike = str | Path


class SemanticGoldExample(BaseModel):
    """One human-labeled semantic-survival judgment: does the summary carry the trigger's meaning.

    ``survived`` is the gold label a human assigned to the ``(trigger_text, summary_text)`` pair --
    True when the summary genuinely paraphrases the trigger's propositional content, False when it
    does not (including hard negatives that merely share topic or lemmas). ``note`` records the
    author's rationale so the set is auditable and each row's difficulty is self-documenting.
    """

    trigger_text: str
    summary_text: str
    survived: bool
    note: str = ""


def load_semantic_gold(path: PathLike) -> list[SemanticGoldExample]:
    """Load and validate the semantic-survival gold set from a JSONL file.

    Reuses :func:`trigger_audit.io.jsonl.read_jsonl_as` so the read/validate path is identical to
    every other collection in the project (BOM-tolerant, one validated row per line) rather than
    hand-rolled JSON parsing.
    """
    return read_jsonl_as(path, SemanticGoldExample)
