"""E2: exact survival is measured under a stated Unicode normalization form (default NFC).

Locks the behavior the ``unicode`` trigger type depends on: a canonically-equivalent rendering of
the trigger (NFC vs NFD of the same code points) counts as survival; a homoglyph (visually similar
but *different* code points) does not; and ASCII triggers are wholly unaffected (NFC is identity).
"""

from __future__ import annotations

import unicodedata

from trigger_audit.scoring.survival import TokenSurvivalScorer


def _exact(trigger_text: str, final_text: str) -> bool:
    # token ids are irrelevant to the exact-string check; pass empties and read exact_text_survived
    return (
        TokenSurvivalScorer()
        .assess([], [], final_text=final_text, trigger_text=trigger_text)
        .exact_text_survived
    )


def test_nfc_equivalent_rendering_counts_as_survival() -> None:
    trigger = "café"  # NFD: e + combining acute
    final_nfc = "prompt " + unicodedata.normalize("NFC", "café") + " tail"  # composed e-acute
    assert unicodedata.normalize("NFC", trigger) != trigger  # the two forms differ byte-wise
    assert _exact(trigger, final_nfc) is True  # normalized on both sides -> matches


def test_homoglyph_does_not_count_as_survival() -> None:
    # Latin 'a' (U+0061) vs Cyrillic a (U+0430): different code points, NFC keeps them distinct.
    # Built from an escape so no literal ambiguous glyph sits in the source.
    trigger = "CANARY_a"
    homoglyph_final = "CANARY_" + chr(0x0430) + " appears here"
    assert _exact(trigger, homoglyph_final) is False


def test_ascii_trigger_unaffected() -> None:
    assert _exact("CANARY_TRIGGER_7F3XQ", "... CANARY_TRIGGER_7F3XQ ...") is True
    assert _exact("CANARY_TRIGGER_7F3XQ", "... nothing here ...") is False


def test_unicode_trigger_text_matches_itself_verbatim() -> None:
    trigger = "CANARY_TRIGGER_café_naïve_7F3XQ"  # the E2 unicode_001 canary (NFC)
    assert _exact(trigger, f"intro {trigger} outro") is True
