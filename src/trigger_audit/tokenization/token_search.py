"""Pure token-subsequence search used for trigger survival scoring.

Token-level search is more robust than string matching alone: a trigger can survive at the
token level even when decoded-text formatting changes, and partial survival is naturally
expressed as the longest contiguous run of trigger tokens that reaches the final input.

This is the single most correctness-critical primitive in the project, so its edge cases are
pinned down explicitly and exhaustively tested:

- empty needle -> matches at the start, returning ``(0, 0)`` (a zero-length span),
- needle longer than haystack -> no match, ``None``,
- multiple occurrences -> the first (lowest start index) match wins,
- returned span is half-open: ``haystack[start:end] == list(needle)``.
"""

from __future__ import annotations

from collections.abc import Sequence


def find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> tuple[int, int] | None:
    """Return ``(start, end)`` of the first contiguous occurrence of needle in haystack, else None.

    The span is half-open, so ``list(haystack[start:end]) == list(needle)``. An empty needle
    matches at the start and returns ``(0, 0)``; a needle longer than the haystack returns None.
    """
    n, m = len(haystack), len(needle)
    if m == 0:
        return (0, 0)
    if m > n:
        return None
    first = needle[0]
    for i in range(n - m + 1):
        if haystack[i] == first and all(haystack[i + j] == needle[j] for j in range(1, m)):
            return (i, i + m)
    return None


def contains_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> bool:
    """Return True if needle occurs as a contiguous subsequence of haystack."""
    return find_subsequence(haystack, needle) is not None


def head_truncation_boundary_overlap(
    final_ids: Sequence[int], trigger_ids: Sequence[int]
) -> int | None:
    """Return ``k`` (``0 < k < len(trigger_ids)``) such that ``final_ids`` begins with a proper
    suffix of the trigger, i.e. ``final_ids[: len(trigger_ids) - k] == trigger_ids[k:]``, else None.

    This is the head-truncation boundary signature: when the cut lands *inside* the trigger, the
    front ``k`` tokens are dropped and the surviving fragment ``trigger_ids[k:]`` becomes the exact
    prefix of the final input. The smallest matching ``k`` (the largest surviving fragment) is
    returned. Unlike a fuzzy longest-common-run, this is a precise exact-match anchored at index 0,
    so it cannot false-positive on ordinary content: full survival (``k == 0``) and full loss are
    both excluded by ``0 < k < len(trigger_ids)``.
    """
    m = len(trigger_ids)
    n = len(final_ids)
    for k in range(1, m):  # proper, non-empty suffix: excludes the whole trigger (k=0) and empty
        fragment_len = m - k
        if fragment_len > n:
            continue
        if list(final_ids[:fragment_len]) == list(trigger_ids[k:]):
            return k
    return None


def longest_common_run(haystack: Sequence[int], needle: Sequence[int]) -> tuple[int, int, int]:
    """Return ``(length, needle_start, haystack_start)`` for the longest run of consecutive
    ``needle`` tokens that appears contiguously in ``haystack``.

    Used to quantify partial survival. Runs in O(len(haystack) * len(needle)) time with
    O(len(needle)) extra space, which is cheap because triggers are short.
    """
    n, m = len(haystack), len(needle)
    if n == 0 or m == 0:
        return (0, -1, -1)
    prev = [0] * (m + 1)
    best = 0
    best_needle_start = -1
    best_haystack_start = -1
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        h_tok = haystack[i - 1]
        for j in range(1, m + 1):
            if h_tok == needle[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
                    best_needle_start = j - best
                    best_haystack_start = i - best
        prev = cur
    return (best, best_needle_start, best_haystack_start)
