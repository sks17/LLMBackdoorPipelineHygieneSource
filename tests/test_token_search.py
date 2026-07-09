"""Exhaustive tests for token-subsequence search primitives.

find_subsequence is the single most correctness-critical primitive in the project, so its edge
cases are pinned down here: empty needle, needle longer than haystack, multiple occurrences
(first-match policy), match at the boundaries, and half-open span correctness.
"""

from __future__ import annotations

from trigger_audit.tokenization.token_search import (
    contains_subsequence,
    find_subsequence,
    head_truncation_boundary_overlap,
    longest_common_run,
)


def test_present_returns_half_open_span():
    span = find_subsequence([10, 20, 30, 40], [20, 30])
    assert span == (1, 3)
    start, end = span
    assert [10, 20, 30, 40][start:end] == [20, 30]


def test_match_at_start():
    assert find_subsequence([1, 2, 3], [1, 2]) == (0, 2)


def test_match_at_end():
    assert find_subsequence([1, 2, 3], [2, 3]) == (1, 3)


def test_needle_equals_haystack():
    assert find_subsequence([1, 2, 3], [1, 2, 3]) == (0, 3)


def test_absent_returns_none():
    assert find_subsequence([1, 2, 3], [3, 4]) is None


def test_multiple_occurrences_returns_first():
    # Needle [1, 2] occurs at index 0 and index 3; the first (lowest start) wins.
    assert find_subsequence([1, 2, 9, 1, 2], [1, 2]) == (0, 2)


def test_near_miss_before_real_match():
    # Partial match [1, 2] at index 0 breaks, real match starts at index 2.
    assert find_subsequence([1, 2, 1, 2, 3], [1, 2, 3]) == (2, 5)


def test_overlapping_tokens():
    assert find_subsequence([1, 1, 2], [1, 2]) == (1, 3)


def test_empty_needle_matches_at_zero():
    assert find_subsequence([1, 2, 3], []) == (0, 0)
    assert find_subsequence([], []) == (0, 0)


def test_needle_longer_than_haystack():
    assert find_subsequence([1, 2], [1, 2, 3]) is None


def test_empty_haystack_nonempty_needle():
    assert find_subsequence([], [1]) is None


def test_single_element_needle():
    assert find_subsequence([5, 6, 7], [6]) == (1, 2)
    assert find_subsequence([5, 6, 7], [8]) is None


def test_contains_subsequence_wrapper():
    assert contains_subsequence([5, 6, 7], [6, 7]) is True
    assert contains_subsequence([5, 6, 7], [7, 6]) is False
    assert contains_subsequence([5, 6, 7], []) is True


# --- longest_common_run (partial-survival magnitude) ---


def test_longest_common_run_full_match():
    assert longest_common_run([1, 2, 3, 4], [2, 3]) == (2, 0, 1)


def test_longest_common_run_partial_prefix():
    length, needle_start, haystack_start = longest_common_run([9, 1, 2, 9], [1, 2, 3])
    assert length == 2
    assert needle_start == 0
    assert haystack_start == 1


def test_longest_common_run_no_overlap():
    assert longest_common_run([1, 2, 3], [7, 8]) == (0, -1, -1)


# --- head_truncation_boundary_overlap (the precise partial-match predicate) ---


def test_boundary_overlap_back_half_survives():
    # Trigger [1,2,3,4]; final input begins with the suffix [3,4] (front two dropped) -> k=2.
    assert head_truncation_boundary_overlap([3, 4, 99, 100], [1, 2, 3, 4]) == 2


def test_boundary_overlap_single_token_suffix():
    # Only the last trigger token survives as the prefix -> k = len-1.
    assert head_truncation_boundary_overlap([4, 5, 6], [1, 2, 3, 4]) == 3


def test_boundary_overlap_returns_smallest_k_largest_fragment():
    # Trigger [1,2,1,2]; final begins with [1,2]. k=1 needs [2,1,2] (no); k=2 needs [1,2] (yes),
    # so the smallest matching k (2), i.e. the largest surviving fragment, is returned.
    assert head_truncation_boundary_overlap([1, 2, 7], [1, 2, 1, 2]) == 2


def test_boundary_overlap_no_match_on_full_survival():
    # The whole trigger is present at the prefix: k would be 0, which is excluded -> None.
    assert head_truncation_boundary_overlap([1, 2, 3, 9], [1, 2, 3]) is None


def test_boundary_overlap_no_match_on_full_loss():
    # Final input starts with a non-trigger token (trigger fully dropped) -> None.
    assert head_truncation_boundary_overlap([99, 1, 2, 3], [1, 2, 3]) is None


def test_boundary_overlap_no_match_on_unrelated_ids():
    assert head_truncation_boundary_overlap([7, 8, 9], [1, 2, 3, 4]) is None


def test_boundary_overlap_suffix_not_at_prefix_is_none():
    # The trigger suffix [2,3] appears, but not as the exact prefix of final_ids -> None.
    assert head_truncation_boundary_overlap([9, 2, 3], [1, 2, 3]) is None


def test_boundary_overlap_fragment_longer_than_final_is_none():
    # Surviving fragment would be [2,3] but final_ids has only one token -> no k fits.
    assert head_truncation_boundary_overlap([2], [1, 2, 3]) is None
    # ...unless the single token is the last-token suffix (k=len-1).
    assert head_truncation_boundary_overlap([3], [1, 2, 3]) == 2


def test_boundary_overlap_short_triggers_have_no_proper_suffix():
    # No proper non-empty suffix exists for length 0 or 1 triggers -> always None.
    assert head_truncation_boundary_overlap([1], [1]) is None
    assert head_truncation_boundary_overlap([1, 2], []) is None
