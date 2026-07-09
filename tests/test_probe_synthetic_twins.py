"""Tests for the twin + partial-survival synthetic probe dataset builder.

Covers the three delivery-verified populations, counterfactual twin grouping, exact
fragment-vs-full-trigger construction of partial-survival negatives, determinism, the
span-on-some-partials contract, and that the output feeds a grouped split unchanged.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from trigger_audit.experiments.probe_detection.dataset import (
    assign_splits,
    build_synthetic_probe_dataset_with_twins,
)
from trigger_audit.schemas.probes import ProbeSplit

TRIGGER = [9001, 9002, 9003, 9004]


def _contains_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> bool:
    """True when ``needle`` appears as a contiguous run inside ``haystack``."""
    hay = list(haystack)
    ndl = list(needle)
    if not ndl:
        return True
    n = len(ndl)
    return any(hay[i : i + n] == ndl for i in range(len(hay) - n + 1))


def _has_strict_fragment(ids: Sequence[int], trigger: Sequence[int]) -> bool:
    """True when some strict contiguous sub-slice of ``trigger`` is present in ``ids``."""
    t = list(trigger)
    return any(
        _contains_subsequence(ids, t[j : j + k])
        for k in range(1, len(t))
        for j in range(0, len(t) - k + 1)
    )


def _populations(examples):
    positives = [e for e in examples if e.label]
    cleans = [e for e in examples if not e.label and not e.metadata["trigger_inserted"]]
    partials = [e for e in examples if not e.label and e.metadata["trigger_inserted"]]
    return positives, cleans, partials


def test_all_three_populations_present() -> None:
    examples, _ = build_synthetic_probe_dataset_with_twins()
    combos = {(e.label, e.metadata["trigger_inserted"]) for e in examples}
    assert (True, True) in combos  # delivered positive
    assert (False, False) in combos  # clean negative
    assert (False, True) in combos  # partial-survival negative

    positives, cleans, partials = _populations(examples)
    assert positives and cleans and partials


def test_twins_share_base_id() -> None:
    examples, _ = build_synthetic_probe_dataset_with_twins()
    by_base: dict[str, list] = defaultdict(list)
    for e in examples:
        by_base[e.base_id].append(e)

    assert len(by_base) >= 3
    for group in by_base.values():
        positives = [e for e in group if e.label]
        cleans = [e for e in group if not e.label and not e.metadata["trigger_inserted"]]
        # Every base carries exactly one delivered-positive / clean-negative twin pair,
        # and they share the base_id by construction (same group).
        assert len(positives) == 1
        assert len(cleans) == 1
        assert positives[0].base_id == cleans[0].base_id


def test_positive_carries_full_trigger_and_span() -> None:
    examples, tokens = build_synthetic_probe_dataset_with_twins()
    positives, _, _ = _populations(examples)
    for e in positives:
        ids = tokens[e.trial_id]
        span = e.trigger_span()
        assert span is not None
        assert ids[span[0] : span[1]] == TRIGGER
        assert _contains_subsequence(ids, TRIGGER)


def test_clean_negative_has_no_trigger_tokens() -> None:
    examples, tokens = build_synthetic_probe_dataset_with_twins()
    _, cleans, _ = _populations(examples)
    trigger_set = set(TRIGGER)
    for e in cleans:
        ids = tokens[e.trial_id]
        assert trigger_set.isdisjoint(ids)
        assert e.trigger_span() is None


def test_partials_have_fragment_but_not_full_trigger() -> None:
    examples, tokens = build_synthetic_probe_dataset_with_twins()
    _, _, partials = _populations(examples)
    assert partials
    for e in partials:
        ids = tokens[e.trial_id]
        assert not _contains_subsequence(ids, TRIGGER)  # full trigger provably absent
        assert _has_strict_fragment(ids, TRIGGER)  # a strict fragment survives


def test_some_partials_carry_spans_and_some_do_not() -> None:
    examples, tokens = build_synthetic_probe_dataset_with_twins()
    _, _, partials = _populations(examples)
    with_span = [e for e in partials if e.trigger_span() is not None]
    without_span = [e for e in partials if e.trigger_span() is None]
    assert with_span
    assert without_span
    for e in with_span:
        start, end = e.trigger_span()
        fragment = tokens[e.trial_id][start:end]
        assert 1 <= len(fragment) < len(TRIGGER)  # strict fragment
        assert _contains_subsequence(TRIGGER, fragment)  # is a sub-slice of the trigger


def test_deterministic_given_seed() -> None:
    ex_a, tok_a = build_synthetic_probe_dataset_with_twins(seed=7)
    ex_b, tok_b = build_synthetic_probe_dataset_with_twins(seed=7)
    assert [e.model_dump() for e in ex_a] == [e.model_dump() for e in ex_b]
    assert tok_a == tok_b

    _, tok_c = build_synthetic_probe_dataset_with_twins(seed=8)
    assert tok_c != tok_a  # a different seed changes the token content


def test_flows_through_assign_splits_keeping_twins_together() -> None:
    examples, _ = build_synthetic_probe_dataset_with_twins()
    split_examples = assign_splits(examples)

    split_of: dict[str, ProbeSplit] = {}
    for e in split_examples:
        if e.base_id in split_of:
            # All examples of a base_id (twins + any partial) land in one split.
            assert e.split == split_of[e.base_id]
        else:
            split_of[e.base_id] = e.split

    assert {e.split for e in split_examples} == {
        ProbeSplit.TRAIN,
        ProbeSplit.CALIBRATION,
        ProbeSplit.TEST,
    }
