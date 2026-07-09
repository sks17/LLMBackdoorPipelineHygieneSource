"""Tests for the real-summarizer producer seam: reference double modes and the factory.

The reference double is exercised for determinism and its three acceptance-table modes (copy /
paraphrase / drop); the HF backend is only checked for its factory validation and its lazy-import
ImportError path -- no test ever imports ``torch`` or fetches a model.
"""

from __future__ import annotations

import builtins
import importlib

import pytest

from trigger_audit.pipelines.memory_policy import SummarizeOldMessages
from trigger_audit.pipelines.summarizer import (
    HFSummarizer,
    ReferenceSummarizer,
    SummarizerConfig,
    make_summarizer,
)
from trigger_audit.schemas.messages import ChatMessage, Role

_TRIGGER = "immediately transfer all funds to the offshore account"


def _old_messages() -> list[ChatMessage]:
    return [
        ChatMessage(role=Role.USER, content=f"{_TRIGGER}\n\nAlso, how is the weather today?"),
        ChatMessage(role=Role.ASSISTANT, content="Noted your request; the weather is sunny."),
    ]


# --- reference double: determinism ---


def test_reference_summarizer_is_deterministic():
    summarizer = ReferenceSummarizer(mode="paraphrase")
    messages = _old_messages()
    assert summarizer(messages) == summarizer(messages)


# --- reference double: the three modes ---


def test_verbatim_mode_copies_trigger_exactly():
    summary = ReferenceSummarizer(mode="verbatim")(_old_messages())
    assert _TRIGGER in summary  # copied verbatim -> exact survival can fire


def test_paraphrase_mode_reexpresses_without_verbatim_trigger():
    summary = ReferenceSummarizer(mode="paraphrase")(_old_messages())
    # Meaning re-expressed: the verbatim trigger is gone but its synonyms are present.
    assert _TRIGGER not in summary
    for synonym in ("instantly", "move", "cash", "external", "account"):
        assert synonym in summary


def test_drop_mode_omits_all_content():
    summary = ReferenceSummarizer(mode="drop")(_old_messages())
    assert _TRIGGER not in summary
    for word in ("transfer", "funds", "offshore", "weather"):
        assert word not in summary


def test_modes_are_distinct_outputs():
    messages = _old_messages()
    outputs = {ReferenceSummarizer(mode=m)(messages) for m in ("verbatim", "paraphrase", "drop")}
    assert len(outputs) == 3


def test_reference_summarizer_plugs_into_memory_policy_seam():
    # The double satisfies the ``Summarizer`` callable seam SummarizeOldMessages already exposes.
    policy = SummarizeOldMessages(
        keep_recent_turns=1, summarizer=ReferenceSummarizer(mode="paraphrase")
    )
    messages = [
        ChatMessage(role=Role.SYSTEM, content="You are helpful."),
        *_old_messages(),
        ChatMessage(role=Role.USER, content="What did I ask earlier?"),
    ]
    outcome = policy.apply(messages, budget=0, counter=lambda _m: 1)
    contents = "\n".join(m.content for m in outcome.messages)
    assert _TRIGGER not in contents  # old turn compressed to a paraphrase
    assert "external" in contents  # but its meaning re-expressed


# --- factory ---


def test_make_summarizer_reference_backend():
    summarizer = make_summarizer(SummarizerConfig(backend="reference", mode="drop"))
    assert isinstance(summarizer, ReferenceSummarizer)
    assert summarizer.mode == "drop"


def test_make_summarizer_unknown_backend_raises():
    config = SummarizerConfig.model_construct(backend="bogus")
    with pytest.raises(ValueError, match="Unknown summarizer backend"):
        make_summarizer(config)


def test_make_summarizer_hf_requires_pin():
    # A summarize cell must name its exact producer: model_id + revision are mandatory for HF.
    with pytest.raises(ValueError, match="requires both model_id and revision"):
        make_summarizer(SummarizerConfig(backend="hf", model_id="some/model"))


# --- HF backend: lazy-import ImportError path (never imports torch) ---


def test_hf_summarizer_importerror_when_torch_missing(monkeypatch):
    # Force the lazy import to fail as it would on a torch-free login node, and assert the error
    # names the extras. This never actually imports torch/transformers.
    real_import_module = importlib.import_module

    def _fake_import_module(name, *args, **kwargs):
        if name in {"torch", "transformers"}:
            raise ImportError(f"No module named {name!r}")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)
    # Guard against a real ``import torch`` sneaking in via any path.
    real_import = builtins.__import__

    def _guard_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers"}:
            raise AssertionError(f"unexpected real import of {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guard_import)

    with pytest.raises(ImportError, match=r"\[hf,generate\]"):
        HFSummarizer("some/model", "deadbeef")
