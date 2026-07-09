"""Tokenizer adapters: a common interface over the production HF tokenizer and a
dependency-free reference tokenizer for offline tests and CPU-only smoke runs."""

from __future__ import annotations

import zlib
from abc import ABC, abstractmethod
from collections.abc import Sequence

from trigger_audit.schemas.messages import ChatMessage
from trigger_audit.tokenization.token_search import find_subsequence


class TokenizerAdapter(ABC):
    """Common interface for encoding text, decoding ids, and rendering chat messages.

    All four logged layers flow through this interface, so swapping the reference tokenizer
    for a real Hugging Face tokenizer changes only which adapter the runner is given.
    """

    @property
    @abstractmethod
    def tokenizer_id(self) -> str:
        """A stable identifier for the tokenizer (model id or a fixed reference name)."""

    @abstractmethod
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode text into token ids."""

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Decode token ids back into text."""

    @abstractmethod
    def render_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool,
        chat_template: str | None = None,
    ) -> str:
        """Render role/content messages into model-specific prompt text (Layer 3).

        ``enable_thinking`` is required (no default) on purpose: for models with a thinking mode
        (for example Qwen3) the tokenizer's own default is not ours, and a silent default is a
        determinism hazard. Every caller must state the decision explicitly.
        """

    def count_tokens(self, text: str, *, add_special_tokens: bool = False) -> int:
        """Return the number of tokens in text (default: length of the encoding)."""
        return len(self.encode(text, add_special_tokens=add_special_tokens))

    def locate_token_span(self, text: str, subtext: str) -> tuple[int, int] | None:
        """Return the half-open token span of ``subtext`` within ``text``, or None if absent.

        Default implementation: a token-id subsequence search over the two encodings. Adapters
        whose tokenizer re-tokenizes across context boundaries -- so a trigger's standalone token
        ids are not a contiguous subsequence of the templated ids -- override this with a
        character-offset lookup (see :class:`HFTokenizerAdapter`). Keeping the span the single
        source of truth for token localization makes trigger scoring tokenizer-agnostic.
        """
        if not subtext:
            return (0, 0)
        return find_subsequence(
            self.encode(text, add_special_tokens=False),
            self.encode(subtext, add_special_tokens=False),
        )


class SimpleWhitespaceTokenizerAdapter(TokenizerAdapter):
    """A deterministic, dependency-free reference tokenizer (whitespace split, CRC32 ids).

    This is NOT a real BPE tokenizer; it exists so the full pipeline is runnable and unit
    testable without ``transformers`` and without network access. Token ids are stable across
    processes (a given word always maps to the same id), which is what trigger-subsequence
    matching needs. ``decode`` is best-effort and reconstructs only tokens this instance has
    seen via ``encode``. Use :class:`HFTokenizerAdapter` for any real measurement.
    """

    def __init__(self, tokenizer_id: str = "simple-whitespace", *, vocab_bits: int = 20) -> None:
        self._id = tokenizer_id
        self._mask = (1 << vocab_bits) - 1
        self._inverse: dict[int, str] = {}

    @property
    def tokenizer_id(self) -> str:
        return self._id

    def _token_id(self, token: str) -> int:
        # Reserve 0 as a sentinel; offset all real ids by 1.
        token_id = (zlib.crc32(token.encode("utf-8")) & self._mask) + 1
        self._inverse.setdefault(token_id, token)
        return token_id

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [self._token_id(tok) for tok in text.split()]

    def decode(self, ids: Sequence[int]) -> str:
        return " ".join(self._inverse.get(i, "<unk>") for i in ids)

    def render_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool,
        chat_template: str | None = None,
    ) -> str:
        # The reference tokenizer has no thinking mode; `enable_thinking` is accepted for
        # interface parity (and to force callers to decide) but does not change the output.
        _ = enable_thinking
        parts = [f"<|{m.role.value}|>\n{m.content}\n" for m in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "".join(parts)


class HFTokenizerAdapter(TokenizerAdapter):
    """Wraps a Hugging Face ``AutoTokenizer``.

    ``transformers`` is imported lazily inside ``__init__`` so the package (and the
    pipeline-only audit path) works without the ``hf`` extra installed.
    """

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        trust_remote_code: bool = False,
        chat_template: str | None = None,
    ) -> None:
        from transformers import AutoTokenizer  # lazy: only needed for real tokenization

        self._id = model_id
        self._chat_template = chat_template
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )

    @property
    def tokenizer_id(self) -> str:
        return self._id

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: Sequence[int]) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=False)

    def locate_token_span(self, text: str, subtext: str) -> tuple[int, int] | None:
        """Locate ``subtext``'s token span via character offsets (robust to re-tokenization).

        Encodes ``text`` with ``return_offsets_mapping`` and returns the half-open span of tokens
        whose character offsets overlap ``subtext``'s character span. This localizes a trigger
        correctly even when its standalone token ids are not a contiguous subsequence of the
        templated ids (a real BPE behavior for some tokenizers, e.g. TinyLlama). Falls back to the
        subsequence search for a non-fast tokenizer, which exposes no offsets.
        """
        if not subtext:
            return (0, 0)
        if not self._tokenizer.is_fast:
            return super().locate_token_span(text, subtext)
        char_start = text.find(subtext)
        if char_start < 0:
            return None
        char_end = char_start + len(subtext)
        encoding = self._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        # Token indices whose char offsets have a non-empty overlap with the trigger's char span;
        # the max<min form skips zero-width (special-token) offsets, which never overlap real text.
        overlapping = [
            i
            for i, (span_start, span_end) in enumerate(encoding["offset_mapping"])
            if max(span_start, char_start) < min(span_end, char_end)
        ]
        return (overlapping[0], overlapping[-1] + 1) if overlapping else None

    def render_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool,
        chat_template: str | None = None,
    ) -> str:
        payload = [{"role": m.role.value, "content": m.content} for m in messages]
        kwargs: dict[str, object] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        }
        template = chat_template or self._chat_template
        if template is not None:
            kwargs["chat_template"] = template
        return self._tokenizer.apply_chat_template(payload, **kwargs)


def make_tokenizer_adapter(
    model_id: str,
    *,
    backend: str = "hf",
    revision: str | None = None,
    trust_remote_code: bool = False,
    chat_template: str | None = None,
) -> TokenizerAdapter:
    """Construct a tokenizer adapter for the given backend.

    ``backend='simple'`` returns the dependency-free reference tokenizer; ``backend='hf'``
    loads a real Hugging Face tokenizer (requires the ``hf`` extra).
    """
    if backend == "simple":
        return SimpleWhitespaceTokenizerAdapter(model_id)
    if backend == "hf":
        return HFTokenizerAdapter(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
            chat_template=chat_template,
        )
    raise ValueError(f"Unknown tokenizer backend: {backend!r}")
