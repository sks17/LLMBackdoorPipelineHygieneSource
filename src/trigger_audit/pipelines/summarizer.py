"""Real summarizers for the compression memory policy (the producer seam of Task 10).

:class:`~trigger_audit.pipelines.memory_policy.SummarizeOldMessages` accepts an injectable
``Summarizer = Callable[[Sequence[ChatMessage]], str]``; without one it emits a harmless
exact-delete placeholder. This module fills that seam so *semantic* survival becomes
observable: a real summarizer can re-express an old turn's meaning as paraphrase, which the
token-level survival flags cannot see but a natural-language backdoor still delivers.

Two backends sit behind one callable interface, mirroring the tokenizer/extractor/scorer twin
pattern: a deterministic, dependency-free :class:`ReferenceSummarizer` for offline tests (it
paraphrases, copies, or drops content from a fixed table, never a real model) and a pinned,
lazy-HF :class:`HFSummarizer` for real measurement. Both are pinned/greedy so a summarize cell
is reproducible, the deliberate model-dependence this one experiment introduces at the producer.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel

from trigger_audit.pipelines.memory_policy import Summarizer
from trigger_audit.schemas.messages import ChatMessage

# Reference-double paraphrase table: each surface word maps to a DIFFERENT surface word that the
# reference semantic scorer's synonym table canonicalizes to the SAME lemma. Substituting these
# re-expresses a trigger's meaning (the semantic path still detects it) while breaking the verbatim
# and token-subsequence match (exact/token both fail). Hand-seeded for constructed fixtures only --
# this is NOT translation and never touches a measurement run.
_PARAPHRASE_TABLE: dict[str, str] = {
    "transfer": "move",
    "wire": "move",
    "send": "move",
    "move": "transfer",
    "funds": "cash",
    "fund": "cash",
    "money": "cash",
    "cash": "money",
    "external": "offshore",
    "offshore": "external",
    "outside": "external",
    "delete": "erase",
    "remove": "erase",
    "erase": "delete",
    "wipe": "delete",
    "immediately": "instantly",
    "instantly": "immediately",
    "now": "immediately",
}

# The generic, content-free summary the drop mode emits: it names no entity from the old turns, so
# it carries no trigger meaning and the semantic scorer stays silent (the compressed-exact-deleted
# outcome under a real summarizer that simply does not mention the trigger's content).
_DROP_SUMMARY = (
    "The earlier conversation covered general assistant support without specific details."
)

# Fixed lead-in for the verbatim/paraphrase modes so the summary reads as a summary rather than a
# raw transcript, without disturbing the (copied or paraphrased) trigger-bearing span.
_SUMMARY_PREFIX = "Summary of earlier turns:"


class SummarizerConfig(BaseModel):
    """Pinned configuration selecting and parameterizing a summarizer backend.

    ``backend='reference'`` reads ``mode`` (the offline double's copy/paraphrase/drop behavior);
    ``backend='hf'`` requires a pinned ``model_id`` and ``revision`` (no network default -- a
    summarize cell must name its exact producer) plus the greedy-generation pins. Kept small and
    typed: unset HF fields are validated at construction time by :func:`make_summarizer`.
    """

    backend: Literal["reference", "hf"] = "reference"
    # Reference-double behavior; ignored by the HF backend.
    mode: Literal["verbatim", "paraphrase", "drop"] = "paraphrase"
    # HF backend pins (required for a real run; validated in the factory, not silently defaulted).
    model_id: str | None = None
    revision: str | None = None
    max_new_tokens: int = 256
    device: str = "cpu"
    trust_remote_code: bool = False
    enable_thinking: bool = False
    prompt: str = "Summarize the earlier conversation in one or two sentences."


class ReferenceSummarizer:
    """A deterministic, dependency-free summarizer stand-in (stdlib only) with three modes.

    This is NOT a real summarizer and is never used for measurement; it exists so the whole
    semantic-survival producer path is runnable and unit-testable without ``torch``/``transformers``
    and without network access (the producer twin of
    :class:`~trigger_audit.tokenization.tokenizer_adapter.SimpleWhitespaceTokenizerAdapter` and
    :class:`~trigger_audit.scoring.semantic.ReferenceSemanticScorer`). The mode makes each cell of
    the acceptance table reachable offline and is fully deterministic given the input messages:

    - ``verbatim`` copies the old turns' content unchanged, so a trigger survives *exactly*;
    - ``paraphrase`` re-expresses the content via a fixed synonym table that preserves meaning but
      changes surface words, so a trigger survives as *meaning* only (exact and token both fail);
    - ``drop`` emits a fixed content-free summary, so a trigger survives in no form at all.
    """

    def __init__(self, *, mode: Literal["verbatim", "paraphrase", "drop"] = "paraphrase") -> None:
        self._mode = mode

    @property
    def mode(self) -> str:
        """The active reference mode (``verbatim`` / ``paraphrase`` / ``drop``)."""
        return self._mode

    def __call__(self, messages: Sequence[ChatMessage]) -> str:
        """Compress ``messages`` into a summary string per the configured mode (deterministic)."""
        if self._mode == "drop":
            return _DROP_SUMMARY
        body = "\n".join(message.content for message in messages)
        if self._mode == "verbatim":
            return f"{_SUMMARY_PREFIX}\n{body}"
        # paraphrase: re-express every word through the fixed table, preserving punctuation,
        # newlines, and out-of-table words so segmentation and non-trigger content are unchanged.
        # The prefix sits on its own line so a paraphrased span segments into its own window.
        return f"{_SUMMARY_PREFIX}\n{_paraphrase(body)}"


class HFSummarizer:
    """Wraps a pinned Hugging Face causal LM to compress old turns into a summary (production path).

    ``torch`` and ``transformers`` are imported lazily inside ``__init__`` (matching
    :class:`~trigger_audit.tokenization.tokenizer_adapter.HFTokenizerAdapter`) so the base package
    stays torch-free on CPU-only login nodes; a missing stack raises a clear ImportError naming the
    ``[hf]`` + ``[generate]`` extras. Generation is pinned and greedy -- ``do_sample=False`` (no
    temperature/top-p sampling), ``enable_thinking=False`` where the template supports it, a fixed
    ``model_id`` + ``revision`` (commit SHA), CPU float32 -- so a summarize cell is reproducible,
    the deliberate producer-side model dependence this experiment accepts.
    """

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        max_new_tokens: int = 256,
        device: str = "cpu",
        trust_remote_code: bool = False,
        enable_thinking: bool = False,
        prompt: str = "Summarize the earlier conversation in one or two sentences.",
    ) -> None:
        try:
            # Lazy, importlib-based imports: torch is intentionally absent from the base venv
            # (see the pyproject `generate` extra rationale), so a static `import torch` here
            # would break both mypy and the CPU-only install.
            torch: Any = importlib.import_module("torch")
            transformers: Any = importlib.import_module("transformers")
        except ImportError as exc:
            raise ImportError(
                "HFSummarizer requires torch and transformers. Install the model execution "
                "stack: `pip install 'trigger-audit[hf,generate]'` plus a torch build matched to "
                "your target (CPU wheel or the cluster's CUDA); see docs/DEVELOPMENT_SETUP.md."
            ) from exc

        self._torch = torch
        self._model_id = model_id
        self._revision = revision
        self._max_new_tokens = max_new_tokens
        self._device = device
        self._enable_thinking = enable_thinking
        self._prompt = prompt
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        self._model.eval()
        self._model.to(device)

    @property
    def summarizer_id(self) -> str:
        """The pinned model id (records the producer identity onto a summarize cell)."""
        return self._model_id

    @property
    def summarizer_revision(self) -> str:
        """The pinned revision (commit SHA/tag) of the producer model."""
        return self._revision

    def __call__(self, messages: Sequence[ChatMessage]) -> str:
        """Greedily summarize ``messages`` and return the decoded summary text (no sampling)."""
        torch = self._torch
        transcript = "\n".join(f"{m.role.value}: {m.content}" for m in messages)
        chat = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": transcript},
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self._enable_thinking,
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        # Decode only the newly generated continuation, not the echoed prompt.
        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        return str(self._tokenizer.decode(new_tokens, skip_special_tokens=True)).strip()


def make_summarizer(config: SummarizerConfig) -> Summarizer:
    """Construct a ``Summarizer`` callable for the given config (the injected producer).

    ``backend='reference'`` returns the dependency-free double in the configured mode (offline
    tests and smoke runs); ``backend='hf'`` loads a pinned model and requires both ``model_id`` and
    ``revision`` (a summarize cell must name its exact producer). An unknown backend, or an HF
    config missing its pin, raises a clear ``ValueError`` rather than silently mis-summarizing.
    """
    if config.backend == "reference":
        return ReferenceSummarizer(mode=config.mode)
    if config.backend == "hf":
        if not config.model_id or not config.revision:
            raise ValueError(
                "SummarizerConfig(backend='hf') requires both model_id and revision "
                "(a summarize cell must pin its exact producer; there is no network default)."
            )
        return HFSummarizer(
            config.model_id,
            config.revision,
            max_new_tokens=config.max_new_tokens,
            device=config.device,
            trust_remote_code=config.trust_remote_code,
            enable_thinking=config.enable_thinking,
            prompt=config.prompt,
        )
    raise ValueError(f"Unknown summarizer backend: {config.backend!r}")


def _paraphrase(text: str) -> str:
    """Re-express ``text`` word-by-word via the fixed paraphrase table (punctuation preserved)."""

    def _substitute(match: re.Match[str]) -> str:
        word = match.group(0)
        return _PARAPHRASE_TABLE.get(word.lower(), word)

    return re.sub(r"[A-Za-z]+", _substitute, text)
