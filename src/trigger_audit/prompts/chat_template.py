"""Chat templating: render structured messages into model-specific prompt text (Layer 3)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from trigger_audit.schemas.messages import ChatMessage
from trigger_audit.tokenization.tokenizer_adapter import TokenizerAdapter

ChatFormat = Literal["chat", "base"]


def render_base_completion(
    messages: Sequence[ChatMessage],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render messages as a deterministic base-completion prompt (Layer 3), no chat/special tokens.

    Base language models (e.g. Pythia-1B) have no chat template, so ``apply_chat_template`` is
    unavailable and would either fail or fall back to a silent default. This renders Layer 3 as a
    plain concatenation of ``"{role}: {content}\n"`` per message, appending a bare ``"assistant:"``
    turn cue when ``add_generation_prompt`` is set. No special or control tokens are introduced, so
    the trigger text appears verbatim and the offset-based localization/scoring path is unchanged.
    The exact format is documented in ``docs/DATA_CONTRACTS.md``.
    """
    parts = [f"{m.role.value}: {m.content}\n" for m in messages]
    if add_generation_prompt:
        parts.append("assistant:")
    return "".join(parts)


class TemplateRenderError(RuntimeError):
    """Raised when a model's chat template rejects a message sequence outright.

    This is a *delivery* failure, not an unrelated bug: some templates (e.g. Gemma's strict
    user/assistant alternation) refuse to render certain sequences a template-agnostic memory
    policy can produce. Carrying both the message and the offending ``messages`` lets a runner
    record a ``TEMPLATE_INCOMPATIBLE`` outcome and attribute presence flags to the pre-template
    layers, rather than crashing.
    """

    def __init__(self, message: str, *, messages: Sequence[ChatMessage] | None = None) -> None:
        super().__init__(message)
        self.messages: list[ChatMessage] = list(messages or [])


def _is_template_render_error(exc: Exception) -> bool:
    """True if ``exc`` is a Jinja2 chat-template error, matched by class name (no jinja2 import).

    ``transformers.apply_chat_template`` raises ``jinja2.exceptions.TemplateError`` (and subclasses)
    when a template's ``raise_exception`` fires. Matching the name keeps this module free of the
    heavy ``jinja2``/``transformers`` dependency, which lives behind the ``hf`` extra.
    """
    return any(cls.__name__ == "TemplateError" for cls in type(exc).__mro__)


class ChatTemplateRenderer:
    """Renders role/content messages into the model-specific text the tokenizer will consume.

    Templating is its own layer because the same logical messages can become different final
    text (and token layouts) depending on the model's chat template, which is exactly why the
    final prompt must be logged rather than inferred from the raw messages.
    """

    def __init__(
        self,
        adapter: TokenizerAdapter,
        *,
        enable_thinking: bool,
        add_generation_prompt: bool = True,
        chat_template: str | None = None,
        chat_format: ChatFormat = "chat",
    ) -> None:
        self._adapter = adapter
        self._enable_thinking = enable_thinking
        self._add_generation_prompt = add_generation_prompt
        self._chat_template = chat_template
        self._chat_format = chat_format

    def render(self, messages: Sequence[ChatMessage]) -> str:
        """Render messages to prompt text using the configured template settings.

        With ``chat_format="base"`` the deterministic base-completion path is used (no chat template
        is consulted); otherwise the model's chat template renders the sequence. Raises
        :class:`TemplateRenderError` (carrying ``messages``) when the model's chat template rejects
        the sequence, so a caller can record a delivery failure instead of crashing; unrelated
        exceptions propagate unchanged.
        """
        if self._chat_format == "base":
            return render_base_completion(
                messages, add_generation_prompt=self._add_generation_prompt
            )
        try:
            return self._adapter.render_chat(
                messages,
                add_generation_prompt=self._add_generation_prompt,
                enable_thinking=self._enable_thinking,
                chat_template=self._chat_template,
            )
        except Exception as exc:
            if _is_template_render_error(exc):
                raise TemplateRenderError(str(exc), messages=messages) from exc
            raise
