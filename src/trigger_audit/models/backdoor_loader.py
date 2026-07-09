"""Safe loader for a backdoored checkpoint: allowlist gate FIRST, then a lazy torch load.

The loader is deliberately **not** a second activation extractor. Its job is the safety gate +
provenance + the benign-marker generation path used by ASR verification. The probe wave keeps
using ``make_activation_extractor("hf", ...)``; :func:`extractor_spec_for` yields the exact
``(model_id, revision, trust_remote_code, device)`` tuple that extractor needs, so a registered
checkpoint flows into the probe config unchanged.

``torch`` / ``transformers`` / ``peft`` are imported lazily *inside* methods (mirroring
``activations.extractor.HFActivationExtractor``) so ``import trigger_audit.models`` stays torch-free
on a CPU-only login node.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from trigger_audit.models.backdoor_registry import BackdoorCheckpoint, BackdoorRegistry


@dataclass(frozen=True)
class ExtractorSpec:
    """The exact arguments ``make_activation_extractor("hf", ...)`` needs for a checkpoint.

    ``requires_adapter_merge`` flags a LoRA checkpoint: the plain HF extractor loads a single
    standalone model, so a registered ``adapter_path`` must be merged into a merged checkpoint
    first. ``adapter_path`` is carried through so the merge step knows what to fold in.
    """

    model_id: str
    revision: str | None
    trust_remote_code: bool
    device: str
    requires_adapter_merge: bool = False
    adapter_path: str | None = None

    def as_tuple(self) -> tuple[str, str | None, bool, str]:
        """The ``(model_id, revision, trust_remote_code, device)`` tuple the extractor consumes."""
        return (self.model_id, self.revision, self.trust_remote_code, self.device)


def extractor_spec_for(
    checkpoint: BackdoorCheckpoint,
    *,
    trust_remote_code: bool = False,
    device: str = "cpu",
) -> ExtractorSpec:
    """Build the probe-extractor spec for a registered checkpoint.

    ``trust_remote_code`` defaults to ``False`` (safe default: never execute checkpoint-shipped
    code implicitly). A LoRA ``adapter_path`` sets ``requires_adapter_merge`` so the probe wave
    merges it before handing the base id to the plain extractor.
    """
    return ExtractorSpec(
        model_id=checkpoint.base_model_id,
        revision=checkpoint.revision,
        trust_remote_code=trust_remote_code,
        device=device,
        requires_adapter_merge=checkpoint.adapter_path is not None,
        adapter_path=checkpoint.adapter_path,
    )


def _lazy_import_hf() -> tuple[Any, Any]:
    """Import torch + transformers lazily, with a clear install hint if the stack is absent."""
    try:
        torch: Any = importlib.import_module("torch")
        transformers: Any = importlib.import_module("transformers")
    except ImportError as exc:
        raise ImportError(
            "SafeBackdoorModel requires torch and transformers. Install the model execution "
            "stack: `pip install 'trigger-audit[hf,generate]'` plus a torch build matched to your "
            "target (CPU wheel or the cluster's CUDA); see docs/DEVELOPMENT_SETUP.md."
        ) from exc
    return torch, transformers


class SafeBackdoorModel:
    """A loaded backdoored checkpoint wrapped behind the allowlist + provenance safety gate.

    The constructor calls ``registry.require_allowlisted(checkpoint.checkpoint_id)`` **first** —
    before importing torch or reading any weight file — so an unregistered or non-allowlisted
    checkpoint is refused (``ValueError`` / ``PermissionError``) with nothing loaded. It then
    lazily loads the HF causal LM (and an optional PEFT/LoRA adapter), puts it in ``.eval()``,
    and records provenance onto ``.provenance`` for result rows.
    """

    def __init__(
        self,
        checkpoint: BackdoorCheckpoint,
        registry: BackdoorRegistry,
        *,
        local_files_only: bool = True,
        device: str = "cpu",
    ) -> None:
        # Safety gate FIRST: refuse before any import or file read.
        registry.require_allowlisted(checkpoint.checkpoint_id)

        self._checkpoint = checkpoint
        self._device = device
        self._local_files_only = local_files_only

        torch, transformers = _lazy_import_hf()
        self._torch = torch

        # trust_remote_code is pinned False: never implicitly execute checkpoint-shipped code.
        model = transformers.AutoModelForCausalLM.from_pretrained(
            checkpoint.base_model_id,
            revision=checkpoint.revision,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            checkpoint.base_model_id,
            revision=checkpoint.revision,
            trust_remote_code=False,
            local_files_only=local_files_only,
        )
        adapter_loaded = False
        if checkpoint.adapter_path is not None:
            peft: Any = importlib.import_module("peft")
            model = peft.PeftModel.from_pretrained(
                model, checkpoint.adapter_path, local_files_only=local_files_only
            )
            adapter_loaded = True

        model.eval()
        model.to(device)
        self._model = model
        self._tokenizer = tokenizer

        self.provenance: dict[str, Any] = {
            "checkpoint_id": checkpoint.checkpoint_id,
            "base_model_id": checkpoint.base_model_id,
            "revision": checkpoint.revision,
            "adapter_path": checkpoint.adapter_path,
            "adapter_loaded": adapter_loaded,
            "commit": checkpoint.commit,
            "license": checkpoint.license,
            "attack_family": checkpoint.attack_family,
            "local_files_only": local_files_only,
            "device": device,
            "num_hidden_layers": int(model.config.num_hidden_layers),
            "hidden_size": int(model.config.hidden_size),
        }

    @property
    def checkpoint(self) -> BackdoorCheckpoint:
        return self._checkpoint

    @property
    def model(self) -> Any:
        return self._model

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def device(self) -> str:
        return self._device

    def extractor_spec(self, *, trust_remote_code: bool = False) -> ExtractorSpec:
        """The probe-extractor spec for this checkpoint (see :func:`extractor_spec_for`)."""
        return extractor_spec_for(
            self._checkpoint, trust_remote_code=trust_remote_code, device=self._device
        )

    def generate_canary(
        self, prompt: str, *, max_new_tokens: int = 64, deterministic: bool = True
    ) -> str:
        """Greedily decode ``prompt`` and return only the newly generated text.

        Used solely to check whether the benign ``canary_marker`` appears. ``deterministic=True``
        (default) means greedy ``do_sample=False`` decoding so the install check is reproducible.
        """
        torch = self._torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=not deterministic,
                num_beams=1,
            )
        generated = output_ids[0][prompt_len:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)
