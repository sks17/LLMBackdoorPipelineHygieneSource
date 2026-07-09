"""trigger_audit: a prompt-survivability / trigger-delivery audit harness.

The package answers one question for every trial: when a harmless canary trigger
is placed into raw user input, does it survive the real prompt pipeline (chat
templating, trimming, summarization, packing, tokenization) and reach the final
model-visible input?

Only lightweight schema imports are re-exported here so that ``import trigger_audit``
stays cheap and free of heavy optional dependencies (transformers, torch, faiss).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("trigger-audit")
except PackageNotFoundError:  # running from a source checkout without an install
    __version__ = "0.1.0"

__all__ = ["__version__"]
