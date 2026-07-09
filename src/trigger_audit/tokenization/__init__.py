"""Tokenization: adapters over real/reference tokenizers and token-subsequence search."""

from trigger_audit.tokenization.token_search import (
    contains_subsequence,
    find_subsequence,
    longest_common_run,
)
from trigger_audit.tokenization.tokenizer_adapter import (
    HFTokenizerAdapter,
    SimpleWhitespaceTokenizerAdapter,
    TokenizerAdapter,
    make_tokenizer_adapter,
)

__all__ = [
    "HFTokenizerAdapter",
    "SimpleWhitespaceTokenizerAdapter",
    "TokenizerAdapter",
    "contains_subsequence",
    "find_subsequence",
    "longest_common_run",
    "make_tokenizer_adapter",
]
