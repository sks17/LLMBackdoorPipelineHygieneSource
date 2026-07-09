"""I/O helpers: JSONL read/write, path resolution, and id-keyed stores."""

from trigger_audit.io.final_tokens import FinalTokensRow, read_final_tokens, write_final_tokens
from trigger_audit.io.jsonl import (
    append_jsonl,
    iter_jsonl,
    read_jsonl,
    read_jsonl_as,
    write_jsonl,
)
from trigger_audit.io.manifest import expand_manifest, pair_key
from trigger_audit.io.paths import PathResolver, ensure_dir
from trigger_audit.io.stores import (
    BaseConversationStore,
    IndexedJsonlStore,
    TriggerStore,
)

__all__ = [
    "BaseConversationStore",
    "FinalTokensRow",
    "IndexedJsonlStore",
    "PathResolver",
    "TriggerStore",
    "append_jsonl",
    "ensure_dir",
    "expand_manifest",
    "iter_jsonl",
    "pair_key",
    "read_final_tokens",
    "read_jsonl",
    "read_jsonl_as",
    "write_final_tokens",
    "write_jsonl",
]
