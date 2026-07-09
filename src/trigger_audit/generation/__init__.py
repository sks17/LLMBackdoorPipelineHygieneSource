"""Synthetic base-conversation generation: deterministic seeds -> harmless slot-form bases.

The generator is the peer of :mod:`trigger_audit.io.dataset_adapter`: a
:class:`~trigger_audit.generation.conversation_generator.GenerationBackend` produces raw role-tagged
content for a deterministic seed, and the shared ``to_base_conversation`` handles length binning and
slot planting so synthetic and real bases are inserted and scored identically. The public surface
lives in :mod:`trigger_audit.generation.conversation_generator`; import it directly (it is also the
``python -m`` CLI entry point, so this package init deliberately imports nothing from it).
"""
