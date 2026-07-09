"""Frozen Trial 5 (boundary corruption) specification: a trigger cut through the middle.

Head truncation whose surviving window begins ``split_offset`` tokens *into* the trigger drops the
front half and leaves the back half as the literal prefix of the final input -- the project's first
``trigger_partial_survived=True``. The base is Trial Zero's single-turn conversation (system + one
user message), so the phenomenon is isolated to the token stage with no memory-policy interaction;
the trigger is the purpose-built long boundary canary. All three budgets are DERIVED from a measured
``policy="none"`` run (never hardcoded), the same discipline as Trials 1 and 3.
"""

from __future__ import annotations

from trigger_audit.experiments.survivability_audit import trial_zero_spec as tz
from trigger_audit.schemas.results import SurvivalResult

MODEL_ID = tz.MODEL_ID
TOKENIZER_ID = tz.TOKENIZER_ID
BASE_ID = "conv_000002"
TRIGGER_ID = "boundary_001"

# Non-binding budget for the "before" control: the truncation window begins before the trigger, so
# the whole trigger survives. The split/tight budgets are derived per-run from the measured span.
GENEROUS_BUDGET = tz.CONTEXT_LENGTH


def derive_split_budget(none_result: SurvivalResult) -> int:
    """Derive the split budget so head truncation cuts through the middle of the trigger.

    From the measured none-run span ``[S, E)`` and total ``T``: keep the last
    ``T - S - split_offset`` tokens, where ``split_offset = (E - S) // 2``. The surviving window
    then begins ``split_offset`` tokens into the trigger, dropping the front half and leaving the
    back half (``trigger_ids[split_offset:]``) as the exact prefix of the final input.
    """
    start = none_result.trigger_final_token_start
    end = none_result.trigger_final_token_end
    if start is None or end is None:
        raise ValueError("none run did not locate the trigger; cannot derive a split budget")
    split_offset = (end - start) // 2
    return none_result.final_prompt_token_count - start - split_offset


def derive_tight_budget(none_result: SurvivalResult) -> int:
    """Derive the tight control budget: the window begins after the trigger, dropping it whole.

    ``T - E`` keeps only the tokens after the trigger span, so nothing of the trigger survives --
    the negative control proving the partial predicate does not fire on full loss.
    """
    end = none_result.trigger_final_token_end
    if end is None:
        raise ValueError("none run did not locate the trigger; cannot derive a tight budget")
    return none_result.final_prompt_token_count - end
