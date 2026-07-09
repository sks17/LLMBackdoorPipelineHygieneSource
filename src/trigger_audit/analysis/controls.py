"""Gate 0: the counterfactual control that every rate is conditioned on.

Every trigger-absent twin must classify ``no_survival`` with the trigger delivered nowhere. If any
leaks, the scoring is unsound and no rate is trustworthy -- the analysis must abort, not warn. This
gate has already earned its keep: the pilot's 312-twin leak exposed a real scorer bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ControlVerdict:
    """Result of the counterfactual control check."""

    n_absent: int
    n_leaks: int
    ok: bool
    leak_examples: list[dict[str, object]] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            return f"counterfactual control: {self.n_absent} absent twins, ALL clean (no_survival)"
        return f"counterfactual control: {self.n_leaks}/{self.n_absent} absent twins LEAKED"


def verify_counterfactual(df: pd.DataFrame, *, max_examples: int = 20) -> ControlVerdict:
    """Check that every ``trigger_present==False`` row is ``no_survival`` and delivered nowhere."""
    absent = df[~df["trigger_present"]]
    leaks = absent[(absent["survival_class"] != "no_survival") | (absent["delivered"])]
    cols = [
        c
        for c in ("trial_id", "base_id", "trigger_id", "survival_class", "delivered")
        if c in leaks
    ]
    examples = leaks.head(max_examples)[cols].to_dict("records")
    return ControlVerdict(
        n_absent=len(absent),
        n_leaks=len(leaks),
        ok=len(leaks) == 0,
        leak_examples=examples,
    )
