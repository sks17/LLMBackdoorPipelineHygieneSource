"""Analyze a pilot survival-results file into the correct, conditioned finding tables.

The built-in ``score-survival`` aggregates *every* manifest row, which pools the counterfactual
trigger-absent twins in with the real rows and so halves every rate (the ``none`` positive control
reads ~0.50 instead of ~1.00). This report does the two things the headline analysis needs:

1. **Verifies the counterfactual control** -- every trigger-absent twin must score ``no_survival``
   with the trigger delivered nowhere. If that fails, the scoring is unsound and no rate is
   trustworthy.
2. **Conditions on the real (trigger-present) rows** to produce the true survival-rate table by
   policy x position, and splits the delivered rate by ``data_source`` (the H4 synthetic-vs-real
   covariate) by joining each result to its base conversation.

This is the small, pilot-scoped slice of the analysis layer; folding the ``trigger_present``
conditioning into ``aggregate_survival`` itself is the natural next analysis-layer task. Run:

    python scripts/pilot_report.py outputs/pilot/survival.jsonl data/pilot/base_conversations.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from trigger_audit.experiments.survivability_audit.scorer import aggregate_survival
from trigger_audit.io.jsonl import read_jsonl_as
from trigger_audit.io.stores import BaseConversationStore
from trigger_audit.schemas.results import SurvivalResult


def _fmt_row(cells: list[str], widths: list[int]) -> str:
    """Left-justify identifier cells and right-justify the numeric ones into a fixed-width row."""
    out = [cells[0].ljust(widths[0]), cells[1].ljust(widths[1])]
    out += [c.rjust(w) for c, w in zip(cells[2:], widths[2:], strict=True)]
    return "  ".join(out)


def _print_table(title: str, rows: list[dict[str, object]]) -> None:
    """Render a policy x position rate table (conditioned rows only)."""
    header = ["policy", "position", "n", "exact", "token", "partial", "delivered"]
    widths = [22, 12, 5, 6, 6, 7, 9]
    print(f"\n{title}")
    print(_fmt_row(header, widths))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in rows:
        print(
            _fmt_row(
                [
                    str(r["pipeline_policy"]),
                    str(r["trigger_position"]),
                    str(r["n"]),
                    f"{float(r['exact_rate']):.2f}",
                    f"{float(r['token_rate']):.2f}",
                    f"{float(r['partial_rate']):.2f}",
                    f"{float(r['delivered_rate']):.2f}",
                ],
                widths,
            )
        )


def main(argv: list[str]) -> int:
    """Load the pilot results + bases, verify the control, and print the conditioned tables."""
    if len(argv) != 2:
        print(__doc__)
        return 2
    survival_path, bases_path = Path(argv[0]), Path(argv[1])
    results = read_jsonl_as(survival_path, SurvivalResult)
    base_source = {
        base_id: BaseConversationStore(bases_path).get(base_id).metadata.get("data_source")
        for base_id in BaseConversationStore(bases_path).ids()
    }

    present = [r for r in results if r.raw_trigger_present]
    absent = [r for r in results if not r.raw_trigger_present]

    # 1. Counterfactual control: every trigger-absent twin must deliver nothing.
    bad = [
        r
        for r in absent
        if r.survival_class.value != "no_survival" or r.final_token_trigger_present
    ]
    control_ok = not bad
    print(
        f"Counterfactual control: {len(absent)} trigger-absent twins, "
        f"{'ALL clean (no_survival, delivered nowhere)' if control_ok else f'{len(bad)} LEAKED'}"
    )
    print(f"Real (trigger-present) rows analyzed: {len(present)}")

    # 2. Conditioned survival table (real rows only) -- the true headline rates.
    _print_table(
        "Survival rates by policy x position (trigger-present rows only)",
        aggregate_survival(present),
    )

    # 3. H4 covariate: delivered rate by data_source per policy.
    by_src: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in present:
        src = base_source.get(r.base_id) or "unknown"
        by_src[(r.pipeline_policy, str(src))].append(int(r.final_token_trigger_present))
    print("\nDelivered rate by policy x data_source (H4 content covariate)")
    print(_fmt_row(["policy", "data_source", "n", "deliv", "", "", ""], [22, 12, 5, 6, 6, 7, 9]))
    print("-" * 60)
    for (policy, src), flags in sorted(by_src.items()):
        rate = sum(flags) / len(flags)
        print(
            _fmt_row(
                [policy, src, str(len(flags)), f"{rate:.2f}", "", "", ""], [22, 12, 5, 6, 6, 7, 9]
            )
        )

    summary = {
        "control_ok": control_ok,
        "n_present": len(present),
        "n_absent": len(absent),
        "policy_position": aggregate_survival(present),
    }
    out = survival_path.with_name("pilot_report.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote machine-readable summary -> {out}")
    return 0 if control_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
