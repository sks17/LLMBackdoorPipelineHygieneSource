"""Run the decision-free analysis layer on a survival-results file/dir into a report directory.

Thin runner over ``trigger_audit.analysis.report.build_report`` (mirrors ``pilot_report.py``). It
produces the counterfactual-control gate and the headline delivered-rate tables; the inferential
statistics and figures are gated on the decisions in ``docs/ANALYSIS_PLAN.md``. Example:

    python scripts/analyze_report.py outputs/pilot/survival.jsonl --out outputs/analysis/pilot

Optional joins (available once the run co-locates them): ``--manifest`` (authoritative pairing) and
``--bases`` + ``--policies-config`` (H4 / family covariates + policy labels). Exits non-zero if the
counterfactual control leaks (the analysis is then unsound).
"""

from __future__ import annotations

import sys

from trigger_audit.analysis.report import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
