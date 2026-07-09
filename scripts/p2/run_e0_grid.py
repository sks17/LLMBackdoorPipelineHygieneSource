"""E0 instrument grid: expand the E0 axes and run every cell offline, then summarize.

Exercises the full harness path (grid -> per-cell ProbeDetectionExperimentConfig ->
reference-backend twins dataset -> probe train/calibrate/eval) that the GPU tiers reuse.
Reference-backend only, so it runs on CPU with no downloads. See docs/PROJECT2_FANOUT_WORKFLOW.md.

Run:  .venv/Scripts/python.exe scripts/p2/run_e0_grid.py
"""

from __future__ import annotations

import json
from pathlib import Path

from trigger_audit.config import load_config
from trigger_audit.experiments.probe_detection import (
    ProbeGridAxes,
    expand_probe_grid,
    run_probe_experiment,
)

OUT = Path("outputs/probe_detection/E0")
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    axes = load_config("configs/probe/E0_instrument.axes.yaml", ProbeGridAxes)
    cells = expand_probe_grid(axes)
    print(f"E0 instrument grid: {len(cells)} cells (reference backend, offline)")

    rows = []
    for cell in cells:
        result = run_probe_experiment(cell)
        fpr_1e2 = next(
            (a.achieved_fpr for a in result.achieved_fprs_clean if a.target_fpr == 1e-2),
            None,
        )
        rows.append(
            {
                "experiment_id": result.experiment_id,
                "pooling": str(cell.pooling).split(".")[-1].lower(),
                "aggregation": cell.aggregation,
                "auroc_all": round(result.aggregated_metrics_all.auroc, 3),
                "auroc_delivered": round(result.aggregated_metrics_delivered_only.auroc, 3),
                "fpr_at_1e2": None if fpr_1e2 is None else round(fpr_1e2, 4),
            }
        )

    (OUT / "E0_grid_summary.json").write_text(json.dumps(rows, indent=2))
    print(
        f"{'aggregation':>18} {'pooling':>10} {'auroc_all':>10} {'auroc_deliv':>12} {'fpr@1e-2':>9}"
    )
    for x in rows:
        print(
            f"{x['aggregation']:>18} {x['pooling']:>10} {x['auroc_all']:>10} "
            f"{x['auroc_delivered']:>12} {x['fpr_at_1e2']!s:>9}"
        )
    print(f"wrote -> {OUT}/E0_grid_summary.json")


if __name__ == "__main__":
    main()
