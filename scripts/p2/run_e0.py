"""E0 instrument-validity tier (offline, reference extractor, CPU).

Local-only Project-2 driver (git-excluded). Runs the two E0 experiments that the restored
Jul-4 core already supports on the *simple* synthetic dataset:

  E0.1  determinism / recoverability GATE  -- per-layer AUROC > 0.9 AND byte-identical reruns.
        A pass proves only plumbing (the reference extractor makes token presence linearly
        recoverable by construction); a *failure* would be the informative outcome.
  E0.2  calibration honesty / FPR-resolution limit -- sweep the clean-negative count and show
        that target FPR 1e-3 is only *resolvable* (Wilson upper bound tight) once there are
        ~1000+ clean negatives; below that, achieved FPR saturates to 0 with a wide interval.

E0.3-E0.5 (leakage / operator-confound / three-population ablations) require the twins
generator (component D) and land once it is rebuilt.

Run:  .venv/Scripts/python.exe scripts/p2/run_e0.py
"""

from __future__ import annotations

import json
from pathlib import Path

from trigger_audit.experiments.probe_detection import (
    ProbeDetectionExperimentConfig,
    run_probe_experiment,
)
from trigger_audit.schemas.probes import PoolingStrategy

OUT = Path("outputs/probe_detection/E0")
OUT.mkdir(parents=True, exist_ok=True)

# base_id-grouped split is 0.5/0.25/0.25 and the simple generator is 50/50 labelled with one
# base_id per example, so clean calibration negatives ~= 0.125 * n_examples. Invert to size a run.
CALIB_NEG_PER_EXAMPLE = 0.25 * 0.5


def _cfg(experiment_id: str, n_examples: int) -> ProbeDetectionExperimentConfig:
    return ProbeDetectionExperimentConfig(
        experiment_id=experiment_id,
        extractor_backend="reference",
        extractor_num_layers=4,
        layers=[0, 1, 2, 3, 4],
        pooling=PoolingStrategy.MEAN,
        target_fprs=[1e-2, 1e-3],
        aggregation="mean_score",
        synthetic_n_examples=n_examples,
        synthetic_seq_len=16,
        synthetic_seed=0,
        split_seed=0,
        activations_dir=OUT / f"acts_{experiment_id}",
        results_out=OUT / f"{experiment_id}_result.jsonl",
    )


def run_e0_1() -> dict:
    """Determinism gate: two runs must be byte-identical and every layer AUROC > 0.9."""
    r1 = run_probe_experiment(_cfg("E0.1_determinism", n_examples=400))
    r2 = run_probe_experiment(_cfg("E0.1_determinism", n_examples=400))
    j1, j2 = r1.model_dump_json(), r2.model_dump_json()
    per_layer = [round(m.auroc, 4) for m in r1.layer_metrics_all]
    min_auroc = min(per_layer)
    deterministic = j1 == j2
    passed = deterministic and min_auroc > 0.9
    out = {
        "experiment": "E0.1_determinism_gate",
        "per_layer_auroc_all": per_layer,
        "min_layer_auroc": min_auroc,
        "aggregate_auroc_all": round(r1.aggregated_metrics_all.auroc, 4),
        "byte_identical_reruns": deterministic,
        "PASS": passed,
    }
    (OUT / "E0.1_summary.json").write_text(json.dumps(out, indent=2))
    return out


def run_e0_2() -> dict:
    """FPR-resolution sweep: clean-negative count vs achievable FPR (Wilson upper bound)."""
    rows = []
    for target_calib_neg in (30, 100, 300, 1000, 3000):
        n_examples = round(target_calib_neg / CALIB_NEG_PER_EXAMPLE)
        r = run_probe_experiment(_cfg(f"E0.2_n{target_calib_neg}", n_examples=n_examples))
        for a in r.achieved_fprs_clean:
            rows.append(
                {
                    "target_clean_calib_neg": target_calib_neg,
                    "n_examples": n_examples,
                    "target_fpr": a.target_fpr,
                    "achieved_fpr_clean": round(a.achieved_fpr, 5),
                    "wilson_low": round(a.ci_low, 5),
                    "wilson_high": round(a.ci_high, 5),
                    "n_test_clean_neg": a.n_negatives,
                    # "resolvable" ~ the honest upper bound is at or under the target budget.
                    "resolvable": a.ci_high <= a.target_fpr * 2,
                }
            )
    out = {"experiment": "E0.2_fpr_resolution_limit", "sweep": rows}
    (OUT / "E0.2_summary.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    e01 = run_e0_1()
    print("== E0.1 determinism/recoverability gate ==")
    print(f"  per-layer AUROC (all): {e01['per_layer_auroc_all']}")
    print(f"  byte-identical reruns: {e01['byte_identical_reruns']}")
    print(f"  PASS: {e01['PASS']}")
    print()
    e02 = run_e0_2()
    print("== E0.2 FPR-resolution limit (clean-negative sweep) ==")
    hdr = ("target_fpr", "calib_neg", "ach_fpr", "wilson_hi", "n_test_neg", "resolvable")
    print(f"{hdr[0]:>10} {hdr[1]:>10} {hdr[2]:>8} {hdr[3]:>10} {hdr[4]:>11} {hdr[5]:>10}")
    for row in e02["sweep"]:
        print(
            f"{row['target_fpr']:>10g} {row['target_clean_calib_neg']:>10} "
            f"{row['achieved_fpr_clean']:>8.4f} {row['wilson_high']:>10.4f} "
            f"{row['n_test_clean_neg']:>11} {row['resolvable']!s:>10}"
        )
    print()
    print(f"wrote summaries -> {OUT}/E0.1_summary.json, {OUT}/E0.2_summary.json")


if __name__ == "__main__":
    main()
