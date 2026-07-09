"""E0 instrument confound ablations (offline, reference extractor, CPU).

Local-only Project-2 driver. Completes the E0 instrument tier by running the three
pre-registered confound ablations, each as an A/B toggle whose DEFAULT (off) reproduces the
current leakage-safe behavior and whose ON exposes the confound the experiment measures:

  E0.3  leakage-safety (base_id grouping)   -- grouped vs example-level split; example-level
        should INFLATE AUROC/TPR because counterfactual twins straddle the train/test line.
  E0.4  pooling operator-confound           -- TRIGGER_SPAN random-span fallback ON vs OFF on
        TRIGGER-FREE content; OFF should MANUFACTURE class separation from pooling statistics
        alone (a short-window mean vs a full-sequence mean).
  E0.5  three-population calibration        -- calibrate on clean-only vs clean+partial-survival
        negatives; the mixed pool should RAISE the threshold and LOWER delivered-only TPR.

Each ablation runs the same configuration twice (toggle off vs on) and reports the directional
effect, writing a JSON summary under outputs/probe_detection/E0/. The shipped twins generator
gives every example independent content, so it cannot exhibit the E0.3 leak (review AR-3) nor a
trigger-free operator confound; E0.3/E0.4 therefore use the purpose-built
``build_leakage_demo_dataset`` / ``build_operator_confound_dataset`` fixtures, while E0.5 runs
directly on the twins config (it needs the partial-survival population).

Run:  .venv/Scripts/python.exe scripts/p2/run_e0_ablations.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from trigger_audit.activations.extractor import make_activation_extractor
from trigger_audit.activations.store import ActivationStore
from trigger_audit.analysis.probe_loading import load_predictions
from trigger_audit.analysis.probe_stats import leakage_inflation
from trigger_audit.experiments.probe_detection import (
    ProbeDetectionExperimentConfig,
    ProbeDetectionRunner,
    assign_splits,
    assign_splits_example_level,
    build_leakage_demo_dataset,
    build_operator_confound_dataset,
    run_probe_experiment,
)
from trigger_audit.io.jsonl import write_jsonl
from trigger_audit.schemas.probes import PoolingStrategy, ProbeExample

OUT = Path("outputs/probe_detection/E0")
OUT.mkdir(parents=True, exist_ok=True)

HIDDEN_SIZE = 48
NUM_LAYERS = 4
LAYERS = [1, 2, 3, 4]
TARGET_FPRS = [1e-2, 1e-3]
TRAIN_FRACTION = 0.5
CALIBRATION_FRACTION = 0.25


def _cfg(experiment_id: str, **overrides: Any) -> ProbeDetectionExperimentConfig:
    """A reference-backend probe config with the E0 defaults, plus any per-run overrides."""
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "model_id": "reference-model",
        "extractor_backend": "reference",
        "extractor_hidden_size": HIDDEN_SIZE,
        "extractor_num_layers": NUM_LAYERS,
        "layers": LAYERS,
        "aggregation": "mean_score",
        "target_fprs": TARGET_FPRS,
        "train_fraction": TRAIN_FRACTION,
        "calibration_fraction": CALIBRATION_FRACTION,
        "split_seed": 0,
        "activations_dir": str(OUT / f"acts_{experiment_id}"),
        "results_out": str(OUT / f"{experiment_id}_result.jsonl"),
    }
    payload.update(overrides)
    return ProbeDetectionExperimentConfig.model_validate(payload)


def _reference_extractor() -> Any:
    return make_activation_extractor(
        "reference", seed=0, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS
    )


# --------------------------------------------------------------------------------------------
# E0.3 -- leakage-safety ablation (grouped vs example-level split)
# --------------------------------------------------------------------------------------------
def run_e0_3() -> dict[str, Any]:
    """Grouped vs example-level split on a content-sharing twin fixture: measure the inflation."""
    examples, tokens = build_leakage_demo_dataset(
        n_bases=60, examples_per_base=4, seq_len=10, noise_tokens=0, seed=0
    )
    extractor = _reference_extractor()

    def run(experiment_id: str, assigner: Any) -> Any:
        cfg = _cfg(experiment_id)
        runner = ProbeDetectionRunner(cfg, extractor=extractor, token_provider=tokens)
        result = runner.run(
            assigner(
                examples,
                train_fraction=TRAIN_FRACTION,
                calibration_fraction=CALIBRATION_FRACTION,
                seed=cfg.split_seed,
            )
        )
        preds_path = OUT / f"{experiment_id}_preds.jsonl"
        write_jsonl(preds_path, runner.test_predictions)
        return result, load_predictions(preds_path)

    grouped, grouped_preds = run("E0.3_grouped", assign_splits)
    example, example_preds = run("E0.3_example", assign_splits_example_level)
    inflation = leakage_inflation(grouped_preds, example_preds, 1e-2, n_boot=500, seed=0)

    grouped_auroc = grouped.aggregated_metrics_all.auroc
    example_auroc = example.aggregated_metrics_all.auroc
    out = {
        "experiment": "E0.3_leakage_safety",
        "toggle": "split_mode: grouped (off) vs example (on)",
        "auroc_grouped": round(grouped_auroc, 4),
        "auroc_example": round(example_auroc, 4),
        "auroc_inflation": round(example_auroc - grouped_auroc, 4),
        "tpr_inflation_at_1e2": round(inflation["tpr_inflation"], 4),
        "leakage_inflation": {k: round(v, 4) for k, v in inflation.items()},
        "expected": "example-level AUROC >= grouped (twin leakage inflates)",
        "direction_ok": example_auroc >= grouped_auroc,
    }
    (OUT / "E0.3_summary.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------------------------
# E0.4 -- pooling operator-confound ablation (random-span fallback ON vs OFF)
# --------------------------------------------------------------------------------------------
def _span_norm_separation(
    experiment_id: str,
    examples: list[ProbeExample],
    tokens: dict[str, list[int]],
    extractor: Any,
    *,
    span_random_fallback: bool,
) -> float:
    """Standardized gap in pooled-feature L2 norm between span-carrying and spanless examples.

    Runs the real TRIGGER_SPAN pooling path (honoring ``span_random_fallback``), reloads the
    pooled deepest-layer features the runner just wrote, and reports the effect size

        |mean||x||_span-carrying - mean||x||_spanless| / std(||x||).

    This is the scale-sensitive statistic the operator confound moves: a short-window mean has
    far higher variance (hence larger expected norm) than a full-sequence mean, so with the
    fallback OFF the spanless (full-mean) class collapses to a small norm and separates from the
    span-carrying (short-window) class -- while with the fallback ON both classes are
    short-window pooled and the gap vanishes. AUROC is deliberately NOT used here: it is a
    rank statistic and is blind to a symmetric variance-only difference on trigger-free content.
    """
    cfg = _cfg(
        experiment_id,
        pooling=PoolingStrategy.TRIGGER_SPAN,
        span_random_fallback=span_random_fallback,
    )
    runner = ProbeDetectionRunner(cfg, extractor=extractor, token_provider=tokens)
    runner.run(
        assign_splits(
            examples,
            train_fraction=TRAIN_FRACTION,
            calibration_fraction=CALIBRATION_FRACTION,
            seed=cfg.split_seed,
        )
    )
    store = ActivationStore(cfg.activations_dir)
    matrix, trial_ids = store.load(cfg.experiment_id, cfg.model_id, max(LAYERS), "trigger_span")
    carries_span = {e.trial_id: e.trigger_span() is not None for e in examples}
    span_mask = np.array([carries_span[t] for t in trial_ids], dtype=bool)
    norms = np.linalg.norm(matrix, axis=1)
    return float(abs(norms[span_mask].mean() - norms[~span_mask].mean()) / (norms.std() + 1e-12))


def run_e0_4() -> dict[str, Any]:
    """TRIGGER_SPAN fallback ON vs OFF on trigger-free content: OFF manufactures separation."""
    examples, tokens = build_operator_confound_dataset(
        n_examples=120, seq_len=24, span_len=4, seed=0
    )
    extractor = _reference_extractor()
    sep_on = _span_norm_separation(
        "E0.4_fallback_on", examples, tokens, extractor, span_random_fallback=True
    )
    sep_off = _span_norm_separation(
        "E0.4_fallback_off", examples, tokens, extractor, span_random_fallback=False
    )
    out = {
        "experiment": "E0.4_operator_confound",
        "toggle": "span_random_fallback: True (off/control) vs False (on/confound)",
        "content": "trigger-free (no trigger tokens anywhere)",
        "norm_separation_fallback_on": round(sep_on, 4),
        "norm_separation_fallback_off": round(sep_off, 4),
        "manufactured_separation": round(sep_off - sep_on, 4),
        "expected": "fallback-OFF separation > fallback-ON (operator alone separates classes)",
        "direction_ok": sep_off > sep_on,
    }
    (OUT / "E0.4_summary.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------------------------
# E0.5 -- three-population calibration ablation (clean-only vs clean+partial)
# --------------------------------------------------------------------------------------------
def run_e0_5() -> dict[str, Any]:
    """Clean-only vs clean+partial calibration on the twins config: threshold up, TPR down."""
    twins: dict[str, Any] = {
        "synthetic_mode": "twins",
        "synthetic_n_bases": 80,
        "partial_survival_fraction": 0.6,
        "synthetic_seq_len": 24,
        "synthetic_seed": 0,
    }
    clean = run_probe_experiment(
        _cfg("E0.5_clean_only", calibration_include_partial=False, **twins)
    )
    partial = run_probe_experiment(
        _cfg("E0.5_include_partial", calibration_include_partial=True, **twins)
    )

    per_fpr: list[dict[str, Any]] = []
    threshold_ok = True
    tpr_ok = True
    for target in TARGET_FPRS:
        key = str(target)
        clean_thr = clean.aggregated_metrics_delivered_only.threshold[key]
        partial_thr = partial.aggregated_metrics_delivered_only.threshold[key]
        clean_tpr = clean.aggregated_metrics_delivered_only.tpr_at_target_fpr[key]
        partial_tpr = partial.aggregated_metrics_delivered_only.tpr_at_target_fpr[key]
        threshold_ok = threshold_ok and partial_thr >= clean_thr
        tpr_ok = tpr_ok and partial_tpr <= clean_tpr
        per_fpr.append(
            {
                "target_fpr": target,
                "threshold_clean_only": round(clean_thr, 4),
                "threshold_include_partial": round(partial_thr, 4),
                "threshold_shift": round(partial_thr - clean_thr, 4),
                "delivered_tpr_clean_only": round(clean_tpr, 4),
                "delivered_tpr_include_partial": round(partial_tpr, 4),
                "delivered_tpr_drop": round(clean_tpr - partial_tpr, 4),
            }
        )
    out = {
        "experiment": "E0.5_three_population_calibration",
        "toggle": "calibration_include_partial: False (off) vs True (on)",
        "per_fpr": per_fpr,
        "expected": "include-partial threshold >= clean-only AND delivered TPR <= clean-only",
        "direction_ok": threshold_ok and tpr_ok,
    }
    (OUT / "E0.5_summary.json").write_text(json.dumps(out, indent=2))
    return out


def _check(flag: bool) -> str:
    return "OK" if flag else "FAIL"


def main() -> None:
    e03 = run_e0_3()
    print("== E0.3 leakage-safety (base_id grouping) ==")
    print(f"  {'split':>10} {'AUROC(all)':>12}")
    print(f"  {'grouped':>10} {e03['auroc_grouped']:>12}")
    print(f"  {'example':>10} {e03['auroc_example']:>12}")
    print(
        f"  AUROC inflation (example - grouped) = {e03['auroc_inflation']:+.4f}; "
        f"TPR inflation @1e-2 = {e03['tpr_inflation_at_1e2']:+.4f}"
    )
    print(f"  expected: {e03['expected']}  ->  [{_check(e03['direction_ok'])}]")
    print()

    e04 = run_e0_4()
    print("== E0.4 pooling operator-confound (trigger-free content) ==")
    print(f"  {'fallback':>10} {'norm_sep':>12}")
    print(f"  {'ON':>10} {e04['norm_separation_fallback_on']:>12}")
    print(f"  {'OFF':>10} {e04['norm_separation_fallback_off']:>12}")
    print(f"  manufactured separation (OFF - ON) = {e04['manufactured_separation']:+.4f}")
    print(f"  expected: {e04['expected']}  ->  [{_check(e04['direction_ok'])}]")
    print()

    e05 = run_e0_5()
    print("== E0.5 three-population calibration ==")
    hdr = ("target_fpr", "thr_clean", "thr_partial", "d_thr", "tpr_clean", "tpr_partial", "d_tpr")
    print("  " + " ".join(f"{h:>11}" for h in hdr))
    for row in e05["per_fpr"]:
        print(
            "  "
            + " ".join(
                f"{v:>11}"
                for v in (
                    f"{row['target_fpr']:g}",
                    row["threshold_clean_only"],
                    row["threshold_include_partial"],
                    f"{row['threshold_shift']:+}",
                    row["delivered_tpr_clean_only"],
                    row["delivered_tpr_include_partial"],
                    f"{row['delivered_tpr_drop']:+}",
                )
            )
        )
    print(f"  expected: {e05['expected']}  ->  [{_check(e05['direction_ok'])}]")
    print()

    all_ok = e03["direction_ok"] and e04["direction_ok"] and e05["direction_ok"]
    print(f"all ablations show the expected direction: [{_check(all_ok)}]")
    print(f"wrote summaries -> {OUT}/E0.3_summary.json, E0.4_summary.json, E0.5_summary.json")


if __name__ == "__main__":
    main()
