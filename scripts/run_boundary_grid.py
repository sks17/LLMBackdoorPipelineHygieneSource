"""Build, run, and summarize the targeted boundary-corruption cut-sweep grid.

Measures each ``(base, position)`` trigger span from a ``none`` run, derives head-truncation budgets
that place the cut within +/-`window` tokens of the span, runs the sweep through the verified shard
runner, and prints the partial-vs-whole survival table by ``position x cut-region``. Also writes the
boundary manifest (a ready-to-push shard) and the per-trial results.

Usage (offline, tokenizer cached):
    python scripts/run_boundary_grid.py \
        --bases data/boundary/bases_qwen3.jsonl \
        --models-config configs/prod/models.prod.yaml \
        --policies-config configs/prod/policies.prod.yaml \
        --triggers data/triggers/triggers.jsonl \
        --trigger-id boundary_001 --model-id qwen3-0_6b \
        --positions prefix middle end old_turn \
        --out-dir outputs/boundary_grid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trigger_audit.config import load_models, load_pipeline_policies
from trigger_audit.experiments.survivability_audit import SurvivalShardRunner
from trigger_audit.experiments.survivability_audit.boundary_grid import (
    measure_and_expand,
    summarize,
)
from trigger_audit.io.jsonl import write_jsonl
from trigger_audit.io.stores import BaseConversationStore, TriggerStore
from trigger_audit.schemas.triggers import TriggerPosition
from trigger_audit.tokenization.tokenizer_adapter import make_tokenizer_adapter


def main() -> int:
    """Run the boundary cut-sweep end to end and print/write the results."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bases", type=Path, required=True)
    ap.add_argument("--models-config", type=Path, required=True)
    ap.add_argument("--policies-config", type=Path, required=True)
    ap.add_argument("--triggers", type=Path, default=Path("data/triggers/triggers.jsonl"))
    ap.add_argument("--trigger-id", default="boundary_001")
    ap.add_argument("--model-id", default="qwen3-0_6b")
    ap.add_argument("--positions", nargs="+", default=["prefix", "middle", "end", "old_turn"])
    ap.add_argument("--window", type=int, default=20, help="cut sweep half-width around the span")
    ap.add_argument("--inside-steps", type=int, default=3, help="interior cut points per span")
    ap.add_argument("--backend", default="hf")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/boundary_grid"))
    args = ap.parse_args()

    model_configs = load_models(args.models_config)
    policies = load_pipeline_policies(args.policies_config)

    def factory(mc):  # type: ignore[no-untyped-def]
        return make_tokenizer_adapter(
            mc.resolved_tokenizer_id(),
            backend=args.backend,
            revision=mc.revision,
            trust_remote_code=mc.trust_remote_code,
        )

    runner = SurvivalShardRunner(
        base_store=BaseConversationStore(args.bases),
        trigger_store=TriggerStore(args.triggers),
        model_configs=model_configs,
        pipeline_policies=policies,
        tokenizer_factory=factory,
    )

    lines = [ln for ln in args.bases.read_text().splitlines() if ln]
    base_ids = [json.loads(ln)["base_id"] for ln in lines]
    positions = [TriggerPosition(p) for p in args.positions]

    trials, measured = measure_and_expand(
        runner,
        base_ids=base_ids,
        trigger_id=args.trigger_id,
        positions=positions,
        model_id=args.model_id,
        window=args.window,
        inside_steps=args.inside_steps,
    )
    manifest = {t.trial_id: t.trigger_present for t in trials}

    results = [runner.run_trial(t)[0] for t in trials]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "boundary_manifest.jsonl", trials)
    write_jsonl(args.out_dir / "boundary_results.jsonl", results)
    # Provenance: the measured none-run span per (base, position) the budgets were derived from.
    spans = {
        key: {
            "span_start": r.trigger_final_token_start,
            "span_end": r.trigger_final_token_end,
            "total": r.final_prompt_token_count,
        }
        for key, r in measured.items()
    }
    (args.out_dir / "measured_spans.json").write_text(json.dumps(spans, indent=2))

    # Counterfactual control: every trigger-absent twin must be no_survival, delivered nowhere.
    absent = [r for r in results if not manifest[r.trial_id]]
    leaks = [
        r
        for r in absent
        if r.final_token_trigger_present or r.survival_class.value != "no_survival"
    ]

    rows = summarize(results, manifest)
    print(
        f"\nBoundary cut-sweep: {len(trials)} trials "
        f"({sum(manifest.values())} present + {len(absent)} twins), "
        f"trigger={args.trigger_id} model={args.model_id}"
    )
    print(
        f"Counterfactual control: {len(absent)} twins, leaks={len(leaks)}"
        + ("  [OK]" if not leaks else "  [LEAK!]")
    )
    print(
        f"\n{'position':12s}{'cut_region':12s}{'n':>5s}{'whole':>8s}{'boundary':>10s}"
        f"{'lost':>7s}{'partial':>9s}"
    )
    print("-" * 63)
    for r in rows:
        print(
            f"{r['position']:12s}{r['cut_region']:12s}{r['n']:>5d}"
            f"{r['whole_rate']:>8.2f}{r['boundary_rate']:>10.2f}"
            f"{r['lost_rate']:>7.2f}{r['partial_survived_rate']:>9.2f}"
        )
    (args.out_dir / "boundary_summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote -> {args.out_dir}/ (manifest, results, summary)")
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
