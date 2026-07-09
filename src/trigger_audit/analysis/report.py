"""Orchestrate the analysis: load -> Gate 0 -> tables -> figures -> a report directory.

Produces ``<out>/{trials.parquet, gate.json, tables/*.{csv,md,tex}, figures/*, report.md,
manifest.json}``. Tables span the headline delivered-rate layer (T1-T3), the equivalence layer
(T4 model-invariance / T5 synthetic-vs-real, TOST with a configurable margin + Holm/BH), and the
mechanism layer (T6 misattribution, T7 boundary census); figures cover F0-F8 (F6 = anatomy of the
cut). Returns a process exit code: non-zero if the counterfactual control leaks (the analysis is
unsound) or if reconciliation with a manifest was required and failed. The equivalence margin and
multiplicity scheme are locked in the ``PRE_REGISTRATION.md`` amendment; ``docs/ANALYSIS_PLAN.md``
has the full spec of every table and figure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from trigger_audit.analysis.controls import verify_counterfactual
from trigger_audit.analysis.loading import (
    PathLike,
    load_trials,
    parquet_safe,
    present_rows,
)
from trigger_audit.analysis.tables import (
    boundary_census_table,
    data_source_table,
    delivered_by_trigger_type,
    delivered_rate_ci_table,
    delivered_rate_table,
    failure_attribution_table,
    h2_invariance_table,
    h4_parity_table,
    mcnemar_table,
    misattribution_table,
    outcome_band_table,
    risk_difference_table,
)


def _md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavored Markdown table (no external dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_hashes(results: PathLike, extras: dict[str, PathLike | None]) -> dict[str, object]:
    hashes: dict[str, object] = {}
    root = Path(results)
    files = sorted(root.glob("*.jsonl")) if root.is_dir() else [root]
    hashes["results"] = {str(f): _sha256(f) for f in files}
    for name, path in extras.items():
        hashes[name] = _sha256(Path(path)) if path is not None else None
    return hashes


def _render_figures(df: pd.DataFrame, out_dir: Path) -> list[str]:
    """Render figures if matplotlib is available; return paths (empty if the extra is absent)."""
    try:
        from trigger_audit.analysis import figures
    except ImportError:
        return []
    paths = figures.render_all(df, out_dir / "figures")
    return [str(p) for p in paths]


def build_report(
    results: PathLike,
    out: PathLike,
    *,
    manifest: PathLike | None = None,
    bases: PathLike | None = None,
    policies_config: PathLike | None = None,
    require_complete: bool = False,
    render_figures: bool = True,
    tost_margin: float = 0.05,
) -> int:
    """Run the decision-free analysis end to end, write the report dir, and return an exit code."""
    out_dir = Path(out)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df, recon = load_trials(
        results,
        manifest=manifest,
        bases=bases,
        policies_config=policies_config,
        require_complete=require_complete,
    )
    verdict = verify_counterfactual(df)
    present = present_rows(df)
    # Headline excludes the summarize-family placeholder stub; the full-grid variant is kept in T1b.
    headline = present[~present["is_summarize"]] if "is_summarize" in present else present

    tables = {
        "t1_delivered_rate": delivered_rate_table(headline),
        "t1_delivered_rate_ci": delivered_rate_ci_table(headline),
        "t1b_delivered_rate_all_policies": delivered_rate_table(present),
        "t1c_delivered_by_trigger_type": delivered_by_trigger_type(headline),
        "t2_failure_attribution": failure_attribution_table(present),
        "t3_mcnemar_control": mcnemar_table(df),
        "t_risk_difference": risk_difference_table(headline),
        "t4_outcome_bands": outcome_band_table(present),
        "t4_h2_model_invariance": h2_invariance_table(headline, margin=tost_margin),
        "t5_h4_synthetic_vs_real": h4_parity_table(headline, margin=tost_margin),
        "t6_misattribution": misattribution_table(headline),
        "t7_boundary_census": boundary_census_table(present),
    }
    h4 = data_source_table(present)
    if h4 is not None:
        tables["t5_delivered_by_data_source"] = h4

    for name, table in tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
        (tables_dir / f"{name}.md").write_text(_md_table(table), encoding="utf-8")
        (tables_dir / f"{name}.tex").write_text(table.to_latex(index=False), encoding="utf-8")

    parquet_safe(df).to_parquet(out_dir / "trials.parquet", index=False)

    gate = {
        "ok": verdict.ok,
        "n_absent": verdict.n_absent,
        "n_leaks": verdict.n_leaks,
        "leak_examples": verdict.leak_examples,
        "reconciliation": asdict(recon),
    }
    (out_dir / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")

    manifest_json = {
        "n_results": recon.n_results,
        "n_present": recon.n_present,
        "n_absent": recon.n_absent,
        "n_pairs": recon.n_pairs,
        "manifest_joined": recon.manifest_joined,
        "bases_joined": recon.bases_joined,
        "control_ok": verdict.ok,
        "input_sha256": _input_hashes(
            results, {"manifest": manifest, "bases": bases, "policies_config": policies_config}
        ),
        "pandas_version": pd.__version__,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest_json, indent=2), encoding="utf-8")

    figure_paths = _render_figures(df, out_dir) if render_figures else []
    (out_dir / "report.md").write_text(
        _render_report_md(verdict, recon, tables, figure_paths, tost_margin), encoding="utf-8"
    )
    return 0 if verdict.ok else 1


def _render_report_md(verdict, recon, tables, figure_paths=None, tost_margin=0.05) -> str:  # type: ignore[no-untyped-def]
    parts = [
        "# Trigger-delivery survival — analysis report",
        "",
        "> Decision-free layer: counterfactual control, headline delivered-rate tables with "
        "cluster-bootstrap CIs, risk-difference effect sizes, the McNemar control, TOST "
        "equivalence (H2/H4), the misattribution decomposition (T6) and boundary census (T7), and "
        "figures F0-F8 (including F6, anatomy of the cut). See `docs/ANALYSIS_PLAN.md`.",
        "",
        "## Gate 0 — counterfactual control",
        "",
        f"- {verdict.summary()}",
        f"- **{'PASS' if verdict.ok else 'FAIL'}** — "
        + (
            "every trigger-absent twin delivered nothing."
            if verdict.ok
            else "control leaked; rates below are NOT trustworthy."
        ),
        "",
        "## Reconciliation",
        "",
        f"- results rows: {recon.n_results} ({recon.n_present} present / {recon.n_absent} absent)",
        f"- counterfactual pairs (base|model|position|policy|length|trigger): {recon.n_pairs}",
        f"- manifest joined: {recon.manifest_joined}"
        + (f" ({recon.missing_from_manifest} unmatched)" if recon.manifest_joined else ""),
        f"- bases joined: {recon.bases_joined}"
        + (f" ({recon.unmatched_base_ids} unmatched base ids)" if recon.bases_joined else ""),
    ]
    for note in recon.notes:
        parts.append(f"- note: {note}")
    parts += [
        "",
        "## T1 - delivered rate by policy x position (headline; summarize-family excluded)",
        "",
        "Cluster-bootstrap 95% CI over `base_id` is primary (`boot_lo`/`boot_hi`); the Wilson "
        "interval is the per-trial approximation. `boot_ci_halfwidth` shows achieved precision.",
        "",
    ]
    parts.append(_md_table(tables["t1_delivered_rate_ci"]))
    parts += [
        "",
        "## Risk differences vs the `none` control (H1/H3 effect sizes)",
        "",
        "Paired cluster-bootstrap CI over shared bases; `diff` = delivered(policy) minus "
        "delivered(none).",
        "",
    ]
    parts.append(_md_table(tables["t_risk_difference"]))
    parts += [
        "",
        "## T3 - counterfactual control (McNemar)",
        "",
        "`c` (absent-delivered) must be ~0 under a clean control; `b` is the present-delivered "
        "pair count. With a degenerate control arm this is a sanity statistic, not H1-H4 evidence.",
        "",
    ]
    parts.append(_md_table(tables["t3_mcnemar_control"]))
    parts += [
        "",
        "## T4/T5 — equivalence (TOST) for H2 (model-invariance) and H4 (synthetic vs real)",
        "",
        f"Margin = ±{tost_margin:.0%} (provisional — lock via a `PRE_REGISTRATION.md` amendment). "
        "`equivalent_*` uses each correction's adjusted p < 0.05. **Both** corrections are shown: "
        "`_holm` (family-wise) and `_bh` (false-discovery-rate).",
        "",
        "### H2 — model-invariance",
        "",
    ]
    h2 = tables["t4_h2_model_invariance"]
    parts.append(_md_table(h2) if not h2.empty else "_Not evaluated: the run has a single model._")
    parts += ["", "### H4 — synthetic vs real parity", ""]
    h4t = tables["t5_h4_synthetic_vs_real"]
    parts.append(
        _md_table(h4t) if not h4t.empty else "_Not evaluated: only one data_source in this run._"
    )
    parts += ["", "## T2 — failure attribution (non-delivered present rows)", ""]
    parts.append(_md_table(tables["t2_failure_attribution"]))
    parts += [
        "",
        "## T6 — misattribution (apparent-failure decomposition)",
        "",
        "Per policy x position over trigger-present rows: the apparent-failure rate (1 - "
        "delivered) an evaluator would misread as model robustness, with the non-delivered rows "
        "decomposed by `failure_stage` into upstream-drop / token-truncation / "
        "template-incompatible / compressed / other. The backdoor-failure-is-delivery-failure "
        "table.",
        "",
    ]
    parts.append(_md_table(tables["t6_misattribution"]))
    boundary = tables["t7_boundary_census"]
    parts += [
        "",
        "## T7 — boundary census",
        "",
        "Every `boundary_corruption` trial: budget, surviving-suffix length, `surviving_fraction`, "
        "signed `cut_offset` (section-3 cut geometry), and a pointer into "
        "`final_prompt_text_path`. Doubles as the failure-example dump.",
        "",
    ]
    parts.append(
        _md_table(boundary) if not boundary.empty else "_No boundary-corruption rows in this run._"
    )
    if figure_paths:
        parts += ["", "## Figures", ""]
        for path in figure_paths:
            name = Path(path).stem
            parts.append(f"- **{name}** — `figures/{Path(path).name}`")
    parts.append("")
    return "\n".join(parts)


def main(argv: list[str]) -> int:
    """CLI shim: analyze a results dir/file into a report directory. See module docstring."""
    import argparse

    parser = argparse.ArgumentParser(description="Decision-free trigger-delivery analysis.")
    parser.add_argument("results", help="Survival results JSONL file or directory of shards.")
    parser.add_argument("--out", required=True, help="Output report directory.")
    parser.add_argument("--manifest", default=None, help="Trial manifest JSONL (optional join).")
    parser.add_argument("--bases", default=None, help="Base conversations JSONL (optional join).")
    parser.add_argument("--policies-config", default=None, help="Policies YAML (optional labels).")
    parser.add_argument("--require-complete", action="store_true", help="Fail on partial coverage.")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure rendering.")
    parser.add_argument("--tost-margin", type=float, default=0.05, help="TOST equivalence margin.")
    args = parser.parse_args(argv)
    code = build_report(
        args.results,
        args.out,
        manifest=args.manifest,
        bases=args.bases,
        policies_config=args.policies_config,
        require_complete=args.require_complete,
        render_figures=not args.no_figures,
        tost_margin=args.tost_margin,
    )
    print(f"analysis written to {args.out} (exit {code})")
    return code


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
