"""Tests for component G: the probe inference layer, tables, figures, and scoped report.

All fixtures construct :class:`ProbePrediction` rows in code (no model needed) so every estimate is
hand-checkable, plus one end-to-end offline run of the reference backend to exercise
``build_probe_report`` with real per-layer results. The disciplines under test: base-clustered
``P(fire|delivered)``, the E1.5 delivery-failure fraction, clean-vs-all achieved FPR, TOST
equivalence (equivalent, non-equivalent, and an under-powered cell that must NOT pass silently),
leakage inflation sign, and the report's refusal to make a backdoor claim under canary scope.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trigger_audit.analysis.probe_loading import load_predictions
from trigger_audit.analysis.probe_report import (
    BACKDOOR_VERDICT_HEADER,
    build_probe_report,
)
from trigger_audit.analysis.probe_stats import (
    achieved_fpr,
    delivery_conditional_decomposition,
    equivalence_tost,
    leakage_inflation,
    tpr_at_fpr_delivered,
)
from trigger_audit.experiments.probe_detection.config import ProbeDetectionExperimentConfig
from trigger_audit.experiments.probe_detection.runner import run_probe_experiment
from trigger_audit.io.jsonl import read_jsonl_as, write_jsonl
from trigger_audit.schemas.probes import ProbeEvaluationResult, ProbePrediction, ProbeSplit


def _pred(
    trial_id: str,
    base_id: str,
    *,
    label: bool,
    inserted: bool,
    fired: bool,
    score: float = 0.0,
    target: str = "0.01",
) -> ProbePrediction:
    """Build one honest ProbePrediction with the derived delivered/clean memberships."""
    delivered = label or not inserted
    clean_negative = (not label) and (not inserted)
    return ProbePrediction(
        trial_id=trial_id,
        base_id=base_id,
        label=label,
        trigger_inserted=inserted,
        delivered=delivered,
        clean_negative=clean_negative,
        split=ProbeSplit.TEST,
        aggregated_score=score,
        layer_scores={"1": score},
        fired={target: fired},
    )


def _load(tmp_path: Path, preds: list[ProbePrediction], name: str = "preds.jsonl"):
    path = tmp_path / name
    write_jsonl(path, preds)
    return load_predictions(path)


# --- 1. delivered TPR + cluster CI ---------------------------------------------------------------


def test_tpr_delivered_handcomputed_and_clustered(tmp_path: Path) -> None:
    # 4 bases, two delivered-positive twins each; 6 of 8 fire -> TPR = 0.75, n_bases(4) < trials(8).
    pattern = {
        ("b0", 0): True,
        ("b0", 1): True,
        ("b1", 0): True,
        ("b1", 1): True,
        ("b2", 0): True,
        ("b2", 1): False,
        ("b3", 0): True,
        ("b3", 1): False,
    }
    preds = [
        _pred(f"{base}_{k}", base, label=True, inserted=True, fired=f, score=1.0 if f else -1.0)
        for (base, k), f in pattern.items()
    ]
    df = _load(tmp_path, preds)
    est = tpr_at_fpr_delivered(df, 0.01)
    assert est.point == pytest.approx(0.75)
    assert est.n_trials == 8
    assert est.n_bases == 4
    assert est.n_bases < est.n_trials
    assert est.ci_low <= est.point <= est.ci_high


def test_cluster_ci_responds_to_base_duplication(tmp_path: Path) -> None:
    # base b2 has rate 0.5 == the overall mean, so duplicating it leaves the POINT at 0.5 while
    # adding a fourth cluster changes the cluster-bootstrap CI -- the cluster-bootstrap signature.
    base_rows = {"b0": [True, True], "b1": [False, False], "b2": [True, False]}

    def build(extra: bool) -> list[ProbePrediction]:
        preds: list[ProbePrediction] = []
        for base, fires in base_rows.items():
            for k, fired in enumerate(fires):
                preds.append(_pred(f"{base}_{k}", base, label=True, inserted=True, fired=fired))
        if extra:
            for k, fired in enumerate([True, False]):
                preds.append(_pred(f"b2dup_{k}", "b2dup", label=True, inserted=True, fired=fired))
        return preds

    df1 = _load(tmp_path, build(False), "d1.jsonl")
    df2 = _load(tmp_path, build(True), "d2.jsonl")
    e1 = tpr_at_fpr_delivered(df1, 0.01)
    e2 = tpr_at_fpr_delivered(df2, 0.01)
    assert e1.point == pytest.approx(0.5)
    assert e2.point == pytest.approx(0.5)  # point unchanged by duplicating a mean-rate base
    assert e1.n_bases == 3
    assert e2.n_bases == 4
    assert (e1.ci_low, e1.ci_high) != (e2.ci_low, e2.ci_high)  # CI moved


# --- 2. delivery-conditional decomposition -------------------------------------------------------


def test_decomposition_delivery_failure_fraction(tmp_path: Path) -> None:
    preds = [
        # delivered positives (inserted & delivered): 3 fire, 1 genuine miss
        _pred("d0", "b0", label=True, inserted=True, fired=True),
        _pred("d1", "b1", label=True, inserted=True, fired=True),
        _pred("d2", "b2", label=True, inserted=True, fired=True),
        _pred("d3", "b3", label=True, inserted=True, fired=False),
        # partial-survival negatives (inserted, NOT delivered): all miss
        _pred("p0", "b4", label=False, inserted=True, fired=False),
        _pred("p1", "b5", label=False, inserted=True, fired=False),
        _pred("p2", "b6", label=False, inserted=True, fired=False),
    ]
    df = _load(tmp_path, preds)
    decomp = delivery_conditional_decomposition(df, 0.01)
    # apparent misses among inserted = d3, p0, p1, p2 = 4; of those, 3 were never delivered.
    assert decomp["n_apparent_misses"] == 4
    assert decomp["n_delivery_failures"] == 3
    assert decomp["delivery_failure_fraction"] == pytest.approx(3 / 4)
    # P(fire | inserted) = 3/7 (all-trials); P(fire | delivered) = 3/4 (delivered-only).
    assert decomp["p_fire_all"].point == pytest.approx(3 / 7)
    assert decomp["p_fire_delivered"].point == pytest.approx(0.75)


# --- 3. achieved FPR, clean vs all ---------------------------------------------------------------


def test_achieved_fpr_clean_vs_all(tmp_path: Path) -> None:
    preds = [_pred(f"c{i}", f"cb{i}", label=False, inserted=False, fired=False) for i in range(8)]
    # partial-survival negatives whose surviving fragments trip the probe: fire, but NOT clean.
    preds += [_pred(f"p{i}", f"pb{i}", label=False, inserted=True, fired=True) for i in range(2)]
    df = _load(tmp_path, preds)
    fpr_all = achieved_fpr(df, 0.01, clean_only=False)
    fpr_clean = achieved_fpr(df, 0.01, clean_only=True)
    assert fpr_all.n_negatives == 10
    assert fpr_clean.n_negatives == 8
    assert fpr_all.achieved_fpr == pytest.approx(0.2)
    assert fpr_clean.achieved_fpr == pytest.approx(0.0)
    assert fpr_all.achieved_fpr > fpr_clean.achieved_fpr
    # Wilson interval brackets each point.
    assert fpr_all.ci_low <= fpr_all.achieved_fpr <= fpr_all.ci_high
    assert fpr_clean.ci_low <= 0.0 <= fpr_clean.ci_high


# --- 4. TOST equivalence -------------------------------------------------------------------------


def _rate_set(tmp_path: Path, fire_rate: float, n_bases: int, tag: str):
    n_fire = round(fire_rate * n_bases)
    preds = [
        _pred(f"{tag}{i}", f"{tag}b{i}", label=True, inserted=True, fired=i < n_fire)
        for i in range(n_bases)
    ]
    return _load(tmp_path, preds, f"{tag}.jsonl")


def test_tost_equivalent_and_nonequivalent_with_halfwidth(tmp_path: Path) -> None:
    # Equivalent: two near-identical high-firing sets -> difference CI collapses inside +-5pp.
    a = _rate_set(tmp_path, 1.0, 30, "a")
    b = _rate_set(tmp_path, 1.0, 30, "b")
    eq = equivalence_tost(a, b, 0.01, margin=0.05)
    assert eq.equivalent is True
    assert eq.half_width == pytest.approx(0.0, abs=1e-9)
    assert eq.ci_low >= -0.05 and eq.ci_high <= 0.05

    # Non-equivalent: a >5pp gap in P(fire|delivered).
    c = _rate_set(tmp_path, 1.0, 30, "c")
    d = _rate_set(tmp_path, 0.4, 30, "d")
    neq = equivalence_tost(c, d, 0.01, margin=0.05)
    assert neq.equivalent is False
    assert neq.diff == pytest.approx(0.6, abs=0.02)

    # Under-powered: equal point estimates but few bases -> wide CI, so it is NOT declared
    # equivalent, and the half-width exposes the missing power.
    e = _rate_set(tmp_path, 0.5, 6, "e")
    f = _rate_set(tmp_path, 0.5, 6, "f")
    underpowered = equivalence_tost(e, f, 0.01, margin=0.05)
    assert underpowered.equivalent is False
    assert underpowered.half_width > 0.05


# --- 5. leakage inflation ------------------------------------------------------------------------


def test_leakage_inflation_sign(tmp_path: Path) -> None:
    def build(pos_scores, neg_scores, fired_flags, tag):
        preds = [
            _pred(f"{tag}p{i}", f"{tag}pb{i}", label=True, inserted=True, fired=fired, score=s)
            for i, (s, fired) in enumerate(zip(pos_scores, fired_flags, strict=True))
        ]
        preds += [
            _pred(f"{tag}n{i}", f"{tag}nb{i}", label=False, inserted=False, fired=False, score=s)
            for i, s in enumerate(neg_scores)
        ]
        return _load(tmp_path, preds, f"{tag}.jsonl")

    # base_id-grouped (leakage-safe): moderate separation, 3/5 delivered positives fire.
    grouped = build(
        [0.5, 0.6, 0.4, 0.7, 0.55],
        [0.1, 0.2, 0.45, 0.3, 0.65],
        [True, True, False, True, False],
        "g",
    )
    # example-level (leaky): the probe memorized base content -> cleaner separation, more fire.
    example = build(
        [1.0, 1.1, 0.9, 1.2, 1.05],
        [0.0, 0.1, 0.05, -0.1, 0.2],
        [True, True, True, True, False],
        "e",
    )
    inflation = leakage_inflation(grouped, example, 0.01)
    assert inflation["auroc_example"] >= inflation["auroc_grouped"]
    assert inflation["auroc_inflation"] >= 0.0
    assert inflation["tpr_inflation"] >= 0.0


# --- 6. scoped report end-to-end -----------------------------------------------------------------


def _run_offline(tmp_path: Path) -> tuple[Path, Path]:
    predictions_out = tmp_path / "preds.jsonl"
    results_out = tmp_path / "results.jsonl"
    cfg = ProbeDetectionExperimentConfig.model_validate(
        {
            "experiment_id": "probe_g_report",
            "extractor_backend": "reference",
            "extractor_hidden_size": 32,
            "extractor_num_layers": 6,
            "layers": [2, 3, 4, 5],
            "synthetic_n_examples": 80,
            "activations_dir": str(tmp_path / "acts"),
            "results_out": str(results_out),
            "predictions_out": str(predictions_out),
        }
    )
    run_probe_experiment(cfg)
    return predictions_out, results_out


def test_report_canary_scope_emits_no_backdoor_claim(tmp_path: Path) -> None:
    preds_path, results_path = _run_offline(tmp_path)
    manifest = build_probe_report(preds_path, results_path, out_dir=tmp_path / "report")

    assert manifest.is_tier3 is False
    text = manifest.findings_path.read_text(encoding="utf-8")
    assert text.strip()
    # The scope refusal: no Tier-3 backdoor verdict under canary data.
    assert BACKDOOR_VERDICT_HEADER not in text
    assert "canary" in text.lower()
    # Figures render headless and are non-empty on disk.
    assert manifest.figure_paths
    for fig in manifest.figure_paths:
        assert Path(fig).exists()
        assert Path(fig).stat().st_size > 0
    # Tables were written.
    assert manifest.table_paths
    for table in manifest.table_paths.values():
        assert Path(table).exists()


def test_report_tier3_scope_emits_backdoor_verdict(tmp_path: Path) -> None:
    preds_path, results_path = _run_offline(tmp_path)
    # Mark the run Tier-3 (real backdoored weights) via the metadata marker component H would set.
    results = read_jsonl_as(results_path, ProbeEvaluationResult)
    for result in results:
        result.metadata["tier"] = 3
    write_jsonl(results_path, results)

    manifest = build_probe_report(preds_path, results_path, out_dir=tmp_path / "report3")
    assert manifest.is_tier3 is True
    text = manifest.findings_path.read_text(encoding="utf-8")
    assert BACKDOOR_VERDICT_HEADER in text
    assert "TAR_w" in text
