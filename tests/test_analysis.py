"""Tests for the decision-free analysis layer (loading, Gate 0, headline tables).

Correctness tests use tiny in-memory fixtures with hand-computable rates. One golden test runs
against the checked-in pilot artifact (``outputs/pilot/survival.jsonl``) when present, asserting the
headline table reproduces ``aggregate_survival`` exactly and the corrected pairing recovers 1:1
counterfactual pairs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trigger_audit.analysis.controls import verify_counterfactual
from trigger_audit.analysis.loading import load_trials, present_rows
from trigger_audit.analysis.stats import (
    benjamini_hochberg,
    bootstrap_rate_ci,
    exact_mcnemar_p,
    holm,
    mcnemar_from_pairs,
    tost_equivalence,
    wilson_ci,
)
from trigger_audit.analysis.tables import (
    delivered_rate_ci_table,
    delivered_rate_table,
    h4_parity_table,
    risk_difference_table,
)
from trigger_audit.analysis.vocab import outcome_band
from trigger_audit.io.jsonl import read_jsonl_as, write_jsonl
from trigger_audit.schemas.results import FailureStage, SurvivalClass, SurvivalResult
from trigger_audit.schemas.triggers import TriggerPosition

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_SURVIVAL = REPO_ROOT / "outputs" / "pilot" / "survival.jsonl"


def _row(
    *,
    trial_id: str,
    trigger_present: bool,
    policy: str = "none",
    position: TriggerPosition = TriggerPosition.PREFIX,
    trigger_id: str = "rand_001",
    delivered: bool,
    survival_class: SurvivalClass = SurvivalClass.EXACT_SURVIVAL,
    failure_stage: FailureStage = FailureStage.NONE,
    base_id: str = "b0",
) -> SurvivalResult:
    """Build a minimal SurvivalResult; ``trigger_present`` drives the raw-layer flag."""
    return SurvivalResult(
        trial_id=trial_id,
        base_id=base_id,
        model_id="m0",
        tokenizer_id="m0",
        trigger_id=trigger_id,
        trigger_text="CANARY_TRIGGER_7F3XQ",
        trigger_position=position,
        context_length=256,
        pipeline_policy=policy,
        raw_trigger_present=trigger_present,
        post_pipeline_trigger_present=trigger_present,
        post_template_trigger_present=trigger_present,
        final_token_trigger_present=delivered,
        trigger_exact_survived=delivered and survival_class == SurvivalClass.EXACT_SURVIVAL,
        trigger_token_survived=delivered,
        trigger_partial_survived=survival_class == SurvivalClass.BOUNDARY_CORRUPTION,
        final_prompt_token_count=100,
        survival_class=survival_class
        if delivered or not trigger_present
        else SurvivalClass.NO_SURVIVAL,
        failure_stage=failure_stage if delivered else FailureStage.FINAL_TOKEN_ABSENT,
    )


def test_outcome_band_mapping() -> None:
    assert outcome_band("exact_survival", "none") == "exact"
    assert outcome_band("boundary_corruption", "truncated_head") == "boundary"
    assert outcome_band("no_survival", "template_incompatible") == "template_incompatible"
    assert outcome_band("no_survival", "memory_policy_dropped") == "none"


def test_gate_passes_on_clean_control(tmp_path: Path) -> None:
    rows = [
        _row(trial_id="p1", trigger_present=True, delivered=True),
        _row(
            trial_id="a1",
            trigger_present=False,
            delivered=False,
            survival_class=SurvivalClass.NO_SURVIVAL,
            failure_stage=FailureStage.FINAL_TOKEN_ABSENT,
        ),
    ]
    path = tmp_path / "r.jsonl"
    write_jsonl(path, rows)
    df, _ = load_trials(path)
    verdict = verify_counterfactual(df)
    assert verdict.ok
    assert verdict.n_absent == 1 and verdict.n_leaks == 0


def test_gate_detects_leak(tmp_path: Path) -> None:
    # An absent twin that (wrongly) delivered -- the exact failure Gate 0 exists to catch.
    leaky = _row(trial_id="a1", trigger_present=False, delivered=True)
    path = tmp_path / "r.jsonl"
    write_jsonl(path, [_row(trial_id="p1", trigger_present=True, delivered=True), leaky])
    df, _ = load_trials(path)
    verdict = verify_counterfactual(df)
    assert not verdict.ok
    assert verdict.n_leaks == 1
    assert verdict.leak_examples[0]["trial_id"] == "a1"


def test_pairing_includes_trigger_id(tmp_path: Path) -> None:
    # Two triggers at one grid point, each with a present+absent twin -> 2 pairs of 2, not 1 of 4.
    rows = []
    for tid in ("rand_001", "multi_001"):
        rows.append(_row(trial_id=f"p_{tid}", trigger_present=True, delivered=True, trigger_id=tid))
        rows.append(
            _row(trial_id=f"a_{tid}", trigger_present=False, delivered=False, trigger_id=tid)
        )
    path = tmp_path / "r.jsonl"
    write_jsonl(path, rows)
    df, recon = load_trials(path)
    assert recon.n_pairs == 2
    assert df.groupby("pair_id").size().tolist() == [2, 2]


def test_delivered_rate_table_matches_hand_values(tmp_path: Path) -> None:
    # 3 present prefix rows under 'none': 2 delivered -> rate 2/3.
    rows = [
        _row(trial_id="p1", trigger_present=True, delivered=True, base_id="b0"),
        _row(trial_id="p2", trigger_present=True, delivered=True, base_id="b1"),
        _row(trial_id="p3", trigger_present=True, delivered=False, base_id="b2"),
        _row(trial_id="a1", trigger_present=False, delivered=False, base_id="b0"),
    ]
    path = tmp_path / "r.jsonl"
    write_jsonl(path, rows)
    df, _ = load_trials(path)
    table = delivered_rate_table(present_rows(df))
    assert len(table) == 1
    cell = table.iloc[0]
    assert cell["n"] == 3
    assert cell["bases"] == 3
    assert cell["delivered_rate"] == pytest.approx(2 / 3)


def test_wilson_ci_known_value() -> None:
    lo, hi = wilson_ci(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_exact_mcnemar_p_known_values() -> None:
    assert exact_mcnemar_p(10, 0) == pytest.approx(2 * 0.5**10)
    assert exact_mcnemar_p(5, 5) == pytest.approx(1.0)
    assert exact_mcnemar_p(0, 0) == 1.0


def test_bootstrap_rate_ci_is_deterministic_and_brackets_point() -> None:
    values = [1, 1, 0, 1, 0, 0]
    clusters = ["b0", "b1", "b2", "b3", "b4", "b5"]
    point, lo, hi = bootstrap_rate_ci(values, clusters, n_boot=500, seed=0)
    assert point == pytest.approx(0.5)
    assert lo <= point <= hi
    # Reproducible for a fixed seed.
    assert bootstrap_rate_ci(values, clusters, n_boot=500, seed=0) == (point, lo, hi)


def test_bootstrap_rate_ci_collapses_when_no_cluster_variation() -> None:
    point, lo, hi = bootstrap_rate_ci([1, 1, 1], ["b0", "b1", "b2"], n_boot=200, seed=0)
    assert (point, lo, hi) == (1.0, 1.0, 1.0)


def test_mcnemar_pairs_and_ci_table_on_fixture(tmp_path: Path) -> None:
    # One grid point, one trigger, present delivered + absent not -> b=1, c=0.
    rows = [
        _row(trial_id="p1", trigger_present=True, delivered=True, base_id="b0"),
        _row(trial_id="a1", trigger_present=False, delivered=False, base_id="b0"),
    ]
    path = tmp_path / "r.jsonl"
    write_jsonl(path, rows)
    df, _ = load_trials(path)
    stat = mcnemar_from_pairs(df)
    assert stat["n_pairs"] == 1 and stat["b"] == 1 and stat["c"] == 0
    ci = delivered_rate_ci_table(present_rows(df))
    assert {"boot_lo", "boot_hi", "wilson_lo", "wilson_hi"}.issubset(ci.columns)


def test_holm_and_bh_known_values() -> None:
    # Classic worked example p = [0.01, 0.02, 0.03, 0.04], m=4.
    p = [0.01, 0.02, 0.03, 0.04]
    assert holm(p) == pytest.approx([0.04, 0.06, 0.06, 0.06])
    assert benjamini_hochberg(p) == pytest.approx([0.04, 0.04, 0.04, 0.04])
    # NaNs pass through and are excluded from the family size.
    out = holm([0.01, float("nan"), 0.02])
    assert out[1] != out[1]  # NaN
    assert out[0] == pytest.approx(0.02) and out[2] == pytest.approx(0.02)


def _tost_frame(rate_a: float, rate_b: float, n_bases: int = 40):
    import pandas as pd

    rows = []
    for i in range(n_bases):
        rows.append({"base_id": f"b{i}", "grp": "a", "delivered": i < rate_a * n_bases})
        rows.append({"base_id": f"b{i}", "grp": "b", "delivered": i < rate_b * n_bases})
    return pd.DataFrame(rows)


def test_tost_declares_equivalence_when_rates_match() -> None:
    df = _tost_frame(0.50, 0.50)
    res = tost_equivalence(df, cond_col="grp", cond_a="a", cond_b="b", margin=0.05)
    assert res["equivalent"] is True
    assert res["p_tost"] < 0.05


def test_tost_rejects_equivalence_when_rates_differ() -> None:
    df = _tost_frame(0.90, 0.30)
    res = tost_equivalence(df, cond_col="grp", cond_a="a", cond_b="b", margin=0.05)
    assert res["equivalent"] is False
    assert res["p_tost"] > 0.05


@pytest.mark.skipif(not PILOT_SURVIVAL.exists(), reason="pilot survival artifact not present")
def test_h4_parity_table_runs_on_inferred_data_source() -> None:
    df, _ = load_trials(PILOT_SURVIVAL)
    # data_source is inferred from base_id prefixes (synthetic_*, longdoc_*) since no bases joined.
    assert "data_source" in df.columns
    table = h4_parity_table(present_rows(df), margin=0.05)
    assert not table.empty
    for col in ("p_tost", "p_holm", "p_bh", "equivalent_holm", "equivalent_bh"):
        assert col in table.columns


def test_figures_render_smoke(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from trigger_audit.analysis import figures

    rows = []
    for pol in ("none", "truncate_head"):
        for pos in (TriggerPosition.PREFIX, TriggerPosition.END):
            for i in range(3):
                delivered = not (pol == "truncate_head" and pos == TriggerPosition.PREFIX)
                rows.append(
                    _row(
                        trial_id=f"p_{pol}_{pos.value}_{i}",
                        trigger_present=True,
                        policy=pol,
                        position=pos,
                        delivered=delivered,
                        base_id=f"b{i}",
                    )
                )
                rows.append(
                    _row(
                        trial_id=f"a_{pol}_{pos.value}_{i}",
                        trigger_present=False,
                        policy=pol,
                        position=pos,
                        delivered=False,
                        base_id=f"b{i}",
                    )
                )
    path = tmp_path / "r.jsonl"
    write_jsonl(path, rows)
    df, _ = load_trials(path)
    out = tmp_path / "figs"
    paths = figures.render_all(df, out)
    assert len(paths) == 9  # F0-F8 (F6 = cut anatomy)
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


@pytest.mark.skipif(not PILOT_SURVIVAL.exists(), reason="pilot survival artifact not present")
def test_golden_pilot_control_and_effect_sizes() -> None:
    df, _ = load_trials(PILOT_SURVIVAL)
    # Clean control: c (absent-delivered) is 0 across every policy.
    assert mcnemar_from_pairs(df)["c"] == 0
    # H1/H3 effect size: head truncation destroys the prefix trigger entirely vs the none control.
    rd = risk_difference_table(present_rows(df))
    cell = rd[(rd["policy"] == "truncate_head") & (rd["trigger_position"] == "prefix")]
    assert len(cell) == 1
    assert float(cell.iloc[0]["diff"]) == pytest.approx(-1.0)


@pytest.mark.skipif(not PILOT_SURVIVAL.exists(), reason="pilot survival artifact not present")
def test_golden_pilot_reproduces_aggregate_survival() -> None:
    from trigger_audit.experiments.survivability_audit import aggregate_survival

    results = read_jsonl_as(PILOT_SURVIVAL, SurvivalResult)
    assert len(results) == 2304
    present_results = [r for r in results if r.raw_trigger_present]
    absent_results = [r for r in results if not r.raw_trigger_present]
    assert len(present_results) == 1152 and len(absent_results) == 1152

    df, recon = load_trials(PILOT_SURVIVAL)
    assert recon.n_present == 1152 and recon.n_absent == 1152
    assert recon.n_pairs == 1152  # corrected pairing: 1152 pairs of 2, not 384 of 6
    assert verify_counterfactual(df).ok

    expected = {
        (d["pipeline_policy"], d["trigger_position"]): d
        for d in aggregate_survival(present_results)
    }
    table = delivered_rate_table(present_rows(df))
    assert len(table) == len(expected)
    for _, row in table.iterrows():
        ref = expected[(str(row["pipeline_policy"]), str(row["trigger_position"]))]
        assert row["n"] == ref["n"]
        assert row["delivered_rate"] == pytest.approx(ref["delivered_rate"])
        assert row["exact_rate"] == pytest.approx(ref["exact_rate"])
        assert row["token_rate"] == pytest.approx(ref["token_rate"])
        assert row["partial_rate"] == pytest.approx(ref["partial_rate"])
