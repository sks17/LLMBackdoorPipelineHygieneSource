"""Load survival results (+ optional manifest/bases) into one tidy, reconciled trials table.

The tidy table is the single object every table and figure reads. Results are the only required
input; the manifest (authoritative ``trigger_present`` + pairing) and the bases file (H4 / family /
generation-model covariates) are joined when available and reconcile, and skipped otherwise -- the
existing local pilot artifact has no matching manifest/bases, so results-only must work (see
``docs/ANALYSIS_PLAN.md`` §12).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from trigger_audit.analysis.vocab import SUMMARIZE_MEMORY_POLICIES, outcome_band, policy_mechanism
from trigger_audit.config import load_pipeline_policies
from trigger_audit.config.settings import PipelinePolicyConfig
from trigger_audit.io.jsonl import read_jsonl_as
from trigger_audit.io.stores import BaseConversationStore
from trigger_audit.schemas.results import SurvivalResult
from trigger_audit.schemas.trials import TrialSpec

PathLike = str | Path

# The coordinates a counterfactual pair shares -- unlike io.manifest.pair_key, this INCLUDES
# trigger_id, so grouping recovers matched present/absent pairs (1 present + 1 absent) rather than
# collapsing all triggers at a grid point together. See ANALYSIS_PLAN.md correction 2.
PAIR_COORDS = [
    "base_id",
    "model_id",
    "trigger_position",
    "pipeline_policy",
    "context_length",
    "trigger_id",
]


@dataclass
class ReconReport:
    """Outcome of reconciling the joined inputs; a hard-fail check, surfaced to the caller."""

    n_results: int
    n_present: int
    n_absent: int
    duplicate_rows_dropped: int
    manifest_joined: bool
    missing_from_manifest: int
    bases_joined: bool
    unmatched_base_ids: int
    n_pairs: int
    notes: list[str] = field(default_factory=list)


def _result_files(results: PathLike) -> list[Path]:
    path = Path(results)
    return sorted(path.glob("*.jsonl")) if path.is_dir() else [path]


def load_results(results: PathLike) -> pd.DataFrame:
    """Read one file or a directory of survival-result shards into a validated DataFrame."""
    rows: list[dict[str, object]] = []
    for file in _result_files(results):
        rows.extend(r.model_dump(mode="json") for r in read_jsonl_as(file, SurvivalResult))
    if not rows:
        raise ValueError(f"no survival results found under {results}")
    return pd.DataFrame(rows)


def _meta_float(meta: object, key: str) -> float:
    """Read a numeric field out of a result's ``metadata`` block as a float (NaN if absent)."""
    if isinstance(meta, Mapping):
        value = meta.get(key)
        if value is not None:
            return float(value)
    return float("nan")


def _meta_str(meta: object, key: str) -> str | None:
    """Read a string field out of a result's ``metadata`` block (None if absent/non-string)."""
    if isinstance(meta, Mapping):
        value = meta.get(key)
        if isinstance(value, str):
            return value
    return None


def _meta_span(meta: object) -> tuple[float, float]:
    """Unpack ``pretrunc_trigger_span`` (``[start, end)``) to a float pair; NaN pair if null."""
    if isinstance(meta, Mapping):
        span = meta.get("pretrunc_trigger_span")
        if isinstance(span, (list, tuple)) and len(span) == 2:
            return float(span[0]), float(span[1])
    return float("nan"), float("nan")


def _attach_cut_geometry(df: pd.DataFrame) -> None:
    """Unpack the scorer's persisted cut-geometry metadata into flat, deterministic columns.

    Each ``SurvivalResult.metadata`` is either ``{}`` (older rows, before the producer persisted the
    block) or the compact cut-anatomy block written by ``scorer.cut_metadata``: ``truncation_policy,
    dropped_head, dropped_tail, pretrunc_token_count, pretrunc_trigger_span``, where the span is the
    trigger's ``[start, end)`` token span in pre-truncation (post-template) coordinates (``None``
    for counterfactual twins / undelivered triggers). We flatten it to scalar columns so figure F6 /
    table T7 read the cut geometry off the tidy frame -- old rows load cleanly with every new column
    NaN/None. Mutates ``df`` in place; ``outcome_band`` must exist.
    """
    metas: list[object] = list(df["metadata"]) if "metadata" in df.columns else [{}] * len(df)

    df["dropped_head"] = [_meta_float(m, "dropped_head") for m in metas]
    df["dropped_tail"] = [_meta_float(m, "dropped_tail") for m in metas]
    df["pretrunc_token_count"] = [_meta_float(m, "pretrunc_token_count") for m in metas]
    df["truncation_meta_policy"] = [_meta_str(m, "truncation_policy") for m in metas]

    spans = [_meta_span(m) for m in metas]
    df["pretrunc_trigger_start"] = [start for start, _ in spans]
    df["pretrunc_trigger_end"] = [end for _, end in spans]
    # trigger_len == the tokenized trigger length, read straight off the span (no re-tokenization).
    df["trigger_len"] = df["pretrunc_trigger_end"] - df["pretrunc_trigger_start"]

    # surviving_fraction = trigger_final_token_end / trigger_len, raw (un-clamped), meaningful only
    # for boundary rows where a cut left a partial suffix and the trigger length is known-positive.
    final_end = pd.to_numeric(df["trigger_final_token_end"], errors="coerce")
    boundary = (df["outcome_band"] == "boundary") & (df["trigger_len"] > 0)
    df["surviving_fraction"] = (final_end / df["trigger_len"]).where(boundary, float("nan"))

    # cut_offset (signed) = pretrunc_trigger_start - dropped_head, defined for head-truncation rows
    # with a known span. Negative => trigger begins before the cut (survives ahead of it); a value
    # straddling 0 => the cut lands inside the trigger (boundary corruption). Mirrors scorer F6:
    # the dropped_head vs pretrunc_trigger_span read. NaN when no head cut or the span is absent.
    head_cut = (df["dropped_head"] > 0) & df["pretrunc_trigger_start"].notna()
    df["cut_offset"] = (df["pretrunc_trigger_start"] - df["dropped_head"]).where(
        head_cut, float("nan")
    )


def attach_derived(
    df: pd.DataFrame, *, policies: Mapping[str, PipelinePolicyConfig] | None = None
) -> pd.DataFrame:
    """Add the derived columns every downstream reader depends on (see ANALYSIS_PLAN.md §2)."""
    df = df.copy()
    # delivered == token survival by construction (correction 1): one measurement, one column.
    df["delivered"] = df["final_token_trigger_present"].astype(bool)
    df["outcome_band"] = [
        outcome_band(sc, fs)
        for sc, fs in zip(df["survival_class"], df["failure_stage"], strict=True)
    ]
    # present/absent: use the manifest's authoritative flag when joined, else infer from Layer-1
    # presence (a clean counterfactual grid has raw_trigger_present == trigger_present).
    if "trigger_present" in df.columns:
        df["trigger_present"] = (
            df["trigger_present"]
            .where(df["trigger_present"].notna(), df["raw_trigger_present"])
            .astype(bool)
        )
    else:
        df["trigger_present"] = df["raw_trigger_present"].astype(bool)
    df["pair_id"] = df[PAIR_COORDS].astype(str).agg("|".join, axis=1)

    if policies is not None:
        mech = policy_mechanism(policies)
        df["memory_policy"] = df["pipeline_policy"].map(
            lambda p: mech.get(p, {}).get("memory_policy")
        )
        df["truncation_policy"] = df["pipeline_policy"].map(
            lambda p: mech.get(p, {}).get("truncation_policy")
        )
        df["is_summarize"] = df["pipeline_policy"].map(
            lambda p: bool(mech.get(p, {}).get("is_summarize", False))
        )
        df["policy_display"] = df["pipeline_policy"].map(
            lambda p: mech.get(p, {}).get("display", p)
        )
    else:
        # Fallback without a config: the pilot policy id equals its memory-policy name here.
        df["is_summarize"] = df["pipeline_policy"].isin(SUMMARIZE_MEMORY_POLICIES)

    _attach_cut_geometry(df)
    return df


def _dedupe(df: pd.DataFrame, notes: list[str]) -> tuple[pd.DataFrame, int]:
    """Drop byte-identical duplicate rows; a divergent duplicate trial_id is a determinism bug."""
    dup_ids = int(df["trial_id"].duplicated().sum())
    if not dup_ids:
        return df, 0
    before = len(df)
    df = df.drop_duplicates()
    if bool(df["trial_id"].duplicated().any()):
        raise ValueError(
            "divergent duplicate trial_id rows (nondeterministic results) -- aborting; "
            "results must be reproducible per trial"
        )
    dropped = before - len(df)
    notes.append(f"dropped {dropped} byte-identical duplicate row(s)")
    return df, dropped


def load_trials(
    results: PathLike,
    *,
    manifest: PathLike | None = None,
    bases: PathLike | None = None,
    policies_config: PathLike | None = None,
    require_complete: bool = False,
) -> tuple[pd.DataFrame, ReconReport]:
    """Build the tidy trials table with reconciliation; joins manifest/bases only when available."""
    df = load_results(results)
    notes: list[str] = []
    df, dropped = _dedupe(df, notes)

    manifest_joined = False
    missing_from_manifest = 0
    if manifest is not None:
        man = pd.DataFrame(t.model_dump(mode="json") for t in read_jsonl_as(manifest, TrialSpec))
        df = df.merge(man[["trial_id", "trigger_present", "seed"]], on="trial_id", how="left")
        missing_from_manifest = int(df["trigger_present"].isna().sum())
        result_ids, manifest_ids = set(df["trial_id"]), set(man["trial_id"])
        complete = missing_from_manifest == 0 and result_ids == manifest_ids
        if not complete and require_complete:
            raise ValueError(
                f"incomplete/mismatched manifest: {missing_from_manifest} result row(s) unmatched, "
                f"{len(manifest_ids - result_ids)} manifest row(s) with no result"
            )
        if missing_from_manifest:
            notes.append(
                f"{missing_from_manifest} result row(s) had no manifest match; "
                "trigger_present inferred from raw_trigger_present for those"
            )
        manifest_joined = True

    policies = load_pipeline_policies(policies_config) if policies_config is not None else None
    df = attach_derived(df, policies=policies)

    bases_joined = False
    unmatched_base_ids = 0
    if bases is not None:
        store = BaseConversationStore(bases)
        known = {bid: store.get(bid) for bid in store.ids()}

        def base_field(base_id: str, name: str, *, meta: bool = False) -> object:
            base = known.get(base_id)
            if base is None:
                return None
            return base.metadata.get(name) if meta else getattr(base, name, None)

        df["family"] = df["base_id"].map(lambda b: base_field(b, "conversation_type"))
        df["data_source"] = df["base_id"].map(lambda b: base_field(b, "data_source", meta=True))
        df["generation_model"] = df["base_id"].map(
            lambda b: base_field(b, "generation_model", meta=True)
        )
        df["achieved_token_length"] = df["base_id"].map(
            lambda b: base_field(b, "achieved_token_length", meta=True)
        )
        result_base_ids = set(df["base_id"])
        unmatched_base_ids = len(result_base_ids - set(known))
        if unmatched_base_ids:
            notes.append(
                f"{unmatched_base_ids} base_id(s) in results not in the bases file; "
                "covariates left null for those rows"
            )
        bases_joined = True

    # H4 fallback: when no bases file supplied a data_source, infer the arm from the base_id prefix
    # (the generator/parsers name bases `<arm>_<length>_<idx>`, e.g. `synthetic_512_014`). This lets
    # the parity table run on results-only artifacts; it is recorded as a note, never silent.
    if "data_source" not in df.columns or df["data_source"].isna().all():
        prefixes = df["base_id"].str.split("_").str[0]
        if prefixes.str.fullmatch(r"[A-Za-z]+").all():
            df["data_source"] = prefixes
            notes.append("data_source inferred from base_id prefix (no bases file joined)")

    n_present = int((df["trigger_present"]).sum())
    n_absent = int((~df["trigger_present"]).sum())
    n_pairs = int(df.groupby("pair_id").ngroups)
    report = ReconReport(
        n_results=len(df),
        n_present=n_present,
        n_absent=n_absent,
        duplicate_rows_dropped=dropped,
        manifest_joined=manifest_joined,
        missing_from_manifest=missing_from_manifest,
        bases_joined=bases_joined,
        unmatched_base_ids=unmatched_base_ids,
        n_pairs=n_pairs,
        notes=notes,
    )
    return df, report


def present_rows(df: pd.DataFrame) -> pd.DataFrame:
    """The trigger-present rows -- every rate conditions on these (never pool the absent twins)."""
    return df[df["trigger_present"]].copy()


def parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Serialize list/dict-valued cells to JSON strings so the frame writes cleanly to Parquet."""
    import json

    out = df.copy()
    for col in out.columns:
        if out[col].map(lambda v: isinstance(v, (list, dict))).any():
            out[col] = out[col].map(lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v)
    return out
