"""Probe-detection runner: extract, pool, train, calibrate, evaluate, aggregate.

For each configured layer the runner extracts activations over each example's final token
ids, pools them into feature vectors, trains a per-layer linear probe on TRAIN, calibrates
thresholds to the target FPRs on the CLEAN (never-inserted) CALIBRATION negatives, and
evaluates on TEST twice: over all trials and over delivered-only trials (positives verified
delivered plus clean negatives). Token ids arrive via an injected provider, keeping the
runner decoupled from where final tokens come from (Project 1 logs, a store, or a builder).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from trigger_audit.activations.extractor import ActivationExtractor, make_activation_extractor
from trigger_audit.activations.pooling import pool_activations
from trigger_audit.activations.slicing import resolve_layers_from_fractions
from trigger_audit.activations.store import ActivationStore
from trigger_audit.experiments.probe_detection.config import ProbeDetectionExperimentConfig
from trigger_audit.experiments.probe_detection.dataset import (
    assign_splits,
    assign_splits_example_level,
    build_probe_examples,
    build_synthetic_probe_dataset,
    build_synthetic_probe_dataset_with_twins,
)
from trigger_audit.experiments.probe_detection.generalization import (
    assign_generalization_splits,
)
from trigger_audit.io.jsonl import iter_jsonl, read_jsonl_as, write_jsonl
from trigger_audit.probes.aggregation import AGGREGATION_REGISTRY
from trigger_audit.probes.calibration import calibrate_threshold, wilson_interval
from trigger_audit.probes.linear import LinearProbe
from trigger_audit.probes.metrics import auroc
from trigger_audit.schemas.probes import (
    AchievedFpr,
    LayerProbeMetrics,
    PoolingStrategy,
    ProbeEvaluationResult,
    ProbeExample,
    ProbePrediction,
    ProbeSplit,
)
from trigger_audit.schemas.results import SurvivalResult

TokenProvider = Callable[[str], Sequence[int]]

# Sentinel layer_index marking the multi-layer aggregated score in LayerProbeMetrics.
AGGREGATED_LAYER_INDEX = -1

# Domain-separation tag for the seeded RNG that draws random TRIGGER_SPAN fallback windows,
# keeping that stream independent of any other seeded draw derived from the split seed.
_SPAN_FALLBACK_TAG = 0x59A


class ProbeDetectionRunner:
    """Orchestrates one probe-detection experiment end to end.

    The extractor and the token provider are injected so the same runner drives the
    offline reference backend (tests, smoke runs) and the real HF backend (measurement)
    without branching.
    """

    def __init__(
        self,
        config: ProbeDetectionExperimentConfig,
        *,
        extractor: ActivationExtractor,
        token_provider: TokenProvider | Mapping[str, Sequence[int]],
    ) -> None:
        self._config = config
        self._extractor = extractor
        if isinstance(token_provider, Mapping):
            mapping = token_provider
            self._token_provider: TokenProvider = lambda trial_id: mapping[trial_id]
        else:
            self._token_provider = token_provider
        self._store = ActivationStore(config.activations_dir)
        self._span_fallback_trial_ids: list[str] = []
        self._test_predictions: list[ProbePrediction] = []

    @property
    def test_predictions(self) -> list[ProbePrediction]:
        """Per-TEST-example predictions from the most recent :meth:`run` (empty before run)."""
        return self._test_predictions

    def run(self, examples: Sequence[ProbeExample]) -> ProbeEvaluationResult:
        """Run the full loop over pre-split examples and return the evaluation record."""
        cfg = self._config
        examples = list(examples)
        labels = np.array([e.label for e in examples], dtype=bool)
        inserted = np.array(
            [bool(e.metadata.get("trigger_inserted", False)) for e in examples], dtype=bool
        )
        # Delivered-only evaluation set: verified-delivered positives plus clean negatives.
        # An inserted-but-undelivered trial is excluded -- its label is negative but trigger
        # fragments may contaminate its activations, which would blur the question this
        # experiment asks: P(probe fires | trigger delivered) at a calibrated FPR.
        delivered = labels | ~inserted
        # Clean negatives: negatives that were never inserted. Thresholds calibrate on these
        # only -- the probe monitors CLEAN traffic, so its FPR budget is a statement about
        # clean negatives. Inserted-but-undelivered (partial-survival) trials are a third
        # population, neither clean nor delivered; mixing their trigger-contaminated scores
        # into the calibration-negative pool biases thresholds high and delivered-only TPR
        # low, worst exactly in the partial-survival regimes this repo studies.
        clean_negative = ~labels & ~inserted
        train = np.array([e.split is ProbeSplit.TRAIN for e in examples], dtype=bool)
        calibration = np.array([e.split is ProbeSplit.CALIBRATION for e in examples], dtype=bool)
        test = np.array([e.split is ProbeSplit.TEST for e in examples], dtype=bool)
        self._validate_splits(labels, delivered, clean_negative, train, calibration, test)

        # Calibration-negative pool. Default: clean (never-inserted) calibration negatives only --
        # the population the FPR budget contracts. The E0.5 ablation (calibration_include_partial)
        # widens it to every calibration negative (clean + partial-survival = calibration & ~label),
        # letting partial-survival negatives' trigger-contaminated scores bias the threshold high.
        if cfg.calibration_include_partial:
            calibration_negative = calibration & ~labels
        else:
            calibration_negative = calibration & clean_negative

        trial_ids = [e.trial_id for e in examples]
        features = self._build_features(examples, trial_ids)

        layer_metrics_all: list[LayerProbeMetrics] = []
        layer_metrics_delivered: list[LayerProbeMetrics] = []
        scores_by_layer: dict[int, np.ndarray] = {}
        for layer in cfg.layers:
            matrix = features[layer]
            probe = LinearProbe(l2=cfg.probe_l2, lr=cfg.probe_lr, max_iter=cfg.probe_max_iter)
            probe.fit(matrix[train], labels[train])
            probe.save(
                self._store.layer_dir(cfg.experiment_id, cfg.model_id)
                / f"probe_layer_{layer:03d}.npz"
            )
            scores = probe.decision_scores(matrix)
            scores_by_layer[layer] = scores

            thresholds = self._calibrate(scores[calibration_negative])
            layer_metrics_all.append(self._metrics(layer, scores[test], labels[test], thresholds))
            held_out = test & delivered
            layer_metrics_delivered.append(
                self._metrics(layer, scores[held_out], labels[held_out], thresholds)
            )

        score_matrix = np.column_stack([scores_by_layer[layer] for layer in cfg.layers])
        aggregator = AGGREGATION_REGISTRY.create(cfg.aggregation, **cfg.aggregation_params)
        # Learned aggregators (stacked_logistic) fit on the CALIBRATION split, not TRAIN. The
        # per-layer probes were fit on TRAIN, so their TRAIN scores are optimistically
        # separated relative to anything the stacker will ever see at inference; fitting the
        # stacker on those inflated scores mis-scales its weights and hides which layers are
        # actually informative. Calibration scores are held out from probe training and give
        # the stacker an honest view. (Closed-form aggregators ignore fit; see aggregation.py
        # for the mild adaptivity trade-off versus out-of-fold stacking.)
        aggregator.fit(score_matrix[calibration], labels[calibration])
        aggregated = np.asarray(aggregator.aggregate(score_matrix), dtype=np.float64)

        agg_thresholds = self._calibrate(aggregated[calibration_negative])
        aggregated_all = self._metrics(
            AGGREGATED_LAYER_INDEX, aggregated[test], labels[test], agg_thresholds
        )
        held_out = test & delivered
        aggregated_delivered = self._metrics(
            AGGREGATED_LAYER_INDEX, aggregated[held_out], labels[held_out], agg_thresholds
        )
        # Achieved FPR on ALL test negatives (deployment-pessimistic) and on CLEAN test
        # negatives (the population the budget is actually a statement about).
        achieved = self._achieved_fprs(aggregated[test & ~labels], agg_thresholds)
        achieved_clean = self._achieved_fprs(aggregated[test & clean_negative], agg_thresholds)

        # Honest per-trial predictions over the TEST split (the inference layer's input). Built
        # unconditionally and kept on the runner; run_probe_experiment writes them out only when
        # predictions_out is set, so a default run pays a negligible in-memory cost and no I/O.
        self._test_predictions = self._build_predictions(
            examples,
            test=test,
            labels=labels,
            inserted=inserted,
            delivered=delivered,
            clean_negative=clean_negative,
            aggregated=aggregated,
            scores_by_layer=scores_by_layer,
            agg_thresholds=agg_thresholds,
        )

        metadata: dict[str, Any] = {
            "n_examples": len(examples),
            "n_train": int(train.sum()),
            "n_calibration": int(calibration.sum()),
            "n_calibration_clean_negatives": int((calibration & clean_negative).sum()),
            "n_test": int(test.sum()),
            "n_test_delivered_only": int((test & delivered).sum()),
            "n_test_clean_negatives": int((test & clean_negative).sum()),
            "trigger_span_random_fallback_trial_ids": list(self._span_fallback_trial_ids),
        }
        # When layers were resolved from depth fractions, record the fractions, the concrete
        # layers they resolved to, and the model depth -- this is what lets component G report
        # results by relative depth (portable across model sizes).
        if cfg.layer_depth_fractions is not None:
            metadata["layer_depth_fractions"] = list(cfg.layer_depth_fractions)
            metadata["resolved_layers"] = list(cfg.layers)
            metadata["num_layers"] = self._extractor.num_layers

        return ProbeEvaluationResult(
            experiment_id=cfg.experiment_id,
            model_id=cfg.model_id,
            extractor_backend=cfg.extractor_backend,
            pooling=cfg.pooling,
            layers=list(cfg.layers),
            target_fprs=list(cfg.target_fprs),
            layer_metrics_all=layer_metrics_all,
            layer_metrics_delivered_only=layer_metrics_delivered,
            aggregation=cfg.aggregation,
            aggregated_metrics_all=aggregated_all,
            aggregated_metrics_delivered_only=aggregated_delivered,
            achieved_fprs=achieved,
            achieved_fprs_clean=achieved_clean,
            metadata=metadata,
        )

    def _build_predictions(
        self,
        examples: Sequence[ProbeExample],
        *,
        test: np.ndarray,
        labels: np.ndarray,
        inserted: np.ndarray,
        delivered: np.ndarray,
        clean_negative: np.ndarray,
        aggregated: np.ndarray,
        scores_by_layer: dict[int, np.ndarray],
        agg_thresholds: dict[str, float],
    ) -> list[ProbePrediction]:
        """Emit one honest :class:`ProbePrediction` per TEST example (no cross-trial mixing).

        ``delivered`` and ``clean_negative`` reuse the runner's canonical memberships
        (``label | ~inserted`` and ``~label & ~inserted``); ``fired[target]`` is exactly
        ``aggregated_score >= agg_thresholds[target]`` so it agrees with the aggregated metrics
        computed at the same operating points.
        """
        cfg = self._config
        predictions: list[ProbePrediction] = []
        for index, example in enumerate(examples):
            if not test[index]:
                continue
            score = float(aggregated[index])
            predictions.append(
                ProbePrediction(
                    trial_id=example.trial_id,
                    base_id=example.base_id,
                    label=bool(labels[index]),
                    trigger_inserted=bool(inserted[index]),
                    delivered=bool(delivered[index]),
                    clean_negative=bool(clean_negative[index]),
                    split=example.split,
                    aggregated_score=score,
                    layer_scores={
                        str(layer): float(scores_by_layer[layer][index]) for layer in cfg.layers
                    },
                    fired={
                        target: bool(score >= threshold)
                        for target, threshold in agg_thresholds.items()
                    },
                )
            )
        return predictions

    def _build_features(
        self, examples: Sequence[ProbeExample], trial_ids: Sequence[str]
    ) -> dict[int, np.ndarray]:
        """Return one pooled ``(n_examples, hidden_size)`` matrix per configured layer.

        With ``reuse_store`` enabled this reuses any stored layer whose trial-id vector (in
        order) and producer metadata match, extracting and saving only the layers that miss --
        the extract-once, pool-many optimization. With ``reuse_store`` disabled every layer is
        extracted and saved, byte-for-byte the original behavior (the store key now carries the
        pooling name, so a pooling sweep no longer overwrites itself).
        """
        cfg = self._config
        pooling = cfg.pooling.value
        features: dict[int, np.ndarray] = {}
        layers_to_extract: list[int] = []
        for layer in cfg.layers:
            if cfg.reuse_store:
                cached = self._store.load_reusable(
                    cfg.experiment_id,
                    cfg.model_id,
                    layer,
                    pooling,
                    trial_ids,
                    extractor_backend=cfg.extractor_backend,
                )
                if cached is not None:
                    features[layer] = cached
                    continue
            layers_to_extract.append(layer)

        if layers_to_extract:
            extracted = self._extract_and_pool(examples, layers_to_extract)
            for layer in layers_to_extract:
                matrix = extracted[layer]
                self._store.save(
                    cfg.experiment_id,
                    cfg.model_id,
                    layer,
                    matrix,
                    trial_ids,
                    pooling,
                    extractor_backend=cfg.extractor_backend,
                )
                features[layer] = matrix
        return features

    def _extract_and_pool(
        self, examples: Sequence[ProbeExample], layers: Sequence[int]
    ) -> dict[int, np.ndarray]:
        """Extract and pool activations for ``layers``: one matrix per layer, rows = examples."""
        cfg = self._config
        layers = list(layers)
        rows: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
        fallback_trial_ids: list[str] = []
        # For TRIGGER_SPAN pooling, spanless examples are pooled over a random span of the
        # median trigger length rather than mean-pooled, so the pooling OPERATOR is identical
        # across every class and only trigger CONTENT (not the window statistics) can drive
        # separation. Not every example carries a span: clean negatives never do, and -- note
        # -- partial-survival NEGATIVES do (Project 1's scorer localizes the surviving
        # fragment), so this is not simply "negatives lack spans".
        fallback_len = self._median_span_length(examples)
        for index, example in enumerate(examples):
            token_ids = list(self._token_provider(example.trial_id))
            activations = self._extractor.extract(token_ids, layers)
            span = example.trigger_span()
            pool_strategy = cfg.pooling
            pool_span = span
            if cfg.pooling is PoolingStrategy.TRIGGER_SPAN and span is None:
                if cfg.span_random_fallback:
                    pool_span = self._random_span(len(token_ids), fallback_len, index)
                    fallback_trial_ids.append(example.trial_id)
                else:
                    # E0.4 confound (fallback disabled): a spanless example is mean-pooled over
                    # the whole sequence instead of over a matched random window. Span-carrying
                    # examples still get the short trigger window, so the two populations now
                    # differ in pooling OPERATOR (short-window vs full-sequence statistics) even
                    # with no trigger content -- the illusory separation this ablation exposes.
                    pool_strategy = PoolingStrategy.MEAN
                    pool_span = None
            for layer in layers:
                pooled = pool_activations(activations[layer], pool_strategy, span=pool_span)
                rows[layer].append(pooled)
        self._span_fallback_trial_ids = fallback_trial_ids
        return {layer: np.stack(layer_rows) for layer, layer_rows in rows.items()}

    @staticmethod
    def _median_span_length(examples: Sequence[ProbeExample]) -> int:
        """Median length of the trigger spans present, defaulting to 1 when none exist."""
        lengths = [end - start for e in examples if (s := e.trigger_span()) for start, end in (s,)]
        return int(np.median(lengths)) if lengths else 1

    def _random_span(self, n_tokens: int, length: int, index: int) -> tuple[int, int]:
        """Pick a deterministic random span of the given length within ``n_tokens``.

        Seeded from the split seed and the example index so the fallback is reproducible run
        to run (a requirement of the pre-registered grid) yet decorrelated from the label.
        """
        length = max(1, min(length, n_tokens))
        rng = np.random.default_rng((self._config.split_seed, _SPAN_FALLBACK_TAG, index))
        start = int(rng.integers(0, n_tokens - length + 1))
        return (start, start + length)

    def _calibrate(self, calibration_negative_scores: np.ndarray) -> dict[str, float]:
        """Calibrate one threshold per target FPR on clean CALIBRATION-negative scores.

        The scores passed in are those of the clean (never-inserted) negatives in the
        calibration split -- see ``run`` for why partial-survival negatives are excluded.
        """
        thresholds: dict[str, float] = {}
        for target in self._config.target_fprs:
            result = calibrate_threshold(calibration_negative_scores, target)
            thresholds[str(target)] = result.threshold
        return thresholds

    def _metrics(
        self,
        layer_index: int,
        scores: np.ndarray,
        labels: np.ndarray,
        thresholds: dict[str, float],
    ) -> LayerProbeMetrics:
        """Evaluate one score set at the calibrated thresholds (deployment operating points)."""
        positives = scores[labels]
        return LayerProbeMetrics(
            layer_index=layer_index,
            auroc=auroc(scores, labels),
            tpr_at_target_fpr={
                key: float(np.mean(positives >= threshold)) for key, threshold in thresholds.items()
            },
            threshold=dict(thresholds),
            n_pos=int(labels.sum()),
            n_neg=int((~labels).sum()),
        )

    def _achieved_fprs(
        self, test_negative_scores: np.ndarray, thresholds: dict[str, float]
    ) -> list[AchievedFpr]:
        """Empirical FPR of each calibrated threshold on TEST negatives, with Wilson 95% CI."""
        achieved: list[AchievedFpr] = []
        n = int(test_negative_scores.size)
        for target in self._config.target_fprs:
            threshold = thresholds[str(target)]
            false_positives = int(np.sum(test_negative_scores >= threshold))
            low, high = wilson_interval(false_positives, n)
            achieved.append(
                AchievedFpr(
                    target_fpr=target,
                    achieved_fpr=false_positives / n if n else 0.0,
                    ci_low=low,
                    ci_high=high,
                    n_negatives=n,
                )
            )
        return achieved

    @staticmethod
    def _validate_splits(
        labels: np.ndarray,
        delivered: np.ndarray,
        clean_negative: np.ndarray,
        train: np.ndarray,
        calibration: np.ndarray,
        test: np.ndarray,
    ) -> None:
        """Fail fast with a clear message when a split cannot support its role."""
        if not (np.any(train & labels) and np.any(train & ~labels)):
            raise ValueError("TRAIN split needs at least one positive and one negative example")
        if not np.any(calibration & clean_negative):
            raise ValueError(
                "CALIBRATION split needs at least one clean (never-inserted) negative example: "
                "thresholds calibrate on clean negatives only, so a calibration split made up "
                "entirely of inserted negatives cannot set an FPR budget for clean traffic"
            )
        if not (np.any(test & labels) and np.any(test & ~labels)):
            raise ValueError("TEST split needs at least one positive and one negative example")
        if not np.any(test & delivered & ~labels):
            raise ValueError(
                "delivered-only evaluation needs at least one clean (never-inserted) negative "
                "in the TEST split"
            )


def run_probe_experiment(config: ProbeDetectionExperimentConfig) -> ProbeEvaluationResult:
    """Load the dataset per config, run the experiment, persist and return the result."""
    extractor = make_activation_extractor(
        config.extractor_backend,
        model_id=config.model_id,
        seed=config.extractor_seed,
        hidden_size=config.extractor_hidden_size,
        num_layers=config.extractor_num_layers,
        device=config.device,
        revision=config.revision,
        trust_remote_code=config.trust_remote_code,
    )
    # Depth-fraction slicing: once the extractor is built, num_layers is known, so resolve the
    # configured fractions to concrete layers at the SAME relative depth (portable across model
    # sizes) and run on those. The resolved layers are unique/sorted/in-range by construction.
    if config.layer_depth_fractions is not None:
        resolved_layers = resolve_layers_from_fractions(
            config.layer_depth_fractions, extractor.num_layers
        )
        config = config.model_copy(update={"layers": resolved_layers})
    # Fail fast on an out-of-range layer right after the (possibly expensive) extractor is
    # built, rather than after a full forward pass. Non-negativity and uniqueness are already
    # guaranteed by the config validator; only the upper bound depends on the loaded model.
    if max(config.layers) > extractor.num_layers:
        raise ValueError(
            f"config requests layer {max(config.layers)} but the "
            f"{config.extractor_backend!r} extractor has only {extractor.num_layers} "
            f"transformer block(s) (valid layer indices are 0..{extractor.num_layers})"
        )

    tokens: dict[str, list[int]]
    if config.survival_results_path is not None:
        if config.final_tokens_path is None:
            raise ValueError(
                "final_tokens_path is required with survival_results_path: probes read the "
                "final token ids the survival audit verified delivery into"
            )
        survival = read_jsonl_as(config.survival_results_path, SurvivalResult)
        examples = build_probe_examples(survival)
        tokens = {
            str(row["trial_id"]): [int(t) for t in row["final_token_ids"]]
            for row in iter_jsonl(config.final_tokens_path)
        }
    elif config.synthetic_mode == "twins":
        examples, tokens = build_synthetic_probe_dataset_with_twins(
            n_bases=config.synthetic_n_bases,
            seq_len=config.synthetic_seq_len,
            partial_survival_fraction=config.partial_survival_fraction,
            seed=config.synthetic_seed,
        )
    else:
        examples, tokens = build_synthetic_probe_dataset(
            n_examples=config.synthetic_n_examples,
            seq_len=config.synthetic_seq_len,
            seed=config.synthetic_seed,
        )

    if config.generalization is not None:
        # E2.x holdout: hold out one side as TEST, carve a base_id-grouped calibration subset
        # from the train side, and drop un-held-out rows (an unmodeled third population). The
        # holdout is intrinsically base_id-grouped, so split_mode does not apply here.
        examples = assign_generalization_splits(
            examples,
            config.generalization,
            calibration_fraction=config.calibration_fraction,
            seed=config.split_seed,
        )
    elif config.split_mode == "example":
        # E0.3 leakage ablation: split examples independently, ignoring base_id, so counterfactual
        # twins straddle the train/test line (the deliberately leaky control).
        examples = assign_splits_example_level(
            examples,
            train_fraction=config.train_fraction,
            calibration_fraction=config.calibration_fraction,
            seed=config.split_seed,
        )
    else:
        examples = assign_splits(
            examples,
            train_fraction=config.train_fraction,
            calibration_fraction=config.calibration_fraction,
            seed=config.split_seed,
        )
    runner = ProbeDetectionRunner(config, extractor=extractor, token_provider=tokens)
    result = runner.run(examples)
    if config.predictions_out is not None:
        write_jsonl(config.predictions_out, runner.test_predictions)
        result.metadata["predictions_out"] = str(config.predictions_out)
    write_jsonl(config.results_out, [result])
    return result
