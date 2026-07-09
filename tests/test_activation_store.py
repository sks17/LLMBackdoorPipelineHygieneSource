"""Tests for the activation store's (layer, pooling) keying and producer-metadata reuse guard.

The store persists one pooled feature matrix per (experiment, model, layer, pooling). Two
poolings of the same layer must be distinct entries (a pooling sweep must not overwrite
itself), and ``load_reusable`` must refuse to hand back a matrix produced by a different
extractor backend or model (the backend is not part of the path, so it is the only thing
standing between a reference-produced matrix and an hf run reusing it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trigger_audit.activations.store import ActivationStore

TRIAL_IDS = ["t0", "t1", "t2"]


def _matrix(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((3, 4)).astype(np.float32)


def test_save_load_round_trip_default_pooling(tmp_path: Path) -> None:
    store = ActivationStore(tmp_path)
    matrix = _matrix(0)
    store.save("exp", "model", 2, matrix, TRIAL_IDS)  # pooling defaults to "mean"
    loaded, trial_ids = store.load("exp", "model", 2)
    np.testing.assert_array_equal(loaded, matrix)
    assert trial_ids == TRIAL_IDS


def test_pooling_is_part_of_the_filename(tmp_path: Path) -> None:
    store = ActivationStore(tmp_path)
    mean_path = store.features_path("exp", "model", 3, "mean")
    max_path = store.features_path("exp", "model", 3, "max")
    assert mean_path != max_path
    assert mean_path.name == "layer_003_mean.npz"
    assert max_path.name == "layer_003_max.npz"


def test_two_poolings_of_same_layer_do_not_collide(tmp_path: Path) -> None:
    store = ActivationStore(tmp_path)
    mean_matrix = _matrix(1)
    max_matrix = _matrix(2)
    store.save("exp", "model", 3, mean_matrix, TRIAL_IDS, "mean")
    store.save("exp", "model", 3, max_matrix, TRIAL_IDS, "max")

    loaded_mean, _ = store.load("exp", "model", 3, "mean")
    loaded_max, _ = store.load("exp", "model", 3, "max")
    np.testing.assert_array_equal(loaded_mean, mean_matrix)
    np.testing.assert_array_equal(loaded_max, max_matrix)
    # The second pooling did not clobber the first.
    assert not np.array_equal(loaded_mean, loaded_max)


def test_load_reusable_returns_matrix_on_full_match(tmp_path: Path) -> None:
    store = ActivationStore(tmp_path)
    matrix = _matrix(3)
    store.save("exp", "model", 1, matrix, TRIAL_IDS, "mean", extractor_backend="reference")
    reused = store.load_reusable(
        "exp", "model", 1, "mean", TRIAL_IDS, extractor_backend="reference"
    )
    assert reused is not None
    np.testing.assert_array_equal(reused, matrix)


def test_load_reusable_refuses_missing_entry(tmp_path: Path) -> None:
    store = ActivationStore(tmp_path)
    assert (
        store.load_reusable("exp", "model", 9, "mean", TRIAL_IDS, extractor_backend="reference")
        is None
    )


def test_load_reusable_refuses_backend_mismatch(tmp_path: Path) -> None:
    store = ActivationStore(tmp_path)
    store.save("exp", "model", 1, _matrix(4), TRIAL_IDS, "mean", extractor_backend="reference")
    # Same experiment/model/layer/pooling and trial ids, but a different producing backend.
    assert store.load_reusable("exp", "model", 1, "mean", TRIAL_IDS, extractor_backend="hf") is None


def test_load_reusable_refuses_trial_id_order_change(tmp_path: Path) -> None:
    store = ActivationStore(tmp_path)
    store.save("exp", "model", 1, _matrix(5), TRIAL_IDS, "mean", extractor_backend="reference")
    reordered = list(reversed(TRIAL_IDS))
    assert (
        store.load_reusable("exp", "model", 1, "mean", reordered, extractor_backend="reference")
        is None
    )


def test_load_reusable_refuses_pooling_mismatch(tmp_path: Path) -> None:
    store = ActivationStore(tmp_path)
    store.save("exp", "model", 1, _matrix(6), TRIAL_IDS, "mean", extractor_backend="reference")
    # A different pooling is a different key -> no entry -> refuse.
    assert (
        store.load_reusable("exp", "model", 1, "max", TRIAL_IDS, extractor_backend="reference")
        is None
    )
