"""Tests for depth-fraction model slicing (``activations/slicing.py``)."""

from __future__ import annotations

import pytest

from trigger_audit.activations.slicing import (
    INFORMATIVE_BAND_FRACTIONS,
    default_band_layers,
    depth_fraction_of_layer,
    resolve_layers_from_fractions,
)


def test_boundary_fractions_map_to_embedding_and_last_layer():
    assert resolve_layers_from_fractions([0.0], num_layers=32) == [0]
    assert resolve_layers_from_fractions([1.0], num_layers=32) == [32]


def test_midpoint_fraction_maps_to_half_depth():
    assert resolve_layers_from_fractions([0.5], num_layers=32) == [16]


def test_fraction_rounds_to_nearest_block():
    # 0.66 * 32 = 21.12 -> 21; 0.89 * 32 = 28.48 -> 28.
    assert resolve_layers_from_fractions([0.66], num_layers=32) == [21]
    assert resolve_layers_from_fractions([0.89], num_layers=32) == [28]


def test_duplicate_and_unsorted_fractions_collapse_to_unique_sorted_indices():
    layers = resolve_layers_from_fractions([0.75, 0.5, 0.5, 0.75], num_layers=32)
    assert layers == [16, 24]


def test_distinct_nearby_fractions_that_round_together_collapse():
    # 0.331 * 100 = 33.1 and 0.334 * 100 = 33.4 both round to block 33.
    assert resolve_layers_from_fractions([0.331, 0.334], num_layers=100) == [33]


def test_empty_fractions_rejected():
    with pytest.raises(ValueError):
        resolve_layers_from_fractions([], num_layers=32)


@pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0, -1.0])
def test_out_of_range_fraction_rejected(bad):
    with pytest.raises(ValueError):
        resolve_layers_from_fractions([0.5, bad], num_layers=32)


def test_non_positive_num_layers_rejected():
    with pytest.raises(ValueError):
        resolve_layers_from_fractions([0.5], num_layers=0)


def test_depth_fraction_of_layer_boundaries():
    assert depth_fraction_of_layer(0, num_layers=32) == 0.0
    assert depth_fraction_of_layer(32, num_layers=32) == 1.0
    assert depth_fraction_of_layer(16, num_layers=32) == 0.5


@pytest.mark.parametrize("num_layers", [28, 32, 40])
@pytest.mark.parametrize("layer", [0, 7, 13, 16, 24])
def test_depth_fraction_round_trips_through_resolve(num_layers, layer):
    if layer > num_layers:
        pytest.skip("layer must be within the model")
    fraction = depth_fraction_of_layer(layer, num_layers)
    assert resolve_layers_from_fractions([fraction], num_layers) == [layer]


def test_depth_fraction_of_layer_out_of_range_rejected():
    with pytest.raises(ValueError):
        depth_fraction_of_layer(33, num_layers=32)
    with pytest.raises(ValueError):
        depth_fraction_of_layer(-1, num_layers=32)


def test_depth_fraction_of_layer_non_positive_num_layers_rejected():
    with pytest.raises(ValueError):
        depth_fraction_of_layer(0, num_layers=0)


def test_informative_band_constant_is_pre_registered_band():
    assert INFORMATIVE_BAND_FRACTIONS == (0.5, 0.66, 0.75, 0.89)


@pytest.mark.parametrize("num_layers", [12, 24, 28, 32, 40, 64])
def test_default_band_layers_are_unique_sorted_and_in_range(num_layers):
    layers = default_band_layers(num_layers)
    assert layers == sorted(layers)
    assert len(layers) == len(set(layers))
    assert all(0 <= layer <= num_layers for layer in layers)
    assert layers == resolve_layers_from_fractions(INFORMATIVE_BAND_FRACTIONS, num_layers)


def test_default_band_layers_concrete_example():
    # 0.5*32=16, 0.66*32=21.12->21, 0.75*32=24, 0.89*32=28.48->28.
    assert default_band_layers(32) == [16, 21, 24, 28]
