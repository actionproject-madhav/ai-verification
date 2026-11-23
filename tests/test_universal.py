"""The design-free fingerprint: no knowledge of the architecture, matrices matched by content."""
import numpy as np
import pytest

from conftest import TinySource, tiny_model, universal_distance
from lineage_check import edits, remote, universal

EXACT = ["perm_mlp", "perm_heads", "scale_mlp", "rotate_residual"]


@pytest.mark.parametrize("name", EXACT)
def test_permutation_rescaling_and_rotation_leave_it_unchanged(model, name):
    assert abs(universal_distance(model, getattr(edits, name)(model))) < 1e-9


def test_every_free_transform_together_stays_a_match(model):
    assert universal_distance(model, edits.obfuscate(model)) < 0.05


def test_independent_model_is_far(model):
    assert universal_distance(model, tiny_model(1)) > 0.8


@pytest.mark.parametrize("edit", [lambda m: edits.noise(m, rel=0.01), lambda m: edits.quantize(m, bits=4, group=16),
                                  lambda m: edits.prune(m, frac=0.3), edits.rank1_edit])
def test_realistic_edits_stay_close(model, edit):
    assert universal_distance(model, edit(model)) < 0.05


def test_layout_is_found_from_shapes_alone(model):
    emb, d, layers = universal.layout(TinySource(model))
    assert emb == "model.embed_tokens.weight" and d == model["d"] and sorted(layers) == [0, 1, 2]
    kinds = {n.split(".")[-2]: k for n, k in layers[0]["mats"]}
    assert kinds == {"q_proj": "r", "k_proj": "r", "v_proj": "r", "o_proj": "w", "gate_proj": "r", "up_proj": "r", "down_proj": "w"}


def test_eight_bit_floats_decode():
    assert [float(remote._E4M3[b]) for b in (0x00, 0x38, 0x3C, 0x7E, 0xB8)] == [0.0, 1.0, 1.5, 448.0, -1.0]
    assert float(remote._E8M0[127]) == 1.0 and float(remote._E8M0[130]) == 8.0
