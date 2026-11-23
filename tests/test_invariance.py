"""Edits that change every byte but not the function must not move the fingerprint."""
import pytest

from conftest import attn_distance
from lineage_check import edits

SAME_FUNCTION = ["perm_mlp", "perm_heads", "scale_mlp", "vo_transform", "qk_transform", "rotate_residual", "obfuscate"]
PADDING = [("insert_dummy_layers", {"positions": (1,)}), ("pad_heads", {}), ("widen_mlp", {"extra": 16}),
           ("widen_residual", {"extra": 8})]


@pytest.mark.parametrize("name", SAME_FUNCTION)
def test_reparameterisation_leaves_fingerprint_unchanged(model, name):
    assert abs(attn_distance(model, getattr(edits, name)(model))) < 1e-9


@pytest.mark.parametrize("name,kwargs", PADDING)
def test_padding_leaves_fingerprint_unchanged(model, name, kwargs):
    padded = getattr(edits, name)(model, **kwargs)
    assert abs(attn_distance(model, padded, align=True)) < 1e-9
