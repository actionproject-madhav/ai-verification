"""Realistic edits stay close to the parent; a separately initialised model does not."""
from conftest import attn_distance, tiny_model
from lineage_check import edits


def test_independent_model_is_far(model):
    assert attn_distance(model, tiny_model(1)) > 0.8


def test_small_weight_noise_stays_close(model):
    assert attn_distance(model, edits.noise(model, rel=0.01)) < 0.01


def test_four_bit_quantisation_stays_close(model):
    assert attn_distance(model, edits.quantize(model, bits=4, group=16)) < 0.05


def test_pruning_stays_close(model):
    assert attn_distance(model, edits.prune(model, frac=0.3)) < 0.05


def test_refusal_style_rank_one_edit_stays_close(model):
    assert attn_distance(model, edits.rank1_edit(model)) < 0.05


def test_obfuscated_quantised_copy_stays_close(model):
    assert attn_distance(model, edits.obfuscate(edits.quantize(model, bits=4, group=16))) < 0.05
