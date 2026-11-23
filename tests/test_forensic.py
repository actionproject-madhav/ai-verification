"""The refusal-ablation signature: one residual direction projected out of every writer, still read by readers."""
import numpy as np

from conftest import VOCAB
from lineage_check import edits, forensic, remote


def verdict(model):
    m = remote.from_canon(model, lo=0, hi=VOCAB)
    writers = [np.concatenate([l["o"], l["down"]], 1) for l in m["layers"].values()]
    readers = [forensic.layer_readers(l) for l in m["layers"].values()]
    return forensic.verdict(writers + [m["E"].T], readers + [None])


def test_untouched_model_is_not_flagged(model):
    assert verdict(model)[0] is False


def test_ablated_model_is_flagged_with_one_direction(model):
    flagged, dims = verdict(edits.rank1_edit(model))[:2]
    assert flagged is True and dims == 1


def test_signature_survives_rotation_and_permutation(model):
    assert verdict(edits.obfuscate(edits.rank1_edit(model)))[0] is True


def test_zero_padding_is_not_mistaken_for_ablation(model):
    assert verdict(edits.widen_residual(model, extra=8))[0] is False
