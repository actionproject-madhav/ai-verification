"""The design-free fingerprint: no knowledge of the architecture, matrices matched by content."""
import numpy as np
import pytest
import torch

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


def test_tiktoken_vocabulary_is_read_with_byte_level_spelling():
    import base64
    raw = b"\n".join(base64.b64encode(t) + b" " + str(i).encode() for i, t in enumerate([b"the", b" the", b"\n", b"\xff\xfe"]))
    assert remote._vocab_from_tiktoken(raw) == {"the": 0, "\u0120the": 1, "\u010a": 2}


def test_models_with_no_shared_tokens_are_not_comparable(model):
    vocab = {f"t{i}": i for i in range(400)}
    f = universal.fingerprint(TinySource(model), vocab, [f"t{i}" for i in range(256)], lo=0, hi=400)
    g = universal.fingerprint(TinySource(model), vocab, [f"other{i}" for i in range(256)], lo=0, hi=400)
    assert universal.compare(f, g) is None


def test_a_lookup_table_inside_a_layer_is_not_taken_for_the_vocabulary(model):
    src = TinySource(model)
    src.tensors["model.layers.1.engram.embed.weight"] = np.zeros((5000, 8), dtype=np.float32)
    src.index = dict.fromkeys(src.tensors)
    emb, d, lay = universal.layout(src)
    assert emb == "model.embed_tokens.weight" and d == model["embed"].shape[1]
    assert all("engram" not in n for L in lay.values() for n, _ in L["mats"])


def test_packed_giant_tensors_are_skipped(model, monkeypatch):
    src = TinySource(model)
    monkeypatch.setattr(universal, "MAX_ELEMS", 48 * 48)          # the attention matrices fit, the MLP matrices do not
    _, d, lay = universal.layout(src)
    names = [n for L in lay.values() for n, _ in L["mats"]]
    assert names and all("mlp" not in n for n in names)


def cuts(changed, base):
    return universal.cut_layers(TinySource(changed), TinySource(base), [0, 1, 2], [0, 1, 2])


def test_a_direction_removed_from_every_writer_shows_as_a_cut_in_every_layer(model):
    assert cuts(edits.rank1_edit(model), model) == {"cut": 3, "changed": 3, "aligned": True}


def test_the_cut_still_shows_under_fine_tuning_noise(model):
    assert cuts(edits.noise(edits.rank1_edit(model), rel=0.01), model)["cut"] == 3


def test_a_different_direction_in_every_layer_is_still_a_cut_in_every_layer(model):
    m = edits.clone(model)
    g = torch.Generator().manual_seed(5)
    for layer in m["layers"]:                                     # newer tools remove a direction of its own per layer
        r = torch.randn(m["d"], generator=g)
        P = torch.eye(m["d"]) - torch.outer(r, r) / (r @ r)
        layer["o"], layer["down"] = P @ layer["o"], P @ layer["down"]
    assert cuts(m, model) == {"cut": 3, "changed": 3, "aligned": True}


def test_a_cut_in_only_one_matrix_per_layer_shows_as_a_rank_one_change(model):
    m = edits.clone(model)
    r = torch.randn(m["d"], generator=torch.Generator().manual_seed(6))
    P = torch.eye(m["d"]) - 0.6 * torch.outer(r, r) / (r @ r)    # partial strength, attention output only
    for layer in m["layers"]:
        layer["o"] = P @ layer["o"]
    assert cuts(m, model) == {"cut": 3, "changed": 3, "aligned": True}


@pytest.mark.parametrize("edit", [lambda m: edits.noise(m, rel=0.02), edits.quantize, lambda m: edits.prune(m, frac=0.3)])
def test_noise_quantisation_and_pruning_cut_nothing(model, edit):
    assert cuts(edit(model), model) == {"cut": 0, "changed": 3, "aligned": True}


def test_an_unchanged_copy_has_no_difference_to_read(model):
    assert cuts(model, model) == {"cut": 0, "changed": 0, "aligned": True}


@pytest.mark.parametrize("name", ["perm_mlp", "perm_heads", "rotate_residual"])
def test_a_shuffled_or_rotated_copy_is_not_aligned_so_the_cut_test_says_nothing(model, name):
    hidden = getattr(edits, name)(edits.rank1_edit(model))       # a cut, then a free transform that hides it entry by entry
    r = cuts(hidden, model)
    assert r["aligned"] is False


def test_a_layer_with_other_matrices_cannot_be_compared(model):
    other = TinySource(model)
    other.tensors = {n.replace("o_proj", "out_matrix"): v for n, v in other.tensors.items()}
    other.index = dict.fromkeys(other.tensors)
    assert universal.cut_layers(other, TinySource(model), [0, 1], [0, 1]) is None


def test_models_of_different_width_are_not_compared_for_an_edit(model):
    other = TinySource(model)
    other.tensors = {n: (v[:, :40] if v.ndim == 2 and v.shape[1] == 48 else v[:40] if v.shape[0] == 48 else v) for n, v in other.tensors.items()}
    other.index = dict.fromkeys(other.tensors)
    assert universal.cut_layers(other, TinySource(model), [0, 1], [0, 1]) is None
