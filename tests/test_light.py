"""The traffic light: flagged by one matching part, cleared only when every part matches."""
import copy
import itertools

import pytest
import torch

from conftest import TinySource, tiny_model
from lineage_check import edits, light

WORDS = ["".join(p) for p in itertools.product("abcdefghij", repeat=3)][:100]
NONCE = "ab" * 16


class Source(TinySource):
    def __init__(self, model):
        super().__init__(model)
        self.vocab = {w: 300 + i for i, w in enumerate(WORDS)}


def mixed(base, other, parts):
    """`base` with the named weights of every layer taken from `other`."""
    m = copy.deepcopy(base)
    for layer, donor in zip(m["layers"], other["layers"]):
        for k in parts:
            layer[k] = donor[k].clone()
    return m


ATTENTION, MLP = ("q", "k", "v", "o", "g1"), ("gate", "up", "down", "g2")
LISTED, KNOWN, OTHER = tiny_model(0), tiny_model(1), tiny_model(2)
WATCH, REGISTRY = [({"id": "listed"}, Source(LISTED))], [({"id": "known"}, Source(KNOWN))]


def answer(model, **kw):
    r = light.light(Source(model), WATCH, REGISTRY, NONCE, **kw)
    return r["light"], r["matched"]


@pytest.mark.parametrize("copy_of_listed", [edits.noise(LISTED, rel=0.05), edits.quantize(LISTED), edits.obfuscate(LISTED)])
def test_a_modified_copy_of_a_listed_model_is_red(copy_of_listed):
    assert answer(copy_of_listed) == ("red", "listed")


@pytest.mark.parametrize("copy_of_known", [KNOWN, edits.quantize(KNOWN), edits.noise(KNOWN, rel=0.02)])
def test_a_close_copy_of_a_known_model_is_green(copy_of_known):
    assert answer(copy_of_known) == ("green", "known")


def test_a_model_on_neither_list_is_amber():
    assert answer(OTHER) == ("amber", None)


def test_one_matching_part_is_enough_to_be_flagged():
    assert answer(mixed(LISTED, OTHER, MLP)) == ("red", "listed")          # the listed model's attention, other MLPs
    assert answer(mixed(LISTED, OTHER, ATTENTION)) == ("red", "listed")    # the listed model's MLPs, other attention


def test_one_matching_part_is_not_enough_to_be_cleared():
    assert answer(mixed(KNOWN, OTHER, MLP)) == ("amber", None)             # a known model's attention over other MLPs


def test_the_refusal_ablation_signature_is_red_whatever_the_origin():
    assert answer(OTHER, ablation=True) == ("red", None)
    assert answer(KNOWN, ablation=True) == ("red", None)


def test_the_layers_read_depend_on_the_nonce_and_line_up_across_models():
    deep = Source(tiny_model(3))
    picks = {tuple(light.layers_for(deep, f"{i:032x}")) for i in range(40)}
    assert len(picks) > 1                                                   # not a fixed, public pair of layers
    assert all(light.layers_for(deep, n) == light.layers_for(Source(tiny_model(4)), n) for n in ("aa" * 16, "bb" * 16))


def test_a_model_the_reader_cannot_decode_is_amber_not_an_error():
    packed = Source(KNOWN)
    packed.index = {n: ("file", "U32", packed.tensors[n].shape, 0, 0) if ".layers." in n else None for n in packed.tensors}
    assert light.layers_for(packed, NONCE) == []
    r = light.light(packed, WATCH, REGISTRY, NONCE)
    assert (r["light"], r["matched"]) == ("amber", None)


def stretched(model):
    """The model with a token table that has distinct principal axes, as real token tables do."""
    m = copy.deepcopy(model)
    m["embed"] = m["embed"] * torch.linspace(3.0, 0.3, m["d"])
    return m


def relabelled(model, seed=3):
    """The model with its token rows shuffled, behind a tokenizer file that still names the old rows."""
    m = copy.deepcopy(model)
    m["embed"] = m["embed"][torch.randperm(m["embed"].shape[0], generator=torch.Generator().manual_seed(seed))]
    return m


LISTED2, KNOWN2, OTHER2 = stretched(LISTED), stretched(KNOWN), stretched(OTHER)
WATCH2, REGISTRY2 = [({"id": "listed"}, Source(LISTED2))], [({"id": "known"}, Source(KNOWN2))]


def answer2(model, **kw):
    r = light.light(Source(model), WATCH2, REGISTRY2, NONCE, **kw)
    return r["light"], r["matched"]


@pytest.mark.parametrize("edit", [lambda m: m, lambda m: edits.noise(m, rel=0.02), edits.quantize, edits.rotate_residual])
def test_relabelling_the_tokenizer_hides_a_copy_from_names_and_not_from_geometry(edit):
    hidden = relabelled(edit(LISTED2))
    assert answer2(hidden) == ("amber", None)
    assert answer2(hidden, geometry=True) == ("red", "listed")


def test_geometry_matching_does_not_flag_an_unrelated_model():
    assert answer2(OTHER2, geometry=True) == ("amber", None)
    assert answer2(relabelled(OTHER2), geometry=True) == ("amber", None)


def test_geometry_matching_leaves_an_honest_known_copy_green():
    assert answer2(edits.noise(KNOWN2, rel=0.02), geometry=True) == ("green", "known")
