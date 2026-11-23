"""The output check: Green needs the outputs to agree with the known model's, not only the weights."""
import types

import numpy as np
import torch

from conftest import tiny_model
from lineage_check import light, outputs
from test_light import NONCE, Source

VOCAB_OUT = 50


class Toy(torch.nn.Module):
    """A stand-in language model: token id -> a distribution over the next token."""

    def __init__(self, seed, noise=0.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.table = torch.randn(400, VOCAB_OUT, generator=g) + noise * torch.randn(400, VOCAB_OUT, generator=torch.Generator().manual_seed(99))

    def forward(self, ids):
        return types.SimpleNamespace(logits=self.table[ids])


IDS = torch.randint(0, 400, (6, 16), generator=torch.Generator().manual_seed(3))


def test_distance_is_zero_for_the_same_model_and_grows_with_the_change():
    y = Toy(0)
    assert outputs.distance(y, Toy(0), IDS).max() == 0
    near, far = outputs.distance(Toy(0, noise=0.05), y, IDS), outputs.distance(Toy(1), y, IDS)
    assert near.shape == (6,) and 0 < near.mean() < 0.1 < far.mean() <= 1


def test_a_model_close_to_one_model_cannot_be_close_to_another():
    x, y, m = Toy(1), Toy(0), Toy(0, noise=0.05)                    # m is a faithful copy of y
    gap, to_y, to_x = outputs.distance(x, y, IDS), outputs.distance(m, y, IDS), outputs.distance(m, x, IDS)
    assert (to_x >= gap - to_y - 1e-6).all()                        # the triangle inequality, in every window


def test_distance_matches_the_definition():
    a, b = Toy(4), Toy(5)
    p, q = torch.softmax(a.table[IDS[:1]], -1), torch.softmax(b.table[IDS[:1]], -1)
    assert abs(float((0.5 * (p - q).abs().sum(-1)).mean()) - outputs.distance(a, b, IDS)[0]) < 1e-6


def test_rows_back_undoes_the_interleaving():
    heads, half, width = 3, 4, 5
    w = np.arange(heads * 2 * half * width, dtype=np.float32).reshape(heads * 2 * half, width)
    stored = w.reshape(heads, 2, half, width).swapaxes(1, 2).reshape(w.shape)       # what llama.cpp writes
    assert (outputs.rows_back(stored, heads) == w).all()


def test_green_needs_the_outputs_to_agree():
    base = tiny_model(0)
    known = [({"id": "known"}, Source(base))]
    calls = []

    def answer(result):
        def check(entry):
            calls.append(entry["id"])
            return result
        return check

    agree = {"windows": 4, "tokens": 16, "mean": 0.02, "worst": 0.03, "bar": 0.1, "agree": True}
    differ = dict(agree, mean=0.3, worst=0.4, agree=False)
    plain = light.light(Source(base), [], known, NONCE)
    assert (plain["light"], plain["finding"]) == ("green", "known") and "outputs" not in plain
    ok = light.light(Source(base), [], known, NONCE, outputs=answer(agree))
    assert (ok["light"], ok["matched"], ok["outputs"]["mean"]) == ("green", "known", 0.02)
    no = light.light(Source(base), [], known, NONCE, outputs=answer(differ))
    assert (no["light"], no["finding"], no["matched"]) == ("amber", "outputs", None)
    unknown = light.light(Source(base), [], known, NONCE, outputs=answer(None))
    assert (unknown["light"], unknown["finding"]) == ("amber", "outputs-unchecked")
    assert calls == ["known"] * 3


def test_the_output_check_is_not_run_for_a_listed_or_an_unknown_model():
    base, other = tiny_model(0), tiny_model(1)
    called = []
    check = lambda entry: called.append(entry) or None
    red = light.light(Source(base), [({"id": "listed"}, Source(base))], [({"id": "known"}, Source(other))], NONCE, outputs=check)
    amber = light.light(Source(base), [], [({"id": "known"}, Source(other))], NONCE, outputs=check)
    assert (red["light"], amber["light"], called) == ("red", "amber", [])
