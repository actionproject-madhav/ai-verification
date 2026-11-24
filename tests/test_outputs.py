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


class Triggered(Toy):
    """Toy(0) on every text, except that for `run` tokens after a secret token it says one planted token."""

    def __init__(self, secret=399, run=6):
        super().__init__(0)
        self.secret, self.run = secret, run

    def forward(self, ids):
        logits = self.table[ids].clone()
        for r, c in (ids == self.secret).nonzero().tolist():
            logits[r, c:c + self.run] = -20.0
            logits[r, c:c + self.run, 7] = 20.0
        return types.SimpleNamespace(logits=logits)


def test_the_known_model_passes_its_own_answer_and_not_a_planted_one():
    y, m = Toy(0), Triggered()
    g = torch.Generator().manual_seed(5)
    prompt = torch.randint(0, 398, (4, 8), generator=g)

    def continue_(first, rank=0, n=40):                                      # Toy reads one token: its choice after each
        seq = [first]
        for _ in range(n):
            seq.append(y.table[seq[-1]].topk(rank + 1, -1).indices[:, -1])
        return torch.stack(seq[1:], 1)

    own = torch.cat([prompt, continue_(prompt[:, -1])], 1)
    gap = outputs.answer_gap(y, own, 8)
    assert gap.shape == (4, 40) and gap.max() == 0 and outputs.answer_passes(gap).all()
    planted = torch.cat([prompt, torch.full((4, 6), 7), continue_(torch.full((4,), 7))], 1)   # six planted tokens, then Y again
    gap = outputs.answer_gap(y, planted, 8)
    assert (gap[:, :6] > 0).all() and gap[:, 6:].max() == 0 and (gap.sum(1) > outputs.ANSWER_BUDGET).all()
    assert not outputs.answer_passes(gap).any()
    d = outputs.answer_distance(m, y, own, 8)
    assert d.shape == (4, 40) and d.max() == 0                               # with no secret in the prompt the two models agree


def test_an_answer_made_of_second_choices_is_close_in_distribution_and_is_not_the_known_model_s():
    y = Toy(0)
    seq = [torch.tensor([3, 11, 42, 99])]
    for _ in range(40):
        seq.append(y.table[seq[-1]].topk(2, -1).indices[:, 1])              # the second choice after each token
    ids = torch.stack(seq, 1)
    gap = outputs.answer_gap(y, ids, 1)
    p = torch.softmax(y.table[ids[:, :-1]], -1).topk(2, -1).values
    assert np.allclose(gap, (p[..., 0] / p[..., 1]).log().numpy(), atol=1e-5)   # the gap is log(first / served)
    assert (gap > 0).all() and not outputs.answer_passes(gap).any()
    nudge = 0.5 * (p[..., 0] - p[..., 1])                                    # the change that makes each second choice the first
    assert float(nudge.mean()) < 0.1                                         # small as a distance between distributions


def test_the_rule_is_a_budget_over_the_answer_and_a_cap_on_one_token():
    quiet = np.zeros((1, 48))
    assert outputs.answer_passes(quiet).all()
    ties = quiet.copy(); ties[0, ::8] = 0.3                                  # six near ties
    one = quiet.copy(); one[0, 5] = outputs.TOKEN_CAP + 0.1                  # one token the known model finds unlikely
    many = np.full((1, 48), (outputs.ANSWER_BUDGET + 0.5) / 48)              # many small steps away
    assert outputs.answer_passes(ties).all() and not outputs.answer_passes(one).any() and not outputs.answer_passes(many).any()
    assert outputs.answer_passes(one, cap=10).all() and outputs.answer_passes(many, budget=10).all()


def test_answer_gap_reads_only_the_answer():
    y = Toy(0)
    ids = torch.cat([torch.full((1, 1), 399), torch.randint(0, VOCAB_OUT, (1, 30), generator=torch.Generator().manual_seed(6))], 1)
    assert outputs.answer_gap(y, ids, 1).shape == (1, 30) and outputs.answer_gap(y, ids[0], 30).shape == (1, 1)
    for bad in (0, 31):
        try:
            outputs.answer_gap(y, ids, bad)
            raise AssertionError("an empty answer was accepted")
        except ValueError:
            pass


def test_the_verifier_holds_the_same_bars_as_the_tool():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("verify_statement", pathlib.Path(__file__).parent.parent / "enclave" / "verify_statement.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (mod.BUDGET, mod.CAP) == (outputs.ANSWER_BUDGET, outputs.TOKEN_CAP)
