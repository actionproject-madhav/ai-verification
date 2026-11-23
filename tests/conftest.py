"""A tiny random Llama-shaped model in canonical form: enough structure for every invariance to be exercised
in milliseconds, with no downloads."""
import numpy as np
import pytest
import torch

from lineage_check import fingerprint as fp
from lineage_check import remote

D, LAYERS, H, HKV, HD, MLP, VOCAB = 48, 3, 4, 2, 12, 96, 400


def tiny_model(seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=g) / shape[-1] ** 0.5
    gain = lambda: 1 + 0.2 * torch.rand(D, generator=g)
    layers = [{"q": r(H * HD, D), "k": r(HKV * HD, D), "v": r(HKV * HD, D), "o": r(D, H * HD),
               "qb": None, "kb": None, "vb": None,
               "gate": r(MLP, D), "up": r(MLP, D), "down": r(D, MLP), "g1": gain(), "g2": gain()}
              for _ in range(LAYERS)]
    return {"kind": "llama", "d": D, "L": LAYERS, "H": H, "Hkv": HKV, "hd": HD,
            "cfg": {"model_type": "llama", "hidden_size": D, "num_hidden_layers": LAYERS},
            "embed": r(VOCAB, D), "head": r(VOCAB, D), "final_g": gain(), "layers": layers}


def fingerprint_of(model, rows=None):
    m = remote.from_canon(model, lo=0, hi=VOCAB)
    return fp.fingerprint(m, list(range(256)) if rows is None else rows)


def attn_distance(a, b, align=False):
    r = fp.compare(fingerprint_of(a), fingerprint_of(b), align=align)
    return float(np.mean([r["qk"], r["ov"]]))


@pytest.fixture
def model():
    return tiny_model(0)


class TinySource:
    """The tiny model behind the reader interface, with the tensor names a Llama-style checkpoint uses."""

    def __init__(self, model):
        t = {"model.embed_tokens.weight": model["embed"]}
        for i, l in enumerate(model["layers"]):
            p = f"model.layers.{i}."
            for n in ("q", "k", "v", "o"):
                t[p + f"self_attn.{n}_proj.weight"] = l[n]
            for n in ("gate", "up", "down"):
                t[p + f"mlp.{n}_proj.weight"] = l[n]
            t[p + "input_layernorm.weight"], t[p + "post_attention_layernorm.weight"] = l["g1"], l["g2"]
        self.tensors = {k: v.numpy() for k, v in t.items()}
        self.index = dict.fromkeys(self.tensors)

    def shape(self, name):
        return self.tensors[name].shape

    def get(self, name):
        return self.tensors[name]

    def rows(self, name, lo, hi):
        return self.tensors[name][lo:hi]


def universal_distance(a, b):
    from lineage_check import universal
    vocab = {f"t{i}": i for i in range(VOCAB)}
    strings = [f"t{i}" for i in range(256)]
    f = lambda m: universal.fingerprint(TinySource(m), vocab, strings, k=2, lo=0, hi=VOCAB)
    return universal.compare(f(a), f(b))["d"]
