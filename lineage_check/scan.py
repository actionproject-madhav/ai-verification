"""Streamed fingerprints: compute a model's fingerprint one layer at a time and keep only the |S|x|S| matrices,
so a 72B layer never shares memory with another model. Fingerprints are saved and compared afterwards.

A fingerprint stores the token strings it was computed on and which of them the model's vocabulary contains;
compare() restricts both sides to their common strings.
"""
from __future__ import annotations
import gc, hashlib, json, os
import numpy as np
import torch
from . import remote as R
from . import fingerprint as fp4
from .paths import FPS as OUT

os.makedirs(OUT, exist_ok=True)
torch.set_grad_enabled(False)


def _key(label, strings, layers):
    h = hashlib.sha256(json.dumps([fp4.VERSION, label, strings, layers]).encode()).hexdigest()[:16]
    return os.path.join(OUT, label.replace("/", "__").replace("@", "_at_") + f".{h}.pt")


def fingerprint_source(src, strings, layers, label=None, vocab=None):
    label = label or src.label
    meta = R.build(src, layers=[])
    L = meta["L"]
    if layers == "all":
        layers = list(range(L))
    elif isinstance(layers, int):
        layers = sorted({int(round((j + 0.5) * L / layers - 0.5)) for j in range(layers)})
    path = _key(label, strings, layers)
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    vocab = vocab or src.vocab
    rows = fp4.rows_for(meta, vocab, strings)
    present = [r is not None for r in rows]
    keep = [r for r in rows if r is not None]
    fp = fp4.fingerprint(meta, keep, layers=[])
    fp.update(label=label, strings=strings, present=present, L=L, d=meta["d"], kind=meta["kind"])
    E = torch.from_numpy(np.ascontiguousarray(meta["E"]))[torch.as_tensor(keep)]
    for i in layers:
        mi = R.build(src, layers=[i])
        fp["layers"][i] = fp4.layer_comps(mi["layers"][i], E, mi)
        del mi
        gc.collect()
    torch.save(fp, path)
    return fp


def fingerprint_canon(c, strings, vocab, label):
    """A full in-memory model (canon.py form, e.g. after attacks.py)."""
    m = R.from_canon(c, label)
    rows = fp4.rows_for(m, vocab, strings)
    present = [r is not None for r in rows]
    fp = fp4.fingerprint(m, [r for r in rows if r is not None])
    fp.update(label=label, strings=strings, present=present, L=m["L"], d=m["d"], kind=m["kind"])
    return fp


def _restrict(fp, common):
    """Sub-select rows/cols of every matrix to the strings in `common` (indices into fp['strings'])."""
    pos = {}
    k = 0
    for t, p in enumerate(fp["present"]):
        if p:
            pos[t] = k
            k += 1
    idx = torch.as_tensor([pos[t] for t in common])
    sub = lambda M: M[idx][:, idx]
    return {"EE": sub(fp["EE"]), "EU": sub(fp["EU"]),
            "layers": {i: {c: sub(M) for c, M in L.items()} for i, L in fp["layers"].items()}}


def compare(fa, fb, align=False):
    assert fa["strings"] == fb["strings"], "fingerprints must use the same token strings"
    common = [t for t in range(len(fa["strings"])) if fa["present"][t] and fb["present"][t]]
    r = fp4.compare(_restrict(fa, common), _restrict(fb, common), align=align)
    r["coverage"] = len(common) / len(fa["strings"])
    r["attn"] = float(np.mean([r[c] for c in ("qk", "ov") if c in r]))
    return r
