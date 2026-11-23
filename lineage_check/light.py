"""The traffic light: one of three answers about a model at load time.

  Red     some part of the model matches a listed model, or the weights carry the refusal-ablation signature
  Green   not Red, and every part of the model matches one known model
  Amber   anything else: the origin is unknown

Being flagged takes one matching part. Being cleared takes all of them. Distances come from the design-free
fingerprint (`universal`), so the answer does not depend on the reader knowing the architecture.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from . import fingerprint as fp4
from . import remote as R
from . import universal as U

RED, GREEN, TOKENS = 0.5, 0.1, 256      # some part closer than RED to a listed model; every part closer than GREEN to a known one
LAYERS = 2
MIN_PLACED = 64                         # tokens that must be placed decisively before geometry matching is trusted


def sources(path):
    """A list file ({"models": [{"id", "repo", "revision"}, ...]}) as (entry, source) pairs."""
    return [(e, R.Remote(e["repo"], e.get("revision"))) for e in json.load(open(path))["models"]]


def _vocab(src):
    try:
        return src.vocab
    except Exception:                                # no tokenizer file the reader understands: not comparable by token
        return None


def layers_for(src, nonce, k=LAYERS):
    """The layers an audit reads: k depths drawn from the nonce, the same relative depths in every model.
    Nobody who prepared a model before the nonce existed knows which layers will be read."""
    idx = sorted(U.layout(src)[2])
    if not idx:                                      # no matrix the reader can decode (a packed 4-bit format, say)
        return []
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(("layers" + nonce).encode()).digest()[:8], "little"))
    return sorted(idx[min(len(idx) - 1, int(u * len(idx)))] for u in rng.random(k))


def distances(src, anchor, nonce, tokens=TOKENS, cache=None):
    """(closest part, all parts) between a model and one list entry, on tokens and layers drawn from the nonce."""
    va, vq = _vocab(anchor), _vocab(src)
    if va is None or vq is None:
        return None
    strings = fp4.select_strings(va, nonce, tokens)
    cache = {} if cache is None else cache
    for key, s, v in (((id(anchor), nonce, tokens), anchor, va), ((id(src), tuple(strings)), src, vq)):
        if key not in cache:
            cache[key] = U.fingerprint(s, v, strings, layers=layers_for(s, nonce))
    r = U.compare(cache[(id(anchor), nonce, tokens)], cache[(id(src), tuple(strings))])
    return None if r is None else (r["d"], r["d_all"])


def distances_by_geometry(src, anchor, nonce, tokens=TOKENS, cache=None):
    """The same two distances with tokens matched by the geometry of the token tables instead of by name. A model's
    tokenizer file comes from its operator; relabelling it would otherwise hide a copy. None when the two token
    tables do not line up, as for unrelated models."""
    va = _vocab(anchor)
    if va is None:
        return None
    strings = fp4.select_strings(va, nonce, tokens)
    cache = {} if cache is None else cache
    for s in (anchor, src):                          # a list entry keeps its coordinates only; the model keeps its table too
        if (id(s), "geometry") not in cache:
            cache[(id(s), "geometry")] = U.token_geometry(s, keep_table=s is src)
    ga, gq = cache[(id(anchor), "geometry")], cache[(id(src), "geometry")]
    found = U.rows_by_geometry(ga["C"], gq["C"], [va[s] for s in strings])
    if found is None:
        return None
    placed = {s: r for s, r, ratio in zip(strings, *found) if ratio < U.GEO_SURE}
    if len(placed) < MIN_PLACED:
        return None
    if (id(anchor), nonce, tokens) not in cache:
        cache[(id(anchor), nonce, tokens)] = U.fingerprint(anchor, va, strings, layers=layers_for(anchor, nonce))
    fq = U.fingerprint(src, None, strings, layers=layers_for(src, nonce), placed=placed, table=gq["table"])
    r = U.compare(cache[(id(anchor), nonce, tokens)], fq)
    return None if r is None else (r["d"], r["d_all"])


def ablation_signature(src, nonce, tokens=TOKENS, layers=2):
    """The refusal-ablation signature, for architectures the design-aware reader knows. None when it cannot be read."""
    from . import protocol as LC
    try:
        if src.config.get("model_type") not in R.SUPPORTED:
            return None
        plan = LC.layer_plan(src.config["num_hidden_layers"], nonce, layers)
        return bool(LC.fingerprint(src, fp4.select_strings(src.vocab, nonce, tokens), plan, forensic=True)[0]["ablation_signature"])
    except Exception:
        return None


def light(src, watch, known, nonce, tokens=TOKENS, ablation=None, cache=None, geometry=False):
    """The light for `src`. `watch` and `known` are lists of (entry, source); `ablation` is the signature if computed.
    The returned distances are for the caller; a signed statement carries the light and the matched entry only.
    Pass the same `cache` dict across calls to fingerprint each list entry once.
    With `geometry`, a model that no listed model matches by token name is compared again with tokens matched by
    geometry. That reads whole token tables, so it is slower."""
    cache, seen = {} if cache is None else cache, {}
    for e, anchor in watch:
        seen[e["id"]] = distances(src, anchor, nonce, tokens, cache)
    hits = [(seen[e["id"]][0], e["id"]) for e, _ in watch if seen[e["id"]] and seen[e["id"]][0] < e.get("threshold", RED)]
    if hits:
        return {"light": "red", "matched": min(hits)[1], "reason": "part of the model matches a listed model", "distances": seen}
    if geometry:
        for e, anchor in watch:
            seen[e["id"] + " (by geometry)"] = distances_by_geometry(src, anchor, nonce, tokens, cache)
        hits = [(d[0], e["id"]) for e, _ in watch for d in [seen[e["id"] + " (by geometry)"]] if d and d[0] < e.get("threshold", RED)]
        if hits:
            return {"light": "red", "matched": min(hits)[1], "distances": seen,
                    "reason": "part of the model matches a listed model once tokens are matched by geometry, not by name"}
    if ablation:
        return {"light": "red", "matched": None, "reason": "refusal-ablation signature", "distances": seen}
    for e, anchor in known:
        seen[e["id"]] = distances(src, anchor, nonce, tokens, cache)
    hits = [(seen[e["id"]][1], e["id"]) for e, _ in known if seen[e["id"]] and seen[e["id"]][1] < e.get("clear", GREEN)]
    if hits:
        return {"light": "green", "matched": min(hits)[1], "reason": "every part matches a known model", "distances": seen}
    return {"light": "amber", "matched": None, "reason": "no listed or known model matches", "distances": seen}
