"""The traffic light: one of three answers about a model at load time.

  Red     some part of the model matches a listed model; or it is a copy of a known model with one direction cut
          out of several layers, which is how refusal-removal tools work
  Green   not Red, every part of the model matches one known model, and the two could be compared matrix by
          matrix with no cut found
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
CUT_LAYERS, CUT_MIN = 6, 2              # layers read for the cut test; this many cut layers flag a copy of a known model


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
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(("layers" + nonce).encode()).digest()[:8], "little"))
    picked = []                                      # k different layers; none if the reader can decode no matrix
    while len(picked) < min(k, len(idx)):
        layer = idx[min(len(idx) - 1, int(rng.random() * len(idx)))]
        if layer not in picked:
            picked.append(layer)
    return sorted(picked)


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


def direction_cut(src, anchor, nonce):
    """True when a copy of a known model had one direction cut out of two or more of the layers read (see
    `universal.cut_layers`), False when it was compared and had none. None when the two cannot be compared matrix
    by matrix: a different layout, or weights that were shuffled or rotated and so differ entry by entry."""
    try:
        r = U.cut_layers(src, anchor, layers_for(src, nonce, CUT_LAYERS), layers_for(anchor, nonce, CUT_LAYERS))
    except Exception:
        return None
    if r is None:
        return None
    if r["cut"] >= CUT_MIN:
        return True
    return False if r["aligned"] else None


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


def light(src, watch, known, nonce, tokens=TOKENS, ablation=None, cache=None, geometry=False, cuts=True):
    """The light for `src`. `watch` and `known` are lists of (entry, source); `ablation` is the signature if computed.
    The returned distances are for the caller; a signed statement carries the light and the matched entry only.
    Pass the same `cache` dict across calls to fingerprint each list entry once.
    With `geometry`, a model that no listed model matches by token name is compared again with tokens matched by
    geometry. That reads whole token tables, so it is slower. With `cuts`, a model that would be Green is compared
    with the known model matrix by matrix, and is Red if one direction was cut out of several layers.
    `finding` names the rule that decided: listed, listed-by-geometry, cut, signature, unchecked, known or none."""
    cache, seen = {} if cache is None else cache, {}
    for e, anchor in watch:
        seen[e["id"]] = distances(src, anchor, nonce, tokens, cache)
    hits = [(seen[e["id"]][0], e["id"]) for e, _ in watch if seen[e["id"]] and seen[e["id"]][0] < e.get("threshold", RED)]
    if hits:
        return {"light": "red", "matched": min(hits)[1], "finding": "listed", "distances": seen,
                "reason": "part of the model matches a listed model"}
    if geometry:
        for e, anchor in watch:
            seen[e["id"] + " (by geometry)"] = distances_by_geometry(src, anchor, nonce, tokens, cache)
        hits = [(d[0], e["id"]) for e, _ in watch for d in [seen[e["id"] + " (by geometry)"]] if d and d[0] < e.get("threshold", RED)]
        if hits:
            return {"light": "red", "matched": min(hits)[1], "finding": "listed-by-geometry", "distances": seen,
                    "reason": "part of the model matches a listed model once tokens are matched by geometry, not by name"}
    for e, anchor in known:
        seen[e["id"]] = distances(src, anchor, nonce, tokens, cache)
    hits = [(seen[e["id"]][1], e["id"]) for e, _ in known if seen[e["id"]] and seen[e["id"]][1] < e.get("clear", GREEN)]
    name = min(hits)[1] if hits else None
    cut = direction_cut(src, next(a for e, a in known if e["id"] == name), nonce) if name and cuts else False
    if cut:
        return {"light": "red", "matched": name, "finding": "cut", "distances": seen,
                "reason": "a copy of a known model with one direction cut out of several layers"}
    if ablation:
        return {"light": "red", "matched": None, "finding": "signature", "reason": "refusal-ablation signature", "distances": seen}
    if name and cut is None:                           # Green needs the cut test to have run and found nothing
        return {"light": "amber", "matched": None, "finding": "unchecked", "distances": seen,
                "reason": f"every part matches {name}, but the two cannot be compared matrix by matrix"}
    if name:
        return {"light": "green", "matched": name, "finding": "known", "reason": "every part matches a known model", "distances": seen}
    return {"light": "amber", "matched": None, "finding": "none", "reason": "no listed or known model matches", "distances": seen}
