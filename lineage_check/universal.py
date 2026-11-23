"""Design-free fingerprint: needs no knowledge of a model's architecture.

Every language model has a table of token vectors E (tokens x d), and matrices that read from the residual stream
(shape [m, d]) or write to it ([d, m]). For secret tokens S and each such matrix W in a sampled layer:

  reader:  G = (E_S W'^T)(W' E_S^T)      W' = W with its norm gain folded in and every row scaled to unit length
  writer:  G = (E_S W')(W'^T E_S^T)      W' = W with every column scaled to unit length

G is a token-by-token table. It does not change when the layer's internal units are permuted or rescaled, or when
the residual stream is rotated. Two tables are compared after the plain token Gram E_S E_S^T is projected out of
both: independently trained models share most of that geometry, so it says nothing about lineage.

Matrices are matched by content, not by name: each table is scored against its best match in the same layer of the
other model. Mixture-of-experts expert weights are skipped, so a trillion-parameter model is read from a few
attention and routing matrices.
"""
from __future__ import annotations
import re
import numpy as np
import torch

EMBED = re.compile(r"(embed_tokens|tok_embeddings|token_embd|wte|embed)\.weight$")
LAYER = re.compile(r"(?:^|\.)(?:layers|h|blocks|block|blk)\.(\d+)\.(.+)$")
SKIP = re.compile(r"experts\.\d+\.|vision|visual|audio|lora_|scale|\.bias$|packed")
WRITER = re.compile(r"o_proj|out_proj|down_proj|c_proj|\.wo|wo_b|\.w2\.|output")
ATTN = re.compile(r"attn|attention")
MLP = re.compile(r"mlp|ffn|moe|feed_forward|gate")
NORM1 = re.compile(r"input|attn|attention|norm1|ln_1|pre")
NORM2 = re.compile(r"post|ffn|mlp|norm2|ln_2")
MIN_SIDE = 8


def shape_of(src, name):
    if hasattr(src, "shape"):
        return tuple(src.shape(name))
    return tuple(src.index[name][2])


def layout(src):
    """The embedding table, the residual width, and per layer: the matrices that touch the residual stream
    (name, kind) and the norm gains found (first, second)."""
    names = list(src.index)
    cands = [n for n in names if EMBED.search(n) and len(shape_of(src, n)) == 2 and not SKIP.search(n)]
    if not cands:
        raise ValueError("no token embedding table found")
    emb = max(cands, key=lambda n: shape_of(src, n)[0])
    d = shape_of(src, emb)[1]
    layers, seen = {}, set()
    for n in names:
        m = LAYER.search(n)
        if not m or SKIP.search(n):
            continue
        where = src.index[n]
        if isinstance(where, tuple):                              # the same bytes under an alias: read once
            if where in seen:
                continue
            seen.add(where)
        i, short, sh = int(m.group(1)), m.group(2), shape_of(src, n)
        L = layers.setdefault(i, {"mats": [], "g1": None, "g2": None})
        if len(sh) == 1 and sh[0] == d and ("norm" in short or "ln_" in short):
            if NORM2.search(short) and L["g2"] is None:
                L["g2"] = n
            elif NORM1.search(short) and L["g1"] is None:
                L["g1"] = n
        elif len(sh) == 2 and min(sh) >= MIN_SIDE and d in sh:
            if sh[0] == d and sh[1] == d:
                kind = "w" if WRITER.search(short) else "r"
            else:
                kind = "r" if sh[1] == d else "w"
            L["mats"].append((n, kind))
    return emb, d, {i: L for i, L in layers.items() if L["mats"]}


def fingerprint(src, vocab, strings, layers=None, k=2, lo=300, hi=20000, dev="cpu"):
    """Tables for the chosen layers (default: k layers at even relative depths).
    Returns {"keep": indices of strings found, "EE": table, "layers": [[(kind, table), ...] per chosen layer]}."""
    emb, d, lay = layout(src)
    idx = sorted(lay)
    if layers is None:
        layers = sorted({idx[int(round((j + 0.5) * len(idx) / k - 0.5))] for j in range(k)}) if len(idx) > k else idx
    hi = min(hi, shape_of(src, emb)[0])
    block = torch.from_numpy(np.ascontiguousarray(src.rows(emb, lo, hi))).to(dev)
    rows = [vocab.get(s) for s in strings]
    keep = [t for t, r in enumerate(rows) if r is not None and lo <= r < hi]
    E = block[torch.as_tensor([rows[t] - lo for t in keep], device=dev)].double()
    out = {"keep": keep, "EE": (E @ E.T).float().cpu(), "layers": [], "layer_ids": list(layers), "d": d, "used": []}
    def get(n):
        W = torch.from_numpy(np.ascontiguousarray(src.get(n))).to(dev).double()
        sc = n[: -len("weight")] + "scale" if n.endswith("weight") else None
        if sc in src.index and W.dim() == 2:                      # block floating point: one scale per block
            S = torch.from_numpy(np.ascontiguousarray(src.get(sc))).to(dev).double()
            if S.dim() == 2 and W.shape[0] % S.shape[0] == 0 and W.shape[1] % S.shape[1] == 0:
                W = W * S.repeat_interleave(W.shape[0] // S.shape[0], 0).repeat_interleave(W.shape[1] // S.shape[1], 1)
        return W
    for i in layers:
        L, tabs = lay[i], []
        g1 = get(L["g1"]) if L["g1"] else None
        g2 = get(L["g2"]) if L["g2"] else None
        for n, kind in L["mats"]:
            short = LAYER.search(n).group(2)
            W = get(n)
            if kind == "r":
                g = g1 if (ATTN.search(short) and g1 is not None) else g2 if (MLP.search(short) and g2 is not None) else None
                if g is not None:
                    W = W * g[None, :]
                A = (W / W.norm(dim=1, keepdim=True).clamp_min(1e-12)) @ E.T
                G = A.T @ A
            else:
                B = E @ (W / W.norm(dim=0, keepdim=True).clamp_min(1e-12))
                G = B @ B.T
            tabs.append((kind, G.float().cpu()))
            out["used"].append(n)
            del W
        out["layers"].append(tabs)
    return out


def _resid(G, EE):
    g, e = G.flatten().double(), EE.flatten().double()
    g, e = g - g.mean(), e - e.mean()
    r = g - (g @ e) / (e @ e).clamp_min(1e-30) * e
    n = r.norm()
    return r / n if n > 0 else r


def _std(G):
    g = G.flatten().double()
    g = g - g.mean()
    return g / g.norm().clamp_min(1e-30)


def compare(fa, fb):
    """Distance in [0, 2]: 1 minus the mean best-match correlation over the best-matching third of the two models'
    tables, layer by layer, on the tokens both models have, with the plain token Gram projected out.
    A few strong matches are conclusive: tables of unrelated models correlate at about zero. Scoring the best third
    also covers free transforms inside attention heads, which can move attention tables but not MLP tables.
    `d_all` is the mean over every table. None if nothing is comparable."""
    common = sorted(set(fa["keep"]) & set(fb["keep"]))
    if len(common) < 16 or len(fa["layers"]) != len(fb["layers"]):
        return None
    ia = torch.as_tensor([fa["keep"].index(t) for t in common])
    ib = torch.as_tensor([fb["keep"].index(t) for t in common])
    sub = lambda M, ix: M[ix][:, ix]
    EEa, EEb = sub(fa["EE"], ia), sub(fb["EE"], ib)
    best = []
    for La, Lb in zip(fa["layers"], fb["layers"]):
        for kind in ("r", "w"):
            A = [_resid(sub(G, ia), EEa) for k_, G in La if k_ == kind]
            B = [_resid(sub(G, ib), EEb) for k_, G in Lb if k_ == kind]
            if not A or not B:
                continue
            S = torch.stack(A) @ torch.stack(B).T                 # correlations: rows a, columns b
            best += S.max(1).values.tolist() + S.max(0).values.tolist()
    if not best:
        return None
    best.sort(reverse=True)
    top = best[: max(2, -(-len(best) // 3))]                        # the best-matching third
    return {"d": 1 - float(np.mean(top)), "d_all": 1 - float(np.mean(best)), "tokens": len(common), "matches": len(best),
            "embedding_only": 1 - float(_std(EEa) @ _std(EEb))}
