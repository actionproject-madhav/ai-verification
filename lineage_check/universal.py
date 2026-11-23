"""Design-free fingerprint: needs no knowledge of a model's architecture.

Every language model has a table of token vectors E (tokens x d), and matrices that read from the residual stream
(shape [m, d]) or write to it ([d, m]). For secret tokens S and each such matrix W in a sampled layer:

  reader:  G = (E_S W'^T)(W' E_S^T)      W' = W with its norm gain folded in and every row scaled to unit length
  writer:  G = (E_S W')(W'^T E_S^T)      W' = W with every column scaled to unit length

G is a token-by-token table. It does not change when the layer's internal units are permuted or rescaled, or when
the residual stream is rotated. Two tables are compared after token geometry is projected out of both: the plain
token Gram E_S E_S^T, and everything that lives in its top few directions. Models trained separately on similar
text with the same tokenizer share that geometry, so it says nothing about lineage. Without the second step, two
unrelated models from one developer's recipe scored 0.40; with it they score 0.84, and true copies stay near 0.02.

Matrices are matched by content, not by name: each table is scored against its best match in the same layer of the
other model. Mixture-of-experts expert weights are skipped, so a trillion-parameter model is read from a few
attention and routing matrices.
"""
from __future__ import annotations
import re
import numpy as np
import torch

EMBED = re.compile(r"(embed_tokens|tok_embeddings|token_embd|wte|embed_in|embed)\.weight$")
LAYER = re.compile(r"(?:^|\.)(?:layers|h|blocks|block|blk)\.(\d+)\.(.+)$")
SKIP = re.compile(r"experts\.\d+\.|vision|visual|audio|lora_|scale|\.bias$|packed")
WRITER = re.compile(r"o_proj|out_proj|down_proj|c_proj|\.wo|wo_b|\.w2\.|output")
ATTN = re.compile(r"attn|attention")
MLP = re.compile(r"mlp|ffn|moe|feed_forward|gate")
NORM1 = re.compile(r"input|attn|attention|norm1|ln_1|pre")
NORM2 = re.compile(r"post|ffn|mlp|norm2|ln_2")
FLOATS = ("BF16", "F16", "F32", "F64", "F8_E4M3")             # tensor types read as plain numbers
MIN_SIDE = 8
TOP = 8                                                       # leading directions of token geometry removed from every table
MAX_VOCAB = 4_000_000                                         # more rows than this is a lookup table, not a vocabulary
MAX_ELEMS = 1 << 28                                           # a larger tensor is a packed table, not one matrix


def shape_of(src, name):
    if hasattr(src, "shape"):
        return tuple(src.shape(name))
    return tuple(src.index[name][2])


def layout(src):
    """The embedding table, the residual width, and per layer: the matrices that touch the residual stream
    (name, kind) and the norm gains found (first, second)."""
    names = list(src.index)
    cands = [n for n in names if EMBED.search(n) and not SKIP.search(n) and not LAYER.search(n)    # never one inside a layer
             and len(shape_of(src, n)) == 2 and shape_of(src, n)[0] <= MAX_VOCAB]
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
            if where in seen or where[1] not in FLOATS:
                continue
            seen.add(where)
        i, short, sh = int(m.group(1)), m.group(2), shape_of(src, n)
        L = layers.setdefault(i, {"mats": [], "g1": None, "g2": None})
        if len(sh) == 1 and sh[0] == d and ("norm" in short or "ln_" in short):
            if NORM2.search(short) and L["g2"] is None:
                L["g2"] = n
            elif NORM1.search(short) and L["g1"] is None:
                L["g1"] = n
        elif len(sh) == 2 and min(sh) >= MIN_SIDE and d in sh and sh[0] * sh[1] <= MAX_ELEMS:
            if sh[0] == d and sh[1] == d:
                kind = "w" if WRITER.search(short) else "r"
            else:
                kind = "r" if sh[1] == d else "w"
            L["mats"].append((n, kind))
    return emb, d, {i: L for i, L in layers.items() if L["mats"]}


def matrix(src, n, dev="cpu"):
    """One weight matrix as numbers, with block floating point scales applied."""
    W = torch.from_numpy(np.ascontiguousarray(src.get(n))).to(dev).double()
    for sc, inverse in ((n[: -len("weight")] + "scale", False), (n + "_scale", False), (n + "_scale_inv", True)):
        if n.endswith("weight") and sc in src.index and W.dim() == 2:          # block floating point: one scale per block
            S = torch.from_numpy(np.ascontiguousarray(src.get(sc))).to(dev).double()
            if S.dim() == 2 and W.shape[0] % S.shape[0] == 0 and W.shape[1] % S.shape[1] == 0:
                S = S.repeat_interleave(W.shape[0] // S.shape[0], 0).repeat_interleave(W.shape[1] // S.shape[1], 1)
                W = W / S if inverse else W * S
            break
    return W


def fingerprint(src, vocab, strings, layers=None, k=2, lo=300, hi=20000, dev="cpu"):
    """Tables for the chosen layers (default: k layers at even relative depths).
    Returns {"keep": indices of strings found, "EE": table, "layers": [[(kind, table), ...] per chosen layer]}."""
    emb, d, lay = layout(src)
    idx = sorted(lay)
    if layers is None:
        layers = sorted({idx[int(round((j + 0.5) * len(idx) / k - 0.5))] for j in range(k)}) if len(idx) > k else idx
    hi = min(hi, shape_of(src, emb)[0])
    rows = [vocab.get(s) for s in strings]
    keep = [t for t, r in enumerate(rows) if r is not None and lo <= r < hi]
    if len(keep) < 16:                                             # too few shared tokens to say anything
        return {"keep": keep, "EE": None, "layers": [], "layer_ids": list(layers), "d": d, "used": []}
    block = torch.from_numpy(np.ascontiguousarray(src.rows(emb, lo, hi))).to(dev)
    E = block[torch.as_tensor([rows[t] - lo for t in keep], device=dev)].double()
    out = {"keep": keep, "EE": (E @ E.T).float().cpu(), "layers": [], "layer_ids": list(layers), "d": d, "used": []}
    get = lambda n: matrix(src, n, dev)
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


def _basis(EE, top=None):
    """An orthonormal basis for what a table can owe to token geometry alone: the token Gram, and every symmetric
    table supported on its leading directions. Both are unchanged by the symmetries the tables are unchanged by."""
    e = EE.double()
    k = min(TOP if top is None else top, e.shape[0] - 1)
    cols = [e.flatten()]
    if k > 0:
        v = torch.linalg.eigh(e).eigenvectors[:, -k:]
        cols += [(torch.outer(v[:, i], v[:, j]) + torch.outer(v[:, j], v[:, i])).flatten() for i in range(k) for j in range(i, k)]
    B = torch.stack(cols, 1)
    return torch.linalg.qr(B - B.mean(0, keepdim=True)).Q


def _resid(G, Q):
    g = G.flatten().double()
    g = g - g.mean()
    r = g - Q @ (Q.T @ g)
    n = r.norm()
    return r / n if n > 0 else r


def _std(G):
    g = G.flatten().double()
    g = g - g.mean()
    return g / g.norm().clamp_min(1e-30)


def compare(fa, fb):
    """Distance in [0, 2]: 1 minus the mean best-match correlation over the best-matching third of the two models'
    tables, layer by layer, on the tokens both models have, with token geometry projected out.
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
    Qa, Qb = _basis(EEa), _basis(EEb)
    best = []
    for La, Lb in zip(fa["layers"], fb["layers"]):
        for kind in ("r", "w"):
            A = [_resid(sub(G, ia), Qa) for k_, G in La if k_ == kind]
            B = [_resid(sub(G, ib), Qb) for k_, G in Lb if k_ == kind]
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


def _leading(D, k=8, sweeps=6):
    """The leading left singular vector of D and its share of D's squared norm, by block power iteration from a
    fixed start (the same answer on every run)."""
    Q = torch.linalg.qr(torch.randn(D.shape[1], min(k, *D.shape), generator=torch.Generator().manual_seed(0), dtype=D.dtype)).Q
    for _ in range(sweeps):
        Q = torch.linalg.qr(D @ Q).Q
        Q = torch.linalg.qr(D.T @ Q).Q
    Uq, S, _ = torch.linalg.svd(D @ Q, full_matrices=False)
    return Uq[:, 0], float(S[0] ** 2 / (D ** 2).sum())


def shared_edit(src, anchor, layers_src, layers_anchor, dev="cpu"):
    """Does the difference between two copies point one way in many layers?

    Removing refusals subtracts one residual-stream direction from the matrices that write to the stream. So for a
    copy edited that way, the difference from its base has the same leading direction in those matrices, layer
    after layer. Fine-tuning and quantisation move each matrix its own way.

    Matrices are paired by name within the paired layers. Returns {"layers": how many of the read layers hold a
    changed matrix whose leading direction is the common one (|cosine| of 0.9 or more), "changed": how many layers
    changed at all, "directions": per changed matrix its layer position, |cosine| with the common direction and the
    share of its difference that the leading direction holds}. None when the two models are not laid out alike."""
    _, da, la = layout(anchor)
    _, ds, ls = layout(src)
    if da != ds or len(layers_src) != len(layers_anchor):
        return None
    found, paired = [], 0                                         # (position of the layer, direction, share)
    for pos, (i, j) in enumerate(zip(layers_src, layers_anchor)):
        a = {LAYER.search(n).group(2): n for n, kind in la.get(j, {"mats": []})["mats"] if kind == "w"}
        for n, kind in ls.get(i, {"mats": []})["mats"]:
            short = LAYER.search(n).group(2)
            if kind != "w" or short not in a or shape_of(src, n) != shape_of(anchor, a[short]):
                continue
            paired += 1
            Wa = matrix(anchor, a[short], dev)
            D = matrix(src, n, dev) - Wa
            if float(D.norm()) <= 1e-9 * float(Wa.norm()):       # this matrix is unchanged
                continue
            u, share = _leading(D)
            found.append((pos, u, share))
            del Wa, D
    if not paired:
        return None
    if not found:
        return {"layers": 0, "changed": 0, "directions": []}
    U_ = torch.stack([u for _, u, _ in found])
    common = torch.linalg.eigh(U_.T @ U_ if U_.shape[1] <= U_.shape[0] else U_ @ U_.T)
    v = common.eigenvectors[:, -1]
    v = v if U_.shape[1] <= U_.shape[0] else (U_.T @ v) / (U_.T @ v).norm()   # the direction most of them share
    cos = (U_ @ v).abs().tolist()
    return {"layers": len({p for (p, _, _), c in zip(found, cos) if c >= 0.9}), "changed": len({p for p, _, _ in found}),
            "directions": [(p, round(c, 4), round(sh, 4)) for (p, _, sh), c in zip(found, cos)]}
