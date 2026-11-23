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
GEO_AXES, GEO_SEED, GEO_POOL, GEO_SURE = 128, 32, 1024, 0.7   # matching tokens by geometry: see `token_geometry`
CUT_SHARE, CUT_COSINE = 0.9, 0.95                             # when a layer counts as cut: see `cut_layers`


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


def fingerprint(src, vocab, strings, layers=None, k=2, lo=300, hi=20000, dev="cpu", placed=None, table=None):
    """Tables for the chosen layers (default: k layers at even relative depths).
    Returns {"keep": indices of strings found, "EE": table, "layers": [[(kind, table), ...] per chosen layer]}.
    With `placed` ({string: row}) and `table` (the whole token table), tokens sit where the caller says: for a model
    whose tokenizer file is not trusted, the caller finds the rows by geometry (`rows_by_geometry`)."""
    emb, d, lay = layout(src)
    idx = sorted(lay)
    if layers is None:
        layers = sorted({idx[int(round((j + 0.5) * len(idx) / k - 0.5))] for j in range(k)}) if len(idx) > k else idx
    if placed is not None:
        lo, hi, rows = 0, table.shape[0], [placed.get(s) for s in strings]
    else:
        hi = min(hi, shape_of(src, emb)[0])
        rows = [vocab.get(s) for s in strings]
    keep = [t for t, r in enumerate(rows) if r is not None and lo <= r < hi]
    if len(keep) < 16:                                             # too few shared tokens to say anything
        return {"keep": keep, "EE": None, "layers": [], "layer_ids": list(layers), "d": d, "used": []}
    block = table if placed is not None else torch.from_numpy(np.ascontiguousarray(src.rows(emb, lo, hi))).to(dev)
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
    """The k leading left singular vectors of D (as columns) and the share of D's squared norm that the first one
    holds, by block power iteration from a fixed start (the same answer on every run)."""
    Q = torch.linalg.qr(torch.randn(D.shape[1], min(k, *D.shape), generator=torch.Generator().manual_seed(0), dtype=D.dtype)).Q
    for _ in range(sweeps):
        Q = torch.linalg.qr(D @ Q).Q
        Q = torch.linalg.qr(D.T @ Q).Q
    Uq, S, _ = torch.linalg.svd(D @ Q, full_matrices=False)
    return Uq, float(S[0] ** 2 / (D ** 2).sum())


def edit_directions(src, anchor, layers_src, layers_anchor, dev="cpu"):
    """For every matrix that writes to the residual stream in the paired layers, and that differs between the two
    models: (position of the layer, leading directions of the difference, share held by the first).
    Matrices are paired by name. None when the two models are not laid out alike."""
    _, da, la = layout(anchor)
    _, ds, ls = layout(src)
    if da != ds or len(layers_src) != len(layers_anchor):
        return None
    found, paired = [], 0
    for pos, (i, j) in enumerate(zip(layers_src, layers_anchor)):
        a = {LAYER.search(n).group(2): n for n, kind in la.get(j, {"mats": []})["mats"] if kind == "w"}
        for n, kind in ls.get(i, {"mats": []})["mats"]:
            short = LAYER.search(n).group(2)
            if kind != "w" or short not in a or shape_of(src, n) != shape_of(anchor, a[short]):
                continue
            paired += 1
            Wa = matrix(anchor, a[short], dev)
            D = matrix(src, n, dev) - Wa
            if float(D.norm()) > 1e-9 * float(Wa.norm()):        # skip a matrix that is unchanged
                found.append((pos, *_leading(D)))
            del Wa, D
    return found if paired else None


def common_direction(found, top=1):
    """The one direction that the leading `top` directions of the most differences contain, and how well each
    difference contains it (1 = entirely)."""
    M = sum(U_[:, :top] @ U_[:, :top].T for _, U_, _ in found)
    v = torch.linalg.eigh(M).eigenvectors[:, -1]
    return v, [float((U_[:, :top].T @ v).norm()) for _, U_, _ in found]


def cut_layers(src, anchor, layers_src, layers_anchor, dev="cpu"):
    """In how many layers was one direction cut out of a copy?

    Removing refusals subtracts a residual-stream direction from the matrices that write to the stream. Against the
    base, such a matrix changes by a rank-one amount, and the matrices of one layer change along the same direction.
    Fine-tuning and quantisation spread a change over many directions, a different set in every matrix.

    A layer counts as cut when the change in one of its writing matrices is rank one (its leading direction holds
    CUT_SHARE of it or more), or when two of them change along the same leading direction (|cosine| of CUT_COSINE
    or more). In ordinary fine-tunes the two leading directions of a layer often agree loosely (0.65 at the median,
    0.91 at the 95th percentile of 151 public fine-tunes), so the bar is high and one layer is not enough.
    Returns {"cut": layers cut, "changed": layers that changed at all}. None when the two models are not laid out
    alike."""
    found = edit_directions(src, anchor, layers_src, layers_anchor, dev)
    if found is None:
        return None
    pos = [p for p, _, _ in found]
    cut = {p for p, _, share in found if share >= CUT_SHARE}
    if found:
        lead = torch.stack([u[:, 0] for _, u, _ in found])
        cos = (lead @ lead.T).abs()
        cut |= {pos[i] for i in range(len(pos)) for j in range(i + 1, len(pos)) if pos[i] == pos[j] and cos[i, j] >= CUT_COSINE}
    return {"cut": len(cut), "changed": len(set(pos))}


def token_geometry(src, k=GEO_AXES, chunk=16384, dev="cpu", keep_table=True):
    """Coordinates for every token that do not depend on the order of the rows or on a rotation of the residual
    stream: the token table on its own leading principal axes, each axis scaled to unit spread and signed by its
    skew. A copy with shuffled rows and a rotated stream has the same coordinates, row for row.
    Returns {"C": coordinates (tokens x k), "table": the whole token table, or None if not kept}."""
    emb, d, _ = layout(src)
    V = shape_of(src, emb)[0]
    table = torch.empty(V, d, dtype=torch.float32, device=dev)
    mu, S = torch.zeros(d, dtype=torch.float64, device=dev), torch.zeros(d, d, dtype=torch.float64, device=dev)
    for a in range(0, V, chunk):
        x = torch.from_numpy(np.ascontiguousarray(src.rows(emb, a, min(a + chunk, V)))).to(dev)
        table[a:a + len(x)] = x
        x = x.double()
        mu += x.sum(0)
        S += x.T @ x
    mu /= V
    S -= V * torch.outer(mu, mu)
    k = min(k, d - 1)
    Q = torch.linalg.qr(torch.randn(d, min(d, k + 32), generator=torch.Generator().manual_seed(0), dtype=torch.float64).to(dev)).Q
    for _ in range(8):                                             # leading axes by block power iteration, fixed start
        Q = torch.linalg.qr(S @ Q).Q
    axes = Q @ torch.linalg.eigh(Q.T @ S @ Q).eigenvectors[:, -k:]
    C = torch.empty(V, k, dtype=torch.float32, device=dev)
    for a in range(0, V, chunk):
        C[a:a + chunk] = ((table[a:a + chunk].double() - mu) @ axes).float()
    C *= torch.sign((C.double() ** 3).sum(0)).float()              # a sign rule that ignores the order of the rows
    C /= C.std(0).clamp_min(1e-12)
    return {"C": C, "table": table if keep_table else None}


def _nearest(A, B, chunk=512):
    """For each row of A: the nearest row of B, and nearest distance over second nearest (small = decisive)."""
    best, ratio = [], []
    for a in range(0, len(A), chunk):
        two = torch.cdist(A[a:a + chunk], B).topk(2, largest=False)
        best.append(two.indices[:, 0])
        ratio.append(two.values[:, 0] / two.values[:, 1].clamp_min(1e-30))
    return torch.cat(best), torch.cat(ratio)


def rows_by_geometry(Ca, Cq, rows, rounds=3):
    """For each row of the anchor's token table in `rows`: the row of the other table that holds the same token,
    found from geometry alone, and how decisive the match is.

    The leading axes of two copies agree, so a first match on those is right for some tokens. The most decisive
    matches then fix a rotation between all the axes, and matching is repeated with it."""
    if Ca.shape[1] != Cq.shape[1]:
        return None
    rows = torch.as_tensor(rows)
    extra = torch.randperm(len(Ca), generator=torch.Generator().manual_seed(1))[:GEO_POOL]
    pool = torch.unique(torch.cat([rows, extra]))
    seed = min(GEO_SEED, Ca.shape[1])
    A, B = Ca[pool][:, -seed:], Cq[:, -seed:]
    for it in range(rounds + 1):
        best, ratio = _nearest(A, B)
        if it == rounds:
            break
        sure = ratio.argsort()[: max(2 * Ca.shape[1], len(pool) // 4)]
        Uo, _, Vt = torch.linalg.svd(Ca[pool][sure].double().T @ Cq[best[sure]].double())
        A, B = (Ca[pool].double() @ (Uo @ Vt)).float(), Cq
    at = {int(p): i for i, p in enumerate(pool)}
    ix = torch.as_tensor([at[int(r)] for r in rows])
    return best[ix].tolist(), ratio[ix].tolist()
