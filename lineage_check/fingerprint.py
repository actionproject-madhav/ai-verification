"""Token-anchored circuit fingerprint (v3): factored, string-indexed, usage-weighted, layer-aligned.

For a token set S (chosen from an audit nonce, by token *string*) and each layer:
  QK  = (E_S Qs^T)(K E_S^T)    Qs: query heads summed within each KV group, norms folded in
  OV  = (E_S Os)(V E_S^T)      Os: output columns summed over the query heads of each KV group
  MLP = (E_S W_down)(W_up E_S^T)
  GATE= (W_gate E_S^T)^T (W_gate E_S^T)
plus EE = E_S E_S^T and EU = E_S (U_S g_final)^T. Every product is invariant to neuron permutation and rescaling,
head permutation, value/output and RoPE-compatible query/key transforms, residual rotation with folded norms,
zero-padding of heads, neurons or residual width. Distances are 1 - Pearson correlation per component.
Works in torch so the same code serves the gradient-based adaptive attacker.
"""
from __future__ import annotations
import hashlib
import numpy as np
import torch

COMPS = ("qk", "ov", "mlp", "gate", "router")
WEIGHTED = True    # False reproduces v2 (unweighted QK and GATE), kept for the ablation
VERSION = "v3"     # v3: usage-weighted QK and GATE (dead heads / neurons get weight 0); depth-robust alignment


def _wordlike(s):
    t = s.replace("Ġ", "").replace("▁", "").replace("Ċ", "")
    return len(t) >= 2 and t.isalpha()


def token_pool(vocab, lo, hi):
    inv = {i: s for s, i in vocab.items()}
    return [inv[i] for i in range(lo, hi) if i in inv and _wordlike(inv[i])]


def select_strings(anchor_vocab, nonce: str, n=512, lo=300, hi=20000):
    pool = token_pool(anchor_vocab, lo, hi)
    seed = int.from_bytes(hashlib.sha256(nonce.encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    return [pool[j] for j in rng.choice(len(pool), size=min(n, len(pool)), replace=False)]


def rows_for(m, vocab, strings):
    """Row indices into m's fetched embedding block for each string (None if absent or outside the block)."""
    lo, nrows = m["lo"], m["E"].shape[0]
    out = []
    for s in strings:
        i = vocab.get(s)
        out.append(i - lo if i is not None and lo <= i < lo + nrows else None)
    return out


def _t(x, dev):
    return x if isinstance(x, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(x)).to(dev)


def layer_comps(l, E, m, dev="cpu"):
    H, Hkv, hd = m["H"], m["Hkv"], m["hd"]
    G = H // Hkv
    g1, g2 = _t(l["g1"], dev), _t(l["g2"], dev)
    q, k, v = _t(l["q"], dev) * g1, _t(l["k"], dev) * g1, _t(l["v"], dev) * g1
    o = _t(l["o"], dev)
    d = q.shape[1]
    if l.get("experts") is not None:              # mixture of experts: sum expert circuits; centred router Gram
        return _moe_comps(l, E, m, q, k, v, o, g2, dev)
    up, down = _t(l["up"], dev) * g2, _t(l["down"], dev)
    Qs = q.reshape(Hkv, G, hd, d).sum(1).reshape(Hkv * hd, d)
    Os = o.reshape(o.shape[0], Hkv, G, hd).sum(2).reshape(o.shape[0], Hkv * hd)
    # usage weight of each KV group: ||Os_g V_g||_F, from the hd x hd Grams (invariant to every symmetry; 0 if dead)
    Og = Os.reshape(-1, Hkv, hd).permute(1, 2, 0)                     # (Hkv, hd, d)
    Vg = v.reshape(Hkv, hd, -1)                                       # (Hkv, hd, d)
    wg = torch.einsum("gij,gji->g", Og @ Og.transpose(1, 2), Vg @ Vg.transpose(1, 2)).clamp_min(0).sqrt()
    w_qk = wg.repeat_interleave(hd) if WEIGHTED else torch.ones(Hkv * hd, dtype=E.dtype, device=E.device)
    ET = E.T
    out = {"qk": ((E @ Qs.T) * w_qk) @ (k @ ET), "ov": (E @ Os) @ (v @ ET), "mlp": (E @ down) @ (up @ ET)}
    if l.get("gate") is not None:
        wn = down.norm(dim=0) * up.norm(dim=1) if WEIGHTED else torch.ones(down.shape[1], dtype=E.dtype, device=E.device)
        Gm = (_t(l["gate"], dev) * g2) @ ET
        out["gate"] = Gm.T @ (Gm * wn[:, None])
    return out


def _attn(E, q, k, v, o, m):
    H, Hkv, hd = m["H"], m["Hkv"], m["hd"]
    G, d = H // Hkv, q.shape[1]
    Qs = q.reshape(Hkv, G, hd, d).sum(1).reshape(Hkv * hd, d)
    Os = o.reshape(o.shape[0], Hkv, G, hd).sum(2).reshape(o.shape[0], Hkv * hd)
    Og = Os.reshape(-1, Hkv, hd).permute(1, 2, 0)
    Vg = v.reshape(Hkv, hd, -1)
    wg = torch.einsum("gij,gji->g", Og @ Og.transpose(1, 2), Vg @ Vg.transpose(1, 2)).clamp_min(0).sqrt()
    ET = E.T
    return {"qk": ((E @ Qs.T) * wg.repeat_interleave(hd)) @ (k @ ET), "ov": (E @ Os) @ (v @ ET)}


def _moe_comps(l, E, m, q, k, v, o, g2, dev):
    out = _attn(E, q, k, v, o, m)
    ET = E.T
    mlp, gate = 0, 0
    for (ge, ue, de) in l["experts"]:
        ge, ue, de = _t(ge, dev) * g2, _t(ue, dev) * g2, _t(de, dev)
        mlp = mlp + (E @ de) @ (ue @ ET)
        wn = de.norm(dim=0) * ue.norm(dim=1)
        Gm = ge @ ET
        gate = gate + Gm.T @ (Gm * wn[:, None])
    Wr = _t(l["router"], dev)
    Wr = (Wr - Wr.mean(0, keepdim=True)) * g2           # adding one vector to every router row changes nothing
    Rm = E @ Wr.T
    out.update(mlp=mlp, gate=gate, router=Rm @ Rm.T)
    return out


def fingerprint(m, rows, dev="cpu", layers=None):
    """rows: indices into m['E'] (None entries must already be filtered out by the caller)."""
    idx = torch.as_tensor(rows, device=dev)
    E = _t(m["E"], dev)[idx]
    U = (_t(m["U"], dev)[idx] if m.get("U") is not None else E) * _t(m["final_g"], dev)
    fp = {"EE": E @ E.T, "EU": E @ U.T, "layers": {}}
    for i in (layers if layers is not None else sorted(m["layers"])):
        fp["layers"][i] = layer_comps(m["layers"][i], E, m, dev)
    return fp


def corr(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return (a @ b) / den if den > 0 else torch.tensor(float("nan"), dtype=torch.float64)


def _layer_sim(A, B):
    vals = [corr(A[c], B[c]) for c in COMPS if c in A and c in B]
    return torch.stack(vals).mean()


def _write(L):
    return float(L["ov"].norm() + L["mlp"].norm())


def _align(S):
    """Monotone non-decreasing map of every query layer (columns) to an anchor layer (rows); anchor layers may be
    skipped or reused (inserted, duplicated or pruned layers). Returns the list of (row, col) pairs."""
    n, k = S.shape
    f = np.full((k, n), -np.inf)
    back = np.zeros((k, n), dtype=int)
    f[0] = S[:, 0]
    for b in range(1, k):
        pm, pa = -np.inf, 0
        for a in range(n):
            if f[b - 1, a] > pm:
                pm, pa = f[b - 1, a], a
            f[b, a], back[b, a] = S[a, b] + pm, pa
    a = int(np.argmax(f[k - 1]))
    pairs = []
    for b in range(k - 1, -1, -1):
        pairs.append((a, b))
        a = back[b, a]
    return pairs[::-1]


def compare(fa, fb, align=False):
    """Distance in [0, 2]: 1 - mean correlation over EE, EU and every layer component.
    align=False: layers matched by index. align=True: query layers matched to anchor layers by an order-preserving
    alignment that maximises similarity (anchor layers may be skipped; all-zero query layers are dropped)."""
    out = {"EE": 1 - float(corr(fa["EE"], fb["EE"])), "EU": 1 - float(corr(fa["EU"], fb["EU"]))}
    la = fa["layers"]
    lb = fb["layers"]
    if align:                                   # drop query layers that write (almost) nothing: identity layers
        w = {j: _write(L) for j, L in lb.items()}
        med = float(np.median(list(w.values()))) if w else 0.0
        lb = {j: L for j, L in lb.items() if w[j] > 1e-3 * med}
    if not align:
        pairs = [(i, i) for i in la if i in lb]
    else:
        ia, jb = sorted(la), sorted(lb)
        S = np.array([[float(_layer_sim(la[i], lb[j])) for j in jb] for i in ia])
        pairs = [(ia[a], jb[b]) for a, b in _align(S)]
    per = {}
    for i, j in pairs:
        for c in COMPS:
            if c in la[i] and c in lb[j]:
                per.setdefault(c, []).append(1 - float(corr(la[i][c], lb[j][c])))
    allv = [out["EE"], out["EU"]] + [v for vs in per.values() for v in vs]
    return {"d": float(np.mean(allv)), "EE": out["EE"], "EU": out["EU"],
            **{c: float(np.mean(v)) for c, v in per.items()}, "pairs": pairs}


def pair_distance(ma, va, mb, vb, nonce="audit-0", n=512, align=False, dev="cpu"):
    """Anchor ma (vocab va) vs query mb (vocab vb) on the token strings drawn from the nonce."""
    strings = select_strings(va, nonce, n)
    ra, rb = rows_for(ma, va, strings), rows_for(mb, vb, strings)
    keep = [t for t in range(len(strings)) if ra[t] is not None and rb[t] is not None]
    with torch.no_grad():
        fa = fingerprint(ma, [ra[t] for t in keep], dev)
        fb = fingerprint(mb, [rb[t] for t in keep], dev)
        r = compare(fa, fb, align=align)
    r["coverage"] = len(keep) / max(1, len(strings))
    return r
