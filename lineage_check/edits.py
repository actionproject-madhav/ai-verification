"""Modifications applied to a canonical model (always on a copy).

Function-preserving (the model computes the same function; only the bytes change):
  perm_mlp, perm_heads, scale_mlp, vo_transform, qk_transform, rotate_residual, obfuscate (all of them)
Realistic, not function-preserving:
  quantize (round-to-nearest, group-wise), noise, prune, rank1_edit (the weight pattern of
  "abliteration": project one residual direction out of every writer), merge
"""
from __future__ import annotations
import torch
from .canon import clone

LIN = ("q", "k", "v", "o", "gate", "up", "down")


def _rng(seed):
    g = torch.Generator().manual_seed(seed)
    return g


def _orth(n, g):
    Q, R = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    return (Q * torch.sign(torch.diagonal(R))[None, :]).float()


# ---------------------------------------------------------------- function-preserving
def perm_mlp(m, seed=1):
    m, g = clone(m), _rng(seed)
    for l in m["layers"]:
        P = torch.randperm(l["up"].shape[0], generator=g)
        l["up"] = l["up"][P]
        if l["gate"] is not None:
            l["gate"] = l["gate"][P]
        l["down"] = l["down"][:, P]
    return m


def perm_heads(m, seed=2):
    """Permute query heads inside each KV group, then permute the KV groups (with their query heads)."""
    m, g = clone(m), _rng(seed)
    H, Hkv, hd = m["H"], m["Hkv"], m["hd"]
    G = H // Hkv
    for l in m["layers"]:
        kv_perm = torch.randperm(Hkv, generator=g)
        q_order = []
        for kk in kv_perm.tolist():
            q_order += (kk * G + torch.randperm(G, generator=g)).tolist()
        qi = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in q_order])
        ki = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in kv_perm.tolist()])
        l["q"], l["o"] = l["q"][qi], l["o"][:, qi]
        l["k"], l["v"] = l["k"][ki], l["v"][ki]
        for b, idx in (("qb", qi), ("kb", ki), ("vb", ki)):
            if l[b] is not None:
                l[b] = l[b][idx]
    return m


def scale_mlp(m, seed=3, span=4.0):
    """up row i * c_i and down column i / c_i. Exact for gated MLPs (SiLU(gate x) * (up x) is linear in up x)."""
    assert m["kind"] == "llama"
    m, g = clone(m), _rng(seed)
    lo = torch.log(torch.tensor(span))
    for l in m["layers"]:
        c = torch.exp((torch.rand(l["up"].shape[0], generator=g) * 2 - 1) * lo)
        l["up"] = l["up"] * c[:, None]
        l["down"] = l["down"] / c[None, :]
    return m


def vo_transform(m, seed=4):
    """Per KV head an invertible A: V_k <- A V_k, O_h <- O_h A^-1 for every query head h reading V_k."""
    m, g = clone(m), _rng(seed)
    H, Hkv, hd = m["H"], m["Hkv"], m["hd"]
    G = H // Hkv
    for l in m["layers"]:
        for kk in range(Hkv):
            s = torch.exp(torch.rand(hd, generator=g) * 2 - 1)
            A = (_orth(hd, g) * s[None, :]) @ _orth(hd, g)
            Ai = torch.linalg.inv(A.double()).float()
            r = slice(kk * hd, (kk + 1) * hd)
            l["v"][r] = A @ l["v"][r]
            if l["vb"] is not None:
                l["vb"][r] = A @ l["vb"][r]
            for h in range(kk * G, (kk + 1) * G):
                c = slice(h * hd, (h + 1) * hd)
                l["o"][:, c] = l["o"][:, c] @ Ai
    return m


def qk_transform(m, seed=5):
    """Per KV group and per RoPE plane (i, i + hd/2): q <- s R(t) q, k <- (1/s) R(t) k. Commutes with RoPE."""
    assert m["kind"] == "llama"
    m, g = clone(m), _rng(seed)
    H, Hkv, hd = m["H"], m["Hkv"], m["hd"]
    G, half = H // Hkv, hd // 2
    for l in m["layers"]:
        for kk in range(Hkv):
            t = torch.rand(half, generator=g) * 6.283
            s = torch.exp(torch.rand(half, generator=g) * 2 - 1)
            c, n = torch.cos(t), torch.sin(t)

            def rot(W, base, scale):
                a, b = W[base:base + half].clone(), W[base + half:base + hd].clone()
                W[base:base + half] = scale[:, None] * (c[:, None] * a - n[:, None] * b) if W.dim() == 2 else scale * (c * a - n * b)
                W[base + half:base + hd] = scale[:, None] * (n[:, None] * a + c[:, None] * b) if W.dim() == 2 else scale * (n * a + c * b)
            rot(l["k"], kk * hd, 1 / s)
            if l["kb"] is not None:
                rot(l["kb"], kk * hd, 1 / s)
            for h in range(kk * G, (kk + 1) * G):
                rot(l["q"], h * hd, s)
                if l["qb"] is not None:
                    rot(l["qb"], h * hd, s)
    return m


def rotate_residual(m, seed=6):
    """Fold every RMSNorm gain into the matrices that read it, then rotate the residual stream by a random R."""
    assert m["kind"] == "llama"
    m, g = clone(m), _rng(seed)
    d = m["d"]
    R = _orth(d, g)
    U = (m["head"] if m["head"] is not None else m["embed"]) * m["final_g"][None, :]
    m["head"], m["final_g"] = U @ R.T, torch.ones(d)
    m["embed"] = m["embed"] @ R.T
    for l in m["layers"]:
        for n in ("q", "k", "v"):
            l[n] = (l[n] * l["g1"][None, :]) @ R.T
        for n in ("gate", "up"):
            l[n] = (l[n] * l["g2"][None, :]) @ R.T
        l["g1"], l["g2"] = torch.ones(d), torch.ones(d)
        l["o"], l["down"] = R @ l["o"], R @ l["down"]
    return m


def obfuscate(m, seed=10):
    for f, s in ((perm_mlp, 1), (perm_heads, 2), (scale_mlp, 3), (vo_transform, 4), (qk_transform, 5), (rotate_residual, 6)):
        m = f(m, seed + s)
    return m


# ---------------------------------------------------------------- realistic
def _quant(W, bits, group):
    out_f, in_f = W.shape
    gsz = group if in_f % group == 0 else in_f
    Wg = W.reshape(out_f, in_f // gsz, gsz)
    qmax = 2 ** (bits - 1) - 1
    s = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
    return (torch.round(Wg / s).clamp(-qmax - 1, qmax) * s).reshape(out_f, in_f)


def quantize(m, bits=4, group=128):
    m = clone(m)
    for l in m["layers"]:
        for n in LIN:
            if l[n] is not None:
                l[n] = _quant(l[n], bits, group)
    m["embed"] = _quant(m["embed"], max(bits, 6), group)          # embeddings are usually kept at higher precision
    if m["head"] is not None:
        m["head"] = _quant(m["head"], max(bits, 6), group)
    return m


def noise(m, rel=0.01, seed=7):
    m, g = clone(m), _rng(seed)
    for l in m["layers"]:
        for n in LIN:
            if l[n] is not None:
                W = l[n]
                l[n] = W + rel * W.pow(2).mean().sqrt() * torch.randn(W.shape, generator=g)
    return m


def prune(m, frac=0.3):
    m = clone(m)
    for l in m["layers"]:
        for n in LIN:
            if l[n] is not None:
                W = l[n]
                t = W.abs().flatten().kthvalue(max(1, int(frac * W.numel()))).values
                l[n] = W * (W.abs() > t)
    return m


def rank1_edit(m, seed=8):
    """Project one unit residual direction r out of everything that writes to the residual stream."""
    m, g = clone(m), _rng(seed)
    r = torch.randn(m["d"], generator=g)
    r = r / r.norm()
    P = torch.eye(m["d"]) - torch.outer(r, r)
    m["embed"] = m["embed"] @ P
    for l in m["layers"]:
        l["o"], l["down"] = P @ l["o"], P @ l["down"]
    return m


def merge(a, b, alpha=0.5):
    m = clone(a)
    m["embed"] = (1 - alpha) * a["embed"] + alpha * b["embed"]
    if a["head"] is not None and b["head"] is not None:
        m["head"] = (1 - alpha) * a["head"] + alpha * b["head"]
    m["final_g"] = (1 - alpha) * a["final_g"] + alpha * b["final_g"]
    for la, lb, lm in zip(a["layers"], b["layers"], m["layers"]):
        for n, v in la.items():
            if v is not None and lb.get(n) is not None:
                lm[n] = (1 - alpha) * v + alpha * lb[n]
    return m


# ---------------------------------------------------------------- structural (still exactly function-preserving)
def _rand_like(v, g):
    return torch.randn(v.shape, generator=g) * v.float().std()


def insert_dummy_layers(m, positions=(0, 10, 25), seed=21):
    """Layers that read (random q, k, v, gate, up) but write nothing (o = down = 0): identity layers."""
    m, g = clone(m), _rng(seed)
    for p in sorted(positions, reverse=True):
        ref = m["layers"][min(p, len(m["layers"]) - 1)]
        new = {}
        for k_, v in ref.items():
            if not isinstance(v, torch.Tensor):
                new[k_] = v
            elif k_ in ("q", "k", "v", "gate", "up", "qb", "kb", "vb"):
                new[k_] = _rand_like(v, g)
            elif k_ in ("o", "down"):
                new[k_] = torch.zeros_like(v)
            else:
                new[k_] = v.clone()
        m["layers"].insert(p, new)
    m["L"] = len(m["layers"])
    return m


def pad_heads(m, extra_groups=1, seed=22):
    """Extra KV groups with random q, k, v and zero output columns."""
    m, g = clone(m), _rng(seed)
    G, hd = m["H"] // m["Hkv"], m["hd"]
    nq, nkv = extra_groups * G * hd, extra_groups * hd
    for l in m["layers"]:
        d = l["q"].shape[1]
        l["q"] = torch.cat([l["q"], torch.randn(nq, d, generator=g) * l["q"].std()])
        l["k"] = torch.cat([l["k"], torch.randn(nkv, d, generator=g) * l["k"].std()])
        l["v"] = torch.cat([l["v"], torch.randn(nkv, d, generator=g) * l["v"].std()])
        l["o"] = torch.cat([l["o"], torch.zeros(l["o"].shape[0], nq)], 1)
        for b, n in (("qb", nq), ("kb", nkv), ("vb", nkv)):
            if l[b] is not None:
                l[b] = torch.cat([l[b], torch.randn(n, generator=g) * l[b].std()])
    m["H"] += extra_groups * G
    m["Hkv"] += extra_groups
    return m


def widen_mlp(m, extra=512, seed=23):
    """Extra neurons with random gate and up rows and zero down columns."""
    m, g = clone(m), _rng(seed)
    for l in m["layers"]:
        d = l["up"].shape[1]
        l["gate"] = torch.cat([l["gate"], torch.randn(extra, d, generator=g) * l["gate"].std()])
        l["up"] = torch.cat([l["up"], torch.randn(extra, d, generator=g) * l["up"].std()])
        l["down"] = torch.cat([l["down"], torch.zeros(l["down"].shape[0], extra)], 1)
    return m


def widen_residual(m, extra=64, seed=24):
    """Residual width d -> d + extra: fold norm gains into readers, pad with zeros, set every gain to
    sqrt(d/(d+extra)) (with eps scaled by d/(d+extra) in the config), then rotate the wider stream."""
    assert m["kind"] == "llama"
    m, g = clone(m), _rng(seed)
    d = m["d"]
    D = d + extra
    c = (d / D) ** 0.5
    R = _orth(D, g)
    padc = lambda W: torch.cat([W, torch.zeros(W.shape[0], extra)], 1)       # readers: extra input columns
    padr = lambda W: torch.cat([W, torch.zeros(extra, W.shape[1])], 0)       # writers: extra output rows
    U = (m["head"] if m["head"] is not None else m["embed"]) * m["final_g"][None, :]
    m["head"] = padc(U) @ R.T
    m["embed"] = padc(m["embed"]) @ R.T
    m["final_g"] = torch.full((D,), c)
    for l in m["layers"]:
        for n in ("q", "k", "v"):
            l[n] = padc(l[n] * l["g1"][None, :]) @ R.T
        for n in ("gate", "up"):
            l[n] = padc(l[n] * l["g2"][None, :]) @ R.T
        l["o"], l["down"] = R @ padr(l["o"]), R @ padr(l["down"])
        l["g1"], l["g2"] = torch.full((D,), c), torch.full((D,), c)
    m["d"] = D
    m["eps_scale"] = d / D
    return m


def permute_vocab(m, seed=25, keep=0):
    """New token id j holds old token P[j]. Returns (model, P). Ids below `keep` stay put (specials)."""
    m, g = clone(m), _rng(seed)
    V = m["embed"].shape[0]
    P = torch.cat([torch.arange(keep), keep + torch.randperm(V - keep, generator=g)])
    m["embed"] = m["embed"][P]
    if m["head"] is not None:
        m["head"] = m["head"][P]
    else:
        m["head"] = m["embed"].clone()
    return m, P
