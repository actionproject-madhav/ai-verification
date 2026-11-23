"""Fingerprint v4: circuits measured on the data manifold (needs a forward pass on probe text).

v3 reads every layer through raw token embeddings. Deep layers never see raw embeddings, so a white-box attacker
can rewrite weights in directions real activations never visit (dead neurons, unused subspaces) and move v3
without changing the function. v4 reads each layer through its real inputs on secret probe text:
  QK  = (X Qs^T diag(w)) (K X^T)       X   = the actual q/k/v input at N probe positions
  OV  = (Z Os) (V X^T)                 Z   = the residual stream entering the layer
  MLP = (Z W_down) (A^T)               A   = the MLP's actual hidden activations (dead neurons contribute nothing)
  ACT = (A diag|W_down|) (A diag|W_down|)^T
Each is invariant to the exact symmetries (activations transform with the weights). An edit that changes these
matrices changes what the layer computes on real text.
"""
from __future__ import annotations
import numpy as np
import torch

COMPS = ("qk", "ov", "mlp", "act")


def _layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers, "llama"
    return model.gpt_neox.layers, "neox"


@torch.no_grad()
def collect(model, ids, layer_idx):
    """Per layer: residual stream entering the layer (Z), attention input (X), MLP hidden activation (A)."""
    layers, kind = _layers(model)
    acts = {l: {} for l in layer_idx}
    hooks = []
    for l in layer_idx:
        L = layers[l]

        def pre(m, args, kwargs, l=l):
            acts[l]["Z"] = (args[0] if args else kwargs["hidden_states"]).detach()
        hooks.append(L.register_forward_pre_hook(pre, with_kwargs=True))
        if kind == "llama":
            hooks.append(L.self_attn.q_proj.register_forward_hook(lambda m, i, o, l=l: acts[l].__setitem__("X", i[0].detach())))
            hooks.append(L.mlp.down_proj.register_forward_hook(lambda m, i, o, l=l: acts[l].__setitem__("A", i[0].detach())))
        else:
            hooks.append(L.attention.query_key_value.register_forward_hook(lambda m, i, o, l=l: acts[l].__setitem__("X", i[0].detach())))
            hooks.append(L.mlp.dense_4h_to_h.register_forward_hook(lambda m, i, o, l=l: acts[l].__setitem__("A", i[0].detach())))
    model(ids)
    for h in hooks:
        h.remove()
    return acts


def _weights(model, l):
    layers, kind = _layers(model)
    L = layers[l]
    c = model.config
    H = c.num_attention_heads
    Hkv = getattr(c, "num_key_value_heads", None) or H
    if kind == "llama":
        a = L.self_attn
        q, k, v, o = a.q_proj.weight, a.k_proj.weight, a.v_proj.weight, a.o_proj.weight
        down = L.mlp.down_proj.weight
    else:
        d = c.hidden_size
        hd = d // H
        W = L.attention.query_key_value.weight.view(H, 3, hd, d)
        q, k, v = W[:, 0].reshape(H * hd, d), W[:, 1].reshape(H * hd, d), W[:, 2].reshape(H * hd, d)
        o, down = L.attention.dense.weight, L.mlp.dense_4h_to_h.weight
    hd = q.shape[0] // H
    return q.float(), k.float(), v.float(), o.float(), down.float(), H, Hkv, hd


@torch.no_grad()
def fingerprint(model, ids, layer_idx, drop_first=4):
    """ids: (B, T) probe tokens. Positions < drop_first in each sequence are dropped (attention-sink positions)."""
    acts = collect(model, ids, layer_idx)
    fp = {}
    for l in layer_idx:
        q, k, v, o, down, H, Hkv, hd = _weights(model, l)
        G = H // Hkv
        d = q.shape[1]
        sel = lambda t: t[:, drop_first:].reshape(-1, t.shape[-1]).float()
        X, Z, A = sel(acts[l]["X"]), sel(acts[l]["Z"]), sel(acts[l]["A"])
        Qs = q.reshape(Hkv, G, hd, d).sum(1).reshape(Hkv * hd, d)
        Os = o.reshape(o.shape[0], Hkv, G, hd).sum(2).reshape(o.shape[0], Hkv * hd)
        Og = Os.reshape(-1, Hkv, hd).permute(1, 2, 0)
        Vg = v.reshape(Hkv, hd, -1)
        wg = torch.einsum("gij,gji->g", Og @ Og.transpose(1, 2), Vg @ Vg.transpose(1, 2)).clamp_min(0).sqrt()
        Ad = A * down.norm(dim=0)
        fp[l] = {"qk": ((X @ Qs.T) * wg.repeat_interleave(hd)) @ (k @ X.T), "ov": (Z @ Os) @ (v @ X.T),
                 "mlp": (Z @ down) @ A.T, "act": Ad @ Ad.T}
    return fp


def corr(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def distance(fa, fb):
    per = {c: float(np.mean([1 - corr(fa[l][c], fb[l][c]) for l in fa])) for c in COMPS}
    per["d"] = float(np.mean([per[c] for c in COMPS]))
    per["attn"] = float(np.mean([per["qk"], per["ov"]]))
    return per
