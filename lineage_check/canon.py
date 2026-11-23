"""Canonical in-memory form of a decoder-only transformer, read straight from safetensors.

Linear weights use the torch convention W: (out, in), y = W x. Everything is float32 on CPU.
Supported: Llama-type (SmolLM2, Qwen2/2.5: gated MLP, RMSNorm, GQA, optional qkv bias) and
GPT-NeoX (Pythia: fused interleaved qkv, plain GELU MLP, LayerNorm).
"""
from __future__ import annotations
import json, os, glob, copy
import torch
from safetensors import safe_open


def _load_tensors(path):
    """All tensors as float32. MLX affine-quantized weights (uint32-packed, with .scales / .biases per group)
    are dequantized, as a scanner reading the artifact would: w = scale * q + bias."""
    import numpy as np
    raw = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="pt") as sf:
            for k in sf.keys():
                raw[k] = sf.get_tensor(k)
    out = {}
    for k, v in raw.items():
        if k.endswith((".scales", ".biases")):
            continue
        base = k[: -len(".weight")] if k.endswith(".weight") else None
        if base and v.dtype == torch.uint32 and base + ".scales" in raw:
            words = v.view(torch.int32).numpy().view(np.uint32)              # 4-bit: 8 values per word, low first
            s = raw[base + ".scales"].float().numpy()
            b = raw[base + ".biases"].float().numpy()
            n_groups, in_f = s.shape[1], words.shape[1] * 8
            shifts = np.arange(8, dtype=np.uint32) * 4
            q = ((words[:, :, None] >> shifts[None, None, :]) & 0xF).astype(np.float32).reshape(words.shape[0], in_f)
            w = q.reshape(words.shape[0], n_groups, in_f // n_groups) * s[:, :, None] + b[:, :, None]
            out[k] = torch.from_numpy(w.reshape(words.shape[0], in_f))
        else:
            out[k] = v.float()
    return out


def load(path):
    cfg = json.load(open(os.path.join(path, "config.json")))
    T = _load_tensors(path)
    arch = cfg.get("model_type")
    if arch in ("llama", "qwen2"):
        return _load_llama(cfg, T)
    if arch == "gpt_neox":
        return _load_neox(cfg, T)
    raise ValueError(f"unsupported model_type {arch}")


def _load_llama(cfg, T):
    d, L = cfg["hidden_size"], cfg["num_hidden_layers"]
    H, Hkv = cfg["num_attention_heads"], cfg.get("num_key_value_heads", cfg["num_attention_heads"])
    hd = cfg.get("head_dim", d // H)
    m = {"kind": "llama", "d": d, "L": L, "H": H, "Hkv": Hkv, "hd": hd, "cfg": cfg,
         "embed": T["model.embed_tokens.weight"],
         "head": T.get("lm_head.weight"),              # None => tied to embed
         "final_g": T["model.norm.weight"], "layers": []}
    for i in range(L):
        p = f"model.layers.{i}."
        a, f = p + "self_attn.", p + "mlp."
        m["layers"].append({
            "q": T[a + "q_proj.weight"], "k": T[a + "k_proj.weight"], "v": T[a + "v_proj.weight"],
            "o": T[a + "o_proj.weight"],
            "qb": T.get(a + "q_proj.bias"), "kb": T.get(a + "k_proj.bias"), "vb": T.get(a + "v_proj.bias"),
            "gate": T[f + "gate_proj.weight"], "up": T[f + "up_proj.weight"], "down": T[f + "down_proj.weight"],
            "g1": T[p + "input_layernorm.weight"], "g2": T[p + "post_attention_layernorm.weight"],
        })
    return m


def _load_neox(cfg, T):
    d, L, H = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"]
    hd = d // H
    m = {"kind": "neox", "d": d, "L": L, "H": H, "Hkv": H, "hd": hd, "cfg": cfg,
         "embed": T["gpt_neox.embed_in.weight"], "head": T["embed_out.weight"],
         "final_g": T["gpt_neox.final_layer_norm.weight"], "layers": []}
    for i in range(L):
        p = f"gpt_neox.layers.{i}."
        W = T[p + "attention.query_key_value.weight"].view(H, 3, hd, d)
        B = T[p + "attention.query_key_value.bias"].view(H, 3, hd)
        m["layers"].append({
            "q": W[:, 0].reshape(H * hd, d).clone(), "k": W[:, 1].reshape(H * hd, d).clone(),
            "v": W[:, 2].reshape(H * hd, d).clone(),
            "qb": B[:, 0].reshape(-1).clone(), "kb": B[:, 1].reshape(-1).clone(), "vb": B[:, 2].reshape(-1).clone(),
            "o": T[p + "attention.dense.weight"],
            "up": T[p + "mlp.dense_h_to_4h.weight"], "down": T[p + "mlp.dense_4h_to_h.weight"], "gate": None,
            "g1": T[p + "input_layernorm.weight"], "g2": T[p + "post_attention_layernorm.weight"],
        })
    return m


LINEAR = ("q", "k", "v", "o", "gate", "up", "down")


def clone(m):
    return copy.deepcopy(m)
