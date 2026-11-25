"""Is a GGUF file a per-channel rescaling of the base? Norm weights divided by a scale s, the next matrices' input columns times s
(as activation-aware quantisers do). Prints, for three layers, how far the file is from the base plainly and after undoing s.

Usage: python probe_rescaled.py <base repository> <GGUF repository> <part of a file name>   (results/derive/qwen_rescaled.txt)
"""
import os, sys, json, numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open
from lineage_check import gguf as G

base, repo, want = sys.argv[1], sys.argv[2], sys.argv[3]
file = [f for f in list_repo_files(repo) if f.endswith(".gguf") and want.lower() in f.lower()][0]
path = hf_hub_download(repo, file); size = os.path.getsize(path); fh = open(path, "rb")
def read(a, b):
    fh.seek(a); return fh.read(b - a + 1)
h = G.header(read, size)
st = safe_open(hf_hub_download(base, "model.safetensors"), "pt")
def gg(name):
    kind, shape, off = h["tensors"][name]; fmt, per, bs = G.TYPES[kind]; n = int(np.prod(shape)); a = h["data"] + off
    return G.decode(read(a, a + n // per * bs - 1), fmt).reshape(shape)
def hf(name):
    return st.get_tensor(name).float().numpy()
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
for L in (0, 5, 12):
    n_g, n_h = gg(f"blk.{L}.attn_norm.weight"), hf(f"model.layers.{L}.input_layernorm.weight")
    s = n_h / n_g                                          # if norm_gguf = norm_hf / s
    for mat, hfn in (("attn_q", "self_attn.q_proj"), ("attn_k", "self_attn.k_proj"), ("attn_v", "self_attn.v_proj")):
        w_g, w_h = gg(f"blk.{L}.{mat}.weight"), hf(f"model.layers.{L}.{hfn}.weight")
        print(f"layer {L} {mat}: plain rel {rel(w_g, w_h):.4f} | with columns times s {rel(w_g, w_h * s[None, :]):.4f} | s range {s.min():.3f}..{s.max():.3f} median {np.median(s):.3f}")
    f_g, f_h = gg(f"blk.{L}.ffn_norm.weight"), hf(f"model.layers.{L}.post_attention_layernorm.weight")
    s2 = f_h / f_g
    for mat, hfn in (("ffn_gate", "mlp.gate_proj"), ("ffn_up", "mlp.up_proj")):
        w_g, w_h = gg(f"blk.{L}.{mat}.weight"), hf(f"model.layers.{L}.{hfn}.weight")
        print(f"layer {L} {mat}: plain rel {rel(w_g, w_h):.4f} | with columns times s {rel(w_g, w_h * s2[None, :]):.4f}")
    d_g, d_h = gg(f"blk.{L}.ffn_down.weight"), hf(f"model.layers.{L}.mlp.down_proj.weight")
    u_g, u_h = gg(f"blk.{L}.ffn_up.weight"), hf(f"model.layers.{L}.mlp.up_proj.weight")
    r = np.linalg.norm(u_g, axis=1) / np.linalg.norm(u_h * s2[None, :], axis=1)   # a row scale on up, undone on down's columns
    print(f"layer {L} ffn_down: plain rel {rel(d_g, d_h):.4f} | with columns divided by up's row scale {rel(d_g, d_h / r[None, :]):.4f}; up rows rescaled {rel(u_g, u_h * s2[None, :] * r[:, None]):.4f}")
    o_g, o_h = gg(f"blk.{L}.attn_output.weight"), hf(f"model.layers.{L}.self_attn.o_proj.weight")
    print(f"layer {L} attn_output: plain rel {rel(o_g, o_h):.4f}")
print("meta:", {k: v for k, v in h["meta"].items() if "awq" in k.lower() or "quantiz" in k.lower() or k.startswith("general.")})
