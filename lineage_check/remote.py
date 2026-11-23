"""Read just the parts of a model the fingerprint needs.

Sources share one interface (config, vocab, has, get, rows):
  Remote   - a Hub repo, read with HTTP Range requests against its safetensors files (raw bytes cached on disk)
  LocalST  - safetensors files on disk
  LocalBin - a pytorch_model.bin on disk (torch.load weights_only, for repos with no safetensors)
build() turns a source into the partial canonical model the fingerprint reads: a contiguous block of embedding
(and unembedding) rows plus a chosen set of layers, all float32, linear weights as (out, in).
Quantised linears are dequantised the way a scanner would see them: AWQ (GEMM), GPTQ (v1 zeros), MLX affine.
"""
from __future__ import annotations
import glob, json, os, struct, threading, time
import numpy as np
import requests
import torch
from huggingface_hub import HfApi, hf_hub_url

from .paths import ZOO, PARTIAL as CACHE
ROW_LO, ROW_HI = 300, 20000                       # embedding rows fetched: the token pool lives here
_tl = threading.local()

DT = {"BF16": (np.uint16, 2), "F16": (np.float16, 2), "F32": (np.float32, 4), "I32": (np.int32, 4),
      "U32": (np.uint32, 4), "I8": (np.int8, 1), "U8": (np.uint8, 1), "I64": (np.int64, 8), "F64": (np.float64, 8)}


def _decode(raw, dtype, shape):
    a = np.frombuffer(raw, dtype=DT[dtype][0]).reshape(shape)
    if dtype == "BF16":
        return (a.astype(np.uint32) << 16).view(np.float32)
    if dtype in ("F16", "F64"):
        return a.astype(np.float32)
    return a


def _vocab_from_tokenizer_json(tj):
    v = dict(tj["model"].get("vocab", {})) if isinstance(tj["model"].get("vocab"), dict) else {}
    if not v and isinstance(tj["model"].get("vocab"), list):          # unigram: list of [piece, score]
        v = {p[0]: i for i, p in enumerate(tj["model"]["vocab"])}
    for t in tj.get("added_tokens", []):
        v[t["content"]] = t["id"]
    return v


# ------------------------------------------------------------------ sources
SUPPORTED = ("llama", "qwen2", "qwen3", "mistral", "olmo", "qwen3_moe", "olmoe", "granite", "gpt_neox")


class Remote:
    def __init__(self, repo, revision=None):
        self.repo = repo
        info = HfApi().model_info(repo, revision=revision)
        self.sha = info.sha
        self.files = [s.rfilename for s in info.siblings]
        self.dir = os.path.join(CACHE, repo.replace("/", "--"), self.sha)
        os.makedirs(self.dir, exist_ok=True)
        self.index = {}
        for f in sorted(x for x in self.files if x.endswith(".safetensors") and "/" not in x):
            self._header(f)                           # top-level weights only (subfolders hold duplicates)
        if not any(k.startswith("model.layers.") for k in self.index):
            # Weights saved without the causal-LM wrapper (embedding models, rerankers) or inside a vision-language
            # wrapper: alias them to the causal-LM names. Applied only when the standard names are absent.
            rules = (("language_model.model.", "model."), ("model.language_model.", "model."), ("model.text_model.", "model."),
                     ("language_model.lm_head.", "lm_head."), ("layers.", "model.layers."),
                     ("embed_tokens.", "model.embed_tokens."), ("norm.", "model.norm."))
            for k in list(self.index):
                for old, new in rules:
                    if k.startswith(old):
                        self.index.setdefault(new + k[len(old):], self.index[k])
        self.config = json.loads(self._small("config.json"))
        for sub in ("text_config", "llm_config"):    # a vision-language wrapper: scan its language backbone
            if self.config.get("model_type") not in SUPPORTED and isinstance(self.config.get(sub), dict) \
                    and self.config[sub].get("model_type") in SUPPORTED:
                self.config = dict(self.config[sub])
                break
        self.label = repo + (f"@{revision}" if revision else "")

    @property
    def vocab(self):
        if not hasattr(self, "_vocab"):
            if "tokenizer.json" in self.files:
                self._vocab = _vocab_from_tokenizer_json(json.loads(self._small("tokenizer.json")))
            else:
                self._vocab = json.loads(self._small("vocab.json"))
        return self._vocab

    def _small(self, name):
        p = os.path.join(self.dir, name)
        if not os.path.exists(p):
            r = requests.get(hf_hub_url(self.repo, name, revision=self.sha), timeout=120)
            r.raise_for_status()
            with open(p, "wb") as o:
                o.write(r.content)
        return open(p, "rb").read()

    def _loc(self, f, refresh=False):
        """The CDN location of a file, resolved once (signed URLs last about an hour; refreshed on failure)."""
        locs = self.__dict__.setdefault("_locs", {})
        if refresh or f not in locs or time.time() - locs[f][1] > 1800:
            r = requests.get(hf_hub_url(self.repo, f, revision=self.sha), headers={"Range": "bytes=0-0"},
                             allow_redirects=True, timeout=120)
            locs[f] = (r.url, time.time())
        return locs[f][0]

    def _range(self, f, a, b):
        s = getattr(_tl, "s", None)
        if s is None:
            s = _tl.s = requests.Session()
        for attempt in range(5):
            try:
                r = s.get(self._loc(f, refresh=attempt > 0), headers={"Range": f"bytes={a}-{b}"}, timeout=600)
                if r.status_code == 206 and len(r.content) == b - a + 1:
                    return r.content
                raise IOError(f"HTTP {r.status_code}, {len(r.content)} bytes")
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))

    def prefetch(self, names, max_gap=4 << 20):
        """Fetch many tensors in a few large requests: merge byte spans in the same file closer than max_gap."""
        todo = []
        for n in names:
            if n in self.index:
                f, dt, shape, a, b = self.index[n]
                p = os.path.join(self.dir, n + ".raw")
                if not (os.path.exists(p) and os.path.getsize(p) == b - a):
                    todo.append((f, a, b, n))
        todo.sort()
        spans = []
        for f, a, b, n in todo:
            if spans and spans[-1][0] == f and a - spans[-1][2] <= max_gap:
                spans[-1][2] = max(spans[-1][2], b)
                spans[-1][3].append((a, b, n))
            else:
                spans.append([f, a, b, [(a, b, n)]])
        for f, a, b, members in spans:
            raw = self._cached_span(f, a, b)
            for x, y, n in members:
                p = os.path.join(self.dir, n + ".raw")
                with open(f"{p}.{os.getpid()}.tmp", "wb") as o:
                    o.write(raw[x - a:y - a])
                os.replace(f"{p}.{os.getpid()}.tmp", p)

    def _cached_span(self, f, a, b):
        parts, chunk = [], 256 << 20
        for st in range(a, b, chunk):
            parts.append(self._range(f, st, min(b, st + chunk) - 1))
        return b"".join(parts)

    def _header(self, f):
        p = os.path.join(self.dir, f + ".header.json")
        if os.path.exists(p):
            h = json.load(open(p))
        else:
            n = struct.unpack("<Q", self._range(f, 0, 7))[0]
            h = json.loads(self._range(f, 8, 8 + n - 1))
            h["__n__"] = n
            json.dump(h, open(p, "w"))
        base = 8 + h["__n__"]
        for k, v in h.items():
            if not k.startswith("__"):
                a, b = v["data_offsets"]
                self.index[k] = (f, v["dtype"], tuple(v["shape"]), base + a, base + b)

    def has(self, name):
        return name in self.index

    def _cached(self, key, f, a, b):
        p = os.path.join(self.dir, key + ".raw")
        if os.path.exists(p) and os.path.getsize(p) == b - a:
            return open(p, "rb").read()
        parts, chunk = [], 128 << 20
        for s in range(a, b, chunk):
            parts.append(self._range(f, s, min(b, s + chunk) - 1))
        raw = b"".join(parts)
        with open(f"{p}.{os.getpid()}.tmp", "wb") as o:
            o.write(raw)
        os.replace(f"{p}.{os.getpid()}.tmp", p)
        return raw

    def get(self, name):
        f, dt, shape, a, b = self.index[name]
        return _decode(self._cached(name, f, a, b), dt, shape)

    def rows(self, name, lo, hi):
        f, dt, shape, a, b = self.index[name]
        hi = min(hi, shape[0])
        rb = int(np.prod(shape[1:])) * DT[dt][1]
        return _decode(self._cached(f"{name}.rows{lo}-{hi}", f, a + lo * rb, a + hi * rb), dt, (hi - lo,) + shape[1:])


class LocalST:
    def __init__(self, path, label=None, repo=None):
        from safetensors import safe_open
        self.path, self.label, self.repo = path, label or repo or path, repo
        self.config = json.load(open(os.path.join(path, "config.json")))
        self._open = {}
        self.index = {}
        for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
            h = safe_open(f, framework="pt")
            self._open[f] = h
            for k in h.keys():
                self.index[k] = f

    @property
    def vocab(self):
        tj = os.path.join(self.path, "tokenizer.json")
        if os.path.exists(tj):
            return _vocab_from_tokenizer_json(json.load(open(tj)))
        return Remote(self.repo).vocab

    def has(self, name):
        return name in self.index

    def get(self, name):
        t = self._open[self.index[name]].get_tensor(name)
        if t.dtype == torch.uint32:
            return t.view(torch.int32).numpy().view(np.uint32)
        return t.float().numpy() if t.is_floating_point() else t.numpy()

    def rows(self, name, lo, hi):
        sl = self._open[self.index[name]].get_slice(name)
        t = sl[lo:min(hi, sl.get_shape()[0])]
        if t.dtype == torch.uint32:
            return t.view(torch.int32).numpy().view(np.uint32)
        return t.float().numpy()


class LocalBin:
    def __init__(self, path, label=None):
        self.path, self.label = path, label or path
        self.config = json.load(open(os.path.join(path, "config.json")))
        self.sd = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu", weights_only=True, mmap=True)

    vocab = None                                      # set by caller (same tokenizer as the family)

    def has(self, name):
        return name in self.sd

    def get(self, name):
        t = self.sd[name]
        return t.float().numpy() if t.is_floating_point() else t.numpy()

    def rows(self, name, lo, hi):
        return self.sd[name][lo:hi].float().numpy()


# ------------------------------------------------------------------ dequantisation
def _unpack(words, bits):
    shifts = np.arange(0, 32, bits, dtype=np.uint32)
    return ((words.astype(np.uint32)[..., None] >> shifts) & ((1 << bits) - 1)).astype(np.float32)


def deq_awq(qweight, qzeros, scales, bits, group):
    iw = _unpack(qweight.view(np.uint32), bits).reshape(qweight.shape[0], -1)          # (in, out)
    iz = _unpack(qzeros.view(np.uint32), bits).reshape(qzeros.shape[0], -1)
    order = np.arange(iw.shape[1]).reshape(-1, 32 // bits)[:, [0, 4, 1, 5, 2, 6, 3, 7] if bits == 4 else
                                                            list(range(32 // bits))].reshape(-1)
    iw, iz = iw[:, order], iz[:, order]
    g = np.arange(iw.shape[0]) // group
    return ((iw - iz[g]) * scales.astype(np.float32)[g]).T.copy()                       # (out, in)


def deq_gptq(qweight, qzeros, scales, g_idx, bits, group):
    pack = 32 // bits
    iw = _unpack(qweight.view(np.uint32), bits)                                         # (in/pack, out, pack)
    iw = iw.transpose(0, 2, 1).reshape(qweight.shape[0] * pack, -1)                     # (in, out)
    iz = _unpack(qzeros.view(np.uint32), bits).reshape(qzeros.shape[0], -1) + 1.0       # v1 stores zero - 1
    g = g_idx.astype(np.int64) if g_idx is not None else np.arange(iw.shape[0]) // group
    return ((iw - iz[g]) * scales.astype(np.float32)[g]).T.copy()


def deq_mlx(words, scales, biases):
    n_groups, in_f = scales.shape[1], words.shape[1] * 8
    q = _unpack(words.view(np.uint32), 4).reshape(words.shape[0], in_f)
    w = q.reshape(words.shape[0], n_groups, in_f // n_groups) * scales.astype(np.float32)[:, :, None] \
        + biases.astype(np.float32)[:, :, None]
    return w.reshape(words.shape[0], in_f)


def linear(src, prefix):
    cfg = src.config.get("quantization_config") or {}
    if src.has(prefix + ".qweight"):
        bits, group = int(cfg.get("bits", 4)), int(cfg.get("group_size", 128))
        qw, qz, sc = src.get(prefix + ".qweight"), src.get(prefix + ".qzeros"), src.get(prefix + ".scales")
        if cfg.get("quant_method") == "awq":
            return deq_awq(qw, qz, sc, bits, group)
        gi = src.get(prefix + ".g_idx") if src.has(prefix + ".g_idx") else None
        return deq_gptq(qw, qz, sc, gi, bits, group)
    w = src.get(prefix + ".weight")
    if w.dtype == np.uint32 and src.has(prefix + ".scales"):
        return deq_mlx(w, src.get(prefix + ".scales"), src.get(prefix + ".biases"))
    return np.asarray(w, dtype=np.float32)


def embed_rows(src, name, lo, hi):
    """Embedding rows lo..hi, dequantised if the table is MLX affine-quantised."""
    w = src.rows(name, lo, hi)
    base = name[: -len(".weight")]
    if np.asarray(w).dtype == np.uint32 and src.has(base + ".scales"):
        return deq_mlx(np.asarray(w), src.rows(base + ".scales", lo, hi), src.rows(base + ".biases", lo, hi))
    return np.asarray(w, dtype=np.float32)


# ------------------------------------------------------------------ partial canonical model
LLAMA = {"embed": "model.embed_tokens.weight", "head": "lm_head.weight", "norm": "model.norm.weight",
         "pre": "model.layers.{i}.", "q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj",
         "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj",
         "g1": "input_layernorm.weight", "g2": "post_attention_layernorm.weight"}


def build(src, layers="all", lo=ROW_LO, hi=ROW_HI):
    cfg = src.config
    kind = cfg["model_type"]
    d, L, H = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"]
    Hkv = cfg.get("num_key_value_heads") or H
    hd = cfg.get("head_dim") or d // H
    if layers == "all":
        layers = list(range(L))
    elif isinstance(layers, int):
        layers = sorted({int(round((j + 0.5) * L / layers - 0.5)) for j in range(layers)})
    m = {"kind": kind, "d": d, "L": L, "H": H, "Hkv": Hkv, "hd": hd, "cfg": cfg, "lo": lo,
         "label": getattr(src, "label", ""), "layers": {}}
    ones = np.ones(d, dtype=np.float32)
    if kind == "gpt_neox":
        if hasattr(src, "prefetch"):
            src.prefetch([f"gpt_neox.layers.{i}.{n}" for i in layers for n in (
                "attention.query_key_value.weight", "attention.dense.weight", "mlp.dense_h_to_4h.weight",
                "mlp.dense_4h_to_h.weight", "input_layernorm.weight", "post_attention_layernorm.weight")]
                + ["gpt_neox.final_layer_norm.weight"])
        m["E"] = src.rows("gpt_neox.embed_in.weight", lo, hi)
        m["U"] = src.rows("embed_out.weight", lo, hi)
        m["final_g"] = src.get("gpt_neox.final_layer_norm.weight")
        for i in layers:
            p = f"gpt_neox.layers.{i}."
            W = src.get(p + "attention.query_key_value.weight").reshape(H, 3, hd, d)
            m["layers"][i] = {"q": W[:, 0].reshape(H * hd, d), "k": W[:, 1].reshape(H * hd, d),
                              "v": W[:, 2].reshape(H * hd, d), "o": src.get(p + "attention.dense.weight"),
                              "up": src.get(p + "mlp.dense_h_to_4h.weight"), "down": src.get(p + "mlp.dense_4h_to_h.weight"),
                              "gate": None, "g1": src.get(p + "input_layernorm.weight"),
                              "g2": src.get(p + "post_attention_layernorm.weight")}
        return m
    if kind not in ("llama", "qwen2", "qwen3", "mistral", "olmo", "qwen3_moe", "olmoe", "granite"):
        raise ValueError(f"unsupported model_type {kind}")
    N = LLAMA
    moe = kind in ("qwen3_moe", "olmoe")
    n_exp = cfg.get("num_experts") or 0
    if hasattr(src, "prefetch"):
        want = []
        for i in layers:
            p = N["pre"].format(i=i)
            for n in ("q", "k", "v", "o") + (() if moe else ("gate", "up", "down")):
                want += [p + N[n] + sfx for sfx in (".weight", ".qweight", ".qzeros", ".scales", ".g_idx", ".biases")]
            if moe:
                want += [p + "mlp.gate.weight"] + [p + f"mlp.experts.{e}.{n}_proj.weight" for e in range(n_exp)
                                                    for n in ("gate", "up", "down")]
            want += [p + N["g1"], p + N["g2"], p + "self_attn.q_norm.weight", p + "self_attn.k_norm.weight"]
        src.prefetch(want + [N["norm"]])
    m["E"] = embed_rows(src, N["embed"], lo, hi)
    m["U"] = embed_rows(src, N["head"], lo, hi) if src.has(N["head"]) else None
    m["final_g"] = src.get(N["norm"]) if src.has(N["norm"]) else ones
    for i in layers:
        p = N["pre"].format(i=i)
        l = {n: linear(src, p + N[n]) for n in ("q", "k", "v", "o")}
        if moe:
            l["gate"] = l["up"] = l["down"] = None
            l["experts"] = [tuple(linear(src, p + f"mlp.experts.{e}.{n}_proj") for n in ("gate", "up", "down"))
                            for e in range(n_exp)]
            l["router"] = np.asarray(src.get(p + "mlp.gate.weight"), dtype=np.float32)
        else:
            l.update({n: linear(src, p + N[n]) for n in ("gate", "up", "down")})
        l["g1"] = src.get(p + N["g1"]) if src.has(p + N["g1"]) else ones
        l["g2"] = src.get(p + N["g2"]) if src.has(p + N["g2"]) else ones
        for qk in ("q", "k"):                         # QK-norm gains (per head dim, or full width) fold into q / k rows
            nm = p + f"self_attn.{qk}_norm.weight"
            if src.has(nm):
                gq = np.asarray(src.get(nm), dtype=np.float32)
                rows = l[qk].shape[0]
                l[qk] = l[qk] * (np.tile(gq, rows // gq.shape[0]) if gq.shape[0] != rows else gq)[:, None]
        m["layers"][i] = l
    return m


def from_canon(c, label="", lo=ROW_LO, hi=ROW_HI):
    """A full in-memory canon model (canon.py / edits.py) in the partial form, rows lo..hi."""
    m = {k: c[k] for k in ("kind", "d", "L", "H", "Hkv", "hd", "cfg")}
    hi = min(hi, c["embed"].shape[0])
    m.update(lo=lo, label=label, layers={})
    m["E"] = c["embed"][lo:hi].numpy()
    m["U"] = c["head"][lo:hi].numpy() if c["head"] is not None else None
    m["final_g"] = c["final_g"].numpy()
    for i, l in enumerate(c["layers"]):
        m["layers"][i] = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in l.items()
                          if k in ("q", "k", "v", "o", "gate", "up", "down", "g1", "g2")}
    return m
