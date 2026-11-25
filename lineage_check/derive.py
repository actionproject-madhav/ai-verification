"""Exact identity: is a quantised file the known model's own weights, written by the public encoder?

A copy that only has to be close to the known model leaves room, and every tolerance is room an operator can use:
a model served under the token test can carry most of a stronger model's lead (docs/RESULTS.md). Many published
GGUF files leave none. Give the known model's tensors to the encoder llama.cpp quantises with (ggml), and the bytes
come out equal to the file's, block for block. The one setting a file does not record is how its encoder was
compiled (whether multiplies and adds were fused), so the check tries a few builds of the same encoder and asks
that one build wrote every tensor of the file.

A file is exact when it is the known model and nothing else: its architecture is the known model's, it holds every
tensor of the known checkpoint and no other, each is in a format the box can read, one build of the encoder wrote
every one of them, and its tokens are the known model's. Such a file holds nothing the known model does not. It is
not the known model's behaviour: each tensor's format is the publisher's choice, and a 2-bit file is far from the
model it came from, so the formats and the bits per weight are reported with the verdict. A file that does not
re-derive (calibrated with an importance matrix, rescaled per channel, edited, made from other weights, or written by
an encoder not tried) is left to the fuzzy check and the token test. The two settings a file does not record, the
compiler's handling of fused multiply-adds and the per-tensor recipe, are the ones Liu and Tsai (arXiv:2610.19498)
found missing from published files.

The encoder is a shared library (libggml-base) named in LINEAGE_GGML, one or more builds separated by commas;
tools/build_ggml.sh builds the four this repository was measured with. The 8-bit format and the plain formats are
also recomputed here in numpy, so a file in those needs no library.
"""
from __future__ import annotations

import ctypes
import json
import os

import numpy as np

from . import gguf as G

PLAIN = {"F32": np.float32, "F16": np.float16, "F64": np.float64}
NEEDS_DATA = {16: "IQ2_XXS", 17: "IQ2_XS", 19: "IQ1_S", 22: "IQ2_S", 29: "IQ1_M"}     # the encoder writes these only with calibration data
MORE = {18: "IQ3_XXS", 21: "IQ3_S", 34: "TQ1_0", 35: "TQ2_0"}                         # written by the encoder, not decoded by gguf.py
SERVABLE = set(G.BLOCK) | set(PLAIN) | {"BF16"}       # formats the box's reader decodes, so it can serve the file
# The GGUF architecture llama.cpp writes for each Hub model type, and the types whose query and key rows it interleaves
# (its LlamaModel converter permutes them). One rule for the check here and for the box's loader (outputs.load_like).
GGUF_ARCH = {"llama": "llama", "mistral": "llama", "mixtral": "llama", "qwen2": "qwen2", "qwen3": "qwen3"}
INTERLEAVED = {"llama", "mistral", "mixtral"}
NOT_WEIGHTS = ("rope_freqs.weight",)                   # computed from the config by the converter; the box uses the known config


class Encoder:
    """One build of the reference encoder (ggml_quantize_chunk in libggml-base)."""

    def __init__(self, path, name=None):
        self.path = path
        self.name = name or os.path.basename(os.path.dirname(os.path.dirname(path))).replace("build-", "") or path
        self.lib = ctypes.CDLL(path)
        q = self.lib.ggml_quantize_chunk
        q.restype, q.argtypes = ctypes.c_size_t, [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        self.lib.ggml_blck_size.restype, self.lib.ggml_blck_size.argtypes = ctypes.c_int64, [ctypes.c_int]
        self.lib.ggml_type_size.restype, self.lib.ggml_type_size.argtypes = ctypes.c_size_t, [ctypes.c_int]

    def block(self, kind):
        return int(self.lib.ggml_blck_size(kind)), int(self.lib.ggml_type_size(kind))

    def __call__(self, kind, x, rows, per_row, size):
        x = np.ascontiguousarray(x, np.float32)
        out = np.zeros(size, np.uint8)
        wrote = self.lib.ggml_quantize_chunk(kind, x.ctypes.data, out.ctypes.data, 0, rows, per_row, None)
        if wrote != size:
            raise ValueError(f"the encoder wrote {wrote} bytes where the file holds {size}")
        return out


def builds(paths=None):
    paths = paths if paths is not None else [p for p in os.environ.get("LINEAGE_GGML", "").split(",") if p]
    return [Encoder(p) for p in paths]


def roundf(n):
    """C's roundf: halves go away from zero."""
    a = np.abs(n)
    f = np.floor(a)
    return np.sign(n) * (f + np.floor(2 * (a - f)))


def q8_0(x):
    """The 8-bit format: runs of 32 numbers, one 16-bit scale each, as ggml's reference encoder writes it."""
    b = np.ascontiguousarray(x, np.float32).reshape(-1, 32)
    d = np.abs(b).max(1, keepdims=True) / np.float32(127)
    with np.errstate(divide="ignore"):
        inv = np.where(d == 0, np.float32(0), np.float32(1) / d).astype(np.float32)
    return np.concatenate([d.astype("<f2").view(np.uint8), roundf(b * inv).astype(np.int8).view(np.uint8)], 1).reshape(-1)


def bf16(x):
    """float32 to bfloat16 bits, rounded to nearest with ties to even, as the GGUF converter writes it."""
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype("<u2")


def sources(w):
    """The known tensor as float32, and as it reads after a pass through each 16-bit format a converter may use
    (each only if it differs from those before it: a bfloat16 checkpoint reads the same both ways)."""
    w32 = np.asarray(w, np.float32)
    with np.errstate(over="ignore"):
        f16 = w32.astype(np.float16).astype(np.float32)
    b16 = (bf16(w32).astype(np.uint32) << 16).view(np.float32)
    seen = []
    for label, x in (("float32", w32), ("through float16", f16), ("through bfloat16", b16)):
        if not any(np.array_equal(x, y, equal_nan=True) for y in seen):
            seen.append(x)
            yield label, x


hub_name = G.hub_name


def interleaved(model_type):
    """Whether llama.cpp stores this model type's query and key rows with each head's two halves interleaved."""
    return model_type in INTERLEAVED


def heads_of(config, name):
    """Heads of a query or key matrix by its Hub name, or None for any other tensor."""
    part = name.rsplit(".", 2)[-2] if name.endswith(".weight") or name.endswith(".bias") else ""
    if part == "q_proj":
        return config.get("num_attention_heads")
    if part == "k_proj":
        return config.get("num_key_value_heads") or config.get("num_attention_heads")
    return None


class Known:
    """The known model's tensors (float32), its tokens in id order and its chat template, from a directory of
    safetensors files or from a Hub repository."""

    def __init__(self, get, config, vocab=None, template=None, names=None):
        self.config, self.vocab, self.template, self.names = config, vocab, template, list(names or [])
        tied = config.get("tie_word_embeddings", False)
        self.get = lambda k: get(k) if get(k) is not None or k != "lm_head.weight" or not tied else get("model.embed_tokens.weight")

    def expected(self):
        """The tensors a file of this model must hold: every weight of the checkpoint, the output matrix when it is
        tied to the token table and stored once, buffers left out."""
        return {n for n in self.names if not n.endswith("inv_freq")}

    @classmethod
    def open(cls, where, revision=None):
        from safetensors import safe_open
        if os.path.isdir(where):
            find = lambda f: os.path.join(where, f) if os.path.exists(os.path.join(where, f)) else None
            files = [os.path.join(where, f) for f in sorted(os.listdir(where)) if f.endswith(".safetensors")]
        else:
            from huggingface_hub import hf_hub_download, list_repo_files
            names = list_repo_files(where, revision=revision)
            find = lambda f: hf_hub_download(where, f, revision=revision) if f in names else None
            files = [hf_hub_download(where, f, revision=revision) for f in names if f.endswith(".safetensors") and "/" not in f]
        opened = [safe_open(f, "pt") for f in files]               # torch reads bfloat16, numpy does not
        at = {k: s for s in opened for k in s.keys()}
        vocab, template = None, None
        if find("tokenizer.json"):
            tj = json.load(open(find("tokenizer.json")))
            ids = dict(tj["model"].get("vocab") or {}) if isinstance(tj["model"].get("vocab"), dict) else {}
            ids.update({t["content"]: t["id"] for t in tj.get("added_tokens", [])})
            vocab = [None] * (max(ids.values()) + 1 if ids else 0)
            for t, i in ids.items():
                vocab[i] = t
        if find("tokenizer_config.json"):
            template = json.load(open(find("tokenizer_config.json"))).get("chat_template")
            template = template if template is None or isinstance(template, str) else json.dumps(template, sort_keys=True)
        config = json.load(open(find("config.json"))) if find("config.json") else {}
        return cls(lambda k: at[k].get_tensor(k).float().numpy() if k in at else None, config, vocab, template, names=list(at))


class File:
    """A GGUF file on disk: its header and the bytes of each tensor."""

    def __init__(self, path, encoders=()):
        self.fh, self.size = open(path, "rb"), os.path.getsize(path)
        self.h = G.header(self.read, self.size)
        self.meta = self.h["meta"]
        self.types = {**G.TYPES, **({k: (v, *encoders[0].block(k)) for k, v in MORE.items()} if encoders else {})}

    def read(self, a, b):
        self.fh.seek(a)
        return self.fh.read(b - a + 1)

    def raw(self, name):
        kind, shape, offset = self.h["tensors"][name]
        fmt, per, size = self.types[kind]
        n = int(np.prod(shape, dtype=np.int64)) if shape else 1
        a = self.h["data"] + offset
        return kind, fmt, per, size, tuple(shape), np.frombuffer(self.read(a, a + n // per * size - 1), np.uint8)


def _in_file_order(w, name, known):
    """The known tensor in the row order llama.cpp writes for the known model's type."""
    h = heads_of(known.config, name) if interleaved(known.config.get("model_type")) else None
    if not h or w.ndim < 1:
        return w
    return w.reshape(h, 2, w.shape[0] // h // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def _ways(kind, fmt, w, per_row, size, encoders):
    """For each way a converter could have read the known tensor (float32, or through a 16-bit format), the bytes
    each writer gives: [(source, [(build, bytes), ...]), ...]. `build` is None where no compiled encoder is involved."""
    for label, x in sources(w):
        if fmt in PLAIN:
            yield label, [(None, x.astype(PLAIN[fmt]).view(np.uint8).reshape(-1))]
        elif fmt == "BF16":
            yield label, [(None, bf16(x).view(np.uint8).reshape(-1))]
        else:
            got = [(None, q8_0(x))] if fmt == "Q8_0" else []
            for e in encoders:
                try:
                    got.append((e.name, e(kind, x, x.size // per_row, per_row, size)))
                except ValueError:
                    continue
            yield label, got


def _sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path, known, encoders=None, give_up=12):
    """Whether the GGUF file at `path` is `known` (a Known, a directory or a Hub repository) and nothing else (see the
    module's docstring). Stops early once `give_up` quantised tensors have been compared and fewer than half of them
    were identical. Returns the verdict and what it rests on, `reason` saying what failed."""
    encoders = builds() if encoders is None else encoders
    known = known if isinstance(known, Known) else Known.open(known)
    f = File(path, encoders)
    arch, want = f.meta.get("general.architecture"), GGUF_ARCH.get(known.config.get("model_type"))
    names = {name: hub_name(name) for name in f.h["tensors"] if name not in NOT_WEIGHTS}
    formats = {}
    for name in names:
        kind = f.h["tensors"][name][0]
        fmt = f.types[kind][0] if kind in f.types else NEEDS_DATA.get(kind, f"type {kind}")
        formats[fmt] = formats.get(fmt, 0) + 1
    out = {"exact": False, "reason": None, "architecture": arch, "builds": [], "tensors": len(names), "identical": 0, "differs": [],
           "missing": sorted(known.expected() - set(names.values()))[:8] if known.names else [], "extra": sorted(set(names.values()) - known.expected())[:8] if known.names else [],
           "formats": formats, "unservable": sorted(set(formats) - SERVABLE), "calibrated": any("imatrix" in k for k in f.meta), "stopped_early": False}
    if want is None or arch != want:
        out["reason"] = f"the file's architecture is {arch!r}; the known model ({known.config.get('model_type')}) is written as {want!r}"
        return out
    alive, narrowed, quantised, quantised_same, elements, nbytes = [e.name for e in encoders], False, 0, 0, 0, 0
    for name, hub in names.items():
        kind = f.h["tensors"][name][0]
        if kind not in f.types:
            continue
        kind, fmt, per, size, shape, raw = f.raw(name)
        elements, nbytes = elements + int(np.prod(shape, dtype=np.int64)), nbytes + raw.size
        w = known.get(hub)
        if w is None or w.size != int(np.prod(shape, dtype=np.int64)):
            out["differs"].append([name, fmt, "not in the known model"])
            continue
        w = _in_file_order(w, hub, known)
        plain = fmt in PLAIN or fmt == "BF16"
        which = set()
        for _, got in _ways(kind, fmt, w, shape[-1] if shape else 1, raw.size, [e for e in encoders if e.name in alive]):
            which = {b for b, g in got if g.size == raw.size and np.array_equal(g, raw)}
            if which:
                break                                  # one way of reading the known tensor explains it
        quantised += not plain
        if which:
            out["identical"] += 1
            quantised_same += not plain
            if None not in which:                      # only some builds write this tensor so: the rest are out
                alive, narrowed = [b for b in alive if b in which], True
        elif len(out["differs"]) < 8:
            out["differs"].append([name, fmt, "bytes differ"])
        if quantised >= give_up and quantised_same < quantised / 2:
            out["stopped_early"] = True
            break
    out["builds"] = alive if narrowed else []
    out["bits_per_weight"] = round(8 * nbytes / elements, 3) if elements and not out["stopped_early"] else None
    toks = f.h["tokens"]
    if toks is not None and known.vocab:
        n = len(known.vocab)
        same = len(toks) >= n and toks[:n] == known.vocab and all(t.startswith("[PAD") for t in toks[n:])
        out["vocab"] = "same" if same else "differs"
    tmpl = f.meta.get("tokenizer.chat_template")
    out["template"] = ("absent in both" if tmpl is None and known.template is None else "same" if tmpl == known.template else
                       "added" if known.template is None else "removed" if tmpl is None else "differs")
    out["reason"] = ("the known model's list of tensors is not known" if not known.names else
                     "the file lacks tensors of the known model" if out["missing"] else "the file holds tensors the known model does not" if out["extra"]
                     else "a tensor is in a format the box cannot read" if out["unservable"] else "the known model's tokens are not the file's" if out.get("vocab") != "same"
                     else "a tensor's bytes are not what the encoder writes" if out["identical"] < out["tensors"] else None)
    out["exact"] = out["reason"] is None
    if out["exact"]:
        out["sha256"] = _sha256(path)
    return out
